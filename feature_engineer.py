"""
Feature engineering module for the trading bot
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import talib
from datetime import datetime, timedelta
from ..sentiment_analyzer import SentimentAnalyzer
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
from sklearn.preprocessing import StandardScaler
import yfinance as yf
import logging

class FeatureEngineer:
    """Feature engineering class for technical indicators and market data"""
    
    def __init__(self, data=None):
        """
        Initialize feature engineer with historical data
        
        Args:
            data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        """
        self.data = data.copy() if data is not None else None
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(__name__)
        
        if self.data is not None:
            # Handle MultiIndex columns from yfinance
            if isinstance(self.data.columns, pd.MultiIndex):
                self.data.columns = [col[0].lower() for col in self.data.columns]
            else:
                self.data.columns = self.data.columns.str.lower()
            
            self._validate_data()
        
    def _validate_data(self):
        """Validate input data format"""
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in self.data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
            
        # Convert price columns to float
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            self.data[col] = self.data[col].astype(float)
        
        # Convert volume to int
        self.data['volume'] = self.data['volume'].astype(int)
        
        # Ensure index is datetime
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data index must be a DatetimeIndex")
        
        # Sort by timestamp
        self.data = self.data.sort_index()

    def calculate_rsi(self, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            period: RSI period (default: 14)
            
        Returns:
            RSI values as pandas Series
        """
        close_prices = self.data['close'].astype(float).values
        rsi_values = talib.RSI(close_prices, timeperiod=period)
        return pd.Series(rsi_values, index=self.data.index)
    
    def calculate_ema_crossover(self, fast_period: int = 9, slow_period: int = 21) -> pd.Series:
        """
        Calculate EMA crossover signals
        
        Args:
            fast_period: Fast EMA period (default: 9)
            slow_period: Slow EMA period (default: 21)
            
        Returns:
            Crossover signals: 1 for bullish crossover, -1 for bearish, 0 for no crossover
        """
        close_prices = self.data['close'].astype(float).values
        fast_ema = pd.Series(
            talib.EMA(close_prices, timeperiod=fast_period),
            index=self.data.index
        )
        slow_ema = pd.Series(
            talib.EMA(close_prices, timeperiod=slow_period),
            index=self.data.index
        )
        
        # Calculate crossover signals
        crossover = pd.Series(0, index=self.data.index)
        crossover[fast_ema > slow_ema] = 1
        crossover[fast_ema < slow_ema] = -1
        
        return crossover
    
    def calculate_pcr(self, oi_data: pd.DataFrame) -> pd.Series:
        """
        Calculate Put-Call Ratio (PCR)
        
        Args:
            oi_data: DataFrame with columns ['timestamp', 'put_oi', 'call_oi']
            
        Returns:
            PCR values as pandas Series
        """
        # Merge OI data with price data
        merged = pd.merge_asof(
            self.data[['timestamp']],
            oi_data,
            on='timestamp',
            direction='backward'
        )
        
        # Calculate PCR
        pcr = merged['put_oi'] / merged['call_oi']
        return pcr
    
    def add_sentiment_features(self, symbol: str) -> pd.DataFrame:
        """
        Add sentiment features to the dataset
        
        Args:
            symbol: Stock symbol
            
        Returns:
            DataFrame with added sentiment features
        """
        # Get sentiment data
        sentiment_data = self.sentiment_analyzer.get_combined_sentiment(symbol)
        
        # Add sentiment features
        self.data['sentiment_score'] = sentiment_data['combined_score']
        self.data['sentiment_strength'] = sentiment_data['sentiment_strength']
        self.data['news_sentiment'] = sentiment_data['news_sentiment']
        self.data['social_sentiment'] = sentiment_data['social_sentiment']
        self.data['market_sentiment'] = sentiment_data['market_sentiment']
        self.data['news_count'] = sentiment_data['news_count']
        self.data['social_volume'] = sentiment_data['social_volume']
        self.data['sentiment_volatility'] = sentiment_data['volatility']
        
        # Add sentiment momentum
        self.data['sentiment_momentum'] = self.data['sentiment_score'].pct_change()
        
        # Add sentiment acceleration
        self.data['sentiment_acceleration'] = self.data['sentiment_momentum'].pct_change()
        
        # Add sentiment divergence
        self.data['sentiment_divergence'] = (
            self.data['sentiment_score'] - self.data['market_sentiment']
        )
        
        return self.data

    def calculate_macd(self) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Returns:
            Tuple of (MACD line, Signal line, MACD histogram)
        """
        close_prices = self.data['close'].astype(float).values
        macd, signal, hist = talib.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
        return (
            pd.Series(macd, index=self.data.index),
            pd.Series(signal, index=self.data.index),
            pd.Series(hist, index=self.data.index)
        )

    def calculate_bollinger_bands(self, period: int = 20, num_std: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands
        
        Args:
            period: Moving average period (default: 20)
            num_std: Number of standard deviations (default: 2.0)
        
        Returns:
            Tuple of (Upper band, Middle band, Lower band)
        """
        close_prices = self.data['close'].astype(float).values
        upper, middle, lower = talib.BBANDS(close_prices, timeperiod=period, nbdevup=num_std, nbdevdn=num_std)
        return (
            pd.Series(upper, index=self.data.index),
            pd.Series(middle, index=self.data.index),
            pd.Series(lower, index=self.data.index)
        )

    def calculate_atr(self, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR)
        
        Args:
            period: ATR period (default: 14)
        
        Returns:
            ATR values as pandas Series
        """
        high = self.data['high'].astype(float).values
        low = self.data['low'].astype(float).values
        close = self.data['close'].astype(float).values
        atr = talib.ATR(high, low, close, timeperiod=period)
        return pd.Series(atr, index=self.data.index)

    def calculate_stochastic(self, fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator
        
        Args:
            fastk_period: Fast %K period (default: 14)
            slowk_period: Slow %K period (default: 3)
            slowd_period: Slow %D period (default: 3)
        
        Returns:
            Tuple of (Slow %K, Slow %D)
        """
        high = self.data['high'].astype(float).values
        low = self.data['low'].astype(float).values
        close = self.data['close'].astype(float).values
        slowk, slowd = talib.STOCH(high, low, close, 
                                  fastk_period=fastk_period,
                                  slowk_period=slowk_period,
                                  slowk_matype=0,
                                  slowd_period=slowd_period,
                                  slowd_matype=0)
        return (
            pd.Series(slowk, index=self.data.index),
            pd.Series(slowd, index=self.data.index)
        )

    def calculate_adx(self, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX)
        
        Args:
            period: ADX period (default: 14)
        
        Returns:
            ADX values as pandas Series
        """
        high = self.data['high'].astype(float).values
        low = self.data['low'].astype(float).values
        close = self.data['close'].astype(float).values
        adx = talib.ADX(high, low, close, timeperiod=period)
        return pd.Series(adx, index=self.data.index)

    def calculate_obv(self) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV)
        
        Returns:
            OBV values as pandas Series
        """
        close = self.data['close'].astype(float).values
        volume = self.data['volume'].astype(float).values
        obv = talib.OBV(close, volume)
        return pd.Series(obv, index=self.data.index)

    def calculate_momentum(self, period: int = 10) -> pd.Series:
        """
        Calculate Momentum indicator
        
        Args:
            period: Momentum period (default: 10)
        
        Returns:
            Momentum values as pandas Series
        """
        close = self.data['close'].astype(float).values
        momentum = talib.MOM(close, timeperiod=period)
        return pd.Series(momentum, index=self.data.index)

    def calculate_cci(self, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI)
        
        Args:
            period: CCI period (default: 20)
        
        Returns:
            CCI values as pandas Series
        """
        high = self.data['high'].astype(float).values
        low = self.data['low'].astype(float).values
        close = self.data['close'].astype(float).values
        cci = talib.CCI(high, low, close, timeperiod=period)
        return pd.Series(cci, index=self.data.index)

    def calculate_volume_features(self) -> pd.DataFrame:
        """
        Calculate volume-based features
        
        Returns:
            DataFrame with volume features
        """
        volume = self.data['volume'].astype(float).values
        close = self.data['close'].astype(float).values
        
        # Volume MA
        volume_ma = pd.Series(talib.SMA(volume, timeperiod=20), index=self.data.index)
        
        # Volume ROC
        volume_roc = pd.Series(talib.ROC(volume, timeperiod=10), index=self.data.index)
        
        # Volume Price Trend (custom implementation)
        price_change = np.diff(close, prepend=close[0])
        vpt = pd.Series(np.cumsum(volume * price_change), index=self.data.index)
        
        # Volume RSI
        volume_rsi = pd.Series(talib.RSI(volume, timeperiod=14), index=self.data.index)
        
        # Volume MACD
        volume_macd, volume_signal, volume_hist = talib.MACD(volume, fastperiod=12, slowperiod=26, signalperiod=9)
        volume_macd = pd.Series(volume_macd, index=self.data.index)
        volume_signal = pd.Series(volume_signal, index=self.data.index)
        volume_hist = pd.Series(volume_hist, index=self.data.index)
        
        return pd.DataFrame({
            'volume_ma': volume_ma,
            'volume_roc': volume_roc,
            'vpt': vpt,
            'volume_rsi': volume_rsi,
            'volume_macd': volume_macd,
            'volume_signal': volume_signal,
            'volume_hist': volume_hist
        })

    def add_technical_indicators(self):
        """Add technical indicators to the data"""
        try:
            # Trend indicators
            self.data['sma_20'] = SMAIndicator(close=self.data['close'], window=20).sma_indicator()
            self.data['sma_50'] = SMAIndicator(close=self.data['close'], window=50).sma_indicator()
            self.data['sma_200'] = SMAIndicator(close=self.data['close'], window=200).sma_indicator()
            
            self.data['ema_12'] = EMAIndicator(close=self.data['close'], window=12).ema_indicator()
            self.data['ema_26'] = EMAIndicator(close=self.data['close'], window=26).ema_indicator()
            
            macd = MACD(close=self.data['close'])
            self.data['macd'] = macd.macd()
            self.data['macd_signal'] = macd.macd_signal()
            self.data['macd_diff'] = macd.macd_diff()
            
            # Momentum indicators
            self.data['rsi'] = RSIIndicator(close=self.data['close']).rsi()
            
            stoch = StochasticOscillator(high=self.data['high'], low=self.data['low'], 
                                       close=self.data['close'])
            self.data['stoch_k'] = stoch.stoch()
            self.data['stoch_d'] = stoch.stoch_signal()
            
            self.data['williams_r'] = WilliamsRIndicator(high=self.data['high'], 
                                                       low=self.data['low'], 
                                                       close=self.data['close']).williams_r()
            
            # Volatility indicators
            bb = BollingerBands(close=self.data['close'])
            self.data['bb_high'] = bb.bollinger_hband()
            self.data['bb_low'] = bb.bollinger_lband()
            self.data['bb_mid'] = bb.bollinger_mavg()
            self.data['bb_width'] = (self.data['bb_high'] - self.data['bb_low']) / self.data['bb_mid']
            
            self.data['atr'] = AverageTrueRange(high=self.data['high'], 
                                              low=self.data['low'], 
                                              close=self.data['close']).average_true_range()
            
            # Volume indicators
            self.data['obv'] = OnBalanceVolumeIndicator(close=self.data['close'], 
                                                      volume=self.data['volume']).on_balance_volume()
            
            vwap = VolumeWeightedAveragePrice(high=self.data['high'], 
                                            low=self.data['low'], 
                                            close=self.data['close'], 
                                            volume=self.data['volume'])
            self.data['vwap'] = vwap.volume_weighted_average_price()
            
            # Market regime features
            self.data['trend_strength'] = ADXIndicator(high=self.data['high'], 
                                                     low=self.data['low'], 
                                                     close=self.data['close']).adx()
            
            # Price action features
            self.data['daily_return'] = DailyReturnIndicator(close=self.data['close']).daily_return()
            self.data['cumulative_return'] = CumulativeReturnIndicator(close=self.data['close']).cumulative_return()
            
            # Feature interactions
            self.data['price_volume_ratio'] = self.data['close'] / self.data['volume']
            self.data['volatility_ratio'] = self.data['atr'] / self.data['close']
            self.data['momentum_ratio'] = self.data['rsi'] / self.data['stoch_k']
            
            # Market regime classification
            self.data['market_regime'] = np.where(
                (self.data['close'] > self.data['sma_200']) & (self.data['trend_strength'] > 25), 
                'bullish',
                np.where(
                    (self.data['close'] < self.data['sma_200']) & (self.data['trend_strength'] > 25),
                    'bearish',
                    'neutral'
                )
            )
            
            # One-hot encode market regime
            self.data = pd.get_dummies(self.data, columns=['market_regime'])
            
            self.logger.info(f"Created {len(self.data.columns)} features")
            
        except Exception as e:
            self.logger.error(f"Error in add_technical_indicators: {str(e)}")
            raise

    def prepare_features(self):
        """Prepare features for model training"""
        try:
            # Add technical indicators
            self.add_technical_indicators()
            
            # Calculate target (1 if next day's return is positive, 0 otherwise)
            self.data['target'] = (self.data['close'].shift(-1) > self.data['close']).astype(int)
            
            # Drop rows with NaN values
            self.data.dropna(inplace=True)
            
            # Replace infinite values with NaN and then fill with mean
            feature_columns = [col for col in self.data.columns if col not in ['date', 'target']]
            self.data[feature_columns] = self.data[feature_columns].replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with column means
            for col in feature_columns:
                if self.data[col].isnull().any():
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
            
            # Scale features
            self.data[feature_columns] = self.scaler.fit_transform(self.data[feature_columns])
            
            return self.data[feature_columns], self.data['target']
            
        except Exception as e:
            self.logger.error(f"Error in prepare_features: {str(e)}")
            raise

    def generate_features(self, data):
        features = {
            'technical': {
                'rsi': self.calculate_rsi(),
                'macd': self.calculate_macd(),
                'bollinger_bands': self.calculate_bollinger_bands(),
                'atr': self.calculate_atr()
            },
            'price': {
                'returns': self.data['close'].pct_change(),
                'volatility': self.data['close'].pct_change().rolling(window=20).std(),
                'volume_profile': self.calculate_volume_features()
            },
            'market': {
                'sector_correlation': self._calculate_sector_correlation(data),
                'market_beta': self._calculate_market_beta(data)
            }
        }
        return features 

class MLPerformanceTracker:
    def track_performance(self, predictions, actual_outcomes):
        # Track accuracy
        accuracy = self._calculate_accuracy(predictions, actual_outcomes)
        
        # Track feature importance
        feature_importance = self._calculate_feature_importance()
        
        # Track model drift
        drift = self._detect_model_drift()
        
        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance,
            'drift': drift
        } 