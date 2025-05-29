"""
Main orchestrator for the trading bot
"""

import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import schedule
import time
from dotenv import load_dotenv

from data.market_data import MarketDataCollector
from data.sentiment_data import SentimentAnalyzer
from features.feature_engineer import FeatureEngineer
from preprocessing.preprocessor import DataPreprocessor
from models.predictor import ModelPredictor
from signals.generator import SignalGenerator
from risk.manager import RiskManager
from broker.base import BrokerAdapter
from performance.tracker import PerformanceTracker
from monitoring.logger import setup_logging
from monitoring.alerts import AlertSystem
from utils.helpers import load_config

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize trading bot
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        load_dotenv()
        
        # Setup logging
        self.logger = setup_logging()
        
        # Initialize components
        self._initialize_components()
        
        # Setup scheduling
        self._setup_scheduling()
        
    def _initialize_components(self):
        """Initialize all trading bot components"""
        try:
            # Data collection
            self.market_data = MarketDataCollector(self.config['market_data'])
            self.sentiment_analyzer = SentimentAnalyzer(self.config['sentiment'])
            
            # Feature engineering and preprocessing
            self.feature_engineer = FeatureEngineer(self.config['features'])
            self.preprocessor = DataPreprocessor(self.config['preprocessing'])
            
            # Models and signals
            self.model_predictor = ModelPredictor(self.config['models'])
            self.signal_generator = SignalGenerator(self.config['signals'])
            
            # Risk management
            self.risk_manager = RiskManager(self.config['risk'])
            
            # Broker integration
            self.broker = BrokerAdapter.create(self.config['broker'])
            
            # Performance tracking
            self.performance_tracker = PerformanceTracker(self.config['performance'])
            
            # Monitoring
            self.alert_system = AlertSystem(self.config['alerts'])
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
            
    def _setup_scheduling(self):
        """Setup scheduled tasks"""
        try:
            # Market data updates
            schedule.every(1).minutes.do(self._update_market_data)
            
            # Sentiment analysis
            schedule.every(15).minutes.do(self._update_sentiment)
            
            # Signal generation
            schedule.every(5).minutes.do(self._generate_signals)
            
            # Performance tracking
            schedule.every(1).hours.do(self._track_performance)
            
            # System health check
            schedule.every(30).minutes.do(self._check_system_health)
            
            self.logger.info("Scheduled tasks setup completed")
            
        except Exception as e:
            self.logger.error(f"Error setting up scheduling: {e}")
            raise
            
    def _update_market_data(self):
        """Update market data"""
        try:
            self.logger.info("Updating market data...")
            self.market_data.update_data()
            self.logger.info("Market data updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating market data: {e}")
            self.alert_system.send_alert("Market data update failed", str(e))
            
    def _update_sentiment(self):
        """Update sentiment analysis"""
        try:
            self.logger.info("Updating sentiment analysis...")
            for symbol in self.config['trading']['symbols']:
                sentiment_data = self.sentiment_analyzer.analyze_sentiment(symbol)
                self.sentiment_analyzer.save_sentiment_data(symbol, sentiment_data)
            self.logger.info("Sentiment analysis updated successfully")
        except Exception as e:
            self.logger.error(f"Error updating sentiment: {e}")
            self.alert_system.send_alert("Sentiment update failed", str(e))
            
    def _generate_signals(self):
        """Generate trading signals"""
        try:
            self.logger.info("Generating trading signals...")
            
            for symbol in self.config['trading']['symbols']:
                # Get market data
                market_data = self.market_data.get_data(symbol)
                
                # Get sentiment data
                sentiment_data = self.sentiment_analyzer.get_sentiment_data(symbol)
                
                # Generate features
                features = self.feature_engineer.generate_features(market_data, sentiment_data)
                
                # Preprocess features
                processed_features = self.preprocessor.transform(features)
                
                # Get model predictions
                predictions = self.model_predictor.predict(processed_features)
                
                # Generate signals
                signals = self.signal_generator.generate_signals(
                    market_data,
                    predictions,
                    sentiment_data
                )
                
                # Apply risk management
                signals = self.risk_manager.apply_risk_rules(signals)
                
                # Execute trades if signals are valid
                if signals['action'] != 'hold':
                    self._execute_trades(symbol, signals)
                    
            self.logger.info("Signal generation completed")
            
        except Exception as e:
            self.logger.error(f"Error generating signals: {e}")
            self.alert_system.send_alert("Signal generation failed", str(e))
            
    def _execute_trades(self, symbol: str, signals: Dict):
        """Execute trades based on signals"""
        try:
            self.logger.info(f"Executing trades for {symbol}...")
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                symbol,
                signals['confidence']
            )
            
            # Place order
            order = self.broker.place_order(
                symbol=symbol,
                action=signals['action'],
                quantity=position_size,
                order_type='MARKET'
            )
            
            # Track trade
            self.performance_tracker.add_trade({
                'symbol': symbol,
                'action': signals['action'],
                'quantity': position_size,
                'entry_price': order['price'],
                'confidence': signals['confidence'],
                'ml_prediction': signals.get('ml_prediction'),
                'technical_signals': signals.get('technical_signals'),
                'sentiment_signals': signals.get('sentiment_signals')
            })
            
            self.logger.info(f"Trade executed successfully for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {e}")
            self.alert_system.send_alert("Trade execution failed", str(e))
            
    def _track_performance(self):
        """Track trading performance"""
        try:
            self.logger.info("Tracking performance...")
            
            # Generate performance report
            report = self.performance_tracker.get_performance_report()
            
            # Save performance data
            self.performance_tracker.save_performance_data(
                self.config['performance']['data_directory']
            )
            
            # Check performance metrics
            if report['sharpe_ratio'] < self.config['performance']['min_sharpe']:
                self.alert_system.send_alert(
                    "Performance Alert",
                    f"Sharpe ratio below threshold: {report['sharpe_ratio']}"
                )
                
            self.logger.info("Performance tracking completed")
            
        except Exception as e:
            self.logger.error(f"Error tracking performance: {e}")
            self.alert_system.send_alert("Performance tracking failed", str(e))
            
    def _check_system_health(self):
        """Check system health"""
        try:
            self.logger.info("Checking system health...")
            
            # Check market data connection
            if not self.market_data.check_connection():
                self.alert_system.send_alert(
                    "System Health",
                    "Market data connection issue"
                )
                
            # Check broker connection
            if not self.broker.check_connection():
                self.alert_system.send_alert(
                    "System Health",
                    "Broker connection issue"
                )
                
            # Check API rate limits
            if self.market_data.check_rate_limit():
                self.alert_system.send_alert(
                    "System Health",
                    "Approaching API rate limit"
                )
                
            self.logger.info("System health check completed")
            
        except Exception as e:
            self.logger.error(f"Error checking system health: {e}")
            self.alert_system.send_alert("System health check failed", str(e))
            
    def run(self):
        """Run the trading bot"""
        try:
            self.logger.info("Starting trading bot...")
            
            while True:
                schedule.run_pending()
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Trading bot stopped by user")
        except Exception as e:
            self.logger.error(f"Error running trading bot: {e}")
            self.alert_system.send_alert("Trading bot error", str(e))
            raise
            
if __name__ == "__main__":
    bot = TradingBot()
    bot.run() 