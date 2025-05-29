"""
Risk management module for the trading bot
"""

import os
import logging
from typing import Dict, List, Optional, Set
import pandas as pd
import numpy as np
from datetime import datetime, time
import yfinance as yf
from .base_trading_engine import OrderResponse, OrderStatus

class RiskManager:
    """Risk management system for trading"""
    
    def __init__(self, capital: float = 1000000.0, 
                 max_risk_per_trade: float = 0.01,
                 max_sector_exposure: float = 0.2,
                 vix_threshold: float = 30.0,
                 max_daily_loss: float = 0.03):
        """
        Initialize risk manager
        
        Args:
            capital: Initial trading capital
            max_risk_per_trade: Maximum risk per trade as fraction of capital
            max_sector_exposure: Maximum exposure to any sector
            vix_threshold: VIX threshold for circuit breaker
            max_daily_loss: Maximum daily loss as fraction of capital
        """
        self.capital = capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_sector_exposure = max_sector_exposure
        self.vix_threshold = vix_threshold
        self.max_daily_loss = max_daily_loss
        
        self.positions: Dict[str, Dict] = {}
        self.sector_exposure: Dict[str, float] = {}
        self.blacklist: Set[str] = set()
        self.daily_pnl: float = 0.0
        self.last_reset_time: Optional[datetime] = None
        
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def reset_daily_pnl(self):
        """Reset daily P&L at market open"""
        current_time = datetime.now().time()
        market_open = time(9, 15)  # 9:15 AM
        
        if (self.last_reset_time is None or 
            (current_time >= market_open and 
             (self.last_reset_time.date() < datetime.now().date() or
              self.last_reset_time.time() < market_open))):
            self.daily_pnl = 0.0
            self.last_reset_time = datetime.now()
            self.logger.info("Daily P&L reset")
            
    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L
        
        Args:
            pnl: Profit/loss to add
        """
        self.reset_daily_pnl()
        self.daily_pnl += pnl
        self.logger.info(f"Daily P&L updated: {self.daily_pnl:.2f}")
        
    def check_circuit_breakers(self) -> List[str]:
        """
        Check circuit breaker conditions
        
        Returns:
            List of triggered circuit breakers
        """
        triggered = []
        
        # Check VIX
        vix = self._get_vix()
        if vix > self.vix_threshold:
            triggered.append(f"VIX above threshold: {vix:.2f}")
            
        # Check daily loss
        if self.daily_pnl < -self.max_daily_loss * self.capital:
            triggered.append(f"Daily loss limit reached: {self.daily_pnl:.2f}")
            
        # Check exchange status
        if not self._check_exchange_status():
            triggered.append("Exchange status check failed")
            
        return triggered
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
                              stop_loss: float) -> int:
        """
        Calculate position size based on risk
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            
        Returns:
            Number of shares to buy
        """
        if symbol in self.blacklist:
            self.logger.warning(f"Symbol {symbol} is blacklisted")
            return 0
            
        risk_per_share = abs(entry_price - stop_loss)
        max_risk_amount = self.capital * self.max_risk_per_trade
        position_size = int(max_risk_amount / risk_per_share)
        
        # Check sector exposure
        sector = self._get_sector(symbol)
        if sector:
            current_exposure = self.sector_exposure.get(sector, 0.0)
            max_sector_amount = self.capital * self.max_sector_exposure
            if current_exposure + (position_size * entry_price) > max_sector_amount:
                position_size = int((max_sector_amount - current_exposure) / entry_price)
                
        return max(0, position_size)
        
    def calculate_stop_loss(self, symbol: str, entry_price: float) -> float:
        """
        Calculate dynamic stop loss
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            
        Returns:
            Stop loss price
        """
        # Get ATR
        atr = self._calculate_atr(symbol)
        
        # Use 1.5x ATR for trailing stop
        trailing_stop = entry_price - (1.5 * atr)
        
        # Use 2% for intraday stop
        intraday_stop = entry_price * 0.98
        
        # Use the tighter stop
        return max(trailing_stop, intraday_stop)
        
    def update_position(self, symbol: str, quantity: int, 
                       price: float, trade_type: str):
        """
        Update position and sector exposure
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            price: Trade price
            trade_type: BUY or SELL
        """
        value = quantity * price
        
        if trade_type == "BUY":
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
                self.positions[symbol]['avg_price'] = (
                    (self.positions[symbol]['avg_price'] * 
                     (self.positions[symbol]['quantity'] - quantity) +
                     price * quantity) / self.positions[symbol]['quantity']
                )
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price
                }
                
            # Update sector exposure
            sector = self._get_sector(symbol)
            if sector:
                self.sector_exposure[sector] = (
                    self.sector_exposure.get(sector, 0.0) + value
                )
        else:  # SELL
            if symbol in self.positions:
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
                    
                # Update sector exposure
                sector = self._get_sector(symbol)
                if sector:
                    self.sector_exposure[sector] = (
                        self.sector_exposure.get(sector, 0.0) - value
                    )
                    
    def add_to_blacklist(self, symbol: str, reason: str):
        """
        Add symbol to blacklist
        
        Args:
            symbol: Stock symbol
            reason: Reason for blacklisting
        """
        self.blacklist.add(symbol)
        self.logger.warning(f"Added {symbol} to blacklist: {reason}")
        
    def remove_from_blacklist(self, symbol: str):
        """
        Remove symbol from blacklist
        
        Args:
            symbol: Stock symbol
        """
        if symbol in self.blacklist:
            self.blacklist.remove(symbol)
            self.logger.info(f"Removed {symbol} from blacklist")
            
    def _get_vix(self) -> float:
        """Get current VIX value"""
        try:
            vix = yf.download("^INDIAVIX", period="1d")['Close'].iloc[-1]
            return float(vix)
        except Exception as e:
            self.logger.error(f"Error fetching VIX: {e}")
            return 0.0
            
    def _check_exchange_status(self) -> bool:
        """Check exchange status"""
        # TODO: Implement actual exchange status check
        return True
        
    def _get_sector(self, symbol: str) -> Optional[str]:
        """Get sector for a symbol"""
        # TODO: Implement actual sector lookup
        return None
        
    def _calculate_atr(self, symbol: str, period: int = 14) -> float:
        """Calculate Average True Range"""
        try:
            data = yf.download(symbol, period=f"{period+1}d", interval="1d")
            high = data['High']
            low = data['Low']
            close = data['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean().iloc[-1]
            
            return float(atr)
        except Exception as e:
            self.logger.error(f"Error calculating ATR: {e}")
            return 0.0 