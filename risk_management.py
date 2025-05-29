"""
Risk management module for the trading bot
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional
import yfinance as yf

class RiskManager:
    """Risk management class for position sizing and circuit breakers"""
    
    def __init__(self, capital: float = 1000000.0):
        """
        Initialize risk manager with capital
        
        Args:
            capital: Initial trading capital in INR
        """
        self.capital = capital
        self.daily_pnl = 0.0
        self.positions = {}
        self.trading_paused = False
        self.trading_stopped = False
        
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size based on 1% capital risk
        
        Args:
            entry_price: Entry price of the trade
            stop_loss: Stop loss price
            
        Returns:
            Number of shares to trade
        """
        risk_per_share = abs(entry_price - stop_loss)
        risk_amount = self.capital * 0.01  # 1% capital risk
        position_size = int(risk_amount / risk_per_share)
        return position_size
    
    def check_circuit_breakers(self) -> Dict[str, bool]:
        """
        Check circuit breaker conditions
        
        Returns:
            Dictionary with circuit breaker status
        """
        # Get India VIX data
        vix = yf.download('^INDIAVIX', period='1d')['Close'].iloc[-1]
        vix_prev = yf.download('^INDIAVIX', period='2d')['Close'].iloc[0]
        vix_rising = vix > vix_prev
        
        # Check VIX condition
        vix_condition = vix > 25 and vix_rising
        
        # Check daily loss condition
        daily_loss_condition = self.daily_pnl < -0.03 * self.capital
        
        # Update trading status
        if vix_condition:
            self.trading_paused = True
        if daily_loss_condition:
            self.trading_stopped = True
            
        return {
            'trading_paused': self.trading_paused,
            'trading_stopped': self.trading_stopped,
            'vix_condition': vix_condition,
            'daily_loss_condition': daily_loss_condition
        }
    
    def calculate_stop_loss(self, entry_price: float, atr: float, 
                          position_type: str = 'long') -> float:
        """
        Calculate dynamic stop loss
        
        Args:
            entry_price: Entry price of the trade
            atr: Average True Range
            position_type: 'long' or 'short'
            
        Returns:
            Stop loss price
        """
        # Calculate trailing stop loss
        trailing_sl = atr * 1.5
        
        # Calculate intraday stop loss
        intraday_sl = entry_price * 0.02  # 2% for equities
        
        # Use the tighter stop loss
        if position_type == 'long':
            sl = min(entry_price - trailing_sl, entry_price - intraday_sl)
        else:
            sl = max(entry_price + trailing_sl, entry_price + intraday_sl)
            
        return sl
    
    def update_daily_pnl(self, pnl: float):
        """
        Update daily P&L
        
        Args:
            pnl: Profit/loss for the day
        """
        self.daily_pnl += pnl
        
    def reset_daily_pnl(self):
        """Reset daily P&L at the start of each day"""
        self.daily_pnl = 0.0
        
    def add_position(self, symbol: str, entry_price: float, 
                    stop_loss: float, position_type: str):
        """
        Add a new position
        
        Args:
            symbol: Stock symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            position_type: 'long' or 'short'
        """
        position_size = self.calculate_position_size(entry_price, stop_loss)
        self.positions[symbol] = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'position_type': position_type,
            'size': position_size
        }
        
    def remove_position(self, symbol: str):
        """
        Remove a closed position
        
        Args:
            symbol: Stock symbol
        """
        if symbol in self.positions:
            del self.positions[symbol]
            
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position details
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position details or None if not found
        """
        return self.positions.get(symbol) 