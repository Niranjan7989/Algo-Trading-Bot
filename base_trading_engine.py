"""
Base trading engine with paper/live mode switching
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "PENDING"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"

@dataclass
class Fill:
    """Order fill details"""
    price: float
    quantity: int
    timestamp: datetime

@dataclass
class OrderResponse:
    """Unified order response format"""
    order_id: str
    status: OrderStatus
    fills: List[Fill]
    timestamp: datetime
    error_message: Optional[str] = None

class BaseTradingEngine(ABC):
    """Base trading engine with paper/live mode switching"""
    
    def __init__(self, capital: float = 1000000.0):
        """
        Initialize base trading engine
        
        Args:
            capital: Initial trading capital in INR
        """
        self.capital = capital
        self.trading_mode = os.getenv("TRADING_MODE", "PAPER").upper()
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.orders: Dict[str, Dict] = {}
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{self.trading_mode}")
        
    @property
    def is_paper_trading(self) -> bool:
        """Check if in paper trading mode"""
        return self.trading_mode == "PAPER"
        
    @property
    def is_live_trading(self) -> bool:
        """Check if in live trading mode"""
        return self.trading_mode == "LIVE"
        
    def log_trade(self, trade_type: str, symbol: str, 
                 price: float, quantity: int, 
                 order_id: Optional[str] = None):
        """
        Log trade execution
        
        Args:
            trade_type: Type of trade (BUY/SELL)
            symbol: Stock symbol
            price: Execution price
            quantity: Number of shares
            order_id: Optional order ID
        """
        trade = {
            'timestamp': datetime.now(),
            'trade_type': trade_type,
            'symbol': symbol,
            'price': price,
            'quantity': quantity,
            'order_id': order_id,
            'mode': self.trading_mode
        }
        self.trades.append(trade)
        self.logger.info(
            f"{trade_type} {quantity} shares of {symbol} at {price:.2f} "
            f"(Order ID: {order_id or 'N/A'})"
        )
        
    @abstractmethod
    def place_order(self, symbol: str, order_type: str, 
                   quantity: int, price: Optional[float] = None,
                   order_params: Optional[Dict] = None) -> OrderResponse:
        """
        Place an order (abstract method)
        
        Args:
            symbol: Stock symbol
            order_type: Type of order (BUY/SELL)
            quantity: Number of shares
            price: Optional limit price
            order_params: Additional order parameters
            
        Returns:
            OrderResponse object
        """
        pass
        
    def paper_place_order(self, symbol: str, order_type: str,
                         quantity: int, price: Optional[float] = None,
                         order_params: Optional[Dict] = None) -> OrderResponse:
        """
        Place a paper trading order
        
        Args:
            symbol: Stock symbol
            order_type: Type of order (BUY/SELL)
            quantity: Number of shares
            price: Optional limit price
            order_params: Additional order parameters
            
        Returns:
            Simulated OrderResponse
        """
        order_id = f"PAPER_{datetime.now().strftime('%Y%m%d%H%M%S')}_{symbol}"
        
        # Simulate order execution
        if price is None:
            # Market order - use current price
            price = self.get_current_price(symbol)
            
        # Simulate partial fills
        fills = []
        remaining_qty = quantity
        while remaining_qty > 0:
            fill_qty = min(remaining_qty, np.random.randint(1, remaining_qty + 1))
            fill_price = price * (1 + np.random.normal(0, 0.001))  # Simulate price movement
            fills.append(Fill(
                price=fill_price,
                quantity=fill_qty,
                timestamp=datetime.now()
            ))
            remaining_qty -= fill_qty
            
        # Update positions
        if order_type == "BUY":
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
                    'avg_price': price,
                    'entry_time': datetime.now()
                }
        else:  # SELL
            if symbol in self.positions:
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] == 0:
                    del self.positions[symbol]
                    
        # Log trade
        self.log_trade(order_type, symbol, price, quantity, order_id)
        
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.FILLED,
            fills=fills,
            timestamp=datetime.now()
        )
        
    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price of a symbol (abstract method)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current price
        """
        pass
        
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get current position for a symbol (abstract method)
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position details or None if no position
        """
        pass
        
    def get_portfolio_value(self) -> float:
        """
        Calculate current portfolio value
        
        Returns:
            Total portfolio value
        """
        total_value = self.capital
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            total_value += position['quantity'] * current_price
        return total_value
        
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get trade history as DataFrame
        
        Returns:
            DataFrame with trade history
        """
        return pd.DataFrame(self.trades)
        
    def get_position_summary(self) -> pd.DataFrame:
        """
        Get current positions summary
        
        Returns:
            DataFrame with position summary
        """
        positions = []
        for symbol, position in self.positions.items():
            current_price = self.get_current_price(symbol)
            positions.append({
                'symbol': symbol,
                'quantity': position['quantity'],
                'avg_price': position['avg_price'],
                'current_price': current_price,
                'pnl': (current_price - position['avg_price']) * position['quantity'],
                'entry_time': position['entry_time']
            })
        return pd.DataFrame(positions)
        
    def get_order_status(self, order_id: str) -> OrderResponse:
        """
        Get order status
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderResponse with current status
        """
        if order_id in self.orders:
            return self.orders[order_id]
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.FAILED,
            fills=[],
            timestamp=datetime.now(),
            error_message="Order not found"
        )
        
    def cancel_order(self, order_id: str) -> OrderResponse:
        """
        Cancel an order
        
        Args:
            order_id: Order ID
            
        Returns:
            OrderResponse with cancellation status
        """
        if order_id in self.orders:
            self.orders[order_id]['status'] = OrderStatus.CANCELLED
            return self.orders[order_id]
        return OrderResponse(
            order_id=order_id,
            status=OrderStatus.FAILED,
            fills=[],
            timestamp=datetime.now(),
            error_message="Order not found"
        ) 