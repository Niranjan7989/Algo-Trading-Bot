"""
Advanced order types for the trading bot
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import numpy as np
from .base_trading_engine import OrderResponse, OrderStatus, Fill

@dataclass
class OrderParams:
    """Base order parameters"""
    symbol: str
    quantity: int
    price: Optional[float] = None
    trigger_price: Optional[float] = None
    validity: Optional[timedelta] = None
    order_id: Optional[str] = None

@dataclass
class GTTOrder(OrderParams):
    """Good-Till-Trigger order parameters"""
    trigger_condition: str  # "LTP_GT" or "LTP_LT"
    trigger_value: float

@dataclass
class BracketOrder(OrderParams):
    """Bracket order parameters"""
    stop_loss: float
    target: float
    trailing_stop: Optional[float] = None

@dataclass
class IcebergOrder(OrderParams):
    """Iceberg order parameters"""
    display_quantity: int
    total_quantity: int

@dataclass
class TWAPOrder(OrderParams):
    """Time-Weighted Average Price order parameters"""
    start_time: datetime
    end_time: datetime
    interval: timedelta
    max_slippage: float = 0.001

class OrderValidator:
    """Order validation logic"""
    
    @staticmethod
    def validate_gtt_order(order: GTTOrder) -> List[str]:
        """Validate GTT order parameters"""
        errors = []
        if order.trigger_condition not in ["LTP_GT", "LTP_LT"]:
            errors.append("Invalid trigger condition")
        if order.trigger_value <= 0:
            errors.append("Trigger value must be positive")
        if order.quantity <= 0:
            errors.append("Quantity must be positive")
        return errors
        
    @staticmethod
    def validate_bracket_order(order: BracketOrder) -> List[str]:
        """Validate bracket order parameters"""
        errors = []
        if order.stop_loss >= order.price:
            errors.append("Stop loss must be below entry price")
        if order.target <= order.price:
            errors.append("Target must be above entry price")
        if order.trailing_stop and order.trailing_stop <= 0:
            errors.append("Trailing stop must be positive")
        return errors
        
    @staticmethod
    def validate_iceberg_order(order: IcebergOrder) -> List[str]:
        """Validate iceberg order parameters"""
        errors = []
        if order.display_quantity <= 0:
            errors.append("Display quantity must be positive")
        if order.total_quantity <= order.display_quantity:
            errors.append("Total quantity must be greater than display quantity")
        return errors
        
    @staticmethod
    def validate_twap_order(order: TWAPOrder) -> List[str]:
        """Validate TWAP order parameters"""
        errors = []
        if order.end_time <= order.start_time:
            errors.append("End time must be after start time")
        if order.interval.total_seconds() <= 0:
            errors.append("Interval must be positive")
        if order.max_slippage <= 0:
            errors.append("Max slippage must be positive")
        return errors

class OrderSimulator:
    """Order simulation for paper trading"""
    
    @staticmethod
    def simulate_gtt_fill(order: GTTOrder, current_price: float) -> Optional[OrderResponse]:
        """Simulate GTT order fill"""
        if order.trigger_condition == "LTP_GT" and current_price > order.trigger_value:
            return OrderResponse(
                order_id=order.order_id or f"GTT_{datetime.now().timestamp()}",
                status=OrderStatus.FILLED,
                fills=[Fill(price=current_price, quantity=order.quantity, 
                          timestamp=datetime.now())],
                timestamp=datetime.now()
            )
        elif order.trigger_condition == "LTP_LT" and current_price < order.trigger_value:
            return OrderResponse(
                order_id=order.order_id or f"GTT_{datetime.now().timestamp()}",
                status=OrderStatus.FILLED,
                fills=[Fill(price=current_price, quantity=order.quantity, 
                          timestamp=datetime.now())],
                timestamp=datetime.now()
            )
        return None
        
    @staticmethod
    def simulate_bracket_fill(order: BracketOrder, current_price: float) -> Optional[OrderResponse]:
        """Simulate bracket order fill"""
        if current_price <= order.stop_loss:
            return OrderResponse(
                order_id=order.order_id or f"BRACKET_{datetime.now().timestamp()}",
                status=OrderStatus.FILLED,
                fills=[Fill(price=order.stop_loss, quantity=order.quantity, 
                          timestamp=datetime.now())],
                timestamp=datetime.now()
            )
        elif current_price >= order.target:
            return OrderResponse(
                order_id=order.order_id or f"BRACKET_{datetime.now().timestamp()}",
                status=OrderStatus.FILLED,
                fills=[Fill(price=order.target, quantity=order.quantity, 
                          timestamp=datetime.now())],
                timestamp=datetime.now()
            )
        return None
        
    @staticmethod
    def simulate_iceberg_fill(order: IcebergOrder, current_price: float) -> OrderResponse:
        """Simulate iceberg order fill"""
        fills = []
        remaining_qty = order.total_quantity
        
        while remaining_qty > 0:
            fill_qty = min(remaining_qty, order.display_quantity)
            fill_price = current_price * (1 + np.random.normal(0, 0.001))
            fills.append(Fill(
                price=fill_price,
                quantity=fill_qty,
                timestamp=datetime.now()
            ))
            remaining_qty -= fill_qty
            
        return OrderResponse(
            order_id=order.order_id or f"ICEBERG_{datetime.now().timestamp()}",
            status=OrderStatus.FILLED,
            fills=fills,
            timestamp=datetime.now()
        )
        
    @staticmethod
    def simulate_twap_fill(order: TWAPOrder, current_price: float) -> OrderResponse:
        """Simulate TWAP order fill"""
        fills = []
        total_intervals = int((order.end_time - order.start_time) / order.interval)
        qty_per_interval = order.quantity / total_intervals
        
        for i in range(total_intervals):
            fill_price = current_price * (1 + np.random.normal(0, order.max_slippage))
            fills.append(Fill(
                price=fill_price,
                quantity=int(qty_per_interval),
                timestamp=order.start_time + i * order.interval
            ))
            
        return OrderResponse(
            order_id=order.order_id or f"TWAP_{datetime.now().timestamp()}",
            status=OrderStatus.FILLED,
            fills=fills,
            timestamp=datetime.now()
        ) 