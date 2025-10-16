"""
Real-Time Trading Engine with Order Execution
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN
import json
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """Order statuses"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

@dataclass
class Order:
    """Trading order"""
    id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: int = 0
    filled_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Position:
    """Trading position"""
    symbol: str
    quantity: int
    average_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

@dataclass
class Portfolio:
    """Trading portfolio"""
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    total_value: float = 0.0
    total_pnl: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class BrokerAPI(ABC):
    """Abstract base class for broker APIs"""
    
    @abstractmethod
    async def place_order(self, order: Order) -> bool:
        """Place an order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        pass

class AlpacaBroker(BrokerAPI):
    """Alpaca API implementation"""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = base_url or "https://paper-api.alpaca.markets"
        self.session = None
    
    async def _get_session(self):
        """Get HTTP session"""
        if self.session is None:
            import aiohttp
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def place_order(self, order: Order) -> bool:
        """Place order via Alpaca API"""
        try:
            session = await self._get_session()
            
            # Map order to Alpaca format
            alpaca_order = {
                "symbol": order.symbol,
                "qty": str(order.quantity),
                "side": order.side.value,
                "type": order.order_type.value,
                "time_in_force": "day"
            }
            
            if order.price:
                alpaca_order["limit_price"] = str(order.price)
            if order.stop_price:
                alpaca_order["stop_price"] = str(order.stop_price)
            
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json"
            }
            
            async with session.post(
                f"{self.base_url}/v2/orders",
                json=alpaca_order,
                headers=headers
            ) as response:
                if response.status == 201:
                    data = await response.json()
                    order.id = data["id"]
                    order.status = OrderStatus.SUBMITTED
                    logger.info(f"Order placed successfully: {order.id}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to place order: {error}")
                    order.status = OrderStatus.REJECTED
                    return False
                    
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            order.status = OrderStatus.REJECTED
            return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order via Alpaca API"""
        try:
            session = await self._get_session()
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            async with session.delete(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=headers
            ) as response:
                if response.status == 204:
                    logger.info(f"Order cancelled: {order_id}")
                    return True
                else:
                    error = await response.text()
                    logger.error(f"Failed to cancel order: {error}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error cancelling order: {str(e)}")
            return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status via Alpaca API"""
        try:
            session = await self._get_session()
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            async with session.get(
                f"{self.base_url}/v2/orders/{order_id}",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    status_map = {
                        "new": OrderStatus.PENDING,
                        "pending_new": OrderStatus.PENDING,
                        "accepted": OrderStatus.SUBMITTED,
                        "filled": OrderStatus.FILLED,
                        "partially_filled": OrderStatus.PARTIALLY_FILLED,
                        "canceled": OrderStatus.CANCELLED,
                        "rejected": OrderStatus.REJECTED
                    }
                    return status_map.get(data["status"], OrderStatus.PENDING)
                else:
                    logger.error(f"Failed to get order status: {response.status}")
                    return OrderStatus.PENDING
                    
        except Exception as e:
            logger.error(f"Error getting order status: {str(e)}")
            return OrderStatus.PENDING
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get positions via Alpaca API"""
        try:
            session = await self._get_session()
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            async with session.get(
                f"{self.base_url}/v2/positions",
                headers=headers
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    positions = {}
                    
                    for pos in data:
                        position = Position(
                            symbol=pos["symbol"],
                            quantity=int(pos["qty"]),
                            average_price=float(pos["avg_entry_price"]),
                            current_price=float(pos["current_price"]),
                            unrealized_pnl=float(pos["unrealized_pl"]),
                            realized_pnl=float(pos["realized_pl"])
                        )
                        positions[pos["symbol"]] = position
                    
                    return positions
                else:
                    logger.error(f"Failed to get positions: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting positions: {str(e)}")
            return {}
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info via Alpaca API"""
        try:
            session = await self._get_session()
            headers = {
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.secret_key
            }
            
            async with session.get(
                f"{self.base_url}/v2/account",
                headers=headers
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get account info: {response.status}")
                    return {}
                    
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return {}

class PaperTradingBroker(BrokerAPI):
    """Paper trading broker for simulation"""
    
    def __init__(self, initial_cash: float = 100000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
    
    async def place_order(self, order: Order) -> bool:
        """Simulate order placement"""
        try:
            self.order_counter += 1
            order.id = f"paper_{self.order_counter}"
            order.status = OrderStatus.SUBMITTED
            
            # Simulate order execution
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Check if we have enough cash for buy orders
            if order.side == OrderSide.BUY:
                required_cash = order.quantity * (order.price or 100.0)  # Default price
                if required_cash > self.cash:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Insufficient cash for order {order.id}")
                    return False
            
            # Check if we have enough shares for sell orders
            if order.side == OrderSide.SELL:
                if order.symbol not in self.positions or self.positions[order.symbol].quantity < order.quantity:
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Insufficient shares for order {order.id}")
                    return False
            
            # Simulate order fill
            order.status = OrderStatus.FILLED
            order.filled_quantity = order.quantity
            order.filled_price = order.price or 100.0  # Default price
            
            # Update portfolio
            self._update_portfolio(order)
            
            logger.info(f"Paper order filled: {order.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error placing paper order: {str(e)}")
            order.status = OrderStatus.REJECTED
            return False
    
    def _update_portfolio(self, order: Order):
        """Update portfolio after order fill"""
        if order.side == OrderSide.BUY:
            # Add to position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_quantity = pos.quantity + order.filled_quantity
                total_cost = (pos.quantity * pos.average_price) + (order.filled_quantity * order.filled_price)
                pos.average_price = total_cost / total_quantity
                pos.quantity = total_quantity
            else:
                self.positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.filled_quantity,
                    average_price=order.filled_price,
                    current_price=order.filled_price
                )
            
            # Deduct cash
            self.cash -= order.filled_quantity * order.filled_price
            
        elif order.side == OrderSide.SELL:
            # Remove from position
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                pos.quantity -= order.filled_quantity
                
                # Calculate realized P&L
                realized_pnl = (order.filled_price - pos.average_price) * order.filled_quantity
                pos.realized_pnl += realized_pnl
                
                # Remove position if quantity is 0
                if pos.quantity == 0:
                    del self.positions[order.symbol]
            
            # Add cash
            self.cash += order.filled_quantity * order.filled_price
    
    async def cancel_order(self, order_id: str) -> bool:
        """Simulate order cancellation"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            logger.info(f"Paper order cancelled: {order_id}")
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """Get order status"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.PENDING
    
    async def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.positions.copy()
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account info"""
        total_value = self.cash
        for pos in self.positions.values():
            total_value += pos.quantity * pos.current_price
        
        return {
            "cash": self.cash,
            "total_value": total_value,
            "buying_power": self.cash,
            "positions": len(self.positions)
        }

class TradingEngine:
    """Main trading engine"""
    
    def __init__(self, broker: BrokerAPI, portfolio: Portfolio = None):
        self.broker = broker
        self.portfolio = portfolio or Portfolio(cash=100000.0)
        self.orders = {}
        self.running = False
        self.event_handlers = {}
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def _emit_event(self, event_type: str, data: Any):
        """Emit event to handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in event handler: {str(e)}")
    
    async def place_order(self, symbol: str, side: OrderSide, quantity: int, 
                         order_type: OrderType = OrderType.MARKET, 
                         price: float = None, stop_price: float = None) -> str:
        """Place a trading order"""
        order = Order(
            id="",  # Will be set by broker
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price
        )
        
        success = await self.broker.place_order(order)
        if success:
            self.orders[order.id] = order
            self._emit_event("order_placed", order)
            return order.id
        else:
            self._emit_event("order_rejected", order)
            return None
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        if order_id in self.orders:
            success = await self.broker.cancel_order(order_id)
            if success:
                self.orders[order_id].status = OrderStatus.CANCELLED
                self._emit_event("order_cancelled", self.orders[order_id])
            return success
        return False
    
    async def update_portfolio(self):
        """Update portfolio from broker"""
        try:
            positions = await self.broker.get_positions()
            self.portfolio.positions = positions
            
            # Calculate total value
            total_value = self.portfolio.cash
            total_pnl = 0.0
            
            for symbol, position in positions.items():
                total_value += position.quantity * position.current_price
                total_pnl += position.unrealized_pnl + position.realized_pnl
            
            self.portfolio.total_value = total_value
            self.portfolio.total_pnl = total_pnl
            self.portfolio.updated_at = datetime.now()
            
            self._emit_event("portfolio_updated", self.portfolio)
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
    
    async def start(self):
        """Start the trading engine"""
        self.running = True
        logger.info("Trading engine started")
        
        while self.running:
            try:
                # Update portfolio
                await self.update_portfolio()
                
                # Check order statuses
                await self._check_order_statuses()
                
                # Wait before next update
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in trading engine loop: {str(e)}")
                await asyncio.sleep(5.0)
    
    async def stop(self):
        """Stop the trading engine"""
        self.running = False
        logger.info("Trading engine stopped")
    
    async def _check_order_statuses(self):
        """Check status of all pending orders"""
        for order_id, order in self.orders.items():
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                try:
                    new_status = await self.broker.get_order_status(order_id)
                    if new_status != order.status:
                        order.status = new_status
                        order.updated_at = datetime.now()
                        self._emit_event("order_status_changed", order)
                except Exception as e:
                    logger.error(f"Error checking order status: {str(e)}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get portfolio summary"""
        return {
            "cash": self.portfolio.cash,
            "total_value": self.portfolio.total_value,
            "total_pnl": self.portfolio.total_pnl,
            "positions_count": len(self.portfolio.positions),
            "orders_count": len(self.orders),
            "updated_at": self.portfolio.updated_at
        }
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions"""
        return self.portfolio.positions.copy()
    
    def get_orders(self) -> Dict[str, Order]:
        """Get all orders"""
        return self.orders.copy()

# Global trading engine instance
trading_engine = None

def initialize_trading_engine(broker_type: str = "paper", **kwargs) -> TradingEngine:
    """Initialize trading engine with specified broker"""
    global trading_engine
    
    if broker_type == "alpaca":
        broker = AlpacaBroker(
            api_key=kwargs.get("api_key"),
            secret_key=kwargs.get("secret_key"),
            base_url=kwargs.get("base_url")
        )
    else:  # Default to paper trading
        broker = PaperTradingBroker(
            initial_cash=kwargs.get("initial_cash", 100000.0)
        )
    
    trading_engine = TradingEngine(broker)
    return trading_engine

def get_trading_engine() -> TradingEngine:
    """Get the global trading engine instance"""
    if trading_engine is None:
        return initialize_trading_engine()
    return trading_engine
