"""
Big Mover Tracker - Real-time monitoring system for identifying stocks before they skyrocket
"""

import asyncio
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
from enum import Enum
import logging

from config import get_config
from logger import get_logger, LogContext
from errors import DataSourceError, StockDataError, ErrorRecovery
from cache import get_cache_manager

logger = get_logger(__name__)

class MovementType(Enum):
    """Types of big movements to track"""
    PRICE_SPIKE = "price_spike"
    VOLUME_SURGE = "volume_surge"
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    BREAKOUT = "breakout"
    MOMENTUM = "momentum"

@dataclass
class BigMoverAlert:
    """Alert data structure for big movers"""
    ticker: str
    movement_type: MovementType
    current_price: float
    price_change: float
    price_change_pct: float
    volume: int
    avg_volume: int
    volume_ratio: float
    timestamp: datetime
    confidence_score: float
    technical_indicators: Dict[str, Any]
    news_correlation: Optional[Dict[str, Any]] = None
    reason: str = ""

class BigMoverTracker:
    """Main class for tracking big movers in real-time"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.cache = get_cache_manager()
        self.monitored_tickers = []
        self.price_history = {}
        self.volume_history = {}
        self.alerts = []
        self.is_running = False
        
        # Movement thresholds
        self.price_spike_threshold = 0.05  # 5% price spike
        self.volume_surge_threshold = 2.0  # 2x average volume
        self.gap_threshold = 0.03  # 3% gap
        self.momentum_threshold = 0.02  # 2% momentum
        
        # Time windows for analysis
        self.short_window = 5  # 5 periods for short-term analysis
        self.medium_window = 20  # 20 periods for medium-term analysis
        self.long_window = 50  # 50 periods for long-term analysis
    
    def add_ticker(self, ticker: str):
        """Add a ticker to monitor"""
        if ticker not in self.monitored_tickers:
            self.monitored_tickers.append(ticker)
            self.price_history[ticker] = []
            self.volume_history[ticker] = []
            logger.info(f"Added {ticker} to big mover monitoring")
    
    def remove_ticker(self, ticker: str):
        """Remove a ticker from monitoring"""
        if ticker in self.monitored_tickers:
            self.monitored_tickers.remove(ticker)
            if ticker in self.price_history:
                del self.price_history[ticker]
            if ticker in self.volume_history:
                del self.volume_history[ticker]
            logger.info(f"Removed {ticker} from big mover monitoring")
    
    async def fetch_realtime_data(self, tickers: List[str]) -> Dict[str, Dict[str, Any]]:
        """Fetch real-time data for multiple tickers"""
        data = {}
        
        for ticker in tickers:
            try:
                # Get 1-minute data for the last hour
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=1)
                
                stock = yf.Ticker(ticker)
                hist = stock.history(start=start_time, end=end_time, interval="1m")
                
                if not hist.empty:
                    # Get current price and volume
                    current_price = hist['Close'].iloc[-1]
                    current_volume = hist['Volume'].iloc[-1]
                    
                    # Calculate price change
                    if len(hist) > 1:
                        prev_price = hist['Close'].iloc[-2]
                        price_change = current_price - prev_price
                        price_change_pct = (price_change / prev_price) * 100
                    else:
                        price_change = 0
                        price_change_pct = 0
                    
                    # Calculate average volume (last 20 periods)
                    avg_volume = hist['Volume'].tail(20).mean() if len(hist) >= 20 else current_volume
                    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                    
                    data[ticker] = {
                        'current_price': current_price,
                        'price_change': price_change,
                        'price_change_pct': price_change_pct,
                        'current_volume': current_volume,
                        'avg_volume': avg_volume,
                        'volume_ratio': volume_ratio,
                        'timestamp': datetime.now(),
                        'price_history': hist['Close'].tolist(),
                        'volume_history': hist['Volume'].tolist()
                    }
                    
            except Exception as e:
                logger.error(f"Error fetching real-time data for {ticker}: {str(e)}")
                data[ticker] = None
        
        return data
    
    def detect_price_spike(self, ticker: str, data: Dict[str, Any]) -> Optional[BigMoverAlert]:
        """Detect significant price spikes"""
        if not data or 'price_change_pct' not in data:
            return None
        
        price_change_pct = abs(data['price_change_pct'])
        
        if price_change_pct >= self.price_spike_threshold * 100:
            movement_type = MovementType.GAP_UP if data['price_change_pct'] > 0 else MovementType.GAP_DOWN
            
            # Calculate confidence score based on volume and technical indicators
            confidence = self._calculate_confidence_score(data)
            
            alert = BigMoverAlert(
                ticker=ticker,
                movement_type=movement_type,
                current_price=data['current_price'],
                price_change=data['price_change'],
                price_change_pct=data['price_change_pct'],
                volume=data['current_volume'],
                avg_volume=data['avg_volume'],
                volume_ratio=data['volume_ratio'],
                timestamp=data['timestamp'],
                confidence_score=confidence,
                technical_indicators=self._calculate_technical_indicators(data),
                reason=f"Price {'spike' if data['price_change_pct'] > 0 else 'drop'} of {price_change_pct:.2f}%"
            )
            
            return alert
        
        return None
    
    def detect_volume_surge(self, ticker: str, data: Dict[str, Any]) -> Optional[BigMoverAlert]:
        """Detect unusual volume surges"""
        if not data or 'volume_ratio' not in data:
            return None
        
        volume_ratio = data['volume_ratio']
        
        if volume_ratio >= self.volume_surge_threshold:
            # Calculate confidence score
            confidence = self._calculate_confidence_score(data)
            
            alert = BigMoverAlert(
                ticker=ticker,
                movement_type=MovementType.VOLUME_SURGE,
                current_price=data['current_price'],
                price_change=data['price_change'],
                price_change_pct=data['price_change_pct'],
                volume=data['current_volume'],
                avg_volume=data['avg_volume'],
                volume_ratio=volume_ratio,
                timestamp=data['timestamp'],
                confidence_score=confidence,
                technical_indicators=self._calculate_technical_indicators(data),
                reason=f"Volume surge: {volume_ratio:.1f}x average volume"
            )
            
            return alert
        
        return None
    
    def detect_gap(self, ticker: str, data: Dict[str, Any]) -> Optional[BigMoverAlert]:
        """Detect significant gaps (pre-market or after-hours)"""
        if not data or 'price_history' not in data:
            return None
        
        price_history = data['price_history']
        if len(price_history) < 2:
            return None
        
        # Calculate gap from previous close to current price
        prev_close = price_history[-2]
        current_price = price_history[-1]
        gap_pct = ((current_price - prev_close) / prev_close) * 100
        
        if abs(gap_pct) >= self.gap_threshold * 100:
            movement_type = MovementType.GAP_UP if gap_pct > 0 else MovementType.GAP_DOWN
            
            confidence = self._calculate_confidence_score(data)
            
            alert = BigMoverAlert(
                ticker=ticker,
                movement_type=movement_type,
                current_price=current_price,
                price_change=current_price - prev_close,
                price_change_pct=gap_pct,
                volume=data['current_volume'],
                avg_volume=data['avg_volume'],
                volume_ratio=data['volume_ratio'],
                timestamp=data['timestamp'],
                confidence_score=confidence,
                technical_indicators=self._calculate_technical_indicators(data),
                reason=f"Gap {'up' if gap_pct > 0 else 'down'} of {gap_pct:.2f}%"
            )
            
            return alert
        
        return None
    
    def detect_breakout(self, ticker: str, data: Dict[str, Any]) -> Optional[BigMoverAlert]:
        """Detect technical breakouts"""
        if not data or 'price_history' not in data:
            return None
        
        price_history = data['price_history']
        if len(price_history) < self.medium_window:
            return None
        
        # Calculate resistance and support levels
        recent_highs = max(price_history[-self.medium_window:])
        recent_lows = min(price_history[-self.medium_window:])
        current_price = price_history[-1]
        
        # Check for breakout above resistance
        if current_price > recent_highs * 1.01:  # 1% above recent high
            confidence = self._calculate_confidence_score(data)
            
            alert = BigMoverAlert(
                ticker=ticker,
                movement_type=MovementType.BREAKOUT,
                current_price=current_price,
                price_change=current_price - recent_highs,
                price_change_pct=((current_price - recent_highs) / recent_highs) * 100,
                volume=data['current_volume'],
                avg_volume=data['avg_volume'],
                volume_ratio=data['volume_ratio'],
                timestamp=data['timestamp'],
                confidence_score=confidence,
                technical_indicators=self._calculate_technical_indicators(data),
                reason=f"Breakout above resistance at ${recent_highs:.2f}"
            )
            
            return alert
        
        return None
    
    def detect_momentum(self, ticker: str, data: Dict[str, Any]) -> Optional[BigMoverAlert]:
        """Detect momentum shifts"""
        if not data or 'price_history' not in data:
            return None
        
        price_history = data['price_history']
        if len(price_history) < self.short_window:
            return None
        
        # Calculate momentum over short window
        recent_prices = price_history[-self.short_window:]
        momentum = (recent_prices[-1] - recent_prices[0]) / recent_prices[0] * 100
        
        if abs(momentum) >= self.momentum_threshold * 100:
            confidence = self._calculate_confidence_score(data)
            
            alert = BigMoverAlert(
                ticker=ticker,
                movement_type=MovementType.MOMENTUM,
                current_price=recent_prices[-1],
                price_change=recent_prices[-1] - recent_prices[0],
                price_change_pct=momentum,
                volume=data['current_volume'],
                avg_volume=data['avg_volume'],
                volume_ratio=data['volume_ratio'],
                timestamp=data['timestamp'],
                confidence_score=confidence,
                technical_indicators=self._calculate_technical_indicators(data),
                reason=f"Momentum shift: {momentum:.2f}% over {self.short_window} periods"
            )
            
            return alert
        
        return None
    
    def _calculate_confidence_score(self, data: Dict[str, Any]) -> float:
        """Calculate confidence score for an alert (0-1 scale)"""
        confidence = 0.0
        
        # Volume factor (higher volume = higher confidence)
        if 'volume_ratio' in data:
            volume_factor = min(data['volume_ratio'] / 3.0, 1.0)  # Cap at 3x volume
            confidence += volume_factor * 0.3
        
        # Price movement factor
        if 'price_change_pct' in data:
            price_factor = min(abs(data['price_change_pct']) / 10.0, 1.0)  # Cap at 10%
            confidence += price_factor * 0.3
        
        # Technical indicators factor
        if 'technical_indicators' in data:
            tech_factor = self._calculate_technical_confidence(data['technical_indicators'])
            confidence += tech_factor * 0.4
        
        return min(confidence, 1.0)
    
    def _calculate_technical_indicators(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate technical indicators for the data"""
        if 'price_history' not in data or len(data['price_history']) < 20:
            return {}
        
        prices = np.array(data['price_history'])
        
        # Simple Moving Averages
        sma_5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
        sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else prices[-1]
        
        # RSI calculation
        rsi = self._calculate_rsi(prices)
        
        # Price position relative to moving averages
        current_price = prices[-1]
        above_sma_5 = current_price > sma_5
        above_sma_20 = current_price > sma_20
        
        return {
            'sma_5': sma_5,
            'sma_20': sma_20,
            'rsi': rsi,
            'above_sma_5': above_sma_5,
            'above_sma_20': above_sma_20,
            'trend_strength': (sma_5 - sma_20) / sma_20 * 100 if sma_20 > 0 else 0
        }
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_technical_confidence(self, indicators: Dict[str, Any]) -> float:
        """Calculate confidence based on technical indicators"""
        confidence = 0.0
        
        # RSI factor
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if 30 <= rsi <= 70:  # Not overbought/oversold
                confidence += 0.3
            elif 20 <= rsi <= 80:  # Moderate levels
                confidence += 0.2
        
        # Moving average alignment
        if 'above_sma_5' in indicators and 'above_sma_20' in indicators:
            if indicators['above_sma_5'] and indicators['above_sma_20']:
                confidence += 0.4  # Strong uptrend
            elif indicators['above_sma_5']:
                confidence += 0.2  # Short-term uptrend
        
        # Trend strength
        if 'trend_strength' in indicators:
            trend = abs(indicators['trend_strength'])
            if trend > 2:  # Strong trend
                confidence += 0.3
            elif trend > 1:  # Moderate trend
                confidence += 0.2
        
        return min(confidence, 1.0)
    
    async def scan_for_movers(self) -> List[BigMoverAlert]:
        """Scan all monitored tickers for big movers"""
        if not self.monitored_tickers:
            return []
        
        logger.info(f"Scanning {len(self.monitored_tickers)} tickers for big movers")
        
        # Fetch real-time data
        data = await self.fetch_realtime_data(self.monitored_tickers)
        
        alerts = []
        
        for ticker, ticker_data in data.items():
            if ticker_data is None:
                continue
            
            # Run all detection methods
            detection_methods = [
                self.detect_price_spike,
                self.detect_volume_surge,
                self.detect_gap,
                self.detect_breakout,
                self.detect_momentum
            ]
            
            for method in detection_methods:
                try:
                    alert = method(ticker, ticker_data)
                    if alert:
                        alerts.append(alert)
                        logger.info(f"Big mover detected: {ticker} - {alert.movement_type.value} - {alert.reason}")
                except Exception as e:
                    logger.error(f"Error in detection method {method.__name__} for {ticker}: {str(e)}")
        
        # Sort alerts by confidence score
        alerts.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Store alerts
        self.alerts.extend(alerts)
        
        return alerts
    
    async def start_monitoring(self, scan_interval: int = 60):
        """Start continuous monitoring of big movers"""
        self.is_running = True
        logger.info(f"Starting big mover monitoring with {scan_interval}s interval")
        
        while self.is_running:
            try:
                alerts = await self.scan_for_movers()
                if alerts:
                    logger.info(f"Found {len(alerts)} big mover alerts")
                
                await asyncio.sleep(scan_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(scan_interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        logger.info("Stopped big mover monitoring")
    
    def get_recent_alerts(self, hours: int = 24) -> List[BigMoverAlert]:
        """Get alerts from the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
    
    def get_top_movers(self, limit: int = 10) -> List[BigMoverAlert]:
        """Get top movers by confidence score"""
        return sorted(self.alerts, key=lambda x: x.confidence_score, reverse=True)[:limit]

# Convenience functions for easy integration
def create_big_mover_tracker(tickers: List[str] = None) -> BigMoverTracker:
    """Create and configure a big mover tracker"""
    tracker = BigMoverTracker()
    
    if tickers:
        for ticker in tickers:
            tracker.add_ticker(ticker)
    
    return tracker

async def scan_market_for_movers(tickers: List[str]) -> List[BigMoverAlert]:
    """One-time scan for big movers"""
    tracker = create_big_mover_tracker(tickers)
    return await tracker.scan_for_movers()
