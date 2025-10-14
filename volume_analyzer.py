"""
Volume Analysis System - Advanced volume pattern detection and analysis
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from config import get_config
from logger import get_logger, LogContext
from errors import DataSourceError, StockDataError

logger = get_logger(__name__)

class VolumePattern(Enum):
    """Types of volume patterns"""
    SURGE = "surge"
    SPIKES = "spikes"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT_VOLUME = "breakout_volume"
    EXHAUSTION = "exhaustion"

@dataclass
class VolumeAlert:
    """Volume analysis alert"""
    ticker: str
    pattern_type: VolumePattern
    current_volume: int
    avg_volume: int
    volume_ratio: float
    volume_percentile: float
    timestamp: datetime
    confidence: float
    price_correlation: float
    reason: str
    technical_details: Dict[str, Any]

class VolumeAnalyzer:
    """Advanced volume pattern analysis system"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # Volume thresholds
        self.surge_threshold = 2.0  # 2x average volume
        self.spike_threshold = 3.0  # 3x average volume
        self.breakout_threshold = 2.5  # 2.5x average volume
        self.accumulation_threshold = 1.5  # 1.5x average volume
        
        # Time windows
        self.short_window = 5
        self.medium_window = 20
        self.long_window = 50
        
        # Volume percentiles for comparison
        self.volume_percentiles = [50, 75, 90, 95, 99]
    
    async def analyze_volume_patterns(self, ticker: str, period: str = "1d", interval: str = "1m") -> Dict[str, Any]:
        """Analyze volume patterns for a ticker"""
        try:
            # Fetch historical data
            end_time = datetime.now()
            start_time = end_time - timedelta(days=7)  # Get 7 days of data
            
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_time, end=end_time, interval=interval)
            
            if hist.empty:
                return None
            
            # Calculate volume metrics
            volume_metrics = self._calculate_volume_metrics(hist)
            
            # Detect volume patterns
            patterns = self._detect_volume_patterns(ticker, hist, volume_metrics)
            
            # Calculate price-volume correlation
            price_volume_corr = self._calculate_price_volume_correlation(hist)
            
            return {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'volume_metrics': volume_metrics,
                'patterns': patterns,
                'price_volume_correlation': price_volume_corr,
                'data_points': len(hist)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns for {ticker}: {str(e)}")
            return None
    
    def _calculate_volume_metrics(self, hist: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive volume metrics"""
        volume = hist['Volume']
        close = hist['Close']
        
        # Basic volume statistics
        current_volume = volume.iloc[-1]
        avg_volume_short = volume.tail(self.short_window).mean()
        avg_volume_medium = volume.tail(self.medium_window).mean()
        avg_volume_long = volume.tail(self.long_window).mean()
        
        # Volume ratios
        volume_ratio_short = current_volume / avg_volume_short if avg_volume_short > 0 else 1
        volume_ratio_medium = current_volume / avg_volume_medium if avg_volume_medium > 0 else 1
        volume_ratio_long = current_volume / avg_volume_long if avg_volume_long > 0 else 1
        
        # Volume percentiles
        volume_percentiles = {}
        for p in self.volume_percentiles:
            volume_percentiles[f'p{p}'] = np.percentile(volume, p)
        
        # Current volume percentile
        current_percentile = (volume < current_volume).sum() / len(volume) * 100
        
        # Volume trend
        volume_trend = self._calculate_volume_trend(volume)
        
        # Volume volatility
        volume_volatility = volume.std() / volume.mean() if volume.mean() > 0 else 0
        
        # Price-volume relationship
        price_change = close.pct_change()
        volume_change = volume.pct_change()
        price_volume_correlation = price_change.corr(volume_change)
        
        return {
            'current_volume': int(current_volume),
            'avg_volume_short': float(avg_volume_short),
            'avg_volume_medium': float(avg_volume_medium),
            'avg_volume_long': float(avg_volume_long),
            'volume_ratio_short': float(volume_ratio_short),
            'volume_ratio_medium': float(volume_ratio_medium),
            'volume_ratio_long': float(volume_ratio_long),
            'volume_percentiles': volume_percentiles,
            'current_percentile': float(current_percentile),
            'volume_trend': volume_trend,
            'volume_volatility': float(volume_volatility),
            'price_volume_correlation': float(price_volume_correlation) if not pd.isna(price_volume_correlation) else 0
        }
    
    def _calculate_volume_trend(self, volume: pd.Series) -> str:
        """Calculate volume trend direction"""
        if len(volume) < 10:
            return "insufficient_data"
        
        # Calculate moving averages
        short_ma = volume.tail(5).mean()
        long_ma = volume.tail(20).mean()
        
        if short_ma > long_ma * 1.1:
            return "increasing"
        elif short_ma < long_ma * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _detect_volume_patterns(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> List[VolumeAlert]:
        """Detect various volume patterns"""
        patterns = []
        
        # Volume surge detection
        surge_alert = self._detect_volume_surge(ticker, hist, metrics)
        if surge_alert:
            patterns.append(surge_alert)
        
        # Volume spike detection
        spike_alert = self._detect_volume_spikes(ticker, hist, metrics)
        if spike_alert:
            patterns.append(spike_alert)
        
        # Accumulation pattern
        accum_alert = self._detect_accumulation(ticker, hist, metrics)
        if accum_alert:
            patterns.append(accum_alert)
        
        # Distribution pattern
        dist_alert = self._detect_distribution(ticker, hist, metrics)
        if dist_alert:
            patterns.append(dist_alert)
        
        # Breakout volume
        breakout_alert = self._detect_breakout_volume(ticker, hist, metrics)
        if breakout_alert:
            patterns.append(breakout_alert)
        
        # Exhaustion volume
        exhaustion_alert = self._detect_exhaustion_volume(ticker, hist, metrics)
        if exhaustion_alert:
            patterns.append(exhaustion_alert)
        
        return patterns
    
    def _detect_volume_surge(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[VolumeAlert]:
        """Detect sustained volume surge"""
        volume_ratio = metrics['volume_ratio_medium']
        current_percentile = metrics['current_percentile']
        
        if volume_ratio >= self.surge_threshold and current_percentile >= 75:
            confidence = min(volume_ratio / 3.0, 1.0)  # Cap at 3x volume
            
            return VolumeAlert(
                ticker=ticker,
                pattern_type=VolumePattern.SURGE,
                current_volume=metrics['current_volume'],
                avg_volume=int(metrics['avg_volume_medium']),
                volume_ratio=volume_ratio,
                volume_percentile=current_percentile,
                timestamp=datetime.now(),
                confidence=confidence,
                price_correlation=metrics['price_volume_correlation'],
                reason=f"Sustained volume surge: {volume_ratio:.1f}x average",
                technical_details={
                    'trend': metrics['volume_trend'],
                    'volatility': metrics['volume_volatility']
                }
            )
        
        return None
    
    def _detect_volume_spikes(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[VolumeAlert]:
        """Detect sudden volume spikes"""
        volume_ratio = metrics['volume_ratio_short']
        current_percentile = metrics['current_percentile']
        
        if volume_ratio >= self.spike_threshold and current_percentile >= 90:
            confidence = min(volume_ratio / 5.0, 1.0)  # Cap at 5x volume
            
            return VolumeAlert(
                ticker=ticker,
                pattern_type=VolumePattern.SPIKES,
                current_volume=metrics['current_volume'],
                avg_volume=int(metrics['avg_volume_short']),
                volume_ratio=volume_ratio,
                volume_percentile=current_percentile,
                timestamp=datetime.now(),
                confidence=confidence,
                price_correlation=metrics['price_volume_correlation'],
                reason=f"Volume spike: {volume_ratio:.1f}x recent average",
                technical_details={
                    'percentile': current_percentile,
                    'volatility': metrics['volume_volatility']
                }
            )
        
        return None
    
    def _detect_accumulation(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[VolumeAlert]:
        """Detect accumulation pattern (institutional buying)"""
        volume_ratio = metrics['volume_ratio_medium']
        price_volume_corr = metrics['price_volume_correlation']
        volume_trend = metrics['volume_trend']
        
        # Accumulation: increasing volume with stable/rising prices
        if (volume_ratio >= self.accumulation_threshold and 
            price_volume_corr > 0.3 and 
            volume_trend == "increasing"):
            
            confidence = min(volume_ratio / 2.0, 1.0)
            
            return VolumeAlert(
                ticker=ticker,
                pattern_type=VolumePattern.ACCUMULATION,
                current_volume=metrics['current_volume'],
                avg_volume=int(metrics['avg_volume_medium']),
                volume_ratio=volume_ratio,
                volume_percentile=metrics['current_percentile'],
                timestamp=datetime.now(),
                confidence=confidence,
                price_correlation=price_volume_corr,
                reason=f"Accumulation pattern: {volume_ratio:.1f}x volume with positive price correlation",
                technical_details={
                    'trend': volume_trend,
                    'correlation': price_volume_corr
                }
            )
        
        return None
    
    def _detect_distribution(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[VolumeAlert]:
        """Detect distribution pattern (institutional selling)"""
        volume_ratio = metrics['volume_ratio_medium']
        price_volume_corr = metrics['price_volume_correlation']
        volume_trend = metrics['volume_trend']
        
        # Distribution: high volume with declining prices
        if (volume_ratio >= self.accumulation_threshold and 
            price_volume_corr < -0.3 and 
            volume_trend == "increasing"):
            
            confidence = min(volume_ratio / 2.0, 1.0)
            
            return VolumeAlert(
                ticker=ticker,
                pattern_type=VolumePattern.DISTRIBUTION,
                current_volume=metrics['current_volume'],
                avg_volume=int(metrics['avg_volume_medium']),
                volume_ratio=volume_ratio,
                volume_percentile=metrics['current_percentile'],
                timestamp=datetime.now(),
                confidence=confidence,
                price_correlation=price_volume_corr,
                reason=f"Distribution pattern: {volume_ratio:.1f}x volume with negative price correlation",
                technical_details={
                    'trend': volume_trend,
                    'correlation': price_volume_corr
                }
            )
        
        return None
    
    def _detect_breakout_volume(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[VolumeAlert]:
        """Detect volume supporting price breakouts"""
        volume_ratio = metrics['volume_ratio_short']
        price_volume_corr = metrics['price_volume_correlation']
        
        # Breakout volume: high volume with strong price movement
        if (volume_ratio >= self.breakout_threshold and 
            price_volume_corr > 0.5):
            
            confidence = min(volume_ratio / 3.0, 1.0)
            
            return VolumeAlert(
                ticker=ticker,
                pattern_type=VolumePattern.BREAKOUT_VOLUME,
                current_volume=metrics['current_volume'],
                avg_volume=int(metrics['avg_volume_short']),
                volume_ratio=volume_ratio,
                volume_percentile=metrics['current_percentile'],
                timestamp=datetime.now(),
                confidence=confidence,
                price_correlation=price_volume_corr,
                reason=f"Breakout volume: {volume_ratio:.1f}x volume supporting price movement",
                technical_details={
                    'correlation': price_volume_corr,
                    'percentile': metrics['current_percentile']
                }
            )
        
        return None
    
    def _detect_exhaustion_volume(self, ticker: str, hist: pd.DataFrame, metrics: Dict[str, Any]) -> Optional[VolumeAlert]:
        """Detect exhaustion volume (end of trend)"""
        volume_ratio = metrics['volume_ratio_short']
        price_volume_corr = metrics['price_volume_correlation']
        volume_trend = metrics['volume_trend']
        
        # Exhaustion: very high volume with weak price movement
        if (volume_ratio >= self.spike_threshold and 
            abs(price_volume_corr) < 0.2 and 
            volume_trend == "decreasing"):
            
            confidence = min(volume_ratio / 4.0, 1.0)
            
            return VolumeAlert(
                ticker=ticker,
                pattern_type=VolumePattern.EXHAUSTION,
                current_volume=metrics['current_volume'],
                avg_volume=int(metrics['avg_volume_short']),
                volume_ratio=volume_ratio,
                volume_percentile=metrics['current_percentile'],
                timestamp=datetime.now(),
                confidence=confidence,
                price_correlation=price_volume_corr,
                reason=f"Exhaustion volume: {volume_ratio:.1f}x volume with weak price correlation",
                technical_details={
                    'trend': volume_trend,
                    'correlation': price_volume_corr
                }
            )
        
        return None
    
    def _calculate_price_volume_correlation(self, hist: pd.DataFrame) -> float:
        """Calculate price-volume correlation"""
        price_change = hist['Close'].pct_change().dropna()
        volume_change = hist['Volume'].pct_change().dropna()
        
        if len(price_change) < 2 or len(volume_change) < 2:
            return 0.0
        
        # Align the series
        min_len = min(len(price_change), len(volume_change))
        price_change = price_change.iloc[-min_len:]
        volume_change = volume_change.iloc[-min_len:]
        
        correlation = price_change.corr(volume_change)
        return correlation if not pd.isna(correlation) else 0.0
    
    async def analyze_multiple_tickers(self, tickers: List[str]) -> Dict[str, Any]:
        """Analyze volume patterns for multiple tickers"""
        results = {}
        
        for ticker in tickers:
            try:
                analysis = await self.analyze_volume_patterns(ticker)
                if analysis:
                    results[ticker] = analysis
            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {str(e)}")
        
        return results
    
    def get_volume_rankings(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank tickers by volume activity"""
        rankings = []
        
        for ticker, data in analysis_results.items():
            if 'volume_metrics' in data:
                metrics = data['volume_metrics']
                score = (
                    metrics['volume_ratio_medium'] * 0.4 +
                    metrics['current_percentile'] / 100 * 0.3 +
                    abs(metrics['price_volume_correlation']) * 0.3
                )
                
                rankings.append({
                    'ticker': ticker,
                    'score': score,
                    'volume_ratio': metrics['volume_ratio_medium'],
                    'percentile': metrics['current_percentile'],
                    'correlation': metrics['price_volume_correlation'],
                    'patterns': len(data.get('patterns', []))
                })
        
        return sorted(rankings, key=lambda x: x['score'], reverse=True)

# Convenience functions
async def analyze_ticker_volume(ticker: str) -> Dict[str, Any]:
    """Analyze volume patterns for a single ticker"""
    analyzer = VolumeAnalyzer()
    return await analyzer.analyze_volume_patterns(ticker)

async def get_top_volume_tickers(tickers: List[str], limit: int = 10) -> List[Dict[str, Any]]:
    """Get top volume activity tickers"""
    analyzer = VolumeAnalyzer()
    results = await analyzer.analyze_multiple_tickers(tickers)
    rankings = analyzer.get_volume_rankings(results)
    return rankings[:limit]
