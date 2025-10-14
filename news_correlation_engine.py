"""
News Correlation Engine - Links stock movements to news events and sentiment
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
import re
from collections import defaultdict

from config import get_config
from logger import get_logger, LogContext
from errors import NewsScrapingError, SentimentAnalysisError
from news_scraper import fetch_wsj_news
from sentiment_analyzer import analyze_sentiment, get_named_entities
from big_mover_tracker import BigMoverAlert

logger = get_logger(__name__)

class NewsImpact(Enum):
    """Types of news impact on stock price"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    HIGH_IMPACT = "high_impact"
    BREAKING = "breaking"

@dataclass
class NewsEvent:
    """News event data structure"""
    ticker: str
    title: str
    content: str
    url: str
    source: str
    timestamp: datetime
    sentiment_score: float
    impact_score: float
    impact_type: NewsImpact
    keywords: List[str]
    entities: Dict[str, List[str]]
    price_correlation: Optional[float] = None
    volume_correlation: Optional[float] = None

@dataclass
class NewsCorrelation:
    """Correlation between news and stock movement"""
    ticker: str
    news_event: NewsEvent
    price_movement: float
    price_movement_pct: float
    volume_movement: float
    time_difference: timedelta
    correlation_strength: float
    confidence: float
    impact_analysis: str

class NewsCorrelationEngine:
    """Engine for correlating news events with stock movements"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        
        # News impact keywords
        self.high_impact_keywords = {
            'earnings', 'revenue', 'profit', 'loss', 'guidance', 'forecast',
            'merger', 'acquisition', 'buyout', 'deal', 'partnership',
            'fda', 'approval', 'rejection', 'clinical', 'trial',
            'lawsuit', 'settlement', 'investigation', 'fine', 'penalty',
            'ceo', 'cfo', 'executive', 'resignation', 'appointment',
            'dividend', 'buyback', 'split', 'offering', 'ipo',
            'bankruptcy', 'restructuring', 'layoffs', 'hiring'
        }
        
        self.breaking_keywords = {
            'breaking', 'urgent', 'alert', 'just in', 'developing',
            'exclusive', 'sources say', 'confirmed', 'denied'
        }
        
        # Sentiment impact multipliers
        self.sentiment_multipliers = {
            'positive': 1.2,
            'negative': -1.2,
            'neutral': 0.1
        }
        
        # Time windows for correlation
        self.correlation_windows = {
            'immediate': timedelta(minutes=30),
            'short': timedelta(hours=2),
            'medium': timedelta(hours=6),
            'long': timedelta(days=1)
        }
    
    async def analyze_news_impact(self, ticker: str, time_window: timedelta = None) -> List[NewsEvent]:
        """Analyze news impact for a specific ticker"""
        if time_window is None:
            time_window = timedelta(hours=24)
        
        try:
            # Fetch recent news
            news_articles = fetch_wsj_news(ticker, max_articles=20)
            
            news_events = []
            cutoff_time = datetime.now() - time_window
            
            for article in news_articles:
                # Filter by time window
                if article['date'] < cutoff_time:
                    continue
                
                # Analyze sentiment
                sentiment_score = analyze_sentiment(
                    article['title'] + " " + article.get('content', ''),
                    financial_adjustment=True
                )
                
                # Extract entities
                entities = get_named_entities(article.get('content', ''))
                
                # Calculate impact score
                impact_score = self._calculate_impact_score(
                    article['title'],
                    article.get('content', ''),
                    sentiment_score
                )
                
                # Determine impact type
                impact_type = self._determine_impact_type(impact_score, sentiment_score)
                
                # Extract keywords
                keywords = self._extract_keywords(article['title'] + " " + article.get('content', ''))
                
                news_event = NewsEvent(
                    ticker=ticker,
                    title=article['title'],
                    content=article.get('content', ''),
                    url=article.get('url', ''),
                    source=article.get('source', 'Unknown'),
                    timestamp=article['date'],
                    sentiment_score=sentiment_score,
                    impact_score=impact_score,
                    impact_type=impact_type,
                    keywords=keywords,
                    entities=entities
                )
                
                news_events.append(news_event)
            
            # Sort by impact score
            news_events.sort(key=lambda x: x.impact_score, reverse=True)
            
            return news_events
            
        except Exception as e:
            logger.error(f"Error analyzing news impact for {ticker}: {str(e)}")
            return []
    
    def _calculate_impact_score(self, title: str, content: str, sentiment_score: float) -> float:
        """Calculate news impact score (0-1 scale)"""
        text = (title + " " + content).lower()
        impact_score = 0.0
        
        # Base sentiment impact
        sentiment_impact = abs(sentiment_score) * 0.3
        impact_score += sentiment_impact
        
        # High impact keywords
        high_impact_count = sum(1 for keyword in self.high_impact_keywords if keyword in text)
        high_impact_score = min(high_impact_count * 0.1, 0.4)
        impact_score += high_impact_score
        
        # Breaking news keywords
        breaking_count = sum(1 for keyword in self.breaking_keywords if keyword in text)
        breaking_score = min(breaking_count * 0.2, 0.3)
        impact_score += breaking_score
        
        # Title emphasis (titles are more important)
        title_impact = self._analyze_title_impact(title)
        impact_score += title_impact * 0.2
        
        # Content length factor (more detailed content = higher impact)
        content_length_factor = min(len(content) / 1000, 0.1)
        impact_score += content_length_factor
        
        return min(impact_score, 1.0)
    
    def _analyze_title_impact(self, title: str) -> float:
        """Analyze title for impact indicators"""
        title_lower = title.lower()
        impact_score = 0.0
        
        # Exclamation marks
        impact_score += title.count('!') * 0.1
        
        # Question marks (uncertainty)
        impact_score += title.count('?') * 0.05
        
        # Caps words (emphasis)
        caps_words = len(re.findall(r'\b[A-Z]{2,}\b', title))
        impact_score += caps_words * 0.05
        
        # Numbers (specific data)
        numbers = len(re.findall(r'\$[\d,]+|\d+%|\d+\.\d+', title))
        impact_score += numbers * 0.1
        
        return min(impact_score, 0.5)
    
    def _determine_impact_type(self, impact_score: float, sentiment_score: float) -> NewsImpact:
        """Determine the type of news impact"""
        if impact_score >= 0.7:
            return NewsImpact.HIGH_IMPACT
        elif impact_score >= 0.5:
            if sentiment_score > 0.2:
                return NewsImpact.POSITIVE
            elif sentiment_score < -0.2:
                return NewsImpact.NEGATIVE
            else:
                return NewsImpact.NEUTRAL
        elif sentiment_score > 0.3:
            return NewsImpact.POSITIVE
        elif sentiment_score < -0.3:
            return NewsImpact.NEGATIVE
        else:
            return NewsImpact.NEUTRAL
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        # Simple keyword extraction (can be enhanced with NLP)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'among', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count frequency and return top keywords
        from collections import Counter
        keyword_counts = Counter(keywords)
        return [word for word, count in keyword_counts.most_common(10)]
    
    async def correlate_news_with_movement(self, ticker: str, price_data: Dict[str, Any], 
                                         volume_data: Dict[str, Any]) -> List[NewsCorrelation]:
        """Correlate news events with stock movements"""
        try:
            # Get news events
            news_events = await self.analyze_news_impact(ticker)
            
            correlations = []
            
            for news_event in news_events:
                # Calculate time difference
                time_diff = datetime.now() - news_event.timestamp
                
                # Get price movement around news time
                price_movement = self._calculate_price_movement_around_news(
                    news_event, price_data, time_diff
                )
                
                # Get volume movement around news time
                volume_movement = self._calculate_volume_movement_around_news(
                    news_event, volume_data, time_diff
                )
                
                # Calculate correlation strength
                correlation_strength = self._calculate_correlation_strength(
                    news_event, price_movement, volume_movement, time_diff
                )
                
                # Calculate confidence
                confidence = self._calculate_correlation_confidence(
                    news_event, price_movement, volume_movement, time_diff
                )
                
                # Generate impact analysis
                impact_analysis = self._generate_impact_analysis(
                    news_event, price_movement, volume_movement, correlation_strength
                )
                
                correlation = NewsCorrelation(
                    ticker=ticker,
                    news_event=news_event,
                    price_movement=price_movement['absolute'],
                    price_movement_pct=price_movement['percentage'],
                    volume_movement=volume_movement['absolute'],
                    time_difference=time_diff,
                    correlation_strength=correlation_strength,
                    confidence=confidence,
                    impact_analysis=impact_analysis
                )
                
                correlations.append(correlation)
            
            # Sort by correlation strength
            correlations.sort(key=lambda x: x.correlation_strength, reverse=True)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Error correlating news with movement for {ticker}: {str(e)}")
            return []
    
    def _calculate_price_movement_around_news(self, news_event: NewsEvent, 
                                            price_data: Dict[str, Any], 
                                            time_diff: timedelta) -> Dict[str, float]:
        """Calculate price movement around news event"""
        # This is a simplified version - in practice, you'd need precise timing
        if 'price_history' not in price_data or len(price_data['price_history']) < 2:
            return {'absolute': 0.0, 'percentage': 0.0}
        
        prices = price_data['price_history']
        current_price = prices[-1]
        
        # Use price from before news (simplified)
        if len(prices) >= 2:
            prev_price = prices[-2]
            absolute_change = current_price - prev_price
            percentage_change = (absolute_change / prev_price) * 100
        else:
            absolute_change = 0.0
            percentage_change = 0.0
        
        return {
            'absolute': absolute_change,
            'percentage': percentage_change
        }
    
    def _calculate_volume_movement_around_news(self, news_event: NewsEvent, 
                                             volume_data: Dict[str, Any], 
                                             time_diff: timedelta) -> Dict[str, float]:
        """Calculate volume movement around news event"""
        if 'volume_history' not in volume_data or len(volume_data['volume_history']) < 2:
            return {'absolute': 0.0, 'ratio': 1.0}
        
        volumes = volume_data['volume_history']
        current_volume = volumes[-1]
        
        # Calculate average volume
        avg_volume = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        return {
            'absolute': current_volume - avg_volume,
            'ratio': volume_ratio
        }
    
    def _calculate_correlation_strength(self, news_event: NewsEvent, 
                                      price_movement: Dict[str, float], 
                                      volume_movement: Dict[str, float], 
                                      time_diff: timedelta) -> float:
        """Calculate correlation strength between news and movement"""
        strength = 0.0
        
        # Time factor (closer to news = higher correlation)
        if time_diff <= self.correlation_windows['immediate']:
            time_factor = 1.0
        elif time_diff <= self.correlation_windows['short']:
            time_factor = 0.8
        elif time_diff <= self.correlation_windows['medium']:
            time_factor = 0.6
        elif time_diff <= self.correlation_windows['long']:
            time_factor = 0.4
        else:
            time_factor = 0.2
        
        # News impact factor
        impact_factor = news_event.impact_score
        
        # Price movement factor
        price_factor = min(abs(price_movement['percentage']) / 10.0, 1.0)
        
        # Volume movement factor
        volume_factor = min(volume_movement['ratio'] / 3.0, 1.0)
        
        # Sentiment alignment factor
        sentiment_alignment = 0.0
        if news_event.sentiment_score > 0 and price_movement['percentage'] > 0:
            sentiment_alignment = 1.0
        elif news_event.sentiment_score < 0 and price_movement['percentage'] < 0:
            sentiment_alignment = 1.0
        elif abs(news_event.sentiment_score) < 0.1 and abs(price_movement['percentage']) < 1.0:
            sentiment_alignment = 0.5
        
        # Calculate overall strength
        strength = (
            time_factor * 0.3 +
            impact_factor * 0.3 +
            price_factor * 0.2 +
            volume_factor * 0.1 +
            sentiment_alignment * 0.1
        )
        
        return min(strength, 1.0)
    
    def _calculate_correlation_confidence(self, news_event: NewsEvent, 
                                        price_movement: Dict[str, float], 
                                        volume_movement: Dict[str, float], 
                                        time_diff: timedelta) -> float:
        """Calculate confidence in the correlation"""
        confidence = 0.0
        
        # High impact news = higher confidence
        if news_event.impact_type == NewsImpact.HIGH_IMPACT:
            confidence += 0.4
        elif news_event.impact_type in [NewsImpact.POSITIVE, NewsImpact.NEGATIVE]:
            confidence += 0.3
        else:
            confidence += 0.1
        
        # Strong price movement = higher confidence
        if abs(price_movement['percentage']) > 5:
            confidence += 0.3
        elif abs(price_movement['percentage']) > 2:
            confidence += 0.2
        else:
            confidence += 0.1
        
        # High volume = higher confidence
        if volume_movement['ratio'] > 2:
            confidence += 0.2
        elif volume_movement['ratio'] > 1.5:
            confidence += 0.1
        
        # Recent news = higher confidence
        if time_diff <= self.correlation_windows['short']:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_impact_analysis(self, news_event: NewsEvent, 
                                price_movement: Dict[str, float], 
                                volume_movement: Dict[str, float], 
                                correlation_strength: float) -> str:
        """Generate human-readable impact analysis"""
        analysis_parts = []
        
        # News impact
        if news_event.impact_type == NewsImpact.HIGH_IMPACT:
            analysis_parts.append("High-impact news")
        elif news_event.impact_type == NewsImpact.POSITIVE:
            analysis_parts.append("Positive news")
        elif news_event.impact_type == NewsImpact.NEGATIVE:
            analysis_parts.append("Negative news")
        else:
            analysis_parts.append("Neutral news")
        
        # Price movement
        if abs(price_movement['percentage']) > 5:
            analysis_parts.append(f"significant price movement ({price_movement['percentage']:.1f}%)")
        elif abs(price_movement['percentage']) > 2:
            analysis_parts.append(f"moderate price movement ({price_movement['percentage']:.1f}%)")
        
        # Volume movement
        if volume_movement['ratio'] > 2:
            analysis_parts.append(f"high volume ({volume_movement['ratio']:.1f}x average)")
        elif volume_movement['ratio'] > 1.5:
            analysis_parts.append(f"elevated volume ({volume_movement['ratio']:.1f}x average)")
        
        # Correlation strength
        if correlation_strength > 0.7:
            analysis_parts.append("strong correlation")
        elif correlation_strength > 0.5:
            analysis_parts.append("moderate correlation")
        else:
            analysis_parts.append("weak correlation")
        
        return " | ".join(analysis_parts)
    
    async def analyze_multiple_tickers(self, tickers: List[str]) -> Dict[str, List[NewsCorrelation]]:
        """Analyze news correlations for multiple tickers"""
        results = {}
        
        for ticker in tickers:
            try:
                # This is a simplified version - in practice, you'd need actual price/volume data
                correlations = await self.analyze_news_impact(ticker)
                results[ticker] = correlations
            except Exception as e:
                logger.error(f"Error analyzing news correlations for {ticker}: {str(e)}")
                results[ticker] = []
        
        return results

# Convenience functions
async def analyze_news_impact(ticker: str) -> List[NewsEvent]:
    """Analyze news impact for a single ticker"""
    engine = NewsCorrelationEngine()
    return await engine.analyze_news_impact(ticker)

async def correlate_news_with_movement(ticker: str, price_data: Dict[str, Any], 
                                     volume_data: Dict[str, Any]) -> List[NewsCorrelation]:
    """Correlate news with stock movement for a single ticker"""
    engine = NewsCorrelationEngine()
    return await engine.correlate_news_with_movement(ticker, price_data, volume_data)
