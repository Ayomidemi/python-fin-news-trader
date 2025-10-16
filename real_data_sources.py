"""
Real Data Sources - Professional market data and news APIs
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from dataclasses import dataclass
from enum import Enum
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources"""
    ALPHA_VANTAGE = "alpha_vantage"
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    NEWS_API = "news_api"
    YAHOO_FINANCE = "yahoo_finance"

@dataclass
class MarketData:
    """Standardized market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    source: str
    raw_data: Dict[str, Any] = None

@dataclass
class NewsData:
    """Standardized news data structure"""
    title: str
    content: str
    url: str
    source: str
    published_at: datetime
    sentiment_score: Optional[float] = None
    relevance_score: Optional[float] = None
    raw_data: Dict[str, Any] = None

class BaseDataSource(ABC):
    """Base class for data sources"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv(f"{self.__class__.__name__.upper()}_API_KEY")
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    @abstractmethod
    def get_stock_data(self, symbol: str, period: str = "1d") -> List[MarketData]:
        """Get stock market data"""
        pass
    
    @abstractmethod
    def get_news_data(self, symbol: str, limit: int = 10) -> List[NewsData]:
        """Get news data for a symbol"""
        pass

class AlphaVantageSource(BaseDataSource):
    """Alpha Vantage API data source"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://www.alphavantage.co/query"
        self.rate_limit_delay = 12.0  # 5 calls per minute
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> List[MarketData]:
        """Get stock data from Alpha Vantage"""
        self._rate_limit()
        
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': 'compact'
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return []
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return []
            
            time_series = data.get('Time Series (Daily)', {})
            market_data = []
            
            for date_str, values in time_series.items():
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.strptime(date_str, '%Y-%m-%d'),
                    open=float(values['1. open']),
                    high=float(values['2. high']),
                    low=float(values['3. low']),
                    close=float(values['4. close']),
                    volume=int(values['5. volume']),
                    source="alpha_vantage",
                    raw_data=values
                ))
            
            # Sort by timestamp (oldest first)
            market_data.sort(key=lambda x: x.timestamp)
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return []
    
    def get_news_data(self, symbol: str, limit: int = 10) -> List[NewsData]:
        """Get news data from Alpha Vantage"""
        self._rate_limit()
        
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'apikey': self.api_key,
                'limit': limit
            }
            
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage news error: {data['Error Message']}")
                return []
            
            news_data = []
            for article in data.get('feed', []):
                news_data.append(NewsData(
                    title=article.get('title', ''),
                    content=article.get('summary', ''),
                    url=article.get('url', ''),
                    source=article.get('source', 'Alpha Vantage'),
                    published_at=datetime.fromisoformat(article.get('time_published', '').replace('Z', '+00:00')),
                    sentiment_score=float(article.get('overall_sentiment_score', 0)),
                    relevance_score=float(article.get('relevance_score', 0)),
                    raw_data=article
                ))
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news for {symbol}: {str(e)}")
            return []

class IEXCloudSource(BaseDataSource):
    """IEX Cloud API data source"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://cloud.iexapis.com/stable"
        self.rate_limit_delay = 0.1  # 100 calls per second
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> List[MarketData]:
        """Get stock data from IEX Cloud"""
        self._rate_limit()
        
        try:
            # Map period to IEX Cloud range
            range_map = {
                "1d": "1d",
                "5d": "5d", 
                "1mo": "1m",
                "3mo": "3m",
                "6mo": "6m",
                "1y": "1y",
                "2y": "2y",
                "5y": "5y"
            }
            
            range_param = range_map.get(period, "1m")
            
            url = f"{self.base_url}/stock/{symbol}/chart/{range_param}"
            params = {'token': self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            market_data = []
            for item in data:
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.fromisoformat(item['date']),
                    open=item.get('open', 0),
                    high=item.get('high', 0),
                    low=item.get('low', 0),
                    close=item.get('close', 0),
                    volume=item.get('volume', 0),
                    source="iex_cloud",
                    raw_data=item
                ))
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching IEX Cloud data for {symbol}: {str(e)}")
            return []
    
    def get_news_data(self, symbol: str, limit: int = 10) -> List[NewsData]:
        """Get news data from IEX Cloud"""
        self._rate_limit()
        
        try:
            url = f"{self.base_url}/stock/{symbol}/news/last/{limit}"
            params = {'token': self.api_key}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            news_data = []
            for article in data:
                news_data.append(NewsData(
                    title=article.get('headline', ''),
                    content=article.get('summary', ''),
                    url=article.get('url', ''),
                    source=article.get('source', 'IEX Cloud'),
                    published_at=datetime.fromtimestamp(article.get('datetime', 0) / 1000),
                    raw_data=article
                ))
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching IEX Cloud news for {symbol}: {str(e)}")
            return []

class NewsAPISource(BaseDataSource):
    """NewsAPI data source"""
    
    def __init__(self, api_key: str = None):
        super().__init__(api_key)
        self.base_url = "https://newsapi.org/v2"
        self.rate_limit_delay = 1.0  # 1000 requests per day
    
    def get_stock_data(self, symbol: str, period: str = "1d") -> List[MarketData]:
        """NewsAPI doesn't provide market data"""
        return []
    
    def get_news_data(self, symbol: str, limit: int = 10) -> List[NewsData]:
        """Get news data from NewsAPI"""
        self._rate_limit()
        
        try:
            # Search for news about the company
            url = f"{self.base_url}/everything"
            params = {
                'q': f'"{symbol}" OR "{self._get_company_name(symbol)}"',
                'apiKey': self.api_key,
                'pageSize': limit,
                'sortBy': 'publishedAt',
                'language': 'en',
                'domains': 'bloomberg.com,reuters.com,cnbc.com,marketwatch.com,wsj.com'
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            news_data = []
            for article in data.get('articles', []):
                news_data.append(NewsData(
                    title=article.get('title', ''),
                    content=article.get('description', ''),
                    url=article.get('url', ''),
                    source=article.get('source', {}).get('name', 'NewsAPI'),
                    published_at=datetime.fromisoformat(article.get('publishedAt', '').replace('Z', '+00:00')),
                    raw_data=article
                ))
            
            return news_data
            
        except Exception as e:
            logger.error(f"Error fetching NewsAPI data for {symbol}: {str(e)}")
            return []
    
    def _get_company_name(self, symbol: str) -> str:
        """Get company name from symbol (simplified mapping)"""
        company_names = {
            'AAPL': 'Apple',
            'MSFT': 'Microsoft',
            'GOOGL': 'Google',
            'AMZN': 'Amazon',
            'META': 'Meta',
            'TSLA': 'Tesla',
            'NVDA': 'NVIDIA'
        }
        return company_names.get(symbol, symbol)

class DataSourceManager:
    """Manages multiple data sources with fallback"""
    
    def __init__(self):
        self.sources = {}
        self._initialize_sources()
    
    def _initialize_sources(self):
        """Initialize available data sources"""
        # Try to initialize each source
        try:
            self.sources[DataSource.ALPHA_VANTAGE] = AlphaVantageSource()
        except Exception as e:
            logger.warning(f"Alpha Vantage not available: {str(e)}")
        
        try:
            self.sources[DataSource.IEX_CLOUD] = IEXCloudSource()
        except Exception as e:
            logger.warning(f"IEX Cloud not available: {str(e)}")
        
        try:
            self.sources[DataSource.NEWS_API] = NewsAPISource()
        except Exception as e:
            logger.warning(f"NewsAPI not available: {str(e)}")
    
    def get_stock_data(self, symbol: str, period: str = "1d", preferred_source: DataSource = None) -> List[MarketData]:
        """Get stock data with fallback between sources"""
        sources_to_try = []
        
        if preferred_source and preferred_source in self.sources:
            sources_to_try.append(preferred_source)
        
        # Add other sources as fallback
        for source in self.sources:
            if source != preferred_source:
                sources_to_try.append(source)
        
        for source in sources_to_try:
            try:
                data = self.sources[source].get_stock_data(symbol, period)
                if data:
                    logger.info(f"Got stock data for {symbol} from {source.value}")
                    return data
            except Exception as e:
                logger.warning(f"Failed to get data from {source.value}: {str(e)}")
                continue
        
        logger.error(f"Failed to get stock data for {symbol} from any source")
        return []
    
    def get_news_data(self, symbol: str, limit: int = 10, preferred_source: DataSource = None) -> List[NewsData]:
        """Get news data with fallback between sources"""
        sources_to_try = []
        
        if preferred_source and preferred_source in self.sources:
            sources_to_try.append(preferred_source)
        
        # Add other sources as fallback
        for source in self.sources:
            if source != preferred_source:
                sources_to_try.append(source)
        
        all_news = []
        for source in sources_to_try:
            try:
                news = self.sources[source].get_news_data(symbol, limit)
                if news:
                    all_news.extend(news)
                    logger.info(f"Got {len(news)} news articles for {symbol} from {source.value}")
            except Exception as e:
                logger.warning(f"Failed to get news from {source.value}: {str(e)}")
                continue
        
        # Remove duplicates and sort by date
        unique_news = []
        seen_urls = set()
        for article in all_news:
            if article.url not in seen_urls:
                unique_news.append(article)
                seen_urls.add(article.url)
        
        unique_news.sort(key=lambda x: x.published_at, reverse=True)
        return unique_news[:limit]

# Global data source manager
data_manager = DataSourceManager()

# Convenience functions
def get_stock_data(symbol: str, period: str = "1d", source: DataSource = None) -> List[MarketData]:
    """Get stock data for a symbol"""
    return data_manager.get_stock_data(symbol, period, source)

def get_news_data(symbol: str, limit: int = 10, source: DataSource = None) -> List[NewsData]:
    """Get news data for a symbol"""
    return data_manager.get_news_data(symbol, limit, source)

def get_multiple_stocks_data(symbols: List[str], period: str = "1d") -> Dict[str, List[MarketData]]:
    """Get stock data for multiple symbols"""
    results = {}
    for symbol in symbols:
        results[symbol] = get_stock_data(symbol, period)
    return results

def get_multiple_news_data(symbols: List[str], limit: int = 10) -> Dict[str, List[NewsData]]:
    """Get news data for multiple symbols"""
    results = {}
    for symbol in symbols:
        results[symbol] = get_news_data(symbol, limit)
    return results
