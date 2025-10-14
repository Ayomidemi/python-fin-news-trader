"""
Async processing system for FinNewsTrader to enable parallel data fetching
"""

import asyncio
import aiohttp
import concurrent.futures
from typing import List, Dict, Any, Callable, Optional, Tuple
from datetime import datetime
import time

from config import get_config
from logger import get_logger, LogContext
from errors import DataSourceError, NewsScrapingError, StockDataError, ErrorRecovery
from cache import get_cache_manager, cached

logger = get_logger(__name__)

class AsyncProcessor:
    """Main async processor for parallel operations"""
    
    def __init__(self, max_workers: int = None):
        self.config = get_config()
        self.max_workers = max_workers or min(32, (len(self.config.ui.available_stocks) * 2) + 4)
        self.session = None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers)
        
    async def __aenter__(self):
        """Async context manager entry"""
        timeout = aiohttp.ClientTimeout(total=self.config.data_sources.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
        self.executor.shutdown(wait=True)
    
    async def run_in_executor(self, func: Callable, *args) -> Any:
        """Run a synchronous function in the thread pool executor"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)
    
    async def fetch_multiple_news(self, tickers: List[str], max_articles: int = None) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple tickers in parallel
        
        Args:
            tickers: List of stock ticker symbols
            max_articles: Maximum articles per ticker
            
        Returns:
            Dictionary mapping ticker to list of news articles
        """
        if max_articles is None:
            max_articles = self.config.data_sources.max_articles_per_stock
        
        logger.info(f"Fetching news for {len(tickers)} tickers in parallel")
        
        # Create tasks for each ticker
        tasks = []
        for ticker in tickers:
            task = self._fetch_single_ticker_news(ticker, max_articles)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        news_data = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching news for {ticker}: {str(result)}")
                news_data[ticker] = []
            else:
                news_data[ticker] = result
                logger.info(f"Fetched {len(result)} articles for {ticker}")
        
        return news_data
    
    async def _fetch_single_ticker_news(self, ticker: str, max_articles: int) -> List[Dict]:
        """Fetch news for a single ticker with caching and error handling"""
        cache = get_cache_manager()
        
        # Check cache first
        cached_news = cache.get(cache.cache_key("news", ticker, datetime.now().date().isoformat()))
        if cached_news is not None:
            logger.debug(f"Using cached news for {ticker}")
            return cached_news
        
        # Fetch from source
        try:
            # Import here to avoid circular imports
            from news_scraper import fetch_wsj_news
            
            # Run the synchronous news fetching in executor
            news_articles = await self.run_in_executor(fetch_wsj_news, ticker, max_articles)
            
            # Cache the results
            cache.set(
                cache.cache_key("news", ticker, datetime.now().date().isoformat()),
                news_articles,
                self.config.data_sources.cache_ttl_seconds
            )
            
            return news_articles
            
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {str(e)}")
            raise NewsScrapingError(f"Failed to fetch news for {ticker}: {str(e)}")
    
    async def fetch_multiple_stock_data(self, tickers: List[str], period: str = None) -> Dict[str, Any]:
        """
        Fetch stock data for multiple tickers in parallel
        
        Args:
            tickers: List of stock ticker symbols
            period: Data period (e.g., "3mo", "1y")
            
        Returns:
            Dictionary mapping ticker to stock data
        """
        if period is None:
            period = self.config.data_sources.stock_data_period
        
        logger.info(f"Fetching stock data for {len(tickers)} tickers in parallel")
        
        # Create tasks for each ticker
        tasks = []
        for ticker in tickers:
            task = self._fetch_single_ticker_stock_data(ticker, period)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        stock_data = {}
        for ticker, result in zip(tickers, results):
            if isinstance(result, Exception):
                logger.error(f"Error fetching stock data for {ticker}: {str(result)}")
                stock_data[ticker] = None
            else:
                stock_data[ticker] = result
                logger.info(f"Fetched stock data for {ticker}")
        
        return stock_data
    
    async def _fetch_single_ticker_stock_data(self, ticker: str, period: str) -> Any:
        """Fetch stock data for a single ticker with caching"""
        cache = get_cache_manager()
        
        # Check cache first
        cached_data = cache.get(cache.cache_key("stock_data", ticker, period))
        if cached_data is not None:
            logger.debug(f"Using cached stock data for {ticker}")
            return cached_data
        
        # Fetch from source
        try:
            # Import here to avoid circular imports
            from utils import get_stock_data
            
            # Run the synchronous stock data fetching in executor
            stock_data = await self.run_in_executor(get_stock_data, ticker, period)
            
            # Cache the results
            cache.set(
                cache.cache_key("stock_data", ticker, period),
                stock_data,
                self.config.data_sources.cache_ttl_seconds
            )
            
            return stock_data
            
        except Exception as e:
            logger.error(f"Error fetching stock data for {ticker}: {str(e)}")
            raise StockDataError(f"Failed to fetch stock data for {ticker}: {str(e)}")
    
    async def analyze_sentiment_batch(self, articles: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for multiple articles in parallel
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            List of articles with sentiment analysis added
        """
        logger.info(f"Analyzing sentiment for {len(articles)} articles in parallel")
        
        # Create tasks for sentiment analysis
        tasks = []
        for article in articles:
            task = self._analyze_single_article_sentiment(article)
            tasks.append(task)
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        processed_articles = []
        for article, result in zip(articles, results):
            if isinstance(result, Exception):
                logger.error(f"Error analyzing sentiment for article: {str(result)}")
                # Add default values on error
                article['sentiment'] = 0.0
                article['entities'] = []
            else:
                article.update(result)
            
            processed_articles.append(article)
        
        return processed_articles
    
    async def _analyze_single_article_sentiment(self, article: Dict) -> Dict:
        """Analyze sentiment for a single article with caching"""
        try:
            # Import here to avoid circular imports
            from sentiment_analyzer import analyze_sentiment, get_named_entities
            
            text = article['title'] + " " + article.get('content', '')
            
            # Check cache for sentiment
            cache = get_cache_manager()
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            cache_key = cache.cache_key("sentiment", text_hash)
            
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug("Using cached sentiment analysis")
                return cached_result
            
            # Run sentiment analysis in executor
            sentiment = await self.run_in_executor(
                analyze_sentiment, 
                text, 
                self.config.sentiment.custom_terms_enabled
            )
            
            # Run entity extraction in executor
            entities = await self.run_in_executor(get_named_entities, article.get('content', ''))
            
            result = {
                'sentiment': sentiment,
                'entities': entities
            }
            
            # Cache the result
            cache.set(cache_key, result, self.config.data_sources.cache_ttl_seconds)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            raise
    
    async def process_complete_pipeline(self, tickers: List[str]) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
        """
        Process the complete data pipeline: fetch news, stock data, and analyze sentiment
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Tuple of (news_data, stock_data)
        """
        with LogContext(logger, "complete_pipeline", tickers=tickers):
            start_time = time.time()
            
            # Step 1: Fetch news and stock data in parallel
            logger.info("Step 1: Fetching news and stock data in parallel")
            news_task = self.fetch_multiple_news(tickers)
            stock_task = self.fetch_multiple_stock_data(tickers)
            
            news_data, stock_data = await asyncio.gather(news_task, stock_task)
            
            # Step 2: Flatten news articles and analyze sentiment in parallel
            logger.info("Step 2: Analyzing sentiment for all articles")
            all_articles = []
            ticker_article_map = {}  # Track which articles belong to which ticker
            
            for ticker, articles in news_data.items():
                start_idx = len(all_articles)
                all_articles.extend(articles)
                end_idx = len(all_articles)
                ticker_article_map[ticker] = (start_idx, end_idx)
            
            if all_articles:
                processed_articles = await self.analyze_sentiment_batch(all_articles)
                
                # Reassign processed articles back to tickers
                for ticker, (start_idx, end_idx) in ticker_article_map.items():
                    news_data[ticker] = processed_articles[start_idx:end_idx]
            
            execution_time = time.time() - start_time
            logger.info(f"Complete pipeline processed in {execution_time:.2f} seconds")
            
            return news_data, stock_data

# Convenience functions for sync/async bridge
def run_async_news_fetch(tickers: List[str], max_articles: int = None) -> Dict[str, List[Dict]]:
    """Synchronous wrapper for async news fetching"""
    async def _run():
        async with AsyncProcessor() as processor:
            return await processor.fetch_multiple_news(tickers, max_articles)
    
    return asyncio.run(_run())

def run_async_stock_fetch(tickers: List[str], period: str = None) -> Dict[str, Any]:
    """Synchronous wrapper for async stock data fetching"""
    async def _run():
        async with AsyncProcessor() as processor:
            return await processor.fetch_multiple_stock_data(tickers, period)
    
    return asyncio.run(_run())

def run_async_complete_pipeline(tickers: List[str]) -> Tuple[Dict[str, List[Dict]], Dict[str, Any]]:
    """Synchronous wrapper for complete async pipeline"""
    async def _run():
        async with AsyncProcessor() as processor:
            return await processor.process_complete_pipeline(tickers)
    
    return asyncio.run(_run())

# Performance monitoring decorator
def async_performance_monitor(operation_name: str):
    """Decorator to monitor async operation performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = await func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"Async operation '{operation_name}' completed in {execution_time:.2f}s")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"Async operation '{operation_name}' failed after {execution_time:.2f}s: {str(e)}")
                raise
        return wrapper
    return decorator 