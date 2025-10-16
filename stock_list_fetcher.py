"""
Stock List Fetcher - Dynamically fetch available stocks from various sources
"""

import yfinance as yf
import pandas as pd
import requests
import json
from typing import List, Dict, Set, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class StockSource(Enum):
    """Available stock data sources"""
    YAHOO_FINANCE = "yahoo_finance"
    NASDAQ = "nasdaq"
    NYSE = "nyse"
    SP500 = "sp500"
    RUSSELL_2000 = "russell_2000"
    CUSTOM = "custom"

@dataclass
class StockInfo:
    """Stock information data structure"""
    symbol: str
    name: str
    exchange: str
    market_cap: Optional[float] = None
    sector: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: Optional[str] = None

class StockListFetcher:
    """Fetches and manages stock lists from various sources"""
    
    def __init__(self):
        self.cached_stocks = {}
        self.cache_timestamp = None
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours
        
        # Popular stock lists for quick access
        self.popular_lists = {
            "FAANG": ["META", "AAPL", "AMZN", "NFLX", "GOOGL"],
            "Tech Giants": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA"],
            "Blue Chips": ["JPM", "JNJ", "PG", "KO", "WMT", "V", "UNH", "HD"],
            "Growth Stocks": ["TSLA", "NVDA", "AMD", "CRM", "ADBE", "NFLX", "PYPL"],
            "Dividend Stocks": ["JNJ", "PG", "KO", "PEP", "WMT", "T", "VZ", "IBM"],
            "Banking": ["JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC"],
            "Energy": ["XOM", "CVX", "COP", "EOG", "SLB", "KMI", "PSX", "VLO"],
            "Healthcare": ["JNJ", "PFE", "UNH", "ABBV", "MRK", "TMO", "DHR", "ABT"]
        }
    
    def get_popular_stocks(self, list_name: str = None) -> List[str]:
        """Get popular stock lists"""
        if list_name and list_name in self.popular_lists:
            return self.popular_lists[list_name]
        return list(self.popular_lists.keys())
    
    def fetch_sp500_stocks(self) -> List[StockInfo]:
        """Fetch S&P 500 stocks from Wikipedia"""
        try:
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            
            stocks = []
            for _, row in sp500_table.iterrows():
                stock = StockInfo(
                    symbol=row['Symbol'],
                    name=row['Security'],
                    exchange="NYSE" if '.' not in row['Symbol'] else "NASDAQ",
                    sector=row.get('GICS Sector', ''),
                    industry=row.get('GICS Sub Industry', ''),
                    country="USA"
                )
                stocks.append(stock)
            
            logger.info(f"Fetched {len(stocks)} S&P 500 stocks")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 stocks: {str(e)}")
            return []
    
    def fetch_nasdaq_stocks(self, limit: int = 1000) -> List[StockInfo]:
        """Fetch NASDAQ stocks (limited sample)"""
        try:
            # This is a simplified approach - in practice, you'd use a proper API
            # For now, we'll return a curated list of popular NASDAQ stocks
            nasdaq_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
                "INTC", "ADBE", "CRM", "NFLX", "PYPL", "COIN", "ROKU", "ZM",
                "PTON", "UBER", "LYFT", "SNAP", "TWTR", "SQ", "SHOP", "OKTA",
                "CRWD", "ZS", "DDOG", "NET", "SNOW", "PLTR", "RBLX", "COIN"
            ]
            
            stocks = []
            for symbol in nasdaq_stocks[:limit]:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    stock = StockInfo(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        exchange="NASDAQ",
                        market_cap=info.get('marketCap'),
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        country=info.get('country'),
                        currency=info.get('currency')
                    )
                    stocks.append(stock)
                    
                except Exception as e:
                    logger.warning(f"Error fetching info for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(stocks)} NASDAQ stocks")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching NASDAQ stocks: {str(e)}")
            return []
    
    def fetch_nyse_stocks(self, limit: int = 1000) -> List[StockInfo]:
        """Fetch NYSE stocks (limited sample)"""
        try:
            # Curated list of popular NYSE stocks
            nyse_stocks = [
                "JPM", "JNJ", "PG", "KO", "WMT", "V", "UNH", "HD", "MA", "DIS",
                "PFE", "ABBV", "MRK", "TMO", "DHR", "ABT", "ACN", "NKE", "ADP",
                "TXN", "QCOM", "AVGO", "ORCL", "CRM", "INTC", "CSCO", "IBM",
                "GE", "BA", "CAT", "MMM", "HON", "UPS", "FDX", "LMT", "RTX"
            ]
            
            stocks = []
            for symbol in nyse_stocks[:limit]:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    stock = StockInfo(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        exchange="NYSE",
                        market_cap=info.get('marketCap'),
                        sector=info.get('sector'),
                        industry=info.get('industry'),
                        country=info.get('country'),
                        currency=info.get('currency')
                    )
                    stocks.append(stock)
                    
                except Exception as e:
                    logger.warning(f"Error fetching info for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(stocks)} NYSE stocks")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching NYSE stocks: {str(e)}")
            return []
    
    def fetch_crypto_stocks(self) -> List[StockInfo]:
        """Fetch cryptocurrency-related stocks"""
        try:
            crypto_stocks = [
                "COIN", "MSTR", "RIOT", "MARA", "HUT", "BITF", "CAN", "HIVE",
                "ARB", "BTBT", "EBON", "SOS", "WKEY", "EBANG", "CAN", "HIVE"
            ]
            
            stocks = []
            for symbol in crypto_stocks:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    stock = StockInfo(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        exchange=info.get('exchange', 'NASDAQ'),
                        market_cap=info.get('marketCap'),
                        sector="Cryptocurrency",
                        industry="Cryptocurrency",
                        country=info.get('country'),
                        currency=info.get('currency')
                    )
                    stocks.append(stock)
                    
                except Exception as e:
                    logger.warning(f"Error fetching info for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(stocks)} crypto stocks")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching crypto stocks: {str(e)}")
            return []
    
    def fetch_etf_stocks(self) -> List[StockInfo]:
        """Fetch popular ETFs"""
        try:
            etf_symbols = [
                "SPY", "QQQ", "IWM", "VTI", "VEA", "VWO", "BND", "TLT", "GLD", "SLV",
                "XLF", "XLK", "XLE", "XLI", "XLV", "XLY", "XLP", "XLU", "XLRE", "XLB",
                "ARKK", "ARKQ", "ARKW", "ARKG", "ARKF", "TQQQ", "SQQQ", "UPRO", "SPXU"
            ]
            
            stocks = []
            for symbol in etf_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    
                    stock = StockInfo(
                        symbol=symbol,
                        name=info.get('longName', symbol),
                        exchange=info.get('exchange', 'NYSE'),
                        market_cap=info.get('marketCap'),
                        sector="ETF",
                        industry="Exchange Traded Fund",
                        country=info.get('country'),
                        currency=info.get('currency')
                    )
                    stocks.append(stock)
                    
                except Exception as e:
                    logger.warning(f"Error fetching info for {symbol}: {str(e)}")
                    continue
            
            logger.info(f"Fetched {len(stocks)} ETFs")
            return stocks
            
        except Exception as e:
            logger.error(f"Error fetching ETFs: {str(e)}")
            return []
    
    def search_stocks(self, query: str, limit: int = 50) -> List[StockInfo]:
        """Search for stocks by symbol or name"""
        try:
            # This is a simplified search - in practice, you'd use a proper search API
            all_stocks = self.get_all_stocks()
            
            query_lower = query.lower()
            matching_stocks = []
            
            for stock in all_stocks:
                if (query_lower in stock.symbol.lower() or 
                    query_lower in stock.name.lower()):
                    matching_stocks.append(stock)
                    
                    if len(matching_stocks) >= limit:
                        break
            
            logger.info(f"Found {len(matching_stocks)} stocks matching '{query}'")
            return matching_stocks
            
        except Exception as e:
            logger.error(f"Error searching stocks: {str(e)}")
            return []
    
    def get_all_stocks(self, force_refresh: bool = False) -> List[StockInfo]:
        """Get all available stocks with caching"""
        # Check if cache is valid
        if (not force_refresh and 
            self.cached_stocks and 
            self.cache_timestamp and 
            datetime.now() - self.cache_timestamp < self.cache_duration):
            return list(self.cached_stocks.values())
        
        # Fetch fresh data
        all_stocks = []
        
        # Fetch from different sources
        all_stocks.extend(self.fetch_sp500_stocks())
        all_stocks.extend(self.fetch_nasdaq_stocks(500))
        all_stocks.extend(self.fetch_nyse_stocks(500))
        all_stocks.extend(self.fetch_crypto_stocks())
        all_stocks.extend(self.fetch_etf_stocks())
        
        # Remove duplicates based on symbol
        unique_stocks = {}
        for stock in all_stocks:
            if stock.symbol not in unique_stocks:
                unique_stocks[stock.symbol] = stock
        
        # Update cache
        self.cached_stocks = unique_stocks
        self.cache_timestamp = datetime.now()
        
        logger.info(f"Cached {len(unique_stocks)} unique stocks")
        return list(unique_stocks.values())
    
    def get_stocks_by_sector(self, sector: str) -> List[StockInfo]:
        """Get stocks filtered by sector"""
        all_stocks = self.get_all_stocks()
        return [stock for stock in all_stocks if stock.sector and sector.lower() in stock.sector.lower()]
    
    def get_stocks_by_exchange(self, exchange: str) -> List[StockInfo]:
        """Get stocks filtered by exchange"""
        all_stocks = self.get_all_stocks()
        return [stock for stock in all_stocks if exchange.upper() in stock.exchange.upper()]
    
    def get_top_market_cap_stocks(self, limit: int = 100) -> List[StockInfo]:
        """Get top stocks by market cap"""
        all_stocks = self.get_all_stocks()
        
        # Filter stocks with market cap data and sort
        stocks_with_cap = [stock for stock in all_stocks if stock.market_cap is not None]
        stocks_with_cap.sort(key=lambda x: x.market_cap, reverse=True)
        
        return stocks_with_cap[:limit]
    
    def get_stock_symbols(self, source: StockSource = None) -> List[str]:
        """Get just the stock symbols"""
        if source == StockSource.SP500:
            stocks = self.fetch_sp500_stocks()
        elif source == StockSource.NASDAQ:
            stocks = self.fetch_nasdaq_stocks()
        elif source == StockSource.NYSE:
            stocks = self.fetch_nyse_stocks()
        else:
            stocks = self.get_all_stocks()
        
        return [stock.symbol for stock in stocks]
    
    def get_stock_info(self, symbol: str) -> Optional[StockInfo]:
        """Get detailed info for a specific stock"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return StockInfo(
                symbol=symbol,
                name=info.get('longName', symbol),
                exchange=info.get('exchange', 'Unknown'),
                market_cap=info.get('marketCap'),
                sector=info.get('sector'),
                industry=info.get('industry'),
                country=info.get('country'),
                currency=info.get('currency')
            )
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return None

# Convenience functions
def get_available_stocks(source: str = "all") -> List[str]:
    """Get available stock symbols"""
    fetcher = StockListFetcher()
    
    if source == "sp500":
        return fetcher.get_stock_symbols(StockSource.SP500)
    elif source == "nasdaq":
        return fetcher.get_stock_symbols(StockSource.NASDAQ)
    elif source == "nyse":
        return fetcher.get_stock_symbols(StockSource.NYSE)
    else:
        return fetcher.get_stock_symbols()

def search_stocks(query: str) -> List[str]:
    """Search for stocks and return symbols"""
    fetcher = StockListFetcher()
    stocks = fetcher.search_stocks(query)
    return [stock.symbol for stock in stocks]

def get_popular_stock_lists() -> Dict[str, List[str]]:
    """Get popular stock lists"""
    fetcher = StockListFetcher()
    return fetcher.popular_lists
