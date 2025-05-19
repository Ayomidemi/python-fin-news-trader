import os
import json
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import pickle

def format_currency(value):
    """
    Format a value as currency
    
    Args:
        value (float): Numerical value to format
        
    Returns:
        str: Formatted currency string
    """
    return f"${value:,.2f}"

def get_stock_data(ticker, period="3mo"):
    """
    Get stock price data from yfinance
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period to fetch data for
        
    Returns:
        DataFrame: Stock price data
    """
    try:
        data = yf.Ticker(ticker).history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {str(e)}")
        return pd.DataFrame()

def save_data(data, filename):
    """
    Save data to a file
    
    Args:
        data: Data to save
        filename (str): Filename to save to
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"Error saving data to {filename}: {str(e)}")

def load_data(filename):
    """
    Load data from a file
    
    Args:
        filename (str): Filename to load from
        
    Returns:
        Data loaded from file, or None if file doesn't exist
    """
    try:
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None
    except Exception as e:
        print(f"Error loading data from {filename}: {str(e)}")
        return None

def calculate_simple_moving_average(data, window):
    """
    Calculate simple moving average
    
    Args:
        data (Series): Price data
        window (int): Window size
        
    Returns:
        Series: Moving average values
    """
    return data.rolling(window=window).mean()

def calculate_exponential_moving_average(data, span):
    """
    Calculate exponential moving average
    
    Args:
        data (Series): Price data
        span (int): Span parameter
        
    Returns:
        Series: EMA values
    """
    return data.ewm(span=span, adjust=False).mean()

def calculate_rsi(data, window=14):
    """
    Calculate Relative Strength Index
    
    Args:
        data (Series): Price data
        window (int): Window size
        
    Returns:
        Series: RSI values
    """
    delta = data.diff()
    
    # Make two series: one for gains and one for losses
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = losses.abs()
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=window).mean()
    avg_losses = losses.rolling(window=window).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def truncate_text(text, max_length=200):
    """
    Truncate text to a maximum length
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text with ellipsis if needed
    """
    if not text:
        return ""
        
    if len(text) <= max_length:
        return text
    
    return text[:max_length] + "..."

def get_trending_stocks():
    """
    Get a list of currently trending stocks
    
    Returns:
        list: List of trending stock tickers
    """
    # This would normally use an API, but for now we'll return some common stocks
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA"]

def calculate_beta(stock_returns, market_returns):
    """
    Calculate beta (market correlation) for a stock
    
    Args:
        stock_returns (Series): Stock returns
        market_returns (Series): Market returns (e.g., S&P 500)
        
    Returns:
        float: Beta value
    """
    # Align the two series
    stock_returns, market_returns = stock_returns.align(market_returns, join='inner')
    
    # Calculate covariance matrix
    cov_matrix = np.cov(stock_returns, market_returns)
    
    # Beta = Covariance(stock, market) / Variance(market)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    
    return beta

def get_market_data(period="3mo"):
    """
    Get market index data (S&P 500)
    
    Args:
        period (str): Time period to fetch data for
        
    Returns:
        DataFrame: Market price data
    """
    try:
        # Get S&P 500 ETF (SPY) as proxy for market
        data = yf.Ticker("SPY").history(period=period)
        return data
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return pd.DataFrame()
