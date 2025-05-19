import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from collections import defaultdict

def generate_signals(news_data, stock_data, sentiment_threshold=0.3, position_size=0.1, stop_loss=0.1, take_profit=0.15):
    """
    Generate trading signals based on news sentiment and stock data
    
    Args:
        news_data (list): List of news article dicts with sentiment
        stock_data (dict): Dictionary of stock price data by ticker
        sentiment_threshold (float): Threshold for sentiment to generate signal
        position_size (float): Position size as percentage of portfolio
        stop_loss (float): Stop loss percentage
        take_profit (float): Take profit percentage
        
    Returns:
        list: List of trading signal dictionaries
    """
    signals = []
    
    # Group news by ticker
    news_by_ticker = defaultdict(list)
    for article in news_data:
        if 'ticker' in article:
            news_by_ticker[article['ticker']].append(article)
    
    # Process each ticker
    for ticker, ticker_news in news_by_ticker.items():
        # Skip if no price data available
        if ticker not in stock_data:
            continue
            
        # Get latest price
        price_data = stock_data[ticker]
        if price_data.empty:
            continue
            
        latest_price = price_data['Close'].iloc[-1]
        
        # Calculate average sentiment
        avg_sentiment = sum(article['sentiment'] for article in ticker_news) / len(ticker_news) if ticker_news else 0
        
        # Generate signals based on sentiment
        if avg_sentiment > sentiment_threshold:
            # Strong positive sentiment - BUY signal
            signals.append({
                'timestamp': datetime.now(),
                'ticker': ticker,
                'action': 'BUY',
                'price': latest_price,
                'reason': f'Bullish news sentiment ({avg_sentiment:.2f})',
                'stop_loss': latest_price * (1 - stop_loss),
                'take_profit': latest_price * (1 + take_profit),
                'position_size': position_size
            })
        elif avg_sentiment < -sentiment_threshold:
            # Strong negative sentiment - SELL signal
            signals.append({
                'timestamp': datetime.now(),
                'ticker': ticker,
                'action': 'SELL',
                'price': latest_price,
                'reason': f'Bearish news sentiment ({avg_sentiment:.2f})',
                'stop_loss': latest_price * (1 + stop_loss),
                'take_profit': latest_price * (1 - take_profit),
                'position_size': position_size
            })
            
    # Apply technical filters
    filtered_signals = filter_signals_with_technicals(signals, stock_data)
    
    return filtered_signals

def filter_signals_with_technicals(signals, stock_data):
    """
    Filter signals using technical indicators to confirm news sentiment
    
    Args:
        signals (list): List of sentiment-based signals
        stock_data (dict): Dictionary of stock price data by ticker
        
    Returns:
        list: Filtered list of signals
    """
    filtered_signals = []
    
    for signal in signals:
        ticker = signal['ticker']
        
        # Skip if no price data available
        if ticker not in stock_data:
            continue
            
        price_data = stock_data[ticker]
        if price_data.empty:
            continue
        
        # Calculate technical indicators
        price_data = calculate_indicators(price_data)
        
        # Last row of indicators
        last_row = price_data.iloc[-1]
        
        # Logic for confirming BUY signals
        if signal['action'] == 'BUY':
            # Confirm with moving averages: price > MA50
            if 'MA50' in last_row and last_row['Close'] > last_row['MA50']:
                # Additional check: RSI not overbought
                if 'RSI' in last_row and last_row['RSI'] < 70:
                    filtered_signals.append(signal)
        
        # Logic for confirming SELL signals
        elif signal['action'] == 'SELL':
            # Confirm with moving averages: price < MA50
            if 'MA50' in last_row and last_row['Close'] < last_row['MA50']:
                # Additional check: RSI not oversold
                if 'RSI' in last_row and last_row['RSI'] > 30:
                    filtered_signals.append(signal)
    
    return filtered_signals

def calculate_indicators(df):
    """
    Calculate technical indicators for a price DataFrame
    
    Args:
        df (DataFrame): Price data with OHLC columns
        
    Returns:
        DataFrame: Price data with added technical indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate moving averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal']
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
    
    return df

def evaluate_open_positions(positions, current_prices, news_sentiment=None):
    """
    Evaluate open positions and generate exit signals if needed
    
    Args:
        positions (dict): Dictionary of current positions
        current_prices (dict): Dictionary of current prices by ticker
        news_sentiment (dict, optional): Latest sentiment by ticker
        
    Returns:
        list: List of exit signals
    """
    exit_signals = []
    
    for ticker, position in positions.items():
        # Skip if no current price available
        if ticker not in current_prices:
            continue
            
        current_price = current_prices[ticker]
        entry_price = position['avg_price']
        
        # Check stop loss
        if position['shares'] > 0:  # Long position
            if current_price <= position['stop_loss']:
                exit_signals.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': current_price,
                    'reason': 'Stop Loss',
                    'shares': position['shares']
                })
            elif current_price >= position['take_profit']:
                exit_signals.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': current_price,
                    'reason': 'Take Profit',
                    'shares': position['shares']
                })
        elif position['shares'] < 0:  # Short position
            if current_price >= position['stop_loss']:
                exit_signals.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': current_price,
                    'reason': 'Stop Loss',
                    'shares': -position['shares']  # Cover the short
                })
            elif current_price <= position['take_profit']:
                exit_signals.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': current_price,
                    'reason': 'Take Profit',
                    'shares': -position['shares']  # Cover the short
                })
                
        # Check sentiment-based exits
        if news_sentiment and ticker in news_sentiment:
            sentiment = news_sentiment[ticker]
            
            # Exit long positions on negative sentiment
            if position['shares'] > 0 and sentiment < -0.5:
                exit_signals.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'SELL',
                    'price': current_price,
                    'reason': f'Bearish sentiment shift ({sentiment:.2f})',
                    'shares': position['shares']
                })
            
            # Exit short positions on positive sentiment
            elif position['shares'] < 0 and sentiment > 0.5:
                exit_signals.append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'BUY',
                    'price': current_price,
                    'reason': f'Bullish sentiment shift ({sentiment:.2f})',
                    'shares': -position['shares']
                })
    
    return exit_signals
