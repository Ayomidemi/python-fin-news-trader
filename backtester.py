import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from tqdm import tqdm
import streamlit as st

from news_scraper import fetch_wsj_news
from sentiment_analyzer import analyze_sentiment, get_named_entities
from trading_strategy import generate_signals, calculate_indicators
from utils import get_stock_data

def run_backtest(tickers, start_date, end_date, sentiment_threshold=0.3, position_size=0.1, stop_loss=0.1, take_profit=0.15):
    """
    Run a backtest of the trading strategy over a historical period
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (datetime): Backtest start date
        end_date (datetime): Backtest end date
        sentiment_threshold (float): Threshold for sentiment to generate signal
        position_size (float): Position size as percentage of portfolio
        stop_loss (float): Stop loss percentage
        take_profit (float): Take profit percentage
        
    Returns:
        dict: Backtest results
    """
    # Initialize portfolio
    portfolio = {
        'cash': 100000.0,
        'positions': {},
        'trades': [],
        'equity_curve': [100000.0],
        'dates': [start_date]
    }
    
    # Create date range for simulation
    current_date = start_date
    date_range = []
    
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Skip weekends
            date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Get historical stock data for the entire period
    stock_data = {}
    for ticker in tickers:
        try:
            # Fetch data with some padding before start_date for indicators
            fetch_start = start_date - timedelta(days=100)
            data = yf.download(ticker, start=fetch_start, end=end_date + timedelta(days=1))
            
            if not data.empty:
                # Calculate indicators
                data = calculate_indicators(data)
                stock_data[ticker] = data
        except Exception as e:
            st.error(f"Error fetching historical data for {ticker}: {str(e)}")
    
    # Simulate trading for each day
    with st.spinner("Running backtest simulation..."):
        for day in tqdm(date_range):
            # Skip if not a trading day
            is_trading_day = all(
                ticker in stock_data and day in stock_data[ticker].index 
                for ticker in tickers
            )
            
            if not is_trading_day:
                continue
            
            # Get prices for this day
            day_prices = {}
            for ticker in tickers:
                if ticker in stock_data and day in stock_data[ticker].index:
                    day_prices[ticker] = stock_data[ticker].loc[day, 'Close']
            
            # Simulate sentiment analysis and signal generation
            # In a real backtest, we would need historical news data
            # Here we'll simulate based on price movements as a proxy
            simulated_news = []
            for ticker in tickers:
                if ticker in stock_data and day in stock_data[ticker].index:
                    # Get data up to this day
                    ticker_data = stock_data[ticker].loc[:day]
                    
                    if len(ticker_data) >= 3:
                        # Use recent price changes as a proxy for sentiment
                        recent_return = ticker_data['Close'].pct_change(3).iloc[-1]
                        
                        # Convert return to sentiment score (-1 to 1 scale)
                        proxy_sentiment = max(min(recent_return * 10, 1), -1)
                        
                        # Create a simulated news article
                        simulated_news.append({
                            'ticker': ticker,
                            'date': day,
                            'sentiment': proxy_sentiment,
                            'title': f"Simulated {ticker} news",
                            'content': f"This is simulated content for {ticker} based on price action.",
                            'url': '',
                            'source': 'Simulation'
                        })
            
            # Generate trading signals
            price_data_for_day = {ticker: data.loc[:day] for ticker, data in stock_data.items()}
            signals = generate_signals(
                simulated_news, 
                price_data_for_day, 
                sentiment_threshold,
                position_size,
                stop_loss,
                take_profit
            )
            
            # Process signals
            for signal in signals:
                ticker = signal['ticker']
                action = signal['action']
                price = day_prices.get(ticker, signal['price'])
                
                # Calculate position size
                portfolio_value = portfolio['cash']
                for pos_ticker, pos_data in portfolio['positions'].items():
                    if pos_ticker in day_prices:
                        portfolio_value += pos_data['shares'] * day_prices[pos_ticker]
                
                position_value = portfolio_value * position_size
                shares = int(position_value / price)
                
                # Execute the trade
                if action == 'BUY':
                    # Check if we have enough cash
                    cost = shares * price
                    if cost > portfolio['cash']:
                        # Adjust shares to available cash
                        shares = int(portfolio['cash'] / price)
                        cost = shares * price
                    
                    if shares > 0:
                        # Update cash
                        portfolio['cash'] -= cost
                        
                        # Update position
                        if ticker in portfolio['positions']:
                            # Average down/up existing position
                            current_position = portfolio['positions'][ticker]
                            total_shares = current_position['shares'] + shares
                            total_cost = (current_position['shares'] * current_position['avg_price']) + cost
                            
                            portfolio['positions'][ticker] = {
                                'shares': total_shares,
                                'avg_price': total_cost / total_shares,
                                'stop_loss': price * (1 - stop_loss),
                                'take_profit': price * (1 + take_profit)
                            }
                        else:
                            # Create new position
                            portfolio['positions'][ticker] = {
                                'shares': shares,
                                'avg_price': price,
                                'stop_loss': price * (1 - stop_loss),
                                'take_profit': price * (1 + take_profit)
                            }
                        
                        # Record trade
                        portfolio['trades'].append({
                            'Date': day,
                            'Stock': ticker,
                            'Action': 'BUY',
                            'Price': price,
                            'Shares': shares,
                            'P&L': 0,
                            'Return %': 0
                        })
                
                elif action == 'SELL':
                    # Check if we have the position
                    if ticker in portfolio['positions'] and portfolio['positions'][ticker]['shares'] >= shares:
                        # Calculate proceeds
                        proceeds = shares * price
                        
                        # Calculate P&L
                        avg_price = portfolio['positions'][ticker]['avg_price']
                        profit_loss = proceeds - (shares * avg_price)
                        return_pct = (profit_loss / (shares * avg_price)) * 100
                        
                        # Update cash
                        portfolio['cash'] += proceeds
                        
                        # Update position
                        current_shares = portfolio['positions'][ticker]['shares']
                        if current_shares == shares:
                            # Completely close the position
                            del portfolio['positions'][ticker]
                        else:
                            # Partially close the position
                            portfolio['positions'][ticker]['shares'] -= shares
                        
                        # Record trade
                        portfolio['trades'].append({
                            'Date': day,
                            'Stock': ticker,
                            'Action': 'SELL',
                            'Price': price,
                            'Shares': shares,
                            'P&L': profit_loss,
                            'Return %': return_pct
                        })
            
            # Check stop losses and take profits at the end of day
            # Create a copy of positions to avoid modifying during iteration
            positions_copy = portfolio['positions'].copy()
            
            for ticker, position in positions_copy.items():
                if ticker in day_prices:
                    price = day_prices[ticker]
                    
                    # Check stop loss
                    if price <= position['stop_loss']:
                        # Close position at stop loss
                        shares = position['shares']
                        proceeds = shares * price
                        profit_loss = proceeds - (shares * position['avg_price'])
                        return_pct = (profit_loss / (shares * position['avg_price'])) * 100
                        
                        # Update cash
                        portfolio['cash'] += proceeds
                        
                        # Remove position
                        del portfolio['positions'][ticker]
                        
                        # Record trade
                        portfolio['trades'].append({
                            'Date': day,
                            'Stock': ticker,
                            'Action': 'SELL',
                            'Price': price,
                            'Shares': shares,
                            'P&L': profit_loss,
                            'Return %': return_pct
                        })
                    
                    # Check take profit
                    elif price >= position['take_profit']:
                        # Close position at take profit
                        shares = position['shares']
                        proceeds = shares * price
                        profit_loss = proceeds - (shares * position['avg_price'])
                        return_pct = (profit_loss / (shares * position['avg_price'])) * 100
                        
                        # Update cash
                        portfolio['cash'] += proceeds
                        
                        # Remove position
                        del portfolio['positions'][ticker]
                        
                        # Record trade
                        portfolio['trades'].append({
                            'Date': day,
                            'Stock': ticker,
                            'Action': 'SELL',
                            'Price': price,
                            'Shares': shares,
                            'P&L': profit_loss,
                            'Return %': return_pct
                        })
            
            # Calculate portfolio value at the end of day
            portfolio_value = portfolio['cash']
            for ticker, position in portfolio['positions'].items():
                if ticker in day_prices:
                    portfolio_value += position['shares'] * day_prices[ticker]
            
            # Update equity curve
            portfolio['equity_curve'].append(portfolio_value)
            portfolio['dates'].append(day)
    
    # Calculate backtest metrics
    # Total return
    initial_value = portfolio['equity_curve'][0]
    final_value = portfolio['equity_curve'][-1]
    total_return = final_value - initial_value
    total_return_pct = (total_return / initial_value) * 100
    
    # Win rate
    trades = portfolio['trades']
    if trades:
        winning_trades = [t for t in trades if t['Action'] == 'SELL' and t['P&L'] > 0]
        losing_trades = [t for t in trades if t['Action'] == 'SELL' and t['P&L'] <= 0]
        
        win_rate = (len(winning_trades) / len([t for t in trades if t['Action'] == 'SELL'])) * 100 if trades else 0
        
        # Profit factor
        gross_profit = sum(t['P&L'] for t in winning_trades)
        gross_loss = abs(sum(t['P&L'] for t in losing_trades))
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    # Maximum drawdown
    equity_curve = np.array(portfolio['equity_curve'])
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    max_drawdown = drawdown.min()
    max_drawdown_pct = max_drawdown * 100
    
    # Return backtest results
    return {
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'equity_curve': portfolio['equity_curve'],
        'dates': portfolio['dates'],
        'trades': portfolio['trades']
    }

def simulate_historical_news(ticker, date):
    """
    Simulate historical news based on price movements
    
    Args:
        ticker (str): Stock ticker symbol
        date (datetime): Date to simulate news for
        
    Returns:
        list: List of simulated news articles
    """
    # Get stock data for a period before the date
    start_date = date - timedelta(days=7)
    end_date = date
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1))
        
        if data.empty:
            return []
        
        # Calculate recent return as proxy for sentiment
        if len(data) >= 3:
            recent_return = data['Close'].pct_change(3).iloc[-1]
            
            # Convert return to sentiment score (-1 to 1 scale)
            proxy_sentiment = max(min(recent_return * 10, 1), -1)
            
            # Create a simulated news article
            return [{
                'ticker': ticker,
                'date': date,
                'sentiment': proxy_sentiment,
                'title': f"Simulated {ticker} news for {date.strftime('%Y-%m-%d')}",
                'content': f"This is simulated content for {ticker} based on price action.",
                'url': '',
                'source': 'Simulation'
            }]
        
        return []
    except Exception as e:
        print(f"Error simulating news for {ticker} on {date}: {str(e)}")
        return []
