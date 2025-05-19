import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

def update_portfolio(portfolio, signals, stock_data):
    """
    Update portfolio based on trading signals
    
    Args:
        portfolio (dict): Current portfolio state
        signals (list): List of trading signals
        stock_data (dict): Dictionary of stock price data
        
    Returns:
        dict: Updated portfolio
    """
    # Make a copy of the portfolio to avoid modifying the original
    portfolio = {
        'cash': portfolio['cash'],
        'positions': portfolio['positions'].copy(),
        'history': portfolio['history'].copy(),
        'performance': portfolio['performance'].copy()
    }
    
    # Process each signal
    for signal in signals:
        ticker = signal['ticker']
        action = signal['action']
        price = signal['price']
        
        # Calculate position size
        if 'position_size' in signal:
            # Based on percentage of portfolio value
            portfolio_value = portfolio['cash']
            for pos_ticker, pos_data in portfolio['positions'].items():
                if pos_ticker in stock_data and not stock_data[pos_ticker].empty:
                    current_price = stock_data[pos_ticker]['Close'].iloc[-1]
                    portfolio_value += pos_data['shares'] * current_price
            
            position_value = portfolio_value * signal['position_size']
            shares = int(position_value / price)
        elif 'shares' in signal:
            # Explicit number of shares
            shares = signal['shares']
        else:
            # Default to 10% of cash
            position_value = portfolio['cash'] * 0.1
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
                        'stop_loss': signal.get('stop_loss', current_position.get('stop_loss')),
                        'take_profit': signal.get('take_profit', current_position.get('take_profit'))
                    }
                else:
                    # Create new position
                    portfolio['positions'][ticker] = {
                        'shares': shares,
                        'avg_price': price,
                        'stop_loss': signal.get('stop_loss'),
                        'take_profit': signal.get('take_profit')
                    }
                
                # Record trade in history
                portfolio['history'].append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'BUY',
                    'shares': shares,
                    'price': price,
                    'value': cost,
                    'reason': signal.get('reason', 'N/A')
                })
        
        elif action == 'SELL':
            # Check if we have the position
            if ticker in portfolio['positions'] and portfolio['positions'][ticker]['shares'] >= shares:
                # Calculate proceeds
                proceeds = shares * price
                
                # Update cash
                portfolio['cash'] += proceeds
                
                # Update position
                current_shares = portfolio['positions'][ticker]['shares']
                if current_shares == shares:
                    # Completely close the position
                    avg_price = portfolio['positions'][ticker]['avg_price']
                    profit_loss = proceeds - (shares * avg_price)
                    
                    del portfolio['positions'][ticker]
                else:
                    # Partially close the position
                    avg_price = portfolio['positions'][ticker]['avg_price']
                    profit_loss = proceeds - (shares * avg_price)
                    
                    portfolio['positions'][ticker]['shares'] -= shares
                
                # Record trade in history
                portfolio['history'].append({
                    'timestamp': datetime.now(),
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': shares,
                    'price': price,
                    'value': proceeds,
                    'profit_loss': profit_loss,
                    'reason': signal.get('reason', 'N/A')
                })
    
    # Calculate and update portfolio performance
    portfolio = calculate_performance(portfolio, stock_data)
    
    return portfolio

def calculate_performance(portfolio, stock_data):
    """
    Calculate portfolio performance metrics
    
    Args:
        portfolio (dict): Current portfolio state
        stock_data (dict): Dictionary of stock price data
        
    Returns:
        dict: Portfolio with updated performance metrics
    """
    # Current portfolio value
    portfolio_value = portfolio['cash']
    
    # Add value of all positions
    positions_value = 0
    for ticker, position in portfolio['positions'].items():
        if ticker in stock_data and not stock_data[ticker].empty:
            current_price = stock_data[ticker]['Close'].iloc[-1]
            position_value = position['shares'] * current_price
            positions_value += position_value
    
    portfolio_value += positions_value
    
    # Add performance snapshot
    portfolio['performance'].append({
        'timestamp': datetime.now(),
        'cash': portfolio['cash'],
        'positions_value': positions_value,
        'total_value': portfolio_value
    })
    
    return portfolio

def get_current_prices(tickers):
    """
    Get current prices for a list of tickers
    
    Args:
        tickers (list): List of ticker symbols
        
    Returns:
        dict: Dictionary of current prices by ticker
    """
    prices = {}
    
    for ticker in tickers:
        try:
            data = yf.Ticker(ticker).history(period="1d")
            if not data.empty:
                prices[ticker] = data['Close'].iloc[-1]
        except Exception as e:
            print(f"Error fetching price for {ticker}: {str(e)}")
    
    return prices

def calculate_portfolio_metrics(portfolio, stock_data):
    """
    Calculate various portfolio performance metrics
    
    Args:
        portfolio (dict): Portfolio data
        stock_data (dict): Dictionary of stock price data
        
    Returns:
        dict: Dictionary of portfolio metrics
    """
    # Extract performance history
    if not portfolio['performance']:
        return {
            'total_return': 0,
            'total_return_pct': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'win_rate': 0,
            'profit_factor': 0
        }
    
    performance = portfolio['performance']
    
    # Calculate returns
    initial_value = 100000  # Assuming initial value
    current_value = performance[-1]['total_value']
    total_return = current_value - initial_value
    total_return_pct = (total_return / initial_value) * 100
    
    # Create a series of portfolio values
    dates = [p['timestamp'] for p in performance]
    values = [p['total_value'] for p in performance]
    
    portfolio_series = pd.Series(values, index=dates)
    
    # Calculate daily returns if we have enough data
    if len(portfolio_series) > 1:
        daily_returns = portfolio_series.pct_change().dropna()
        
        # Calculate annualized return
        days = (portfolio_series.index[-1] - portfolio_series.index[0]).days
        if days > 0:
            annualized_return = ((1 + total_return_pct/100) ** (365/days) - 1) * 100
        else:
            annualized_return = 0
        
        # Calculate Sharpe ratio (assuming 0% risk-free rate)
        if len(daily_returns) > 1 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        cumulative_max = portfolio_series.cummax()
        drawdown = (portfolio_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        max_drawdown_pct = max_drawdown * 100
    else:
        annualized_return = 0
        sharpe_ratio = 0
        max_drawdown = 0
        max_drawdown_pct = 0
    
    # Calculate trade metrics
    trades = [t for t in portfolio['history'] if t['action'] == 'SELL']
    
    if trades:
        # Win rate
        profitable_trades = sum(1 for t in trades if t.get('profit_loss', 0) > 0)
        win_rate = (profitable_trades / len(trades)) * 100
        
        # Profit factor
        gross_profit = sum(t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) > 0)
        gross_loss = sum(abs(t.get('profit_loss', 0)) for t in trades if t.get('profit_loss', 0) < 0)
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    else:
        win_rate = 0
        profit_factor = 0
    
    return {
        'total_return': total_return,
        'total_return_pct': total_return_pct,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'win_rate': win_rate,
        'profit_factor': profit_factor
    }
