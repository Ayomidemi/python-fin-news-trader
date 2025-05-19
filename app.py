import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time
import os

# Custom module imports
from news_scraper import fetch_wsj_news
from sentiment_analyzer import analyze_sentiment, get_named_entities
from trading_strategy import generate_signals
from portfolio_tracker import update_portfolio, calculate_performance
from data_visualizer import plot_sentiment_over_time, plot_portfolio_performance, plot_stock_price_with_signals
from backtester import run_backtest
from utils import format_currency, get_stock_data, load_data, save_data

# Set page configuration
st.set_page_config(
    page_title="FinNews Trader",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {
        'cash': 100000.0,
        'positions': {},
        'history': [],
        'performance': [],
    }
    
if 'signals' not in st.session_state:
    st.session_state.signals = []
    
if 'news_data' not in st.session_state:
    st.session_state.news_data = []
    
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# Main app header
st.title("📊 FinNews Trader")
st.subheader("Algorithmic Trading System Based on Financial News Analysis")

# Sidebar configuration
st.sidebar.header("Configuration")

# Select stocks to track
default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
tracked_stocks = st.sidebar.multiselect(
    "Select stocks to track",
    options=["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"],
    default=default_stocks
)

# Strategy parameters
st.sidebar.subheader("Strategy Parameters")
sentiment_threshold = st.sidebar.slider("Sentiment Threshold", -1.0, 1.0, 0.3, 0.05)
position_size_pct = st.sidebar.slider("Position Size (%)", 5, 30, 10, 5) / 100
stop_loss_pct = st.sidebar.slider("Stop Loss (%)", 5, 20, 10, 1) / 100
take_profit_pct = st.sidebar.slider("Take Profit (%)", 5, 30, 15, 1) / 100

# Backtest settings
use_backtest = st.sidebar.checkbox("Run Backtest")
if use_backtest:
    backtest_days = st.sidebar.slider("Backtest Days", 7, 90, 30)
    backtest_start = datetime.now() - timedelta(days=backtest_days)
    backtest_end = datetime.now()

# Fetch new data button
if st.sidebar.button("Fetch Latest Data"):
    with st.spinner("Fetching latest financial news and stock data..."):
        # Fetch news data
        news_data = []
        for stock in tracked_stocks:
            try:
                stock_news = fetch_wsj_news(stock)
                for article in stock_news:
                    article['ticker'] = stock
                news_data.extend(stock_news)
            except Exception as e:
                st.error(f"Error fetching news for {stock}: {str(e)}")
        
        # Sort by date
        news_data = sorted(news_data, key=lambda x: x['date'], reverse=True)
        
        # Analyze sentiment and extract entities
        for article in news_data:
            try:
                sentiment = analyze_sentiment(article['title'] + " " + article['content'])
                entities = get_named_entities(article['content'])
                article['sentiment'] = sentiment
                article['entities'] = entities
            except Exception as e:
                st.error(f"Error in sentiment analysis: {str(e)}")
                article['sentiment'] = 0
                article['entities'] = []
        
        # Fetch stock data
        stock_data = {}
        for stock in tracked_stocks:
            try:
                stock_data[stock] = get_stock_data(stock)
            except Exception as e:
                st.error(f"Error fetching stock data for {stock}: {str(e)}")
        
        # Generate trading signals
        signals = generate_signals(
            news_data, 
            stock_data, 
            sentiment_threshold,
            position_size_pct,
            stop_loss_pct,
            take_profit_pct
        )
        
        # Update session state
        st.session_state.news_data = news_data
        st.session_state.signals = signals
        st.session_state.last_update = datetime.now()
        
        # Update portfolio based on signals
        st.session_state.portfolio = update_portfolio(
            st.session_state.portfolio,
            signals,
            stock_data
        )
        
        st.success("Data updated successfully!")

# Main content area - use tabs
tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "News Analysis", "Trading Signals", "Portfolio"])

with tab1:
    # Dashboard Overview
    st.header("Trading Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        portfolio_value = st.session_state.portfolio['cash']
        for ticker, position in st.session_state.portfolio['positions'].items():
            try:
                current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                portfolio_value += position['shares'] * current_price
            except:
                st.warning(f"Could not fetch current price for {ticker}")
        
        st.metric(
            "Portfolio Value", 
            format_currency(portfolio_value), 
            delta=format_currency(portfolio_value - 100000)
        )
    
    with col2:
        signal_count = len(st.session_state.signals)
        buy_signals = sum(1 for s in st.session_state.signals if s['action'] == 'BUY')
        sell_signals = sum(1 for s in st.session_state.signals if s['action'] == 'SELL')
        
        st.metric("Active Signals", f"{signal_count}", f"{buy_signals} Buy, {sell_signals} Sell")
    
    with col3:
        if st.session_state.last_update:
            last_update = st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")
        else:
            last_update = "Never"
        
        st.metric("Last Data Update", last_update)
    
    # Performance Chart
    st.subheader("Portfolio Performance")
    if len(st.session_state.portfolio['performance']) > 0:
        plot_portfolio_performance(st.session_state.portfolio['performance'])
    else:
        st.info("No performance data available yet. Fetch latest data to begin tracking.")
    
    # Stock Charts with Signals
    st.subheader("Stock Prices & Signals")
    if tracked_stocks:
        for stock in tracked_stocks:
            try:
                stock_signals = [s for s in st.session_state.signals if s['ticker'] == stock]
                stock_data = get_stock_data(stock)
                plot_stock_price_with_signals(stock, stock_data, stock_signals)
            except Exception as e:
                st.error(f"Error displaying chart for {stock}: {str(e)}")
    else:
        st.info("Select stocks to track in the sidebar")

with tab2:
    # News Analysis
    st.header("Financial News Analysis")
    
    # Sentiment Overview
    st.subheader("News Sentiment Overview")
    
    if st.session_state.news_data:
        # Calculate average sentiment per stock
        sentiment_by_stock = {}
        for article in st.session_state.news_data:
            ticker = article['ticker']
            if ticker not in sentiment_by_stock:
                sentiment_by_stock[ticker] = []
            sentiment_by_stock[ticker].append(article['sentiment'])
        
        avg_sentiment = {ticker: sum(scores)/len(scores) for ticker, scores in sentiment_by_stock.items()}
        
        # Create a bar chart
        fig = px.bar(
            x=list(avg_sentiment.keys()),
            y=list(avg_sentiment.values()),
            labels={'x': 'Stock', 'y': 'Average Sentiment'},
            color=list(avg_sentiment.values()),
            color_continuous_scale='RdYlGn',
            range_color=[-1, 1]
        )
        fig.update_layout(title_text="Average Sentiment by Stock")
        st.plotly_chart(fig, use_container_width=True)
        
        # Sentiment over time
        st.subheader("Sentiment Trends")
        plot_sentiment_over_time(st.session_state.news_data)
        
        # News Articles Table
        st.subheader("Recent Financial News")
        news_df = pd.DataFrame(st.session_state.news_data)
        if len(news_df) > 0:
            news_df = news_df[['date', 'ticker', 'title', 'sentiment']]
            news_df.columns = ['Date', 'Stock', 'Headline', 'Sentiment']
            news_df = news_df.sort_values('Date', ascending=False)
            
            # Apply color to sentiment values
            def color_sentiment(val):
                if val > 0.2:
                    return 'background-color: rgba(0, 255, 0, 0.2)'
                elif val < -0.2:
                    return 'background-color: rgba(255, 0, 0, 0.2)'
                else:
                    return 'background-color: rgba(255, 255, 0, 0.1)'
            
            styled_df = news_df.style.applymap(color_sentiment, subset=['Sentiment'])
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No news data available. Click 'Fetch Latest Data' to get news.")
    else:
        st.info("No news data available. Click 'Fetch Latest Data' to get news.")

with tab3:
    # Trading Signals
    st.header("Trading Signals")
    
    if use_backtest:
        st.subheader("Backtest Results")
        
        backtest_results = run_backtest(
            tracked_stocks,
            backtest_start,
            backtest_end,
            sentiment_threshold,
            position_size_pct,
            stop_loss_pct,
            take_profit_pct
        )
        
        # Display backtest metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Return", f"{backtest_results['total_return_pct']:.2f}%")
        
        with col2:
            st.metric("Win Rate", f"{backtest_results['win_rate']:.2f}%")
            
        with col3:
            st.metric("Profit Factor", f"{backtest_results['profit_factor']:.2f}")
            
        with col4:
            st.metric("Max Drawdown", f"{backtest_results['max_drawdown_pct']:.2f}%")
        
        # Plot backtest equity curve
        st.subheader("Backtest Equity Curve")
        fig = px.line(
            x=backtest_results['dates'],
            y=backtest_results['equity_curve'],
            labels={'x': 'Date', 'y': 'Portfolio Value'},
        )
        fig.update_layout(title_text="Backtest Equity Curve")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display trades
        st.subheader("Backtest Trades")
        trades_df = pd.DataFrame(backtest_results['trades'])
        if len(trades_df) > 0:
            trades_df.columns = ['Date', 'Stock', 'Action', 'Price', 'Shares', 'P&L', 'Return %']
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No trades were executed in the backtest period.")
    
    # Recent Signals
    st.subheader("Recent Trading Signals")
    if st.session_state.signals:
        signals_df = pd.DataFrame(st.session_state.signals)
        signals_df = signals_df[['timestamp', 'ticker', 'action', 'price', 'reason', 'stop_loss', 'take_profit']]
        signals_df.columns = ['Timestamp', 'Stock', 'Action', 'Price', 'Reason', 'Stop Loss', 'Take Profit']
        signals_df = signals_df.sort_values('Timestamp', ascending=False)
        
        # Apply color to action values
        def color_action(val):
            if val == 'BUY':
                return 'background-color: rgba(0, 255, 0, 0.2)'
            elif val == 'SELL':
                return 'background-color: rgba(255, 0, 0, 0.2)'
            else:
                return ''
        
        styled_df = signals_df.style.applymap(color_action, subset=['Action'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No trading signals available. Click 'Fetch Latest Data' to generate signals.")

with tab4:
    # Portfolio Tracker
    st.header("Portfolio Tracker")
    
    # Portfolio Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Cash Balance", format_currency(st.session_state.portfolio['cash']))
    
    with col2:
        positions_value = 0
        for ticker, position in st.session_state.portfolio['positions'].items():
            try:
                current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                positions_value += position['shares'] * current_price
            except:
                st.warning(f"Could not fetch current price for {ticker}")
        
        st.metric("Positions Value", format_currency(positions_value))
    
    with col3:
        st.metric("Total Value", format_currency(st.session_state.portfolio['cash'] + positions_value))
    
    # Current Positions
    st.subheader("Current Positions")
    if st.session_state.portfolio['positions']:
        positions_data = []
        for ticker, position in st.session_state.portfolio['positions'].items():
            try:
                current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                market_value = position['shares'] * current_price
                unrealized_pl = market_value - (position['shares'] * position['avg_price'])
                unrealized_pl_pct = (unrealized_pl / (position['shares'] * position['avg_price'])) * 100
                
                positions_data.append({
                    'Stock': ticker,
                    'Shares': position['shares'],
                    'Avg Price': position['avg_price'],
                    'Current Price': current_price,
                    'Market Value': market_value,
                    'Unrealized P&L': unrealized_pl,
                    'Return %': unrealized_pl_pct
                })
            except:
                st.warning(f"Could not fetch current price for {ticker}")
        
        positions_df = pd.DataFrame(positions_data)
        
        # Apply color to P&L values
        def color_pl(val):
            if val > 0:
                return 'background-color: rgba(0, 255, 0, 0.2)'
            elif val < 0:
                return 'background-color: rgba(255, 0, 0, 0.2)'
            else:
                return ''
                
        styled_df = positions_df.style.applymap(color_pl, subset=['Unrealized P&L', 'Return %'])
        st.dataframe(styled_df, use_container_width=True)
    else:
        st.info("No open positions. Signals will be converted to positions when you fetch new data.")
    
    # Trade History
    st.subheader("Trade History")
    if st.session_state.portfolio['history']:
        history_df = pd.DataFrame(st.session_state.portfolio['history'])
        history_df = history_df.sort_values('timestamp', ascending=False)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No trade history available.")

# Footer
st.markdown("---")
st.caption("FinNews Trader - Algorithmic Trading System Based on Financial News Analysis")
