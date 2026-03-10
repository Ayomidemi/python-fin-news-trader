import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import time
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration and logging
from config import get_config
from logger import get_app_logger, LogContext
from errors import (
    FinNewsError, DataSourceError, NewsScrapingError, 
    SentimentAnalysisError, TradingError, handle_errors,
    ErrorRecovery, validate_ticker
)
from cache import get_cache_manager
from async_processor import run_async_complete_pipeline, run_async_news_fetch, run_async_stock_fetch

# Custom module imports
from news_scraper import fetch_wsj_news
from sentiment_analyzer import analyze_sentiment, get_named_entities
from trading_strategy import generate_signals
from portfolio_tracker import update_portfolio, calculate_performance
from data_visualizer import plot_sentiment_over_time, plot_portfolio_performance, plot_stock_price_with_signals
from backtester import run_backtest
from utils import format_currency, get_stock_data, load_data, save_data
from big_mover_dashboard import run_big_mover_dashboard
from stock_list_fetcher import get_available_stocks, search_stocks, get_popular_stock_lists, clear_stock_cache
from edge_screener import run_full_scan

# Initialize configuration and logging
config = get_config()
logger = get_app_logger()

# Set page configuration
st.set_page_config(
    page_title="FinNews Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for refined blue/black/white UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    .main {
        padding-top: 1rem;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 25%, #334155 50%, #1e293b 75%, #0f172a 100%);
        font-family: 'Inter', sans-serif;
        min-height: 100vh;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(15, 23, 42, 0.3);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(59, 130, 246, 0.5);
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(59, 130, 246, 0.7);
    }
    
    /* Refined Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.1);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #3b82f6, #1d4ed8, #1e40af);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(15, 23, 42, 0.3);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    .metric-card h3 {
        color: #0f172a;
        font-size: 1rem;
        font-weight: 600;
        margin: 0 0 1rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-card .value {
        color: #0f172a;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .metric-card .delta {
        font-size: 0.85rem;
        font-weight: 500;
        margin: 0.5rem 0 0 0;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        display: inline-block;
    }
    
    .metric-card .delta.positive {
        color: #059669;
        background: rgba(5, 150, 105, 0.1);
    }
    
    .metric-card .delta.negative {
        color: #dc2626;
        background: rgba(220, 38, 38, 0.1);
    }
    
    /* Refined Sidebar */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(248, 250, 252, 0.95) 100%);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        margin: 1rem;
        padding: 2rem;
        box-shadow: 0 20px 40px rgba(15, 23, 42, 0.2);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Refined Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 50%, #1e40af 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-size: 0.9rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 50%, #1e3a8a 100%);
    }
    
    /* Refined Form Elements */
    .stSelectbox > div > div,
    .stMultiSelect > div > div,
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(59, 130, 246, 0.2);
        color: #0f172a;
        font-weight: 500;
    }
    
    .stSelectbox > div > div:hover,
    .stMultiSelect > div > div:hover,
    .stTextInput > div > div > input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: rgba(255, 255, 255, 1);
    }
    
    /* Refined Slider */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #1d4ed8, #1e40af);
        border-radius: 6px;
    }
    
    /* Refined Messages */
    .stInfo, .stSuccess, .stWarning, .stError {
        border-radius: 10px;
        backdrop-filter: blur(10px);
        border-left: 4px solid;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1);
        border-left-color: #3b82f6;
        color: #1e40af;
    }
    
    .stSuccess {
        background: rgba(5, 150, 105, 0.1);
        border-left-color: #059669;
        color: #047857;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1);
        border-left-color: #f59e0b;
        color: #d97706;
    }
    
    .stError {
        background: rgba(220, 38, 38, 0.1);
        border-left-color: #dc2626;
        color: #b91c1c;
    }
    
    /* Refined Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background: rgba(15, 23, 42, 0.1);
        padding: 6px;
        border-radius: 12px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        transition: all 0.3s ease;
        border: 1px solid transparent;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
    }
    
    /* Refined Header */
    .main-header {
        text-align: center;
        padding: 3rem 0;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
        border-radius: 24px;
        margin-bottom: 2rem;
        backdrop-filter: blur(20px);
        border: 1px solid rgba(59, 130, 246, 0.2);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, rgba(59, 130, 246, 0.1) 0%, transparent 50%, rgba(29, 78, 216, 0.1) 100%);
        pointer-events: none;
    }
    
    .main-header h1 {
        color: white;
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        position: relative;
        z-index: 1;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        font-weight: 300;
        position: relative;
        z-index: 1;
    }
    
    /* Data Tables */
    .dataframe {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 32px rgba(15, 23, 42, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.1);
    }
    
    /* Section Headers */
    .stHeader {
        color: #0f172a;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
        display: inline-block;
    }
    
    /* Subsection Headers */
    .stSubheader {
        color: #1e293b;
        font-weight: 600;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

logger.info("Starting FinNews Trader application")

def initialize_session_state():
    """Initialize session state with configuration defaults"""
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'cash': config.backtest.initial_capital,
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
    if 'config' not in st.session_state:
        st.session_state.config = config
    if 'edge_scan' not in st.session_state:
        st.session_state.edge_scan = None  # {unusual_volume, top_performers, watchlist, tickers_scanned}

# Check if running in Streamlit context
if __name__ == "__main__" or "streamlit" in sys.modules:
    # Initialize session state
    initialize_session_state()
    
    # Main app header with refined styling
    st.markdown("""
    <div class="main-header">
        <h1>📊 FinNews Trader</h1>
        <p>Professional Algorithmic Trading Platform with Advanced News Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Stock selection section
st.sidebar.subheader("Stock Selection")

# Stock source selection
stock_source = st.sidebar.selectbox(
    "Stock List Source",
    ["Popular Lists", "S&P 500", "NASDAQ", "NYSE", "All Stocks", "Search"],
    index=0,
    help="Popular Lists work without API keys. S&P 500, NASDAQ, NYSE use real-time data when API keys are available."
)

# Get available stocks based on source (with caching)
try:
    if stock_source == "Popular Lists":
        popular_lists = get_popular_stock_lists()
        selected_list = st.sidebar.selectbox(
            "Choose a popular list",
            list(popular_lists.keys()),
            index=0
        )
        available_stocks = popular_lists[selected_list]
    elif stock_source == "S&P 500":
        available_stocks = get_available_stocks("sp500")
    elif stock_source == "NASDAQ":
        available_stocks = get_available_stocks("nasdaq")
    elif stock_source == "NYSE":
        available_stocks = get_available_stocks("nyse")
    elif stock_source == "All Stocks":
        available_stocks = get_available_stocks("all")
    else:  # Search
        search_query = st.sidebar.text_input("Search for stocks", placeholder="e.g., AAPL, Apple, Tesla")
        if search_query:
            available_stocks = search_stocks(search_query)
        else:
            available_stocks = config.ui.available_stocks
    
    # Fallback to config stocks if no stocks are available
    if not available_stocks:
        available_stocks = config.ui.available_stocks or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        
except Exception as e:
    logger.error(f"Error fetching stocks: {str(e)}")
    # Fallback to config stocks
    available_stocks = config.ui.available_stocks or ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]

# Limit the number of stocks shown for performance
if len(available_stocks) > 200:
    available_stocks = available_stocks[:200]
    st.sidebar.info(f"Showing first 200 stocks. Total available: {len(available_stocks)}")

# Show performance info
if available_stocks:
    st.sidebar.success(f"✅ Loaded {len(available_stocks)} stocks")
else:
    st.sidebar.warning("⚠️ No stocks loaded")

# Select stocks to track
# Ensure default stocks are in the available options
default_stocks = []
if config.ui.default_stocks:
    # Filter default stocks to only include those in available_stocks
    default_stocks = [stock for stock in config.ui.default_stocks if stock in available_stocks]
    
# If no valid defaults, use first few available stocks
if not default_stocks and available_stocks:
    default_stocks = available_stocks[:5]

tracked_stocks = st.sidebar.multiselect(
    "Select stocks to track",
    options=available_stocks,
    default=default_stocks
)

# Strategy parameters
st.sidebar.subheader("Strategy Parameters")
sentiment_threshold = st.sidebar.slider(
    "Sentiment Threshold", 
    -1.0, 1.0, 
    config.trading.default_sentiment_threshold, 
    0.05
)
position_size_pct = st.sidebar.slider(
    "Position Size (%)", 
    5, 30, 
    int(config.trading.default_position_size * 100), 
    5
) / 100
stop_loss_pct = st.sidebar.slider(
    "Stop Loss (%)", 
    5, 20, 
    int(config.trading.default_stop_loss * 100), 
    1
) / 100
take_profit_pct = st.sidebar.slider(
    "Take Profit (%)", 
    5, 30, 
    int(config.trading.default_take_profit * 100), 
    1
) / 100

# API Status
st.sidebar.subheader("API Status")
api_keys = {
    'Alpha Vantage': os.getenv('ALPHA_VANTAGE_API_KEY'),
    'IEX Cloud': os.getenv('IEX_CLOUD_API_KEY'),
    'News API': os.getenv('NEWS_API_KEY'),
    'Polygon': os.getenv('POLYGON_API_KEY'),
}

for api, key in api_keys.items():
    if key and key != f"your_{api.lower().replace(' ', '_')}_key_here":
        st.sidebar.success(f"✅ {api}")
    else:
        st.sidebar.warning(f"❌ {api}")

if not any(key and key != f"your_{api.lower().replace(' ', '_')}_key_here" for api, key in api_keys.items()):
    st.sidebar.info("💡 Run `python setup_api_keys.py` to set up API keys")

# Configuration display (if debug mode enabled)
if config.ui.show_debug_info:
    st.sidebar.subheader("Debug Info")
    st.sidebar.text(f"Config file: {os.getenv('CONFIG_FILE', 'default')}")
    st.sidebar.text(f"Log level: {config.logging.level}")
    st.sidebar.text(f"Cache TTL: {config.data_sources.cache_ttl_seconds}s")
    st.sidebar.text(f"Async processing: {config.data_sources.use_async_processing}")
    
    # Cache statistics
    cache_manager = get_cache_manager()
    if hasattr(cache_manager.backend, 'get_stats'):
        cache_stats = cache_manager.backend.get_stats()
        st.sidebar.text(f"Cache entries: {cache_stats.get('entries', 'N/A')}")
    
    # Clear cache buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear Data Cache"):
            cache_manager.clear()
            st.success("Data cache cleared!")
    with col2:
        if st.button("Clear Stock Cache"):
            clear_stock_cache()
            st.success("Stock cache cleared!")

# Backtest settings
use_backtest = st.sidebar.checkbox("Run Backtest")
if use_backtest:
    backtest_days = st.sidebar.slider(
        "Backtest Days", 
        7, 90, 
        config.backtest.default_period_days
    )
    backtest_start = datetime.now() - timedelta(days=backtest_days)
    backtest_end = datetime.now()

# Fetch new data button
if st.sidebar.button("Fetch Latest Data"):
    with LogContext(logger, "fetch_latest_data"):
        with st.spinner("Fetching latest financial news and stock data..."):
            # Validate tracked stocks
            validated_stocks = []
            for stock in tracked_stocks:
                try:
                    validated_stock = validate_ticker(stock)
                    validated_stocks.append(validated_stock)
                except Exception as e:
                    st.error(f"Invalid ticker {stock}: {str(e)}")
                    logger.warning(f"Invalid ticker provided: {stock}")
            
            if not validated_stocks:
                st.error("No valid stocks selected")
                logger.error("No valid stocks selected for data fetching")
            else:
                # Use async processing if enabled (local variable so we don't mutate config)
                use_async_this_run = config.data_sources.use_async_processing
                news_data = []
                stock_data = {}
                if use_async_this_run:
                    try:
                        with st.spinner("Fetching data using parallel processing..."):
                            with LogContext(logger, "async_complete_pipeline"):
                                news_data_dict, stock_data = run_async_complete_pipeline(validated_stocks)
                                news_data = []
                                for ticker, articles in news_data_dict.items():
                                    for article in articles:
                                        article['ticker'] = ticker
                                        news_data.append(article)
                                news_data = sorted(news_data, key=lambda x: x['date'], reverse=True)
                                logger.info(f"Async pipeline: fetched {len(news_data)} articles total")
                    except Exception as e:
                        st.error(f"Async processing failed: {str(e)}")
                        logger.error(f"Async processing failed, falling back to sync: {str(e)}")
                        use_async_this_run = False
                # Synchronous processing (fallback or if async disabled)
                if not use_async_this_run:
                    # Fetch news data with error handling
                    news_data = []
                    progress_bar = st.progress(0)
                    total_stocks = len(validated_stocks)
                    
                    for i, stock in enumerate(validated_stocks):
                        try:
                            with LogContext(logger, f"fetch_news_{stock}", ticker=stock):
                                stock_news = fetch_wsj_news(stock, config.data_sources.max_articles_per_stock)
                                for article in stock_news:
                                    article['ticker'] = stock
                                news_data.extend(stock_news)
                                logger.info(f"Fetched {len(stock_news)} articles for {stock}")
                        except NewsScrapingError as e:
                            st.warning(f"Could not fetch news for {stock}: {e.message}")
                            logger.warning(f"News scraping failed for {stock}: {e.message}")
                        except Exception as e:
                            st.error(f"Unexpected error fetching news for {stock}: {str(e)}")
                            logger.error(f"Unexpected error fetching news for {stock}: {str(e)}", exc_info=True)
                        
                        progress_bar.progress((i + 1) / total_stocks)
                    
                    progress_bar.empty()
                    
                    # Sort by date
                    news_data = sorted(news_data, key=lambda x: x['date'], reverse=True)
                    logger.info(f"Total articles fetched: {len(news_data)}")
                    
                    # Analyze sentiment and extract entities
                    with st.spinner("Analyzing sentiment..."):
                        for article in news_data:
                            try:
                                with LogContext(logger, f"sentiment_analysis", article_title=article['title'][:50]):
                                    sentiment = analyze_sentiment(
                                        article['title'] + " " + article['content'],
                                        financial_adjustment=config.sentiment.custom_terms_enabled
                                    )
                                    entities = get_named_entities(article['content'])
                                    article['sentiment'] = sentiment
                                    article['entities'] = entities
                            except SentimentAnalysisError as e:
                                st.warning(f"Sentiment analysis failed for article: {e.message}")
                                article['sentiment'] = 0
                                article['entities'] = []
                            except Exception as e:
                                logger.error(f"Unexpected error in sentiment analysis: {str(e)}", exc_info=True)
                                article['sentiment'] = 0
                                article['entities'] = []
                    
                    # Fetch stock data with error handling
                    stock_data = {}
                    with st.spinner("Fetching stock data..."):
                        for stock in validated_stocks:
                            try:
                                with LogContext(logger, f"fetch_stock_data_{stock}", ticker=stock):
                                    stock_data[stock] = get_stock_data(stock, config.data_sources.stock_data_period)
                                    logger.info(f"Fetched stock data for {stock}")
                            except Exception as e:
                                st.error(f"Error fetching stock data for {stock}: {str(e)}")
                                logger.error(f"Error fetching stock data for {stock}: {str(e)}", exc_info=True)
                
                # Generate trading signals with error handling - this is the core of the trading strategy
                try:
                    with LogContext(logger, "generate_signals"):
                        signals = generate_signals(
                            news_data, 
                            stock_data, 
                            sentiment_threshold,
                            position_size_pct,
                            stop_loss_pct,
                            take_profit_pct
                        )
                        logger.info(f"Generated {len(signals)} trading signals")
                except TradingError as e:
                    st.error(f"Error generating signals: {e.message}")
                    signals = []
                except Exception as e:
                    st.error(f"Unexpected error generating signals: {str(e)}")
                    logger.error(f"Unexpected error generating signals: {str(e)}", exc_info=True)
                    signals = []
                
                # Update session state
                st.session_state.news_data = news_data
                st.session_state.signals = signals
                st.session_state.last_update = datetime.now()
                
                # Update portfolio based on signals
                try:
                    with LogContext(logger, "update_portfolio"):
                        st.session_state.portfolio = update_portfolio(
                            st.session_state.portfolio,
                            signals,
                            stock_data
                        )
                        logger.info("Portfolio updated successfully")
                except Exception as e:
                    st.error(f"Error updating portfolio: {str(e)}")
                    logger.error(f"Error updating portfolio: {str(e)}", exc_info=True)
                
                st.success(f"Data updated successfully! Fetched {len(news_data)} articles and generated {len(signals)} signals.")
                logger.info("Data fetch completed successfully")

# Main content area - use tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dashboard",
    "🚀 Edge / Watchlist",
    "News Analysis",
    "Trading Signals",
    "Portfolio",
    "Big Mover Tracker",
])

with tab1:
    # Dashboard Overview - a single page with all the information
    st.header("Trading Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        try:
            portfolio_value = st.session_state.portfolio['cash']
            for ticker, position in st.session_state.portfolio['positions'].items():
                try:
                    current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                    portfolio_value += position['shares'] * current_price
                except Exception as e:
                    logger.warning(f"Could not fetch current price for {ticker}: {e}")
                    st.warning(f"Could not fetch current price for {ticker}")
        except (KeyError, AttributeError):
            portfolio_value = 100000.0
        
        delta = portfolio_value - 100000
        delta_class = "positive" if delta >= 0 else "negative"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Portfolio Value</h3>
            <div class="value">{format_currency(portfolio_value)}</div>
            <div class="delta {delta_class}">
                {format_currency(delta)} ({delta/1000:.1f}%)
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        try:
            signal_count = len(st.session_state.signals)
            buy_signals = sum(1 for s in st.session_state.signals if s['action'] == 'BUY')
            sell_signals = sum(1 for s in st.session_state.signals if s['action'] == 'SELL')
        except (KeyError, AttributeError):
            signal_count = 0
            buy_signals = 0
            sell_signals = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Trading Signals</h3>
        <div class="value">{signal_count}</div>
        <div class="delta">
            🟢 {buy_signals} Buy | 🔴 {sell_signals} Sell
        </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        try:
            if st.session_state.last_update:
                last_update = st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")
            else:
                last_update = "Never"
        except (KeyError, AttributeError):
            last_update = "Never"
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔄 Last Update</h3>
            <div class="value">{last_update}</div>
            <div class="delta">
                Data freshness
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
    # Edge / Watchlist - catch blow-ups early
    st.header("🚀 Edge / Watchlist")
    st.markdown(
        "**Catch movers early:** screen for **unusual volume** and **momentum**. "
        "Stocks with volume spikes and positive price action often move before headlines."
    )
    scan_tickers = tracked_stocks if tracked_stocks else (available_stocks[:50] if available_stocks else [])
    col_scan, col_info = st.columns([1, 2])
    with col_scan:
        do_scan = st.button("🔍 Scan for edge", type="primary", use_container_width=True)
    with col_info:
        if scan_tickers:
            st.caption(f"Scanning up to {min(50, len(scan_tickers))} tickers from your list.")
        else:
            st.caption("Add stocks to track in the sidebar, or pick a list above to scan.")
    if do_scan and scan_tickers:
        to_scan = (scan_tickers[:50] if len(scan_tickers) > 50 else scan_tickers)
        with st.spinner("Scanning volume & momentum..."):
            try:
                uv_df, tp_df, wl_df = run_full_scan(to_scan, period="1mo", min_volume_ratio=1.2, top_n=25)
                st.session_state.edge_scan = {
                    "unusual_volume": uv_df,
                    "top_performers": tp_df,
                    "watchlist": wl_df,
                    "tickers_scanned": to_scan,
                }
                st.success(f"Scanned {len(to_scan)} tickers.")
            except Exception as e:
                logger.error(f"Edge scan failed: {e}", exc_info=True)
                st.error(f"Scan failed: {str(e)}")
    edge = st.session_state.edge_scan
    has_any = edge and (
        (edge.get("unusual_volume") is not None and not edge["unusual_volume"].empty)
        or (edge.get("top_performers") is not None and not edge["top_performers"].empty)
        or (edge.get("watchlist") is not None and not edge["watchlist"].empty)
    )
    if has_any:
        uv = edge.get("unusual_volume")
        if uv is not None and not uv.empty:
            st.subheader("📊 Unusual volume (today vs 20d avg)")
            uv_display = uv.rename(columns={
                "volume_ratio": "Vol ratio",
                "current_volume": "Volume",
                "avg_volume": "Avg vol",
                "return_1d_pct": "1d %",
                "return_5d_pct": "5d %",
            })
            st.dataframe(uv_display[["ticker", "price", "Vol ratio", "Volume", "Avg vol", "1d %", "5d %"]], use_container_width=True)
        else:
            st.subheader("📊 Unusual volume")
            st.caption("No tickers above 1.2× average volume in this scan.")
        tp = edge.get("top_performers")
        if tp is not None and not tp.empty:
            st.subheader("📈 Top performers (5d return)")
            tp_display = tp.rename(columns={"return_1d_pct": "1d %", "return_5d_pct": "5d %", "volume_ratio": "Vol ratio"})
            st.dataframe(tp_display[["ticker", "price", "1d %", "5d %", "Vol ratio"]], use_container_width=True)
        wl = edge.get("watchlist")
        if wl is not None and not wl.empty:
            st.subheader("🎯 Watchlist (volume + momentum score)")
            st.caption("Higher score = unusual volume + positive momentum — potential early movers.")
            wl_display = wl.rename(columns={"edge_score": "Score", "return_5d_pct": "5d %", "volume_ratio": "Vol ratio"})
            st.dataframe(wl_display[["ticker", "price", "Score", "Vol ratio", "5d %"]], use_container_width=True)
    else:
        if not do_scan or not scan_tickers:
            st.info("Click **Scan for edge** to find unusual volume and top performers. Use the sidebar to choose stocks to scan.")
        else:
            st.warning("No results for this list. Try more tickers or check data availability.")

with tab3:
    # News Analysis
    st.header("📰 Financial News Analysis")
    
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

with tab4:
    # Trading Signals
    st.header("📊 Trading Signals")
    
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

with tab5:
    # Portfolio Tracker
    st.header("💼 Portfolio Tracker")
    
    # Portfolio Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>💰 Cash Balance</h3>
            <div class="value">{format_currency(st.session_state.portfolio['cash'])}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        positions_value = 0
        for ticker, position in st.session_state.portfolio['positions'].items():
            try:
                current_price = yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1]
                positions_value += position['shares'] * current_price
            except Exception as e:
                logger.warning(f"Could not fetch current price for {ticker}: {e}")
                st.warning(f"Could not fetch current price for {ticker}")
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Positions Value</h3>
            <div class="value">{format_currency(positions_value)}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        total_value = st.session_state.portfolio['cash'] + positions_value
        st.markdown(f"""
        <div class="metric-card">
            <h3>💎 Total Value</h3>
            <div class="value">{format_currency(total_value)}</div>
        </div>
        """, unsafe_allow_html=True)
    
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
            except Exception as e:
                logger.warning(f"Could not fetch current price for {ticker}: {e}")
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

with tab6:
    # Big Mover Tracker
    st.header("🚀 Big Mover Tracker")
    st.subheader("Real-time monitoring for stocks before they skyrocket")
    
    # Run the big mover dashboard
    run_big_mover_dashboard()

# Footer
st.markdown("""
<div style="text-align: center; padding: 2rem 0; margin-top: 3rem; border-top: 1px solid rgba(59, 130, 246, 0.2);">
    <p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 0; font-weight: 300;">
        FinNews Trader - Professional Algorithmic Trading Platform
    </p>
    <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.8rem; margin: 0.5rem 0 0 0; font-weight: 300;">
        Powered by Advanced News Intelligence & Real-Time Market Analysis
    </p>
</div>
""", unsafe_allow_html=True)
