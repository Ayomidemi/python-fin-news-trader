# FinNews Trader

An algorithmic trading system that analyzes financial news and generates buy/sell signals.

## Features

- **News Analysis**: Analyzes financial news from Yahoo Finance and uses NLP to determine sentiment
- **Trading Signals**: Generates buy/sell signals based on news sentiment and technical indicators
- **Portfolio Management**: Tracks portfolio performance with position sizing and risk management
- **Backtesting**: Tests trading strategies on historical data
- **Interactive Dashboard**: Visualizes portfolio performance, stock prices, and sentiment trends

## Components

- `app.py`: Main Streamlit application
- `news_scraper.py`: Fetches financial news from various sources
- `sentiment_analyzer.py`: Analyzes sentiment of financial news
- `trading_strategy.py`: Generates trading signals based on sentiment and technical analysis
- `portfolio_tracker.py`: Tracks portfolio performance
- `data_visualizer.py`: Visualizes data using Plotly
- `backtester.py`: Backtests trading strategies
- `utils.py`: Utility functions

## Setup

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Run the app:
   ```
   streamlit run app.py
   ```

## Required Packages

- streamlit
- pandas
- numpy
- plotly
- yfinance
- nltk
- spacy
- trafilatura
- beautifulsoup4
- tqdm

## Usage

1. Select stocks to track in the sidebar
2. Set strategy parameters (sentiment threshold, position size, stop loss, take profit)
3. Click "Fetch Latest Data" to generate signals
4. Explore the dashboard, news analysis, trading signals, and portfolio tabs

## Configuration

You can modify strategy parameters in the sidebar:
- **Sentiment Threshold**: Threshold for sentiment to generate signal (-1.0 to 1.0)
- **Position Size**: Percentage of portfolio to allocate to each position
- **Stop Loss**: Percentage below entry price to exit losing positions
- **Take Profit**: Percentage above entry price to exit winning positions