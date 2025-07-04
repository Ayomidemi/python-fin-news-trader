import argparse
from datetime import datetime, timedelta
import yfinance as yf
from news_scraper import fetch_wsj_news
from sentiment_analyzer import analyze_sentiment, get_named_entities
from trading_strategy import generate_signals
from portfolio_tracker import update_portfolio, calculate_performance
from utils import format_currency, get_stock_data

def analyze_stock(stock_symbol, days=30):
    """Analyze a single stock and generate trading signals."""
    print(f"\nAnalyzing {stock_symbol} for the past {days} days...")
    
    # Fetch news
    print("Fetching news...")
    news = fetch_wsj_news(stock_symbol)
    
    # Analyze sentiment
    print("Analyzing sentiment...")
    for article in news:
        sentiment = analyze_sentiment(article['title'] + " " + article['content'])
        print(f"\nArticle: {article['title']}")
        print(f"Date: {article['date']}")
        print(f"Sentiment: {sentiment:.2f}")
    
    # Get stock data
    print("\nFetching stock data...")
    stock_data = get_stock_data(stock_symbol)
    
    # Generate signals
    print("\nGenerating trading signals...")
    signals = generate_signals(
        news,
        {stock_symbol: stock_data},
        sentiment_threshold=0.3,
        position_size_pct=0.1,
        stop_loss_pct=0.1,
        take_profit_pct=0.15
    )
    
    # Display signals
    print("\nTrading Signals:")
    for signal in signals:
        print(f"- {signal['action']} {signal['ticker']} at ${signal['price']:.2f}")
        print(f"  Reason: {signal['reason']}")
        print(f"  Stop Loss: ${signal['stop_loss']:.2f}")
        print(f"  Take Profit: ${signal['take_profit']:.2f}")

def main():
    parser = argparse.ArgumentParser(description='FinNews Trader CLI')
    parser.add_argument('--stock', required=True, help='Stock symbol to analyze')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    args = parser.parse_args()

    try:
        analyze_stock(args.stock, args.days)
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 