# FinNews Trader 🚀

An advanced algorithmic trading system that analyzes financial news, generates trading signals, and includes a powerful **Big Mover Tracker** to identify stocks before they skyrocket.

## 🎯 Features

### Core Trading System

- **News Analysis**: Analyzes financial news from Yahoo Finance and uses NLP to determine sentiment
- **Trading Signals**: Generates buy/sell signals based on news sentiment and technical indicators
- **Portfolio Management**: Tracks portfolio performance with position sizing and risk management
- **Backtesting**: Tests trading strategies on historical data
- **Interactive Dashboard**: Visualizes portfolio performance, stock prices, and sentiment trends

### 🚀 Big Mover Tracker (NEW!)

- **Real-Time Price Monitoring**: Detects stocks before they experience significant price movements
- **Volume Analysis**: Identifies unusual volume patterns and institutional activity
- **News Correlation**: Links stock movements to breaking news events
- **Smart Alerts**: Multi-channel notifications with confidence scoring
- **Live Dashboard**: Real-time monitoring interface with visualizations

## 🏗️ Architecture

### Core Components

#### Trading System

- `app.py`: Main Streamlit application
- `news_scraper.py`: Fetches financial news from various sources
- `sentiment_analyzer.py`: Analyzes sentiment of financial news
- `trading_strategy.py`: Generates trading signals based on sentiment and technical analysis
- `portfolio_tracker.py`: Tracks portfolio performance
- `data_visualizer.py`: Visualizes data using Plotly
- `backtester.py`: Backtests trading strategies
- `utils.py`: Utility functions

#### Big Mover Tracker System

- `big_mover_tracker.py`: Main monitoring engine for price movements
- `volume_analyzer.py`: Advanced volume pattern detection and analysis
- `news_correlation_engine.py`: Links news events to stock movements
- `alert_system.py`: Multi-channel alert and notification system
- `big_mover_dashboard.py`: Streamlit UI for big mover tracking

#### Supporting Systems

- `config.py`: Configuration management with YAML support
- `logger.py`: Centralized logging system
- `errors.py`: Custom error handling and recovery
- `cache.py`: Caching system for performance optimization
- `async_processor.py`: Parallel data processing

## 🚀 Quick Start

### Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download spaCy model:
   ```bash
   python -m spacy download en_core_web_sm
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

### Basic Usage

1. **Trading System**:

   - Select stocks to track in the sidebar
   - Set strategy parameters (sentiment threshold, position size, stop loss, take profit)
   - Click "Fetch Latest Data" to generate signals
   - Explore the dashboard, news analysis, trading signals, and portfolio tabs

2. **Big Mover Tracker**:
   - Go to the "Big Mover Tracker" tab
   - Add tickers to monitor (e.g., AAPL, MSFT, TSLA)
   - Configure alert thresholds in the sidebar
   - Start monitoring and watch for alerts!

## 📊 Big Mover Tracker

### What It Detects

#### Price Movements

- **Price Spikes**: Significant price movements (configurable thresholds)
- **Gap Detection**: Pre-market and after-hours gaps
- **Breakout Patterns**: Technical breakouts above resistance levels
- **Momentum Shifts**: Trend changes and momentum analysis

#### Volume Patterns

- **Volume Surges**: 2x, 3x, 5x+ average volume activity
- **Accumulation**: Institutional buying patterns
- **Distribution**: Institutional selling patterns
- **Breakout Volume**: Volume supporting price breakouts
- **Exhaustion Volume**: End-of-trend signals

#### News Impact

- **High Impact News**: Major news events with significant market impact
- **Sentiment Shifts**: Significant changes in news sentiment
- **Breaking News**: Urgent news updates
- **Movement Correlation**: Links between news events and price movements

### Detection Algorithms

#### Price Movement Detection

1. **Price Spike**: Compares current price change to threshold
2. **Gap Detection**: Analyzes price gaps between sessions
3. **Breakout**: Identifies breakouts above resistance levels
4. **Momentum**: Tracks short-term momentum changes

#### Volume Pattern Detection

1. **Volume Surge**: Sustained high volume activity
2. **Volume Spikes**: Sudden volume bursts
3. **Accumulation**: Increasing volume with stable/rising prices
4. **Distribution**: High volume with declining prices
5. **Breakout Volume**: Volume supporting price breakouts
6. **Exhaustion Volume**: High volume with weak price movement

#### News Impact Analysis

1. **Sentiment Scoring**: VADER sentiment analysis with financial adjustments
2. **Impact Scoring**: Keyword-based impact assessment
3. **Correlation Analysis**: Links news events to price movements
4. **Confidence Scoring**: Multi-factor confidence calculation

### Alert System

#### Alert Types

- **Movement Alerts**: Price spikes, gaps, breakouts, momentum
- **Volume Alerts**: Surges, spikes, accumulation, distribution
- **News Alerts**: High impact news, sentiment shifts, correlations

#### Alert Channels

- **Email**: SMTP-based email notifications
- **SMS**: Text message alerts
- **Webhook**: HTTP POST to custom endpoints
- **Console**: Terminal output
- **Dashboard**: In-app notifications

#### Alert Configuration

```python
# Example alert rule
rule = AlertRule(
    name="price_spike",
    enabled=True,
    priority=AlertPriority.HIGH,
    channels=[AlertChannel.EMAIL, AlertChannel.CONSOLE],
    conditions={
        "movement_type": MovementType.PRICE_SPIKE,
        "min_price_change_pct": 5.0,
        "min_confidence": 0.7
    },
    cooldown_minutes=15
)
```

## 💻 API Usage

### Big Mover Tracker

```python
from big_mover_tracker import create_big_mover_tracker
import asyncio

async def main():
    # Create tracker
    tracker = create_big_mover_tracker()

    # Add tickers to monitor
    tracker.add_ticker("AAPL")
    tracker.add_ticker("MSFT")
    tracker.add_ticker("GOOGL")

    # Scan for big movers
    alerts = await tracker.scan_for_movers()

    for alert in alerts:
        print(f"Big mover: {alert.ticker} - {alert.reason}")

asyncio.run(main())
```

### Volume Analysis

```python
from volume_analyzer import analyze_ticker_volume

async def analyze_volume():
    analysis = await analyze_ticker_volume("AAPL")

    if analysis and 'patterns' in analysis:
        for pattern in analysis['patterns']:
            print(f"Volume pattern: {pattern.pattern_type.value}")

asyncio.run(analyze_volume())
```

### News Correlation

```python
from news_correlation_engine import analyze_news_impact

async def analyze_news():
    news_events = await analyze_news_impact("TSLA")

    for event in news_events:
        print(f"News: {event.title}")
        print(f"Impact: {event.impact_score}")

asyncio.run(analyze_news())
```

### Continuous Monitoring

```python
async def continuous_monitoring():
    tracker = create_big_mover_tracker(["AAPL", "MSFT", "GOOGL"])

    # Start monitoring (scans every 60 seconds)
    await tracker.start_monitoring(scan_interval=60)

asyncio.run(continuous_monitoring())
```

## ⚙️ Configuration

### Trading System Parameters

You can modify strategy parameters in the sidebar:

- **Sentiment Threshold**: Threshold for sentiment to generate signal (-1.0 to 1.0)
- **Position Size**: Percentage of portfolio to allocate to each position
- **Stop Loss**: Percentage below entry price to exit losing positions
- **Take Profit**: Percentage above entry price to exit winning positions

### Big Mover Tracker Settings

#### Movement Detection Thresholds

```python
# Price spike threshold (5% by default)
tracker.price_spike_threshold = 0.05

# Volume surge threshold (2x average by default)
tracker.volume_surge_threshold = 2.0

# Gap threshold (3% by default)
tracker.gap_threshold = 0.03

# Momentum threshold (2% by default)
tracker.momentum_threshold = 0.02
```

#### Alert Configuration

- **Price Change Threshold**: Minimum price change percentage to trigger alerts
- **Volume Surge Threshold**: Minimum volume ratio to trigger alerts
- **Confidence Threshold**: Minimum confidence score for alerts
- **Cooldown Period**: Time between alerts for the same ticker

## 📈 Performance Features

### Optimization

- **Caching**: Price data, news data, and sentiment analysis caching
- **Async Processing**: Parallel data fetching for multiple tickers
- **Resource Management**: Configurable worker limits and memory management
- **Smart Filtering**: Reduces false positives with technical confirmation

### Monitoring

- **Real-Time Updates**: Live data refresh and monitoring
- **Performance Metrics**: Alert statistics and delivery success rates
- **Error Handling**: Robust error recovery and fallback mechanisms
- **Logging**: Comprehensive logging for debugging and monitoring

## 🔧 Troubleshooting

### Common Issues

1. **No Alerts Generated**

   - Check if tickers are properly added
   - Verify thresholds are not too high
   - Ensure monitoring is started

2. **High False Positives**

   - Increase confidence thresholds
   - Adjust movement thresholds
   - Enable technical confirmation

3. **Missing News Data**

   - Check internet connection
   - Verify news API access
   - Check rate limiting

4. **Performance Issues**
   - Reduce number of monitored tickers
   - Increase scan intervals
   - Enable caching

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📚 Required Packages

### Core Dependencies

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

### Additional Dependencies

- requests
- pyyaml
- aiohttp

## 🚀 Example Scripts

### Run Examples

```bash
# Run the example script
python example_big_mover.py

# Run the main application
streamlit run app.py
```

### Example Output

```
🚀 Big Mover Tracker Example
==================================================
Monitoring tickers: AAPL, MSFT, GOOGL, TSLA, NVDA

🔍 Scanning for big movers...

📊 Found 2 big mover alerts:
  • AAPL: price_spike - Price spike of 5.2%
    Price: $150.25 (+5.2%)
    Volume: 45,000,000 (2.3x avg)
    Confidence: 85%

  • TSLA: volume_surge - Volume surge: 3.1x average volume
    Price: $245.80 (+2.1%)
    Volume: 78,000,000 (3.1x avg)
    Confidence: 78%
```

## 🤝 Contributing

To contribute to FinNews Trader:

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Write tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Yahoo Finance for market data
- NLTK and spaCy for NLP capabilities
- Streamlit for the dashboard interface
- Plotly for data visualization
- The open-source community for various libraries and tools

## 📞 Support

For questions, issues, or feature requests:

- Create an issue on GitHub
- Check the troubleshooting section
- Review the example scripts
- Consult the API documentation

---

**Happy Trading! 🚀📈**
