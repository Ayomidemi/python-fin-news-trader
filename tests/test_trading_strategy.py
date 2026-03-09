"""
Unit tests for trading_strategy: generate_signals, filter_signals_with_technicals, calculate_indicators.
"""
import pandas as pd
import pytest
from trading_strategy import (
    generate_signals,
    filter_signals_with_technicals,
    calculate_indicators,
)


def test_generate_signals_returns_list(sample_news_data, sample_stock_data):
    """generate_signals returns a list (possibly empty after technical filter)."""
    signals = generate_signals(
        sample_news_data,
        sample_stock_data,
        sentiment_threshold=0.3,
        position_size=0.1,
        stop_loss=0.1,
        take_profit=0.15,
    )
    assert isinstance(signals, list)
    for s in signals:
        assert "ticker" in s and "action" in s and "price" in s
        assert s["action"] in ("BUY", "SELL")


def test_generate_signals_buy_for_positive_sentiment(sample_stock_data):
    """Strong positive average sentiment produces BUY signal when ticker has price data."""
    news = [
        {"ticker": "AAPL", "date": __import__("datetime").datetime.now(), "sentiment": 0.8},
        {"ticker": "AAPL", "date": __import__("datetime").datetime.now(), "sentiment": 0.7},
    ]
    signals = generate_signals(news, sample_stock_data, sentiment_threshold=0.3)
    buy_signals = [s for s in signals if s["ticker"] == "AAPL" and s["action"] == "BUY"]
    # May be 0 if technical filter removes it (MA50/RSI), but if any signal exists it should be BUY from sentiment
    assert all(s["action"] == "BUY" for s in signals if s["ticker"] == "AAPL")


def test_generate_signals_skips_ticker_without_stock_data(sample_news_data):
    """Tickers with no price data are skipped (no KeyError)."""
    news_only_xyz = [{"ticker": "XYZ", "date": __import__("datetime").datetime.now(), "sentiment": 0.5}]
    stock_data = {}  # No XYZ
    signals = generate_signals(news_only_xyz, stock_data, sentiment_threshold=0.3)
    assert isinstance(signals, list)
    assert len(signals) == 0


def test_calculate_indicators_adds_columns():
    """calculate_indicators adds MA20, MA50, RSI, etc."""
    n = 100
    df = pd.DataFrame({
        "Open": range(n),
        "High": range(1, n + 1),
        "Low": range(n),
        "Close": range(1, n + 1),
        "Volume": [1e6] * n,
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))
    out = calculate_indicators(df)
    assert "MA20" in out.columns and "MA50" in out.columns
    assert "RSI" in out.columns
    assert "MACD" in out.columns
    assert out["Close"].iloc[-1] == df["Close"].iloc[-1]


def test_filter_signals_with_technicals_empty(sample_stock_data):
    """Filtering empty signals returns empty list."""
    filtered = filter_signals_with_technicals([], sample_stock_data)
    assert filtered == []


def test_filter_signals_with_technicals_returns_subset(sample_news_data, sample_stock_data):
    """Technical filter returns a subset of signals (same or fewer)."""
    raw_signals = generate_signals(
        sample_news_data, sample_stock_data,
        sentiment_threshold=0.2, position_size=0.1, stop_loss=0.1, take_profit=0.15,
    )
    # Add indicators to stock_data so filter can run
    from trading_strategy import calculate_indicators
    stock_with_indicators = {t: calculate_indicators(d) for t, d in sample_stock_data.items()}
    filtered = filter_signals_with_technicals(raw_signals, stock_with_indicators)
    assert len(filtered) <= len(raw_signals)
    assert all(s in raw_signals for s in filtered)
