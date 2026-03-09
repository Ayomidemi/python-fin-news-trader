"""
Pytest fixtures for FinNews Trader tests.
Provides sample data and mocks so tests don't hit real APIs.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytest


@pytest.fixture
def sample_news_data():
    """News articles with sentiment for testing signal generation."""
    return [
        {"ticker": "AAPL", "date": datetime.now(), "title": "Apple beats", "content": "Strong results", "sentiment": 0.6},
        {"ticker": "AAPL", "date": datetime.now(), "title": "Apple growth", "content": "Revenue up", "sentiment": 0.5},
        {"ticker": "MSFT", "date": datetime.now(), "title": "Microsoft warning", "content": "Guidance cut", "sentiment": -0.5},
    ]


@pytest.fixture
def sample_stock_data():
    """DataFrames with OHLC for testing. Enough rows for MA50/RSI."""
    n = 60
    dates = pd.date_range(end=datetime.now(), periods=n, freq="B")
    def _frame(ticker, base=100):
        np.random.seed(hash(ticker) % 2**32)
        close = base + np.cumsum(np.random.randn(n) * 0.5)
        close = np.maximum(close, 1)
        return pd.DataFrame({
            "Open": close - 0.5,
            "High": close + 0.5,
            "Low": close - 1,
            "Close": close,
            "Volume": 1_000_000,
        }, index=dates)
    return {
        "AAPL": _frame("AAPL", 150),
        "MSFT": _frame("MSFT", 380),
    }


@pytest.fixture
def sample_portfolio():
    """Initial portfolio state."""
    return {
        "cash": 100_000.0,
        "positions": {},
        "history": [],
        "performance": [],
    }


@pytest.fixture(autouse=True)
def mock_streamlit(monkeypatch):
    """Mock Streamlit so backtester and app code can run in tests."""
    try:
        import streamlit as st
        # Replace spinner with a no-op context manager
        class NoOpSpinner:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        monkeypatch.setattr(st, "spinner", lambda _: NoOpSpinner())
        monkeypatch.setattr(st, "error", lambda _: None)
        progress_bar = type("Progress", (), {"progress": lambda self, x: None, "empty": lambda self: None})()
        monkeypatch.setattr(st, "progress", lambda _: progress_bar)
    except ImportError:
        pass
