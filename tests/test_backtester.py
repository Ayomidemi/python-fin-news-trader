"""
Unit tests for backtester: run_backtest with mocked yfinance and streamlit.
"""
import pandas as pd
import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


def test_run_backtest_returns_expected_keys():
    """run_backtest returns a dict with total_return_pct, win_rate, equity_curve, trades, etc."""
    tickers = ["AAPL"]
    start = datetime.now() - timedelta(days=5)
    end = datetime.now()
    # Build minimal OHLC data for yf.download
    dates = pd.date_range(start=start - timedelta(days=100), end=end + timedelta(days=1), freq="B")
    fake_df = pd.DataFrame({
        "Open": [100.0] * len(dates),
        "High": [101.0] * len(dates),
        "Low": [99.0] * len(dates),
        "Close": [100.0 + (i % 5) for i in range(len(dates))],
        "Adj Close": [100.0] * len(dates),
        "Volume": [1e6] * len(dates),
    }, index=dates)
    if isinstance(fake_df.columns, pd.MultiIndex):
        fake_df.columns = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    fake_df = fake_df.loc[~fake_df.index.duplicated(keep="first")]

    with patch("yfinance.download", return_value=fake_df):
        with patch("streamlit.spinner", lambda x: MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None)):
            with patch("streamlit.error", lambda x: None):
                from backtester import run_backtest
                result = run_backtest(
                    tickers,
                    start,
                    end,
                    sentiment_threshold=0.3,
                    position_size=0.1,
                    stop_loss=0.1,
                    take_profit=0.15,
                )
    assert isinstance(result, dict)
    assert "total_return_pct" in result
    assert "win_rate" in result
    assert "profit_factor" in result
    assert "max_drawdown_pct" in result
    assert "equity_curve" in result
    assert "dates" in result
    assert "trades" in result
    assert isinstance(result["equity_curve"], list)
    assert isinstance(result["trades"], list)


def test_run_backtest_empty_tickers():
    """run_backtest with no tickers still returns valid structure (no download)."""
    with patch("yfinance.download", return_value=pd.DataFrame()) as mock_dl:
        with patch("streamlit.spinner", lambda x: MagicMock(__enter__=lambda s: s, __exit__=lambda s, *a: None)):
            with patch("streamlit.error", lambda x: None):
                from backtester import run_backtest
                start = datetime.now() - timedelta(days=7)
                end = datetime.now()
                result = run_backtest([], start, end)
    assert "equity_curve" in result
    assert result["equity_curve"][0] == 100000.0
    assert result["total_return_pct"] == 0.0
