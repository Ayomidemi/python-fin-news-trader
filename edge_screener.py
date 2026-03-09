"""
Edge Screener - Surface stocks with unusual volume and momentum to catch blow-ups early.
Used by the Edge / Watchlist UI tab.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import logging

from logger import get_logger

logger = get_logger(__name__)

# Default lookback for volume average (trading days)
VOLUME_AVG_DAYS = 20
# Min days of data required
MIN_DAYS = 5


def _fetch_hist(ticker: str, period: str = "1mo") -> pd.DataFrame:
    """Fetch daily history for one ticker. Returns empty DataFrame on error."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1d")
        if hist is None or hist.empty or len(hist) < MIN_DAYS:
            return pd.DataFrame()
        return hist
    except Exception as e:
        logger.debug(f"edge_screener: fetch failed for {ticker}: {e}")
        return pd.DataFrame()


def _volume_and_returns(hist: pd.DataFrame, ticker: str) -> Optional[Dict[str, Any]]:
    """From daily OHLCV, compute current volume, avg volume, volume ratio, and 1d/5d returns."""
    if hist.empty or len(hist) < MIN_DAYS:
        return None
    close = hist["Close"]
    volume = hist["Volume"]
    current_price = float(close.iloc[-1])
    current_vol = int(volume.iloc[-1])
    avg_vol = float(volume.tail(VOLUME_AVG_DAYS).mean())
    if avg_vol <= 0:
        volume_ratio = 1.0
    else:
        volume_ratio = current_vol / avg_vol
    # Returns
    return_1d = float(close.pct_change(1).iloc[-1]) * 100 if len(close) >= 2 else 0.0
    return_5d = float(close.pct_change(min(5, len(close) - 1)).iloc[-1]) * 100 if len(close) > 5 else return_1d
    return {
        "ticker": ticker,
        "price": current_price,
        "current_volume": current_vol,
        "avg_volume": int(avg_vol),
        "volume_ratio": round(volume_ratio, 2),
        "return_1d_pct": round(return_1d, 2),
        "return_5d_pct": round(return_5d, 2),
    }


def get_unusual_volume(
    tickers: List[str],
    period: str = "1mo",
    min_volume_ratio: float = 1.2,
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Screen for unusual volume (today vs 20d avg). Sorted by volume_ratio descending.
    """
    rows = []
    for ticker in tickers:
        hist = _fetch_hist(ticker, period)
        row = _volume_and_returns(hist, ticker)
        if row and row["volume_ratio"] >= min_volume_ratio:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ticker", "price", "current_volume", "avg_volume", "volume_ratio", "return_1d_pct", "return_5d_pct"])
    df = pd.DataFrame(rows).sort_values("volume_ratio", ascending=False).head(top_n)
    return df.reset_index(drop=True)


def get_top_performers(
    tickers: List[str],
    period: str = "1mo",
    sort_by: str = "return_5d_pct",
    top_n: int = 25,
) -> pd.DataFrame:
    """
    Screen for top performers by 1d or 5d return. Sorted by sort_by descending.
    """
    rows = []
    for ticker in tickers:
        hist = _fetch_hist(ticker, period)
        row = _volume_and_returns(hist, ticker)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["ticker", "price", "current_volume", "avg_volume", "volume_ratio", "return_1d_pct", "return_5d_pct"])
    if sort_by not in ("return_1d_pct", "return_5d_pct"):
        sort_by = "return_5d_pct"
    df = pd.DataFrame(rows).sort_values(sort_by, ascending=False).head(top_n)
    return df.reset_index(drop=True)


def get_watchlist_scores(
    tickers: List[str],
    period: str = "1mo",
    volume_weight: float = 0.5,
    momentum_weight: float = 0.5,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Combined edge score: normalize volume_ratio and 5d return, then blend.
    Higher score = unusual volume + positive momentum (catch blow-ups early).
    """
    rows = []
    for ticker in tickers:
        hist = _fetch_hist(ticker, period)
        row = _volume_and_returns(hist, ticker)
        if row:
            rows.append(row)
    if not rows:
        return pd.DataFrame(
            columns=["ticker", "price", "volume_ratio", "return_5d_pct", "edge_score", "current_volume", "avg_volume", "return_1d_pct"]
        )
    df = pd.DataFrame(rows)
    # Normalize volume_ratio (log scale to reduce outlier effect): score 0–1
    vr = df["volume_ratio"].clip(lower=0.1)
    df["vol_score"] = (np.log1p(vr - 0.1) / np.log1p(10)).clip(0, 1)
    # Normalize return_5d: -20% -> 0, +20% -> 1, linear
    ret = df["return_5d_pct"]
    df["mom_score"] = ((ret - ret.min()) / (ret.max() - ret.min() + 1e-9)).fillna(0.5)
    df["edge_score"] = round(
        (volume_weight * df["vol_score"] + momentum_weight * df["mom_score"]) * 100, 1
    )
    out = df.sort_values("edge_score", ascending=False).head(top_n)
    return out[
        ["ticker", "price", "volume_ratio", "return_5d_pct", "edge_score", "current_volume", "avg_volume", "return_1d_pct"]
    ].reset_index(drop=True)


def run_full_scan(
    tickers: List[str],
    period: str = "1mo",
    min_volume_ratio: float = 1.2,
    top_n: int = 25,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run all three screens in one pass (one fetch per ticker, reuse for all).
    Returns (unusual_volume_df, top_performers_df, watchlist_df).
    """
    all_rows = []
    for ticker in tickers:
        hist = _fetch_hist(ticker, period)
        row = _volume_and_returns(hist, ticker)
        if row:
            all_rows.append(row)
    if not all_rows:
        empty = pd.DataFrame(columns=["ticker", "price", "current_volume", "avg_volume", "volume_ratio", "return_1d_pct", "return_5d_pct"])
        return empty.copy(), empty.copy(), empty.copy()
    df = pd.DataFrame(all_rows)
    # Unusual volume
    uv = df[df["volume_ratio"] >= min_volume_ratio].sort_values("volume_ratio", ascending=False).head(top_n)
    # Top performers
    tp = df.sort_values("return_5d_pct", ascending=False).head(top_n)
    # Watchlist score
    vr = df["volume_ratio"].clip(lower=0.1)
    df["vol_score"] = (np.log1p(vr - 0.1) / np.log1p(10)).clip(0, 1)
    ret = df["return_5d_pct"]
    df["mom_score"] = ((ret - ret.min()) / (ret.max() - ret.min() + 1e-9)).fillna(0.5)
    df["edge_score"] = round((0.5 * df["vol_score"] + 0.5 * df["mom_score"]) * 100, 1)
    wl = df.sort_values("edge_score", ascending=False).head(top_n)[
        ["ticker", "price", "volume_ratio", "return_5d_pct", "edge_score", "current_volume", "avg_volume", "return_1d_pct"]
    ]
    return uv.reset_index(drop=True), tp.reset_index(drop=True), wl.reset_index(drop=True)
