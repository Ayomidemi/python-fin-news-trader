"""
Unit tests for portfolio_tracker: update_portfolio, calculate_performance.
"""
import pandas as pd
import pytest
from datetime import datetime
from portfolio_tracker import update_portfolio, calculate_performance


def test_update_portfolio_does_not_mutate_input(sample_portfolio, sample_stock_data):
    """update_portfolio returns a new dict and does not mutate the input portfolio."""
    signals = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "price": 150.0,
            "position_size": 0.1,
            "stop_loss": 135.0,
            "take_profit": 172.5,
            "reason": "test",
        }
    ]
    cash_before = sample_portfolio["cash"]
    pos_before = len(sample_portfolio["positions"])
    out = update_portfolio(sample_portfolio, signals, sample_stock_data)
    assert out is not sample_portfolio
    assert sample_portfolio["cash"] == cash_before
    assert len(sample_portfolio["positions"]) == pos_before
    assert out["cash"] <= cash_before
    assert "AAPL" in out["positions"] or out["cash"] < cash_before


def test_update_portfolio_buy_reduces_cash(sample_portfolio, sample_stock_data):
    """A BUY signal reduces cash and adds a position."""
    price = 100.0
    signals = [
        {
            "ticker": "AAPL",
            "action": "BUY",
            "price": price,
            "position_size": 0.1,
            "stop_loss": price * 0.9,
            "take_profit": price * 1.15,
            "reason": "test",
        }
    ]
    out = update_portfolio(sample_portfolio, signals, sample_stock_data)
    assert out["cash"] < sample_portfolio["cash"]
    assert "AAPL" in out["positions"]
    assert out["positions"]["AAPL"]["shares"] >= 0
    assert out["positions"]["AAPL"]["avg_price"] == price
    assert len(out["history"]) >= 1
    assert out["history"][-1]["action"] == "BUY"


def test_update_portfolio_sell_increases_cash(sample_portfolio, sample_stock_data):
    """BUY then SELL: cash goes down then back up (minus spread)."""
    price = 100.0
    buy_signal = {
        "ticker": "AAPL",
        "action": "BUY",
        "price": price,
        "position_size": 0.05,
        "stop_loss": price * 0.9,
        "take_profit": price * 1.15,
        "reason": "buy",
    }
    after_buy = update_portfolio(sample_portfolio, [buy_signal], sample_stock_data)
    shares = after_buy["positions"]["AAPL"]["shares"]
    sell_signal = {
        "ticker": "AAPL",
        "action": "SELL",
        "price": 105.0,
        "shares": shares,
        "reason": "sell",
    }
    after_sell = update_portfolio(after_buy, [sell_signal], sample_stock_data)
    assert "AAPL" not in after_sell["positions"]
    assert after_sell["cash"] > after_buy["cash"]
    assert any(h["action"] == "SELL" for h in after_sell["history"])


def test_calculate_performance_adds_snapshot(sample_portfolio, sample_stock_data):
    """calculate_performance appends a performance snapshot."""
    perf = sample_portfolio["performance"].copy()
    out = calculate_performance(sample_portfolio, sample_stock_data)
    assert len(out["performance"]) == len(perf) + 1
    last = out["performance"][-1]
    assert "total_value" in last
    assert "cash" in last
    assert last["total_value"] >= out["cash"]


def test_calculate_performance_empty_positions(sample_portfolio, sample_stock_data):
    """With no positions, total_value equals cash."""
    out = calculate_performance(sample_portfolio, sample_stock_data)
    assert out["performance"][-1]["total_value"] == sample_portfolio["cash"]
    assert out["performance"][-1]["positions_value"] == 0
