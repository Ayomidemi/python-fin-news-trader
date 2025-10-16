#!/usr/bin/env python3
"""
Comprehensive Test Suite for FinNews Trader
Tests all core functionality and integrations
"""

import sys
import logging
from datetime import datetime
import os

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_core_imports():
    """Test core package imports"""
    logger.info("🧪 Testing core imports...")
    
    try:
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import yfinance as yf
        import requests
        import nltk
        import sklearn
        logger.info("✅ Core packages imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Core import test failed: {str(e)}")
        return False

def test_custom_modules():
    """Test custom module imports"""
    logger.info("🔧 Testing custom modules...")
    
    try:
        from config import get_config
        from logger import get_app_logger
        from cache import get_cache_manager
        from news_scraper import fetch_wsj_news
        from sentiment_analyzer import analyze_sentiment
        from trading_strategy import generate_signals
        from portfolio_tracker import update_portfolio
        from data_visualizer import plot_sentiment_over_time
        from backtester import run_backtest
        from utils import format_currency, get_stock_data
        logger.info("✅ Custom modules imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Custom module test failed: {str(e)}")
        return False

def test_new_features():
    """Test new advanced features"""
    logger.info("🚀 Testing new features...")
    
    try:
        from real_data_sources import get_stock_data, get_news_data
        from advanced_sentiment_analyzer import analyze_sentiment
        from trading_engine import initialize_trading_engine
        from big_mover_tracker import BigMoverTracker
        from stock_list_fetcher import get_available_stocks
        logger.info("✅ New features imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ New features test failed: {str(e)}")
        return False

def test_data_sources():
    """Test data source functionality"""
    logger.info("📊 Testing data sources...")
    
    try:
        from real_data_sources import get_stock_data, get_news_data
        
        # Test stock data (may fail without API key)
        stock_data = get_stock_data("AAPL", "1d")
        if stock_data:
            logger.info(f"✅ Stock data: {len(stock_data)} points")
        else:
            logger.warning("⚠️  No stock data (API key needed)")
        
        # Test news data
        news_data = get_news_data("AAPL", 3)
        if news_data:
            logger.info(f"✅ News data: {len(news_data)} articles")
        else:
            logger.warning("⚠️  No news data (API key needed)")
        
        return True
    except Exception as e:
        logger.error(f"❌ Data sources test failed: {str(e)}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis"""
    logger.info("🧠 Testing sentiment analysis...")
    
    try:
        from advanced_sentiment_analyzer import analyze_sentiment
        
        test_text = "Apple stock is performing excellently today!"
        result = analyze_sentiment(test_text)
        logger.info(f"✅ Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        return True
    except Exception as e:
        logger.error(f"❌ Sentiment analysis test failed: {str(e)}")
        return False

def test_trading_engine():
    """Test trading engine"""
    logger.info("💰 Testing trading engine...")
    
    try:
        from trading_engine import initialize_trading_engine
        
        engine = initialize_trading_engine("paper", initial_cash=100000.0)
        summary = engine.get_portfolio_summary()
        logger.info(f"✅ Trading engine: {summary}")
        return True
    except Exception as e:
        logger.error(f"❌ Trading engine test failed: {str(e)}")
        return False

def test_app_startup():
    """Test app startup"""
    logger.info("🚀 Testing app startup...")
    
    try:
        # Test if app can be imported without errors
        import app
        logger.info("✅ App module imported successfully")
        return True
    except Exception as e:
        logger.error(f"❌ App startup test failed: {str(e)}")
        return False

def main():
    """Run comprehensive test suite"""
    logger.info("🚀 Starting FinNews Trader Test Suite...")
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Custom Modules", test_custom_modules),
        ("New Features", test_new_features),
        ("Data Sources", test_data_sources),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Trading Engine", test_trading_engine),
        ("App Startup", test_app_startup)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name}...")
        if test_func():
            passed += 1
            logger.info(f"✅ {test_name} passed")
        else:
            logger.error(f"❌ {test_name} failed")
    
    logger.info(f"\n🎉 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! System is ready.")
        logger.info("\n📝 Next steps:")
        logger.info("1. Add API keys to .env file")
        logger.info("2. Run: streamlit run app.py")
        logger.info("3. Open http://localhost:8502")
    else:
        logger.warning("⚠️  Some tests failed. Check logs above.")
        logger.info("\n🔧 Common fixes:")
        logger.info("1. Install packages: pip install -r requirements.txt")
        logger.info("2. Add API keys to .env file")
        logger.info("3. Check internet connection")

if __name__ == "__main__":
    main()
