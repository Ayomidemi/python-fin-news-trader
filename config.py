import os
from dataclasses import dataclass
from typing import List, Dict, Any
import yaml
from pathlib import Path

@dataclass
class TradingConfig:
    """Trading strategy configuration"""
    default_position_size: float = 0.10
    default_stop_loss: float = 0.10
    default_take_profit: float = 0.15
    default_sentiment_threshold: float = 0.3
    max_portfolio_allocation: float = 0.80
    min_cash_reserve: float = 0.20

@dataclass
class SentimentConfig:
    """Sentiment analysis configuration"""
    model: str = "vader"  # Options: vader, finbert
    threshold: float = 0.3
    financial_weight: float = 0.05
    custom_terms_enabled: bool = True

@dataclass
class DataSourceConfig:
    """Data source configuration"""
    news_apis: List[str] = None
    max_articles_per_stock: int = 10
    stock_data_period: str = "3mo"
    cache_ttl_seconds: int = 300
    retry_attempts: int = 3
    timeout_seconds: int = 30
    
    def __post_init__(self):
        if self.news_apis is None:
            self.news_apis = ["yahoo_finance"]

@dataclass
class UIConfig:
    """User interface configuration"""
    default_stocks: List[str] = None
    available_stocks: List[str] = None
    refresh_interval_seconds: int = 60
    chart_height: int = 500
    show_debug_info: bool = False
    
    def __post_init__(self):
        if self.default_stocks is None:
            self.default_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
        if self.available_stocks is None:
            self.available_stocks = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
                "TSLA", "NVDA", "JPM", "V", "WMT"
            ]

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100000.0
    commission_per_trade: float = 0.0
    slippage_bps: float = 0.0
    default_period_days: int = 30
    min_trade_amount: float = 100.0

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/trader.log"
    max_file_size_mb: int = 10
    backup_count: int = 3
    console_logging: bool = True

class Config:
    """Main configuration class that holds all settings"""
    
    def __init__(self, config_file: str = None):
        self.trading = TradingConfig()
        self.sentiment = SentimentConfig()
        self.data_sources = DataSourceConfig()
        self.ui = UIConfig()
        self.backtest = BacktestConfig()
        self.logging = LoggingConfig()
        
        # Load from file if provided
        if config_file and Path(config_file).exists():
            self.load_from_file(config_file)
        
        # Override with environment variables
        self._load_from_env()
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)
            
            if 'trading' in data:
                self._update_dataclass(self.trading, data['trading'])
            if 'sentiment' in data:
                self._update_dataclass(self.sentiment, data['sentiment'])
            if 'data_sources' in data:
                self._update_dataclass(self.data_sources, data['data_sources'])
            if 'ui' in data:
                self._update_dataclass(self.ui, data['ui'])
            if 'backtest' in data:
                self._update_dataclass(self.backtest, data['backtest'])
            if 'logging' in data:
                self._update_dataclass(self.logging, data['logging'])
                
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _update_dataclass(self, obj, data: Dict[str, Any]):
        """Update dataclass fields from dictionary"""
        for key, value in data.items():
            if hasattr(obj, key):
                setattr(obj, key, value)
    
    def _load_from_env(self):
        """Load configuration from environment variables"""
        # Trading config
        if os.getenv('TRADING_POSITION_SIZE'):
            self.trading.default_position_size = float(os.getenv('TRADING_POSITION_SIZE'))
        if os.getenv('TRADING_STOP_LOSS'):
            self.trading.default_stop_loss = float(os.getenv('TRADING_STOP_LOSS'))
        if os.getenv('TRADING_TAKE_PROFIT'):
            self.trading.default_take_profit = float(os.getenv('TRADING_TAKE_PROFIT'))
        
        # Sentiment config
        if os.getenv('SENTIMENT_THRESHOLD'):
            self.sentiment.threshold = float(os.getenv('SENTIMENT_THRESHOLD'))
        if os.getenv('SENTIMENT_MODEL'):
            self.sentiment.model = os.getenv('SENTIMENT_MODEL')
        
        # Data sources
        if os.getenv('MAX_ARTICLES'):
            self.data_sources.max_articles_per_stock = int(os.getenv('MAX_ARTICLES'))
        if os.getenv('CACHE_TTL'):
            self.data_sources.cache_ttl_seconds = int(os.getenv('CACHE_TTL'))
        
        # Logging
        if os.getenv('LOG_LEVEL'):
            self.logging.level = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.logging.log_file = os.getenv('LOG_FILE')
    
    def save_to_file(self, config_file: str):
        """Save current configuration to YAML file"""
        config_data = {
            'trading': {
                'default_position_size': self.trading.default_position_size,
                'default_stop_loss': self.trading.default_stop_loss,
                'default_take_profit': self.trading.default_take_profit,
                'default_sentiment_threshold': self.trading.default_sentiment_threshold,
                'max_portfolio_allocation': self.trading.max_portfolio_allocation,
                'min_cash_reserve': self.trading.min_cash_reserve,
            },
            'sentiment': {
                'model': self.sentiment.model,
                'threshold': self.sentiment.threshold,
                'financial_weight': self.sentiment.financial_weight,
                'custom_terms_enabled': self.sentiment.custom_terms_enabled,
            },
            'data_sources': {
                'news_apis': self.data_sources.news_apis,
                'max_articles_per_stock': self.data_sources.max_articles_per_stock,
                'stock_data_period': self.data_sources.stock_data_period,
                'cache_ttl_seconds': self.data_sources.cache_ttl_seconds,
                'retry_attempts': self.data_sources.retry_attempts,
                'timeout_seconds': self.data_sources.timeout_seconds,
            },
            'ui': {
                'default_stocks': self.ui.default_stocks,
                'available_stocks': self.ui.available_stocks,
                'refresh_interval_seconds': self.ui.refresh_interval_seconds,
                'chart_height': self.ui.chart_height,
                'show_debug_info': self.ui.show_debug_info,
            },
            'backtest': {
                'initial_capital': self.backtest.initial_capital,
                'commission_per_trade': self.backtest.commission_per_trade,
                'slippage_bps': self.backtest.slippage_bps,
                'default_period_days': self.backtest.default_period_days,
                'min_trade_amount': self.backtest.min_trade_amount,
            },
            'logging': {
                'level': self.logging.level,
                'format': self.logging.format,
                'log_file': self.logging.log_file,
                'max_file_size_mb': self.logging.max_file_size_mb,
                'backup_count': self.logging.backup_count,
                'console_logging': self.logging.console_logging,
            }
        }
        
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, indent=2)

# Global configuration instance
_config = None

def get_config() -> Config:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        config_file = os.getenv('CONFIG_FILE', 'config/settings.yaml')
        _config = Config(config_file)
    return _config

def reload_config(config_file: str = None):
    """Reload configuration from file"""
    global _config
    if config_file is None:
        config_file = os.getenv('CONFIG_FILE', 'config/settings.yaml')
    _config = Config(config_file)
    return _config 