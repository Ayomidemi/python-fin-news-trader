import logging
import logging.handlers
import os
import time
from pathlib import Path
from config import get_config

class FinNewsLogger:
    """Centralized logging system for FinNewsTrader"""
    
    _loggers = {}
    _initialized = False
    
    @classmethod
    def setup_logging(cls):
        """Initialize the logging system"""
        if cls._initialized:
            return
        
        config = get_config()
        
        # Create logs directory if it doesn't exist
        log_file_path = Path(config.logging.log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, config.logging.level.upper()))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Create formatter
        formatter = logging.Formatter(config.logging.format)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            config.logging.log_file,
            maxBytes=config.logging.max_file_size_mb * 1024 * 1024,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(getattr(logging, config.logging.level.upper()))
        root_logger.addHandler(file_handler)
        
        # Console handler (optional)
        if config.logging.console_logging:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(getattr(logging, config.logging.level.upper()))
            root_logger.addHandler(console_handler)
        
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger instance for a specific module"""
        if not cls._initialized:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]

# Convenience function to get logger
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return FinNewsLogger.get_logger(name)

# Module-specific loggers
def get_app_logger():
    """Get logger for main application"""
    return get_logger('app')

def get_news_logger():
    """Get logger for news scraping"""
    return get_logger('news_scraper')

def get_sentiment_logger():
    """Get logger for sentiment analysis"""
    return get_logger('sentiment_analyzer')

def get_trading_logger():
    """Get logger for trading strategy"""
    return get_logger('trading_strategy')

def get_portfolio_logger():
    """Get logger for portfolio tracking"""
    return get_logger('portfolio_tracker')

def get_backtest_logger():
    """Get logger for backtesting"""
    return get_logger('backtester')

def get_data_viz_logger():
    """Get logger for data visualization"""
    return get_logger('data_visualizer')

def get_utils_logger():
    """Get logger for utilities"""
    return get_logger('utils')

# Logging decorators for common use cases
def log_function_call(logger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} completed successfully")
                return result
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                raise
        return wrapper
    return decorator

def log_execution_time(logger):
    """Decorator to log function execution time"""
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.error(f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}")
                raise
        return wrapper
    return decorator

# Context manager for structured logging
class LogContext:
    """Context manager for structured logging with additional context"""
    
    def __init__(self, logger, operation: str, **context):
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        context_str = ", ".join([f"{k}={v}" for k, v in self.context.items()])
        self.logger.info(f"Starting {self.operation} - {context_str}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        execution_time = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.info(f"Completed {self.operation} in {execution_time:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {execution_time:.2f}s: {exc_val}")
        
        return False  # Don't suppress exceptions

# Initialize logging when module is imported
FinNewsLogger.setup_logging() 