"""
Custom exception classes and error handling utilities for FinNewsTrader
"""

import functools
import traceback
from typing import Optional, Dict, Any, Callable
from logger import get_logger

# Base exception classes
class FinNewsError(Exception):
    """Base exception class for FinNewsTrader"""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self):
        """Convert exception to dictionary for logging/serialization"""
        return {
            'error_type': self.__class__.__name__,
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details
        }

# Data-related exceptions
class DataSourceError(FinNewsError):
    """Exception raised when data source operations fail"""
    pass

class NewsScrapingError(DataSourceError):
    """Exception raised when news scraping fails"""
    pass

class StockDataError(DataSourceError):
    """Exception raised when stock data fetching fails"""
    pass

class APIError(DataSourceError):
    """Exception raised when API calls fail"""
    
    def __init__(self, message: str, status_code: int = None, response_data: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_data = response_data
        self.details = {
            'status_code': status_code,
            'response_data': str(response_data) if response_data else None
        }

# Analysis-related exceptions
class SentimentAnalysisError(FinNewsError):
    """Exception raised when sentiment analysis fails"""
    pass

class TechnicalAnalysisError(FinNewsError):
    """Exception raised when technical analysis fails"""
    pass

# Trading-related exceptions
class TradingError(FinNewsError):
    """Exception raised when trading operations fail"""
    pass

class SignalGenerationError(TradingError):
    """Exception raised when signal generation fails"""
    pass

class PortfolioError(TradingError):
    """Exception raised when portfolio operations fail"""
    pass

class BacktestError(TradingError):
    """Exception raised when backtesting fails"""
    pass

# Configuration-related exceptions
class ConfigurationError(FinNewsError):
    """Exception raised when configuration is invalid"""
    pass

class ValidationError(FinNewsError):
    """Exception raised when data validation fails"""
    pass

# Error handler decorator
def handle_errors(
    logger=None,
    default_return=None,
    reraise: bool = True,
    error_message: str = None
):
    """
    Decorator to handle errors in functions
    
    Args:
        logger: Logger instance to use for error logging
        default_return: Value to return if error occurs and reraise=False
        reraise: Whether to reraise the exception after logging
        error_message: Custom error message prefix
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_logger = logger or get_logger(func.__module__)
            
            try:
                return func(*args, **kwargs)
            except FinNewsError as e:
                # Log our custom exceptions with full context
                func_logger.error(
                    f"{error_message or f'Error in {func.__name__}'}: {e.message}",
                    extra={'error_details': e.to_dict()}
                )
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # Log unexpected exceptions
                func_logger.error(
                    f"{error_message or f'Unexpected error in {func.__name__}'}: {str(e)}",
                    exc_info=True
                )
                if reraise:
                    # Wrap in our custom exception
                    raise FinNewsError(
                        f"Unexpected error in {func.__name__}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        details={'original_exception': str(e), 'traceback': traceback.format_exc()}
                    )
                return default_return
        return wrapper
    return decorator

# Specific error handlers for different operations
def handle_data_source_errors(func: Callable):
    """Decorator specifically for data source operations"""
    return handle_errors(
        error_message=f"Data source error in {func.__name__}",
        reraise=True
    )(func)

def handle_analysis_errors(func: Callable):
    """Decorator specifically for analysis operations"""
    return handle_errors(
        error_message=f"Analysis error in {func.__name__}",
        reraise=True
    )(func)

def handle_trading_errors(func: Callable):
    """Decorator specifically for trading operations"""
    return handle_errors(
        error_message=f"Trading error in {func.__name__}",
        reraise=True
    )(func)

# Error recovery utilities
class ErrorRecovery:
    """Utilities for error recovery and retry logic"""
    
    @staticmethod
    def retry_on_error(
        max_attempts: int = 3,
        delay: float = 1.0,
        backoff_factor: float = 2.0,
        exceptions: tuple = (Exception,),
        logger=None
    ):
        """
        Decorator to retry function calls on specific exceptions
        
        Args:
            max_attempts: Maximum number of attempts
            delay: Initial delay between attempts (seconds)
            backoff_factor: Factor to multiply delay by after each attempt
            exceptions: Tuple of exceptions to catch and retry on
            logger: Logger instance for retry logging
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                func_logger = logger or get_logger(func.__module__)
                current_delay = delay
                
                for attempt in range(1, max_attempts + 1):
                    try:
                        return func(*args, **kwargs)
                    except exceptions as e:
                        if attempt == max_attempts:
                            func_logger.error(
                                f"Function {func.__name__} failed after {max_attempts} attempts"
                            )
                            raise
                        
                        func_logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                            f"Retrying in {current_delay:.1f} seconds..."
                        )
                        
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                
            return wrapper
        return decorator

# Validation utilities
def validate_ticker(ticker: str) -> str:
    """Validate stock ticker symbol"""
    if not ticker or not isinstance(ticker, str):
        raise ValidationError("Ticker must be a non-empty string")
    
    ticker = ticker.upper().strip()
    if not ticker.isalpha() or len(ticker) > 5:
        raise ValidationError(f"Invalid ticker format: {ticker}")
    
    return ticker

def validate_percentage(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Validate percentage value"""
    if not isinstance(value, (int, float)):
        raise ValidationError("Percentage must be a number")
    
    if value < min_val or value > max_val:
        raise ValidationError(f"Percentage must be between {min_val} and {max_val}")
    
    return float(value)

def validate_sentiment_score(score: float) -> float:
    """Validate sentiment score"""
    if not isinstance(score, (int, float)):
        raise ValidationError("Sentiment score must be a number")
    
    if score < -1.0 or score > 1.0:
        raise ValidationError("Sentiment score must be between -1.0 and 1.0")
    
    return float(score)

# Context manager for error handling
class ErrorContext:
    """Context manager for structured error handling"""
    
    def __init__(self, operation: str, logger=None, reraise: bool = True):
        self.operation = operation
        self.logger = logger or get_logger(__name__)
        self.reraise = reraise
        self.error = None
    
    def __enter__(self):
        self.logger.debug(f"Starting operation: {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.logger.debug(f"Operation completed successfully: {self.operation}")
            return False
        
        self.error = exc_val
        
        if isinstance(exc_val, FinNewsError):
            self.logger.error(
                f"Operation failed: {self.operation} - {exc_val.message}",
                extra={'error_details': exc_val.to_dict()}
            )
        else:
            self.logger.error(
                f"Operation failed: {self.operation} - {str(exc_val)}",
                exc_info=True
            )
        
        if not self.reraise:
            return True  # Suppress the exception
        
        return False  # Let the exception propagate

# Error reporting utilities
def create_error_report(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a standardized error report"""
    report = {
        'timestamp': None,
        'error_type': type(error).__name__,
        'message': str(error),
        'traceback': traceback.format_exc(),
        'context': context or {}
    }
    
    # Add timestamp
    from datetime import datetime
    report['timestamp'] = datetime.now().isoformat()
    
    # Add custom error details if available
    if isinstance(error, FinNewsError):
        report.update(error.to_dict())
    
    return report

def log_error_report(error: Exception, logger=None, context: Dict[str, Any] = None):
    """Log a standardized error report"""
    report = create_error_report(error, context)
    error_logger = logger or get_logger(__name__)
    
    error_logger.error(
        f"Error Report: {report['error_type']} - {report['message']}",
        extra={'error_report': report}
    ) 