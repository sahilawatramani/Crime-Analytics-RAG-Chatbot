"""
Utility functions for the RAG Chatbot application.
"""

import json
import hashlib
import time
import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import jwt
from functools import wraps
from config import API_KEYS

logger = logging.getLogger(__name__)

def safe_json_dumps(obj: Any) -> str:
    """Safely convert an object to JSON string, handling non-serializable types"""
    def json_serial(obj):
        if isinstance(obj, (datetime, timedelta)):
            return str(obj)
        if isinstance(obj, set):
            return list(obj)
        raise TypeError(f"Type {type(obj)} not serializable")
    
    return json.dumps(obj, default=json_serial)

def get_query_hash(query: str) -> str:
    """Generate a unique hash for a query"""
    return hashlib.sha256(query.encode()).hexdigest()

def log_error(error: Exception, context: Optional[Dict] = None):
    """Log an error with context information"""
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'context': context or {}
    }
    logger.error(f"Error occurred: {safe_json_dumps(error_info)}")

def encrypt_data(data: str, fernet: Fernet) -> str:
    """Encrypt sensitive data using Fernet symmetric encryption"""
    try:
        if isinstance(data, str):
            return fernet.encrypt(data.encode()).decode()
        return fernet.encrypt(safe_json_dumps(data).encode()).decode()
    except Exception as e:
        log_error(e, {'function': 'encrypt_data', 'data_type': type(data).__name__})
        raise

def decrypt_data(encrypted_data: str, fernet: Fernet) -> str:
    """Decrypt data that was encrypted using Fernet"""
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except Exception as e:
        log_error(e, {'function': 'decrypt_data'})
        raise

def generate_token(user_id: str, secret: str, expiry: int = 3600) -> str:
    """Generate a JWT token for user authentication"""
    try:
        payload = {
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(seconds=expiry),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, secret, algorithm='HS256')
    except Exception as e:
        log_error(e, {'function': 'generate_token', 'user_id': user_id})
        raise

def verify_token(token: str, secret: str) -> bool:
    """Verify a JWT token"""
    try:
        jwt.decode(token, secret, algorithms=['HS256'])
        return True
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return False
    except jwt.InvalidTokenError as e:
        log_error(e, {'function': 'verify_token'})
        return False

def rate_limit(user_id: str, rate_limits: Dict) -> bool:
    """Implement rate limiting for API requests"""
    current_time = time.time()
    
    if user_id not in rate_limits:
        rate_limits[user_id] = {
            'requests': [],
            'last_reset': current_time
        }
    
    # Reset counter if more than 1 minute has passed
    if current_time - rate_limits[user_id]['last_reset'] > 60:
        rate_limits[user_id]['requests'] = []
        rate_limits[user_id]['last_reset'] = current_time
    
    # Remove requests older than 1 minute
    rate_limits[user_id]['requests'] = [
        req_time for req_time in rate_limits[user_id]['requests']
        if current_time - req_time <= 60
    ]
    
    # Check if user has exceeded rate limit (60 requests per minute)
    if len(rate_limits[user_id]['requests']) >= 60:
        return False
    
    # Add current request
    rate_limits[user_id]['requests'].append(current_time)
    return True

def format_timestamp(timestamp: Union[str, datetime]) -> str:
    """Format timestamp in a user-friendly way"""
    if isinstance(timestamp, str):
        timestamp = datetime.fromisoformat(timestamp)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")

def calculate_confidence_score(scores: list) -> float:
    """Calculate confidence score from a list of similarity scores"""
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection attacks"""
    # Remove any potentially dangerous characters
    sanitized = ''.join(c for c in text if c.isprintable())
    # Limit length
    return sanitized[:1000]

def validate_query(query: str) -> bool:
    """Validate user query for security and relevance"""
    if not query or len(query.strip()) == 0:
        return False
    
    # Check for minimum length
    if len(query.strip()) < 3:
        return False
    
    # Check for maximum length
    if len(query) > 500:
        return False
    
    # Add more validation rules as needed
    return True

def get_elapsed_time(start_time: float) -> float:
    """Calculate elapsed time in seconds"""
    return time.time() - start_time

def format_number(number: Union[int, float]) -> str:
    """Format numbers with appropriate suffixes (K, M, B)"""
    if number >= 1_000_000_000:
        return f"{number/1_000_000_000:.1f}B"
    elif number >= 1_000_000:
        return f"{number/1_000_000:.1f}M"
    elif number >= 1_000:
        return f"{number/1_000:.1f}K"
    return str(number)

def cache_key_generator(*args, **kwargs) -> str:
    """Generate a unique cache key from function arguments"""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
    return hashlib.md5("|".join(key_parts).encode()).hexdigest()

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying functions on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                    time.sleep(delay * (attempt + 1))
            return None
        return wrapper
    return decorator

def validate_api_key(api_key: str, service: str) -> bool:
    """Validate API key for external services"""
    if not api_key:
        return False
    
    # Add service-specific validation if needed
    if service == 'openai':
        return api_key.startswith('sk-') and len(api_key) > 20
    elif service == 'huggingface':
        return api_key.startswith('hf_') and len(api_key) > 20
    
    return True

def format_error_message(error: Exception) -> str:
    """Format error message for user display"""
    if isinstance(error, ValueError):
        return "Invalid input. Please check your query and try again."
    elif isinstance(error, TimeoutError):
        return "Request timed out. Please try again."
    elif isinstance(error, ConnectionError):
        return "Connection error. Please check your internet connection."
    else:
        return "An unexpected error occurred. Please try again later." 