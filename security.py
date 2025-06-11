"""
Security module for the RAG Chatbot.
Handles encryption, authentication, and rate limiting.
"""

import logging
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import jwt
from config import API_KEYS, SECURITY_CONFIG
from utils import log_error

logger = logging.getLogger(__name__)

class SecurityManager:
    """Manages security features including encryption, authentication, and rate limiting"""
    
    def __init__(self):
        """Initialize security manager with encryption and JWT settings"""
        try:
            # Initialize encryption
            self.fernet = Fernet(API_KEYS['encryption_key'].encode() 
                               if isinstance(API_KEYS['encryption_key'], str) 
                               else API_KEYS['encryption_key'])
            
            # JWT settings
            self.jwt_secret = API_KEYS['jwt_secret']
            self.token_expiry = SECURITY_CONFIG['token_expiry']
            
            # Rate limiting
            self.rate_limits: Dict[str, Dict[str, Any]] = {}
            self.requests_per_minute = SECURITY_CONFIG['rate_limit']['requests_per_minute']
            self.burst_limit = SECURITY_CONFIG['rate_limit']['burst_limit']
            
            logger.info("Security manager initialized successfully")
        
        except Exception as e:
            log_error(e, {'function': 'SecurityManager.__init__'})
            raise
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data using Fernet symmetric encryption"""
        try:
            if not isinstance(data, str):
                data = str(data)
            return self.fernet.encrypt(data.encode()).decode()
        except Exception as e:
            log_error(e, {'function': 'encrypt_sensitive_data'})
            raise
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt data that was encrypted using Fernet"""
        try:
            return self.fernet.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            log_error(e, {'function': 'decrypt_sensitive_data'})
            raise
    
    def generate_auth_token(self, user_id: str, additional_claims: Optional[Dict] = None) -> str:
        """Generate JWT token for user authentication"""
        try:
            claims = {
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(seconds=self.token_expiry),
                'iat': datetime.utcnow()
            }
            if additional_claims:
                claims.update(additional_claims)
            
            return jwt.encode(claims, self.jwt_secret, algorithm='HS256')
        except Exception as e:
            log_error(e, {'function': 'generate_auth_token', 'user_id': user_id})
            raise
    
    def verify_auth_token(self, token: str) -> Dict[str, Any]:
        """Verify JWT token and return claims"""
        try:
            return jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            raise
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid token: {str(e)}")
            raise
        except Exception as e:
            log_error(e, {'function': 'verify_auth_token'})
            raise
    
    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user has exceeded rate limit"""
        try:
            current_time = time.time()
            
            # Initialize rate limit tracking for new users
            if user_id not in self.rate_limits:
                self.rate_limits[user_id] = {
                    'requests': [],
                    'last_reset': current_time
                }
            
            # Clean up old requests
            user_limits = self.rate_limits[user_id]
            user_limits['requests'] = [
                req_time for req_time in user_limits['requests']
                if current_time - req_time < 60  # Keep only last minute
            ]
            
            # Check burst limit
            if len(user_limits['requests']) >= self.burst_limit:
                return False
            
            # Check requests per minute
            if len(user_limits['requests']) >= self.requests_per_minute:
                return False
            
            # Add new request
            user_limits['requests'].append(current_time)
            return True
        
        except Exception as e:
            log_error(e, {'function': 'check_rate_limit', 'user_id': user_id})
            raise
    
    def validate_password(self, password: str) -> bool:
        """Validate password against security requirements"""
        try:
            if len(password) < SECURITY_CONFIG['password_min_length']:
                return False
            
            # Check for at least one uppercase letter
            if not any(c.isupper() for c in password):
                return False
            
            # Check for at least one lowercase letter
            if not any(c.islower() for c in password):
                return False
            
            # Check for at least one digit
            if not any(c.isdigit() for c in password):
                return False
            
            # Check for at least one special character
            if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in password):
                return False
            
            return True
        
        except Exception as e:
            log_error(e, {'function': 'validate_password'})
            raise
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        try:
            # In a real application, use a proper password hashing library like bcrypt
            # This is a simplified example
            import hashlib
            salt = os.urandom(32)
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            return salt.hex() + key.hex()
        except Exception as e:
            log_error(e, {'function': 'hash_password'})
            raise
    
    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify password against hashed password"""
        try:
            # In a real application, use a proper password hashing library like bcrypt
            # This is a simplified example
            import hashlib
            salt = bytes.fromhex(hashed_password[:64])
            key = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt,
                100000
            )
            return hashed_password[64:] == key.hex()
        except Exception as e:
            log_error(e, {'function': 'verify_password'})
            raise
    
    def get_security_info(self) -> Dict[str, Any]:
        """Get security configuration information"""
        return {
            'rate_limits': {
                'requests_per_minute': self.requests_per_minute,
                'burst_limit': self.burst_limit,
                'active_users': len(self.rate_limits)
            },
            'token_expiry': self.token_expiry,
            'password_requirements': {
                'min_length': SECURITY_CONFIG['password_min_length']
            }
        } 