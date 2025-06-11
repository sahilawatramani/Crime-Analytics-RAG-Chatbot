"""
Configuration settings for the RAG Chatbot application.
"""

import os
import torch
from cryptography.fernet import Fernet
from typing import Dict, Any

# Cache settings
CACHE_TTL = 3600  # Time-to-live for cached items in seconds
MAX_CONTEXT_LENGTH = 512  # Maximum context length for models
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence score for answers
MAX_RETRIEVED_DOCS = 5  # Maximum number of documents to retrieve
TYPING_DELAY = 0.5  # Delay for typing indicator in seconds

# API Keys and Secrets
API_KEYS = {
    'encryption_key': os.getenv('ENCRYPTION_KEY', Fernet.generate_key().decode()),
    'jwt_secret': os.getenv('JWT_SECRET', 'your-secret-key-here'),
    'openai_api_key': os.getenv('OPENAI_API_KEY', ''),
    'huggingface_api_key': os.getenv('HUGGINGFACE_API_KEY', '')
}

# Supported languages for translation
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi',
    'bn': 'Bengali',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi'
}

# Model configurations
MODEL_CONFIGS = {
    'cache_dir': 'models/cache',  # Directory for caching models
    'embedding': {
        'name': 'sentence-transformers/all-MiniLM-L6-v2',
        'device': 'cpu',  # Use CPU to avoid device issues
        'max_length': 512
    },
    'qa': {
        'name': 'deepset/roberta-base-squad2',
        'device': 'cpu',  # Use CPU to avoid device issues
        'max_length': 512
    },
    'generator': {
        'name': 'gpt2',
        'device': 'cpu',  # Use CPU to avoid device issues
        'max_length': 1024
    },
    'nlp': {
        'name': 'en_core_web_sm',
        'device': 'cpu'  # Use CPU to avoid device issues
    }
}

# Security settings
SECURITY_CONFIG = {
    'rate_limit': {
        'requests_per_minute': 60,
        'burst_limit': 10
    },
    'token_expiry': 3600,  # JWT token expiry in seconds
    'max_login_attempts': 5,
    'password_min_length': 8
}

# Integration settings
INTEGRATION_CONFIG = {
    'slack': {
        'webhook_url': os.getenv('SLACK_WEBHOOK_URL', ''),
        'channel': '#crime-analytics'
    },
    'email': {
        'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(os.getenv('SMTP_PORT', '587')),
        'sender_email': os.getenv('SENDER_EMAIL', ''),
        'sender_password': os.getenv('SENDER_PASSWORD', '')
    }
}

# UI settings
UI_CONFIG = {
    'theme': {
        'primary_color': '#d62828',
        'secondary_color': '#4a9eff',
        'background_color': '#ffffff',
        'text_color': '#333333',
        'font_family': 'Inter, sans-serif'
    },
    'chat': {
        'max_messages': 100,
        'message_display_limit': 50,
        'typing_speed': 0.05
    }
}

# Data settings
DATA_CONFIG = {
    'data_file': 'data/crime_data.csv',
    'cache_dir': 'cache',
    'log_dir': 'logs',
    'export_dir': 'exports'
}

# Analytics settings
ENABLE_ANALYTICS = True
ANALYTICS_CONFIG = {
    'track_queries': True,
    'track_response_times': True,
    'track_satisfaction': True,
    'track_errors': True,
    'track_feature_usage': True,
    'retention_days': 30
}

# Feature flags
FEATURE_FLAGS = {
    'enable_voice': True,
    'enable_translation': True,
    'enable_analytics': True,
    'enable_export': True,
    'enable_feedback': True,
    'enable_visualizations': True
}

# Create necessary directories
for directory in [DATA_CONFIG['cache_dir'], DATA_CONFIG['log_dir'], DATA_CONFIG['export_dir']]:
    os.makedirs(directory, exist_ok=True)

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(DATA_CONFIG['log_dir'], 'app.log'),
            'formatter': 'standard',
            'level': 'INFO'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        }
    },
    'loggers': {
        '': {
            'handlers': ['file', 'console'],
            'level': 'INFO',
            'propagate': True
        }
    }
} 