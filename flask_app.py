import os
import re
import time
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd
import faiss
import spacy
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import StringIO, BytesIO
import hashlib
import requests
from PIL import Image
import torch
import logging
from cryptography.fernet import Fernet
import jwt
from functools import wraps
import concurrent.futures
import html

# Import modularized components
from config import (
    CACHE_TTL, MAX_CONTEXT_LENGTH, CONFIDENCE_THRESHOLD,
    MAX_RETRIEVED_DOCS, TYPING_DELAY, ENABLE_ANALYTICS,
    API_KEYS, MODEL_CONFIGS
)
from data_loader import (
    load_data, 
    prepare_documents, 
    load_cached_data,
    clear_cache,
    debug_data_loading,
    debug_embeddings,
    DataLoader
)
from model_loader import load_models, get_embeddings, FALLBACK_MODE
from query_processor import QueryProcessor
from analytical_processor import AnalyticalProcessor
from ui_components import (
    get_processed_results, get_source_documents_data,
    generate_visualizations_json, get_query_suggestions,
    generate_pdf, get_analytics_dashboard_data,
    generate_automatic_chart_json
)
from utils import (
    safe_json_dumps, get_query_hash, log_error, encrypt_data,
    decrypt_data, generate_token, verify_token, rate_limit
)
from cache import QueryCache, ModelCache


# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize caches
query_cache = QueryCache(max_size=1000)
model_cache = ModelCache()


from flask import Flask, request, jsonify, render_template, session

app = Flask(__name__)
app.secret_key = os.urandom(24) # Replace with a strong, permanent secret key in production

# Initialize components
data_loader = None
query_processor = None

def initialize_components():
    """Initialize data loader and query processor"""
    global data_loader, query_processor
    try:
        data_loader = DataLoader()
        query_processor = QueryProcessor(data_loader.df)
        logger.info("Components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {str(e)}", exc_info=True)
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/quick_stats', methods=['GET'])
def get_quick_stats():
    """Get quick statistics about the dataset"""
    try:
        if data_loader is None:
            initialize_components()
        
        stats = data_loader.get_data_summary()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting quick stats: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error retrieving quick stats: {str(e)}'}), 500

@app.route('/api/filter_options', methods=['GET'])
def get_filter_options():
    """Get available filter options for the dataset"""
    try:
        if data_loader is None:
            initialize_components()
        
        filter_options = data_loader.get_filter_options()
        return jsonify(filter_options)
    except Exception as e:
        logger.error(f"Error getting filter options: {str(e)}", exc_info=True)
        return jsonify({'error': f'Error retrieving filter options: {str(e)}'}), 500

@app.route('/api/sample_queries', methods=['GET'])
def get_sample_queries():
    try:
        with open('data/sample_queries.json', 'r', encoding='utf-8') as f:
            sample_queries = json.load(f)
        return jsonify(sample_queries)
    except FileNotFoundError:
        logger.error("sample_queries.json not found.")
        return jsonify({'error': 'Sample queries file not found.'}), 404
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding sample_queries.json: {e}")
        return jsonify({'error': f'Error reading sample queries: {e}'}), 500

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a natural language query about crime statistics"""
    try:
        if query_processor is None:
            initialize_components()
        
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        logger.info(f"Processing query: {query}")
        
        # Process the query
        result = query_processor.process_query(query)
        
        # Format the response
        formatted_response = query_processor.format_response(result)
        
        logger.info(f"Successfully processed query: {query}")
        return jsonify({
            'response': formatted_response,
            'raw_data': result
        })
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return jsonify({
            'error': f'Error processing query: {str(e)}',
            'details': 'Please try rephrasing your query or contact support if the issue persists.'
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_status():
    """Debug endpoint to check the status of loaded components"""
    try:
        status = {
            'data_loader_initialized': data_loader is not None,
            'query_processor_initialized': query_processor is not None
        }
        
        if data_loader is not None:
            status.update(data_loader.get_data_summary())
        
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': f'Debug error: {str(e)}'}), 500

if __name__ == '__main__':
    try:
        initialize_components()
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True) 