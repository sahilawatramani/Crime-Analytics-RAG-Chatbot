import streamlit as st
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
    debug_embeddings
)
from model_loader import load_models, get_embeddings, FALLBACK_MODE
from query_processor import EnhancedQueryProcessor
from ui_components import (
    inject_custom_css, render_chat_message,
    render_typing_indicator, render_results, render_source_documents,
    generate_visualizations, get_query_suggestions, export_conversation,
    collect_feedback, render_analytics_dashboard, render_filters,
    generate_automatic_chart
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

# --- MUST BE FIRST STREAMLIT COMMAND ---
st.set_page_config(
    page_title="🚨 Advanced Crime Analytics RAG Chatbot",
    page_icon="👮‍♀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initialize Session State ---
def initialize_session_state():
    """Initialize all session state variables with enhanced features"""
    defaults = {
        'conversation': [],
        'user_profile': {
            'preferences': {
                'theme': 'dark',
                'voice_enabled': False,
                'notifications': True
            },
            'query_history': [],
            'favorite_queries': set(),
            'custom_filters': {}
        },
        'current_query_id': None,
        'feedback_data': {},
        'theme': 'dark',
        'last_query_time': None,
        'search_suggestions': [],
        'user_id': str(uuid.uuid4()),
        'analytics_data': {
            'queries': [],
            'response_times': [],
            'satisfaction_scores': [],
            'error_rates': [],
            'feature_usage': {}
        },
        'active_integrations': set(),
        'model_versions': {},
        'error_log': [],
        'latest_query_result': None,
        'current_query': '', # Ensure current_query is always initialized
        'display_query_input': '', # For displaying in text input
        'processing_needed': False # New flag to control query processing
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Main Application ---
def main():
    # Initialize session state and inject custom CSS
    initialize_session_state()
    inject_custom_css()
    
    # Main title with modern styling
    st.markdown(f"""
    <div style="
        text-align: center; 
        padding: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin: 0 0 2rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    ">
        <div style="
            background: rgba(255,255,255,0.1);
            border-radius: 50%;
            width: 80px;
            height: 80px;
            margin: 0 auto 15px auto;
            display: flex;
            align-items: center;
            justify-content: center;
            backdrop-filter: blur(10px);
        ">
            <span style="font-size: 40px;">👮‍♀️</span>
        </div>
        <h1 style="
            font-size: 2.5rem; 
            font-weight: 700; 
            color: white; 
            margin-bottom: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.3);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            Crime Analytics RAG Chatbot: India (2001-2014)
        </h1>
        <p style="
            font-size: 1.1rem; 
            color: rgba(255,255,255,0.9); 
            margin-top: 10px;
            font-weight: 300;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            AI-Powered Analysis of Crimes Against Women in India
        </p>
        <div style="
            margin-top: 15px;
            padding: 8px 20px;
            background: rgba(255,255,255,0.2);
            border-radius: 25px;
            display: inline-block;
            backdrop-filter: blur(10px);
        ">
            <span style="color: white; font-weight: 500; font-size: 0.9rem;">
                🔍 Comprehensive Data Analysis | 📊 Interactive Visualizations | ✨ Humanized Insights
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and models with caching using session state
    if 'df' not in st.session_state or 'models' not in st.session_state or 'documents' not in st.session_state or 'metadata' not in st.session_state or 'embedding_matrix' not in st.session_state:
        # Create a progress container
        progress_container = st.empty()
        progress_bar = progress_container.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load and cache data (20%) - Force regeneration for new document format
            status_text.text("📊 Loading and caching data...")
            df, documents, metadata, embedding_matrix = load_cached_data(force_regenerate=True)  # Force regeneration
            st.session_state.df = df
            st.session_state.documents = documents
            st.session_state.metadata = metadata
            st.session_state.embedding_matrix = embedding_matrix
            progress_bar.progress(20)
            
            # Step 2: Load models (60%)
            status_text.text("🤖 Loading AI models...")
            st.session_state.models = load_models()
            progress_bar.progress(80)
            
            # Step 3: Initialize query processor (20%)
            status_text.text("🔄 Initializing query processor...")
            embedding_model = st.session_state.models['embedding']
            nlp_model = st.session_state.models['nlp']
            qa_model = st.session_state.models['qa']
            generator_model = st.session_state.models['generator']
            query_processor = EnhancedQueryProcessor(
                df=st.session_state.df,
                embedding_model=embedding_model,
                nlp_model=nlp_model,
                qa_model=qa_model,
                generator_model=generator_model,
                documents=st.session_state.documents,
                metadata=st.session_state.metadata,
                embedding_matrix=embedding_matrix,
                csv_file_path="data/crime_data.csv"  # Add CSV file path
            )
            st.session_state.processor = query_processor
            progress_bar.progress(100)
            status_text.text("✅ Ready!")
            
            # Clear progress indicators after a short delay
            time.sleep(1)
            progress_container.empty()
            status_text.empty()
            
        except Exception as e:
            progress_container.empty()
            status_text.empty()
            st.error('Error during initialization: ' + str(e))
            raise
    
    # Use cached data and models
    df = st.session_state.df
    models = st.session_state.models
    documents = st.session_state.documents
    metadata = st.session_state.metadata
    embedding_matrix = st.session_state.embedding_matrix
    processor = st.session_state.processor
    
    # Sidebar with enhanced features
    with st.sidebar:
        # Quick filters
        # Filters for Year, State/UT, Crime Types
        render_filters(df, lambda years, states, crimes: update_filters(years, states, crimes, processor))
        
        st.markdown("---")
        
        # Sample Queries for easy testing
        st.markdown("### 💬 Sample Queries")
        queries_per_page = 5
        
        # Load sample queries from JSON file
        try:
            with open('data/sample_queries.json', 'r', encoding='utf-8') as f:
                sample_queries = json.load(f)
        except FileNotFoundError:
            st.error("Error: sample_queries.json not found in the 'data' directory.")
            sample_queries = {}
        except json.JSONDecodeError:
            st.error("Error: Could not decode sample_queries.json. Please check its format.")
            sample_queries = {}

        all_queries = []
        for category, queries in sample_queries.items():
            for query_text in queries:
                all_queries.append({"category": category, "query": query_text})
        
        total_pages = (len(all_queries) + queries_per_page - 1) // queries_per_page
        
        if 'query_page' not in st.session_state:
            st.session_state.query_page = 0
        
        with st.expander("Browse Sample Queries", expanded=True):
            # Pagination controls at the top of the expander
            st.markdown(f"<div style='text-align: center; font-size: 1.0em; font-weight: 500;'>Page {st.session_state.query_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            
            # Simple pagination buttons without columns at the top
            col_top_prev, col_top_next = st.columns(2)
            with col_top_prev:
                if st.button("⬅️ Previous", disabled=st.session_state.query_page == 0, key="prev_page_button_top", use_container_width=True):
                    st.session_state.query_page = max(0, st.session_state.query_page - 1)
                    # When navigating, we want to pre-fill the display_query_input with the first query of the new page
                    if len(all_queries) > st.session_state.query_page * queries_per_page:
                        st.session_state.current_query = all_queries[st.session_state.query_page * queries_per_page]["query"]
                        st.session_state.display_query_input = all_queries[st.session_state.query_page * queries_per_page]["query"]
                        st.session_state.processing_needed = True # Signal that processing is needed for the pre-filled query
                    st.rerun()
            with col_top_next:
                if st.button("Next ➡️", disabled=st.session_state.query_page >= total_pages - 1, key="next_page_button_top", use_container_width=True):
                    st.session_state.query_page = min(total_pages - 1, st.session_state.query_page + 1)
                    # When navigating, we want to pre-fill the display_query_input with the first query of the new page
                    if len(all_queries) > st.session_state.query_page * queries_per_page:
                        st.session_state.current_query = all_queries[st.session_state.query_page * queries_per_page]["query"]
                        st.session_state.display_query_input = all_queries[st.session_state.query_page * queries_per_page]["query"]
                        st.session_state.processing_needed = True # Signal that processing is needed for the pre-filled query
                    st.rerun()
            
            start_idx = st.session_state.query_page * queries_per_page
            end_idx = min(start_idx + queries_per_page, len(all_queries))
            
            # Show current category (moved to top of expander for better visibility)
            if all_queries:
                current_category = all_queries[start_idx]["category"]
                st.markdown(f"**📂 Current Category: {current_category}**")
            
            # Display current page queries
            for i, item in enumerate(all_queries[start_idx:end_idx]):
                query = item["query"]
                
                # Use a simple button layout without nested columns
                if st.button(f"📝 {query[:50]}...", key=f"sample_query_btn_{start_idx + i}_{hash(query)}", use_container_width=True):
                    st.session_state.current_query = query # Directly set current_query for processing
                    st.session_state.display_query_input = query # Update display input
                    st.session_state.processing_needed = True # Signal that processing is needed
                    st.rerun()
            
            # Pagination controls at the bottom of the expander
            st.markdown(f"<div style='text-align: center; font-size: 1.0em; font-weight: 500;'>Page {st.session_state.query_page + 1} of {total_pages}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='text-align: center; font-size: 0.9em; opacity: 0.7;'>Showing {start_idx + 1}-{end_idx} of {len(all_queries)} queries</div>", unsafe_allow_html=True)
            
            # Simple pagination buttons without columns
            if st.button("⬅️ Previous", disabled=st.session_state.query_page == 0, key="prev_page_button_bottom", use_container_width=True):
                st.session_state.query_page = max(0, st.session_state.query_page - 1)
                # When navigating, we want to pre-fill the display_query_input with the first query of the new page
                if len(all_queries) > st.session_state.query_page * queries_per_page:
                    st.session_state.current_query = all_queries[st.session_state.query_page * queries_per_page]["query"]
                    st.session_state.display_query_input = all_queries[st.session_state.query_page * queries_per_page]["query"]
                    st.session_state.processing_needed = True # Signal that processing is needed for the pre-filled query
                st.rerun()
            if st.button("Next ➡️", disabled=st.session_state.query_page >= total_pages - 1, key="next_page_button_bottom", use_container_width=True):
                st.session_state.query_page = min(total_pages - 1, st.session_state.query_page + 1)
                # When navigating, we want to pre-fill the display_query_input with the first query of the new page
                if len(all_queries) > st.session_state.query_page * queries_per_page:
                    st.session_state.current_query = all_queries[st.session_state.query_page * queries_per_page]["query"]
                    st.session_state.display_query_input = all_queries[st.session_state.query_page * queries_per_page]["query"]
                    st.session_state.processing_needed = True # Signal that processing is needed for the pre-filled query
                st.rerun()
        
        # Quick category jump within an expander
        with st.expander("Jump to Category", expanded=False):
            for i, category in enumerate(list(sample_queries.keys())):
                # Use a simple button layout without nested columns
                if st.button(f"📁 {category}", key=f"cat_jump_btn_{i}", use_container_width=True):
                    total_queries_before_category = 0
                    for j, (cat, queries) in enumerate(sample_queries.items()):
                        if cat == category:
                            st.session_state.query_page = total_queries_before_category // queries_per_page
                            # When jumping to category, pre-fill display_query_input with the first query of that category
                            if queries:
                                st.session_state.current_query = queries[0] # Directly set current_query
                                st.session_state.display_query_input = queries[0]
                                st.session_state.processing_needed = True # Signal that processing is needed
                            st.rerun()
                            break
                        total_queries_before_category += len(queries)
        
        st.markdown("---")
        
        # Export conversation
        export_conversation()
        
        st.markdown("---")
        

    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Query input with autocomplete
        with st.form("chat_form", clear_on_submit=True):
            query_input = st.text_input(
                "💬 Ask me anything about crime statistics:",
                value=st.session_state.display_query_input, # Use display_query_input here
                placeholder="e.g., 'Show me rape cases in Delhi from 2010 to 2012'",
                key="main_query_input_form"
            )
            submit_button = st.form_submit_button("Send 🚀")
        
        # Update session state with current input (if it's a new input via form submission)
        if submit_button and query_input and query_input.strip():
            st.session_state.current_query = query_input # Set current_query for processing
            st.session_state.display_query_input = query_input # Keep display updated
            st.session_state.processing_needed = True # Signal that processing is needed
            st.rerun() # Trigger rerun to process the query immediately
        
        # The main query processing logic will now be triggered only if current_query is set
        # by a form submission, filter, or sample query button.
        
    with col2:
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        st.metric("Total Records", f"{len(df):,}")
        st.metric("Years Covered", f"{df['Year'].min()}-{df['Year'].max()}")
        st.metric("States/UTs", len(df['STATE/UT'].unique()))
        st.metric("Crime Types", len([col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']]))
    
    # Process query
    if st.session_state.processing_needed: # Use processing_needed flag to trigger processing
        try:
            # Process the query
            start_time = time.time()
            result = processor.process_with_context(st.session_state.current_query) # Use current_query for processing
            processing_time = time.time() - start_time
            
            # Store processing_time in the result for persistent display
            result['processing_time'] = processing_time

            # Store the full result for visualizations
            st.session_state.latest_query_result = result
            
            # Store in conversation
            conversation_entry = {
                'user': st.session_state.current_query,
                'bot': result['answer'],
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'confidence': result['confidence'],
                'sources': result['sources'],
                'query_id': result['query_id']
            }
            
            # Defensive check: Only append user message if it's new or not a consecutive duplicate
            # If the last entry is an identical user query without a bot response, update it instead of appending.
            if st.session_state.conversation and \
               st.session_state.conversation[-1].get('user') == conversation_entry['user'] and \
               st.session_state.conversation[-1].get('bot') is None:
                # Update the existing entry with the bot response
                st.session_state.conversation[-1].update(conversation_entry)
            else:
                # Append the new entry (which will contain both user and bot info for this exchange)
                st.session_state.conversation.append(conversation_entry)
            
            # Clear current_query to prevent re-running on subsequent reruns,
            # but keep display_query_input to show last query in input box
            st.session_state.current_query = ""
            
            # After successful processing, set processing_needed to False
            st.session_state.processing_needed = False
            
            # Analytics tracking
            if ENABLE_ANALYTICS:
                st.session_state.analytics_data['queries'].append(st.session_state.current_query)
                st.session_state.analytics_data['response_times'].append(processing_time)
            
        finally:
            pass

    # Display the latest bot response, automatic visualization, feedback, and performance metrics
    # This block ensures that the output persists across reruns, even if current_query is cleared.
    print(f"DEBUG: Before latest_query_result display block. latest_query_result is None: {st.session_state.latest_query_result is None}")
    if st.session_state.latest_query_result:
        result = st.session_state.latest_query_result
        query_input = st.session_state.display_query_input # Use the display_query_input for visual representation
        
        # Display user query in a clean, simple format (if it was the last one processed)
        st.markdown("---") # Add a separator
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: none;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 18px;">👤</span>
                <span style="font-size: 1.1em; font-weight: 500;">{html.escape(query_input)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display bot response with full styling
        # Note: We need the full conversation_entry from the stored conversation
        # to ensure the timestamp and other details are correct for the last message.
        # Find the last bot message for this query_id if available, or just the last overall bot message.
        last_bot_message_entry = None
        if st.session_state.conversation:
            # Try to find the specific bot response for the latest_query_result
            if 'query_id' in result:
                for entry in reversed(st.session_state.conversation):
                    if entry.get('query_id') == result['query_id'] and 'bot' in entry:
                        last_bot_message_entry = entry
                        break
            if not last_bot_message_entry and 'bot' in st.session_state.conversation[-1]: # Fallback to last message if not found by query_id
                 last_bot_message_entry = st.session_state.conversation[-1]
        
        if last_bot_message_entry:
            render_chat_message(last_bot_message_entry, is_user=False)
        else:
            st.markdown(result['answer']) # Fallback if specific entry not found

        # Generate automatic visualization based on query results
        results_data = None
        analysis_type = 'general'
        
        if result.get('results'):
            results_data = result['results']
            analysis_type = result.get('analysis_type', 'general')
        elif result.get('raw_data') and result['raw_data'].get('results'):
            results_data = result['raw_data']['results']
            analysis_type = result['raw_data'].get('analysis_type', 'general')
        elif result.get('raw_data'):
            results_data = result['raw_data']
            analysis_type = result.get('analysis_type', 'general')
        
        if results_data:
            st.markdown("### 📊 Automatic Analysis Visualization")
            auto_chart = generate_automatic_chart(results_data, df, analysis_type)
            if auto_chart:
                st.plotly_chart(auto_chart, use_container_width=True, config={'displayModeBar': True})
            else:
                st.info("📈 No suitable visualization available for this query type.")
        else:
            st.info("📈 No data available for visualization.")

        # Collect feedback
        collect_feedback(result['query_id'], result['answer'])
        
        # Show Analytics checkbox moved to main content area
        st.markdown("---") # Add a separator before analytics
        show_analytics_checkbox = st.checkbox("📊 Show Analytics Dashboard", value=st.session_state.get('show_analytics', False))
        st.session_state.show_analytics = show_analytics_checkbox
        print(f"DEBUG: Analytics checkbox toggled. show_analytics: {st.session_state.show_analytics}")
        
        if st.session_state.show_analytics:
            render_analytics_dashboard()
        
        # Performance metrics - Improved styling
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 15px 20px;
            border-radius: 10px;
            margin: 15px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border: none;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 16px;">⚡</span>
                    <span style="font-weight: 500;">{result.get('processing_time', 0):.2f}s</span>
                </div>
                <div style="display: flex; align-items: center; gap: 5px;">
                    <span style="font-size: 16px;">🎯</span>
                    <span style="font-weight: 500;">{result['confidence']:.1%}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Conversation history
    print(f"DEBUG: Before conversation history display. Conversation length: {len(st.session_state.conversation)}")
    if st.session_state.conversation and len(st.session_state.conversation) > 1:
        st.markdown("---")
        st.markdown("### 💬 Conversation History")
        st.info(f"📊 **Total Conversations:** {len(st.session_state.conversation)} | 📅 **Session Started:** {st.session_state.conversation[0]['timestamp'] if st.session_state.conversation else 'N/A'}")
        # Display conversation history
        for message in reversed(st.session_state.conversation):
            if 'user' in message:
                render_chat_message(message, is_user=True)
            elif 'bot' in message:
                render_chat_message(message, is_user=False)


# Filter update function
def update_filters(selected_years, selected_states, selected_crimes, processor):
    st.session_state.filtered_years = selected_years
    st.session_state.filtered_states = selected_states
    st.session_state.filtered_crimes = selected_crimes

    # Construct a natural language query based on applied filters
    query_parts = []
    if selected_crimes:
        query_parts.append(f"crime types {', '.join(selected_crimes)}")
    if selected_states:
        query_parts.append(f"states {', '.join(selected_states)}")
    if selected_years:
        query_parts.append(f"years {', '.join(map(str, selected_years))}")

    generated_query = ""
    if query_parts:
        generated_query = f"Show me statistics for {', and '.join(query_parts)}."
    else:
        generated_query = "Show me overall crime statistics."
    
    # Set the generated query to the main input field via session state
    st.session_state.current_query = generated_query
    st.session_state.display_query_input = generated_query # Update display input
    st.session_state.processing_needed = True # Signal that processing is needed

    st.success("Filters applied and query generated!")
    st.rerun()


if __name__ == "__main__":
    main()