"""
Data loading and preprocessing module for the RAG Chatbot.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Any
import logging
from pathlib import Path
from config import DATA_CONFIG, MODEL_CONFIGS
from sentence_transformers import SentenceTransformer
import torch
from utils import log_error
from model_loader import ModelManager, EmbeddingModel, validate_and_truncate_texts  # Add EmbeddingModel and validate_and_truncate_texts import
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class DataLoader:
    """Class to handle data loading and preprocessing for the crime analytics application."""
    
    def __init__(self, data_file: str = None):
        """Initialize the DataLoader with optional data file path."""
        self.data_file = data_file or DATA_CONFIG['data_file']
        self.df = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and preprocess the crime data."""
        try:
            # Load data from CSV
            self.df = pd.read_csv(self.data_file)
            
            # Basic preprocessing
            self.df = self.df.fillna(0)  # Fill missing values with 0
            
            # Convert numeric columns to appropriate types
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].astype(np.int32)
            
            # Ensure Year column is integer
            self.df['Year'] = self.df['Year'].astype(int)
            
            # Clean state names: strip, upper, and apply specific mappings for consistency
            self.df['STATE/UT'] = self.df['STATE/UT'].str.strip().str.upper()
            
            # Specific state name mappings for consistency and readability
            state_name_mapping = {
                'A & N ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
                'A&N ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
                'DELHI UT': 'DELHI',
                'D & N HAVELI': 'DADRA AND NAGAR HAVELI',
                'D&N HAVELI': 'DADRA AND NAGAR HAVELI',
                'DAMAN & DIU': 'DAMAN AND DIU',
                'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
                'PUDUCHERRY': 'PUDUCHERRY'  # Ensure consistent casing
            }
            
            # Apply mapping
            self.df['STATE/UT'] = self.df['STATE/UT'].replace(state_name_mapping, regex=True)
            
            # Ensure all states are in a consistent format (e.g., proper spacing)
            self.df['STATE/UT'] = self.df['STATE/UT'].apply(lambda x: ' '.join(x.split()))
            
            # Clean district names
            self.df['DISTRICT'] = self.df['DISTRICT'].str.strip().str.title()
            
            logger.info(f"Loaded data with shape: {self.df.shape}")
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the loaded data."""
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            return {
                'total_records': len(self.df),
                'years_range': [int(self.df['Year'].min()), int(self.df['Year'].max())],
                'states_count': len(self.df['STATE/UT'].unique()),
                'crime_types': [col for col in self.df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']],
                'data_shape': list(self.df.shape)
            }
        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}", exc_info=True)
            raise
    
    def get_filter_options(self) -> Dict[str, List]:
        """Get available filter options for the dataset."""
        try:
            if self.df is None:
                raise ValueError("No data loaded")
            
            return {
                'years': sorted(self.df['Year'].unique().tolist()),
                'states': sorted(self.df['STATE/UT'].unique().tolist()),
                'crime_types': sorted([col for col in self.df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT', 'Unnamed: 0']])
            }
        except Exception as e:
            logger.error(f"Error getting filter options: {str(e)}", exc_info=True)
            raise

def load_data() -> pd.DataFrame:
    """Load and preprocess the crime data"""
    try:
        # Load data from CSV
        df = pd.read_csv(DATA_CONFIG['data_file'])
        
        # Basic preprocessing
        df = df.fillna(0)  # Fill missing values with 0
        
        # Convert numeric columns to appropriate types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].astype(np.int32)
        
        # Ensure Year column is integer
        df['Year'] = df['Year'].astype(int)
        
        # Clean state names: strip, upper, and apply specific mappings for consistency
        df['STATE/UT'] = df['STATE/UT'].str.strip().str.upper()
        
        # Specific state name mappings for consistency and readability
        state_name_mapping = {
            'A & N ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
            'A&N ISLANDS': 'ANDAMAN AND NICOBAR ISLANDS',
            'DELHI UT': 'DELHI',
            'D & N HAVELI': 'DADRA AND NAGAR HAVELI',
            'D&N HAVELI': 'DADRA AND NAGAR HAVELI',
            'DAMAN & DIU': 'DAMAN AND DIU',
            'JAMMU & KASHMIR': 'JAMMU AND KASHMIR',
            'PUDUCHERRY': 'PUDUCHERRY' # Ensure consistent casing
        }
        
        # Apply mapping
        df['STATE/UT'] = df['STATE/UT'].replace(state_name_mapping, regex=True)
        
        # Ensure all states are in a consistent format (e.g., proper spacing)
        df['STATE/UT'] = df['STATE/UT'].apply(lambda x: ' '.join(x.split()))
        
        # Clean district names
        df['DISTRICT'] = df['DISTRICT'].str.strip().str.title()
        
        logger.info(f"Loaded data with shape: {df.shape}")
        return df
    
    except Exception as e:
        log_error(e, {'function': 'load_data'})
        raise

def prepare_documents(df: pd.DataFrame) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Prepare documents for embedding with optimized approach for speed"""
    try:
        documents = []
        metadata = []
        
        # Process in smaller chunks for better performance
        chunk_size = 1000  # Process 1000 rows at a time
        total_rows = len(df)
        
        logger.info(f"Preparing documents from {total_rows} rows in chunks of {chunk_size}")
        
        # Get all crime types for better context
        crime_types = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT']]
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk_df = df.iloc[start_idx:end_idx]
            
            logger.info(f"Processing chunk {start_idx//chunk_size + 1}/{(total_rows + chunk_size - 1)//chunk_size} (rows {start_idx}-{end_idx})")
            
            for idx, row in chunk_df.iterrows():
                try:
                    # Create a unique document ID
                    doc_id = f"doc_{idx}"
                    
                    # Extract basic information
                    year = row['Year']
                    state = row['STATE/UT']
                    district = row['DISTRICT']
                    
                    # Calculate total crimes and create crime statistics
                    crime_stats = []
                    total_crimes = 0
                    for crime_type in crime_types:
                        if pd.notna(row[crime_type]) and row[crime_type] > 0:
                            crime_stats.append(f"{crime_type}: {int(row[crime_type])}")
                            total_crimes += int(row[crime_type])
                    
                    if crime_stats:  # Only create document if there's crime data
                        # Create a more structured and informative document
                        doc = f"""Crime Report for {district}, {state} ({year})
Location: {district} district in {state} state
Year: {year}
Total Crimes: {total_crimes}
Crime Statistics:
{chr(10).join(f"- {stat}" for stat in crime_stats)}

This report provides detailed crime statistics for {district} district in {state} state during {year}. The data shows {total_crimes} total reported crimes, including {', '.join(crime_stats)}."""
                        
                        # Enhanced metadata with more information
                        meta = {
                            'doc_id': doc_id,
                            'year': int(year),
                            'state': state,
                            'district': district,
                            'total_crimes': total_crimes,
                            'crime_types': [crime_type for crime_type in crime_types if pd.notna(row[crime_type]) and row[crime_type] > 0],
                            'crime_counts': {crime_type: int(row[crime_type]) for crime_type in crime_types if pd.notna(row[crime_type]) and row[crime_type] > 0},
                            'location': f"{district}, {state}",
                            'time_period': str(year)
                        }
                        
                        documents.append(doc)
                        metadata.append(meta)
                
                except Exception as e:
                    logger.warning(f"Error processing row {start_idx}: {e}")
                    continue
        
        logger.info(f"Generated {len(documents)} documents from {total_rows} rows")
        return documents, metadata
        
    except Exception as e:
        log_error(e, {'function': 'prepare_documents'})
        raise

def get_embeddings(texts: List[str], model: EmbeddingModel) -> np.ndarray:
    """Get embeddings for a list of texts using the provided model
    
    Args:
        texts: List of texts to embed
        model: EmbeddingModel instance to use
        
    Returns:
        numpy array of embeddings
    """
    try:
        if not texts:
            return np.empty((0, 384))  # Return empty array with correct dimensions
        
        return model.encode(texts)
    except Exception as e:
        log_error(e, {'function': 'get_embeddings'})
        raise

def debug_data_loading():
    """Debug function to verify data loading and processing"""
    try:
        print("\n[DEBUG] === DATA LOADING DEBUG ===")
        
        # Step 1: Load raw data
        print("[DEBUG] Step 1: Loading raw CSV data...")
        df = load_data()
        print(f"[DEBUG] Raw data shape: {df.shape}")
        print(f"[DEBUG] Columns: {list(df.columns)}")
        
        # Step 2: Check specific data points
        print("\n[DEBUG] Step 2: Checking specific data points...")
        
        # Check for Chandigarh data
        chandigarh_data = df[df['STATE/UT'] == 'CHANDIGARH']
        print(f"[DEBUG] Chandigarh data rows: {len(chandigarh_data)}")
        if len(chandigarh_data) > 0:
            print("[DEBUG] Chandigarh sample data:")
            print(chandigarh_data[['Year', 'STATE/UT', 'DISTRICT', 'Cruelty by Husband or his Relatives', 'Dowry Deaths']].head())
        
        # Check for Arunachal Pradesh data
        arunachal_data = df[df['STATE/UT'] == 'ARUNACHAL PRADESH']
        print(f"[DEBUG] Arunachal Pradesh data rows: {len(arunachal_data)}")
        if len(arunachal_data) > 0:
            print("[DEBUG] Arunachal Pradesh sample data:")
            print(arunachal_data[['Year', 'STATE/UT', 'DISTRICT', 'Cruelty by Husband or his Relatives', 'Dowry Deaths']].head())
        
        # Check for specific years
        year_2004 = df[df['Year'] == 2004]
        year_2007 = df[df['Year'] == 2007]
        print(f"[DEBUG] 2004 data rows: {len(year_2004)}")
        print(f"[DEBUG] 2007 data rows: {len(year_2007)}")
        
        # Step 3: Test document preparation
        print("\n[DEBUG] Step 3: Testing document preparation...")
        sample_df = df.head(10)  # Test with 10 rows
        documents, metadata = prepare_documents(sample_df)
        print(f"[DEBUG] Generated {len(documents)} documents from {len(sample_df)} rows")
        
        # Show some sample documents
        print("\n[DEBUG] Sample documents:")
        for i, doc in enumerate(documents[:3]):
            print(f"[DEBUG] Doc {i+1}: {doc[:200]}...")
        
        # Step 4: Check for specific crime types
        print("\n[DEBUG] Step 4: Checking crime type columns...")
        crime_cols = [col for col in df.columns if col not in ['Year', 'STATE/UT', 'DISTRICT']]
        print(f"[DEBUG] Crime columns: {crime_cols}")
        
        # Check if the specific crime types exist
        cruelty_col = 'Cruelty by Husband or his Relatives'
        dowry_col = 'Dowry Deaths'
        
        if cruelty_col in df.columns:
            print(f"[DEBUG] ✓ Found column: {cruelty_col}")
            cruelty_data = df[df[cruelty_col] > 0]
            print(f"[DEBUG] Rows with {cruelty_col} > 0: {len(cruelty_data)}")
        else:
            print(f"[DEBUG] ✗ Missing column: {cruelty_col}")
        
        if dowry_col in df.columns:
            print(f"[DEBUG] ✓ Found column: {dowry_col}")
            dowry_data = df[df[dowry_col] > 0]
            print(f"[DEBUG] Rows with {dowry_col} > 0: {len(dowry_data)}")
        else:
            print(f"[DEBUG] ✗ Missing column: {dowry_col}")
        
        print("\n[DEBUG] === END DATA LOADING DEBUG ===\n")
        return True
        
    except Exception as e:
        print(f"[DEBUG] Error in debug_data_loading: {e}")
        return False

def test_document_preparation():
    """Test function to verify document preparation works correctly"""
    try:
        logger.info("Testing document preparation...")
        
        # Load a small sample of data
        df = load_data()
        sample_df = df.head(5)  # Just 5 rows for testing
        
        # Prepare documents
        documents, metadata = prepare_documents(sample_df)
        
        # Check results
        logger.info(f"Generated {len(documents)} documents from {len(sample_df)} rows")
        
        # Check document lengths
        for i, doc in enumerate(documents[:3]):
            word_count = len(doc.split())
            char_count = len(doc)
            logger.info(f"Document {i+1}: {word_count} words, {char_count} chars")
            logger.info(f"Preview: {doc[:100]}...")
            
            # Verify length limits
            if word_count > 400:
                logger.error(f"Document {i+1} is too long: {word_count} words")
                return False
        
        logger.info("Document preparation test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Document preparation test failed: {e}")
        return False

def clear_cache():
    """Clear all cached data to force regeneration"""
    try:
        cache_dir = Path(DATA_CONFIG['cache_dir'])
        
        # Remove cache files
        cache_files = [
            cache_dir / 'processed_data.parquet',
            cache_dir / 'embeddings.pt',
            cache_dir / 'documents.txt',
            cache_dir / 'metadata.json'
        ]
        
        for cache_file in cache_files:
            if cache_file.exists():
                cache_file.unlink()
                logger.info(f"Removed cache file: {cache_file}")
        
        logger.info("Cache cleared successfully")
        
    except Exception as e:
        log_error(e, {'function': 'clear_cache'})
        raise

def load_cached_data(force_regenerate: bool = False) -> Tuple[pd.DataFrame, List[str], List[Dict], torch.Tensor]:
    """Load data from cache if available, otherwise load and cache"""
    cache_dir = Path(DATA_CONFIG['cache_dir'])
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    df_cache = cache_dir / 'processed_data.parquet'
    embeddings_cache = cache_dir / 'embeddings.pt'
    
    try:
        # Check if cached data exists and force_regenerate is False
        if not force_regenerate and df_cache.exists() and embeddings_cache.exists():
            logger.info("Loading data from cache")
            df = pd.read_parquet(df_cache)
            embeddings = torch.load(embeddings_cache)
            with open(cache_dir / 'documents.txt', 'r', encoding='utf-8') as f:
                documents = f.read().splitlines()
            with open(cache_dir / 'metadata.json', 'r', encoding='utf-8') as f:
                metadata = pd.read_json(f).to_dict('records')
            return df, documents, metadata, embeddings
        
        # If cache doesn't exist or force_regenerate is True, load and process data
        logger.info("Cache not found or force regeneration requested, processing data")
        df = load_data()
        documents, metadata = prepare_documents(df)
        
        # Save processed data
        df.to_parquet(df_cache)
        with open(cache_dir / 'documents.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(documents))
        pd.DataFrame(metadata).to_json(cache_dir / 'metadata.json')
        
        # Generate and save embeddings using ModelManager
        model_manager = ModelManager()
        model = model_manager.load_embedding_model()
        embeddings = get_embeddings(documents, model)
        torch.save(embeddings, embeddings_cache)
        
        return df, documents, metadata, embeddings
    
    except Exception as e:
        log_error(e, {'function': 'load_cached_data'})
        raise

def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate summary statistics for the dataset"""
    try:
        summary = {
            'total_records': len(df),
            'years': {
                'min': int(df['Year'].min()),
                'max': int(df['Year'].max()),
                'unique': sorted(df['Year'].unique().tolist())
            },
            'states': {
                'count': len(df['STATE/UT'].unique()),
                'list': sorted(df['STATE/UT'].unique().tolist())
            },
            'districts': {
                'count': len(df['DISTRICT'].unique()),
                'list': sorted(df['DISTRICT'].unique().tolist())
            },
            'crime_types': {
                'count': len([col for col in df.columns 
                            if col not in ['Year', 'STATE/UT', 'DISTRICT']]),
                'list': [col for col in df.columns 
                        if col not in ['Year', 'STATE/UT', 'DISTRICT']]
            },
            'total_crimes': {
                col: int(df[col].sum()) for col in df.columns 
                if col not in ['Year', 'STATE/UT', 'DISTRICT']
            }
        }
        return summary
    
    except Exception as e:
        log_error(e, {'function': 'get_data_summary'})
        raise

def debug_embeddings():
    """Debug function to verify embeddings are generated correctly"""
    try:
        print("\n[DEBUG] === EMBEDDINGS DEBUG ===")
        
        # Load data and generate embeddings
        print("[DEBUG] Loading data and generating embeddings...")
        df, documents, metadata, embedding_matrix = load_cached_data(force_regenerate=True)
        
        print(f"[DEBUG] Data shape: {df.shape}")
        print(f"[DEBUG] Documents count: {len(documents)}")
        print(f"[DEBUG] Embedding matrix shape: {embedding_matrix.shape}")
        
        # Check if embeddings match documents
        if embedding_matrix.shape[0] != len(documents):
            print(f"[DEBUG] ⚠️ MISMATCH: {embedding_matrix.shape[0]} embeddings vs {len(documents)} documents")
            return False
        else:
            print(f"[DEBUG] ✓ Embeddings match documents: {embedding_matrix.shape[0]} == {len(documents)}")
        
        # Check for specific documents
        print("\n[DEBUG] Looking for specific documents...")
        
        # Search for Chandigarh documents
        chandigarh_docs = [i for i, doc in enumerate(documents) if 'CHANDIGARH' in doc.upper()]
        print(f"[DEBUG] Found {len(chandigarh_docs)} Chandigarh documents")
        for i in chandigarh_docs[:3]:
            print(f"[DEBUG] Chandigarh doc {i}: {documents[i][:150]}...")
        
        # Search for Arunachal Pradesh documents
        arunachal_docs = [i for i, doc in enumerate(documents) if 'ARUNACHAL' in doc.upper()]
        print(f"[DEBUG] Found {len(arunachal_docs)} Arunachal Pradesh documents")
        for i in arunachal_docs[:3]:
            print(f"[DEBUG] Arunachal doc {i}: {documents[i][:150]}...")
        
        # Search for cruelty documents
        cruelty_docs = [i for i, doc in enumerate(documents) if 'cruelty' in doc.lower()]
        print(f"[DEBUG] Found {len(cruelty_docs)} cruelty documents")
        for i in cruelty_docs[:3]:
            print(f"[DEBUG] Cruelty doc {i}: {documents[i][:150]}...")
        
        # Search for dowry documents
        dowry_docs = [i for i, doc in enumerate(documents) if 'dowry' in doc.lower()]
        print(f"[DEBUG] Found {len(dowry_docs)} dowry documents")
        for i in dowry_docs[:3]:
            print(f"[DEBUG] Dowry doc {i}: {documents[i][:150]}...")
        
        # Check for year 2004 and 2007 documents
        year_2004_docs = [i for i, doc in enumerate(documents) if '2004' in doc]
        year_2007_docs = [i for i, doc in enumerate(documents) if '2007' in doc]
        print(f"[DEBUG] Found {len(year_2004_docs)} 2004 documents")
        print(f"[DEBUG] Found {len(year_2007_docs)} 2007 documents")
        
        print("\n[DEBUG] === END EMBEDDINGS DEBUG ===\n")
        return True
        
    except Exception as e:
        print(f"[DEBUG] Error in debug_embeddings: {e}")
        return False 