"""
Model loading and management module for the RAG Chatbot.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    AutoModel,
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertForQuestionAnswering
)
import spacy
import subprocess
import sys
from config import MODEL_CONFIGS, API_KEYS
from utils import log_error, retry_on_failure

logger = logging.getLogger(__name__)

FALLBACK_MODE = False
try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    
    # Check PyTorch version - if 2.6.0 or higher, force fallback mode
    import torch
    torch_version = torch.__version__
    if torch_version.startswith('2.6') or torch_version.startswith('2.7'):
        FALLBACK_MODE = True
        logging.warning(f"PyTorch {torch_version} detected - forcing TF-IDF fallback mode to avoid meta tensor issues")
    else:
        # Test if sentence-transformers works
        test_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=None)
        del test_model  # Clean up
        logging.info("Sentence-transformers test successful")
        
except Exception as e:
    FALLBACK_MODE = True
    logging.warning(f"Neural embeddings unavailable: {e}")

def safe_to_device(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Safely move a model to the specified device"""
    try:
        if hasattr(model, 'to_empty'):
            # Handle meta tensor case
            return model.to_empty(device=device)
        else:
            # Regular device placement
            return model.to(device)
    except Exception as e:
        log_error(e, {'function': 'safe_to_device', 'device': device})
        raise

class EmbeddingModel:
    """Wrapper class for sentence transformer model with proper device handling"""
    
    def __init__(self, model_name: str, device: str = 'cpu'):
        """Initialize the embedding model with proper PyTorch 2.6.0 meta tensor handling"""
        try:
            if FALLBACK_MODE:
                from ultra_simple_embedding import UltraSimpleEmbeddingModel
                logging.warning("Using ultra-simple TF-IDF embedding model (fallback mode)")
                logging.info("This provides good keyword-based search but less semantic accuracy than neural embeddings")
                self.ultra_simple_model = UltraSimpleEmbeddingModel(model_name, device)
                self.use_sentence_transformer = False
                logging.info("✅ TF-IDF embedding model initialized successfully")
            else:
                from sentence_transformers import SentenceTransformer
                
                # Create SentenceTransformer without device specification to avoid meta tensor issues
                logging.info(f"Creating SentenceTransformer without device specification")
                self.sentence_transformer = SentenceTransformer(model_name, device=None)
                
                # Manually move to device using to_empty() for PyTorch 2.6.0 compatibility
                logging.info(f"Moving SentenceTransformer to device {device} using to_empty()")
                if hasattr(self.sentence_transformer, 'to_empty'):
                    self.sentence_transformer = self.sentence_transformer.to_empty(device=device)
                else:
                    # Fallback for older versions
                    self.sentence_transformer = self.sentence_transformer.to(device)
                
                self.use_sentence_transformer = True
                logging.info("✅ SentenceTransformer initialized successfully with PyTorch 2.6.0 compatibility")
        except Exception as e:
            logging.error(f"Failed to initialize embedding model: {e}")
            # Force fallback mode if neural embeddings fail
            try:
                from ultra_simple_embedding import UltraSimpleEmbeddingModel
                logging.warning("Falling back to TF-IDF embedding model due to neural embedding failure")
                self.ultra_simple_model = UltraSimpleEmbeddingModel(model_name, device)
                self.use_sentence_transformer = False
                logging.info("✅ TF-IDF fallback model initialized successfully")
            except Exception as fallback_error:
                logging.error(f"Both neural and TF-IDF embedding models failed: {fallback_error}")
                raise
    
    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Encode texts to embeddings using ultra-simple TF-IDF approach"""
        try:
            if hasattr(self, 'sentence_transformer') and self.use_sentence_transformer:
                return self.sentence_transformer.encode(texts, convert_to_tensor=True)
            elif hasattr(self, 'ultra_simple_model'):
                arr = self.ultra_simple_model.encode(texts)
                return torch.from_numpy(arr).float()
            else:
                raise Exception("No embedding model available!")
            
        except Exception as e:
            logger.error(f"Error in TF-IDF encoding: {e}")
            raise

class ModelCache:
    """Cache for storing and managing loaded models"""
    
    def __init__(self):
        """Initialize an empty model cache"""
        self._cache = {}
        logger.info("Initialized model cache")
    
    def __getitem__(self, key: str) -> Any:
        """Get a model from cache"""
        if key not in self._cache:
            raise KeyError(f"Model {key} not found in cache")
        return self._cache[key]
    
    def __setitem__(self, key: str, model: Any) -> None:
        """Add a model to cache"""
        self._cache[key] = model
        logger.info(f"Added model {key} to cache")
    
    def __contains__(self, key: str) -> bool:
        """Check if a model is in cache"""
        return key in self._cache
    
    def clear(self) -> None:
        """Clear the cache"""
        self._cache.clear()
        logger.info("Cleared model cache")
    
    def remove(self, key: str) -> None:
        """Remove a model from cache"""
        if key in self._cache:
            del self._cache[key]
            logger.info(f"Removed model {key} from cache")

class ModelManager:
    """Manager class for loading and caching models"""
    
    def __init__(self):
        """Initialize model manager with cache"""
        self.cache = ModelCache()
        self.device = MODEL_CONFIGS['embedding']['device']
        self.models = {}  # Initialize models dictionary
        logger.info(f"Initialized ModelManager with device: {self.device}")
    
    def load_embedding_model(self) -> EmbeddingModel:
        """Load embedding model with proper device handling"""
        try:
            model_name = MODEL_CONFIGS['embedding']['name']
            
            # Clear any existing cache to force fresh loading
            cache_key = f"embedding_{model_name}"
            if cache_key in self.cache:
                logger.info(f"Clearing cached embedding model to force fresh loading")
                del self.cache[cache_key]
            
            # Always load new model (don't cache to ensure fresh instance)
            logger.info(f"Loading fresh embedding model: {model_name}")
            model = EmbeddingModel(
                model_name=model_name,
                device=self.device
            )
            
            # Verify it's using sentence-transformers
            if hasattr(model, 'use_sentence_transformer'):
                if model.use_sentence_transformer:
                    logger.info("✅ Embedding model is using sentence-transformers")
                else:
                    logger.info("✅ Embedding model is using a non-sentence-transformers approach (e.g., TF-IDF)")
            else:
                logger.info("✅ Embedding model loaded (no use_sentence_transformer attribute)")
            
            return model
            
        except Exception as e:
            log_error(e, {'function': 'load_embedding_model'})
            raise
    
    @retry_on_failure(max_retries=3)
    def load_qa_model(self) -> Tuple[AutoModelForQuestionAnswering, AutoTokenizer]:
        """Load the question answering model"""
        try:
            if 'qa' not in self.models:
                logger.info("Loading QA model...")
                
                # Load tokenizer using AutoTokenizer to match the model
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_CONFIGS['qa']['name']
                )
                
                # Load model with specific configuration
                model = AutoModelForQuestionAnswering.from_pretrained(
                    MODEL_CONFIGS['qa']['name'],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=None
                )
                
                # Move model to device safely
                logger.info(f"Moving QA model to {self.device}...")
                model = safe_to_device(model, self.device)
                
                self.models['qa'] = (model, tokenizer)
                logger.info("QA model loaded successfully")
            return self.models['qa']
        except Exception as e:
            log_error(e, {'function': 'load_qa_model'})
            raise
    
    @retry_on_failure(max_retries=3)
    def get_qa_pipeline(self) -> pipeline:
        """Get the question answering pipeline"""
        try:
            if 'qa_pipeline' not in self.models:
                logger.info("Creating QA pipeline...")
                model, tokenizer = self.load_qa_model()
                
                # Create pipeline with loaded model and tokenizer
                qa_pipeline = pipeline(
                    'question-answering',
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device
                )
                
                self.models['qa_pipeline'] = qa_pipeline
                logger.info("QA pipeline created successfully")
            return self.models['qa_pipeline']
        except Exception as e:
            log_error(e, {'function': 'get_qa_pipeline'})
            raise
    
    @retry_on_failure(max_retries=3)
    def load_generator_model(self) -> pipeline:
        """Load the text generation model"""
        try:
            if 'generator' not in self.models:
                logger.info("Loading generator model...")
                
                # Load tokenizer with proper configuration
                tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS['generator']['name'])
                
                # Ensure tokenizer has proper EOS and pad tokens
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with specific configuration to avoid meta tensor issues
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_CONFIGS['generator']['name'],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=None  # Don't use device_map to handle device placement manually
                )
                
                # Move model to device safely using to_empty
                logger.info(f"Moving generator model to {self.device}...")
                model = safe_to_device(model, self.device)
                
                # Create pipeline with loaded model and tokenizer
                generator = pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=self.device,
                    max_length=MODEL_CONFIGS['generator']['max_length'],
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                self.models['generator'] = generator
                logger.info("Generator model loaded successfully")
            return self.models['generator']
        except Exception as e:
            log_error(e, {'function': 'load_generator_model'})
            raise
    
    @retry_on_failure(max_retries=3)
    def load_nlp_model(self) -> spacy.language.Language:
        """Load the spaCy NLP model"""
        try:
            if 'nlp' not in self.models:
                logger.info("Loading NLP model...")
                
                # Check if spaCy model is installed, install if not
                try:
                    nlp = spacy.load(MODEL_CONFIGS['nlp']['name'])
                except OSError:
                    logger.info(f"Installing spaCy model: {MODEL_CONFIGS['nlp']['name']}")
                    subprocess.check_call([
                        sys.executable, 
                        "-m", 
                        "spacy", 
                        "download", 
                        MODEL_CONFIGS['nlp']['name']
                    ])
                    nlp = spacy.load(MODEL_CONFIGS['nlp']['name'])
                
                # Disable unnecessary pipeline components for better performance
                nlp.disable_pipe("ner")  # Disable named entity recognition
                nlp.disable_pipe("parser")  # Disable dependency parsing
                
                self.models['nlp'] = nlp
                logger.info("NLP model loaded successfully")
            return self.models['nlp']
        except Exception as e:
            log_error(e, {'function': 'load_nlp_model'})
            raise

def load_models() -> Dict[str, Any]:
    """Load all required models"""
    try:
        manager = ModelManager()
        return {
            'embedding': manager.load_embedding_model(),
            'qa': manager.get_qa_pipeline(),
            'generator': manager.load_generator_model(),
            'nlp': manager.load_nlp_model()
        }
    except Exception as e:
        log_error(e, {'function': 'load_models'})
        raise

def clear_model_cache():
    """Clear the model cache"""
    try:
        manager = ModelManager()
        manager.models.clear()
        logger.info("Model cache cleared")
    except Exception as e:
        log_error(e, {'function': 'clear_model_cache'})
        raise

def get_model_info() -> Dict[str, Any]:
    """Get information about loaded models"""
    try:
        manager = ModelManager()
        return {
            'device': str(manager.device),
            'loaded_models': list(manager.models.keys()),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    except Exception as e:
        log_error(e, {'function': 'get_model_info'})
        raise

def get_embeddings(model: EmbeddingModel, texts: List[str]) -> torch.Tensor:
    """Generate embeddings for a list of texts"""
    try:
        return model.encode(
            texts,
        )
    except Exception as e:
        log_error(e, {'function': 'get_embeddings'})
        raise 