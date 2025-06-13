"""
Model loading and management module for the RAG Chatbot.
Fixed version with proper token length handling and input validation.
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
import numpy as np

logger = logging.getLogger(__name__)

FALLBACK_MODE = False
try:
    import sentence_transformers
    from sentence_transformers import SentenceTransformer
    
    # Removed PyTorch version check - assume compatible setup for SentenceTransformer
    # Test if sentence-transformers works
    test_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=None)
    del test_model  # Clean up
    logging.info("Sentence-transformers test successful")
        
except Exception as e:
    FALLBACK_MODE = True
    logging.warning(f"Neural embeddings unavailable, falling back: {e}")

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

def validate_and_truncate_texts(texts: List[str], model_name: str = None, max_length: int = 512) -> List[str]:
    """
    Validate and truncate texts to ensure they don't exceed model's max length.
    
    Args:
        texts: List of input texts
        model_name: Name of the model (for tokenizer compatibility)
        max_length: Maximum allowed token length
    
    Returns:
        List of validated and truncated texts
    """
    try:
        # Filter out empty texts
        filtered_texts = [text.strip() for text in texts if text and text.strip()]
        
        if not filtered_texts:
            logger.warning("All input texts are empty after filtering")
            return [""]  # Return minimal valid input
        
        # If model_name is provided, use its tokenizer for accurate length checking
        if model_name:
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                validated_texts = []
                for text in filtered_texts:
                    # Tokenize and check length
                    tokens = tokenizer.tokenize(text)
                    if len(tokens) > max_length:
                        logger.warning(f"Text length {len(tokens)} exceeds max {max_length}, truncating")
                        # Truncate using tokenizer
                        encoded = tokenizer(
                            text, 
                            truncation=True, 
                            max_length=max_length,
                            return_tensors=None
                        )
                        truncated_text = tokenizer.decode(encoded['input_ids'], skip_special_tokens=True)
                        validated_texts.append(truncated_text)
                    else:
                        validated_texts.append(text)
                
                return validated_texts
                
            except Exception as tokenizer_error:
                logger.warning(f"Could not load tokenizer for {model_name}: {tokenizer_error}")
                # Fall back to character-based truncation
                pass
        
        # Fallback: Character-based truncation (rough estimate: 4 chars per token)
        char_limit = max_length * 4
        validated_texts = []
        for text in filtered_texts:
            if len(text) > char_limit:
                logger.warning(f"Text character length {len(text)} exceeds estimated limit {char_limit}, truncating")
                validated_texts.append(text[:char_limit])
            else:
                validated_texts.append(text)
        
        return validated_texts
        
    except Exception as e:
        logger.error(f"Error in validate_and_truncate_texts: {e}")
        # Return safe fallback
        return [text[:2000] for text in texts if text and text.strip()]

class EmbeddingModel:
    """Handles text embedding using a pre-trained model"""
    
    def __init__(self, model_name: str, device: str = 'cpu', max_length: int = 256):
        """Initialize the embedding model
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cpu' or 'cuda')
            max_length: Maximum sequence length for tokenization
        """
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(device)
        
        logger.info(f"Initialized embedding model: {model_name} on {device}")
    
    def encode(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """Encode texts into embeddings
        
        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for processing
            
        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**encoded)
                embeddings = outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
                
            # Move to CPU and convert to numpy
            embeddings = embeddings.cpu().numpy()
            all_embeddings.append(embeddings)
            
        # Concatenate all batches
        return np.vstack(all_embeddings)

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
    """Manages loading and caching of all models used in the RAG system"""
    
    def __init__(self):
        """Initialize the model manager with caching"""
        self.model_cache = ModelCache()
        self._embedding_model = None
        self._qa_model = None
        self._qa_tokenizer = None
        self._generator_model = None
        self._nlp_model = None
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all required models"""
        try:
            # Load embedding model
            self._embedding_model = self.load_embedding_model()
            
            # Load QA model and tokenizer
            self._qa_model, self._qa_tokenizer = self.load_qa_model()
            
            # Load generator model
            self._generator_model = self.load_generator_model()
            
            # Load NLP model
            self._nlp_model = self.load_nlp_model()
            
            logger.info("All models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            raise
    
    @property
    def embedding_model(self) -> EmbeddingModel:
        """Get the embedding model"""
        if self._embedding_model is None:
            self._embedding_model = self.load_embedding_model()
        return self._embedding_model
    
    @property
    def qa_model(self) -> AutoModelForQuestionAnswering:
        """Get the QA model"""
        if self._qa_model is None:
            self._qa_model, self._qa_tokenizer = self.load_qa_model()
        return self._qa_model
    
    @property
    def qa_tokenizer(self) -> AutoTokenizer:
        """Get the QA tokenizer"""
        if self._qa_tokenizer is None:
            self._qa_model, self._qa_tokenizer = self.load_qa_model()
        return self._qa_tokenizer
    
    @property
    def generator_model(self) -> pipeline:
        """Get the generator model"""
        if self._generator_model is None:
            self._generator_model = self.load_generator_model()
        return self._generator_model
    
    @property
    def nlp_model(self) -> spacy.language.Language:
        """Get the NLP model"""
        if self._nlp_model is None:
            self._nlp_model = self.load_nlp_model()
        return self._nlp_model
    
    def load_embedding_model(self) -> EmbeddingModel:
        """Load embedding model with proper device handling and input validation"""
        try:
            model_name = MODEL_CONFIGS['embedding']['name']
            max_length = MODEL_CONFIGS['embedding'].get('max_length', 256)
            
            # Clear any existing cache to force fresh loading
            cache_key = f"embedding_{model_name}"
            if cache_key in self.model_cache:
                logger.info(f"Clearing cached embedding model to force fresh loading")
                del self.model_cache[cache_key]
            
            # Always load new model (don't cache to ensure fresh instance)
            logger.info(f"Loading fresh embedding model: {model_name} with max_length: {max_length}")
            model = EmbeddingModel(
                model_name=model_name,
                device=MODEL_CONFIGS['embedding']['device'],
                max_length=max_length
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
        """Load the question answering model with input validation"""
        try:
            if 'qa' not in self.model_cache:
                logger.info("Loading QA model...")
                
                # Load tokenizer using AutoTokenizer to match the model
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_CONFIGS['qa']['name']
                )
                
                # Ensure tokenizer has proper settings
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with specific configuration
                model = AutoModelForQuestionAnswering.from_pretrained(
                    MODEL_CONFIGS['qa']['name'],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=None
                )
                
                # Move model to device safely
                logger.info(f"Moving QA model to {MODEL_CONFIGS['embedding']['device']}...")
                model = safe_to_device(model, MODEL_CONFIGS['embedding']['device'])
                
                self.model_cache['qa'] = (model, tokenizer)
                logger.info("QA model loaded successfully")
            return self.model_cache['qa']
        except Exception as e:
            log_error(e, {'function': 'load_qa_model'})
            raise
    
    def validate_qa_inputs(self, question: str, context: str, max_length: int = 512) -> Tuple[str, str]:
        """Validate and truncate question-answering inputs to prevent token length errors"""
        try:
            # Load tokenizer from model name
            tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS['qa']['name'])
            
            # Calculate available tokens for context (75% of max_length)
            available_for_context = int(max_length * 0.75)
            # Reserve some tokens for special tokens and question
            reserved_tokens = 50  # For special tokens and buffer
            
            # Tokenize question and context separately
            question_tokens = tokenizer.encode(question, add_special_tokens=False)
            context_tokens = tokenizer.encode(context, add_special_tokens=False)
            
            # Calculate how many tokens we can use for context
            max_context_tokens = available_for_context - len(question_tokens) - reserved_tokens
            
            # If context is too long, truncate it intelligently
            if len(context_tokens) > max_context_tokens:
                # Try to find a good truncation point (e.g., at sentence boundary)
                truncated_context = context[:max_context_tokens * 4]  # Approximate character length
                # Find last complete sentence
                last_period = truncated_context.rfind('.')
                if last_period > 0:
                    truncated_context = truncated_context[:last_period + 1]
                context = truncated_context
            
            # Ensure question isn't too long (25% of max_length)
            max_question_tokens = int(max_length * 0.25)
            if len(question_tokens) > max_question_tokens:
                question = question[:max_question_tokens * 4]  # Approximate character length
            
            return question, context
            
        except Exception as e:
            logger.error(f"Error in token validation: {str(e)}")
            # Fallback to character-based truncation with more generous limits
            max_context_chars = max_length * 6  # More generous character limit
            max_question_chars = max_length * 4  # More generous character limit
            
            if len(context) > max_context_chars:
                context = context[:max_context_chars]
            if len(question) > max_question_chars:
                question = question[:max_question_chars]
            
            return question, context
    
    @retry_on_failure(max_retries=3)
    def get_qa_pipeline(self) -> pipeline:
        """Get the question answering pipeline with input validation"""
        try:
            if 'qa_pipeline' not in self.model_cache:
                logger.info("Creating QA pipeline...")
                model, tokenizer = self.load_qa_model()
                
                # Create pipeline with loaded model and tokenizer
                qa_pipeline = pipeline(
                    'question-answering',
                    model=model,
                    tokenizer=tokenizer,
                    device=MODEL_CONFIGS['embedding']['device'],
                    max_length=MODEL_CONFIGS['qa'].get('max_length', 256),
                    truncation=True  # Enable truncation in pipeline
                )
                
                self.model_cache['qa_pipeline'] = qa_pipeline
                logger.info("QA pipeline created successfully")
                
            return self.model_cache['qa_pipeline']
        except Exception as e:
            log_error(e, {'function': 'get_qa_pipeline'})
            raise
    
    @retry_on_failure(max_retries=3)
    def load_generator_model(self) -> pipeline:
        """Load the text generation model with input validation"""
        try:
            if 'generator' not in self.model_cache:
                logger.info("Loading generator model...")
                
                # Load tokenizer with proper configuration
                tokenizer = AutoTokenizer.from_pretrained(
                    MODEL_CONFIGS['generator']['name'],
                    model_max_length=MODEL_CONFIGS['generator'].get('max_length', 512)
                )
                
                # Ensure tokenizer has proper EOS and pad tokens
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model with specific configuration
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_CONFIGS['generator']['name'],
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map=None
                )
                
                # Move model to device safely
                logger.info(f"Moving generator model to {MODEL_CONFIGS['embedding']['device']}...")
                model = safe_to_device(model, MODEL_CONFIGS['embedding']['device'])
                
                # Create pipeline with loaded model and tokenizer
                generator = pipeline(
                    'text-generation',
                    model=model,
                    tokenizer=tokenizer,
                    device=MODEL_CONFIGS['embedding']['device'],
                    max_length=MODEL_CONFIGS['generator'].get('max_length', 512),
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    truncation=True,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2
                )
                
                self.model_cache['generator'] = generator
                logger.info("Generator model loaded successfully")
            return self.model_cache['generator']
        except Exception as e:
            log_error(e, {'function': 'load_generator_model'})
            raise
    
    def validate_generator_input(self, text: str) -> str:
        """Validate and truncate generator input"""
        try:
            max_length = MODEL_CONFIGS['generator'].get('max_length', 512)
            model_name = MODEL_CONFIGS['generator']['name']
            
            # Load tokenizer for accurate length checking
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Tokenize and truncate if necessary
            tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
            return tokenizer.decode(tokens, skip_special_tokens=True)
            
        except Exception as e:
            logger.error(f"Error validating generator input: {e}")
            # Fallback truncation
            return text[:4000] if text else ""
    
    @retry_on_failure(max_retries=3)
    def load_nlp_model(self) -> spacy.language.Language:
        """Load the spaCy NLP model"""
        try:
            if 'nlp' not in self.model_cache:
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
                
                self.model_cache['nlp'] = nlp
                logger.info("NLP model loaded successfully")
            return self.model_cache['nlp']
        except Exception as e:
            log_error(e, {'function': 'load_nlp_model'})
            raise

def load_models() -> Dict[str, Any]:
    """Load all required models"""
    try:
        manager = ModelManager()
        return {
            'embedding': manager.embedding_model,
            'qa': manager.get_qa_pipeline(),
            'generator': manager.generator_model,
            'nlp': manager.nlp_model,
            'manager': manager  # Include manager for validation methods
        }
    except Exception as e:
        log_error(e, {'function': 'load_models'})
        raise

def clear_model_cache():
    """Clear the model cache"""
    try:
        manager = ModelManager()
        manager.model_cache.clear()
        logger.info("Model cache cleared")
    except Exception as e:
        log_error(e, {'function': 'clear_model_cache'})
        raise

def get_model_info() -> Dict[str, Any]:
    """Get information about loaded models"""
    try:
        manager = ModelManager()
        return {
            'device': str(MODEL_CONFIGS['embedding']['device']),
            'loaded_models': list(manager.model_cache.keys()),
            'max_lengths': MODEL_CONFIGS['embedding'].get('max_length', 256),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        }
    except Exception as e:
        log_error(e, {'function': 'get_model_info'})
        raise

def get_embeddings(model: EmbeddingModel, texts: List[str], max_length: int = None) -> torch.Tensor:
    """
    Generate embeddings for a list of texts with proper input validation
    
    Args:
        model: The embedding model
        texts: List of texts to embed
        max_length: Maximum token length (optional)
    
    Returns:
        Embeddings tensor
    """
    try:
        if not texts:
            logger.warning("Empty texts provided to get_embeddings")
            return torch.empty((0, 384))  # Return empty tensor with correct dimensions
        
        return model.encode(texts)
    except Exception as e:
        log_error(e, {'function': 'get_embeddings'})
        logger.error(f"Input texts count: {len(texts) if texts else 0}")
        if texts:
            logger.error(f"Sample text lengths: {[len(str(t)) for t in texts[:3]]}")
        raise

# Utility function for safe QA processing
def safe_qa_process(qa_pipeline, question: str, context: str, manager: ModelManager = None) -> Dict[str, Any]:
    """
    Safely process QA with input validation
    
    Args:
        qa_pipeline: The QA pipeline
        question: Question text
        context: Context text
        manager: ModelManager instance for validation
    
    Returns:
        QA result dictionary
    """
    try:
        if manager:
            question, context = manager.validate_qa_inputs(question, context)
        
        result = qa_pipeline(question=question, context=context)
        return result
        
    except Exception as e:
        logger.error(f"Error in safe_qa_process: {e}")
        return {'answer': 'Error processing question', 'score': 0.0}

# Utility function for safe text generation
def safe_generate_text(generator_pipeline, prompt: str, manager: ModelManager = None, **kwargs) -> str:
    """
    Safely generate text with input validation
    
    Args:
        generator_pipeline: The generation pipeline
        prompt: Input prompt
        manager: ModelManager instance for validation
        **kwargs: Additional generation parameters
    
    Returns:
        Generated text
    """
    try:
        if manager:
            prompt = manager.validate_generator_input(prompt)
        
        # Set safe defaults
        kwargs.setdefault('max_length', 512)
        kwargs.setdefault('truncation', True)
        kwargs.setdefault('do_sample', True)
        kwargs.setdefault('temperature', 0.7)
        
        result = generator_pipeline(prompt, **kwargs)
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        return str(result)
        
    except Exception as e:
        logger.error(f"Error in safe_generate_text: {e}")
        return f"Error generating text: {str(e)}"