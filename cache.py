"""
Caching module for the RAG Chatbot application.
Provides efficient caching for queries and model instances.
"""

import time
import logging
from typing import Dict, Any, Optional, Tuple
from collections import OrderedDict
import torch
from config import CACHE_TTL, MODEL_CONFIGS

logger = logging.getLogger(__name__)

class QueryCache:
    """LRU Cache for query results with TTL"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if it exists and is not expired"""
        if key in self.cache:
            # Check if item has expired
            if time.time() - self.timestamps[key] > CACHE_TTL:
                self._remove(key)
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Add item to cache with timestamp"""
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.max_size:
            # Remove least recently used item
            self._remove(self.cache.popitem(last=False)[0])
        
        self.cache[key] = value
        self.timestamps[key] = time.time()
    
    def _remove(self, key: str) -> None:
        """Remove item from cache and timestamps"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def clear(self) -> None:
        """Clear all cached items"""
        self.cache.clear()
        self.timestamps.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'expired_items': sum(1 for ts in self.timestamps.values() 
                               if time.time() - ts > CACHE_TTL)
        }

class ModelCache:
    """Cache for ML models with device management"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
    
    def get_model(self, model_name: str) -> Optional[Any]:
        """Get model from cache if it exists"""
        return self.models.get(model_name)
    
    def add_model(self, model_name: str, model: Any) -> None:
        """Add model to cache"""
        if model_name in MODEL_CONFIGS:
            # Move model to appropriate device
            if hasattr(model, 'to'):
                model = model.to(self.device)
            self.models[model_name] = model
        else:
            logger.warning(f"Model {model_name} not in MODEL_CONFIGS")
    
    def remove_model(self, model_name: str) -> None:
        """Remove model from cache"""
        if model_name in self.models:
            # Clear CUDA cache if model was on GPU
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            del self.models[model_name]
    
    def clear(self) -> None:
        """Clear all cached models"""
        self.models.clear()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_available_models(self) -> list:
        """Get list of cached model names"""
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a cached model"""
        if model_name not in self.models:
            return {}
        
        model = self.models[model_name]
        info = {
            'device': str(next(model.parameters()).device) if hasattr(model, 'parameters') else 'unknown',
            'type': type(model).__name__,
            'config': MODEL_CONFIGS.get(model_name, {})
        }
        
        # Add model-specific information
        if hasattr(model, 'config'):
            info['model_config'] = model.config.to_dict()
        
        return info

class EmbeddingCache:
    """Cache for document embeddings"""
    
    def __init__(self):
        self.embeddings: Dict[str, torch.Tensor] = {}
        self.metadata: Dict[str, Dict] = {}
    
    def get_embeddings(self, key: str) -> Tuple[Optional[torch.Tensor], Optional[Dict]]:
        """Get embeddings and metadata for a key"""
        return self.embeddings.get(key), self.metadata.get(key)
    
    def add_embeddings(self, key: str, embeddings: torch.Tensor, metadata: Dict) -> None:
        """Add embeddings and metadata to cache"""
        self.embeddings[key] = embeddings
        self.metadata[key] = metadata
    
    def clear(self) -> None:
        """Clear all cached embeddings"""
        self.embeddings.clear()
        self.metadata.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            'num_embeddings': len(self.embeddings),
            'total_vectors': sum(e.shape[0] for e in self.embeddings.values()),
            'vector_dim': next(iter(self.embeddings.values())).shape[1] if self.embeddings else 0
        } 