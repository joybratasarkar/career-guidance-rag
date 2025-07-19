import logging
from typing import List, Union, Optional
import asyncio
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)


class SharedEmbeddingService:
    """Shared embedding service using SentenceTransformer"""
    
    _instance: Optional['SharedEmbeddingService'] = None
    _model: Optional[SentenceTransformer] = None
    _model_name: str = "sentence-transformers/all-MiniLM-L6-V2"
    
    def __new__(cls):
        """Singleton pattern to ensure only one instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding service"""
        if self._model is None:
            logger.info(f"Loading embedding model: {self._model_name}")
            self._model = SentenceTransformer(self._model_name)
            logger.info("Embedding model loaded successfully")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get the embedding model"""
        if self._model is None:
            self.__init__()
        return self._model
    
    def encode(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Encode texts to embeddings"""
        return self.model.encode(texts, **kwargs)
    
    async def encode_async(self, texts: Union[str, List[str]], **kwargs) -> np.ndarray:
        """Async wrapper for encoding"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.encode, texts, **kwargs)
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        return self.model.get_sentence_embedding_dimension()
    
    @classmethod
    def get_instance(cls) -> 'SharedEmbeddingService':
        """Get the singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
