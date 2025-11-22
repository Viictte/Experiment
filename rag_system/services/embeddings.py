"""Embedding service using bge-m3"""

from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_system.core.config import get_config

class EmbeddingService:
    def __init__(self):
        config = get_config()
        model_name = config.get('embeddings.model', 'BAAI/bge-m3')
        self.batch_size = config.get('embeddings.batch_size', 32)
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
    
    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100
        )
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_text(query)

_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
