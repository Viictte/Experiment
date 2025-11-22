"""Cross-encoder reranker service"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sentence_transformers import CrossEncoder
from rag_system.core.config import get_config
from rag_system.services.redis_service import get_redis_service

class RerankerService:
    def __init__(self):
        config = get_config()
        model_name = config.get('reranker.model', 'BAAI/bge-reranker-large')
        self.batch_size = config.get('reranker.batch_size', 32)
        self.top_k = config.get('reranker.top_k', 24)
        
        self.model = CrossEncoder(model_name)
        self.redis_service = get_redis_service()
    
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.top_k
        
        if len(documents) == 0:
            return []
        
        pairs = []
        cached_scores = []
        indices_to_compute = []
        
        for i, doc in enumerate(documents):
            doc_id = doc.get('id', '')
            cached_score = self.redis_service.get_rerank_cache(doc_id, query)
            
            if cached_score is not None:
                cached_scores.append((i, cached_score))
            else:
                pairs.append([query, doc['text']])
                indices_to_compute.append(i)
        
        computed_scores = []
        if pairs:
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            computed_scores = list(zip(indices_to_compute, scores))
            
            for idx, score in computed_scores:
                doc_id = documents[idx].get('id', '')
                self.redis_service.set_rerank_cache(doc_id, query, float(score))
        
        all_scores = cached_scores + computed_scores
        all_scores.sort(key=lambda x: x[0])
        
        for i, (idx, score) in enumerate(all_scores):
            documents[idx]['rerank_score'] = float(score)
        
        documents.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)
        
        return documents[:top_k]

_reranker_service = None

def get_reranker_service() -> RerankerService:
    global _reranker_service
    if _reranker_service is None:
        _reranker_service = RerankerService()
    return _reranker_service
