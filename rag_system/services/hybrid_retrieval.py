"""Hybrid retrieval with RRF (Reciprocal Rank Fusion)"""

from typing import List, Dict, Any, Optional
import math
from datetime import datetime
from collections import defaultdict
from rag_system.core.config import get_config
from rag_system.services.qdrant_service import get_qdrant_service
from rag_system.services.elasticsearch_service import get_elasticsearch_service
from rag_system.services.reranker import get_reranker_service
from rag_system.services.redis_service import get_redis_service

class HybridRetrievalService:
    def __init__(self):
        config = get_config()
        self.dense_top_k = config.get('retrieval.dense_top_k', 50)
        self.bm25_top_k = config.get('retrieval.bm25_top_k', 50)
        self.rrf_k = config.get('retrieval.rrf_k', 60)
        self.final_top_k = config.get('retrieval.final_top_k', 50)
        
        self.cross_encoder_weight = config.get('scoring.cross_encoder_weight', 0.55)
        self.base_retrieval_weight = config.get('scoring.base_retrieval_weight', 0.25)
        self.freshness_weight = config.get('scoring.freshness_weight', 0.12)
        self.credibility_weight = config.get('scoring.credibility_weight', 0.08)
        self.freshness_tau = config.get('scoring.freshness_tau_days', 30)
        self.dedup_threshold = config.get('scoring.dedup_threshold', 0.9)
        
        self.qdrant = get_qdrant_service()
        self.elasticsearch = get_elasticsearch_service()
        self.reranker = get_reranker_service()
        self.redis = get_redis_service()
    
    def _reciprocal_rank_fusion(self, dense_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        doc_scores = defaultdict(lambda: {'score': 0, 'doc': None})
        
        for rank, doc in enumerate(dense_results):
            doc_id = doc['id']
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc_id]['score'] += rrf_score
            doc_scores[doc_id]['doc'] = doc
        
        for rank, doc in enumerate(bm25_results):
            doc_id = doc['id']
            rrf_score = 1.0 / (self.rrf_k + rank + 1)
            doc_scores[doc_id]['score'] += rrf_score
            if doc_scores[doc_id]['doc'] is None:
                doc_scores[doc_id]['doc'] = doc
        
        merged_docs = []
        for doc_id, data in doc_scores.items():
            doc = data['doc']
            doc['rrf_score'] = data['score']
            merged_docs.append(doc)
        
        merged_docs.sort(key=lambda x: x['rrf_score'], reverse=True)
        return merged_docs[:self.final_top_k]
    
    def _calculate_freshness_score(self, updated_at: str) -> float:
        if not updated_at:
            return 0.5
        
        try:
            updated_date = datetime.fromisoformat(updated_at.replace('Z', '+00:00'))
            now = datetime.now(updated_date.tzinfo)
            days_old = (now - updated_date).days
            freshness = math.exp(-days_old / self.freshness_tau)
            return freshness
        except:
            return 0.5
    
    def _calculate_final_score(self, doc: Dict[str, Any]) -> float:
        rerank_score = doc.get('rerank_score', 0)
        rrf_score = doc.get('rrf_score', 0)
        freshness_score = self._calculate_freshness_score(doc.get('updated_at', ''))
        credibility_score = doc.get('credibility_score', 0.5)
        
        final_score = (
            self.cross_encoder_weight * rerank_score +
            self.base_retrieval_weight * rrf_score +
            self.freshness_weight * freshness_score +
            self.credibility_weight * credibility_score
        )
        
        return final_score
    
    def _deduplicate(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if len(documents) <= 1:
            return documents
        
        unique_docs = []
        seen_texts = []
        
        for doc in documents:
            text = doc['text']
            is_duplicate = False
            
            for seen_text in seen_texts:
                similarity = self._text_similarity(text, seen_text)
                if similarity > self.dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                seen_texts.append(text)
        
        return unique_docs
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if top_k is None:
            top_k = self.reranker.top_k
        
        cached_results = self.redis.get_query_cache(query, top_k)
        if cached_results:
            return cached_results
        
        dense_results = self.qdrant.search(query, self.dense_top_k)
        bm25_results = self.elasticsearch.search(query, self.bm25_top_k)
        
        merged_docs = self._reciprocal_rank_fusion(dense_results, bm25_results)
        
        reranked_docs = self.reranker.rerank(query, merged_docs, top_k)
        
        for doc in reranked_docs:
            doc['final_score'] = self._calculate_final_score(doc)
        
        reranked_docs.sort(key=lambda x: x['final_score'], reverse=True)
        
        unique_docs = self._deduplicate(reranked_docs)
        
        self.redis.set_query_cache(query, top_k, unique_docs)
        
        return unique_docs

_hybrid_retrieval_service = None

def get_hybrid_retrieval_service() -> HybridRetrievalService:
    global _hybrid_retrieval_service
    if _hybrid_retrieval_service is None:
        _hybrid_retrieval_service = HybridRetrievalService()
    return _hybrid_retrieval_service
