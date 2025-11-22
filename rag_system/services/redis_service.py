"""Redis caching service"""

import json
import hashlib
from typing import Any, Optional
import redis
from rag_system.core.config import get_config

class RedisService:
    def __init__(self):
        config = get_config()
        host = config.get('redis.host', 'localhost')
        port = config.get('redis.port', 6379)
        db = config.get('redis.db', 0)
        self.ttl = config.get('redis.ttl_seconds', 3600)
        
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
    
    def _make_key(self, prefix: str, data: Any) -> str:
        data_str = json.dumps(data, sort_keys=True)
        hash_val = hashlib.sha256(data_str.encode()).hexdigest()[:16]
        return f"{prefix}:{hash_val}"
    
    def get(self, key: str) -> Optional[Any]:
        value = self.client.get(key)
        if value:
            return json.loads(value)
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        if ttl is None:
            ttl = self.ttl
        self.client.setex(key, ttl, json.dumps(value))
    
    def get_query_cache(self, query: str, top_k: int) -> Optional[Any]:
        key = self._make_key('query', {'query': query, 'top_k': top_k})
        return self.get(key)
    
    def set_query_cache(self, query: str, top_k: int, results: Any):
        key = self._make_key('query', {'query': query, 'top_k': top_k})
        self.set(key, results)
    
    def get_rerank_cache(self, doc_id: str, query: str) -> Optional[float]:
        key = self._make_key('rerank', {'doc_id': doc_id, 'query': query})
        result = self.get(key)
        return result if result is None else float(result)
    
    def set_rerank_cache(self, doc_id: str, query: str, score: float):
        key = self._make_key('rerank', {'doc_id': doc_id, 'query': query})
        self.set(key, score)
    
    def get_answer_cache(self, prompt: str, citations: str) -> Optional[str]:
        key = self._make_key('answer', {'prompt': prompt, 'citations': citations})
        return self.get(key)
    
    def set_answer_cache(self, prompt: str, citations: str, answer: str):
        key = self._make_key('answer', {'prompt': prompt, 'citations': citations})
        self.set(key, answer)
    
    def get_tool_cache(self, tool_name: str, params: dict) -> Optional[Any]:
        """Get cached tool output (for weather, finance, transport)"""
        key = self._make_key(f'tool:{tool_name}', params)
        return self.get(key)
    
    def set_tool_cache(self, tool_name: str, params: dict, result: Any, ttl: int = 300):
        """Cache tool output with shorter TTL (default 5 minutes for time-sensitive data)"""
        key = self._make_key(f'tool:{tool_name}', params)
        self.set(key, result, ttl=ttl)
    
    def clear_all(self):
        self.client.flushdb()

_redis_service = None

def get_redis_service() -> RedisService:
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
    return _redis_service
