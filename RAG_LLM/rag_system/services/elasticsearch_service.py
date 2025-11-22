"""Elasticsearch BM25 service"""

from typing import List, Dict, Any, Optional
import uuid
from elasticsearch import Elasticsearch
from rag_system.core.config import get_config

class ElasticsearchService:
    def __init__(self):
        config = get_config()
        host = config.get('elasticsearch.host', 'localhost')
        port = config.get('elasticsearch.port', 9200)
        self.index_name = config.get('elasticsearch.index_name', 'rag_documents')
        
        self.client = Elasticsearch([f"http://{host}:{port}"])
        self._ensure_index()
    
    def _ensure_index(self):
        if not self.client.indices.exists(index=self.index_name):
            mapping = {
                "mappings": {
                    "properties": {
                        "text": {"type": "text", "analyzer": "standard"},
                        "source": {"type": "keyword"},
                        "url": {"type": "keyword"},
                        "doc_id": {"type": "keyword"},
                        "created_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                        "updated_at": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
                        "page": {"type": "integer"},
                        "section": {"type": "text"},
                        "credibility_score": {"type": "float"}
                    }
                }
            }
            self.client.indices.create(index=self.index_name, body=mapping)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        doc_ids = []
        for doc in documents:
            doc_id = doc.get('id', str(uuid.uuid4()))
            doc_ids.append(doc_id)
            
            body = {
                'text': doc['text'],
                'source': doc.get('source', ''),
                'url': doc.get('url', ''),
                'doc_id': doc.get('doc_id', ''),
                'created_at': doc.get('created_at', ''),
                'updated_at': doc.get('updated_at', ''),
                'page': doc.get('page', 0),
                'section': doc.get('section', ''),
                'credibility_score': doc.get('credibility_score', 0.5),
            }
            
            self.client.index(index=self.index_name, id=doc_id, body=body)
        
        self.client.indices.refresh(index=self.index_name)
        return doc_ids
    
    def search(self, query: str, top_k: int = 50) -> List[Dict[str, Any]]:
        search_body = {
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": ["text^2", "section"],
                    "type": "best_fields"
                }
            },
            "size": top_k
        }
        
        results = self.client.search(index=self.index_name, body=search_body)
        
        documents = []
        for hit in results['hits']['hits']:
            doc = {
                'id': hit['_id'],
                'score': hit['_score'],
                **hit['_source']
            }
            documents.append(doc)
        
        return documents
    
    def delete_all(self):
        if self.client.indices.exists(index=self.index_name):
            self.client.indices.delete(index=self.index_name)
        self._ensure_index()

_elasticsearch_service = None

def get_elasticsearch_service() -> ElasticsearchService:
    global _elasticsearch_service
    if _elasticsearch_service is None:
        _elasticsearch_service = ElasticsearchService()
    return _elasticsearch_service
