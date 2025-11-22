"""Qdrant vector database service"""

from typing import List, Dict, Any, Optional
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    SearchRequest, ScalarQuantization, ScalarQuantizationConfig,
    ScalarType, QuantizationSearchParams
)
from rag_system.core.config import get_config
from rag_system.services.embeddings import get_embedding_service

class QdrantService:
    def __init__(self):
        config = get_config()
        host = config.get('qdrant.host', 'localhost')
        port = config.get('qdrant.port', 6333)
        self.collection_name = config.get('qdrant.collection_name', 'rag_documents')
        self.hnsw_m = config.get('qdrant.hnsw_m', 16)
        self.hnsw_ef_search = config.get('qdrant.hnsw_ef_search', 64)
        self.quantization = config.get('qdrant.quantization', True)
        
        self.client = QdrantClient(host=host, port=port)
        self.embedding_service = get_embedding_service()
        self._ensure_collection()
    
    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            quantization_config = None
            if self.quantization:
                quantization_config = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_service.dimension,
                    distance=Distance.COSINE
                ),
                hnsw_config={
                    "m": self.hnsw_m,
                    "ef_construct": 100
                },
                quantization_config=quantization_config
            )
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        texts = [doc['text'] for doc in documents]
        embeddings = self.embedding_service.embed_texts(texts)
        
        points = []
        doc_ids = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = doc.get('id', str(uuid.uuid4()))
            doc_ids.append(doc_id)
            
            payload = {
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
            
            points.append(PointStruct(
                id=doc_id,
                vector=embedding.tolist(),
                payload=payload
            ))
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return doc_ids
    
    def search(self, query: str, top_k: int = 50, filter_dict: Optional[Dict] = None) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_service.embed_query(query)
        
        search_params = None
        if self.quantization:
            search_params = QuantizationSearchParams(
                ignore=False,
                rescore=True
            )
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k,
            search_params={"hnsw_ef": self.hnsw_ef_search},
            with_payload=True
        )
        
        documents = []
        for result in results:
            doc = {
                'id': result.id,
                'score': result.score,
                **result.payload
            }
            documents.append(doc)
        
        return documents
    
    def delete_all(self):
        self.client.delete_collection(collection_name=self.collection_name)
        self._ensure_collection()

_qdrant_service = None

def get_qdrant_service() -> QdrantService:
    global _qdrant_service
    if _qdrant_service is None:
        _qdrant_service = QdrantService()
    return _qdrant_service
