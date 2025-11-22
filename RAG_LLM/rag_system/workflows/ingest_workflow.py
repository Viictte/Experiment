"""Document ingestion workflow"""

from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from rag_system.parsers.document_parser import get_document_parser
from rag_system.services.qdrant_service import get_qdrant_service
from rag_system.services.elasticsearch_service import get_elasticsearch_service

class IngestWorkflow:
    def __init__(self):
        self.parser = get_document_parser()
        self.qdrant = get_qdrant_service()
        self.elasticsearch = get_elasticsearch_service()
    
    def ingest_path(self, path: str) -> Dict[str, Any]:
        path_obj = Path(path)
        
        if path.startswith('http://') or path.startswith('https://'):
            return self.ingest_url(path)
        elif path_obj.is_file():
            return self.ingest_file(str(path_obj))
        elif path_obj.is_dir():
            return self.ingest_directory(str(path_obj))
        else:
            return {'error': f'Invalid path: {path}'}
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        try:
            chunks = self.parser.parse_file(file_path)
            
            if not chunks:
                return {
                    'file': file_path,
                    'chunks': 0,
                    'status': 'no_content'
                }
            
            qdrant_ids = self.qdrant.add_documents(chunks)
            es_ids = self.elasticsearch.add_documents(chunks)
            
            return {
                'file': file_path,
                'chunks': len(chunks),
                'status': 'success'
            }
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'status': 'error'
            }
    
    def ingest_url(self, url: str) -> Dict[str, Any]:
        try:
            chunks = self.parser.parse_url(url)
            
            if not chunks:
                return {
                    'url': url,
                    'chunks': 0,
                    'status': 'no_content'
                }
            
            qdrant_ids = self.qdrant.add_documents(chunks)
            es_ids = self.elasticsearch.add_documents(chunks)
            
            return {
                'url': url,
                'chunks': len(chunks),
                'status': 'success'
            }
        except Exception as e:
            return {
                'url': url,
                'error': str(e),
                'status': 'error'
            }
    
    def ingest_directory(self, dir_path: str) -> Dict[str, Any]:
        dir_obj = Path(dir_path)
        
        if not dir_obj.is_dir():
            return {'error': f'Not a directory: {dir_path}'}
        
        files = list(dir_obj.rglob('*'))
        files = [f for f in files if f.is_file()]
        
        results = []
        total_chunks = 0
        
        for file_path in tqdm(files, desc="Ingesting files"):
            result = self.ingest_file(str(file_path))
            results.append(result)
            total_chunks += result.get('chunks', 0)
        
        success_count = sum(1 for r in results if r.get('status') == 'success')
        error_count = sum(1 for r in results if r.get('status') == 'error')
        
        return {
            'directory': dir_path,
            'total_files': len(files),
            'success': success_count,
            'errors': error_count,
            'total_chunks': total_chunks,
            'results': results
        }

_ingest_workflow = None

def get_ingest_workflow() -> IngestWorkflow:
    global _ingest_workflow
    if _ingest_workflow is None:
        _ingest_workflow = IngestWorkflow()
    return _ingest_workflow
