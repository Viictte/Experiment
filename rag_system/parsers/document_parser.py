"""Document parser with multimodal support"""

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import requests
from urllib.parse import urlparse
from rag_system.core.config import get_config

class DocumentParser:
    def __init__(self):
        self.config = get_config()
        self.chunk_target_size = self.config.get('chunking.target_size', 800)
        self.chunk_overlap = self.config.get('chunking.overlap', 100)
        self.chunk_min_size = self.config.get('chunking.min_size', 200)
        self.chunk_max_size = self.config.get('chunking.max_size', 1200)
    
    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = file_path.suffix.lower()
        
        if ext == '.pdf':
            return self._parse_pdf(file_path)
        elif ext in ['.txt', '.md']:
            return self._parse_text(file_path)
        elif ext in ['.html', '.htm']:
            return self._parse_html(file_path)
        elif ext in ['.doc', '.docx']:
            return self._parse_doc(file_path)
        elif ext in ['.png', '.jpg', '.jpeg', '.gif']:
            return self._parse_image(file_path)
        else:
            return self._parse_text(file_path)
    
    def parse_url(self, url: str) -> List[Dict[str, Any]]:
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            
            if 'text/html' in content_type:
                return self._parse_html_content(response.text, url)
            else:
                return self._parse_text_content(response.text, url)
        except Exception as e:
            raise Exception(f"Failed to parse URL {url}: {str(e)}")
    
    def _parse_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            import pymupdf
            
            doc = pymupdf.open(file_path)
            chunks = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                
                if text.strip():
                    page_chunks = self._chunk_text(text)
                    for chunk in page_chunks:
                        chunks.append({
                            'text': chunk,
                            'source': str(file_path),
                            'doc_id': file_path.stem,
                            'page': page_num + 1,
                            'created_at': datetime.now().isoformat(),
                            'updated_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            'credibility_score': self._get_credibility_score(str(file_path))
                        })
            
            return chunks
        except ImportError:
            return self._parse_text(file_path)
    
    def _parse_text(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        chunks = self._chunk_text(text)
        
        return [{
            'text': chunk,
            'source': str(file_path),
            'doc_id': file_path.stem,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
            'credibility_score': self._get_credibility_score(str(file_path))
        } for chunk in chunks]
    
    def _parse_html(self, file_path: Path) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        return self._parse_html_content(html_content, str(file_path))
    
    def _parse_html_content(self, html_content: str, source: str) -> List[Dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            chunks = self._chunk_text(text)
            
            return [{
                'text': chunk,
                'source': source,
                'url': source if source.startswith('http') else '',
                'doc_id': urlparse(source).path.split('/')[-1] if source.startswith('http') else Path(source).stem,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat(),
                'credibility_score': self._get_credibility_score(source)
            } for chunk in chunks]
        except ImportError:
            return self._parse_text_content(html_content, source)
    
    def _parse_text_content(self, text: str, source: str) -> List[Dict[str, Any]]:
        chunks = self._chunk_text(text)
        
        return [{
            'text': chunk,
            'source': source,
            'url': source if source.startswith('http') else '',
            'doc_id': urlparse(source).path.split('/')[-1] if source.startswith('http') else 'unknown',
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'credibility_score': self._get_credibility_score(source)
        } for chunk in chunks]
    
    def _parse_doc(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            import docx
            
            doc = docx.Document(file_path)
            text = '\n'.join([para.text for para in doc.paragraphs])
            
            chunks = self._chunk_text(text)
            
            return [{
                'text': chunk,
                'source': str(file_path),
                'doc_id': file_path.stem,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'credibility_score': self._get_credibility_score(str(file_path))
            } for chunk in chunks]
        except ImportError:
            return self._parse_text(file_path)
    
    def _parse_image(self, file_path: Path) -> List[Dict[str, Any]]:
        try:
            import pytesseract
            from PIL import Image
            
            image = Image.open(file_path)
            text = pytesseract.image_to_string(image)
            
            if not text.strip():
                return []
            
            chunks = self._chunk_text(text)
            
            return [{
                'text': chunk,
                'source': str(file_path),
                'doc_id': file_path.stem,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                'credibility_score': self._get_credibility_score(str(file_path))
            } for chunk in chunks]
        except ImportError:
            return []
    
    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        
        if len(words) <= self.chunk_target_size:
            return [text] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + self.chunk_target_size
            chunk_words = words[start:end]
            chunk = ' '.join(chunk_words)
            
            if len(chunk) >= self.chunk_min_size:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            
            if start >= len(words):
                break
        
        return chunks
    
    def _get_credibility_score(self, source: str) -> float:
        credibility_priors = self.config.get('credibility_priors', {})
        default_score = credibility_priors.get('default', 0.5)
        domains = credibility_priors.get('domains', {})
        
        if source.startswith('http'):
            parsed = urlparse(source)
            domain = parsed.netloc
            
            for known_domain, score in domains.items():
                if known_domain in domain:
                    return score
        
        return default_score

_document_parser = None

def get_document_parser() -> DocumentParser:
    global _document_parser
    if _document_parser is None:
        _document_parser = DocumentParser()
    return _document_parser
