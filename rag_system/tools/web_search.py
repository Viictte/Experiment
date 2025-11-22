"""Web search tool with Google Custom Search and Tavily API support"""

from typing import Dict, Any, List, Optional
import requests
from rag_system.core.config import get_config
import os

class WebSearchTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.web_search.enabled', True)
        
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        self.tavily_api_key = self.config.get('tools.web_search.api_key') or os.getenv('TAVILY_API_KEY')
        
        if self.google_api_key and self.google_cse_id:
            self.provider = 'google'
        elif self.tavily_api_key:
            self.provider = 'tavily'
        else:
            self.provider = None
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Web search tool is disabled'}
        
        if not self.provider:
            return {
                'query': query,
                'results': [],
                'error': 'No web search provider configured. Set GOOGLE_API_KEY + GOOGLE_CSE_ID or TAVILY_API_KEY.'
            }
        
        if self.provider == 'google':
            return self._search_google(query, max_results)
        else:
            return self._search_tavily(query, max_results)
    
    def _search_google(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            url = "https://www.googleapis.com/customsearch/v1"
            params = {
                'key': self.google_api_key,
                'cx': self.google_cse_id,
                'q': query,
                'num': min(max_results, 10)
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            results = []
            for item in data.get('items', []):
                results.append({
                    'title': item.get('title', ''),
                    'url': item.get('link', ''),
                    'content': item.get('snippet', ''),
                    'snippet': item.get('snippet', '')
                })
            
            return {
                'query': query,
                'results': results,
                'provider': 'google'
            }
        except Exception as e:
            if self.tavily_api_key:
                return self._search_tavily(query, max_results)
            return {'error': str(e), 'query': query, 'results': []}
    
    def _search_tavily(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        try:
            url = "https://api.tavily.com/search"
            headers = {
                'Content-Type': 'application/json'
            }
            body = {
                'api_key': self.tavily_api_key,
                'query': query,
                'max_results': max_results,
                'search_depth': 'basic',
                'include_answer': True
            }
            
            response = requests.post(url, json=body, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'query': query,
                'results': data.get('results', []),
                'answer': data.get('answer', ''),
                'provider': 'tavily'
            }
        except Exception as e:
            return {'error': str(e), 'query': query, 'results': []}

_web_search_tool = None

def get_web_search_tool() -> WebSearchTool:
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
