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
        
        if self.google_api_key and self.google_cse_id:
            self.provider = 'google'
        else:
            self.provider = None
    
    def search(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Web search tool is disabled'}
        
        if not self.provider:
            return {
                'query': query,
                'results': [],
                'error': 'No web search provider configured. Set GOOGLE_API_KEY + GOOGLE_CSE_ID.'
            }
        
        return self._search_google(query, max_results)
    
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
            return {'error': str(e), 'query': query, 'results': []}
    
_web_search_tool = None

def get_web_search_tool() -> WebSearchTool:
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
