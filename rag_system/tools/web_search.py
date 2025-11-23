"""Web search tool with Google Custom Search and domain filtering"""

from typing import Dict, Any, List, Optional
import requests
from rag_system.core.config import get_config
import os
from urllib.parse import urlparse

class WebSearchTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.web_search.enabled', True)
        
        # Prefer Tavily over Google
        self.tavily_api_key = os.getenv('TAVILY_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.google_cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if self.tavily_api_key:
            self.provider = 'tavily'
        elif self.google_api_key and self.google_cse_id:
            self.provider = 'google'
        else:
            self.provider = None
    
    def search(self, query: str, max_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the web with optional domain filtering.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            filters: Optional dict with keys:
                - must_domains: List of domains that results MUST come from
                - preferred_domains: List of domains to prioritize in ranking
                - blocked_domains: List of domains to exclude from results
        """
        if not self.enabled:
            return {'error': 'Web search tool is disabled'}
        
        if not self.provider:
            return {
                'query': query,
                'results': [],
                'error': 'No web search provider configured. Set TAVILY_API_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID.'
            }
        
        if self.provider == 'tavily':
            return self._search_tavily(query, max_results, filters)
        else:
            return self._search_google(query, max_results, filters)
    
    def _domain_of(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return ""
    
    def _search_google(self, query: str, max_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
                url_str = item.get('link', '')
                domain = self._domain_of(url_str)
                results.append({
                    'title': item.get('title', ''),
                    'url': url_str,
                    'domain': domain,
                    'content': item.get('snippet', ''),
                    'snippet': item.get('snippet', '')
                })
            
            # Apply filters if provided
            if filters:
                results = self._apply_filters(results, filters)
            
            return {
                'query': query,
                'results': results,
                'provider': 'google'
            }
        except Exception as e:
            return {'error': str(e), 'query': query, 'results': []}
    
    def _search_tavily(self, query: str, max_results: int = 5, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Search using Tavily API with full content snippets"""
        try:
            url = "https://api.tavily.com/search"
            headers = {'Content-Type': 'application/json'}
            
            # Build Tavily request with advanced features
            data = {
                'api_key': self.tavily_api_key,
                'query': query,
                'search_depth': 'advanced',  # Advanced mode for better quality
                'max_results': max_results,
                'include_answer': True,  # Use Tavily's high-quality answer synthesis
                'include_raw_content': False  # Snippets are sufficient
            }
            
            # Map our filters to Tavily's native domain filtering
            if filters:
                must_domains = filters.get('must_domains', [])
                preferred_domains = filters.get('preferred_domains', [])
                blocked_domains = filters.get('blocked_domains', [])
                
                # Tavily uses include_domains and exclude_domains
                if must_domains:
                    data['include_domains'] = must_domains
                elif preferred_domains:
                    # For preferred (not must), we'll apply local filtering instead
                    pass
                
                if blocked_domains:
                    data['exclude_domains'] = blocked_domains
            
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            tavily_result = response.json()
            
            # Normalize Tavily results to match our expected format
            results = []
            for item in tavily_result.get('results', []):
                url_str = item.get('url', '')
                domain = self._domain_of(url_str)
                content = item.get('content', '')
                
                results.append({
                    'title': item.get('title', ''),
                    'url': url_str,
                    'domain': domain,
                    'content': content,
                    'snippet': content,  # Backward compatibility
                    'score': item.get('score', 0.8)  # Tavily's relevance score
                })
            
            # Apply local filtering for preferred_domains (not must_domains)
            if filters and filters.get('preferred_domains') and not filters.get('must_domains'):
                results = self._apply_filters(results, filters)
            
            # Capture Tavily's answer field if present
            response_data = {
                'query': query,
                'results': results,
                'provider': 'tavily'
            }
            
            # Include Tavily's synthesized answer if available
            if 'answer' in tavily_result and tavily_result['answer']:
                response_data['answer'] = tavily_result['answer']
            
            return response_data
        except Exception as e:
            return {'error': str(e), 'query': query, 'results': [], 'provider': 'tavily'}
    
    def _apply_filters(self, results: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply domain filters to search results"""
        must_domains = set(d.lower() for d in filters.get('must_domains', []) if d)
        blocked_domains = set(d.lower() for d in filters.get('blocked_domains', []) if d)
        preferred_domains = set(d.lower() for d in filters.get('preferred_domains', []) if d)
        
        # Filter out blocked domains and enforce must_domains
        filtered = []
        for result in results:
            domain = result.get('domain', '')
            
            # Block unwanted domains
            if blocked_domains and any(domain.endswith(b) or b in domain for b in blocked_domains):
                continue
            
            # Enforce must_domains if specified
            if must_domains and not any(domain.endswith(m) or m in domain for m in must_domains):
                continue
            
            filtered.append(result)
        
        if not filtered:
            # If filtering removed everything, return original results
            # (better to have some results than none)
            filtered = results
        
        # Reorder to prioritize preferred domains
        if preferred_domains:
            filtered.sort(
                key=lambda r: (
                    0 if any(r.get('domain', '').endswith(p) or p in r.get('domain', '') for p in preferred_domains) else 1
                )
            )
        
        return filtered
    
_web_search_tool = None

def get_web_search_tool() -> WebSearchTool:
    global _web_search_tool
    if _web_search_tool is None:
        _web_search_tool = WebSearchTool()
    return _web_search_tool
