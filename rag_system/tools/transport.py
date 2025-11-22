"""Transport tool using OpenRouteService"""

from typing import Dict, Any, Optional
import requests
from rag_system.core.config import get_config
import os

class TransportTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.transport.enabled', True)
        self.api_key = self.config.get('tools.transport.api_key') or os.getenv('OPENROUTESERVICE_API_KEY')
    
    def get_route(self, origin: str, destination: str, mode: str = 'driving') -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Transport tool is disabled'}
        
        if not self.api_key:
            return {'error': 'OpenRouteService API key not configured. Using mock data.', 'mock': True}
        
        try:
            origin_coords = self._geocode(origin)
            dest_coords = self._geocode(destination)
            
            if not origin_coords or not dest_coords:
                return {'error': 'Could not geocode locations'}
            
            route_data = self._get_route_data(origin_coords, dest_coords, mode)
            
            return {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'data': route_data
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _geocode(self, location: str) -> Optional[tuple]:
        try:
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 1
            }
            headers = {'User-Agent': 'RAG-System/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            if results:
                return (float(results[0]['lon']), float(results[0]['lat']))
            return None
        except Exception:
            return None
    
    def _get_route_data(self, origin: tuple, destination: tuple, mode: str) -> Dict[str, Any]:
        if not self.api_key:
            return {
                'distance': 10000,
                'duration': 600,
                'mock': True
            }
        
        url = f"https://api.openrouteservice.org/v2/directions/{mode}"
        headers = {
            'Authorization': self.api_key,
            'Content-Type': 'application/json'
        }
        body = {
            'coordinates': [list(origin), list(destination)]
        }
        
        response = requests.post(url, json=body, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if 'routes' in data and len(data['routes']) > 0:
            route = data['routes'][0]
            return {
                'distance': route['summary']['distance'],
                'duration': route['summary']['duration']
            }
        
        return {'error': 'No route found'}

_transport_tool = None

def get_transport_tool() -> TransportTool:
    global _transport_tool
    if _transport_tool is None:
        _transport_tool = TransportTool()
    return _transport_tool
