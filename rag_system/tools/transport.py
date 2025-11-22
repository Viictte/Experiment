"""Transport tool using Google Maps Directions API"""

from typing import Dict, Any, Optional, List
import requests
from rag_system.core.config import get_config
import os

class TransportTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.transport.enabled', True)
        self.api_key = self.config.get('tools.transport.api_key') or os.getenv('GOOGLE_MAPS_API_KEY')
        self.base_url = "https://maps.googleapis.com/maps/api/directions/json"
    
    def get_route(self, origin: str, destination: str, mode: str = 'transit') -> Dict[str, Any]:
        """
        Get route from origin to destination using Google Maps Directions API.
        
        Args:
            origin: Starting location (e.g., "K11 MUSEA")
            destination: Destination location (e.g., "HKUST")
            mode: Travel mode - 'transit', 'driving', 'walking', 'bicycling'
        
        Returns:
            Dict with route information including steps, duration, distance
        """
        if not self.enabled:
            return {'error': 'Transport tool is disabled'}
        
        if not self.api_key:
            return {'error': 'Google Maps API key not configured'}
        
        try:
            # Call Google Maps Directions API
            params = {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'key': self.api_key,
                'language': 'en',
                'alternatives': 'true'  # Get alternative routes
            }
            
            response = requests.get(self.base_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'OK':
                return {'error': f"Google Maps API error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"}
            
            # Parse routes
            routes = []
            for route in data.get('routes', []):
                route_info = self._parse_route(route)
                routes.append(route_info)
            
            return {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'routes': routes,
                'status': 'success'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_route(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single route from Google Maps API response"""
        legs = route.get('legs', [])
        if not legs:
            return {'error': 'No legs found in route'}
        
        # For now, focus on the first leg (origin to destination)
        leg = legs[0]
        
        # Extract steps
        steps = []
        for step in leg.get('steps', []):
            step_info = {
                'instruction': step.get('html_instructions', '').replace('<b>', '').replace('</b>', '').replace('<div>', ' ').replace('</div>', ''),
                'distance': step.get('distance', {}).get('text', ''),
                'duration': step.get('duration', {}).get('text', ''),
                'travel_mode': step.get('travel_mode', '')
            }
            
            # For transit, add transit details
            if 'transit_details' in step:
                transit = step['transit_details']
                step_info['transit'] = {
                    'line': transit.get('line', {}).get('short_name', transit.get('line', {}).get('name', '')),
                    'vehicle': transit.get('line', {}).get('vehicle', {}).get('name', ''),
                    'departure_stop': transit.get('departure_stop', {}).get('name', ''),
                    'arrival_stop': transit.get('arrival_stop', {}).get('name', ''),
                    'num_stops': transit.get('num_stops', 0),
                    'headsign': transit.get('headsign', '')
                }
            
            steps.append(step_info)
        
        return {
            'summary': route.get('summary', ''),
            'total_distance': leg.get('distance', {}).get('text', ''),
            'total_duration': leg.get('duration', {}).get('text', ''),
            'start_address': leg.get('start_address', ''),
            'end_address': leg.get('end_address', ''),
            'steps': steps
        }

_transport_tool = None

def get_transport_tool() -> TransportTool:
    global _transport_tool
    if _transport_tool is None:
        _transport_tool = TransportTool()
    return _transport_tool
