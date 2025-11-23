"""Transport tool using Google Directions API with HERE fallback"""

from typing import Dict, Any, Optional, List
import requests
from rag_system.core.config import get_config
import os
from datetime import datetime

class TransportTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.transport.enabled', True)
        
        # Prefer Google Directions over HERE
        self.google_api_key = os.getenv('GOOGLE_DIRECTIONS_API_KEY')
        self.here_api_key = self.config.get('tools.transport.api_key') or os.getenv('HERE_API_KEY')
        
        if self.google_api_key:
            self.provider = 'google'
            self.api_key = self.google_api_key
        elif self.here_api_key:
            self.provider = 'here'
            self.api_key = self.here_api_key
            self.transit_url = "https://transit.router.hereapi.com/v8/routes"
            self.geocode_url = "https://geocode.search.hereapi.com/v1/geocode"
        else:
            self.provider = None
            self.api_key = None
    
    def get_route(self, origin: str, destination: str, mode: str = 'transit') -> Dict[str, Any]:
        """
        Get route from origin to destination using Google Directions API (primary) or HERE (fallback).
        
        Args:
            origin: Starting location (e.g., "K11 MUSEA")
            destination: Destination location (e.g., "HKUST")
            mode: Travel mode - 'transit' (public transport), 'driving', 'walking', 'bicycling'
        
        Returns:
            Dict with route information including steps, duration, distance, legs
        """
        if not self.enabled:
            return {'error': 'Transport tool is disabled'}
        
        if not self.api_key:
            return {'error': 'No transport API key configured (need GOOGLE_DIRECTIONS_API_KEY or HERE_API_KEY)'}
        
        try:
            if self.provider == 'google':
                return self._get_google_route(origin, destination, mode)
            else:
                return self._get_here_route(origin, destination, mode)
        except Exception as e:
            return {'error': str(e)}
    
    def _get_google_route(self, origin: str, destination: str, mode: str = 'transit') -> Dict[str, Any]:
        """Get route using Google Directions API"""
        try:
            url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'key': self.google_api_key,
                'language': 'zh-CN',  # Chinese for better Hong Kong support
                'alternatives': 'true'  # Get multiple route options
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('status') != 'OK':
                return {'error': f"Google Directions API error: {data.get('status')} - {data.get('error_message', 'Unknown error')}"}
            
            # Parse all routes
            routes = []
            for route in data.get('routes', []):
                route_info = self._parse_google_route(route)
                routes.append(route_info)
            
            return {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'routes': routes,
                'provider': 'google',
                'status': 'success'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_google_route(self, route: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single route from Google Directions API"""
        legs = route.get('legs', [])
        if not legs:
            return {'error': 'No legs found in route'}
        
        # Parse all legs (for multi-waypoint routes)
        parsed_legs = []
        for leg in legs:
            steps = []
            for step in leg.get('steps', []):
                step_info = {
                    'instruction': step.get('html_instructions', '').replace('<b>', '').replace('</b>', '').replace('<div>', ' ').replace('</div>', ''),
                    'distance': step.get('distance', {}).get('text', ''),
                    'duration': step.get('duration', {}).get('text', ''),
                    'travel_mode': step.get('travel_mode', '')
                }
                
                # For transit, add detailed transit information
                if 'transit_details' in step:
                    transit = step['transit_details']
                    step_info['transit'] = {
                        'line': transit.get('line', {}).get('short_name', transit.get('line', {}).get('name', '')),
                        'vehicle': transit.get('line', {}).get('vehicle', {}).get('name', ''),
                        'departure_stop': transit.get('departure_stop', {}).get('name', ''),
                        'arrival_stop': transit.get('arrival_stop', {}).get('name', ''),
                        'num_stops': transit.get('num_stops', 0),
                        'headsign': transit.get('headsign', ''),
                        'departure_time': transit.get('departure_time', {}).get('text', ''),
                        'arrival_time': transit.get('arrival_time', {}).get('text', '')
                    }
                
                steps.append(step_info)
            
            parsed_legs.append({
                'start_address': leg.get('start_address', ''),
                'end_address': leg.get('end_address', ''),
                'distance': leg.get('distance', {}).get('text', ''),
                'duration': leg.get('duration', {}).get('text', ''),
                'steps': steps
            })
        
        return {
            'summary': route.get('summary', ''),
            'legs': parsed_legs,
            'total_distance': legs[0].get('distance', {}).get('text', '') if legs else '',
            'total_duration': legs[0].get('duration', {}).get('text', '') if legs else ''
        }
    
    def _get_here_route(self, origin: str, destination: str, mode: str = 'transit') -> Dict[str, Any]:
        """Get route using HERE API (fallback)"""
        try:
            # Geocode origin and destination
            origin_coords = self._geocode(origin)
            dest_coords = self._geocode(destination)
            
            if not origin_coords:
                return {'error': f'Could not find location: {origin}'}
            if not dest_coords:
                return {'error': f'Could not find location: {destination}'}
            
            # Get transit route
            if mode == 'transit':
                route_data = self._get_transit_route(origin_coords, dest_coords)
            else:
                # For car/pedestrian, use regular routing API
                route_data = self._get_regular_route(origin_coords, dest_coords, mode)
            
            if 'error' in route_data:
                return route_data
            
            return {
                'origin': origin,
                'destination': destination,
                'mode': mode,
                'route': route_data,
                'provider': 'here',
                'status': 'success'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _geocode(self, location: str) -> Optional[str]:
        """Geocode location to coordinates using HERE Geocoding API"""
        try:
            params = {
                'q': location,
                'apiKey': self.api_key,
                'limit': 1
            }
            
            response = requests.get(self.geocode_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data.get('items') and len(data['items']) > 0:
                pos = data['items'][0]['position']
                return f"{pos['lat']},{pos['lng']}"
            return None
        except Exception:
            return None
    
    def _get_transit_route(self, origin: str, destination: str) -> Dict[str, Any]:
        """Get transit route using HERE Transit API"""
        try:
            params = {
                'origin': origin,
                'destination': destination,
                'apiKey': self.api_key
            }
            
            response = requests.get(self.transit_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'routes' not in data or not data['routes']:
                return {'error': 'No routes found'}
            
            # Parse the first route
            route = data['routes'][0]
            sections = route.get('sections', [])
            
            # Extract route details
            steps = []
            for section in sections:
                step = self._parse_transit_section(section)
                steps.append(step)
            
            # Calculate total duration
            total_duration_minutes = 0
            if sections:
                first_departure = sections[0].get('departure', {}).get('time')
                last_arrival = sections[-1].get('arrival', {}).get('time')
                
                if first_departure and last_arrival:
                    try:
                        dep_time = datetime.fromisoformat(first_departure.replace('Z', '+00:00'))
                        arr_time = datetime.fromisoformat(last_arrival.replace('Z', '+00:00'))
                        total_duration_minutes = (arr_time - dep_time).total_seconds() / 60
                    except:
                        pass
            
            return {
                'steps': steps,
                'total_duration_minutes': round(total_duration_minutes, 1),
                'departure_time': sections[0].get('departure', {}).get('time') if sections else None,
                'arrival_time': sections[-1].get('arrival', {}).get('time') if sections else None,
                'num_steps': len(steps)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _parse_transit_section(self, section: Dict[str, Any]) -> Dict[str, Any]:
        """Parse a single transit section"""
        transport = section.get('transport', {})
        departure = section.get('departure', {})
        arrival = section.get('arrival', {})
        
        step = {
            'type': section.get('type', 'unknown'),
            'mode': transport.get('mode', 'unknown')
        }
        
        # Add transit details for bus/train
        if transport.get('mode') in ['bus', 'train', 'subway', 'tram']:
            step['transit'] = {
                'line': transport.get('name', 'N/A'),
                'category': transport.get('category', 'N/A'),
                'headsign': transport.get('headsign', 'N/A')
            }
        
        # Add departure info
        if departure.get('place'):
            step['from'] = departure['place'].get('name', 'N/A')
        if departure.get('time'):
            step['departure_time'] = departure['time']
        
        # Add arrival info
        if arrival.get('place'):
            step['to'] = arrival['place'].get('name', 'N/A')
        if arrival.get('time'):
            step['arrival_time'] = arrival['time']
        
        return step
    
    def _get_regular_route(self, origin: str, destination: str, mode: str) -> Dict[str, Any]:
        """Get car/pedestrian route using HERE Routing API"""
        try:
            routing_url = "https://router.hereapi.com/v8/routes"
            params = {
                'transportMode': mode,
                'origin': origin,
                'destination': destination,
                'return': 'summary',
                'apiKey': self.api_key
            }
            
            response = requests.get(routing_url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            if 'routes' not in data or not data['routes']:
                return {'error': 'No routes found'}
            
            route = data['routes'][0]
            sections = route.get('sections', [])
            
            if sections:
                summary = sections[0].get('summary', {})
                return {
                    'duration_minutes': round(summary.get('duration', 0) / 60, 1),
                    'distance_km': round(summary.get('length', 0) / 1000, 1)
                }
            
            return {'error': 'No route data found'}
        except Exception as e:
            return {'error': str(e)}

_transport_tool = None

def get_transport_tool() -> TransportTool:
    global _transport_tool
    if _transport_tool is None:
        _transport_tool = TransportTool()
    return _transport_tool
