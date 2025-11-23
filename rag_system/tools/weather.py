"""Weather tool with WeatherAPI.com and Open-Meteo support"""

from typing import Dict, Any, Optional
import requests
import os
from datetime import datetime, timedelta
from rag_system.core.config import get_config

# WMO Weather Code Descriptions
WMO_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail"
}

class WeatherTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.weather.enabled', True)
        # Default timezone for time-based queries (can be overridden)
        self.default_timezone = self.config.get('tools.weather.default_timezone', 'Asia/Hong_Kong')
        
        # Prefer WeatherAPI over Open-Meteo
        self.weatherapi_key = os.getenv('WEATHERAPI_KEY')
        self.google_air_quality_key = os.getenv('GOOGLE_AIR_QUALITY_API_KEY')
        
        if self.weatherapi_key:
            self.provider = 'weatherapi'
        else:
            self.provider = 'open_meteo'
    
    def get_weather(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {'error': 'Weather tool is disabled'}
        
        try:
            if self.provider == 'weatherapi':
                return self._get_weatherapi_weather(location, date)
            else:
                return self._get_openmeteo_weather(location, date)
        except Exception as e:
            return {'error': str(e)}
    
    def _get_openmeteo_weather(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get weather using Open-Meteo API (fallback)"""
        coords = self._geocode(location)
        if not coords:
            return {'error': f'Could not geocode location: {location}'}
        
        lat, lon = coords
        
        if date:
            weather_data = self._get_forecast(lat, lon, date)
        else:
            weather_data = self._get_current(lat, lon)
        
        return {
            'location': location,
            'latitude': lat,
            'longitude': lon,
            'data': weather_data,
            'provider': 'open_meteo'
        }
    
    def _get_weatherapi_weather(self, location: str, date: Optional[str] = None) -> Dict[str, Any]:
        """Get weather using WeatherAPI.com (primary)"""
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': self.weatherapi_key,
            'q': location,
            'days': 1,
            'aqi': 'no',
            'alerts': 'no'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Normalize to match expected format
        current = data.get('current', {})
        location_data = data.get('location', {})
        
        return {
            'location': location,
            'latitude': location_data.get('lat'),
            'longitude': location_data.get('lon'),
            'current_time': location_data.get('localtime'),
            'timezone': location_data.get('tz_id'),
            'temperature': current.get('temp_c'),
            'precipitation': current.get('precip_mm'),
            'windspeed': current.get('wind_kph'),
            'humidity': current.get('humidity'),
            'feels_like': current.get('feelslike_c'),
            'weather_description': current.get('condition', {}).get('text'),
            'weathercode': current.get('condition', {}).get('code'),
            'data': data,
            'provider': 'weatherapi'
        }
    
    def _geocode(self, location: str) -> Optional[tuple]:
        try:
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'q': location,
                'format': 'json',
                'limit': 5,
                'addressdetails': 1
            }
            headers = {'User-Agent': 'RAG-System/1.0'}
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            results = response.json()
            if not results:
                return None
            
            location_lower = location.lower()
            for result in results:
                address = result.get('address', {})
                display_name = result.get('display_name', '').lower()
                
                if 'taipei' in location_lower:
                    if 'taiwan' in display_name or address.get('country_code') == 'tw':
                        return (float(result['lat']), float(result['lon']))
                
                if any(keyword in location_lower for keyword in ['tokyo', '東京']):
                    if 'japan' in display_name or address.get('country_code') == 'jp':
                        return (float(result['lat']), float(result['lon']))
                
                if any(keyword in location_lower for keyword in ['beijing', '北京']):
                    if 'china' in display_name or address.get('country_code') == 'cn':
                        return (float(result['lat']), float(result['lon']))
            
            return (float(results[0]['lat']), float(results[0]['lon']))
        except Exception:
            return None
    
    def _get_current(self, lat: float, lon: float) -> Dict[str, Any]:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current_weather': True,
            'hourly': 'temperature_2m,precipitation,windspeed_10m',
            'timezone': self.default_timezone
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Add WMO code description to current weather
        if 'current_weather' in data:
            weathercode = data['current_weather'].get('weathercode', 0)
            data['current_weather']['weather_description'] = WMO_CODES.get(weathercode, f"Unknown code {weathercode}")
        
        return data
    
    def _get_forecast(self, lat: float, lon: float, date: str) -> Dict[str, Any]:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'hourly': 'temperature_2m,precipitation,windspeed_10m,weathercode',
            'start_date': date,
            'end_date': date,
            'timezone': self.default_timezone
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Add WMO code descriptions to hourly data
        if 'hourly' in data and 'weathercode' in data['hourly']:
            data['hourly']['weather_descriptions'] = [
                WMO_CODES.get(code, f"Unknown code {code}")
                for code in data['hourly']['weathercode']
            ]
        
        return data
    
    def get_afternoon_forecast(self, location: str) -> Dict[str, Any]:
        """Get forecast specifically for this afternoon (12:00-18:00 local time)"""
        if not self.enabled:
            return {'error': 'Weather tool is disabled'}
        
        try:
            if self.provider == 'weatherapi':
                return self._get_weatherapi_afternoon(location)
            else:
                return self._get_openmeteo_afternoon(location)
        except Exception as e:
            return {'error': str(e)}
    
    def _get_weatherapi_afternoon(self, location: str) -> Dict[str, Any]:
        """Get afternoon forecast using WeatherAPI.com"""
        url = "http://api.weatherapi.com/v1/forecast.json"
        params = {
            'key': self.weatherapi_key,
            'q': location,
            'days': 1,
            'aqi': 'no',
            'alerts': 'no'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract location and time info
        location_data = data.get('location', {})
        current_time_str = location_data.get('localtime', '')
        current_time = datetime.fromisoformat(current_time_str) if current_time_str else datetime.now()
        
        # Define afternoon window (12:00-18:00)
        afternoon_start = current_time.replace(hour=12, minute=0, second=0, microsecond=0)
        afternoon_end = current_time.replace(hour=18, minute=0, second=0, microsecond=0)
        
        # Filter hourly data for afternoon hours
        hourly = data.get('forecast', {}).get('forecastday', [{}])[0].get('hour', [])
        
        afternoon_data = []
        for hour in hourly:
            time_str = hour.get('time', '')
            if not time_str:
                continue
            
            time_obj = datetime.fromisoformat(time_str)
            if afternoon_start <= time_obj <= afternoon_end:
                afternoon_data.append({
                    'time': time_str,
                    'temperature': hour.get('temp_c'),
                    'precipitation': hour.get('precip_mm'),
                    'windspeed': hour.get('wind_kph'),
                    'humidity': hour.get('humidity'),
                    'weathercode': hour.get('condition', {}).get('code'),
                    'weather_description': hour.get('condition', {}).get('text')
                })
        
        # Determine if afternoon has passed
        afternoon_status = "current" if current_time < afternoon_end else "past"
        if current_time < afternoon_start:
            afternoon_status = "upcoming"
        
        return {
            'location': location,
            'latitude': location_data.get('lat'),
            'longitude': location_data.get('lon'),
            'current_time': current_time_str,
            'timezone': location_data.get('tz_id'),
            'afternoon_status': afternoon_status,
            'afternoon_window': f"{afternoon_start.strftime('%H:%M')}-{afternoon_end.strftime('%H:%M')}",
            'afternoon_data': afternoon_data,
            'data': data,
            'provider': 'weatherapi'
        }
    
    def _get_openmeteo_afternoon(self, location: str) -> Dict[str, Any]:
        """Get afternoon forecast using Open-Meteo (fallback)"""
        coords = self._geocode(location)
        if not coords:
            return {'error': f'Could not geocode location: {location}'}
        
        lat, lon = coords
        
        # Get current weather data with timezone
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat,
            'longitude': lon,
            'current_weather': True,
            'hourly': 'temperature_2m,precipitation,windspeed_10m,weathercode',
            'timezone': self.default_timezone
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Parse current time from API response
        current_time_str = data.get('current_weather', {}).get('time', '')
        if current_time_str:
            current_time = datetime.fromisoformat(current_time_str)
        else:
            current_time = datetime.now()
        
        # Define afternoon window (12:00-18:00)
        afternoon_start = current_time.replace(hour=12, minute=0, second=0, microsecond=0)
        afternoon_end = current_time.replace(hour=18, minute=0, second=0, microsecond=0)
        
        # Filter hourly data for afternoon hours
        hourly = data.get('hourly', {})
        times = hourly.get('time', [])
        temps = hourly.get('temperature_2m', [])
        precip = hourly.get('precipitation', [])
        wind = hourly.get('windspeed_10m', [])
        codes = hourly.get('weathercode', [])
        
        afternoon_data = []
        for i, time_str in enumerate(times):
            time_obj = datetime.fromisoformat(time_str)
            if afternoon_start <= time_obj <= afternoon_end:
                afternoon_data.append({
                    'time': time_str,
                    'temperature': temps[i] if i < len(temps) else None,
                    'precipitation': precip[i] if i < len(precip) else None,
                    'windspeed': wind[i] if i < len(wind) else None,
                    'weathercode': codes[i] if i < len(codes) else None,
                    'weather_description': WMO_CODES.get(codes[i], f"Unknown code {codes[i]}") if i < len(codes) else None
                })
        
        # Determine if afternoon has passed
        afternoon_status = "current" if current_time < afternoon_end else "past"
        if current_time < afternoon_start:
            afternoon_status = "upcoming"
        
        return {
            'location': location,
            'latitude': lat,
            'longitude': lon,
            'current_time': current_time_str,
            'afternoon_status': afternoon_status,
            'afternoon_window': f"{afternoon_start.strftime('%H:%M')}-{afternoon_end.strftime('%H:%M')}",
            'afternoon_data': afternoon_data,
            'data': data,
            'provider': 'open_meteo'
        }
    
    def get_air_quality(self, location: str) -> Dict[str, Any]:
        """
        Get air quality information for a location using Google Air Quality API.
        
        Args:
            location: Location name (e.g., "Hong Kong Central", "香港中環")
        
        Returns:
            Dict with air quality data including AQI, pollutants, health recommendations
        """
        if not self.enabled:
            return {'error': 'Weather tool is disabled'}
        
        if not self.google_air_quality_key:
            return {'error': 'Google Air Quality API key not configured'}
        
        try:
            # First geocode the location to get coordinates
            coords = self._geocode(location)
            if not coords:
                return {'error': f'Could not geocode location: {location}'}
            
            lat, lon = coords
            
            # Call Google Air Quality API
            url = "https://airquality.googleapis.com/v1/currentConditions:lookup"
            params = {
                'key': self.google_air_quality_key
            }
            
            payload = {
                "location": {
                    "latitude": lat,
                    "longitude": lon
                },
                "extraComputations": [
                    "HEALTH_RECOMMENDATIONS",
                    "DOMINANT_POLLUTANT_CONCENTRATION",
                    "POLLUTANT_CONCENTRATION",
                    "LOCAL_AQI",
                    "POLLUTANT_ADDITIONAL_INFO"
                ],
                "languageCode": "zh-CN"
            }
            
            response = requests.post(url, params=params, json=payload, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract key information
            indexes = data.get('indexes', [])
            pollutants = data.get('pollutants', [])
            health_recommendations = data.get('healthRecommendations', {})
            
            # Find the main AQI index (usually the first one)
            main_aqi = None
            if indexes:
                main_aqi = indexes[0]
            
            return {
                'location': location,
                'latitude': lat,
                'longitude': lon,
                'aqi': main_aqi.get('aqi') if main_aqi else None,
                'aqi_display': main_aqi.get('aqiDisplay') if main_aqi else None,
                'category': main_aqi.get('category') if main_aqi else None,
                'dominant_pollutant': main_aqi.get('dominantPollutant') if main_aqi else None,
                'indexes': indexes,
                'pollutants': pollutants,
                'health_recommendations': health_recommendations,
                'provider': 'google_air_quality',
                'data': data
            }
        except Exception as e:
            return {'error': str(e)}

_weather_tool = None

def get_weather_tool() -> WeatherTool:
    global _weather_tool
    if _weather_tool is None:
        _weather_tool = WeatherTool()
    return _weather_tool
