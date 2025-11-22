"""Time tool using WorldTimeAPI.org"""

from typing import Dict, Any, Optional
import requests
from datetime import datetime

# Location to timezone mapping
LOCATION_TO_TIMEZONE = {
    # Hong Kong
    'hong kong': 'Asia/Hong_Kong',
    'hk': 'Asia/Hong_Kong',
    '香港': 'Asia/Hong_Kong',
    
    # China
    'beijing': 'Asia/Shanghai',
    'shanghai': 'Asia/Shanghai',
    'china': 'Asia/Shanghai',
    '北京': 'Asia/Shanghai',
    '上海': 'Asia/Shanghai',
    '中国': 'Asia/Shanghai',
    
    # Japan
    'tokyo': 'Asia/Tokyo',
    'japan': 'Asia/Tokyo',
    '東京': 'Asia/Tokyo',
    '日本': 'Asia/Tokyo',
    
    # USA
    'new york': 'America/New_York',
    'los angeles': 'America/Los_Angeles',
    'chicago': 'America/Chicago',
    'san francisco': 'America/Los_Angeles',
    'washington': 'America/New_York',
    'usa': 'America/New_York',
    
    # UK
    'london': 'Europe/London',
    'uk': 'Europe/London',
    'england': 'Europe/London',
    
    # Europe
    'paris': 'Europe/Paris',
    'berlin': 'Europe/Berlin',
    'rome': 'Europe/Rome',
    'madrid': 'Europe/Madrid',
    'amsterdam': 'Europe/Amsterdam',
    'france': 'Europe/Paris',
    'germany': 'Europe/Berlin',
    'italy': 'Europe/Rome',
    'spain': 'Europe/Madrid',
    
    # Asia
    'singapore': 'Asia/Singapore',
    'bangkok': 'Asia/Bangkok',
    'seoul': 'Asia/Seoul',
    'taipei': 'Asia/Taipei',
    'manila': 'Asia/Manila',
    'kuala lumpur': 'Asia/Kuala_Lumpur',
    'jakarta': 'Asia/Jakarta',
    'hanoi': 'Asia/Ho_Chi_Minh',
    'mumbai': 'Asia/Kolkata',
    'delhi': 'Asia/Kolkata',
    'dubai': 'Asia/Dubai',
    
    # Australia
    'sydney': 'Australia/Sydney',
    'melbourne': 'Australia/Melbourne',
    'brisbane': 'Australia/Brisbane',
    'perth': 'Australia/Perth',
    'australia': 'Australia/Sydney',
    
    # Others
    'moscow': 'Europe/Moscow',
    'toronto': 'America/Toronto',
    'vancouver': 'America/Vancouver',
    'montreal': 'America/Toronto',
    'canada': 'America/Toronto',
}

class TimeTool:
    """Tool for getting current time using WorldTimeAPI.org"""
    
    def __init__(self):
        self.base_url = "https://worldtimeapi.org/api/timezone"
        self.default_timezone = "Asia/Hong_Kong"  # Default to Hong Kong
    
    def get_current_time(self, location: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current time for a location.
        
        Args:
            location: Location name (e.g., "Hong Kong", "New York")
                     If None, defaults to Hong Kong
        
        Returns:
            Dict with time information
        """
        # Map location to timezone
        timezone = self._map_location_to_timezone(location)
        
        # Try API call with retries
        for attempt in range(3):
            try:
                # Call WorldTimeAPI
                url = f"{self.base_url}/{timezone}"
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                
                data = response.json()
                
                # Parse datetime
                datetime_str = data.get('datetime', '')
                dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
                
                return {
                    'location': location or 'Hong Kong',
                    'timezone': timezone,
                    'datetime': datetime_str,
                    'formatted_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'formatted_date': dt.strftime('%Y-%m-%d'),
                    'formatted_time_only': dt.strftime('%H:%M:%S'),
                    'day_of_week': data.get('day_of_week'),
                    'day_of_week_name': dt.strftime('%A'),
                    'abbreviation': data.get('abbreviation'),
                    'utc_offset': data.get('utc_offset'),
                    'unixtime': data.get('unixtime'),
                    'data': data
                }
            except Exception as e:
                if attempt == 2:  # Last attempt
                    # Fallback to Python datetime if API fails
                    return self._get_time_fallback(location, timezone)
                continue
        
        # Should not reach here, but fallback just in case
        return self._get_time_fallback(location, timezone)
    
    def _get_time_fallback(self, location: Optional[str], timezone: str) -> Dict[str, Any]:
        """Fallback to Python datetime if API fails"""
        try:
            from zoneinfo import ZoneInfo
            dt = datetime.now(ZoneInfo(timezone))
            
            return {
                'location': location or 'Hong Kong',
                'timezone': timezone,
                'datetime': dt.isoformat(),
                'formatted_time': dt.strftime('%Y-%m-%d %H:%M:%S'),
                'formatted_date': dt.strftime('%Y-%m-%d'),
                'formatted_time_only': dt.strftime('%H:%M:%S'),
                'day_of_week': dt.weekday(),
                'day_of_week_name': dt.strftime('%A'),
                'abbreviation': dt.strftime('%Z'),
                'utc_offset': dt.strftime('%z'),
                'source': 'fallback'
            }
        except Exception as e:
            return {'error': f'Failed to get time: {str(e)}'}
    
    def _map_location_to_timezone(self, location: Optional[str]) -> str:
        """Map location name to timezone identifier"""
        if not location:
            return self.default_timezone
        
        # Normalize location
        location_lower = location.lower().strip()
        
        # Direct lookup
        if location_lower in LOCATION_TO_TIMEZONE:
            return LOCATION_TO_TIMEZONE[location_lower]
        
        # Partial match (e.g., "in Hong Kong" → "hong kong")
        for loc_key, timezone in LOCATION_TO_TIMEZONE.items():
            if loc_key in location_lower:
                return timezone
        
        # Default to Hong Kong if no match
        return self.default_timezone

_time_tool = None

def get_time_tool() -> TimeTool:
    """Get singleton time tool instance"""
    global _time_tool
    if _time_tool is None:
        _time_tool = TimeTool()
    return _time_tool
