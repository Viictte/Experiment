"""Vision tool using Gemini API for image recognition"""

import base64
import requests
import json
from typing import Optional
import os
from rag_system.core.config import get_config


class VisionTool:
    def __init__(self):
        self.config = get_config()
        self.enabled = self.config.get('tools.vision.enabled', True)
        self.api_key = self.config.get('tools.vision.api_key') or os.getenv('VISION_API_KEY')
        self.api_url = self.config.get('tools.vision.api_url', 'https://yinli.one/v1/chat/completions')
        self.model = self.config.get('tools.vision.model', 'gemini-2.5-flash-lite')
        self.max_tokens = self.config.get('tools.vision.max_tokens', 500)
        self.timeout = self.config.get('tools.vision.timeout', 30)
    
    def describe_image(self, image_bytes: bytes, prompt: str = "Describe this image in detail. If it's a logo or emblem, identify the organization.") -> Optional[str]:
        """
        Describe an image using the Gemini vision API.
        
        Args:
            image_bytes: Raw image bytes
            prompt: The prompt to send with the image
            
        Returns:
            Description of the image, or None if vision is disabled or fails
        """
        if not self.enabled:
            return None
        
        if not self.api_key:
            return None
        
        try:
            # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Prepare the API request
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": self.max_tokens
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            
            # Make the API request
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and result['choices']:
                    content = result['choices'][0]['message']['content']
                    return content
                else:
                    return None
            else:
                # Log error but don't raise - graceful degradation
                return None
                
        except Exception as e:
            # Graceful degradation - return None on any error
            return None


# Singleton instance
_vision_tool = None

def get_vision_tool() -> VisionTool:
    """Get or create the vision tool singleton"""
    global _vision_tool
    if _vision_tool is None:
        _vision_tool = VisionTool()
    return _vision_tool
