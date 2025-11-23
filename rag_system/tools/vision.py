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
        
        # Candidate models to try in order of preference
        self.candidate_models = [
            "gemini-2.5-flash-lite-preview-06-17",
            "gemini-2.5-flash-lite",
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite-preview-06-17-nothinking",
            "gemini-2.5-flash-lite-nothinking",
            "gemini-2.5-flash-nothinking",
            "gemini-2.5-flash-preview-05-20-nothinking",
        ]
        
        self.model = self.config.get('tools.vision.model', self.candidate_models[0])
        self.max_tokens = self.config.get('tools.vision.max_tokens', 500)
        self.timeout = self.config.get('tools.vision.timeout', 30)
        
        # Model selection state
        self._model_checked = False
        self._last_error = None
    
    def _test_model(self, model: str) -> bool:
        """
        Test if a specific Gemini model is available and working.
        
        Args:
            model: Model name to test
            
        Returns:
            True if model works, False otherwise
        """
        try:
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "ping"}
                        ],
                    }
                ],
                "max_tokens": 1,
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=5  # Short timeout for testing
            )
            
            if response.status_code == 200:
                data = response.json()
                # Verify response has expected structure
                return "choices" in data and len(data["choices"]) > 0
            else:
                self._last_error = f"{response.status_code}: {response.text[:100]}"
                return False
                
        except Exception as e:
            self._last_error = str(e)
            return False
    
    def select_working_model(self, force: bool = False) -> Optional[str]:
        """
        Test candidate models and select the first working one.
        
        Args:
            force: If True, re-test even if already checked
            
        Returns:
            Working model name, or None if all models fail
        """
        if self._model_checked and not force:
            return self.model if self.model else None
        
        if not self.enabled or not self.api_key:
            self._model_checked = True
            self._last_error = "Vision tool disabled or API key not configured"
            return None
        
        # Test each candidate model
        for candidate in self.candidate_models:
            if self._test_model(candidate):
                self.model = candidate
                self._model_checked = True
                return candidate
        
        # All models failed
        self._model_checked = True
        return None
    
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
        
        # Auto-select working model if not yet checked
        if not self._model_checked:
            model = self.select_working_model(force=False)
            if not model:
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
