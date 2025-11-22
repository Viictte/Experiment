"""Configuration management for RAG system"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    def __init__(self, config_path: Optional[str] = None):
        project_root = Path(__file__).parent.parent.parent
        load_dotenv(dotenv_path=project_root / ".env")
        
        if config_path is None:
            config_path = project_root / "config" / "config.yaml"
        
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
        
        self._env_overrides()
    
    def _env_overrides(self):
        if os.getenv('DEEPSEEK_API_KEY'):
            self._config.setdefault('llm', {})['api_key'] = os.getenv('DEEPSEEK_API_KEY')
        
        if os.getenv('OPENROUTESERVICE_API_KEY'):
            self._config.setdefault('tools', {}).setdefault('transport', {})['api_key'] = os.getenv('OPENROUTESERVICE_API_KEY')
        
        if os.getenv('REDIS_HOST'):
            self._config.setdefault('redis', {})['host'] = os.getenv('REDIS_HOST')
        if os.getenv('REDIS_PORT'):
            self._config.setdefault('redis', {})['port'] = int(os.getenv('REDIS_PORT'))
        
        if os.getenv('QDRANT_HOST'):
            self._config.setdefault('qdrant', {})['host'] = os.getenv('QDRANT_HOST')
        if os.getenv('QDRANT_PORT'):
            self._config.setdefault('qdrant', {})['port'] = int(os.getenv('QDRANT_PORT'))
        
        if os.getenv('ELASTICSEARCH_HOST'):
            self._config.setdefault('elasticsearch', {})['host'] = os.getenv('ELASTICSEARCH_HOST')
        if os.getenv('ELASTICSEARCH_PORT'):
            self._config.setdefault('elasticsearch', {})['port'] = int(os.getenv('ELASTICSEARCH_PORT'))
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def set(self, key: str, value: Any):
        keys = key.split('.')
        config = self._config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    def save(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
        
        with open(config_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config

_global_config = None

def get_config() -> Config:
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config

def set_config(config: Config):
    global _global_config
    _global_config = config
