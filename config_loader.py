import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional

# Load environment variables from .env file if it exists (Issue #8)
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env from current working directory
except ImportError:
    pass  # python-dotenv not installed, continue without env loading


class ConfigLoader:
    """
    Central configuration management for the RAG Pipeline.
    Loads settings from config.yaml with environment variable overrides.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        self._load_config()
        self._apply_env_overrides()
        
    def _load_config(self) -> None:
        """Load YAML configuration into memory."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides for sensitive or deployment settings."""
        # LM Studio connection
        if os.getenv("LM_STUDIO_BASE_URL"):
            self.config['lmstudio']['base_url'] = os.getenv("LM_STUDIO_BASE_URL")
            
        if os.getenv("LM_STUDIO_API_KEY"):
            self.config['lmstudio']['api_key'] = os.getenv("LM_STUDIO_API_KEY")
            
        # Server settings
        if os.getenv("SERVER_PORT"):
            try:
                self.config['server']['port'] = int(os.getenv("SERVER_PORT"))
            except ValueError:
                pass
                
        if os.getenv("SERVER_HOST"):
            self.config['server']['host'] = os.getenv("SERVER_HOST")
            
        # Paths can be overridden via environment variables for different environments
        env_paths = ['SOURCE_DIR', 'CHROMA_PATH', 'LOCAL_MODELS_DIR']
        path_map = {
            'SOURCE_DIR': 'source_dir',
            'CHROMA_PATH': 'chroma_path', 
            'LOCAL_MODELS_DIR': 'local_models_dir'
        }
        
        for env_var, config_key in path_map.items():
            if os.getenv(env_var):
                self.config['paths'][config_key] = os.getenv(env_var)

    def get(self, *keys: str, default: Any = None) -> Any:
        """Safely retrieve nested configuration values."""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value
    
    def get_paths(self) -> Dict[str, Path]:
        """Return all path configurations as resolved Path objects.
        
        FIX #7: Paths are now relative to config file location, not cwd.
        This ensures consistent behavior regardless of where the user runs from.
        """
        paths_cfg = self.config['paths']
        # Use config file's parent directory as base, not current working directory
        base_dir = self.config_path.parent.resolve()
        
        return {
            'source_dir': base_dir / paths_cfg['source_dir'],
            'chroma_path': base_dir / paths_cfg['chroma_path'],
            'local_models_dir': base_dir / paths_cfg['local_models_dir']
        }

    def validate(self) -> None:
        """Perform basic validation on configuration values."""
        errors = []
        
        # Validate required sections exist
        required_sections = ['paths', 'collection', 'lmstudio', 'server']
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required config section: {section}")
                
        # Validate paths exist where applicable
        source_dir = Path(self.get('paths', 'source_dir'))
        chroma_path = Path(self.get('paths', 'chroma_path'))
        
        if not source_dir.exists():
            print(f"[WARNING] Source directory does not exist: {source_dir}")
            
        # Validate numeric ranges
        chunk_size = self.get('chunking', 'chunk_size')
        if chunk_size and (chunk_size < 100 or chunk_size > 4000):
            errors.append("Chunk size should be between 100-4000")
            
        top_k = self.get('retrieval', 'top_k')
        if top_k and (top_k < 1 or top_k > 50):
            errors.append("Retrieval top_k should be between 1-50")
            
        if errors:
            for error in errors:
                print(f"[ERROR] {error}")
            raise ValueError("Configuration validation failed")

# Singleton instance pattern for convenience
_config_instance: Optional[ConfigLoader] = None


def get_config(config_path: str = "config.yaml") -> ConfigLoader:
    """Get or create the singleton config loader."""
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


# Export for direct import
__all__ = ['ConfigLoader', 'get_config']
