"""
Configuration management for OpenMisty.
"""

import threading
import tomllib
from pathlib import Path
from typing import Dict, Optional, Literal

from pydantic import BaseModel, Field


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = get_project_root()
WORKSPACE_ROOT = PROJECT_ROOT / "workspace"
MODELS_ROOT = PROJECT_ROOT / "models"


class HardwareSettings(BaseModel):
    """Hardware optimization settings."""
    
    optimize_for_hardware: bool = Field(True, description="Enable hardware-aware optimization")
    max_memory_gb: float = Field(0, description="Maximum memory usage in GB (0 for auto-detect)")
    cpu_threads: int = Field(0, description="Number of CPU threads to use (0 for auto-detect)")
    mixed_precision: bool = Field(True, description="Enable mixed precision for faster inference")


class LLMSettings(BaseModel):
    """LLM configuration settings."""
    
    # Common settings
    mode: Literal["local", "api"] = Field("api", description="LLM mode (local or API)")
    model: str = Field(..., description="Model name or HuggingFace model ID")
    temperature: float = Field(0.0, description="Sampling temperature")
    
    # Local model settings
    quantization: Optional[Literal["4bit", "8bit", "none"]] = Field(
        "4bit", description="Quantization level for local models"
    )
    device: Optional[Literal["cuda", "cpu", "mps", "auto"]] = Field(
        "auto", description="Device to run the model on"
    )
    max_seq_length: Optional[int] = Field(
        4096, description="Maximum sequence length for local models"
    )
    
    # API settings
    api_type: Optional[Literal["openai", "azure", "anthropic"]] = Field(
        "openai", description="API type"
    )
    api_key: Optional[str] = Field("", description="API key")
    base_url: Optional[str] = Field("https://api.openai.com/v1", description="API base URL")
    api_version: Optional[str] = Field("", description="API version (required for Azure)")
    max_tokens: Optional[int] = Field(4096, description="Maximum number of tokens per request")


class AppConfig(BaseModel):
    """Application configuration."""
    
    llm: Dict[str, LLMSettings] = Field(..., description="LLM configuration")
    hardware: Optional[HardwareSettings] = Field(
        default_factory=HardwareSettings, description="Hardware settings"
    )


class Config:
    """Configuration singleton for OpenMisty."""
    
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config = None
                    self._load_initial_config()
                    self._initialized = True

    @staticmethod
    def _get_config_path() -> Path:
        """Get the path to the configuration file."""
        root = PROJECT_ROOT
        config_path = root / "config" / "config.toml"
        if config_path.exists():
            return config_path
        example_path = root / "config" / "config.example.toml"
        if example_path.exists():
            return example_path
        raise FileNotFoundError("No configuration file found in config directory")

    def _load_config(self) -> dict:
        """Load the configuration from the TOML file."""
        config_path = self._get_config_path()
        with config_path.open("rb") as f:
            return tomllib.load(f)

    def _load_initial_config(self):
        """Load the initial configuration."""
        raw_config = self._load_config()
        
        # Process LLM configuration
        base_llm = raw_config.get("llm", {})
        llm_overrides = {
            k: v for k, v in raw_config.get("llm", {}).items() if isinstance(v, dict)
        }
        
        # Create default LLM settings
        default_settings = LLMSettings(
            mode=base_llm.get("mode", "api"),
            model=base_llm.get("model", "gpt-4o"),
            temperature=base_llm.get("temperature", 0.0),
            # Local model settings
            quantization=base_llm.get("quantization", "4bit"),
            device=base_llm.get("device", "auto"),
            max_seq_length=base_llm.get("max_seq_length", 4096),
            # API settings
            api_type=base_llm.get("api_type", "openai"),
            api_key=base_llm.get("api_key", ""),
            base_url=base_llm.get("base_url", "https://api.openai.com/v1"),
            api_version=base_llm.get("api_version", ""),
            max_tokens=base_llm.get("max_tokens", 4096),
        )
        
        # Create LLM configuration dictionary
        llm_config = {
            "default": default_settings.model_dump(),
            **{
                name: {**default_settings.model_dump(), **override_config}
                for name, override_config in llm_overrides.items()
            },
        }
        
        # Process hardware configuration
        hardware_config = raw_config.get("hardware", {})
        
        # Create the final configuration
        config_dict = {
            "llm": llm_config,
            "hardware": hardware_config,
        }
        
        self._config = AppConfig(**config_dict)

    @property
    def llm(self) -> Dict[str, LLMSettings]:
        """Get the LLM configuration."""
        return self._config.llm
    
    @property
    def hardware(self) -> HardwareSettings:
        """Get the hardware configuration."""
        return self._config.hardware


# Create a singleton instance
config = Config() 