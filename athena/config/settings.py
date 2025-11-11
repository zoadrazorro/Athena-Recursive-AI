"""
Configuration management for Athena MoE system.

Loads settings from environment variables and YAML configuration files.
"""

import os
from pathlib import Path
from typing import Dict, Optional
import yaml
from pydantic import BaseModel, Field, validator
from dotenv import load_dotenv
from loguru import logger


# Load environment variables
load_dotenv()


class ModelEndpoint(BaseModel):
    """Configuration for a single model endpoint."""

    url: str = Field(..., description="API endpoint URL")
    model_name: str = Field(..., description="Name of the model")
    gpu_id: int = Field(default=0, description="GPU device ID (0 or 1)")
    max_tokens: int = Field(default=2048, description="Maximum tokens per request")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    timeout: int = Field(default=120, description="Request timeout in seconds")


class GWTConfig(BaseModel):
    """Global Workspace Theory configuration."""

    attention_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum attention weight for workspace entry"
    )
    workspace_size: int = Field(
        default=5,
        ge=1,
        description="Maximum items in global workspace"
    )
    broadcast_decay: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Decay rate for workspace broadcasts"
    )
    enable_competition: bool = Field(
        default=True,
        description="Enable attention competition between experts"
    )


class ExpertRoutingConfig(BaseModel):
    """Configuration for expert routing and consultation."""

    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel expert consultation"
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for accepting responses"
    )
    max_retries: int = Field(
        default=3,
        ge=1,
        description="Maximum retries per expert"
    )
    timeout_per_expert: int = Field(
        default=60,
        ge=10,
        description="Timeout for each expert in seconds"
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(
        default="INFO",
        description="Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    file: Optional[str] = Field(
        default="athena.log",
        description="Log file path (None for no file logging)"
    )
    format: str = Field(
        default="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        description="Log format string"
    )
    rotation: str = Field(
        default="10 MB",
        description="Log rotation size"
    )


class AthenaConfig(BaseModel):
    """Main configuration for the Athena MoE system."""

    # Model endpoints
    orchestrator: ModelEndpoint
    reasoning_expert: ModelEndpoint
    creative_expert: ModelEndpoint
    technical_expert: ModelEndpoint
    memory_expert: ModelEndpoint

    # System configuration
    gwt: GWTConfig = Field(default_factory=GWTConfig)
    routing: ExpertRoutingConfig = Field(default_factory=ExpertRoutingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # Context configuration
    max_context_length: int = Field(default=8192, ge=1024)
    conversation_history_size: int = Field(default=10, ge=1)

    @classmethod
    def from_env(cls) -> "AthenaConfig":
        """
        Load configuration from environment variables.

        Returns:
            AthenaConfig instance populated from environment
        """
        return cls(
            orchestrator=ModelEndpoint(
                url=os.getenv("ORCHESTRATOR_URL", "http://localhost:1234/v1"),
                model_name=os.getenv("ORCHESTRATOR_MODEL", "qwen2.5-14b-instruct"),
                gpu_id=int(os.getenv("ORCHESTRATOR_GPU", "0")),
                max_tokens=int(os.getenv("MAX_TOKENS", "2048")),
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                top_p=float(os.getenv("TOP_P", "0.9")),
                timeout=int(os.getenv("TIMEOUT", "120")),
            ),
            reasoning_expert=ModelEndpoint(
                url=os.getenv("REASONING_EXPERT_URL", "http://localhost:1235/v1"),
                model_name=os.getenv("REASONING_MODEL", "phi-3.5-mini-instruct"),
                gpu_id=int(os.getenv("REASONING_GPU", "0")),  # GPU 1 for logical tasks
            ),
            creative_expert=ModelEndpoint(
                url=os.getenv("CREATIVE_EXPERT_URL", "http://localhost:1237/v1"),
                model_name=os.getenv("CREATIVE_MODEL", "mistral-7b-instruct"),
                gpu_id=int(os.getenv("CREATIVE_GPU", "1")),  # GPU 2 for creative tasks
            ),
            technical_expert=ModelEndpoint(
                url=os.getenv("TECHNICAL_EXPERT_URL", "http://localhost:1238/v1"),
                model_name=os.getenv("TECHNICAL_MODEL", "codeqwen-7b-instruct"),
                gpu_id=int(os.getenv("TECHNICAL_GPU", "1")),  # GPU 2 for code tasks
            ),
            memory_expert=ModelEndpoint(
                url=os.getenv("MEMORY_EXPERT_URL", "http://localhost:1236/v1"),
                model_name=os.getenv("MEMORY_MODEL", "llama-3.1-8b-instruct"),
                gpu_id=int(os.getenv("MEMORY_GPU", "0")),  # GPU 1 for context-heavy tasks
            ),
            gwt=GWTConfig(
                attention_threshold=float(os.getenv("GWT_ATTENTION_THRESHOLD", "0.6")),
                workspace_size=int(os.getenv("GWT_WORKSPACE_SIZE", "5")),
                broadcast_decay=float(os.getenv("GWT_BROADCAST_DECAY", "0.9")),
            ),
            routing=ExpertRoutingConfig(
                enable_parallel=os.getenv("ENABLE_PARALLEL_CONSULTATION", "true").lower() == "true",
                confidence_threshold=float(os.getenv("CONFIDENCE_THRESHOLD", "0.7")),
                max_retries=int(os.getenv("MAX_EXPERT_RETRIES", "3")),
            ),
            logging=LoggingConfig(
                level=os.getenv("LOG_LEVEL", "INFO"),
                file=os.getenv("LOG_FILE", "athena.log"),
            ),
            max_context_length=int(os.getenv("MAX_CONTEXT_LENGTH", "8192")),
        )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "AthenaConfig":
        """
        Load configuration from a YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            AthenaConfig instance
        """
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def save_yaml(self, config_path: Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            config_path: Path to save the YAML file
        """
        config_dict = self.model_dump(mode="json")

        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Saved configuration to {config_path}")

    def get_expert_endpoints(self) -> Dict[str, ModelEndpoint]:
        """
        Get a dictionary of all expert endpoints.

        Returns:
            Dict mapping expert names to ModelEndpoint configs
        """
        return {
            "reasoning": self.reasoning_expert,
            "creative": self.creative_expert,
            "technical": self.technical_expert,
            "memory": self.memory_expert,
        }


def setup_logging(config: LoggingConfig) -> None:
    """
    Configure logging based on the LoggingConfig.

    Args:
        config: Logging configuration
    """
    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        lambda msg: print(msg, end=""),
        level=config.level,
        format=config.format,
        colorize=True,
    )

    # Add file logger if specified
    if config.file:
        logger.add(
            config.file,
            level=config.level,
            format=config.format,
            rotation=config.rotation,
            compression="zip",
        )

    logger.info(f"Logging configured: level={config.level}, file={config.file}")


# Global configuration instance
_config: Optional[AthenaConfig] = None


def get_config() -> AthenaConfig:
    """
    Get the global configuration instance.

    Loads from environment variables on first call.

    Returns:
        AthenaConfig instance
    """
    global _config

    if _config is None:
        _config = AthenaConfig.from_env()
        setup_logging(_config.logging)

    return _config


def set_config(config: AthenaConfig) -> None:
    """
    Set the global configuration instance.

    Args:
        config: AthenaConfig to set as global
    """
    global _config
    _config = config
    setup_logging(config.logging)
