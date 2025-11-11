"""
Athena Recursive AI - MoE Agentic Meta-LLM System

A hierarchical meta-reasoning system implementing Global Workspace Theory-inspired
coordination between specialized expert models.
"""

__version__ = "0.1.0"
__author__ = "zoadrazorro"

from .core.orchestrator import MetaOrchestrator
from .core.workspace import GlobalWorkspace
from .config.settings import AthenaConfig, get_config, set_config
from .communication.schemas import (
    UserQuery,
    ExpertType,
    RoutingStrategy,
    TaskComplexity,
)

__all__ = [
    "MetaOrchestrator",
    "GlobalWorkspace",
    "AthenaConfig",
    "get_config",
    "set_config",
    "UserQuery",
    "ExpertType",
    "RoutingStrategy",
    "TaskComplexity",
]
