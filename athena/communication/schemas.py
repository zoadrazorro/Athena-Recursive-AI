"""
Message schemas for inter-model communication in the Athena MoE system.

These Pydantic models define the structure for all messages exchanged between
the Meta-Orchestrator and Expert models, implementing a consciousness-inspired
communication protocol.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator


class ExpertType(str, Enum):
    """Types of expert models available in the system."""
    REASONING = "reasoning"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    MEMORY = "memory"


class MessageType(str, Enum):
    """Types of messages in the communication protocol."""
    QUERY = "query"
    RESPONSE = "response"
    DECOMPOSITION = "decomposition"
    SYNTHESIS = "synthesis"
    ATTENTION_BROADCAST = "attention_broadcast"
    WORKSPACE_UPDATE = "workspace_update"


class TaskComplexity(str, Enum):
    """Complexity classification for incoming queries."""
    TRIVIAL = "trivial"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT_REQUIRED = "expert_required"


class RoutingStrategy(str, Enum):
    """Strategies for routing queries to experts."""
    DIRECT = "direct"  # Orchestrator responds directly
    SINGLE_EXPERT = "single_expert"  # Route to one expert
    PARALLEL = "parallel"  # Consult multiple experts simultaneously
    SEQUENTIAL = "sequential"  # Chain experts in sequence


class ExpertQuery(BaseModel):
    """A query sent from the orchestrator to an expert model."""

    query_id: str = Field(..., description="Unique identifier for this query")
    expert_type: ExpertType = Field(..., description="Target expert type")
    prompt: str = Field(..., description="The specialized prompt for this expert")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context needed by the expert"
    )
    parent_query_id: Optional[str] = Field(
        None,
        description="ID of the original user query if this is a sub-task"
    )
    max_tokens: int = Field(default=2048, description="Maximum tokens for response")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class ExpertResponse(BaseModel):
    """Response from an expert model to the orchestrator."""

    query_id: str = Field(..., description="ID of the query being responded to")
    expert_type: ExpertType = Field(..., description="Type of expert responding")
    response: str = Field(..., description="The expert's response content")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Expert's confidence in this response"
    )
    reasoning_trace: Optional[List[str]] = Field(
        None,
        description="Step-by-step reasoning process (if applicable)"
    )
    uncertainty_flags: List[str] = Field(
        default_factory=list,
        description="Aspects of the response the expert is uncertain about"
    )
    attention_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Proposed attention weight for Global Workspace"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the response"
    )
    processing_time: Optional[float] = Field(
        None,
        description="Time taken to generate response (seconds)"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class UserQuery(BaseModel):
    """A query from the user to the system."""

    query_id: str = Field(..., description="Unique identifier for this query")
    content: str = Field(..., description="The user's query content")
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional additional context from the user"
    )
    conversation_history: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Previous conversation turns for context"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class QueryAnalysis(BaseModel):
    """Orchestrator's analysis of a user query."""

    query_id: str = Field(..., description="ID of the analyzed query")
    complexity: TaskComplexity = Field(..., description="Assessed complexity level")
    required_experts: List[ExpertType] = Field(
        default_factory=list,
        description="Experts needed to answer this query"
    )
    routing_strategy: RoutingStrategy = Field(
        ...,
        description="Strategy for routing to experts"
    )
    decomposed_tasks: List[str] = Field(
        default_factory=list,
        description="Sub-tasks if query was decomposed"
    )
    rationale: str = Field(
        ...,
        description="Reasoning behind the routing decision"
    )
    estimated_difficulty: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Estimated difficulty score"
    )


class SynthesizedResponse(BaseModel):
    """Final synthesized response from the orchestrator."""

    query_id: str = Field(..., description="ID of the original query")
    response: str = Field(..., description="Synthesized response content")
    expert_contributions: Dict[ExpertType, float] = Field(
        default_factory=dict,
        description="Contribution weight from each expert"
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Overall confidence in the synthesized response"
    )
    reasoning_summary: Optional[str] = Field(
        None,
        description="Summary of the reasoning process"
    )
    sources: List[ExpertType] = Field(
        default_factory=list,
        description="Expert sources consulted"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about synthesis"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class WorkspaceState(BaseModel):
    """
    Represents the current state of the Global Workspace.

    Inspired by Global Workspace Theory, this tracks what information
    is currently 'conscious' in the system.
    """

    active_queries: List[str] = Field(
        default_factory=list,
        description="Currently active query IDs"
    )
    attention_focus: Optional[ExpertType] = Field(
        None,
        description="Current primary attention focus"
    )
    workspace_contents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Information currently in the global workspace"
    )
    expert_activations: Dict[ExpertType, float] = Field(
        default_factory=dict,
        description="Current activation levels for each expert"
    )
    context_summary: str = Field(
        default="",
        description="Summary of current conversation context"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class AttentionBroadcast(BaseModel):
    """
    A broadcast message to the Global Workspace.

    Implements the GWT concept of broadcasting information to make it
    available to all cognitive processes.
    """

    broadcast_id: str = Field(..., description="Unique broadcast identifier")
    source: ExpertType = Field(..., description="Expert broadcasting the information")
    content: str = Field(..., description="Information being broadcast")
    attention_weight: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Importance weight for workspace allocation"
    )
    decay_rate: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Rate at which this information decays from workspace"
    )
    related_query_id: Optional[str] = Field(
        None,
        description="Query this broadcast is related to"
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class HealthCheck(BaseModel):
    """Health status of a model endpoint."""

    endpoint: str = Field(..., description="The endpoint being checked")
    model_name: str = Field(..., description="Name of the model")
    status: str = Field(..., description="Status: healthy, degraded, or unavailable")
    response_time: Optional[float] = Field(
        None,
        description="Response time in seconds"
    )
    error: Optional[str] = Field(None, description="Error message if unhealthy")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
