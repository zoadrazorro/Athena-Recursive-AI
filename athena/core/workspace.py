"""
Global Workspace implementation for Athena MoE system.

Implements a computational model inspired by Global Workspace Theory (GWT),
where information competing for attention is broadcast to a shared workspace
accessible to all cognitive processes (expert models).
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, field
from loguru import logger

from ..communication.schemas import (
    AttentionBroadcast,
    WorkspaceState,
    ExpertType,
    ExpertResponse,
)
from ..config.settings import GWTConfig


@dataclass
class WorkspaceItem:
    """
    An item in the Global Workspace.

    Represents a piece of information that has won the competition
    for conscious attention and is being broadcast system-wide.
    """

    broadcast_id: str
    source: ExpertType
    content: str
    attention_weight: float
    decay_rate: float
    created_at: datetime
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    related_query_id: Optional[str] = None

    def update_access(self) -> None:
        """Update access tracking when this item is retrieved."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1

    def get_current_weight(self, now: Optional[datetime] = None) -> float:
        """
        Calculate current attention weight with decay.

        Args:
            now: Current time (defaults to utcnow)

        Returns:
            Decayed attention weight
        """
        if now is None:
            now = datetime.utcnow()

        elapsed = (now - self.created_at).total_seconds()
        # Simple exponential decay
        decayed_weight = self.attention_weight * (self.decay_rate ** (elapsed / 60))

        return max(0.0, min(1.0, decayed_weight))

    def is_expired(self, threshold: float = 0.1, max_age: int = 3600) -> bool:
        """
        Check if this item has expired from the workspace.

        Args:
            threshold: Minimum weight to remain active
            max_age: Maximum age in seconds

        Returns:
            True if expired, False otherwise
        """
        age = (datetime.utcnow() - self.created_at).total_seconds()

        return self.get_current_weight() < threshold or age > max_age


class GlobalWorkspace:
    """
    Global Workspace for conscious information processing.

    Implements GWT-inspired attention mechanism where expert responses
    compete for limited workspace capacity. High-attention information
    is broadcast and made available to all experts.
    """

    def __init__(self, config: GWTConfig):
        """
        Initialize the Global Workspace.

        Args:
            config: GWT configuration parameters
        """
        self.config = config
        self.workspace: Dict[str, WorkspaceItem] = {}
        self.attention_history: deque = deque(maxlen=100)
        self.expert_activations: Dict[ExpertType, float] = {
            expert: 0.0 for expert in ExpertType
        }

    def broadcast(self, broadcast: AttentionBroadcast) -> bool:
        """
        Attempt to broadcast information to the global workspace.

        Information must compete for limited workspace slots based on
        attention weight. Lower-weight items may be evicted.

        Args:
            broadcast: AttentionBroadcast message

        Returns:
            True if broadcast succeeded, False if rejected
        """
        # Create workspace item
        item = WorkspaceItem(
            broadcast_id=broadcast.broadcast_id,
            source=broadcast.source,
            content=broadcast.content,
            attention_weight=broadcast.attention_weight,
            decay_rate=broadcast.decay_rate,
            created_at=datetime.utcnow(),
            related_query_id=broadcast.related_query_id,
        )

        # Check if attention weight meets threshold
        if item.attention_weight < self.config.attention_threshold:
            logger.debug(
                f"Broadcast rejected: attention weight {item.attention_weight:.2f} "
                f"below threshold {self.config.attention_threshold:.2f}"
            )
            return False

        # Remove expired items
        self._cleanup_expired()

        # Check if workspace is full
        if len(self.workspace) >= self.config.workspace_size:
            if self.config.enable_competition:
                # Find lowest-weight item
                current_weights = {
                    bid: item.get_current_weight()
                    for bid, item in self.workspace.items()
                }
                min_broadcast_id = min(current_weights, key=current_weights.get)
                min_weight = current_weights[min_broadcast_id]

                # Compete for workspace slot
                if item.attention_weight > min_weight:
                    # Evict lowest-weight item
                    evicted = self.workspace.pop(min_broadcast_id)
                    logger.info(
                        f"Evicted broadcast {min_broadcast_id} (weight: {min_weight:.2f}) "
                        f"for new broadcast {broadcast.broadcast_id} (weight: {item.attention_weight:.2f})"
                    )
                else:
                    logger.debug(
                        f"Broadcast rejected: weight {item.attention_weight:.2f} "
                        f"insufficient to compete (min in workspace: {min_weight:.2f})"
                    )
                    return False
            else:
                logger.warning(
                    f"Workspace full and competition disabled. "
                    f"Broadcast {broadcast.broadcast_id} rejected."
                )
                return False

        # Add to workspace
        self.workspace[broadcast.broadcast_id] = item

        # Update expert activation
        self.expert_activations[broadcast.source] = max(
            self.expert_activations.get(broadcast.source, 0.0),
            broadcast.attention_weight
        )

        # Record in history
        self.attention_history.append({
            "broadcast_id": broadcast.broadcast_id,
            "source": broadcast.source.value,
            "weight": broadcast.attention_weight,
            "timestamp": datetime.utcnow().isoformat(),
        })

        logger.info(
            f"Broadcast accepted: {broadcast.broadcast_id} from {broadcast.source.value} "
            f"(weight: {broadcast.attention_weight:.2f}, workspace: {len(self.workspace)}/{self.config.workspace_size})"
        )

        return True

    def retrieve(
        self,
        query_id: Optional[str] = None,
        expert_type: Optional[ExpertType] = None,
        min_weight: Optional[float] = None
    ) -> List[WorkspaceItem]:
        """
        Retrieve items from the workspace.

        Args:
            query_id: Filter by related query ID
            expert_type: Filter by source expert
            min_weight: Minimum current attention weight

        Returns:
            List of matching WorkspaceItems
        """
        results = []

        for item in self.workspace.values():
            # Apply filters
            if query_id and item.related_query_id != query_id:
                continue

            if expert_type and item.source != expert_type:
                continue

            current_weight = item.get_current_weight()
            if min_weight and current_weight < min_weight:
                continue

            # Update access tracking
            item.update_access()
            results.append(item)

        # Sort by current attention weight (descending)
        results.sort(key=lambda x: x.get_current_weight(), reverse=True)

        return results

    def get_state(self) -> WorkspaceState:
        """
        Get the current state of the global workspace.

        Returns:
            WorkspaceState object
        """
        # Get active query IDs
        active_queries = list(set(
            item.related_query_id
            for item in self.workspace.values()
            if item.related_query_id
        ))

        # Determine primary attention focus
        if self.expert_activations:
            focus = max(self.expert_activations, key=self.expert_activations.get)
        else:
            focus = None

        # Build workspace contents summary
        contents = [
            {
                "broadcast_id": item.broadcast_id,
                "source": item.source.value,
                "weight": item.get_current_weight(),
                "age": (datetime.utcnow() - item.created_at).total_seconds(),
                "content_preview": item.content[:100] + "..." if len(item.content) > 100 else item.content,
            }
            for item in sorted(
                self.workspace.values(),
                key=lambda x: x.get_current_weight(),
                reverse=True
            )
        ]

        # Generate context summary
        if contents:
            context_summary = f"Workspace contains {len(contents)} active broadcasts. "
            context_summary += f"Primary focus: {focus.value if focus else 'none'}. "
            context_summary += f"Top item: {contents[0]['source']} (weight: {contents[0]['weight']:.2f})"
        else:
            context_summary = "Workspace empty."

        return WorkspaceState(
            active_queries=active_queries,
            attention_focus=focus,
            workspace_contents=contents,
            expert_activations=self.expert_activations.copy(),
            context_summary=context_summary,
        )

    def update_expert_activation(
        self,
        expert_type: ExpertType,
        activation: float
    ) -> None:
        """
        Update the activation level for an expert.

        Args:
            expert_type: Expert to update
            activation: New activation level (0.0 to 1.0)
        """
        activation = max(0.0, min(1.0, activation))
        self.expert_activations[expert_type] = activation

        logger.debug(f"Updated {expert_type.value} activation to {activation:.2f}")

    def decay_activations(self, decay_rate: float = 0.95) -> None:
        """
        Apply decay to all expert activations.

        Args:
            decay_rate: Decay multiplier (0.0 to 1.0)
        """
        for expert in self.expert_activations:
            self.expert_activations[expert] *= decay_rate

    def _cleanup_expired(self) -> None:
        """Remove expired items from the workspace."""
        expired = [
            bid for bid, item in self.workspace.items()
            if item.is_expired(
                threshold=self.config.attention_threshold * 0.5,
                max_age=3600
            )
        ]

        for bid in expired:
            removed = self.workspace.pop(bid)
            logger.debug(
                f"Removed expired broadcast {bid} from {removed.source.value} "
                f"(final weight: {removed.get_current_weight():.2f})"
            )

    def clear(self) -> None:
        """Clear all workspace contents."""
        count = len(self.workspace)
        self.workspace.clear()
        self.expert_activations = {expert: 0.0 for expert in ExpertType}
        logger.info(f"Cleared workspace ({count} items removed)")

    def get_attention_summary(self) -> Dict[str, Any]:
        """
        Get a summary of attention allocation.

        Returns:
            Dict with attention statistics
        """
        total_weight = sum(
            item.get_current_weight() for item in self.workspace.values()
        )

        expert_weights = {expert: 0.0 for expert in ExpertType}
        for item in self.workspace.values():
            expert_weights[item.source] += item.get_current_weight()

        return {
            "total_items": len(self.workspace),
            "total_weight": total_weight,
            "expert_weights": {
                expert.value: weight
                for expert, weight in expert_weights.items()
            },
            "expert_activations": {
                expert.value: activation
                for expert, activation in self.expert_activations.items()
            },
            "capacity_used": f"{len(self.workspace)}/{self.config.workspace_size}",
        }
