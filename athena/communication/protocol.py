"""
Communication protocol implementation for Athena MoE system.

Handles message serialization, validation, and routing between the
Meta-Orchestrator and Expert models.
"""

import json
from typing import Any, Dict, Optional, Union
from loguru import logger

from .schemas import (
    ExpertQuery,
    ExpertResponse,
    UserQuery,
    QueryAnalysis,
    SynthesizedResponse,
    WorkspaceState,
    AttentionBroadcast,
    HealthCheck,
    MessageType,
)


class MessageProtocol:
    """
    Handles encoding and decoding of messages between system components.

    This class implements a JSON-based protocol with Pydantic validation
    for type safety and schema enforcement.
    """

    @staticmethod
    def encode_expert_query(query: ExpertQuery) -> Dict[str, Any]:
        """Encode an ExpertQuery into a JSON-serializable dict."""
        return {
            "type": MessageType.QUERY.value,
            "data": query.model_dump(mode="json")
        }

    @staticmethod
    def encode_expert_response(response: ExpertResponse) -> Dict[str, Any]:
        """Encode an ExpertResponse into a JSON-serializable dict."""
        return {
            "type": MessageType.RESPONSE.value,
            "data": response.model_dump(mode="json")
        }

    @staticmethod
    def encode_user_query(query: UserQuery) -> Dict[str, Any]:
        """Encode a UserQuery into a JSON-serializable dict."""
        return {
            "type": MessageType.QUERY.value,
            "data": query.model_dump(mode="json")
        }

    @staticmethod
    def encode_synthesized_response(response: SynthesizedResponse) -> Dict[str, Any]:
        """Encode a SynthesizedResponse into a JSON-serializable dict."""
        return {
            "type": MessageType.SYNTHESIS.value,
            "data": response.model_dump(mode="json")
        }

    @staticmethod
    def encode_attention_broadcast(broadcast: AttentionBroadcast) -> Dict[str, Any]:
        """Encode an AttentionBroadcast into a JSON-serializable dict."""
        return {
            "type": MessageType.ATTENTION_BROADCAST.value,
            "data": broadcast.model_dump(mode="json")
        }

    @staticmethod
    def encode_workspace_state(state: WorkspaceState) -> Dict[str, Any]:
        """Encode a WorkspaceState into a JSON-serializable dict."""
        return {
            "type": MessageType.WORKSPACE_UPDATE.value,
            "data": state.model_dump(mode="json")
        }

    @staticmethod
    def decode_expert_response(data: Dict[str, Any]) -> ExpertResponse:
        """Decode a dict into an ExpertResponse object."""
        try:
            return ExpertResponse(**data.get("data", data))
        except Exception as e:
            logger.error(f"Failed to decode expert response: {e}")
            raise ValueError(f"Invalid expert response format: {e}")

    @staticmethod
    def decode_user_query(data: Dict[str, Any]) -> UserQuery:
        """Decode a dict into a UserQuery object."""
        try:
            return UserQuery(**data.get("data", data))
        except Exception as e:
            logger.error(f"Failed to decode user query: {e}")
            raise ValueError(f"Invalid user query format: {e}")

    @staticmethod
    def serialize(message: Union[ExpertQuery, ExpertResponse, UserQuery,
                                  SynthesizedResponse, AttentionBroadcast,
                                  WorkspaceState]) -> str:
        """Serialize a message object to a JSON string."""
        if isinstance(message, ExpertQuery):
            data = MessageProtocol.encode_expert_query(message)
        elif isinstance(message, ExpertResponse):
            data = MessageProtocol.encode_expert_response(message)
        elif isinstance(message, UserQuery):
            data = MessageProtocol.encode_user_query(message)
        elif isinstance(message, SynthesizedResponse):
            data = MessageProtocol.encode_synthesized_response(message)
        elif isinstance(message, AttentionBroadcast):
            data = MessageProtocol.encode_attention_broadcast(message)
        elif isinstance(message, WorkspaceState):
            data = MessageProtocol.encode_workspace_state(message)
        else:
            raise ValueError(f"Unsupported message type: {type(message)}")

        return json.dumps(data, indent=2)

    @staticmethod
    def deserialize(message_str: str, expected_type: Optional[type] = None) -> Any:
        """
        Deserialize a JSON string to a message object.

        Args:
            message_str: JSON string to deserialize
            expected_type: Expected message class (for validation)

        Returns:
            Deserialized message object
        """
        try:
            data = json.loads(message_str)
            message_type = data.get("type")

            if message_type == MessageType.QUERY.value:
                result = MessageProtocol.decode_user_query(data)
            elif message_type == MessageType.RESPONSE.value:
                result = MessageProtocol.decode_expert_response(data)
            else:
                raise ValueError(f"Unknown message type: {message_type}")

            if expected_type and not isinstance(result, expected_type):
                raise ValueError(
                    f"Expected {expected_type.__name__}, got {type(result).__name__}"
                )

            return result

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            raise ValueError(f"Invalid JSON format: {e}")


class ConflictResolver:
    """
    Resolves conflicts when multiple experts provide different responses.

    Implements weighted confidence scoring and consensus-building strategies.
    """

    @staticmethod
    def weighted_confidence_merge(
        responses: list[ExpertResponse],
        weights: Optional[Dict[str, float]] = None
    ) -> tuple[str, float]:
        """
        Merge multiple expert responses using weighted confidence scores.

        Args:
            responses: List of expert responses to merge
            weights: Optional custom weights for each expert type

        Returns:
            Tuple of (merged_response, overall_confidence)
        """
        if not responses:
            return "", 0.0

        if len(responses) == 1:
            return responses[0].response, responses[0].confidence

        # Calculate weighted scores
        total_weight = 0.0
        weighted_responses = []

        for response in responses:
            # Use custom weight if provided, otherwise use attention weight
            expert_weight = (
                weights.get(response.expert_type.value, 1.0)
                if weights
                else response.attention_weight
            )

            score = response.confidence * expert_weight
            total_weight += score

            weighted_responses.append({
                "response": response.response,
                "score": score,
                "confidence": response.confidence,
                "expert": response.expert_type.value
            })

        # Sort by score
        weighted_responses.sort(key=lambda x: x["score"], reverse=True)

        # For now, use highest-scoring response
        # TODO: Implement more sophisticated merging strategies
        primary_response = weighted_responses[0]

        # Calculate overall confidence as weighted average
        overall_confidence = sum(
            r["score"] * r["confidence"] for r in weighted_responses
        ) / total_weight if total_weight > 0 else 0.0

        logger.info(
            f"Merged {len(responses)} responses. "
            f"Primary expert: {primary_response['expert']}, "
            f"Confidence: {overall_confidence:.2f}"
        )

        return primary_response["response"], overall_confidence

    @staticmethod
    def consensus_threshold(
        responses: list[ExpertResponse],
        threshold: float = 0.7
    ) -> Optional[str]:
        """
        Check if responses reach a consensus threshold.

        Args:
            responses: List of expert responses
            threshold: Minimum average confidence for consensus

        Returns:
            Consensus response if threshold met, None otherwise
        """
        if not responses:
            return None

        avg_confidence = sum(r.confidence for r in responses) / len(responses)

        if avg_confidence >= threshold:
            # Return response from most confident expert
            best_response = max(responses, key=lambda r: r.confidence)
            return best_response.response

        return None

    @staticmethod
    def detect_conflicts(responses: list[ExpertResponse]) -> list[str]:
        """
        Detect significant conflicts between expert responses.

        Args:
            responses: List of expert responses to analyze

        Returns:
            List of detected conflicts/disagreements
        """
        conflicts = []

        if len(responses) < 2:
            return conflicts

        # Check for low confidence across all experts
        avg_confidence = sum(r.confidence for r in responses) / len(responses)
        if avg_confidence < 0.5:
            conflicts.append(
                f"Low average confidence across experts: {avg_confidence:.2f}"
            )

        # Check for high uncertainty flags
        total_uncertainties = sum(len(r.uncertainty_flags) for r in responses)
        if total_uncertainties > len(responses):
            conflicts.append(
                f"High number of uncertainty flags: {total_uncertainties}"
            )

        # Check for large confidence variance
        confidences = [r.confidence for r in responses]
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        if variance > 0.1:
            conflicts.append(
                f"High confidence variance among experts: {variance:.3f}"
            )

        return conflicts
