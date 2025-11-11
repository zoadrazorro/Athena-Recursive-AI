"""
Base class for expert models in the Athena MoE system.

Defines the interface and common functionality for all specialized
expert models in the system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
from loguru import logger

from ..communication.schemas import (
    ExpertQuery,
    ExpertResponse,
    ExpertType,
)
from ..communication.lm_studio_client import LMStudioClient
from ..config.settings import ModelEndpoint


class ExpertModel(ABC):
    """
    Abstract base class for expert models.

    Each expert specializes in a specific cognitive domain and implements
    domain-specific prompt engineering and response formatting.
    """

    def __init__(
        self,
        expert_type: ExpertType,
        endpoint_config: ModelEndpoint,
        client: Optional[LMStudioClient] = None
    ):
        """
        Initialize the expert model.

        Args:
            expert_type: Type of this expert
            endpoint_config: Configuration for the model endpoint
            client: Optional pre-configured LM Studio client
        """
        self.expert_type = expert_type
        self.config = endpoint_config

        if client:
            self.client = client
        else:
            self.client = LMStudioClient(
                base_url=endpoint_config.url,
                model_name=endpoint_config.model_name,
                timeout=endpoint_config.timeout,
            )

        self.query_count = 0
        self.total_tokens = 0
        self.average_confidence = 0.0

    @abstractmethod
    def build_system_prompt(self) -> str:
        """
        Build the system prompt for this expert.

        Returns:
            System prompt string defining the expert's role and capabilities
        """
        pass

    @abstractmethod
    def format_query(self, query: ExpertQuery) -> List[Dict[str, str]]:
        """
        Format the query into messages for the LLM.

        Args:
            query: ExpertQuery to process

        Returns:
            List of message dicts with 'role' and 'content'
        """
        pass

    @abstractmethod
    def extract_reasoning_trace(self, response_text: str) -> Optional[List[str]]:
        """
        Extract step-by-step reasoning from the response.

        Args:
            response_text: Raw response from the LLM

        Returns:
            List of reasoning steps, or None if not applicable
        """
        pass

    @abstractmethod
    def calculate_confidence(
        self,
        response_text: str,
        query: ExpertQuery
    ) -> float:
        """
        Calculate confidence score for this response.

        Args:
            response_text: Generated response
            query: Original query

        Returns:
            Confidence score between 0.0 and 1.0
        """
        pass

    @abstractmethod
    def identify_uncertainties(self, response_text: str) -> List[str]:
        """
        Identify aspects of the response the expert is uncertain about.

        Args:
            response_text: Generated response

        Returns:
            List of uncertainty descriptions
        """
        pass

    async def process_query(self, query: ExpertQuery) -> ExpertResponse:
        """
        Process a query and generate an expert response.

        This is the main entry point for querying an expert.

        Args:
            query: ExpertQuery to process

        Returns:
            ExpertResponse with the expert's answer
        """
        start_time = datetime.utcnow()

        try:
            # Format messages
            messages = self.format_query(query)

            # Call LLM
            logger.debug(
                f"{self.expert_type.value} expert processing query {query.query_id}"
            )

            response = await self.client.chat_completion(
                messages=messages,
                temperature=query.temperature,
                max_tokens=query.max_tokens,
                top_p=self.config.top_p,
            )

            # Extract response text
            response_text = await self.client.extract_response_text(response)

            # Process response
            confidence = self.calculate_confidence(response_text, query)
            reasoning_trace = self.extract_reasoning_trace(response_text)
            uncertainties = self.identify_uncertainties(response_text)

            # Calculate attention weight (can be overridden by subclasses)
            attention_weight = self._calculate_attention_weight(
                confidence, query, response_text
            )

            # Update statistics
            self.query_count += 1
            self.average_confidence = (
                (self.average_confidence * (self.query_count - 1) + confidence)
                / self.query_count
            )

            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()

            logger.info(
                f"{self.expert_type.value} expert completed query {query.query_id} "
                f"(confidence: {confidence:.2f}, time: {processing_time:.2f}s)"
            )

            return ExpertResponse(
                query_id=query.query_id,
                expert_type=self.expert_type,
                response=response_text,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                uncertainty_flags=uncertainties,
                attention_weight=attention_weight,
                processing_time=processing_time,
                metadata={
                    "model": self.config.model_name,
                    "temperature": query.temperature,
                    "expert_query_count": self.query_count,
                }
            )

        except Exception as e:
            logger.error(
                f"{self.expert_type.value} expert failed on query {query.query_id}: {e}"
            )

            # Return low-confidence error response
            return ExpertResponse(
                query_id=query.query_id,
                expert_type=self.expert_type,
                response=f"Error processing query: {str(e)}",
                confidence=0.0,
                uncertainty_flags=[f"Processing error: {str(e)}"],
                attention_weight=0.0,
                metadata={"error": str(e)}
            )

    def _calculate_attention_weight(
        self,
        confidence: float,
        query: ExpertQuery,
        response: str
    ) -> float:
        """
        Calculate attention weight for Global Workspace.

        Default implementation uses confidence as primary factor.
        Subclasses can override for domain-specific weighting.

        Args:
            confidence: Calculated confidence score
            query: Original query
            response: Generated response

        Returns:
            Attention weight between 0.0 and 1.0
        """
        # Base weight on confidence
        weight = confidence

        # Boost for queries explicitly targeting this expert
        if query.expert_type == self.expert_type:
            weight = min(1.0, weight * 1.2)

        # Penalize very short responses (likely errors or uncertainty)
        if len(response.split()) < 20:
            weight *= 0.7

        return max(0.0, min(1.0, weight))

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about this expert's performance.

        Returns:
            Dict with performance metrics
        """
        return {
            "expert_type": self.expert_type.value,
            "model": self.config.model_name,
            "query_count": self.query_count,
            "average_confidence": self.average_confidence,
            "endpoint": self.config.url,
        }

    async def health_check(self) -> bool:
        """
        Check if this expert's endpoint is healthy.

        Returns:
            True if healthy, False otherwise
        """
        try:
            health = await self.client.health_check()
            return health.status == "healthy"
        except Exception as e:
            logger.error(f"{self.expert_type.value} health check failed: {e}")
            return False


def extract_uncertainty_markers(text: str) -> List[str]:
    """
    Helper function to extract common uncertainty markers from text.

    Args:
        text: Text to analyze

    Returns:
        List of identified uncertainty markers
    """
    uncertainties = []

    uncertainty_phrases = [
        "i'm not sure",
        "i don't know",
        "possibly",
        "maybe",
        "might be",
        "could be",
        "unclear",
        "uncertain",
        "cannot determine",
        "hard to say",
        "difficult to",
        "not certain",
        "not confident",
    ]

    text_lower = text.lower()

    for phrase in uncertainty_phrases:
        if phrase in text_lower:
            # Find context around the phrase
            idx = text_lower.find(phrase)
            start = max(0, idx - 20)
            end = min(len(text), idx + len(phrase) + 30)
            context = text[start:end].strip()
            uncertainties.append(context)

    return uncertainties


def calculate_response_length_confidence(text: str) -> float:
    """
    Calculate confidence modifier based on response length.

    Args:
        text: Response text

    Returns:
        Confidence modifier (0.5 to 1.0)
    """
    word_count = len(text.split())

    # Very short responses indicate low confidence
    if word_count < 10:
        return 0.5
    elif word_count < 20:
        return 0.7
    elif word_count < 50:
        return 0.9
    else:
        return 1.0
