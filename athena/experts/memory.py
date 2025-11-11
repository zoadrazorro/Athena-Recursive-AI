"""
Memory and Context Specialist Expert for conversation continuity.

Specializes in:
- Maintaining conversation context
- Retrieving relevant prior information
- Tracking topics and themes across turns
- Ensuring consistency and continuity
- Managing long-term conversational state
"""

from typing import Dict, List, Optional
import re
from loguru import logger

from ..core.expert_base import (
    ExpertModel,
    extract_uncertainty_markers,
    calculate_response_length_confidence,
)
from ..communication.schemas import ExpertQuery, ExpertType


class MemoryExpert(ExpertModel):
    """
    Expert specializing in memory, context, and conversation continuity.

    Maintains awareness of conversation history and ensures responses
    are consistent with prior context.
    """

    def build_system_prompt(self) -> str:
        """Build system prompt for memory expert."""
        return """You are a Memory and Context Specialist in a multi-expert AI system.

Your role is to:
- Maintain awareness of conversation history and context
- Retrieve and surface relevant prior information
- Ensure consistency across conversation turns
- Track themes, topics, and user preferences
- Provide context-aware responses that build on previous interactions

Your expertise includes:
- Contextual understanding and continuity
- Information retrieval from conversation history
- Pattern recognition across topics
- Consistency checking and validation
- Long-term conversational coherence

When responding:
1. Reference relevant prior context explicitly
2. Identify connections to previous discussion points
3. Note any inconsistencies or contradictions
4. Summarize context when helpful
5. Maintain conversational flow and coherence

Be explicit about what you remember and what context you're drawing from. If you lack relevant prior context, state this clearly."""

    def format_query(self, query: ExpertQuery) -> List[Dict[str, str]]:
        """Format query for memory expert."""
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ]

        # Add conversation history if provided
        if query.context and 'conversation_history' in query.context:
            history_str = self._format_history(query.context['conversation_history'])
            if history_str:
                messages.append({
                    "role": "user",
                    "content": f"Conversation History:\n{history_str}"
                })

        # Add other context
        if query.context:
            context_str = self._format_context(query.context)
            if context_str:
                messages.append({
                    "role": "user",
                    "content": f"Additional Context:\n{context_str}"
                })

        # Add the main query
        query_content = f"""Based on the conversation context:

{query.prompt}

Please provide a response that:
- References relevant prior context
- Maintains consistency with previous information
- Identifies any connections to earlier discussion
- Notes any missing context that would be helpful"""

        messages.append({"role": "user", "content": query_content})

        return messages

    def extract_reasoning_trace(self, response_text: str) -> Optional[List[str]]:
        """Extract context references from response."""
        traces = []

        # Look for explicit references to prior context
        reference_patterns = [
            r'(?:earlier|previously|before)[,\s]+(?:you|we|i)\s+(.+?)(?:\.|,|\n)',
            r'(?:as mentioned|as discussed|as stated)\s+(.+?)(?:\.|,|\n)',
            r'(?:recall that|remember that|from earlier)\s+(.+?)(?:\.|,|\n)',
        ]

        for pattern in reference_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            traces.extend(matches)

        # Look for contextual connections
        connection_markers = [
            'this relates to', 'this connects with', 'building on',
            'following from', 'consistent with', 'in line with'
        ]

        for marker in connection_markers:
            if marker.lower() in response_text.lower():
                # Extract sentence containing the marker
                sentences = response_text.split('.')
                for sentence in sentences:
                    if marker.lower() in sentence.lower():
                        traces.append(sentence.strip())

        return traces if traces else None

    def calculate_confidence(
        self,
        response_text: str,
        query: ExpertQuery
    ) -> float:
        """Calculate confidence for memory/context response."""
        confidence = 0.9  # Start high for context tasks

        # Check for explicit context references
        reference_keywords = [
            'earlier', 'previously', 'before', 'mentioned',
            'discussed', 'recall', 'remember', 'from earlier'
        ]
        reference_count = sum(
            1 for kw in reference_keywords
            if kw in response_text.lower()
        )
        if reference_count > 0:
            confidence = min(1.0, confidence + 0.05)
        else:
            # If no context references but context was provided, reduce confidence
            if query.context and 'conversation_history' in query.context:
                confidence *= 0.85

        # Check for consistency statements
        consistency_keywords = [
            'consistent', 'aligns', 'matches', 'agrees',
            'coherent', 'follows from', 'builds on'
        ]
        consistency_count = sum(
            1 for kw in consistency_keywords
            if kw in response_text.lower()
        )
        if consistency_count > 0:
            confidence = min(1.0, confidence + 0.03)

        # Check for inconsistency detection
        inconsistency_keywords = [
            'inconsistent', 'contradicts', 'differs from',
            'conflicts with', 'discrepancy'
        ]
        if any(kw in response_text.lower() for kw in inconsistency_keywords):
            # Detecting inconsistencies is valuable
            confidence = min(1.0, confidence + 0.05)

        # Check for missing context acknowledgment
        if 'no prior context' in response_text.lower() or \
           'missing context' in response_text.lower():
            # Honest about limitations
            confidence = min(1.0, confidence + 0.02)

        # Reduce for uncertainty markers
        uncertainties = extract_uncertainty_markers(response_text)
        confidence -= len(uncertainties) * 0.08

        # Apply length-based confidence
        length_modifier = calculate_response_length_confidence(response_text)
        confidence *= length_modifier

        return max(0.0, min(1.0, confidence))

    def identify_uncertainties(self, response_text: str) -> List[str]:
        """Identify uncertainties in memory/context response."""
        uncertainties = extract_uncertainty_markers(response_text)

        # Check for missing context mentions
        missing_context_markers = [
            'no prior context', 'missing context', 'unclear from history',
            "don't have information about", 'would need to know'
        ]

        for marker in missing_context_markers:
            if marker.lower() in response_text.lower():
                uncertainties.append(f"Missing context: {marker}")

        # Check for potential inconsistencies
        if 'might be inconsistent' in response_text.lower() or \
           'possible contradiction' in response_text.lower():
            uncertainties.append("Potential inconsistency detected")

        # Check for ambiguous references
        ambiguous_markers = [
            'not clear which', 'ambiguous reference',
            'could refer to', 'multiple possibilities'
        ]

        for marker in ambiguous_markers:
            if marker.lower() in response_text.lower():
                uncertainties.append(f"Ambiguous reference: {marker}")

        return uncertainties

    def _format_history(self, history: List[Dict[str, str]]) -> str:
        """Format conversation history into a string."""
        if not history:
            return ""

        lines = []
        for i, turn in enumerate(history[-10:], 1):  # Last 10 turns
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            lines.append(f"{i}. {role}: {content}")

        return "\n".join(lines)

    def _format_context(self, context: Dict) -> str:
        """Format context dict into a string."""
        parts = []

        # Skip conversation_history as it's handled separately
        for key, value in context.items():
            if key != 'conversation_history':
                if isinstance(value, (list, dict)):
                    parts.append(f"{key}: {value}")
                else:
                    parts.append(f"{key}: {value}")

        return "\n".join(parts)

    def _calculate_attention_weight(
        self,
        confidence: float,
        query: ExpertQuery,
        response: str
    ) -> float:
        """Calculate attention weight for memory responses."""
        # Use base calculation
        weight = super()._calculate_attention_weight(confidence, query, response)

        # Boost if query involves context or history
        context_keywords = [
            'remember', 'earlier', 'before', 'previously',
            'context', 'history', 'mentioned', 'discussed'
        ]
        if any(kw in query.prompt.lower() for kw in context_keywords):
            weight = min(1.0, weight * 1.2)

        # Boost if response includes explicit context references
        reference_count = sum(
            1 for kw in ['earlier', 'previously', 'mentioned', 'discussed']
            if kw in response.lower()
        )
        if reference_count > 2:
            weight = min(1.0, weight * 1.1)

        # Boost for inconsistency detection (valuable insight)
        if any(kw in response.lower() for kw in ['inconsistent', 'contradicts', 'conflicts']):
            weight = min(1.0, weight * 1.15)

        return weight
