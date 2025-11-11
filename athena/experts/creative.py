"""
Creative Synthesis Specialist Expert for ideation and conceptual thinking.

Specializes in:
- Creative ideation and brainstorming
- Conceptual blending and metaphor
- Philosophical exploration
- Abstract reasoning
- Novel perspective generation
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


class CreativeExpert(ExpertModel):
    """
    Expert specializing in creative synthesis and conceptual thinking.

    Encourages divergent thinking, metaphorical reasoning, and
    philosophical exploration.
    """

    def build_system_prompt(self) -> str:
        """Build system prompt for creative expert."""
        return """You are a Creative Synthesis Specialist in a multi-expert AI system.

Your role is to:
- Generate creative ideas and novel perspectives
- Explore conceptual connections and metaphors
- Engage in philosophical and abstract reasoning
- Synthesize concepts from different domains
- Think divergently and consider unconventional approaches

Your strengths:
- Conceptual blending: combining ideas from different fields
- Metaphorical thinking: using analogies to illuminate complex concepts
- Philosophical depth: exploring implications and meaning
- Creative problem-solving: finding novel solutions

When responding:
1. Think beyond conventional boundaries
2. Draw connections between disparate concepts
3. Use metaphors and analogies to explain abstract ideas
4. Consider multiple perspectives and interpretations
5. Explore the philosophical implications

Be creative but grounded. If an idea is speculative, acknowledge it. Express confidence in your insights while being honest about their exploratory nature."""

    def format_query(self, query: ExpertQuery) -> List[Dict[str, str]]:
        """Format query for creative expert."""
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ]

        # Add context if provided
        if query.context:
            context_str = self._format_context(query.context)
            if context_str:
                messages.append({
                    "role": "user",
                    "content": f"Context:\n{context_str}"
                })

        # Add the main query with creative framing
        query_content = f"""Please explore this topic creatively and conceptually:

{query.prompt}

Feel free to:
- Draw analogies and metaphors
- Connect ideas from different domains
- Explore philosophical implications
- Consider unconventional perspectives
- Synthesize novel insights"""

        messages.append({"role": "user", "content": query_content})

        return messages

    def extract_reasoning_trace(self, response_text: str) -> Optional[List[str]]:
        """Extract creative reasoning process from response."""
        traces = []

        # Look for conceptual connections
        connection_patterns = [
            r'(?:similar to|like|analogous to|reminds me of)\s+(.+?)(?:\.|,|\n)',
            r'(?:consider|imagine|think of)\s+(.+?)(?:\.|,|\n)',
            r'(?:this connects to|relates to|parallels)\s+(.+?)(?:\.|,|\n)',
        ]

        for pattern in connection_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            traces.extend(matches)

        # Look for perspective shifts
        perspective_markers = [
            'from another angle', 'alternatively', 'consider this',
            'looking at it differently', 'another way to think'
        ]

        for marker in perspective_markers:
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
        """Calculate confidence for creative response."""
        confidence = 0.85  # Start slightly lower for creative/speculative content

        # Creative responses are inherently more speculative
        # Adjust based on content characteristics

        # Check for metaphors and analogies (positive indicator)
        metaphor_markers = ['like', 'similar to', 'analogous to', 'metaphor', 'analogy']
        metaphor_count = sum(
            1 for marker in metaphor_markers
            if marker in response_text.lower()
        )
        confidence = min(1.0, confidence + metaphor_count * 0.03)

        # Check for conceptual depth
        depth_markers = [
            'implies', 'suggests', 'reveals', 'illuminates',
            'deeper', 'fundamental', 'essence', 'nature of'
        ]
        depth_count = sum(
            1 for marker in depth_markers
            if marker in response_text.lower()
        )
        confidence = min(1.0, confidence + depth_count * 0.02)

        # Check for multiple perspectives
        perspective_markers = [
            'on one hand', 'on the other hand', 'alternatively',
            'another perspective', 'from this angle', 'conversely'
        ]
        perspective_count = sum(
            1 for marker in perspective_markers
            if marker in response_text.lower()
        )
        if perspective_count > 0:
            confidence = min(1.0, confidence + 0.05)

        # Reduce confidence for excessive uncertainty
        uncertainties = extract_uncertainty_markers(response_text)
        if len(uncertainties) > 3:
            confidence -= 0.15

        # Apply length-based confidence
        length_modifier = calculate_response_length_confidence(response_text)
        confidence *= length_modifier

        return max(0.0, min(1.0, confidence))

    def identify_uncertainties(self, response_text: str) -> List[str]:
        """Identify uncertainties in creative response."""
        uncertainties = extract_uncertainty_markers(response_text)

        # Creative content is inherently speculative
        # Check for acknowledgment of speculation
        speculation_markers = [
            'speculative', 'hypothetical', 'one interpretation',
            'possible', 'potential', 'could be seen as'
        ]

        for marker in speculation_markers:
            if marker.lower() in response_text.lower():
                uncertainties.append(f"Acknowledged speculation: {marker}")

        # Check for questions posed (exploration vs. assertion)
        question_count = response_text.count('?')
        if question_count > 2:
            uncertainties.append(f"Exploratory questions raised: {question_count}")

        # Check for caveats
        caveat_markers = ['however', 'but', 'although', 'that said', 'caveat']
        caveat_count = sum(
            1 for marker in caveat_markers
            if marker.lower() in response_text.lower()
        )
        if caveat_count > 1:
            uncertainties.append(f"Caveats presented: {caveat_count}")

        return uncertainties

    def _format_context(self, context: Dict) -> str:
        """Format context dict into a string."""
        parts = []

        if 'theme' in context:
            parts.append(f"Theme: {context['theme']}")

        if 'domains' in context:
            parts.append(f"Related domains: {context['domains']}")

        if 'perspectives' in context:
            parts.append(f"Perspectives to consider: {context['perspectives']}")

        if 'philosophical_context' in context:
            parts.append(f"Philosophical context: {context['philosophical_context']}")

        return "\n".join(parts)

    def _calculate_attention_weight(
        self,
        confidence: float,
        query: ExpertQuery,
        response: str
    ) -> float:
        """Calculate attention weight for creative responses."""
        # Use base calculation
        weight = super()._calculate_attention_weight(confidence, query, response)

        # Boost for philosophical or creative keywords in query
        creative_keywords = [
            'creative', 'philosophical', 'conceptual', 'abstract',
            'meaning', 'metaphor', 'perspective', 'insight'
        ]
        if any(kw in query.prompt.lower() for kw in creative_keywords):
            weight = min(1.0, weight * 1.2)

        # Boost for rich conceptual content
        if len(response.split()) > 150:  # Substantial creative content
            weight = min(1.0, weight * 1.1)

        return weight
