"""
Reasoning Specialist Expert for logical inference and analytical thinking.

Specializes in:
- Logical reasoning and inference
- Mathematical problem-solving
- Structured problem decomposition
- Step-by-step analytical thinking
"""

import re
from typing import Dict, List, Optional
from loguru import logger

from ..core.expert_base import (
    ExpertModel,
    extract_uncertainty_markers,
    calculate_response_length_confidence,
)
from ..communication.schemas import ExpertQuery, ExpertType


class ReasoningExpert(ExpertModel):
    """
    Expert specializing in logical reasoning and analytical thinking.

    Uses structured prompting to encourage step-by-step reasoning
    and mathematical precision.
    """

    def build_system_prompt(self) -> str:
        """Build system prompt for reasoning expert."""
        return """You are a Reasoning Specialist in a multi-expert AI system.

Your role is to:
- Perform logical inference and deductive reasoning
- Solve mathematical and analytical problems step-by-step
- Decompose complex problems into structured sub-problems
- Identify logical fallacies and inconsistencies
- Provide rigorous, evidence-based conclusions

Always structure your responses with clear reasoning steps:
1. Analyze the problem
2. Identify key assumptions and constraints
3. Apply logical principles or mathematical methods
4. Show your work step-by-step
5. Provide a clear conclusion

Be explicit about your certainty level. If you're unsure about any aspect, clearly state what you're uncertain about and why.

Focus on precision, clarity, and logical rigor."""

    def format_query(self, query: ExpertQuery) -> List[Dict[str, str]]:
        """Format query for reasoning expert."""
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

        # Add the main query
        query_content = f"""Please analyze this problem using step-by-step reasoning:

{query.prompt}

Provide your analysis with clear reasoning steps. If you make any assumptions, state them explicitly."""

        messages.append({"role": "user", "content": query_content})

        return messages

    def extract_reasoning_trace(self, response_text: str) -> Optional[List[str]]:
        """Extract reasoning steps from the response."""
        steps = []

        # Look for numbered steps
        numbered_pattern = r'^\d+[\.)]\s+(.+)$'
        for line in response_text.split('\n'):
            match = re.match(numbered_pattern, line.strip())
            if match:
                steps.append(match.group(1))

        # Look for bullet points
        if not steps:
            bullet_pattern = r'^[-*•]\s+(.+)$'
            for line in response_text.split('\n'):
                match = re.match(bullet_pattern, line.strip())
                if match:
                    steps.append(match.group(1))

        # Look for "Step N:" format
        if not steps:
            step_pattern = r'[Ss]tep\s+\d+[:)]\s+(.+?)(?=[Ss]tep\s+\d+|$)'
            steps = re.findall(step_pattern, response_text, re.DOTALL)
            steps = [s.strip() for s in steps]

        return steps if steps else None

    def calculate_confidence(
        self,
        response_text: str,
        query: ExpertQuery
    ) -> float:
        """Calculate confidence for reasoning response."""
        confidence = 1.0

        # Check for uncertainty markers
        uncertainties = extract_uncertainty_markers(response_text)
        confidence -= len(uncertainties) * 0.1

        # Check for reasoning structure
        reasoning_steps = self.extract_reasoning_trace(response_text)
        if reasoning_steps:
            # More steps generally indicate more thorough reasoning
            if len(reasoning_steps) >= 3:
                confidence = min(1.0, confidence + 0.1)
        else:
            # No clear reasoning structure reduces confidence
            confidence *= 0.8

        # Check for mathematical notation (indicates precision)
        if any(char in response_text for char in ['=', '+', '-', '*', '/', '^', '√']):
            confidence = min(1.0, confidence + 0.05)

        # Check for logical keywords
        logical_keywords = [
            'therefore', 'thus', 'hence', 'because', 'since',
            'given that', 'it follows', 'consequently', 'implies'
        ]
        keyword_count = sum(
            1 for kw in logical_keywords
            if kw in response_text.lower()
        )
        confidence = min(1.0, confidence + keyword_count * 0.02)

        # Apply length-based confidence
        length_modifier = calculate_response_length_confidence(response_text)
        confidence *= length_modifier

        return max(0.0, min(1.0, confidence))

    def identify_uncertainties(self, response_text: str) -> List[str]:
        """Identify uncertainties in reasoning response."""
        uncertainties = extract_uncertainty_markers(response_text)

        # Check for incomplete reasoning
        if "..." in response_text:
            uncertainties.append("Response contains incomplete sections")

        # Check for questions in response (indicates uncertainty)
        if "?" in response_text:
            questions = [
                line.strip() for line in response_text.split('\n')
                if '?' in line
            ]
            if questions:
                uncertainties.append(f"Unresolved questions: {len(questions)}")

        # Check for alternative possibilities
        alternative_markers = ['alternatively', 'or', 'could also', 'might also']
        for marker in alternative_markers:
            if marker in response_text.lower():
                uncertainties.append(f"Multiple possibilities considered ({marker})")

        return uncertainties

    def _format_context(self, context: Dict) -> str:
        """Format context dict into a string."""
        parts = []

        if 'background' in context:
            parts.append(f"Background: {context['background']}")

        if 'constraints' in context:
            parts.append(f"Constraints: {context['constraints']}")

        if 'assumptions' in context:
            parts.append(f"Known assumptions: {context['assumptions']}")

        if 'previous_steps' in context:
            parts.append(f"Previous analysis: {context['previous_steps']}")

        return "\n".join(parts)

    def _calculate_attention_weight(
        self,
        confidence: float,
        query: ExpertQuery,
        response: str
    ) -> float:
        """Calculate attention weight for reasoning responses."""
        # Use base calculation
        weight = super()._calculate_attention_weight(confidence, query, response)

        # Boost weight if response contains mathematical reasoning
        if any(char in response for char in ['=', '∴', '∵', '⇒', '∀', '∃']):
            weight = min(1.0, weight * 1.15)

        # Boost weight if response has clear structure
        if self.extract_reasoning_trace(response):
            weight = min(1.0, weight * 1.1)

        return weight
