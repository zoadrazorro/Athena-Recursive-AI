"""
Technical Implementation Specialist Expert for code and engineering.

Specializes in:
- Code generation and optimization
- Software architecture and design
- Debugging and problem diagnosis
- Technical documentation
- Best practices and patterns
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


class TechnicalExpert(ExpertModel):
    """
    Expert specializing in technical implementation and software engineering.

    Focuses on code quality, architectural soundness, and practical
    implementation details.
    """

    def build_system_prompt(self) -> str:
        """Build system prompt for technical expert."""
        return """You are a Technical Implementation Specialist in a multi-expert AI system.

Your role is to:
- Generate clean, efficient, well-documented code
- Design robust software architectures
- Debug and diagnose technical problems
- Explain technical concepts clearly
- Recommend best practices and design patterns

Your expertise spans:
- Multiple programming languages and frameworks
- Software architecture and system design
- Algorithms and data structures
- Code optimization and performance
- Testing and debugging strategies

When responding:
1. Provide practical, working solutions
2. Write clean, readable, well-commented code
3. Explain your technical decisions
4. Consider edge cases and error handling
5. Follow language/framework best practices

Be precise and thorough. If you're uncertain about a technical detail, explicitly state what you'd need to verify. Prefer working, tested approaches over experimental ones."""

    def format_query(self, query: ExpertQuery) -> List[Dict[str, str]]:
        """Format query for technical expert."""
        messages = [
            {"role": "system", "content": self.build_system_prompt()}
        ]

        # Add technical context if provided
        if query.context:
            context_str = self._format_context(query.context)
            if context_str:
                messages.append({
                    "role": "user",
                    "content": f"Technical Context:\n{context_str}"
                })

        # Add the main query with technical framing
        query_content = f"""Technical Task:

{query.prompt}

Please provide:
1. A clear technical solution
2. Well-commented code (if applicable)
3. Explanation of your approach
4. Considerations for edge cases or potential issues
5. Any assumptions you're making"""

        messages.append({"role": "user", "content": query_content})

        return messages

    def extract_reasoning_trace(self, response_text: str) -> Optional[List[str]]:
        """Extract technical reasoning from response."""
        traces = []

        # Look for implementation steps
        step_patterns = [
            r'(?:first|step 1)[,:]?\s+(.+?)(?:\n|$)',
            r'(?:next|then|step 2)[,:]?\s+(.+?)(?:\n|$)',
            r'(?:finally|lastly|step 3)[,:]?\s+(.+?)(?:\n|$)',
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            traces.extend(matches)

        # Look for technical decision points
        decision_markers = [
            'i chose', 'we use', 'this approach', 'the reason',
            'because', 'to ensure', 'this allows'
        ]

        for marker in decision_markers:
            pattern = rf'{marker}\s+(.+?)(?:\.|;|\n)'
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            traces.extend(matches[:3])  # Limit to avoid noise

        return traces if traces else None

    def calculate_confidence(
        self,
        response_text: str,
        query: ExpertQuery
    ) -> float:
        """Calculate confidence for technical response."""
        confidence = 1.0

        # Check for code blocks (indicates concrete implementation)
        code_block_count = len(re.findall(r'```[\s\S]*?```', response_text))
        if code_block_count > 0:
            confidence = min(1.0, confidence + 0.1)
        else:
            # If query seems to request code but none provided, reduce confidence
            code_keywords = ['implement', 'write', 'code', 'function', 'class']
            if any(kw in query.prompt.lower() for kw in code_keywords):
                confidence *= 0.8

        # Check for technical specificity
        technical_indicators = [
            'function', 'class', 'method', 'variable', 'parameter',
            'return', 'import', 'module', 'library', 'framework',
            'algorithm', 'complexity', 'performance'
        ]
        specificity_count = sum(
            1 for indicator in technical_indicators
            if indicator in response_text.lower()
        )
        confidence = min(1.0, confidence + specificity_count * 0.01)

        # Check for error handling mentions
        error_handling_keywords = [
            'try', 'catch', 'except', 'error', 'exception',
            'edge case', 'validation', 'check'
        ]
        error_handling_count = sum(
            1 for kw in error_handling_keywords
            if kw in response_text.lower()
        )
        if error_handling_count > 0:
            confidence = min(1.0, confidence + 0.05)

        # Check for best practices mentions
        best_practice_keywords = [
            'best practice', 'recommended', 'pattern', 'convention',
            'standard', 'idiomatic', 'clean code'
        ]
        if any(kw in response_text.lower() for kw in best_practice_keywords):
            confidence = min(1.0, confidence + 0.05)

        # Reduce for uncertainty markers
        uncertainties = extract_uncertainty_markers(response_text)
        confidence -= len(uncertainties) * 0.08

        # Check for TODO or FIXME comments (indicates incomplete solution)
        if 'TODO' in response_text or 'FIXME' in response_text:
            confidence *= 0.9

        # Apply length-based confidence
        length_modifier = calculate_response_length_confidence(response_text)
        confidence *= length_modifier

        return max(0.0, min(1.0, confidence))

    def identify_uncertainties(self, response_text: str) -> List[str]:
        """Identify uncertainties in technical response."""
        uncertainties = extract_uncertainty_markers(response_text)

        # Check for incomplete implementation markers
        incomplete_markers = ['TODO', 'FIXME', 'XXX', '...', 'etc']
        for marker in incomplete_markers:
            if marker in response_text:
                uncertainties.append(f"Incomplete implementation: {marker} found")

        # Check for conditional statements indicating alternatives
        conditional_markers = [
            'depending on', 'if you want', 'alternatively',
            'you could also', 'another option'
        ]
        for marker in conditional_markers:
            if marker.lower() in response_text.lower():
                uncertainties.append(f"Multiple approaches suggested: {marker}")

        # Check for requests for clarification
        if any(q in response_text for q in ['which framework?', 'which version?', 'what language?']):
            uncertainties.append("Needs clarification on technical details")

        # Check for untested code disclaimers
        disclaimer_markers = ['untested', 'not tested', 'might need adjustment']
        for marker in disclaimer_markers:
            if marker.lower() in response_text.lower():
                uncertainties.append(f"Code validation needed: {marker}")

        return uncertainties

    def _format_context(self, context: Dict) -> str:
        """Format technical context dict into a string."""
        parts = []

        if 'language' in context:
            parts.append(f"Language: {context['language']}")

        if 'framework' in context:
            parts.append(f"Framework: {context['framework']}")

        if 'environment' in context:
            parts.append(f"Environment: {context['environment']}")

        if 'constraints' in context:
            parts.append(f"Constraints: {context['constraints']}")

        if 'existing_code' in context:
            parts.append(f"Existing code context:\n{context['existing_code']}")

        if 'dependencies' in context:
            parts.append(f"Dependencies: {context['dependencies']}")

        return "\n".join(parts)

    def _calculate_attention_weight(
        self,
        confidence: float,
        query: ExpertQuery,
        response: str
    ) -> float:
        """Calculate attention weight for technical responses."""
        # Use base calculation
        weight = super()._calculate_attention_weight(confidence, query, response)

        # Boost for code-heavy responses
        code_block_count = len(re.findall(r'```[\s\S]*?```', response))
        if code_block_count > 0:
            weight = min(1.0, weight * (1.0 + code_block_count * 0.05))

        # Boost for comprehensive technical responses
        if len(response.split()) > 200:  # Detailed technical explanation
            weight = min(1.0, weight * 1.1)

        # Check if query explicitly requests technical expertise
        technical_keywords = [
            'implement', 'code', 'function', 'algorithm',
            'architecture', 'debug', 'optimize'
        ]
        if any(kw in query.prompt.lower() for kw in technical_keywords):
            weight = min(1.0, weight * 1.15)

        return weight
