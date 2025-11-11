"""
Meta-Orchestrator for the Athena MoE system.

Implements the central coordination layer that analyzes queries,
routes them to appropriate experts, and synthesizes responses.
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from ..communication.schemas import (
    UserQuery,
    ExpertQuery,
    ExpertResponse,
    QueryAnalysis,
    SynthesizedResponse,
    TaskComplexity,
    RoutingStrategy,
    ExpertType,
    AttentionBroadcast,
)
from ..communication.protocol import ConflictResolver
from ..communication.lm_studio_client import LMStudioClient, ModelPool
from ..core.workspace import GlobalWorkspace
from ..core.expert_base import ExpertModel
from ..experts.reasoning import ReasoningExpert
from ..experts.creative import CreativeExpert
from ..experts.technical import TechnicalExpert
from ..experts.memory import MemoryExpert
from ..config.settings import AthenaConfig


class MetaOrchestrator:
    """
    Central orchestrator implementing GWT-inspired meta-reasoning.

    Coordinates expert models, manages the global workspace, and
    synthesizes coherent responses from multiple expert inputs.
    """

    def __init__(self, config: AthenaConfig):
        """
        Initialize the Meta-Orchestrator.

        Args:
            config: System configuration
        """
        self.config = config
        self.workspace = GlobalWorkspace(config.gwt)

        # Initialize LM Studio client for orchestrator
        self.orchestrator_client = LMStudioClient(
            base_url=config.orchestrator.url,
            model_name=config.orchestrator.model_name,
            timeout=config.orchestrator.timeout,
        )

        # Initialize expert models
        self.experts: Dict[ExpertType, ExpertModel] = {}
        self._initialize_experts()

        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Statistics
        self.query_count = 0
        self.expert_usage_count: Dict[ExpertType, int] = {
            expert: 0 for expert in ExpertType
        }

    def _initialize_experts(self) -> None:
        """Initialize all expert models."""
        logger.info("Initializing expert models...")

        # Reasoning Expert
        reasoning_client = LMStudioClient(
            base_url=self.config.reasoning_expert.url,
            model_name=self.config.reasoning_expert.model_name,
            timeout=self.config.reasoning_expert.timeout,
        )
        self.experts[ExpertType.REASONING] = ReasoningExpert(
            expert_type=ExpertType.REASONING,
            endpoint_config=self.config.reasoning_expert,
            client=reasoning_client,
        )

        # Creative Expert
        creative_client = LMStudioClient(
            base_url=self.config.creative_expert.url,
            model_name=self.config.creative_expert.model_name,
            timeout=self.config.creative_expert.timeout,
        )
        self.experts[ExpertType.CREATIVE] = CreativeExpert(
            expert_type=ExpertType.CREATIVE,
            endpoint_config=self.config.creative_expert,
            client=creative_client,
        )

        # Technical Expert
        technical_client = LMStudioClient(
            base_url=self.config.technical_expert.url,
            model_name=self.config.technical_expert.model_name,
            timeout=self.config.technical_expert.timeout,
        )
        self.experts[ExpertType.TECHNICAL] = TechnicalExpert(
            expert_type=ExpertType.TECHNICAL,
            endpoint_config=self.config.technical_expert,
            client=technical_client,
        )

        # Memory Expert
        memory_client = LMStudioClient(
            base_url=self.config.memory_expert.url,
            model_name=self.config.memory_expert.model_name,
            timeout=self.config.memory_expert.timeout,
        )
        self.experts[ExpertType.MEMORY] = MemoryExpert(
            expert_type=ExpertType.MEMORY,
            endpoint_config=self.config.memory_expert,
            client=memory_client,
        )

        logger.info(f"Initialized {len(self.experts)} expert models")

    async def process_query(self, user_query: UserQuery) -> SynthesizedResponse:
        """
        Process a user query through the MoE system.

        This is the main entry point for query processing.

        Args:
            user_query: UserQuery from the user

        Returns:
            SynthesizedResponse with the final answer
        """
        self.query_count += 1

        logger.info(f"Processing query {user_query.query_id} (total: {self.query_count})")

        # Add conversation history to query context
        if not user_query.conversation_history:
            user_query.conversation_history = self.conversation_history.copy()

        # Analyze the query
        analysis = await self.analyze_query(user_query)

        logger.info(
            f"Query analysis: complexity={analysis.complexity.value}, "
            f"strategy={analysis.routing_strategy.value}, "
            f"experts={[e.value for e in analysis.required_experts]}"
        )

        # Route based on complexity and strategy
        if analysis.routing_strategy == RoutingStrategy.DIRECT:
            # Orchestrator responds directly
            response = await self._direct_response(user_query, analysis)

        elif analysis.routing_strategy == RoutingStrategy.SINGLE_EXPERT:
            # Route to single expert
            response = await self._single_expert_response(user_query, analysis)

        elif analysis.routing_strategy == RoutingStrategy.PARALLEL:
            # Consult multiple experts in parallel
            response = await self._parallel_expert_response(user_query, analysis)

        elif analysis.routing_strategy == RoutingStrategy.SEQUENTIAL:
            # Chain experts sequentially
            response = await self._sequential_expert_response(user_query, analysis)

        else:
            raise ValueError(f"Unknown routing strategy: {analysis.routing_strategy}")

        # Update conversation history
        self._update_conversation_history(user_query.content, response.response)

        # Decay workspace activations
        self.workspace.decay_activations()

        logger.info(
            f"Completed query {user_query.query_id} "
            f"(confidence: {response.confidence:.2f})"
        )

        return response

    async def analyze_query(self, user_query: UserQuery) -> QueryAnalysis:
        """
        Analyze a user query to determine routing strategy.

        Args:
            user_query: Query to analyze

        Returns:
            QueryAnalysis with routing decision
        """
        # Build analysis prompt
        workspace_state = self.workspace.get_state()

        analysis_prompt = f"""Analyze this user query and determine the best approach to answer it.

User Query: {user_query.content}

Current Workspace State:
{workspace_state.context_summary}

Available Experts:
1. Reasoning: Logical inference, mathematical reasoning, analytical thinking
2. Creative: Creative ideation, philosophical exploration, conceptual synthesis
3. Technical: Code generation, software architecture, technical implementation
4. Memory: Conversation context, consistency checking, continuity

Classify the query:
1. Complexity: trivial, simple, moderate, complex, expert_required
2. Required experts: Which experts should be consulted?
3. Routing strategy: direct (you answer), single_expert, parallel (multiple experts), sequential (chained)
4. Rationale: Brief explanation of your decision

Respond in this format:
Complexity: [complexity]
Required Experts: [expert1, expert2, ...]
Routing Strategy: [strategy]
Rationale: [explanation]"""

        messages = [
            {
                "role": "system",
                "content": "You are a meta-cognitive coordinator analyzing queries to route them to appropriate specialized experts."
            },
            {"role": "user", "content": analysis_prompt}
        ]

        response = await self.orchestrator_client.chat_completion(
            messages=messages,
            temperature=0.3,  # Low temperature for consistent analysis
            max_tokens=500,
        )

        response_text = await self.orchestrator_client.extract_response_text(response)

        # Parse the response
        return self._parse_analysis(response_text, user_query.query_id)

    def _parse_analysis(self, analysis_text: str, query_id: str) -> QueryAnalysis:
        """Parse the orchestrator's analysis into a QueryAnalysis object."""
        # Simple parsing (could be made more robust)
        complexity = TaskComplexity.MODERATE  # Default
        required_experts = []
        routing_strategy = RoutingStrategy.PARALLEL  # Default
        rationale = ""

        lines = analysis_text.lower().split('\n')

        for line in lines:
            if 'complexity:' in line:
                complexity_str = line.split('complexity:')[1].strip()
                for c in TaskComplexity:
                    if c.value in complexity_str:
                        complexity = c
                        break

            elif 'required experts:' in line:
                experts_str = line.split('required experts:')[1].strip()
                for expert in ExpertType:
                    if expert.value in experts_str:
                        required_experts.append(expert)

            elif 'routing strategy:' in line:
                strategy_str = line.split('routing strategy:')[1].strip()
                for s in RoutingStrategy:
                    if s.value in strategy_str:
                        routing_strategy = s
                        break

            elif 'rationale:' in line:
                rationale = line.split('rationale:')[1].strip()

        # If no experts identified but strategy requires them, default to reasoning
        if not required_experts and routing_strategy != RoutingStrategy.DIRECT:
            required_experts = [ExpertType.REASONING]

        # Calculate difficulty estimate
        difficulty_map = {
            TaskComplexity.TRIVIAL: 0.1,
            TaskComplexity.SIMPLE: 0.3,
            TaskComplexity.MODERATE: 0.5,
            TaskComplexity.COMPLEX: 0.7,
            TaskComplexity.EXPERT_REQUIRED: 0.9,
        }

        return QueryAnalysis(
            query_id=query_id,
            complexity=complexity,
            required_experts=required_experts,
            routing_strategy=routing_strategy,
            decomposed_tasks=[],  # Could implement task decomposition
            rationale=rationale or "No specific rationale provided",
            estimated_difficulty=difficulty_map.get(complexity, 0.5),
        )

    async def _direct_response(
        self,
        user_query: UserQuery,
        analysis: QueryAnalysis
    ) -> SynthesizedResponse:
        """Generate direct response from orchestrator."""
        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant providing direct responses to simple queries."
            }
        ]

        # Add conversation history
        for turn in user_query.conversation_history[-5:]:
            messages.append(turn)

        messages.append({"role": "user", "content": user_query.content})

        response = await self.orchestrator_client.chat_completion(
            messages=messages,
            temperature=self.config.orchestrator.temperature,
            max_tokens=self.config.orchestrator.max_tokens,
        )

        response_text = await self.orchestrator_client.extract_response_text(response)

        return SynthesizedResponse(
            query_id=user_query.query_id,
            response=response_text,
            expert_contributions={},
            confidence=0.85,  # Moderate confidence for direct responses
            sources=[],
            metadata={"strategy": "direct", "model": self.config.orchestrator.model_name}
        )

    async def _single_expert_response(
        self,
        user_query: UserQuery,
        analysis: QueryAnalysis
    ) -> SynthesizedResponse:
        """Route to single expert and return response."""
        expert_type = analysis.required_experts[0]
        expert = self.experts[expert_type]

        # Create expert query
        expert_query = ExpertQuery(
            query_id=user_query.query_id,
            expert_type=expert_type,
            prompt=user_query.content,
            context=user_query.context or {},
            max_tokens=self.config.orchestrator.max_tokens,
            temperature=self.config.orchestrator.temperature,
        )

        # Add conversation history to context
        expert_query.context['conversation_history'] = user_query.conversation_history

        # Query expert
        expert_response = await expert.process_query(expert_query)

        # Update usage stats
        self.expert_usage_count[expert_type] += 1

        # Broadcast to workspace
        await self._broadcast_to_workspace(expert_response, user_query.query_id)

        return SynthesizedResponse(
            query_id=user_query.query_id,
            response=expert_response.response,
            expert_contributions={expert_type: 1.0},
            confidence=expert_response.confidence,
            reasoning_summary="\n".join(expert_response.reasoning_trace or []),
            sources=[expert_type],
            metadata={
                "strategy": "single_expert",
                "expert": expert_type.value,
                "uncertainties": expert_response.uncertainty_flags,
            }
        )

    async def _parallel_expert_response(
        self,
        user_query: UserQuery,
        analysis: QueryAnalysis
    ) -> SynthesizedResponse:
        """Consult multiple experts in parallel."""
        # Create expert queries
        expert_queries = []

        for expert_type in analysis.required_experts:
            expert_query = ExpertQuery(
                query_id=user_query.query_id,
                expert_type=expert_type,
                prompt=user_query.content,
                context=user_query.context or {},
                max_tokens=self.config.orchestrator.max_tokens,
                temperature=self.config.orchestrator.temperature,
            )
            expert_query.context['conversation_history'] = user_query.conversation_history
            expert_queries.append((expert_type, expert_query))

        # Query all experts in parallel
        tasks = [
            self.experts[expert_type].process_query(query)
            for expert_type, query in expert_queries
        ]

        expert_responses = await asyncio.gather(*tasks)

        # Update usage stats
        for expert_type in analysis.required_experts:
            self.expert_usage_count[expert_type] += 1

        # Broadcast to workspace
        for response in expert_responses:
            await self._broadcast_to_workspace(response, user_query.query_id)

        # Synthesize responses
        synthesized = await self._synthesize_responses(
            user_query, expert_responses, analysis
        )

        return synthesized

    async def _sequential_expert_response(
        self,
        user_query: UserQuery,
        analysis: QueryAnalysis
    ) -> SynthesizedResponse:
        """Chain experts sequentially."""
        expert_responses = []
        accumulated_context = user_query.context or {}

        for expert_type in analysis.required_experts:
            # Create query with accumulated context
            expert_query = ExpertQuery(
                query_id=user_query.query_id,
                expert_type=expert_type,
                prompt=user_query.content,
                context=accumulated_context.copy(),
                max_tokens=self.config.orchestrator.max_tokens,
                temperature=self.config.orchestrator.temperature,
            )

            # Query expert
            response = await self.experts[expert_type].process_query(expert_query)
            expert_responses.append(response)

            # Update usage stats
            self.expert_usage_count[expert_type] += 1

            # Broadcast to workspace
            await self._broadcast_to_workspace(response, user_query.query_id)

            # Add response to context for next expert
            accumulated_context[f'{expert_type.value}_response'] = response.response

        # Synthesize final response
        return await self._synthesize_responses(
            user_query, expert_responses, analysis
        )

    async def _synthesize_responses(
        self,
        user_query: UserQuery,
        expert_responses: List[ExpertResponse],
        analysis: QueryAnalysis
    ) -> SynthesizedResponse:
        """Synthesize multiple expert responses into coherent answer."""
        # Use conflict resolver to merge responses
        merged_text, overall_confidence = ConflictResolver.weighted_confidence_merge(
            expert_responses
        )

        # Calculate expert contributions
        total_weight = sum(r.attention_weight for r in expert_responses)
        contributions = {
            r.expert_type: r.attention_weight / total_weight if total_weight > 0 else 0.0
            for r in expert_responses
        }

        # Detect conflicts
        conflicts = ConflictResolver.detect_conflicts(expert_responses)
        if conflicts:
            logger.warning(f"Conflicts detected: {conflicts}")

        # Build reasoning summary
        reasoning_parts = []
        for response in expert_responses:
            if response.reasoning_trace:
                reasoning_parts.append(
                    f"{response.expert_type.value}: " + "; ".join(response.reasoning_trace[:3])
                )

        reasoning_summary = "\n".join(reasoning_parts) if reasoning_parts else None

        return SynthesizedResponse(
            query_id=user_query.query_id,
            response=merged_text,
            expert_contributions=contributions,
            confidence=overall_confidence,
            reasoning_summary=reasoning_summary,
            sources=[r.expert_type for r in expert_responses],
            metadata={
                "strategy": analysis.routing_strategy.value,
                "conflicts": conflicts,
                "expert_count": len(expert_responses),
            }
        )

    async def _broadcast_to_workspace(
        self,
        response: ExpertResponse,
        query_id: str
    ) -> None:
        """Broadcast expert response to global workspace."""
        broadcast = AttentionBroadcast(
            broadcast_id=str(uuid.uuid4()),
            source=response.expert_type,
            content=response.response[:500],  # First 500 chars
            attention_weight=response.attention_weight,
            decay_rate=self.config.gwt.broadcast_decay,
            related_query_id=query_id,
        )

        accepted = self.workspace.broadcast(broadcast)

        if accepted:
            # Update expert activation
            self.workspace.update_expert_activation(
                response.expert_type,
                response.confidence
            )

    def _update_conversation_history(self, user_message: str, assistant_message: str) -> None:
        """Update conversation history with latest exchange."""
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": assistant_message})

        # Trim to max size
        max_history = self.config.conversation_history_size * 2  # user + assistant per turn
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all model endpoints."""
        logger.info("Running health checks on all endpoints...")

        results = {"orchestrator": False}

        # Check orchestrator
        try:
            orch_health = await self.orchestrator_client.health_check()
            results["orchestrator"] = orch_health.status == "healthy"
        except Exception as e:
            logger.error(f"Orchestrator health check failed: {e}")

        # Check experts
        for expert_type, expert in self.experts.items():
            try:
                healthy = await expert.health_check()
                results[expert_type.value] = healthy
            except Exception as e:
                logger.error(f"{expert_type.value} health check failed: {e}")
                results[expert_type.value] = False

        return results

    def get_statistics(self) -> Dict:
        """Get system statistics."""
        expert_stats = {
            expert_type.value: expert.get_statistics()
            for expert_type, expert in self.experts.items()
        }

        return {
            "total_queries": self.query_count,
            "expert_usage": {
                k.value: v for k, v in self.expert_usage_count.items()
            },
            "expert_stats": expert_stats,
            "workspace_state": self.workspace.get_attention_summary(),
            "conversation_turns": len(self.conversation_history) // 2,
        }

    async def close(self) -> None:
        """Clean up resources."""
        logger.info("Closing orchestrator and expert connections...")

        await self.orchestrator_client.close()

        for expert in self.experts.values():
            await expert.client.close()

        logger.info("All connections closed")
