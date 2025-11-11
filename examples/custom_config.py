"""
Custom configuration example for Athena Recursive AI.

Demonstrates how to create and use custom configurations
programmatically without relying on environment variables.
"""

import asyncio
import uuid
from pathlib import Path
from athena import MetaOrchestrator, UserQuery
from athena.config.settings import (
    AthenaConfig,
    ModelEndpoint,
    GWTConfig,
    ExpertRoutingConfig,
    LoggingConfig,
    set_config,
)


async def main():
    """Run example with custom configuration."""
    print("Athena Custom Configuration Example")
    print("=" * 60)

    # Create custom configuration programmatically
    custom_config = AthenaConfig(
        # Orchestrator configuration
        orchestrator=ModelEndpoint(
            url="http://localhost:1234/v1",
            model_name="qwen2.5-14b-instruct",
            gpu_id=0,
            max_tokens=3000,  # Custom: higher token limit
            temperature=0.6,   # Custom: lower temperature for consistency
            top_p=0.92,
            timeout=150,
        ),
        # Expert configurations
        reasoning_expert=ModelEndpoint(
            url="http://localhost:1235/v1",
            model_name="phi-3.5-mini-instruct",
            gpu_id=1,
            temperature=0.4,  # Low temp for precise reasoning
        ),
        creative_expert=ModelEndpoint(
            url="http://localhost:1236/v1",
            model_name="mistral-7b-instruct",
            gpu_id=1,
            temperature=0.95,  # High temp for creativity
        ),
        technical_expert=ModelEndpoint(
            url="http://localhost:1237/v1",
            model_name="codeqwen-7b-instruct",
            gpu_id=1,
            temperature=0.2,   # Very low temp for code accuracy
            max_tokens=4096,   # More tokens for code
        ),
        memory_expert=ModelEndpoint(
            url="http://localhost:1238/v1",
            model_name="llama-3.1-8b-instruct",
            gpu_id=1,
        ),
        # Custom GWT parameters
        gwt=GWTConfig(
            attention_threshold=0.7,  # Higher threshold for stricter filtering
            workspace_size=3,         # Smaller workspace for focused attention
            broadcast_decay=0.85,     # Faster decay
            enable_competition=True,
        ),
        # Custom routing configuration
        routing=ExpertRoutingConfig(
            enable_parallel=True,
            confidence_threshold=0.8,  # Higher confidence requirement
            max_retries=2,
            timeout_per_expert=45,
        ),
        # Custom logging
        logging=LoggingConfig(
            level="DEBUG",  # Verbose logging
            file="custom_athena.log",
        ),
        # Other settings
        max_context_length=6144,
        conversation_history_size=8,
    )

    # Set as global config
    set_config(custom_config)

    # Optionally save to YAML for reuse
    config_path = Path("examples/my_custom_config.yaml")
    custom_config.save_yaml(config_path)
    print(f"\nCustom configuration saved to: {config_path}")

    print("\nCustom Configuration Settings:")
    print(f"  - Orchestrator temperature: {custom_config.orchestrator.temperature}")
    print(f"  - Creative expert temperature: {custom_config.creative_expert.temperature}")
    print(f"  - Technical expert temperature: {custom_config.technical_expert.temperature}")
    print(f"  - GWT attention threshold: {custom_config.gwt.attention_threshold}")
    print(f"  - Workspace size: {custom_config.gwt.workspace_size}")
    print(f"  - Confidence threshold: {custom_config.routing.confidence_threshold}")

    # Initialize orchestrator with custom config
    print("\nInitializing orchestrator with custom configuration...")
    orchestrator = MetaOrchestrator(custom_config)

    # Test query
    query = UserQuery(
        query_id=str(uuid.uuid4()),
        content=(
            "Create a creative metaphor to explain how the Global Workspace Theory "
            "relates to consciousness, then provide a Python code example demonstrating "
            "a simple attention mechanism."
        ),
    )

    print(f"\nQuery: {query.content}")
    print("\nProcessing with custom configuration...\n")

    try:
        # Process query
        response = await orchestrator.process_query(query)

        # Display results
        print("=" * 60)
        print("RESPONSE:")
        print("=" * 60)
        print(response.response)
        print()
        print("-" * 60)
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Sources: {', '.join(s.value for s in response.sources)}")

        # Show configuration impact
        print("\nConfiguration Impact:")
        if 'creative' in [s.value for s in response.sources]:
            print("  ✓ Creative expert consulted (high temperature)")
        if 'technical' in [s.value for s in response.sources]:
            print("  ✓ Technical expert consulted (low temperature for code)")

        # Show workspace state with custom parameters
        workspace_state = orchestrator.workspace.get_state()
        print(f"\nWorkspace items: {len(workspace_state.workspace_contents)}/{custom_config.gwt.workspace_size}")

        if workspace_state.workspace_contents:
            print("Top workspace item:")
            top_item = workspace_state.workspace_contents[0]
            print(f"  Source: {top_item['source']}")
            print(f"  Weight: {top_item['weight']:.3f}")
            print(f"  (Threshold: {custom_config.gwt.attention_threshold})")

    finally:
        # Cleanup
        await orchestrator.close()
        print("\nDone!")

    # Show how to load config from YAML
    print("\n" + "=" * 60)
    print("Loading Configuration from YAML")
    print("=" * 60)

    print(f"\nTo use this configuration in future sessions:")
    print(f"  athena interactive --config {config_path}")
    print("\nOr in code:")
    print(f"  config = AthenaConfig.from_yaml('{config_path}')")


if __name__ == "__main__":
    asyncio.run(main())
