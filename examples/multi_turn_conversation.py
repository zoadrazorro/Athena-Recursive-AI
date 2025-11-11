"""
Multi-turn conversation example for Athena Recursive AI.

Demonstrates how Athena maintains context and consistency across
multiple conversation turns using the Memory Expert.
"""

import asyncio
import uuid
from athena import MetaOrchestrator, UserQuery, get_config


async def main():
    """Run a multi-turn conversation example."""
    print("Athena Multi-Turn Conversation Example")
    print("=" * 60)

    # Load configuration
    config = get_config()

    # Initialize orchestrator
    print("\nInitializing orchestrator...")
    orchestrator = MetaOrchestrator(config)

    # Define a series of related queries
    queries = [
        "Explain the concept of consciousness in simple terms.",
        "How does Global Workspace Theory explain consciousness?",
        "Can you relate what you just said about GWT to artificial intelligence?",
        "Based on our discussion, do you think AI systems can be conscious?",
    ]

    try:
        for i, query_text in enumerate(queries, 1):
            print(f"\n{'=' * 60}")
            print(f"Turn {i}")
            print(f"{'=' * 60}")

            # Create query
            query = UserQuery(
                query_id=str(uuid.uuid4()),
                content=query_text,
            )

            print(f"\nUser: {query_text}")
            print("\nProcessing...\n")

            # Process query (conversation history is maintained internally)
            response = await orchestrator.process_query(query)

            # Display response
            print("Athena:", response.response[:500])
            if len(response.response) > 500:
                print("...\n(response truncated for display)")
            else:
                print()

            print(f"\nConfidence: {response.confidence:.2%}")
            print(f"Sources: {', '.join(s.value for s in response.sources)}")

            # Check if Memory Expert was consulted
            if 'memory' in [s.value for s in response.sources]:
                print("✓ Memory Expert consulted for context")

        # Show final statistics
        print(f"\n{'=' * 60}")
        print("Conversation Statistics")
        print(f"{'=' * 60}")

        stats = orchestrator.get_statistics()
        print(f"\nTotal queries: {stats['total_queries']}")
        print(f"Conversation turns: {stats['conversation_turns']}")

        print("\nExpert usage:")
        for expert, count in stats['expert_usage'].items():
            print(f"  - {expert}: {count} times")

        # Show workspace state
        workspace_summary = stats['workspace_state']
        print(f"\nWorkspace capacity: {workspace_summary['capacity_used']}")
        print(f"Total attention weight: {workspace_summary['total_weight']:.2f}")

    finally:
        # Cleanup
        await orchestrator.close()
        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
