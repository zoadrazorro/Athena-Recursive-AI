"""
Basic query example for Athena Recursive AI.

Demonstrates simple usage of the Meta-Orchestrator to process a single query.
"""

import asyncio
import uuid
from athena import MetaOrchestrator, UserQuery, get_config


async def main():
    """Run a basic query example."""
    print("Athena Basic Query Example")
    print("=" * 50)

    # Load configuration from environment
    config = get_config()

    # Initialize orchestrator
    print("\nInitializing orchestrator...")
    orchestrator = MetaOrchestrator(config)

    # Create a simple query
    query = UserQuery(
        query_id=str(uuid.uuid4()),
        content="What is the difference between recursion and iteration in programming?",
    )

    print(f"\nQuery: {query.content}")
    print("\nProcessing...\n")

    try:
        # Process the query
        response = await orchestrator.process_query(query)

        # Display results
        print("=" * 50)
        print("RESPONSE:")
        print("=" * 50)
        print(response.response)
        print()
        print("-" * 50)
        print(f"Confidence: {response.confidence:.2%}")
        print(f"Sources: {', '.join(s.value for s in response.sources)}")

        if response.reasoning_summary:
            print("\nReasoning:")
            print(response.reasoning_summary)

        # Show expert contributions
        if response.expert_contributions:
            print("\nExpert Contributions:")
            for expert, contribution in response.expert_contributions.items():
                print(f"  - {expert.value}: {contribution:.2%}")

    finally:
        # Cleanup
        await orchestrator.close()
        print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
