"""
Global Workspace inspection example for Athena Recursive AI.

Demonstrates how to inspect the Global Workspace state and understand
how information flows through the attention-based system.
"""

import asyncio
import uuid
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from athena import MetaOrchestrator, UserQuery, get_config


console = Console()


async def main():
    """Run workspace inspection example."""
    console.print(Panel.fit(
        "[bold cyan]Athena Global Workspace Inspection Example[/bold cyan]\n"
        "Demonstrating GWT-inspired attention mechanisms",
        border_style="cyan"
    ))

    # Load configuration
    config = get_config()

    # Initialize orchestrator
    console.print("\n[yellow]Initializing orchestrator...[/yellow]")
    orchestrator = MetaOrchestrator(config)

    # Complex query that will activate multiple experts
    query = UserQuery(
        query_id=str(uuid.uuid4()),
        content=(
            "I'm building a Unity game with AI-driven NPCs. "
            "Can you help me design a system where NPCs have believable "
            "decision-making that adapts to player behavior? Include both "
            "the conceptual architecture and some example code."
        ),
    )

    console.print(f"\n[bold]Query:[/bold] {query.content}\n")

    try:
        # Process query
        console.print("[yellow]Processing query...[/yellow]\n")
        response = await orchestrator.process_query(query)

        # Show response summary
        console.print(Panel(
            f"[green]{response.response[:300]}...[/green]",
            title=f"Response Preview (Confidence: {response.confidence:.2%})",
            border_style="green"
        ))

        # Inspect workspace state
        console.print("\n[bold cyan]Global Workspace State:[/bold cyan]\n")

        workspace_state = orchestrator.workspace.get_state()

        # Show workspace contents
        if workspace_state.workspace_contents:
            workspace_table = Table(title="Workspace Contents")
            workspace_table.add_column("Source", style="cyan")
            workspace_table.add_column("Attention Weight", style="yellow")
            workspace_table.add_column("Age (s)", style="dim")
            workspace_table.add_column("Content Preview", style="white")

            for item in workspace_state.workspace_contents:
                workspace_table.add_row(
                    item['source'],
                    f"{item['weight']:.3f}",
                    f"{item['age']:.1f}",
                    item['content_preview'][:60] + "..."
                )

            console.print(workspace_table)
        else:
            console.print("[dim]Workspace is empty[/dim]")

        # Show expert activations
        console.print("\n[bold cyan]Expert Activation Levels:[/bold cyan]\n")

        activation_table = Table()
        activation_table.add_column("Expert", style="cyan")
        activation_table.add_column("Activation", style="yellow")
        activation_table.add_column("Bar", style="green")

        for expert, activation in workspace_state.expert_activations.items():
            bar = "█" * int(activation * 20)
            activation_table.add_row(
                expert.value,
                f"{activation:.3f}",
                bar
            )

        console.print(activation_table)

        # Show attention summary
        console.print("\n[bold cyan]Attention Summary:[/bold cyan]\n")

        attention = orchestrator.workspace.get_attention_summary()

        summary_table = Table()
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="yellow")

        summary_table.add_row("Total Items", str(attention['total_items']))
        summary_table.add_row("Total Weight", f"{attention['total_weight']:.3f}")
        summary_table.add_row("Capacity Used", attention['capacity_used'])

        console.print(summary_table)

        # Show expert weights in workspace
        console.print("\n[bold cyan]Expert Weight Distribution in Workspace:[/bold cyan]\n")

        weight_table = Table()
        weight_table.add_column("Expert", style="cyan")
        weight_table.add_column("Total Weight", style="yellow")

        for expert, weight in attention['expert_weights'].items():
            weight_table.add_row(expert, f"{weight:.3f}")

        console.print(weight_table)

        # Show system statistics
        console.print("\n[bold cyan]System Statistics:[/bold cyan]\n")

        stats = orchestrator.get_statistics()

        stats_table = Table()
        stats_table.add_column("Expert", style="cyan")
        stats_table.add_column("Queries", style="yellow")
        stats_table.add_column("Avg Confidence", style="green")

        for expert_name, expert_stats in stats['expert_stats'].items():
            stats_table.add_row(
                expert_name,
                str(expert_stats['query_count']),
                f"{expert_stats['average_confidence']:.2%}"
            )

        console.print(stats_table)

    finally:
        # Cleanup
        await orchestrator.close()
        console.print("\n[green]Done![/green]")


if __name__ == "__main__":
    asyncio.run(main())
