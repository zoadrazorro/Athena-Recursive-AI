"""
Command-line interface for Athena Recursive AI.

Provides an interactive REPL and command-line tools for interacting
with the MoE system.
"""

import asyncio
import uuid
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from loguru import logger

from .core.orchestrator import MetaOrchestrator
from .communication.schemas import UserQuery
from .config.settings import AthenaConfig, get_config, set_config
from .utils.logging import setup_logger


app = typer.Typer(
    name="athena",
    help="Athena Recursive AI - MoE Agentic Meta-LLM System",
    add_completion=False,
)
console = Console()


@app.command()
def interactive(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file"
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        "-l",
        help="Logging level"
    ),
):
    """
    Start an interactive session with Athena.

    This launches a REPL where you can chat with the MoE system,
    which will automatically route your queries to appropriate experts.
    """
    # Load configuration
    if config_file:
        config = AthenaConfig.from_yaml(config_file)
        set_config(config)
    else:
        config = get_config()

    # Override log level
    config.logging.level = log_level
    setup_logger(
        level=config.logging.level,
        log_file=config.logging.file,
    )

    # Run interactive session
    asyncio.run(_interactive_session(config))


async def _interactive_session(config: AthenaConfig):
    """Run the interactive REPL session."""
    console.print(Panel.fit(
        "[bold cyan]Athena Recursive AI[/bold cyan]\n"
        "MoE Agentic Meta-LLM System\n\n"
        "Type your queries and press Enter. Type 'exit' or 'quit' to end.\n"
        "Special commands: /stats, /health, /workspace, /help",
        border_style="cyan"
    ))

    # Initialize orchestrator
    console.print("\n[yellow]Initializing system...[/yellow]")
    orchestrator = MetaOrchestrator(config)

    # Health check
    console.print("[yellow]Running health checks...[/yellow]")
    health_results = await orchestrator.health_check_all()

    health_table = Table(title="System Health")
    health_table.add_column("Component", style="cyan")
    health_table.add_column("Status", style="green")

    for component, healthy in health_results.items():
        status = "✓ Healthy" if healthy else "✗ Unavailable"
        style = "green" if healthy else "red"
        health_table.add_row(component, f"[{style}]{status}[/{style}]")

    console.print(health_table)

    unhealthy_count = sum(1 for h in health_results.values() if not h)
    if unhealthy_count > 0:
        console.print(
            f"\n[yellow]Warning: {unhealthy_count} component(s) unavailable. "
            "Some features may not work.[/yellow]"
        )

    console.print("\n[bold green]System ready![/bold green]\n")

    # REPL loop
    try:
        while True:
            try:
                # Get user input
                user_input = console.input("[bold blue]You:[/bold blue] ").strip()

                if not user_input:
                    continue

                # Check for exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    console.print("\n[yellow]Shutting down...[/yellow]")
                    break

                # Check for special commands
                if user_input.startswith('/'):
                    await _handle_command(user_input, orchestrator)
                    continue

                # Process query
                query = UserQuery(
                    query_id=str(uuid.uuid4()),
                    content=user_input,
                )

                console.print("\n[dim]Processing...[/dim]")

                response = await orchestrator.process_query(query)

                # Display response
                console.print("\n[bold green]Athena:[/bold green]")
                console.print(Panel(
                    Markdown(response.response),
                    border_style="green",
                    title=f"Confidence: {response.confidence:.2%}",
                    subtitle=f"Sources: {', '.join(s.value for s in response.sources)}" if response.sources else None
                ))

                # Show reasoning if available
                if response.reasoning_summary:
                    console.print("\n[dim]Reasoning:[/dim]")
                    console.print(f"[dim]{response.reasoning_summary}[/dim]")

                console.print()

            except KeyboardInterrupt:
                console.print("\n\n[yellow]Use 'exit' or 'quit' to end the session.[/yellow]\n")
                continue

            except Exception as e:
                logger.exception("Error processing query")
                console.print(f"\n[red]Error: {e}[/red]\n")
                continue

    finally:
        await orchestrator.close()
        console.print("[green]Goodbye![/green]\n")


async def _handle_command(command: str, orchestrator: MetaOrchestrator):
    """Handle special commands."""
    cmd = command.lower()

    if cmd == '/help':
        help_text = """
**Available Commands:**

- `/stats` - Show system statistics
- `/health` - Run health checks on all endpoints
- `/workspace` - Show current workspace state
- `/clear` - Clear conversation history
- `/help` - Show this help message
- `exit` or `quit` - End the session
"""
        console.print(Markdown(help_text))

    elif cmd == '/stats':
        stats = orchestrator.get_statistics()

        stats_table = Table(title="System Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")

        stats_table.add_row("Total Queries", str(stats['total_queries']))
        stats_table.add_row("Conversation Turns", str(stats['conversation_turns']))

        console.print(stats_table)

        # Expert usage
        usage_table = Table(title="Expert Usage")
        usage_table.add_column("Expert", style="cyan")
        usage_table.add_column("Queries", style="green")
        usage_table.add_column("Avg Confidence", style="yellow")

        for expert_name, expert_stat in stats['expert_stats'].items():
            usage_table.add_row(
                expert_name,
                str(expert_stat['query_count']),
                f"{expert_stat['average_confidence']:.2%}"
            )

        console.print(usage_table)

        # Workspace
        workspace = stats['workspace_state']
        console.print(f"\n[cyan]Workspace:[/cyan] {workspace['capacity_used']}")

    elif cmd == '/health':
        console.print("[yellow]Running health checks...[/yellow]")
        health_results = await orchestrator.health_check_all()

        health_table = Table(title="System Health")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")

        for component, healthy in health_results.items():
            status = "✓ Healthy" if healthy else "✗ Unavailable"
            style = "green" if healthy else "red"
            health_table.add_row(component, f"[{style}]{status}[/{style}]")

        console.print(health_table)

    elif cmd == '/workspace':
        workspace_state = orchestrator.workspace.get_state()

        console.print(Panel.fit(
            f"**Context:** {workspace_state.context_summary}\n\n"
            f"**Active Queries:** {len(workspace_state.active_queries)}\n"
            f"**Workspace Items:** {len(workspace_state.workspace_contents)}\n"
            f"**Attention Focus:** {workspace_state.attention_focus.value if workspace_state.attention_focus else 'None'}",
            title="Global Workspace State",
            border_style="cyan"
        ))

        if workspace_state.workspace_contents:
            content_table = Table(title="Workspace Contents")
            content_table.add_column("Source", style="cyan")
            content_table.add_column("Weight", style="yellow")
            content_table.add_column("Preview", style="dim")

            for item in workspace_state.workspace_contents[:5]:
                content_table.add_row(
                    item['source'],
                    f"{item['weight']:.2f}",
                    item['content_preview']
                )

            console.print(content_table)

    elif cmd == '/clear':
        orchestrator.conversation_history.clear()
        orchestrator.workspace.clear()
        console.print("[green]Conversation history and workspace cleared.[/green]")

    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("[dim]Type /help for available commands.[/dim]")


@app.command()
def query(
    text: str = typer.Argument(..., help="Query text"),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file"
    ),
    log_level: str = typer.Option(
        "WARNING",
        "--log-level",
        "-l",
        help="Logging level"
    ),
):
    """
    Send a single query to Athena and get a response.

    This is useful for scripting or one-off queries.
    """
    # Load configuration
    if config_file:
        config = AthenaConfig.from_yaml(config_file)
        set_config(config)
    else:
        config = get_config()

    config.logging.level = log_level
    setup_logger(level=config.logging.level)

    asyncio.run(_single_query(text, config))


async def _single_query(text: str, config: AthenaConfig):
    """Process a single query."""
    orchestrator = MetaOrchestrator(config)

    try:
        query = UserQuery(
            query_id=str(uuid.uuid4()),
            content=text,
        )

        response = await orchestrator.process_query(query)

        # Print response
        console.print(Panel(
            Markdown(response.response),
            border_style="green",
            title=f"Confidence: {response.confidence:.2%}",
        ))

    finally:
        await orchestrator.close()


@app.command()
def health(
    config_file: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML configuration file"
    ),
):
    """
    Check the health of all model endpoints.
    """
    if config_file:
        config = AthenaConfig.from_yaml(config_file)
        set_config(config)
    else:
        config = get_config()

    asyncio.run(_health_check(config))


async def _health_check(config: AthenaConfig):
    """Run health check."""
    orchestrator = MetaOrchestrator(config)

    try:
        health_results = await orchestrator.health_check_all()

        health_table = Table(title="System Health Check")
        health_table.add_column("Component", style="cyan")
        health_table.add_column("Status", style="green")

        all_healthy = True

        for component, healthy in health_results.items():
            status = "✓ Healthy" if healthy else "✗ Unavailable"
            style = "green" if healthy else "red"
            health_table.add_row(component, f"[{style}]{status}[/{style}]")

            if not healthy:
                all_healthy = False

        console.print(health_table)

        if all_healthy:
            console.print("\n[bold green]All systems operational![/bold green]")
        else:
            console.print("\n[bold red]Some systems are unavailable.[/bold red]")
            raise typer.Exit(1)

    finally:
        await orchestrator.close()


@app.command()
def version():
    """Show version information."""
    console.print("[cyan]Athena Recursive AI v0.1.0[/cyan]")
    console.print("MoE Agentic Meta-LLM System")


def main():
    """Main entry point."""
    app()


if __name__ == "__main__":
    main()
