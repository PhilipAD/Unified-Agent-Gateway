"""Unified Agents SDK CLI.

Usage::

    uag serve            # start the HTTP gateway
    uag chat "prompt"    # query a provider directly (no server needed)
    uag providers        # list registered providers
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="uag",
    help="Unified Agents SDK -- One API. Every LLM. Any tool.",
    add_completion=False,
    no_args_is_help=True,
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8000, help="Port number"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
    workers: int = typer.Option(1, help="Number of worker processes"),
) -> None:
    """Start the Unified Agents SDK HTTP server."""
    import uvicorn

    from runtime.bootstrap import bootstrap_and_configure_app

    asyncio.get_event_loop().run_until_complete(bootstrap_and_configure_app())

    console.print(
        f"[bold green]UAG[/bold green] serving on "
        f"[bold]http://{host}:{port}[/bold]  "
        f"(reload={'on' if reload else 'off'})"
    )

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
    )


@app.command()
def chat(
    prompt: str = typer.Argument(..., help="User message to send"),
    profile: str = typer.Option("default", "--profile", "-p", help="Agent profile name"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream tokens as they arrive"),
    output_json: bool = typer.Option(False, "--json", "-j", help="Output raw JSON response"),
    system: Optional[str] = typer.Option(None, "--system", help="System prompt"),
) -> None:
    """Send a single query to a provider (no running server needed)."""
    asyncio.run(_chat_async(prompt, profile, stream, output_json, system))


async def _chat_async(
    prompt: str,
    profile: str,
    stream: bool,
    output_json: bool,
    system: Optional[str],
) -> None:
    from config.settings import GatewaySettings, ProviderSettings
    from core.agent_loop import AgentLoop
    from core.types import NormalizedMessage, Role
    from runtime.router import create_provider, resolve_provider_config

    provider_settings = ProviderSettings()
    gateway_settings = GatewaySettings()

    cfg = resolve_provider_config(provider_settings, gateway_settings, profile=profile)
    provider = create_provider(cfg)

    messages = []
    if system:
        messages.append(NormalizedMessage(role=Role.SYSTEM, content=system))
    messages.append(NormalizedMessage(role=Role.USER, content=prompt))

    if stream:
        async for event in provider.stream(messages):
            if event.type == "chunk" and event.delta:
                sys.stdout.write(event.delta)
                sys.stdout.flush()
            elif event.type == "error" and event.error:
                console.print(f"\n[red]Error:[/red] {event.error}", highlight=False)
        sys.stdout.write("\n")
    else:
        loop = AgentLoop(provider=provider)
        response = await loop.run_conversation(messages)

        if output_json:
            console.print_json(json.dumps(response.to_dict(), default=str))
        else:
            for msg in response.messages:
                if msg.role == Role.ASSISTANT and msg.content:
                    console.print(msg.content, highlight=False)

            if response.usage:
                usage_parts = []
                for k, v in response.usage.items():
                    usage_parts.append(f"{k}={v}")
                console.print(
                    f"\n[dim]provider={cfg.provider_name}  model={cfg.model}  "
                    f"{' '.join(usage_parts)}[/dim]"
                )


@app.command()
def providers() -> None:
    """List all registered providers and their configuration keys."""
    from config.settings import ProviderSettings
    from runtime.router import PROVIDERS

    settings = ProviderSettings()

    env_keys = {
        "openai_compatible": "OPENAI_API_KEY",
        "openai_responses": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "xai": "XAI_API_KEY",
    }

    table = Table(title="Registered Providers")
    table.add_column("Provider", style="bold cyan")
    table.add_column("Adapter Class")
    table.add_column("Env Key")
    table.add_column("Configured", justify="center")

    for name, cls in PROVIDERS.items():
        env_key = env_keys.get(name, "")
        configured = ""
        if env_key:
            val = getattr(settings, env_key, None)
            configured = "[green]yes[/green]" if val else "[dim]no[/dim]"
        table.add_row(name, cls.__name__, env_key, configured)

    console.print(table)


if __name__ == "__main__":
    app()
