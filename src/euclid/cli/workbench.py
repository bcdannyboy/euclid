from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from euclid.workbench.server import serve_workbench

workbench_app = typer.Typer(
    add_completion=False,
    help="Launch the live market workbench UI on a local HTTP server.",
)


@workbench_app.command("serve")
def serve_command(
    host: str = typer.Option(
        "127.0.0.1",
        help="Host interface to bind the server to.",
    ),
    port: int = typer.Option(8765, help="TCP port to bind the server to."),
    output_root: Path = typer.Option(
        Path("build") / "workbench",
        file_okay=False,
        help=(
            "Directory where datasets, manifests, runs, and saved analyses "
            "are written."
        ),
    ),
    project_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Optional project root override for the Euclid runtime.",
    ),
    api_key_env_var: str = typer.Option(
        "FMP_API_KEY",
        help=(
            "Environment variable to fall back to when the form omits an "
            "API key."
        ),
    ),
    open_browser: bool = typer.Option(
        False,
        "--open-browser/--no-open-browser",
        help="Open the workbench URL in the default browser after startup.",
    ),
) -> None:
    """Serve the Euclid Market Workbench UI locally."""
    typer.echo(f"Euclid workbench listening on http://{host}:{port}")
    serve_workbench(
        host=host,
        port=port,
        output_root=output_root,
        project_root=project_root,
        api_key_env_var=api_key_env_var,
        open_browser=open_browser,
    )


def register(app: typer.Typer) -> None:
    app.add_typer(workbench_app, name="workbench")


__all__ = ["register", "serve_command", "workbench_app"]
