from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from euclid.operator_runtime.run import format_operator_run_summary, run_operator
from euclid.release import write_operator_run_evidence_report


def run_command(
    config: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to the operator-facing run config.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help="Directory where the run artifacts are written.",
    ),
    evidence_report: Optional[Path] = typer.Option(
        None,
        dir_okay=False,
        help="Optional path where a certification evidence report is written.",
    ),
) -> None:
    """Run the certified operator workflow from a manifest config."""
    result = run_operator(
        manifest_path=config,
        output_root=output_root,
    )
    if evidence_report is not None:
        write_operator_run_evidence_report(
            result=result,
            report_path=evidence_report,
            scope_id=(
                "full_vision"
                if result.request.request_id == "full-vision-run"
                else "current_release"
            ),
        )
    typer.echo(format_operator_run_summary(result))


def register(app: typer.Typer) -> None:
    app.command("run")(run_command)


__all__ = ["register", "run_command"]
