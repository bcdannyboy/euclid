from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from euclid.operator_runtime.replay import (
    format_operator_replay_summary,
    replay_operator,
)
from euclid.release import write_operator_replay_evidence_report


def replay_command(
    run_id: str = typer.Option(
        ...,
        help="Run identifier whose saved summary should be replayed.",
    ),
    output_root: Optional[Path] = typer.Option(
        None,
        file_okay=False,
        help=(
            "Optional output root override instead of the default "
            "build/operator/<run-id> path."
        ),
    ),
    evidence_report: Optional[Path] = typer.Option(
        None,
        dir_okay=False,
        help="Optional path where a certification replay evidence report is written.",
    ),
) -> None:
    """Replay a previously saved operator run from its run id."""
    result = replay_operator(output_root=output_root, run_id=run_id)
    if evidence_report is not None:
        write_operator_replay_evidence_report(
            run_id=run_id,
            result=result,
            report_path=evidence_report,
            scope_id=(
                "full_vision" if run_id == "full-vision-run" else "current_release"
            ),
        )
    typer.echo(format_operator_replay_summary(run_id=run_id, result=result))


def register(app: typer.Typer) -> None:
    app.command("replay")(replay_command)


__all__ = ["register", "replay_command"]
