from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workflow:
    name: str
    description: str


@dataclass(frozen=True)
class RuntimeDependency:
    capability: str
    package: str
    purpose: str


_WORKFLOWS = (
    Workflow(
        name="run.current_release",
        description=(
            "Run the active current_release operator path against a local manifest and "
            "dataset pair."
        ),
    ),
    Workflow(
        name="replay.current_release",
        description=(
            "Replay the active current_release operator path from sealed artifacts."
        ),
    ),
    Workflow(
        name="benchmarks.current_release",
        description=(
            "Run the declared current_release benchmark suite and emit the suite "
            "summary plus task reports."
        ),
    ),
    Workflow(
        name="release.status",
        description=(
            "Recompute current_release, full_vision, and shipped_releasable status."
        ),
    ),
    Workflow(
        name="release.repo_test_matrix",
        description=(
            "Run the required certification repo test matrix with zero-skip enforcement."
        ),
    ),
    Workflow(
        name="release.certify_research_readiness",
        description=(
            "Fail closed unless the full research-readiness certification battery is green."
        ),
    ),
)

_RUNTIME_STACK = (
    RuntimeDependency(
        capability="manifests",
        package="pydantic==2.12.5",
        purpose="Typed manifest models and validation entrypoints.",
    ),
    RuntimeDependency(
        capability="numerics",
        package="numpy==2.3.5",
        purpose="Ordered numeric series handling and later reducer math.",
    ),
    RuntimeDependency(
        capability="cli",
        package="typer==0.15.2",
        purpose="Readable local commands for smoke, demo, and replay entrypoints.",
    ),
    RuntimeDependency(
        capability="storage",
        package="sqlite3 (stdlib) + sqlalchemy==2.0.38",
        purpose="Immutable local metadata storage with a future-friendly query layer.",
    ),
    RuntimeDependency(
        capability="notebook_execution",
        package="nbformat==5.10.4 + nbclient==0.10.2",
        purpose="Notebook artifact authoring and deterministic execution hooks.",
    ),
    RuntimeDependency(
        capability="profiling",
        package="pyinstrument==5.0.0",
        purpose="Local runtime profiling during search, replay, and suite execution.",
    ),
)


def get_workflows() -> tuple[Workflow, ...]:
    return _WORKFLOWS


def get_runtime_stack() -> tuple[RuntimeDependency, ...]:
    return _RUNTIME_STACK


def smoke_summary() -> str:
    lines = [
        "Euclid bootstrap runtime",
        "",
        "Available workflows:",
    ]
    for workflow in _WORKFLOWS:
        lines.append(f"- {workflow.name}: {workflow.description}")
    lines.append("")
    lines.append("Locked runtime stack:")
    lines.extend(
        f"- {dependency.capability}: {dependency.package} ({dependency.purpose})"
        for dependency in _RUNTIME_STACK
    )
    return "\n".join(lines)
