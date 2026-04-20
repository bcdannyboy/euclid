from __future__ import annotations

import euclid


def test_package_exports_only_canonical_active_workflow_names() -> None:
    workflows = euclid.get_workflows()
    stack = euclid.get_runtime_stack()

    assert euclid.ACTIVE_PACKAGE_SCOPE == "current_release"
    assert euclid.NORTH_STAR_SCOPE == "full_vision"
    assert [workflow.name for workflow in workflows] == [
        "run.current_release",
        "replay.current_release",
        "benchmarks.current_release",
        "release.status",
        "release.repo_test_matrix",
        "release.certify_research_readiness",
    ]
    assert {"run_demo", "replay_demo", "build_prototype_intake_plan"} <= set(
        euclid.COMPATIBILITY_ONLY_EXPORTS
    )
    assert callable(euclid.run_operator)
    assert callable(euclid.replay_operator)
    assert [dependency.capability for dependency in stack] == [
        "manifests",
        "numerics",
        "cli",
        "storage",
        "notebook_execution",
        "profiling",
    ]
    assert callable(euclid.run_demo)
    assert callable(euclid.replay_demo)


def test_smoke_summary_is_human_readable() -> None:
    summary = euclid.smoke_summary()

    assert "Euclid bootstrap runtime" in summary
    assert "Available workflows" in summary
    assert "benchmarks.current_release" in summary
    assert "release.repo_test_matrix" in summary
    assert "release.certify_research_readiness" in summary
    assert "demo.replay" not in summary
    assert "pyinstrument" in summary
