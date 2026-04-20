from __future__ import annotations

from pathlib import Path

import euclid.release as release


def test_release_module_loads_current_and_full_vision_packaged_policies(
    project_root: Path,
) -> None:
    policies = release.load_packaged_release_policies(project_root=project_root)

    assert set(policies) == {
        "current_release_v1",
        "full_vision_v1",
        "shipped_releasable_v1",
    }

    current = policies["current_release_v1"]
    full = policies["full_vision_v1"]
    shipped = policies["shipped_releasable_v1"]

    assert current["policy_id"] == "current_release_v1"
    assert full["policy_id"] == "full_vision_v1"
    assert shipped["policy_id"] == "shipped_releasable_v1"
    assert current["matrix_path"] == "schemas/readiness/full-vision-matrix.yaml"
    assert full["matrix_path"] == "schemas/readiness/full-vision-matrix.yaml"
    assert shipped["matrix_path"] == "schemas/readiness/full-vision-matrix.yaml"
    assert "README.md" in current["scope_authority_refs"]
    assert {"docs/system.md", "docs/benchmarks-readiness.md"} <= set(
        full["scope_authority_refs"]
    )


def test_release_status_blocks_current_and_shipped_when_clean_install_proof_missing(
    project_root: Path,
    tmp_path: Path,
) -> None:
    clean_install_report = (
        project_root / "build" / "reports" / "clean-install-certification.json"
    )
    original_clean_install_report = (
        clean_install_report.read_text(encoding="utf-8")
        if clean_install_report.is_file()
        else None
    )
    if clean_install_report.exists():
        clean_install_report.unlink()

    try:
        status = release.get_release_status(
            project_root=project_root,
            benchmark_root=tmp_path / "benchmarks",
            notebook_output_root=tmp_path / "notebook-smoke",
        )

        assert status.target_ready is False
        assert set(status.policy_judgments) == {
            "current_release_v1",
            "full_vision_v1",
            "shipped_releasable_v1",
        }

        current_release = status.policy_judgments["current_release_v1"]
        full_vision = status.policy_judgments["full_vision_v1"]
        shipped_releasable = status.policy_judgments["shipped_releasable_v1"]

        assert current_release.final_verdict == "blocked"
        assert any(
            "determinism.same_seed" in code for code in current_release.reason_codes
        )
        assert any(
            "performance.runtime_smoke" in code for code in current_release.reason_codes
        )

        assert full_vision.final_verdict == "blocked"
        assert any(
            code.startswith("clean_install_surface.")
            for code in full_vision.reason_codes
        )

        assert shipped_releasable.final_verdict == "blocked"
        assert any(
            "determinism.same_seed" in code
            for code in shipped_releasable.reason_codes
        )
        assert any(
            "performance.runtime_smoke" in code
            for code in shipped_releasable.reason_codes
        )
        assert status.shipped_releasable_judgment.final_verdict == "blocked"
    finally:
        if original_clean_install_report is not None:
            clean_install_report.parent.mkdir(parents=True, exist_ok=True)
            clean_install_report.write_text(
                original_clean_install_report,
                encoding="utf-8",
            )
