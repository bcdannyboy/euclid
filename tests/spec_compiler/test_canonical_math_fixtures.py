from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MATH_FIXTURE_ROOT = REPO_ROOT / "fixtures/canonical/math"
EXPECTED_FIXTURE_PATHS = [
    "fixtures/canonical/math/descriptive-only-analytic-piecewise.yaml",
    "fixtures/canonical/math/mechanistically-compatible-hypothesis.yaml",
    "fixtures/canonical/math/predictively-supported-distribution.yaml",
    "fixtures/canonical/math/predictively-supported-point.yaml",
    "fixtures/canonical/math/shared-plus-local-predictive.yaml",
]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_yaml(path: Path) -> dict:
    assert path.is_file(), f"missing required file: {path.relative_to(REPO_ROOT).as_posix()}"
    return yaml.safe_load(path.read_text())


def _copy_repo(tmp_path: Path) -> Path:
    destination = tmp_path / "repo-copy"
    shutil.copytree(
        REPO_ROOT,
        destination,
        ignore=shutil.ignore_patterns(".DS_Store", "__pycache__", ".pytest_cache", "build"),
    )
    return destination


def test_canonical_math_fixtures_exist_and_cover_required_paths() -> None:
    assert MATH_FIXTURE_ROOT.is_dir(), "fixtures/canonical/math must exist"

    payloads = {}
    for relative_path in EXPECTED_FIXTURE_PATHS:
        path = REPO_ROOT / relative_path
        payload = _load_yaml(path)
        assert payload["kind"] == "canonical_math_fixture"
        assert payload["version"] == 1
        payloads[relative_path] = payload

    fixtures = list(payloads.values())

    assert any(
        fixture["claim_lane"] == "descriptive_only" and "predictive_gate_policy_id" not in fixture
        for fixture in fixtures
    )
    assert any(
        fixture["claim_lane"] == "predictively_supported" and fixture["forecast_object_type"] == "point"
        for fixture in fixtures
    )
    assert any(
        fixture["claim_lane"] == "predictively_supported"
        and fixture["forecast_object_type"] in {"distribution", "interval", "quantile", "event_probability"}
        for fixture in fixtures
    )
    assert any(fixture["claim_lane"] == "mechanistically_compatible_hypothesis" for fixture in fixtures)
    assert any(
        fixture["reducer_object"]["composition_operator"] == "shared_plus_local_decomposition"
        for fixture in fixtures
    )


def test_live_repo_build_reports_math_fixture_inventory(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    result = build_pack(source_root=REPO_ROOT, output_root=tmp_path / "repo-build")
    payload = json.loads(result.json_path.read_text())

    assert payload["validation_summary"]["math_fixtures_loaded"] == len(EXPECTED_FIXTURE_PATHS)
    assert [fixture["path"] for fixture in payload["math_fixtures"]] == EXPECTED_FIXTURE_PATHS


def test_build_pack_rejects_unknown_enum_inside_math_fixture(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    fixture_path = source_root / "fixtures/canonical/math/predictively-supported-point.yaml"
    payload = yaml.safe_load(fixture_path.read_text())
    payload["claim_lane"] = "unsupported_lane"
    fixture_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(SpecCompilerError, match="unknown enum"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_rejects_unknown_math_object_reference(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    fixture_path = source_root / "fixtures/canonical/math/descriptive-only-analytic-piecewise.yaml"
    payload = yaml.safe_load(fixture_path.read_text())
    payload["descriptive_admissibility_object"]["reducer_object_ref"] = "missing_reducer_object"
    fixture_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(SpecCompilerError, match="unknown math object"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_requires_predictive_policy_bindings_for_predictive_fixtures(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_repo(tmp_path)
    fixture_path = source_root / "fixtures/canonical/math/predictively-supported-distribution.yaml"
    payload = yaml.safe_load(fixture_path.read_text())
    payload.pop("predictive_gate_policy_id")
    fixture_path.write_text(yaml.safe_dump(payload, sort_keys=False))

    with pytest.raises(SpecCompilerError, match="predictive_gate_policy_id"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")
