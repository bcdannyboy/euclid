from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = REPO_ROOT / "fixtures/canonical/golden-pack"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _copy_fixture(tmp_path: Path) -> Path:
    destination = tmp_path / "golden-pack"
    shutil.copytree(FIXTURE_ROOT, destination)
    return destination


def test_build_pack_generates_expected_artifacts_for_fixture(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    source_root = _copy_fixture(tmp_path)
    output_root = tmp_path / "build"

    result = build_pack(source_root=source_root, output_root=output_root)

    assert result.markdown_path == output_root / "euclid-canonical-pack.md"
    assert result.json_path == output_root / "euclid-canonical-pack.json"
    assert result.markdown_path.is_file()
    assert result.json_path.is_file()

    payload = json.loads(result.json_path.read_text())
    expected_payload = json.loads((FIXTURE_ROOT / "expected-build.json").read_text())
    assert payload == expected_payload

    markdown = result.markdown_path.read_text()
    expected_markdown = (FIXTURE_ROOT / "expected-build.md").read_text()
    assert markdown == expected_markdown


def test_build_pack_loads_contract_and_readiness_artifacts_for_fixture(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    source_root = _copy_fixture(tmp_path)
    result = build_pack(source_root=source_root, output_root=tmp_path / "build")

    payload = json.loads(result.json_path.read_text())

    assert payload["validation_summary"]["contract_artifacts_loaded"] == 2
    assert payload["validation_summary"]["owners_declared"] == 3
    assert [contract["path"] for contract in payload["contracts"]] == [
        "schemas/contracts/module-ownership.json",
        "schemas/readiness/readiness-charter.yaml",
    ]


def test_build_pack_detects_missing_required_reference(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(tmp_path)
    system_path = source_root / "schemas/core/euclid-system.yaml"
    system_text = system_path.read_text()
    system_path.write_text(
        system_text.replace(
            "docs/canonical/system-map.md",
            "docs/canonical/missing-system-map.md",
            1,
        )
    )

    with pytest.raises(SpecCompilerError, match="missing required reference"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_detects_unknown_enum_values(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(tmp_path)
    contract_path = source_root / "schemas/contracts/module-ownership.json"
    payload = json.loads(contract_path.read_text())
    payload["contracts"][0]["claim_lane"] = "unsupported_lane"
    contract_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(SpecCompilerError, match="unknown enum"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_build_pack_detects_duplicate_owner_ids(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import SpecCompilerError, build_pack

    source_root = _copy_fixture(tmp_path)
    contract_path = source_root / "schemas/contracts/module-ownership.json"
    payload = json.loads(contract_path.read_text())
    payload["owners"].append({"id": "control_plane", "display_name": "Duplicate Control Plane"})
    contract_path.write_text(json.dumps(payload, indent=2) + "\n")

    with pytest.raises(SpecCompilerError, match="duplicate owner"):
        build_pack(source_root=source_root, output_root=tmp_path / "build")


def test_cli_build_entrypoint_writes_requested_output_directory(tmp_path: Path) -> None:
    source_root = _copy_fixture(tmp_path)
    output_root = tmp_path / "cli-build"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.spec_compiler",
            "build",
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output_root / "euclid-canonical-pack.md").is_file()
    assert (output_root / "euclid-canonical-pack.json").is_file()
    assert "euclid-canonical-pack.json" in completed.stdout


def test_live_repo_canonical_spine_compiles_without_manual_intervention(tmp_path: Path) -> None:
    from tools.spec_compiler.compiler import build_pack

    result = build_pack(source_root=REPO_ROOT, output_root=tmp_path / "repo-build")
    payload = json.loads(result.json_path.read_text())

    assert payload["project_name"] == "Euclid"
    assert payload["validation_summary"]["required_refs_checked"] == 17
    assert payload["validation_summary"]["closed_vocabularies_loaded"] == 7
