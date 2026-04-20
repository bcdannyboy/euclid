from __future__ import annotations

from pathlib import Path

import pytest

from euclid.operator_runtime.resources import (
    default_run_output_root,
    resolve_asset_root,
    resolve_contract_root,
    resolve_example_path,
    resolve_notebook_path,
)


def test_packaged_resource_lookup_resolves_certified_assets(tmp_path: Path) -> None:
    asset_root = resolve_asset_root()
    contract_root = resolve_contract_root()
    current_release_example = resolve_example_path("current_release_run.yaml")
    full_vision_example = resolve_example_path("full_vision_run.yaml")
    notebook_path = resolve_notebook_path()

    assert asset_root == contract_root
    assert current_release_example.is_file()
    assert full_vision_example.is_file()
    assert notebook_path.name == "current_release.ipynb"
    assert default_run_output_root("current-release-run", project_root=tmp_path) == (
        tmp_path / "build" / "operator" / "current-release-run"
    )


def test_missing_packaged_asset_raises_explicit_error() -> None:
    with pytest.raises(FileNotFoundError, match="missing packaged example"):
        resolve_example_path("missing-example.yaml")
