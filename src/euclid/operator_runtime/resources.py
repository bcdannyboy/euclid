from __future__ import annotations

import atexit
import os
from contextlib import ExitStack
from importlib import resources
from pathlib import Path

_ASSET_STACK = ExitStack()
_PACKAGED_ASSET_ROOT: Path | None = None


class EuclidAssetError(FileNotFoundError):
    """Typed failure for canonical packaged asset resolution."""

    def __init__(self, message: str, *, code: str, asset_path: Path) -> None:
        self.code = code
        self.asset_path = asset_path
        super().__init__(message)


def _close_asset_stack() -> None:
    _ASSET_STACK.close()


atexit.register(_close_asset_stack)


def _configured_project_root(project_root: Path | str | None = None) -> Path | None:
    if project_root is not None:
        return Path(project_root).resolve()
    configured_root = os.environ.get("EUCLID_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).resolve()
    return None


def _has_runtime_assets(root: Path) -> bool:
    return (
        (root / "schemas" / "contracts" / "schema-registry.yaml").is_file()
        and (root / "examples").is_dir()
    )


def _asset_root_candidates(root: Path) -> tuple[Path, ...]:
    packaged_mirror = root / "src" / "euclid" / "_assets"
    candidates: list[Path] = []
    if _has_runtime_assets(packaged_mirror):
        candidates.append(packaged_mirror)
    if _has_runtime_assets(root):
        candidates.append(root)
    return tuple(candidates)


def _packaged_asset_root() -> Path:
    global _PACKAGED_ASSET_ROOT
    if _PACKAGED_ASSET_ROOT is None:
        traversable = resources.files("euclid._assets")
        _PACKAGED_ASSET_ROOT = _ASSET_STACK.enter_context(
            resources.as_file(traversable)
        )
    return _PACKAGED_ASSET_ROOT


def resolve_asset_root(project_root: Path | str | None = None) -> Path:
    configured_root = _configured_project_root(project_root)
    if configured_root is not None:
        candidates = _asset_root_candidates(configured_root)
        if candidates:
            return candidates[0]
    return _packaged_asset_root()


def resolve_workspace_root(project_root: Path | str | None = None) -> Path:
    configured_root = _configured_project_root(project_root)
    if configured_root is not None:
        return configured_root
    return Path.cwd().resolve()


def resolve_checkout_root(project_root: Path | str | None = None) -> Path:
    configured_root = _configured_project_root(project_root)
    if configured_root is not None and (configured_root / "pyproject.toml").is_file():
        return configured_root
    candidate = Path(__file__).resolve().parents[3]
    if (candidate / "pyproject.toml").is_file():
        return candidate
    return resolve_workspace_root(project_root)


def resolve_contract_root(project_root: Path | str | None = None) -> Path:
    root = resolve_asset_root(project_root)
    contract_registry = root / "schemas" / "contracts" / "schema-registry.yaml"
    if not contract_registry.is_file():
        raise EuclidAssetError(
            f"missing packaged contract catalog at {contract_registry}",
            code="euclid_asset_missing",
            asset_path=contract_registry,
        )
    return root


def resolve_example_path(
    name: str,
    *,
    project_root: Path | str | None = None,
) -> Path:
    path = resolve_asset_root(project_root) / "examples" / name
    if not path.is_file():
        raise EuclidAssetError(
            f"missing packaged example {name!r} at {path}",
            code="euclid_asset_missing",
            asset_path=path,
        )
    return path


def resolve_fixture_path(
    *relative_parts: str,
    project_root: Path | str | None = None,
) -> Path:
    path = resolve_asset_root(project_root) / "fixtures" / Path(*relative_parts)
    if not path.is_file():
        raise EuclidAssetError(
            f"missing packaged fixture {'/'.join(relative_parts)!r} at {path}",
            code="euclid_asset_missing",
            asset_path=path,
        )
    return path


def resolve_notebook_path(
    name: str = "current_release.ipynb",
    *,
    project_root: Path | str | None = None,
) -> Path:
    path = (
        resolve_asset_root(project_root)
        / "output"
        / "jupyter-notebook"
        / name
    )
    if not path.is_file():
        raise EuclidAssetError(
            f"missing packaged notebook {name!r} at {path}",
            code="euclid_asset_missing",
            asset_path=path,
        )
    return path


def default_run_output_root(
    run_id: str,
    *,
    project_root: Path | str | None = None,
) -> Path:
    return resolve_workspace_root(project_root) / "build" / "operator" / run_id


__all__ = [
    "EuclidAssetError",
    "default_run_output_root",
    "resolve_asset_root",
    "resolve_checkout_root",
    "resolve_contract_root",
    "resolve_example_path",
    "resolve_fixture_path",
    "resolve_notebook_path",
    "resolve_workspace_root",
]
