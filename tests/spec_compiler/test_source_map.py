from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MAP_PATH = REPO_ROOT / "schemas/core/source-map.yaml"
REQUIRED_DIRECTORIES = [
    REPO_ROOT / "docs/reference",
    REPO_ROOT / "schemas/core",
    REPO_ROOT / "schemas/contracts",
    REPO_ROOT / "schemas/readiness",
    REPO_ROOT / "benchmarks",
    REPO_ROOT / "tests/spec_compiler",
    REPO_ROOT / "tools/spec_compiler",
]
ALLOWED_STATUSES = {
    "entrypoint",
    "certified_surface",
    "compatibility_surface",
    "modeling_surface",
    "search_surface",
    "authority_surface",
    "ui_surface",
    "truthfulness_surface",
    "executable_reference",
}


def _load_source_map() -> dict:
    return yaml.safe_load(SOURCE_MAP_PATH.read_text(encoding="utf-8"))


def test_required_reference_directories_exist() -> None:
    missing = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in REQUIRED_DIRECTORIES
        if not path.is_dir()
    ]
    assert not missing, f"missing reference directories: {missing}"


def test_source_map_uses_known_status_values() -> None:
    entries = _load_source_map()["entries"]
    statuses = {entry["status"] for entry in entries}
    assert statuses <= ALLOWED_STATUSES


def test_source_map_targets_resolve_in_repo() -> None:
    missing_targets = []
    for entry in _load_source_map()["entries"]:
        for target in entry["canonical_targets"]:
            if not (REPO_ROOT / target).exists():
                missing_targets.append((entry["source"], target))

    assert not missing_targets, (
        "source-map canonical targets must resolve in-repo: "
        + ", ".join(f"{source} -> {target}" for source, target in missing_targets)
    )
