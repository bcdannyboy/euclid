from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
CERTIFIED_ENTRYPOINTS = (
    SRC_ROOT / "euclid/cli/run.py",
    SRC_ROOT / "euclid/cli/replay.py",
    SRC_ROOT / "euclid/operator_runtime/run.py",
    SRC_ROOT / "euclid/operator_runtime/replay.py",
    SRC_ROOT / "euclid/operator_runtime/resources.py",
    SRC_ROOT / "euclid/release.py",
    SRC_ROOT / "euclid/runtime/profiling.py",
    SRC_ROOT / "euclid/contracts/loader.py",
)
COMPATIBILITY_ONLY_MODULES = {
    SRC_ROOT / "euclid/demo.py",
    SRC_ROOT / "euclid/inspection.py",
    SRC_ROOT / "euclid/__init__.py",
    SRC_ROOT / "euclid/cli/__init__.py",
    SRC_ROOT / "euclid/prototype/__init__.py",
    SRC_ROOT / "euclid/prototype/intake_planning.py",
    SRC_ROOT / "euclid/prototype/workflow.py",
}
BANNED_MODULE_PREFIXES = ("euclid.demo", "euclid.prototype")


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
        elif isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
    return modules


def _module_path(module_name: str) -> Path | None:
    if not module_name.startswith("euclid."):
        return None
    relative_module = module_name.replace(".", "/")
    file_candidate = SRC_ROOT / f"{relative_module}.py"
    if file_candidate.is_file():
        return file_candidate
    package_candidate = SRC_ROOT / relative_module / "__init__.py"
    if package_candidate.is_file():
        return package_candidate
    return None


def _transitive_import_graph(start_paths: tuple[Path, ...]) -> dict[Path, set[str]]:
    graph: dict[Path, set[str]] = {}
    pending = list(start_paths)
    while pending:
        path = pending.pop()
        if path in graph:
            continue
        imports = _imports(path)
        graph[path] = imports
        for module_name in imports:
            module_path = _module_path(module_name)
            if module_path is not None and module_path not in graph:
                pending.append(module_path)
    return graph


def test_certified_runtime_modules_do_not_import_demo_or_prototype() -> None:
    graph = _transitive_import_graph(CERTIFIED_ENTRYPOINTS)
    for path, imports in graph.items():
        for module_name in imports:
            assert not module_name.startswith(BANNED_MODULE_PREFIXES), (
                f"{path.relative_to(REPO_ROOT)} still routes through banned module "
                f"{module_name}"
            )


def test_demo_dependencies_remain_confined_to_compatibility_only_modules() -> None:
    offenders: list[str] = []
    for path in sorted((SRC_ROOT / "euclid").rglob("*.py")):
        imports = _imports(path)
        if not any(
            module_name.startswith(BANNED_MODULE_PREFIXES) for module_name in imports
        ):
            continue
        if path not in COMPATIBILITY_ONLY_MODULES:
            offenders.append(path.relative_to(REPO_ROOT).as_posix())
    assert offenders == [], (
        "demo/prototype imports escaped the compatibility-only surface: "
        + ", ".join(offenders)
    )


def test_release_truth_surfaces_do_not_route_through_demo_or_prototype() -> None:
    graph = _transitive_import_graph((SRC_ROOT / "euclid/release.py",))
    banned_paths = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in graph
        if path not in {SRC_ROOT / "euclid/release.py"}
        and path not in COMPATIBILITY_ONLY_MODULES
        and any(
            module_name.startswith(BANNED_MODULE_PREFIXES)
            for module_name in graph[path]
        )
    ]
    assert banned_paths == []
    assert all(
        not any(module_name.startswith(BANNED_MODULE_PREFIXES) for module_name in imports)
        for imports in graph.values()
    ), "release truth surfaces still depend on demo/prototype imports"
