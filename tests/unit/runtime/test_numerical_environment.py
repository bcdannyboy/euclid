from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata

from euclid.modules.replay import build_runtime_environment_metadata
from euclid.runtime.numerical_environment import (
    capture_numerical_environment,
    compare_numerical_environments,
)

CORE_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sympy": "sympy",
    "pint": "pint",
    "statsmodels": "statsmodels",
    "sklearn": "scikit-learn",
    "pysindy": "pysindy",
    "pysr": "pysr",
    "egglog": "egglog",
    "joblib": "joblib",
    "sqlalchemy": "sqlalchemy",
    "pydantic": "pydantic",
    "yaml": "PyYAML",
    "typer": "typer",
    "pyarrow": "pyarrow",
    "httpx": "httpx",
    "dotenv": "python-dotenv",
    "vcr": "vcrpy",
    "responses": "responses",
    "respx": "respx",
    "pytest_timeout": "pytest-timeout",
    "xdist": "pytest-xdist",
    "hypothesis": "hypothesis",
}


def test_scientific_and_gate_dependencies_are_importable() -> None:
    missing = []
    for module_name, package_name in CORE_IMPORTS.items():
        try:
            if module_name == "pysr":
                importlib_metadata.version("pysr")
            else:
                importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - failure message path.
            missing.append(f"{package_name}: {type(exc).__name__}: {exc}")

    assert not missing


def test_capture_numerical_environment_records_versions_and_runtime_metadata() -> None:
    environment = capture_numerical_environment()

    assert environment["schema_version"] == "numerical-runtime-v1"
    assert environment["python"]["version"]
    assert environment["platform"]["machine"] or environment["platform"]["processor"]
    assert "blas_lapack" in environment
    for package_name in (
        "numpy",
        "pandas",
        "scipy",
        "sympy",
        "pint",
        "statsmodels",
        "scikit-learn",
        "pysindy",
        "pysr",
        "egglog",
    ):
        assert environment["libraries"][package_name]["status"] == "available"
        assert environment["libraries"][package_name]["version"]
    assert "julia" in environment


def test_capture_numerical_environment_uses_metadata_without_heavy_imports(
    monkeypatch,
) -> None:
    imported_modules: list[str] = []
    original_import_module = importlib.import_module

    def _recording_import(module_name: str):
        imported_modules.append(module_name)
        return original_import_module(module_name)

    monkeypatch.setattr(
        "euclid.runtime.numerical_environment.importlib.import_module",
        _recording_import,
    )

    environment = capture_numerical_environment()

    assert environment["libraries"]["scipy"]["status"] == "available"
    assert not {
        "pandas",
        "scipy",
        "sympy",
        "statsmodels",
        "sklearn",
        "pysindy",
        "pysr",
        "egglog",
        "joblib",
    } & set(imported_modules)


def test_compare_numerical_environments_reports_replay_mismatch_diagnostics() -> None:
    first = capture_numerical_environment()
    second = {
        **first,
        "libraries": {
            **first["libraries"],
            "numpy": {**first["libraries"]["numpy"], "version": "0.0-mismatch"},
        },
    }

    diagnostics = compare_numerical_environments(first, second)

    assert diagnostics["status"] == "mismatch"
    assert diagnostics["reason_codes"] == ["library_version_mismatch"]
    assert diagnostics["mismatches"]["libraries.numpy.version"]["expected"]


def test_replay_environment_metadata_includes_numerical_versions() -> None:
    metadata = build_runtime_environment_metadata()

    assert metadata["numerical_schema_version"] == "numerical-runtime-v1"
    assert metadata["library.numpy.status"] == "available"
    assert metadata["library.scipy.version"]
    assert "blas_lapack.status" in metadata
