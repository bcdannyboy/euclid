from __future__ import annotations

import importlib
import importlib.metadata as importlib_metadata
import platform
import subprocess
import sys
from typing import Any, Mapping

_LIBRARIES: tuple[tuple[str, str, str], ...] = (
    ("numpy", "numpy", "numpy"),
    ("pandas", "pandas", "pandas"),
    ("scipy", "scipy", "scipy"),
    ("sympy", "sympy", "sympy"),
    ("pint", "pint", "pint"),
    ("statsmodels", "statsmodels", "statsmodels"),
    ("scikit-learn", "sklearn", "scikit-learn"),
    ("pysindy", "pysindy", "pysindy"),
    ("pysr", "pysr", "pysr"),
    ("egglog", "egglog", "egglog"),
    ("joblib", "joblib", "joblib"),
)


def capture_numerical_environment() -> dict[str, Any]:
    libraries = {
        name: _library_status(module, distribution)
        for name, module, distribution in _LIBRARIES
    }
    return {
        "schema_version": "numerical-runtime-v1",
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "libraries": libraries,
        "julia": _julia_runtime_status(),
        "blas_lapack": _blas_lapack_status(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "cpu": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_compiler": platform.python_compiler(),
        },
    }


def compare_numerical_environments(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, Any]:
    mismatches: dict[str, dict[str, Any]] = {}
    expected_libraries = expected.get("libraries", {})
    actual_libraries = actual.get("libraries", {})
    if isinstance(expected_libraries, Mapping) and isinstance(actual_libraries, Mapping):
        for library_name in sorted(set(expected_libraries) | set(actual_libraries)):
            expected_info = expected_libraries.get(library_name, {})
            actual_info = actual_libraries.get(library_name, {})
            if not isinstance(expected_info, Mapping) or not isinstance(actual_info, Mapping):
                continue
            expected_version = expected_info.get("version")
            actual_version = actual_info.get("version")
            if expected_version != actual_version:
                mismatches[f"libraries.{library_name}.version"] = {
                    "expected": expected_version,
                    "actual": actual_version,
                }
    return {
        "status": "mismatch" if mismatches else "matched",
        "reason_codes": ["library_version_mismatch"] if mismatches else [],
        "mismatches": mismatches,
    }


def flatten_numerical_environment(environment: Mapping[str, Any] | None = None) -> dict[str, str]:
    payload = dict(environment or capture_numerical_environment())
    flat: dict[str, str] = {
        "numerical_schema_version": str(payload.get("schema_version", "")),
    }
    python_info = payload.get("python", {})
    if isinstance(python_info, Mapping):
        flat["python_version"] = str(python_info.get("version", ""))
        flat["python_implementation"] = str(python_info.get("implementation", ""))
    libraries = payload.get("libraries", {})
    if isinstance(libraries, Mapping):
        for library_name, info in libraries.items():
            if isinstance(info, Mapping):
                flat[f"library.{library_name}.status"] = str(info.get("status", ""))
                flat[f"library.{library_name}.version"] = str(info.get("version", ""))
    julia = payload.get("julia", {})
    if isinstance(julia, Mapping):
        flat["julia.status"] = str(julia.get("status", ""))
        flat["julia.version"] = str(julia.get("version", ""))
    blas_lapack = payload.get("blas_lapack", {})
    if isinstance(blas_lapack, Mapping):
        flat["blas_lapack.status"] = str(blas_lapack.get("status", ""))
        flat["blas_lapack.summary"] = str(blas_lapack.get("summary", ""))
    platform_info = payload.get("platform", {})
    if isinstance(platform_info, Mapping):
        flat["platform.system"] = str(platform_info.get("system", ""))
        flat["platform.machine"] = str(platform_info.get("machine", ""))
        flat["platform.processor"] = str(platform_info.get("processor", ""))
    return flat


def _library_status(import_name: str, distribution_name: str) -> dict[str, str]:
    diagnostic = (
        "top_level_import_deferred_until_julia_runtime_available"
        if import_name == "pysr"
        else ""
    )
    try:
        version = importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return _import_library_status(
            import_name,
            distribution_name,
            diagnostic=diagnostic,
        )
    return {"status": "available", "version": version, "diagnostic": diagnostic}


def _import_library_status(
    import_name: str,
    distribution_name: str,
    *,
    diagnostic: str,
) -> dict[str, str]:
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:
        return {
            "status": "unavailable",
            "version": "",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }
    version = getattr(module, "__version__", None)
    if not isinstance(version, str) or not version:
        try:
            version = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            version = "unknown"
    fallback_diagnostic = "package_metadata_unavailable"
    if diagnostic:
        fallback_diagnostic = f"{diagnostic};{fallback_diagnostic}"
    return {
        "status": "available",
        "version": version,
        "diagnostic": fallback_diagnostic,
    }


def _julia_runtime_status() -> dict[str, str]:
    try:
        completed = subprocess.run(
            ["julia", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "status": "unavailable",
            "version": "",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }
    output = (completed.stdout or completed.stderr).strip()
    return {
        "status": "available" if completed.returncode == 0 else "unavailable",
        "version": output,
        "diagnostic": "" if completed.returncode == 0 else output,
    }


def _blas_lapack_status() -> dict[str, str]:
    try:
        import numpy as np
    except Exception as exc:
        return {
            "status": "unavailable",
            "summary": "",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }
    try:
        config = np.__config__
        show_config = getattr(config, "show", None)
        summary = "numpy_config_available" if callable(show_config) else "unknown"
    except Exception as exc:
        return {
            "status": "unavailable",
            "summary": "",
            "diagnostic": f"{type(exc).__name__}: {exc}",
        }
    return {"status": "available", "summary": summary, "diagnostic": ""}


__all__ = [
    "capture_numerical_environment",
    "compare_numerical_environments",
    "flatten_numerical_environment",
]
