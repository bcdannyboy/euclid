#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback.
    tomllib = None

PLAN_PATH = Path("docs/plans/2026-04-21-euclid-enhancement-master-plan.md")
PHASE_IDS = tuple(f"P{index:02d}" for index in range(17))

REQUIRED_ENV_EXAMPLE_NAMES = (
    "EUCLID_LIVE_API_TESTS",
    "EUCLID_LIVE_API_STRICT",
    "FMP_API_KEY",
    "OPENAI_API_KEY",
    "EUCLID_OPENAI_EXPLAINER_MODEL",
    "EUCLID_LIVE_TEST_TIMEOUT_SECONDS",
    "EUCLID_LIVE_ARTIFACT_DIR",
)
REQUIRED_RUNTIME_DEPENDENCIES = (
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
    "joblib",
    "sqlalchemy",
    "pydantic",
    "pyyaml",
    "typer",
    "pyarrow",
    "httpx",
    "python-dotenv",
)
REQUIRED_DEV_DEPENDENCIES = (
    "pytest",
    "pytest-cov",
    "pytest-timeout",
    "pytest-xdist",
    "hypothesis",
    "responses",
    "respx",
    "vcrpy",
)
REQUIRED_SCAFFOLD_PATHS = (
    ".env.example",
    "docs/reference/live-api-test-policy.md",
    "schemas/contracts/live-api-evidence.yaml",
    "schemas/contracts/fixture-provenance.yaml",
    "scripts/live_api_smoke.sh",
    "src/euclid/runtime/env.py",
    "src/euclid/testing/fixtures.py",
    "src/euclid/testing/gate_manifest.py",
    "src/euclid/testing/live_api.py",
    "src/euclid/testing/redaction.py",
    "tests/live/README.md",
)
REQUIRED_GATE_SECTIONS = (
    "covered_ids",
    "fixture_unit",
    "fixture_integration",
    "fixture_regression",
    "live_api",
    "redaction",
    "replay",
    "claim_scope",
    "edge_cases",
    "id_gates_schema",
    "id_gates",
)
REQUIRED_ID_GATE_FIELDS = (
    "status",
    "implementation_files",
    "test_files",
    "gate_refs",
    "evidence_refs",
    "edge_cases",
    "redaction_assertions",
    "replay_assertions",
    "claim_scope_assertions",
)
LEGACY_SHIM_PATTERNS = (
    (
        "legacy.sindy_shim",
        Path("src/euclid/adapters/sparse_library.py"),
        ("sindy-sparse-library", "SparseLibraryAdapter"),
    ),
    (
        "legacy.decomposition_shim",
        Path("src/euclid/adapters/decomposition.py"),
        ("ai_feynman-decomposition", "DecompositionAdapter"),
    ),
    (
        "legacy.sort_only_equality_saturation",
        Path("src/euclid/search/backends.py"),
        ("EqualitySaturationHeuristicBackend", "_equality_extractor_sort_key"),
    ),
)


@dataclass(frozen=True)
class Finding:
    check_id: str
    status: str
    message: str
    detail: str = ""


@dataclass(frozen=True)
class AuditReport:
    project_root: str
    findings: tuple[Finding, ...]

    @property
    def failed_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == "failed")

    @property
    def warning_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == "warning")

    @property
    def passed_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == "passed")


def extract_plan_ids(plan_text: str) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {phase_id: set() for phase_id in PHASE_IDS}
    for match in re.finditer(r"\b(P\d{2}-T\d{2}(?:-S\d{2})?)\b", plan_text):
        item_id = match.group(1)
        grouped.setdefault(item_id[:3], set()).add(item_id)
    return grouped


def run_audit(
    project_root: Path,
    *,
    run_fixtures: bool,
    run_live: bool,
    strict_warnings: bool = False,
) -> AuditReport:
    project_root = project_root.resolve()
    findings: list[Finding] = []
    plan_text = _read_text(project_root / PLAN_PATH)
    plan_ids = extract_plan_ids(plan_text) if plan_text is not None else {}

    findings.extend(_check_plan(project_root, plan_text))
    findings.extend(_check_gate_manifests(project_root, plan_ids))
    findings.extend(_check_env_example(project_root))
    findings.extend(_check_pyproject_dependencies(project_root))
    findings.extend(_check_scaffold_paths(project_root))
    findings.extend(_check_legacy_removal(project_root))
    if run_fixtures:
        findings.extend(_run_fixture_commands(project_root))
    if run_live:
        findings.extend(_run_live_commands(project_root))

    if strict_warnings:
        findings = [
            Finding(
                finding.check_id,
                "failed" if finding.status == "warning" else finding.status,
                finding.message,
                finding.detail,
            )
            for finding in findings
        ]
    return AuditReport(project_root=str(project_root), findings=tuple(findings))


def _check_plan(project_root: Path, plan_text: str | None) -> list[Finding]:
    if plan_text is None:
        return [_failed("plan.exists", f"Missing plan file: {PLAN_PATH}")]
    findings = [_passed("plan.exists", f"Found {PLAN_PATH}")]
    for phase_id in PHASE_IDS:
        if f"{phase_id}:" in plan_text or f"{phase_id} " in plan_text:
            findings.append(_passed("plan.phase", f"{phase_id} appears in plan"))
        else:
            findings.append(_failed("plan.phase", f"{phase_id} missing from plan"))
    for heading in (
        "Final 100% Completion Checklist",
        "Mandatory Phase-Level Dual Gates",
        "Task And Subtask Gate Manifest Requirements",
        "Release Certification Dual-Gate Command Set",
    ):
        if heading in plan_text:
            findings.append(_passed("plan.section", f"Found section: {heading}"))
        else:
            findings.append(_failed("plan.section", f"Missing section: {heading}"))
    return findings


def _check_gate_manifests(
    project_root: Path,
    plan_ids: Mapping[str, set[str]],
) -> list[Finding]:
    findings: list[Finding] = []
    for phase_id in PHASE_IDS:
        manifest_path = project_root / "tests" / "gates" / f"{phase_id}.yaml"
        relative_manifest = manifest_path.relative_to(project_root)
        required_ids = plan_ids.get(phase_id, set())
        if not manifest_path.is_file():
            findings.append(
                _failed(
                    "gate_manifest.exists",
                    f"Missing {relative_manifest}",
                    detail=f"{len(required_ids)} plan ids require coverage",
                )
            )
            continue
        payload = _read_yaml_mapping(manifest_path)
        if payload is None:
            findings.append(
                _failed(
                    "gate_manifest.parse",
                    f"{relative_manifest} is not valid YAML mapping",
                )
            )
            continue
        if payload.get("phase_id") != phase_id:
            findings.append(
                _failed(
                    "gate_manifest.phase_id",
                    f"{relative_manifest} has wrong phase_id",
                    detail=str(payload.get("phase_id")),
                )
            )
        else:
            findings.append(
                _passed(
                    "gate_manifest.phase_id",
                    f"{relative_manifest} declares {phase_id}",
                )
            )
        for section in REQUIRED_GATE_SECTIONS:
            if section not in payload:
                findings.append(
                    _failed(
                        "gate_manifest.section",
                        f"{relative_manifest} missing {section}",
                    )
                )
            else:
                findings.append(
                    _passed(
                        "gate_manifest.section",
                        f"{relative_manifest} has {section}",
                    )
                )
        covered_ids = {
            str(item).strip()
            for item in payload.get("covered_ids", ())
            if str(item).strip()
        }
        missing_ids = sorted(required_ids - covered_ids)
        if missing_ids:
            findings.append(
                _failed(
                    "gate_manifest.coverage",
                    f"{relative_manifest} missing plan ids",
                    detail=", ".join(missing_ids[:40])
                    + (" ..." if len(missing_ids) > 40 else ""),
                )
            )
        else:
            findings.append(
                _passed(
                    "gate_manifest.coverage",
                    f"{relative_manifest} covers {len(required_ids)} plan ids",
                )
            )
        id_gates = payload.get("id_gates", {})
        id_gates_schema = {
            str(field).strip()
            for field in payload.get("id_gates_schema", ())
            if str(field).strip()
        }
        missing_schema_fields = sorted(set(REQUIRED_ID_GATE_FIELDS) - id_gates_schema)
        if missing_schema_fields:
            findings.append(
                _failed(
                    "gate_manifest.id_gates_schema",
                    f"{relative_manifest} id_gates_schema is incomplete",
                    detail=", ".join(missing_schema_fields),
                )
            )
        else:
            findings.append(
                _passed(
                    "gate_manifest.id_gates_schema",
                    f"{relative_manifest} declares per-ID gate schema",
                )
            )
        if not isinstance(id_gates, Mapping):
            findings.append(
                _failed(
                    "gate_manifest.id_gates",
                    f"{relative_manifest} id_gates is not a mapping",
                )
            )
            continue
        missing_gate_rows = sorted(required_ids - {str(key) for key in id_gates})
        if missing_gate_rows:
            findings.append(
                _failed(
                    "gate_manifest.id_gates",
                    f"{relative_manifest} missing per-ID gate rows",
                    detail=", ".join(missing_gate_rows[:40])
                    + (" ..." if len(missing_gate_rows) > 40 else ""),
                )
            )
            continue
        blank_fields: list[str] = []
        for item_id in sorted(required_ids):
            row = id_gates.get(item_id)
            if not isinstance(row, Mapping):
                blank_fields.append(f"{item_id}:row")
                continue
            for field in REQUIRED_ID_GATE_FIELDS:
                if field not in row or not row[field]:
                    blank_fields.append(f"{item_id}:{field}")
        if blank_fields:
            findings.append(
                _failed(
                    "gate_manifest.id_gates",
                    f"{relative_manifest} has blank per-ID gate fields",
                    detail=", ".join(blank_fields[:40])
                    + (" ..." if len(blank_fields) > 40 else ""),
                )
            )
        else:
            findings.append(
                _passed(
                    "gate_manifest.id_gates",
                    f"{relative_manifest} has non-empty checks for every plan id",
                )
            )
    return findings


def _check_env_example(project_root: Path) -> list[Finding]:
    env_example = project_root / ".env.example"
    text = _read_text(env_example)
    if text is None:
        return [_failed("env_example.exists", "Missing .env.example")]
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[key.strip()] = value.strip()
    findings: list[Finding] = []
    for name in REQUIRED_ENV_EXAMPLE_NAMES:
        if name not in parsed:
            findings.append(_failed("env_example.name", f"{name} missing"))
        elif parsed[name]:
            findings.append(
                _failed("env_example.blank", f"{name} must be blank in .env.example")
            )
        else:
            findings.append(_passed("env_example.name", f"{name} present and blank"))
    return findings


def _check_pyproject_dependencies(project_root: Path) -> list[Finding]:
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return [_failed("pyproject.exists", "Missing pyproject.toml")]
    payload = _load_pyproject(pyproject_path)
    runtime_deps = _dependency_names(payload.get("project", {}).get("dependencies", ()))
    dev_deps = _dependency_names(
        payload.get("project", {})
        .get("optional-dependencies", {})
        .get("dev", ())
    )
    findings: list[Finding] = []
    for name in REQUIRED_RUNTIME_DEPENDENCIES:
        if name in runtime_deps:
            findings.append(_passed("pyproject.runtime_dependency", f"{name} declared"))
        else:
            findings.append(
                _failed("pyproject.runtime_dependency", f"{name} missing")
            )
    for name in REQUIRED_DEV_DEPENDENCIES:
        if name in dev_deps:
            findings.append(_passed("pyproject.dev_dependency", f"{name} declared"))
        else:
            findings.append(_failed("pyproject.dev_dependency", f"{name} missing"))
    return findings


def _load_pyproject(pyproject_path: Path) -> Mapping[str, Any]:
    if tomllib is not None:
        with pyproject_path.open("rb") as handle:
            return tomllib.load(handle)
    text = pyproject_path.read_text(encoding="utf-8")
    return {
        "project": {
            "dependencies": _extract_toml_list(text, "dependencies"),
            "optional-dependencies": {
                "dev": _extract_toml_list(text, "dev"),
            },
        }
    }


def _extract_toml_list(text: str, key: str) -> tuple[str, ...]:
    match = re.search(rf"(?m)^{re.escape(key)}\s*=\s*\[(.*?)\]", text, re.DOTALL)
    if not match:
        return ()
    return tuple(
        item.strip().strip('"').strip("'")
        for item in match.group(1).split(",")
        if item.strip().strip('"').strip("'")
    )


def _check_scaffold_paths(project_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative in REQUIRED_SCAFFOLD_PATHS:
        path = project_root / relative
        if path.exists():
            findings.append(_passed("scaffold.exists", f"{relative} exists"))
        else:
            findings.append(_failed("scaffold.exists", f"{relative} missing"))
    return findings


def _check_legacy_removal(project_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for check_id, relative_path, patterns in LEGACY_SHIM_PATTERNS:
        path = project_root / relative_path
        if not path.exists():
            findings.append(_passed(check_id, f"{relative_path} removed"))
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        matched = [pattern for pattern in patterns if pattern in text]
        if matched:
            findings.append(
                _failed(
                    check_id,
                    f"{relative_path} still contains legacy shim markers",
                    detail=", ".join(matched),
                )
            )
        else:
            findings.append(
                _passed(
                    check_id,
                    f"{relative_path} contains no configured shim markers",
                )
            )
    return findings


def _run_fixture_commands(project_root: Path) -> list[Finding]:
    commands = (
        ("commands.unit", ("python3", "-m", "pytest", "-q", "tests/unit")),
        (
            "commands.integration",
            ("python3", "-m", "pytest", "-q", "tests/integration"),
        ),
        ("commands.regression", ("python3", "-m", "pytest", "-q", "tests/regression")),
        ("commands.golden", ("python3", "-m", "pytest", "-q", "tests/golden")),
        ("commands.benchmarks", ("python3", "-m", "pytest", "-q", "tests/benchmarks")),
        (
            "commands.spec_compiler",
            ("python3", "-m", "pytest", "-q", "tests/spec_compiler"),
        ),
        ("commands.release_status", ("python3", "-m", "euclid", "release", "status")),
        (
            "commands.verify_completion",
            ("python3", "-m", "euclid", "release", "verify-completion"),
        ),
        (
            "commands.certify_research_readiness",
            ("python3", "-m", "euclid", "release", "certify-research-readiness"),
        ),
    )
    return [
        _run_command(project_root, check_id, command)
        for check_id, command in commands
    ]


def _run_live_commands(project_root: Path) -> list[Finding]:
    env = os.environ.copy()
    env["EUCLID_LIVE_API_TESTS"] = "1"
    env["EUCLID_LIVE_API_STRICT"] = "1"
    commands = (
        (
            "commands.live_api_smoke",
            ("bash", "scripts/live_api_smoke.sh"),
        ),
        (
            "commands.live_pytest",
            ("python3", "-m", "pytest", "-q", "tests/live"),
        ),
    )
    return [
        _run_command(project_root, check_id, command, env=env)
        for check_id, command in commands
    ]


def _run_command(
    project_root: Path,
    check_id: str,
    command: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
) -> Finding:
    command_env = dict(os.environ if env is None else env)
    pythonpath = f"{project_root / 'src'}{os.pathsep}"
    pythonpath += command_env.get("PYTHONPATH", "")
    command_env["PYTHONPATH"] = pythonpath.rstrip(os.pathsep)
    result = subprocess.run(
        list(command),
        cwd=project_root,
        env=command_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    detail = _tail(result.stdout)
    if result.returncode == 0:
        return _passed(check_id, "Command passed", detail=" ".join(command))
    return _failed(
        check_id,
        f"Command failed with exit {result.returncode}: {' '.join(command)}",
        detail=detail,
    )


def _dependency_names(dependencies: Iterable[str]) -> set[str]:
    names = set()
    for dependency in dependencies:
        name = re.split(r"[<>=!~; \[]", str(dependency), maxsplit=1)[0]
        normalized = name.strip().lower().replace("_", "-")
        if normalized:
            names.add(normalized)
    return names


def _read_text(path: Path) -> str | None:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def _read_yaml_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return payload


def _tail(text: str, *, max_lines: int = 40) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-max_lines:])


def _passed(check_id: str, message: str, detail: str = "") -> Finding:
    return Finding(check_id=check_id, status="passed", message=message, detail=detail)


def _failed(check_id: str, message: str, detail: str = "") -> Finding:
    return Finding(check_id=check_id, status="failed", message=message, detail=detail)


def _warning(check_id: str, message: str, detail: str = "") -> Finding:
    return Finding(check_id=check_id, status="warning", message=message, detail=detail)


def format_report(report: AuditReport) -> str:
    lines = [
        "Euclid enhancement audit",
        f"Project root: {report.project_root}",
        (
            "Summary: "
            f"{report.passed_count} passed, "
            f"{report.warning_count} warnings, "
            f"{report.failed_count} failed"
        ),
        "",
    ]
    for finding in report.findings:
        marker = {"passed": "PASS", "warning": "WARN", "failed": "FAIL"}[
            finding.status
        ]
        lines.append(f"[{marker}] {finding.check_id}: {finding.message}")
        if finding.detail:
            lines.append(f"        {finding.detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Euclid enhancement-plan implementation readiness."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--run-fixtures",
        action="store_true",
        help="Run fixture/unit/integration/regression/benchmark/release commands.",
    )
    parser.add_argument(
        "--run-live",
        action="store_true",
        help="Run strict live API commands. Requires .env or CI secrets.",
    )
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="Treat warnings as failures.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for machine-readable JSON audit output.",
    )
    args = parser.parse_args(argv)

    report = run_audit(
        args.project_root,
        run_fixtures=args.run_fixtures,
        run_live=args.run_live,
        strict_warnings=args.strict_warnings,
    )
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(asdict(report), indent=2) + "\n",
            encoding="utf-8",
        )
    print(format_report(report))
    return 1 if report.failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
