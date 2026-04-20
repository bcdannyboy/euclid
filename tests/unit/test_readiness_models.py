from __future__ import annotations

from pathlib import Path

from euclid.contracts.loader import load_contract_catalog
from euclid.manifests.runtime_models import (
    ReadinessGateRecord,
    ReadinessJudgmentManifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
READINESS_SCHEMA_PATH = PROJECT_ROOT / "schemas/readiness/euclid-readiness.yaml"


def test_readiness_judgment_manifest_roundtrip_preserves_gate_records() -> None:
    catalog = load_contract_catalog(PROJECT_ROOT)
    model = ReadinessJudgmentManifest(
        object_id="demo_readiness_judgment",
        judgment_id="demo_readiness_judgment",
        final_verdict="review_required",
        catalog_scope="internal",
        verdict_summary=(
            "Required release gates passed but operator review is still "
            "required."
        ),
        reason_codes=("performance.runtime_smoke_missing",),
        judged_at="2026-04-14T00:00:00Z",
        required_gate_count=3,
        passed_gate_count=2,
        failed_gate_count=0,
        missing_gate_count=1,
        gate_records=(
            ReadinessGateRecord(
                gate_id="contracts.catalog",
                status="passed",
                required=True,
                summary="Contract catalog loaded successfully.",
                evidence={"schema_count": 42},
            ),
            ReadinessGateRecord(
                gate_id="performance.runtime_smoke",
                status="missing",
                required=True,
                summary="Runtime smoke gate has not been captured yet.",
                evidence={},
            ),
        ),
    )

    manifest = model.to_manifest(catalog)
    restored = ReadinessJudgmentManifest.from_manifest(manifest)

    assert restored == model


def test_readiness_schema_declares_release_verdicts_and_required_gates() -> None:
    import yaml

    payload = yaml.safe_load(READINESS_SCHEMA_PATH.read_text(encoding="utf-8"))

    assert payload["kind"] == "euclid_readiness_policy"
    assert {item["id"] for item in payload["verdicts"]} == {
        "ready",
        "review_required",
        "blocked",
    }
    assert {
        item["gate_id"]
        for item in payload["retained_core_release"]["required_gates"]
    } == {
        "contracts.catalog",
        "benchmarks.rediscovery",
        "benchmarks.predictive_generalization",
        "benchmarks.adversarial_honesty",
        "notebook.smoke",
        "determinism.same_seed",
        "performance.runtime_smoke",
    }
