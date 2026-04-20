from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from euclid.contracts.refs import TypedRef
from euclid.modules.claims import resolve_claim_publication
from euclid.modules.mechanistic_evidence import evaluate_mechanistic_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "fixtures/runtime/mechanistic"
POSITIVE_FIXTURE = FIXTURE_ROOT / "mechanistic-positive-evidence.yaml"
NEGATIVE_FIXTURE = FIXTURE_ROOT / "mechanistic-overlap-evidence.yaml"
POSITIVE_GOLDEN = FIXTURE_ROOT / "mechanistic-publication-golden.json"
NEGATIVE_GOLDEN = FIXTURE_ROOT / "mechanistic-downgrade-golden.json"


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _snapshot(path: Path, *, lower_claim_ceiling: str) -> dict[str, Any]:
    fixture = _load_yaml(path)
    evaluation = evaluate_mechanistic_evidence(
        mechanistic_evidence_id=str(fixture["bundle_id"]),
        candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        prediction_artifact_ref=_ref(
            "prediction_artifact_manifest@1.1.0", "prediction"
        ),
        external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0",
            str(fixture["bundle_id"]),
        ),
        lower_claim_ceiling=lower_claim_ceiling,
        term_bindings=tuple(fixture["term_bindings"]),
        term_units=tuple(fixture["term_units"]),
        invariance_checks=tuple(fixture["invariance_checks"]),
        external_evidence_records=tuple(fixture["raw_sources"]),
        predictive_evidence_refs=(
            _ref("point_score_result_manifest@1.0.0", "score_result"),
        ),
    )
    claim = resolve_claim_publication(
        scorecard_body={
            "descriptive_status": "passed",
            "descriptive_reason_codes": [],
            "predictive_status": "passed",
            "predictive_reason_codes": [],
            "mechanistic_status": (
                "passed"
                if evaluation.dossier.status == "passed"
                else "downgraded_to_predictively_supported"
            ),
            "mechanistic_reason_codes": list(evaluation.dossier.reason_codes),
            "forecast_object_type": "point",
        }
    )
    return {
        "dossier": {
            "status": evaluation.dossier.status,
            "resolved_claim_ceiling": evaluation.dossier.resolved_claim_ceiling,
            "reason_codes": list(evaluation.dossier.reason_codes),
        },
        "claim": {
            "claim_type": claim.claim_type,
            "claim_ceiling": claim.claim_ceiling,
            "allowed_interpretation_codes": list(claim.allowed_interpretation_codes),
        },
    }


def test_mechanistic_positive_publication_matches_golden_fixture() -> None:
    assert _snapshot(
        POSITIVE_FIXTURE,
        lower_claim_ceiling="predictively_supported",
    ) == json.loads(POSITIVE_GOLDEN.read_text(encoding="utf-8"))


def test_mechanistic_overlap_downgrade_matches_golden_fixture() -> None:
    assert _snapshot(
        NEGATIVE_FIXTURE,
        lower_claim_ceiling="predictively_supported",
    ) == json.loads(NEGATIVE_GOLDEN.read_text(encoding="utf-8"))
