from __future__ import annotations

from pathlib import Path

import pytest

from euclid.contracts.errors import ContractValidationError
from euclid.contracts.loader import load_contract_catalog
from euclid.contracts.refs import TypedRef
from euclid.manifests.runtime_models import (
    ArtifactHashRecord,
    ReadinessJudgmentManifest,
    ReproducibilityBundleManifest,
    SchemaLifecycleIntegrationClosureManifest,
    SeedRecord,
)
from euclid.modules.evaluation_governance import (
    ComparisonKey,
    build_comparison_universe,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _ref(schema_name: str, object_id: str) -> TypedRef:
    return TypedRef(schema_name=schema_name, object_id=object_id)


def _comparison_universe_manifest(*, include_comparator_records: bool):
    catalog = load_contract_catalog(PROJECT_ROOT)
    key = ComparisonKey(
        forecast_object_type="point",
        score_policy_ref=_ref("point_score_policy_manifest@1.0.0", "score_policy"),
        horizon_set=(1,),
        scored_origin_set_id="origins",
    )
    universe = build_comparison_universe(
        selected_candidate_id="candidate",
        baseline_id="constant_baseline",
        candidate_primary_score=0.1,
        baseline_primary_score=0.2,
        candidate_comparison_key=key,
        baseline_comparison_key=key,
        candidate_score_result_ref=_ref(
            "point_score_result_manifest@1.0.0", "candidate_score"
        ),
        baseline_score_result_ref=_ref(
            "point_score_result_manifest@1.0.0", "baseline_score"
        ),
        comparator_score_result_refs=(
            (
                _ref("point_score_result_manifest@1.0.0", "baseline_score"),
            )
            if include_comparator_records
            else ()
        ),
        paired_comparison_records=(
            (
                {
                    "comparator_id": "constant_baseline",
                    "comparator_kind": "baseline",
                    "comparison_status": "comparable",
                    "candidate_primary_score": 0.1,
                    "comparator_primary_score": 0.2,
                    "primary_score_delta": -0.1,
                    "mean_loss_differential": -0.1,
                    "score_result_ref": _ref(
                        "point_score_result_manifest@1.0.0", "baseline_score"
                    ).as_dict(),
                },
            )
            if include_comparator_records
            else ()
        ),
    )
    return universe.to_manifest(catalog)


def _bundle_manifest(*, replay_status: str = "verified"):
    catalog = load_contract_catalog(PROJECT_ROOT)
    bundle = ReproducibilityBundleManifest(
        object_id="demo_bundle",
        bundle_id="demo_bundle",
        scope_id="euclid_v1_binding_scope@1.0.0",
        bundle_mode="candidate_publication",
        dataset_snapshot_ref=_ref("dataset_snapshot_manifest@1.0.0", "snapshot"),
        feature_view_ref=_ref("feature_view_manifest@1.0.0", "feature_view"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        run_result_ref=_ref("run_result_manifest@1.1.0", "run_result"),
        required_manifest_refs=(
            _ref("reducer_artifact_manifest@1.0.0", "candidate"),
            _ref("scorecard_manifest@1.1.0", "scorecard"),
            _ref("claim_card_manifest@1.1.0", "claim"),
        ),
        artifact_hash_records=(
            ArtifactHashRecord(
                artifact_role="dataset_snapshot",
                sha256="sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            ),
        ),
        seed_records=(SeedRecord(seed_scope="search", seed_value="0"),),
        replay_verification_status=replay_status,
        failure_reason_codes=(
            ()
            if replay_status == "verified"
            else ("nondeterministic_replay",)
        ),
    )
    return bundle.to_manifest(catalog)


def _readiness_manifest(*, final_verdict: str = "ready"):
    catalog = load_contract_catalog(PROJECT_ROOT)
    return ReadinessJudgmentManifest(
        object_id="demo_readiness",
        judgment_id="demo_readiness",
        final_verdict=final_verdict,
        catalog_scope="public",
    ).to_manifest(catalog)


def _closure_manifest():
    catalog = load_contract_catalog(PROJECT_ROOT)
    return SchemaLifecycleIntegrationClosureManifest(
        object_id="demo_closure",
        closure_id="demo_closure",
        status="passed",
    ).to_manifest(catalog)


def test_build_run_result_manifest_for_abstention_omits_candidate_refs() -> None:
    from euclid.modules.catalog_publishing import build_run_result_manifest

    model = build_run_result_manifest(
        object_id="demo_run_result",
        run_id="demo_run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        publication_mode="abstention_only_publication",
        abstention_ref=_ref("abstention_manifest@1.1.0", "abstention"),
        primary_validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0", "validation_scope"
        ),
        prediction_artifact_refs=(
            _ref("prediction_artifact_manifest@1.1.0", "prediction"),
        ),
        primary_external_evidence_ref=_ref(
            "external_evidence_manifest@1.0.0", "external_bundle"
        ),
        robustness_report_refs=(
            _ref("robustness_report_manifest@1.1.0", "robustness"),
        ),
    )

    assert model.result_mode == "abstention_only_publication"
    assert model.forecast_object_type == "point"
    assert model.primary_validation_scope_ref == _ref(
        "validation_scope_manifest@1.0.0", "validation_scope"
    )
    assert model.primary_abstention_ref == _ref(
        "abstention_manifest@1.1.0", "abstention"
    )
    assert model.primary_external_evidence_ref == _ref(
        "external_evidence_manifest@1.0.0", "external_bundle"
    )
    assert model.primary_reducer_artifact_ref is None
    assert model.primary_scorecard_ref is None
    assert model.primary_claim_card_ref is None


def test_build_run_result_manifest_rejects_candidate_publication_without_claim_refs(
) -> None:
    from euclid.modules.catalog_publishing import build_run_result_manifest

    with pytest.raises(ContractValidationError) as excinfo:
        build_run_result_manifest(
            object_id="demo_run_result",
            run_id="demo_run_result",
            scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
            search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
            evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
            comparison_universe_ref=_ref(
                "comparison_universe_manifest@1.0.0", "comparison"
            ),
            evaluation_event_log_ref=_ref(
                "evaluation_event_log_manifest@1.0.0", "event_log"
            ),
            evaluation_governance_ref=_ref(
                "evaluation_governance_manifest@1.1.0", "governance"
            ),
            reproducibility_bundle_ref=_ref(
                "reproducibility_bundle_manifest@1.0.0", "bundle"
            ),
            publication_mode="candidate_publication",
            selected_candidate_ref=_ref(
                "reducer_artifact_manifest@1.0.0", "candidate"
            ),
        )

    assert excinfo.value.code == "candidate_publication_missing_required_refs"


def test_build_run_result_manifest_rejects_hidden_candidate_refs_on_abstention(
) -> None:
    from euclid.modules.catalog_publishing import build_run_result_manifest

    with pytest.raises(ContractValidationError) as excinfo:
        build_run_result_manifest(
            object_id="demo_run_result",
            run_id="demo_run_result",
            scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
            search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
            evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
            comparison_universe_ref=_ref(
                "comparison_universe_manifest@1.0.0", "comparison"
            ),
            evaluation_event_log_ref=_ref(
                "evaluation_event_log_manifest@1.0.0", "event_log"
            ),
            evaluation_governance_ref=_ref(
                "evaluation_governance_manifest@1.1.0", "governance"
            ),
            reproducibility_bundle_ref=_ref(
                "reproducibility_bundle_manifest@1.0.0", "bundle"
            ),
            publication_mode="abstention_only_publication",
            selected_candidate_ref=_ref(
                "reducer_artifact_manifest@1.0.0", "candidate"
            ),
            scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
            claim_card_ref=_ref("claim_card_manifest@1.1.0", "claim"),
            abstention_ref=_ref("abstention_manifest@1.1.0", "abstention"),
        )

    assert excinfo.value.code == "invalid_result_mode_payload"


def test_build_publication_record_requires_candidate_comparator_exposure() -> None:
    from euclid.modules.catalog_publishing import (
        build_publication_record_manifest,
        build_run_result_manifest,
    )

    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result = build_run_result_manifest(
        object_id="demo_run_result",
        run_id="demo_run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        publication_mode="candidate_publication",
        selected_candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        claim_card_ref=_ref("claim_card_manifest@1.1.0", "claim"),
        primary_validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0", "validation_scope"
        ),
    ).to_manifest(catalog)

    with pytest.raises(ContractValidationError) as excinfo:
        build_publication_record_manifest(
            object_id="demo_publication_record",
            publication_id="demo_publication_record",
            run_result_manifest=run_result,
            comparison_universe_manifest=_comparison_universe_manifest(
                include_comparator_records=False
            ),
            reproducibility_bundle_manifest=_bundle_manifest(),
            readiness_judgment_manifest=_readiness_manifest(),
            schema_lifecycle_integration_closure_ref=_closure_manifest().ref,
            catalog_scope="public",
            published_at="2026-04-12T00:00:00Z",
        )

    assert excinfo.value.code == "candidate_publication_requires_comparator_exposure"


def test_build_publication_record_blocks_public_scope_without_ready_judgment() -> None:
    from euclid.modules.catalog_publishing import (
        build_publication_record_manifest,
        build_run_result_manifest,
    )

    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result = build_run_result_manifest(
        object_id="demo_run_result",
        run_id="demo_run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        primary_validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0", "validation_scope"
        ),
        publication_mode="candidate_publication",
        selected_candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        claim_card_ref=_ref("claim_card_manifest@1.1.0", "claim"),
    ).to_manifest(catalog)

    with pytest.raises(ContractValidationError) as excinfo:
        build_publication_record_manifest(
            object_id="demo_publication_record",
            publication_id="demo_publication_record",
            run_result_manifest=run_result,
            comparison_universe_manifest=_comparison_universe_manifest(
                include_comparator_records=True
            ),
            reproducibility_bundle_manifest=_bundle_manifest(),
            readiness_judgment_manifest=_readiness_manifest(final_verdict="blocked"),
            schema_lifecycle_integration_closure_ref=_closure_manifest().ref,
            catalog_scope="public",
            published_at="2026-04-12T00:00:00Z",
        )

    assert excinfo.value.code == "public_catalog_requires_ready_judgment"


def test_build_publication_record_emits_abstention_mode_without_candidate_exposure(
) -> None:
    from euclid.modules.catalog_publishing import (
        build_publication_record_manifest,
        build_run_result_manifest,
    )

    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result = build_run_result_manifest(
        object_id="demo_run_result",
        run_id="demo_run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        publication_mode="abstention_only_publication",
        abstention_ref=_ref("abstention_manifest@1.1.0", "abstention"),
    ).to_manifest(catalog)

    publication = build_publication_record_manifest(
        object_id="demo_publication_record",
        publication_id="demo_publication_record",
        run_result_manifest=run_result,
        comparison_universe_manifest=_comparison_universe_manifest(
            include_comparator_records=False
        ),
        reproducibility_bundle_manifest=_bundle_manifest(),
        readiness_judgment_manifest=_readiness_manifest(),
        schema_lifecycle_integration_closure_ref=_closure_manifest().ref,
        catalog_scope="public",
        published_at="2026-04-12T00:00:00Z",
    )

    assert publication.publication_mode == "abstention_only_publication"
    assert publication.comparator_exposure_status == "not_applicable_abstention_only"


def test_build_local_publication_catalog_entry_for_candidate_keeps_published_refs(
) -> None:
    from euclid.modules.catalog_publishing import (
        build_local_publication_catalog_entry,
        build_publication_record_manifest,
        build_run_result_manifest,
    )

    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result = build_run_result_manifest(
        object_id="demo_run_result",
        run_id="demo_run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        primary_validation_scope_ref=_ref(
            "validation_scope_manifest@1.0.0", "validation_scope"
        ),
        publication_mode="candidate_publication",
        selected_candidate_ref=_ref("reducer_artifact_manifest@1.0.0", "candidate"),
        scorecard_ref=_ref("scorecard_manifest@1.1.0", "scorecard"),
        claim_card_ref=_ref("claim_card_manifest@1.1.0", "claim"),
    ).to_manifest(catalog)
    publication_record = build_publication_record_manifest(
        object_id="demo_publication_record",
        publication_id="demo_publication_record",
        run_result_manifest=run_result,
        comparison_universe_manifest=_comparison_universe_manifest(
            include_comparator_records=True
        ),
        reproducibility_bundle_manifest=_bundle_manifest(),
        readiness_judgment_manifest=_readiness_manifest(),
        schema_lifecycle_integration_closure_ref=_closure_manifest().ref,
        catalog_scope="public",
        published_at="2026-04-12T00:00:00Z",
    ).to_manifest(catalog)

    entry = build_local_publication_catalog_entry(
        request_id="demo-request",
        publication_record_manifest=publication_record,
        run_result_manifest=run_result,
        reproducibility_bundle_manifest=_bundle_manifest(),
    )

    assert entry.request_id == "demo-request"
    assert entry.publication_mode == "candidate_publication"
    assert entry.forecast_object_type == "point"
    assert entry.validation_scope_ref == _ref(
        "validation_scope_manifest@1.0.0", "validation_scope"
    )
    assert entry.publication_record_ref == publication_record.ref
    assert entry.run_result_ref == run_result.ref
    assert entry.scorecard_ref == _ref("scorecard_manifest@1.1.0", "scorecard")
    assert entry.claim_card_ref == _ref("claim_card_manifest@1.1.0", "claim")
    assert entry.abstention_ref is None


def test_build_local_publication_catalog_entry_for_abstention_omits_candidate_refs(
) -> None:
    from euclid.modules.catalog_publishing import (
        build_local_publication_catalog_entry,
        build_publication_record_manifest,
        build_run_result_manifest,
    )

    catalog = load_contract_catalog(PROJECT_ROOT)
    run_result = build_run_result_manifest(
        object_id="demo_run_result",
        run_id="demo_run_result",
        scope_ledger_ref=_ref("scope_ledger_manifest@1.0.0", "scope"),
        search_plan_ref=_ref("search_plan_manifest@1.0.0", "search"),
        evaluation_plan_ref=_ref("evaluation_plan_manifest@1.1.0", "evaluation"),
        comparison_universe_ref=_ref(
            "comparison_universe_manifest@1.0.0", "comparison"
        ),
        evaluation_event_log_ref=_ref(
            "evaluation_event_log_manifest@1.0.0", "event_log"
        ),
        evaluation_governance_ref=_ref(
            "evaluation_governance_manifest@1.1.0", "governance"
        ),
        reproducibility_bundle_ref=_ref(
            "reproducibility_bundle_manifest@1.0.0", "bundle"
        ),
        publication_mode="abstention_only_publication",
        abstention_ref=_ref("abstention_manifest@1.1.0", "abstention"),
    ).to_manifest(catalog)
    publication_record = build_publication_record_manifest(
        object_id="demo_publication_record",
        publication_id="demo_publication_record",
        run_result_manifest=run_result,
        comparison_universe_manifest=_comparison_universe_manifest(
            include_comparator_records=False
        ),
        reproducibility_bundle_manifest=_bundle_manifest(),
        readiness_judgment_manifest=_readiness_manifest(),
        schema_lifecycle_integration_closure_ref=_closure_manifest().ref,
        catalog_scope="public",
        published_at="2026-04-12T00:00:00Z",
    ).to_manifest(catalog)

    entry = build_local_publication_catalog_entry(
        request_id="demo-request",
        publication_record_manifest=publication_record,
        run_result_manifest=run_result,
        reproducibility_bundle_manifest=_bundle_manifest(),
    )

    assert entry.request_id == "demo-request"
    assert entry.publication_mode == "abstention_only_publication"
    assert entry.forecast_object_type == "point"
    assert entry.scorecard_ref is None
    assert entry.claim_card_ref is None
    assert entry.abstention_ref == _ref("abstention_manifest@1.1.0", "abstention")
