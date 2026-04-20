from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from euclid.contracts.refs import TypedRef
from euclid.modules.features import default_feature_spec, materialize_feature_view
from euclid.modules.search_planning import (
    build_canonicalization_policy,
    build_search_plan,
)
from euclid.modules.snapshotting import FrozenDatasetSnapshot, SnapshotRow
from euclid.modules.split_planning import build_evaluation_plan
from euclid.modules.timeguard import audit_snapshot_time_safety
from euclid.search.backends import (
    DescriptiveSearchProposal,
    run_descriptive_search_backends,
)


def test_exact_search_fixture_collapses_duplicate_candidate_encodings(
    phase04_runtime_fixture_dir: Path,
) -> None:
    fixture = _load_fixture(
        phase04_runtime_fixture_dir / "duplicate-canonical-candidates.yaml"
    )

    result = _run_fixture_search(fixture, search_class="exact_finite_enumeration")

    assert _accepted_candidate_ids(result) == fixture["expected"]["exact"][
        "accepted_candidate_ids"
    ]
    assert _rejected_reason_codes(result) == fixture["expected"]["exact"][
        "rejected_reason_codes"
    ]


def test_bounded_search_fixture_reports_duplicate_screening_and_omissions(
    phase04_runtime_fixture_dir: Path,
) -> None:
    fixture = _load_fixture(
        phase04_runtime_fixture_dir / "duplicate-canonical-candidates.yaml"
    )

    result = _run_fixture_search(fixture, search_class="bounded_heuristic")

    assert _accepted_candidate_ids(result) == fixture["expected"]["bounded"][
        "accepted_candidate_ids"
    ]
    assert _rejected_reason_codes(result) == fixture["expected"]["bounded"][
        "rejected_reason_codes"
    ]
    assert result.coverage.omitted_candidate_count == fixture["expected"]["bounded"][
        "omitted_candidate_count"
    ]
    assert [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.frontier.frozen_shortlist_cir_candidates
    ] == fixture["expected"]["bounded"]["accepted_candidate_ids"]


def test_exact_search_fixture_frontier_only_uses_surviving_candidates(
    phase04_runtime_fixture_dir: Path,
) -> None:
    fixture = _load_fixture(
        phase04_runtime_fixture_dir / "duplicate-canonical-candidates.yaml"
    )

    result = _run_fixture_search(fixture, search_class="exact_finite_enumeration")
    accepted_ids = set(_accepted_candidate_ids(result))
    frontier_ids = {
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.frontier.retained_frontier_cir_candidates
    }
    shortlist_ids = {
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.frontier.frozen_shortlist_cir_candidates
    }

    assert result.frontier.coverage.comparable_axes == (
        "structure_code_bits",
        "description_gain_bits",
        "inner_primary_score",
    )
    assert frontier_ids
    assert frontier_ids.issubset(accepted_ids)
    assert shortlist_ids
    assert shortlist_ids.issubset(frontier_ids)


def _load_fixture(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text())
    assert isinstance(payload, dict)
    return payload


def _run_fixture_search(
    fixture: dict[str, Any],
    *,
    search_class: str,
):
    feature_view, audit = _feature_view(fixture)
    evaluation_plan = build_evaluation_plan(
        feature_view=feature_view,
        audit=audit,
        min_train_size=3,
        horizon=1,
    )
    canonicalization_policy = build_canonicalization_policy()
    search_fixture = fixture["search"]["exact"]
    if search_class == "bounded_heuristic":
        search_fixture = fixture["search"]["bounded"]
    search_plan = build_search_plan(
        evaluation_plan=evaluation_plan,
        canonicalization_policy_ref=TypedRef(
            "canonicalization_policy_manifest@1.0.0",
            canonicalization_policy.canonicalization_policy_id,
        ),
        codelength_policy_ref=TypedRef(
            "codelength_policy_manifest@1.1.0",
            "mdl_policy_default",
        ),
        reference_description_policy_ref=TypedRef(
            "reference_description_policy_manifest@1.1.0",
            "reference_description_default",
        ),
        observation_model_ref=TypedRef(
            "observation_model_manifest@1.1.0",
            "observation_model_default",
        ),
        candidate_family_ids=tuple(search_fixture["candidate_ids"]),
        search_class=search_class,
        proposal_limit=search_fixture["proposal_limit"],
        seasonal_period=fixture["search"]["seasonal_period"],
    )
    proposals = tuple(
        DescriptiveSearchProposal(
            candidate_id=proposal["candidate_id"],
            primitive_family=proposal["primitive_family"],
            form_class=proposal["form_class"],
            parameter_values=proposal.get("parameter_values", {}),
            literal_values=proposal.get("literal_values", {}),
            persistent_state=proposal.get("persistent_state", {}),
            composition_payload=proposal.get("composition_payload"),
            history_access_mode=proposal.get("history_access_mode", "full_prefix"),
            max_lag=proposal.get("max_lag"),
            required_observation_model_family=proposal.get(
                "required_observation_model_family",
                "gaussian_location_scale",
            ),
        )
        for proposal in fixture["proposals"]
    )
    return run_descriptive_search_backends(
        search_plan=search_plan,
        feature_view=feature_view,
        include_default_grammar=False,
        proposal_specs=proposals,
    )


def _feature_view(fixture: dict[str, Any]):
    snapshot_payload = fixture["snapshot"]
    partition_values = snapshot_payload.get("piecewise_partition_values")
    snapshot = FrozenDatasetSnapshot(
        series_id=snapshot_payload["series_id"],
        cutoff_available_at=snapshot_payload["cutoff_available_at"],
        revision_policy=snapshot_payload["revision_policy"],
        rows=tuple(
            SnapshotRow(
                event_time=f"2026-01-0{index + 1}T00:00:00Z",
                available_at=f"2026-01-0{index + 1}T00:00:00Z",
                observed_value=value,
                revision_id=0,
                payload_hash=f"sha256:{index}",
            )
            for index, value in enumerate(snapshot_payload["observed_values"])
        ),
    )
    audit = audit_snapshot_time_safety(snapshot)
    feature_view = materialize_feature_view(
        snapshot=snapshot,
        audit=audit,
        feature_spec=default_feature_spec(),
    )
    if partition_values is not None:
        assert len(partition_values) == len(snapshot_payload["observed_values"])
        feature_view = replace(
            feature_view,
            feature_names=(*feature_view.feature_names, "piecewise_partition_value"),
            rows=tuple(
                {
                    **row,
                    "piecewise_partition_value": partition_values[
                        int(row["entity_row_index"])
                    ],
                }
                for row in feature_view.rows
            ),
        )
    return feature_view, audit


def _accepted_candidate_ids(result) -> list[str]:
    return [
        candidate.evidence_layer.backend_origin_record.source_candidate_id
        for candidate in result.accepted_candidates
    ]


def _rejected_reason_codes(result) -> dict[str, str]:
    return {
        diagnostic.candidate_id: diagnostic.reason_code
        for diagnostic in result.rejected_diagnostics
    }
