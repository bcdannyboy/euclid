from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from euclid.contracts.refs import TypedRef
from euclid.control_plane import ManifestMetadataRecord, ManifestReferenceRecord
from euclid.demo import (
    _build_registry,
    _demo_paths,
    _load_catalog,
    _read_run_summary,
    replay_demo,
)
from euclid.manifest_registry import StoreValidationReport
from euclid.manifests.base import ManifestEnvelope
from euclid.modules.catalog_publishing import (
    LocalPublicationCatalogEntry,
    build_local_publication_catalog_entry,
)
from euclid.modules.replay import (
    ReplayInspection,
    inspect_reproducibility_bundle,
    resolve_scorecard,
)


@dataclass(frozen=True)
class ArtifactInspection:
    ref: TypedRef
    manifest: ManifestEnvelope
    metadata: ManifestMetadataRecord
    parents: tuple[TypedRef, ...]
    children: tuple[TypedRef, ...]
    references: tuple[ManifestReferenceRecord, ...]
    referrers: tuple[ManifestReferenceRecord, ...]


@dataclass(frozen=True)
class RunArtifactGraph:
    run_id: str
    root_ref: TypedRef
    manifests: tuple[ArtifactInspection, ...]
    request_id: str | None = None

    @property
    def manifest_count(self) -> int:
        return len(self.manifests)

    def inspect(self, ref: str | TypedRef | Mapping[str, object]) -> ArtifactInspection:
        typed_ref = _coerce_typed_ref(ref)
        for manifest in self.manifests:
            if manifest.ref == typed_ref:
                return manifest
        ref_label = f"{typed_ref.schema_name}:{typed_ref.object_id}"
        raise KeyError(f"artifact graph does not include {ref_label}")

    def children_for(
        self, ref: str | TypedRef | Mapping[str, object]
    ) -> tuple[TypedRef, ...]:
        return self.inspect(ref).children

    def parents_for(
        self, ref: str | TypedRef | Mapping[str, object]
    ) -> tuple[TypedRef, ...]:
        return self.inspect(ref).parents


@dataclass(frozen=True)
class DemoPublicationCatalog:
    catalog_root: Path
    entries: tuple[LocalPublicationCatalogEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class PublishedRunInspection:
    entry: LocalPublicationCatalogEntry
    run_result: ArtifactInspection
    publication_record: ArtifactInspection
    replay_bundle: ReplayInspection
    scorecard: ArtifactInspection | None
    claim_card: ArtifactInspection | None
    abstention: ArtifactInspection | None


@dataclass(frozen=True)
class PointPredictionInspection:
    run_id: str
    prediction_artifact_ref: TypedRef
    point_score_result_ref: TypedRef
    candidate_id: str
    stage_id: str
    fit_window_id: str
    test_window_id: str
    forecast_object_type: str
    horizon_set: tuple[int, ...]
    scored_origin_set_id: str | None
    scored_origin_count: int
    row_count: int
    missing_origin_reason_codes: tuple[str, ...]
    timeguard_failure_count: int
    aggregated_primary_score: float
    rows: tuple[Mapping[str, Any], ...]
    per_horizon_scores: tuple[Mapping[str, Any], ...]
    missing_scored_origins: tuple[Mapping[str, Any], ...]
    timeguard_checks: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class ProbabilisticPredictionInspection:
    request_id: str
    prediction_artifact_ref: TypedRef
    score_result_ref: TypedRef
    candidate_id: str
    stage_id: str
    fit_window_id: str
    test_window_id: str
    forecast_object_type: str
    score_law_id: str
    horizon_set: tuple[int, ...]
    scored_origin_set_id: str | None
    row_count: int
    aggregated_primary_score: float
    rows: tuple[Mapping[str, Any], ...]
    per_horizon_scores: tuple[Mapping[str, Any], ...]
    missing_scored_origins: tuple[Mapping[str, Any], ...]
    timeguard_checks: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class CalibrationInspection:
    request_id: str
    forecast_object_type: str
    calibration_contract_ref: TypedRef
    calibration_result_ref: TypedRef
    prediction_artifact_ref: TypedRef
    status: str
    passed: bool | None
    failure_reason_code: str | None
    gate_effect: str
    diagnostics: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class BaselineComparisonInspection:
    run_id: str
    comparison_universe_ref: TypedRef
    selected_candidate_id: str
    baseline_id: str
    comparison_class_status: str
    candidate_primary_score: float
    baseline_primary_score: float
    score_delta: float
    candidate_beats_baseline: bool
    candidate_comparison_key: Mapping[str, Any]
    baseline_comparison_key: Mapping[str, Any]
    practical_significance_margin: float | None
    paired_comparison_records: tuple[Mapping[str, Any], ...]


def load_demo_run_artifact_graph(
    *,
    output_root: Path | None = None,
    run_id: str | None = None,
) -> RunArtifactGraph:
    paths, summary = _resolve_demo_paths_and_summary(
        output_root=output_root,
        run_id=run_id,
    )
    registry = _build_registry(catalog=_load_catalog(), paths=paths)
    resolved_run_id, root_ref = _resolve_root_ref(
        run_id=run_id,
        summary=summary,
        registry=registry,
    )

    discovered: list[TypedRef] = []
    queue: list[TypedRef] = [root_ref]
    seen: set[tuple[str, str]] = set()

    while queue:
        current = queue.pop(0)
        key = (current.schema_name, current.object_id)
        if key in seen:
            continue
        seen.add(key)
        discovered.append(current)
        queue.extend(registry.list_lineage_parents(current))
        queue.extend(registry.list_lineage_children(current))

    inspections = tuple(
        _inspect_artifact(registry=registry, ref=ref)
        for ref in _ordered_refs(root_ref=root_ref, refs=discovered)
    )
    request_id = summary.get("request_id")
    return RunArtifactGraph(
        run_id=resolved_run_id,
        root_ref=root_ref,
        manifests=inspections,
        request_id=request_id if isinstance(request_id, str) else None,
    )


def resolve_demo_artifact(
    *,
    output_root: Path | None = None,
    ref: str | TypedRef | Mapping[str, object],
) -> ArtifactInspection:
    paths, _ = _resolve_demo_paths_and_summary(
        output_root=output_root,
        run_id=None,
    )
    registry = _build_registry(
        catalog=_load_catalog(),
        paths=paths,
    )
    return _inspect_artifact(registry=registry, ref=_coerce_typed_ref(ref))


def inspect_demo_replay_bundle(
    *,
    output_root: Path | None = None,
    bundle_ref: str | TypedRef | Mapping[str, object] | None = None,
) -> ReplayInspection:
    paths, summary = _resolve_demo_paths_and_summary(
        output_root=output_root,
        run_id=None,
    )
    registry = _build_registry(
        catalog=_load_catalog(),
        paths=paths,
    )
    effective_bundle_ref = _summary_typed_ref(summary, "bundle_ref")
    if bundle_ref is not None:
        effective_bundle_ref = _coerce_typed_ref(bundle_ref)
    return inspect_reproducibility_bundle(registry.resolve(effective_bundle_ref))


def inspect_demo_point_prediction(
    *,
    output_root: Path | None = None,
    run_id: str | None = None,
) -> PointPredictionInspection:
    graph = load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
    paths, _ = _resolve_demo_paths_and_summary(
        output_root=output_root,
        run_id=graph.run_id,
    )
    registry = _build_registry(
        catalog=_load_catalog(),
        paths=paths,
    )
    run_artifact = graph.inspect(graph.root_ref)
    prediction_artifact_ref = _single_prediction_artifact_ref(
        run_artifact.manifest.body
    )
    prediction_artifact = graph.inspect(prediction_artifact_ref)
    comparison = compare_demo_baseline(output_root=output_root, run_id=graph.run_id)
    point_score_result = _resolve_point_score_result_artifact(
        graph=graph,
        registry=registry,
        run_artifact=run_artifact,
        prediction_artifact_ref=prediction_artifact_ref,
    )

    prediction_body = prediction_artifact.manifest.body
    comparison_key = _comparison_key_from_prediction_body(
        prediction_body,
        fallback=comparison.candidate_comparison_key,
    )
    rows = tuple(_dict_items(prediction_body.get("rows", ())))
    missing_scored_origins = tuple(
        _dict_items(prediction_body.get("missing_scored_origins", ()))
    )
    timeguard_checks = tuple(_dict_items(prediction_body.get("timeguard_checks", ())))
    horizon_set = _horizon_set_from_prediction(
        prediction_body=prediction_body,
        comparison_key=comparison_key,
    )
    scored_origin_set_id = _string_or_none(
        prediction_body.get("scored_origin_set_id")
    ) or _string_or_none(comparison_key.get("scored_origin_set_id"))
    reason_codes = tuple(
        sorted(
            {
                str(item["reason_code"])
                for item in missing_scored_origins
                if "reason_code" in item
            }
        )
    )
    failed_timeguard_checks = sum(
        1
        for check in timeguard_checks
        if str(check.get("status", "")).lower() == "failed"
    )
    point_score_body = point_score_result.manifest.body

    return PointPredictionInspection(
        run_id=graph.run_id,
        prediction_artifact_ref=prediction_artifact_ref,
        point_score_result_ref=point_score_result.ref,
        candidate_id=str(prediction_body["candidate_id"]),
        stage_id=str(prediction_body["stage_id"]),
        fit_window_id=str(prediction_body["fit_window_id"]),
        test_window_id=str(prediction_body["test_window_id"]),
        forecast_object_type=str(prediction_body.get("forecast_object_type", "point")),
        horizon_set=horizon_set,
        scored_origin_set_id=scored_origin_set_id,
        scored_origin_count=len(rows),
        row_count=len(rows),
        missing_origin_reason_codes=reason_codes,
        timeguard_failure_count=failed_timeguard_checks,
        aggregated_primary_score=float(point_score_body["aggregated_primary_score"]),
        rows=rows,
        per_horizon_scores=tuple(_dict_items(point_score_body.get("per_horizon", ()))),
        missing_scored_origins=missing_scored_origins,
        timeguard_checks=timeguard_checks,
    )


def inspect_demo_probabilistic_prediction(
    *,
    output_root: Path | None = None,
) -> ProbabilisticPredictionInspection:
    paths, summary = _resolve_demo_paths_and_summary(
        output_root=output_root, run_id=None
    )
    registry = _build_registry(catalog=_load_catalog(), paths=paths)
    prediction_artifact_ref = _summary_typed_ref(summary, "prediction_artifact_ref")
    score_result_ref = _summary_typed_ref(summary, "score_result_ref")
    prediction_artifact = _inspect_artifact(
        registry=registry, ref=prediction_artifact_ref
    )
    score_result = _inspect_artifact(registry=registry, ref=score_result_ref)
    prediction_body = prediction_artifact.manifest.body
    score_body = score_result.manifest.body
    rows = tuple(_dict_items(prediction_body.get("rows", ())))
    return ProbabilisticPredictionInspection(
        request_id=_summary_string(summary, "request_id"),
        prediction_artifact_ref=prediction_artifact_ref,
        score_result_ref=score_result_ref,
        candidate_id=str(prediction_body["candidate_id"]),
        stage_id=str(prediction_body["stage_id"]),
        fit_window_id=str(prediction_body["fit_window_id"]),
        test_window_id=str(prediction_body["test_window_id"]),
        forecast_object_type=str(prediction_body["forecast_object_type"]),
        score_law_id=str(prediction_body["score_law_id"]),
        horizon_set=_horizon_set_from_prediction(
            prediction_body=prediction_body,
            comparison_key=dict(prediction_body.get("comparison_key", {})),
        ),
        scored_origin_set_id=_string_or_none(
            prediction_body.get("scored_origin_set_id")
        ),
        row_count=len(rows),
        aggregated_primary_score=float(score_body["aggregated_primary_score"]),
        rows=rows,
        per_horizon_scores=tuple(_dict_items(score_body.get("per_horizon", ()))),
        missing_scored_origins=tuple(
            _dict_items(prediction_body.get("missing_scored_origins", ()))
        ),
        timeguard_checks=tuple(
            _dict_items(prediction_body.get("timeguard_checks", ()))
        ),
    )


def inspect_demo_calibration(
    *,
    output_root: Path | None = None,
) -> CalibrationInspection:
    paths, summary = _resolve_demo_paths_and_summary(
        output_root=output_root, run_id=None
    )
    registry = _build_registry(catalog=_load_catalog(), paths=paths)
    calibration_contract_ref = _summary_typed_ref(summary, "calibration_contract_ref")
    calibration_result_ref = _summary_typed_ref(summary, "calibration_result_ref")
    calibration_result = _inspect_artifact(
        registry=registry, ref=calibration_result_ref
    )
    body = calibration_result.manifest.body
    return CalibrationInspection(
        request_id=_summary_string(summary, "request_id"),
        forecast_object_type=str(body["forecast_object_type"]),
        calibration_contract_ref=calibration_contract_ref,
        calibration_result_ref=calibration_result_ref,
        prediction_artifact_ref=_coerce_typed_ref(body["prediction_artifact_ref"]),
        status=str(body["status"]),
        passed=(bool(body["pass"]) if isinstance(body.get("pass"), bool) else None),
        failure_reason_code=_string_or_none(body.get("failure_reason_code")),
        gate_effect=str(body["gate_effect"]),
        diagnostics=tuple(_dict_items(body.get("diagnostics", ()))),
    )


def compare_demo_baseline(
    *,
    output_root: Path | None = None,
    run_id: str | None = None,
) -> BaselineComparisonInspection:
    graph = load_demo_run_artifact_graph(output_root=output_root, run_id=run_id)
    run_artifact = graph.inspect(graph.root_ref)
    comparison_universe_ref = _coerce_typed_ref(
        run_artifact.manifest.body["comparison_universe_ref"]
    )
    comparison_universe = graph.inspect(comparison_universe_ref)
    body = comparison_universe.manifest.body
    candidate_primary_score = float(body["candidate_primary_score"])
    baseline_primary_score = float(body["baseline_primary_score"])
    return BaselineComparisonInspection(
        run_id=graph.run_id,
        comparison_universe_ref=comparison_universe_ref,
        selected_candidate_id=str(body["selected_candidate_id"]),
        baseline_id=str(body["baseline_id"]),
        comparison_class_status=str(body["comparison_class_status"]),
        candidate_primary_score=candidate_primary_score,
        baseline_primary_score=baseline_primary_score,
        score_delta=baseline_primary_score - candidate_primary_score,
        candidate_beats_baseline=bool(body["candidate_beats_baseline"]),
        candidate_comparison_key=dict(body.get("candidate_comparison_key", {})),
        baseline_comparison_key=dict(body.get("baseline_comparison_key", {})),
        practical_significance_margin=_float_or_none(
            body.get("practical_significance_margin")
        ),
        paired_comparison_records=tuple(
            _dict_items(body.get("paired_comparison_records", ()))
        ),
    )


def validate_demo_store(
    *,
    output_root: Path | None = None,
) -> StoreValidationReport:
    paths, _ = _resolve_demo_paths_and_summary(
        output_root=output_root,
        run_id=None,
    )
    registry = _build_registry(
        catalog=_load_catalog(),
        paths=paths,
    )
    return registry.validate_store()


def publish_demo_run_to_catalog(
    *,
    output_root: Path | None = None,
    run_id: str | None = None,
) -> LocalPublicationCatalogEntry:
    paths, summary = _resolve_demo_paths_and_summary(
        output_root=output_root,
        run_id=run_id,
    )
    catalog = _load_catalog()
    registry = _build_registry(catalog=catalog, paths=paths)
    bundle_ref = _summary_typed_ref(summary, "bundle_ref")
    replay = replay_demo(
        output_root=paths.output_root,
        run_id=run_id,
        bundle_ref=bundle_ref,
    )
    if replay.summary.replay_verification_status != "verified":
        reasons = ", ".join(replay.summary.failure_reason_codes) or "unknown failure"
        raise ValueError(
            "local catalog publication requires replay verification to pass; "
            f"got {reasons}"
        )

    run_result = registry.resolve(replay.summary.run_result_ref)
    bundle = registry.resolve(bundle_ref)
    publication_record = _resolve_publication_record_for_run_result(
        registry=registry,
        run_result_ref=run_result.manifest.ref,
    )
    entry = build_local_publication_catalog_entry(
        request_id=_summary_string(summary, "request_id"),
        publication_record_manifest=publication_record.manifest,
        run_result_manifest=run_result.manifest,
        reproducibility_bundle_manifest=bundle.manifest,
    )
    existing_entries = [
        candidate
        for candidate in load_demo_publication_catalog(
            output_root=paths.output_root
        ).entries
        if candidate.publication_id != entry.publication_id
    ]
    existing_entries.append(entry)
    _write_demo_publication_catalog(
        catalog_root=_catalog_root(paths.output_root),
        entries=tuple(existing_entries),
    )
    return entry


def load_demo_publication_catalog(
    *,
    output_root: Path | None = None,
) -> DemoPublicationCatalog:
    catalog_root = _catalog_root(output_root)
    index_path = catalog_root / "index.json"
    if not index_path.is_file():
        return DemoPublicationCatalog(catalog_root=catalog_root, entries=())

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    entries_payload = payload.get("entries", [])
    if not isinstance(entries_payload, list):
        raise ValueError("demo publication catalog index must contain an entries list")
    entries = tuple(
        LocalPublicationCatalogEntry.from_mapping(item)
        for item in entries_payload
        if isinstance(item, Mapping)
    )
    return DemoPublicationCatalog(catalog_root=catalog_root, entries=entries)


def inspect_demo_catalog_entry(
    *,
    output_root: Path | None = None,
    publication_id: str | None = None,
    run_id: str | None = None,
) -> PublishedRunInspection:
    catalog = load_demo_publication_catalog(output_root=output_root)
    entry = _select_catalog_entry(
        entries=catalog.entries,
        publication_id=publication_id,
        run_id=run_id,
    )
    paths = _demo_paths(
        request=None,
        output_root=output_root,
        request_id=entry.request_id,
    )
    registry = _build_registry(catalog=_load_catalog(), paths=paths)
    run_result = _inspect_artifact(registry=registry, ref=entry.run_result_ref)
    scorecard_ref = (
        entry.scorecard_ref
        if entry.scorecard_ref is not None
        else resolve_scorecard(run_result.manifest.body, registry).manifest.ref
    )
    return PublishedRunInspection(
        entry=entry,
        run_result=run_result,
        publication_record=_inspect_artifact(
            registry=registry,
            ref=entry.publication_record_ref,
        ),
        replay_bundle=inspect_reproducibility_bundle(
            registry.resolve(entry.reproducibility_bundle_ref)
        ),
        scorecard=(
            _inspect_artifact(registry=registry, ref=scorecard_ref)
            if scorecard_ref is not None
            else None
        ),
        claim_card=(
            _inspect_artifact(registry=registry, ref=entry.claim_card_ref)
            if entry.claim_card_ref is not None
            else None
        ),
        abstention=(
            _inspect_artifact(registry=registry, ref=entry.abstention_ref)
            if entry.abstention_ref is not None
            else None
        ),
    )


def format_demo_artifact_graph(graph: RunArtifactGraph) -> str:
    lines = [
        "Euclid demo artifact inspection",
        f"Run id: {graph.run_id}",
        f"Root ref: {_format_typed_ref(graph.root_ref)}",
    ]
    if graph.request_id is not None:
        lines.append(f"Request id: {graph.request_id}")
    lines.extend(
        [
            f"Manifest count: {graph.manifest_count}",
            "",
            "Connected manifests:",
        ]
    )
    for manifest in graph.manifests:
        lines.append(
            "- "
            f"{_format_typed_ref(manifest.ref)} "
            f"(parents={len(manifest.parents)}, "
            f"children={len(manifest.children)}, "
            f"typed_refs={len(manifest.references)})"
        )
    return "\n".join(lines)


def format_resolved_artifact(artifact: ArtifactInspection) -> str:
    lines = [
        "Resolved manifest",
        f"Ref: {_format_typed_ref(artifact.ref)}",
        f"Schema: {artifact.manifest.schema_name}",
        f"Object id: {artifact.manifest.object_id}",
        f"Module: {artifact.manifest.module_id}",
        (
            f"Run id: {artifact.metadata.run_id}"
            if artifact.metadata.run_id is not None
            else "Run id: none"
        ),
        f"Content hash: {artifact.metadata.content_hash}",
        f"Artifact path: {artifact.metadata.artifact_path}",
        "Parents:",
    ]
    lines.extend(_format_ref_block(artifact.parents))
    lines.append("Children:")
    lines.extend(_format_ref_block(artifact.children))
    lines.append("Typed refs:")
    lines.extend(
        _format_reference_block(
            artifact.references,
            prefix="source",
        )
    )
    lines.append("Referrers:")
    lines.extend(
        _format_reference_block(
            artifact.referrers,
            prefix="target",
        )
    )
    return "\n".join(lines)


def format_demo_lineage_graph(graph: RunArtifactGraph) -> str:
    lines = [
        "Euclid demo lineage graph",
        f"Run id: {graph.run_id}",
        f"Root ref: {_format_typed_ref(graph.root_ref)}",
        "",
        "Upstream lineage:",
        _format_typed_ref(graph.root_ref),
    ]
    lines.extend(
        _format_lineage_branch(
            graph=graph,
            ref=graph.root_ref,
            direction="parents",
            prefix="",
            seen={(graph.root_ref.schema_name, graph.root_ref.object_id)},
        )
    )
    lines.extend(["", "Downstream lineage:", _format_typed_ref(graph.root_ref)])
    lines.extend(
        _format_lineage_branch(
            graph=graph,
            ref=graph.root_ref,
            direction="children",
            prefix="",
            seen={(graph.root_ref.schema_name, graph.root_ref.object_id)},
        )
    )
    return "\n".join(lines)


def format_demo_store_validation(report: StoreValidationReport) -> str:
    lines = [
        "Euclid demo store validation",
        f"Manifest count: {report.manifest_count}",
        f"Store valid: {'yes' if report.is_valid else 'no'}",
        f"Issue count: {report.issue_count}",
    ]
    if report.issues:
        lines.append("")
        lines.append("Issues:")
        for issue in report.issues:
            location = (
                _format_typed_ref(issue.ref)
                if issue.ref is not None
                else str(issue.path)
                if issue.path is not None
                else "store"
            )
            lines.append(f"- {issue.code} at {location}: {issue.message}")
    return "\n".join(lines)


def format_demo_publication_catalog(catalog: DemoPublicationCatalog) -> str:
    lines = [
        "Euclid demo publication catalog",
        f"Catalog root: {catalog.catalog_root}",
        f"Entries: {catalog.entry_count}",
    ]
    if not catalog.entries:
        lines.append("")
        lines.append("Published entries: none")
        return "\n".join(lines)

    lines.extend(["", "Published entries:"])
    for entry in sorted(
        catalog.entries,
        key=lambda item: (item.published_at, item.publication_id),
    ):
        lines.append(
            "- "
            f"{entry.publication_id} "
            f"(request_id={entry.request_id}, "
            f"mode={entry.publication_mode}, "
            f"published_at={entry.published_at})"
        )
    return "\n".join(lines)


def format_demo_catalog_entry(inspection: PublishedRunInspection) -> str:
    lines = [
        "Euclid demo catalog entry",
        f"Publication id: {inspection.entry.publication_id}",
        f"Request id: {inspection.entry.request_id}",
        f"Run id: {inspection.entry.run_id}",
        f"Publication mode: {inspection.entry.publication_mode}",
        f"Catalog scope: {inspection.entry.catalog_scope}",
        f"Published at: {inspection.entry.published_at}",
        (
            "Publication record: "
            f"{_format_typed_ref(inspection.entry.publication_record_ref)}"
        ),
        f"Run result: {_format_typed_ref(inspection.entry.run_result_ref)}",
        (
            "Reproducibility bundle: "
            f"{_format_typed_ref(inspection.entry.reproducibility_bundle_ref)}"
        ),
        f"Replay verification: {inspection.entry.replay_verification_status}",
        (
            "Comparator exposure: "
            f"{inspection.entry.comparator_exposure_status}"
        ),
        (
            "Scorecard: "
            + (
                _format_typed_ref(inspection.entry.scorecard_ref)
                if inspection.entry.scorecard_ref is not None
                else "none"
            )
        ),
        (
            "Claim card: "
            + (
                _format_typed_ref(inspection.entry.claim_card_ref)
                if inspection.entry.claim_card_ref is not None
                else "none"
            )
        ),
        (
            "Abstention: "
            + (
                _format_typed_ref(inspection.entry.abstention_ref)
                if inspection.entry.abstention_ref is not None
                else "none"
            )
        ),
    ]
    return "\n".join(lines)


def format_replay_bundle_inspection(inspection: ReplayInspection) -> str:
    lines = [
        "Euclid demo replay bundle",
        f"Bundle ref: {_format_typed_ref(inspection.bundle_ref)}",
        f"Run result ref: {_format_typed_ref(inspection.run_result_ref)}",
        f"Bundle mode: {inspection.bundle_mode}",
        f"Replay verification: {inspection.replay_verification_status}",
        (
            "Failure reasons: " + ", ".join(inspection.failure_reason_codes)
            if inspection.failure_reason_codes
            else "Failure reasons: none"
        ),
        "Artifact hashes:",
    ]
    for record in inspection.artifact_hash_records:
        lines.append(f"- {record.artifact_role}: {record.sha256}")
    lines.append("Seed records:")
    for record in inspection.seed_records:
        lines.append(f"- {record.seed_scope}: {record.seed_value}")
    lines.append("Environment metadata:")
    for key, value in sorted(inspection.environment_metadata.items()):
        lines.append(f"- {key}: {value}")
    lines.append("Stage order:")
    for record in inspection.stage_order_records:
        lines.append(
            "- " f"{record.stage_id} -> {_format_typed_ref(record.manifest_ref)}"
        )
    return "\n".join(lines)


def format_point_prediction_inspection(inspection: PointPredictionInspection) -> str:
    lines = [
        "Euclid demo point prediction",
        f"Run id: {inspection.run_id}",
        f"Prediction artifact: {_format_typed_ref(inspection.prediction_artifact_ref)}",
        f"Point score result: {_format_typed_ref(inspection.point_score_result_ref)}",
        f"Candidate id: {inspection.candidate_id}",
        f"Stage: {inspection.stage_id}",
        f"Fit window: {inspection.fit_window_id}",
        f"Test window: {inspection.test_window_id}",
        f"Forecast object type: {inspection.forecast_object_type}",
        f"Horizon set: {_format_horizon_set(inspection.horizon_set)}",
        f"Scored origin set: {inspection.scored_origin_set_id or 'none'}",
        f"Scored origin count: {inspection.scored_origin_count}",
        f"Row count: {inspection.row_count}",
        f"Timeguard failures: {inspection.timeguard_failure_count}",
        f"Aggregated primary score: {inspection.aggregated_primary_score:.6f}",
        "Per-horizon scores:",
    ]
    if inspection.per_horizon_scores:
        for item in inspection.per_horizon_scores:
            lines.append(
                "- "
                f"horizon {int(item['horizon'])}: "
                f"mean_point_loss={float(item['mean_point_loss']):.6f}, "
                f"valid_origin_count={int(item['valid_origin_count'])}"
            )
    else:
        lines.append("- none")
    lines.append("Missing origin reasons:")
    if inspection.missing_origin_reason_codes:
        lines.extend(f"- {reason}" for reason in inspection.missing_origin_reason_codes)
    else:
        lines.append("- none")
    return "\n".join(lines)


def format_probabilistic_prediction_inspection(
    inspection: ProbabilisticPredictionInspection,
) -> str:
    lines = [
        "Euclid demo probabilistic prediction",
        f"Request id: {inspection.request_id}",
        f"Prediction artifact: {_format_typed_ref(inspection.prediction_artifact_ref)}",
        f"Score result: {_format_typed_ref(inspection.score_result_ref)}",
        f"Candidate id: {inspection.candidate_id}",
        f"Stage: {inspection.stage_id}",
        f"Fit window: {inspection.fit_window_id}",
        f"Test window: {inspection.test_window_id}",
        f"Forecast object type: {inspection.forecast_object_type}",
        f"Score law: {inspection.score_law_id}",
        f"Horizon set: {_format_horizon_set(inspection.horizon_set)}",
        f"Scored origin set: {inspection.scored_origin_set_id or 'none'}",
        f"Row count: {inspection.row_count}",
        f"Aggregated primary score: {inspection.aggregated_primary_score:.6f}",
    ]
    return "\n".join(lines)


def format_calibration_inspection(inspection: CalibrationInspection) -> str:
    lines = [
        "Euclid demo calibration",
        f"Request id: {inspection.request_id}",
        f"Forecast object type: {inspection.forecast_object_type}",
        (
            "Calibration contract: "
            f"{_format_typed_ref(inspection.calibration_contract_ref)}"
        ),
        f"Calibration result: {_format_typed_ref(inspection.calibration_result_ref)}",
        f"Prediction artifact: {_format_typed_ref(inspection.prediction_artifact_ref)}",
        f"Status: {inspection.status}",
        f"Pass: {inspection.passed if inspection.passed is not None else 'n/a'}",
        f"Gate effect: {inspection.gate_effect}",
        f"Failure reason: {inspection.failure_reason_code or 'none'}",
        "Diagnostics:",
    ]
    if inspection.diagnostics:
        for diagnostic in inspection.diagnostics:
            lines.append(f"- {dict(diagnostic)}")
    else:
        lines.append("- none")
    return "\n".join(lines)


def format_baseline_comparison(comparison: BaselineComparisonInspection) -> str:
    lines = [
        "Euclid demo baseline comparison",
        f"Run id: {comparison.run_id}",
        f"Comparison universe: {_format_typed_ref(comparison.comparison_universe_ref)}",
        f"Candidate id: {comparison.selected_candidate_id}",
        f"Baseline id: {comparison.baseline_id}",
        f"Comparison class: {comparison.comparison_class_status}",
        f"Candidate primary score: {comparison.candidate_primary_score:.6f}",
        f"Baseline primary score: {comparison.baseline_primary_score:.6f}",
        f"Primary score delta (baseline - candidate): {comparison.score_delta:.6f}",
        "Candidate beats baseline: "
        f"{'yes' if comparison.candidate_beats_baseline else 'no'}",
    ]
    if comparison.practical_significance_margin is not None:
        lines.append(
            "Practical significance margin: "
            f"{comparison.practical_significance_margin:.6f}"
        )
    lines.append("Paired comparison records:")
    if comparison.paired_comparison_records:
        for record in comparison.paired_comparison_records:
            lines.append(
                "- "
                f"{record['comparator_id']}: "
                f"status={record['comparison_status']}, "
                f"delta={float(record['primary_score_delta']):.6f}"
            )
    else:
        lines.append("- none")
    return "\n".join(lines)


def _inspect_artifact(*, registry, ref: TypedRef) -> ArtifactInspection:
    registered = registry.resolve(ref)
    return ArtifactInspection(
        ref=ref,
        manifest=registered.manifest,
        metadata=registered.metadata,
        parents=registry.list_lineage_parents(ref),
        children=registry.list_lineage_children(ref),
        references=registry.list_referenced_manifests(ref),
        referrers=registry.list_referrers(ref),
    )


def _resolve_publication_record_for_run_result(
    *,
    registry,
    run_result_ref: TypedRef,
) -> ArtifactInspection:
    candidates = [
        record.source_ref
        for record in registry.list_referrers(run_result_ref)
        if record.source_ref.schema_name == "publication_record_manifest@1.1.0"
        and record.field_path == "body.run_result_ref"
    ]
    if len(candidates) != 1:
        raise ValueError(
            "expected exactly one publication record for the selected run result"
        )
    return _inspect_artifact(registry=registry, ref=candidates[0])


def _catalog_root(output_root: Path | None) -> Path:
    return _demo_paths(request=None, output_root=output_root).output_root / "catalog"


def _write_demo_publication_catalog(
    *,
    catalog_root: Path,
    entries: tuple[LocalPublicationCatalogEntry, ...],
) -> None:
    catalog_root.mkdir(parents=True, exist_ok=True)
    ordered_entries = sorted(
        entries,
        key=lambda entry: (entry.published_at, entry.publication_id),
    )
    payload = {
        "catalog_version": 1,
        "entries": [entry.as_dict() for entry in ordered_entries],
    }
    (catalog_root / "index.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _select_catalog_entry(
    *,
    entries: tuple[LocalPublicationCatalogEntry, ...],
    publication_id: str | None,
    run_id: str | None,
) -> LocalPublicationCatalogEntry:
    if publication_id is not None:
        for entry in entries:
            if entry.publication_id == publication_id:
                return entry
        raise KeyError(f"publication catalog does not include {publication_id!r}")

    if run_id is not None:
        matches = [entry for entry in entries if entry.run_id == run_id]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise KeyError(f"publication catalog does not include run_id {run_id!r}")
        raise ValueError(
            "multiple publication catalog entries matched the requested run_id"
        )

    if len(entries) == 1:
        return entries[0]
    if not entries:
        raise FileNotFoundError("demo publication catalog is empty")
    raise ValueError(
        "multiple published entries found; pass publication_id or run_id to select one"
    )


def _resolve_demo_paths_and_summary(
    *,
    output_root: Path | None,
    run_id: str | None,
):
    default_paths = _demo_paths(request=None, output_root=output_root)
    if default_paths.run_summary_path.is_file():
        return default_paths, _read_run_summary(default_paths.run_summary_path)
    if output_root is None:
        return default_paths, _read_run_summary(default_paths.run_summary_path)

    summary_files = sorted(
        (output_root.resolve() / "sealed-runs").glob("*/run-summary.json")
    )
    if not summary_files:
        return default_paths, _read_run_summary(default_paths.run_summary_path)

    selected_summary_path: Path | None = None
    selected_summary: Mapping[str, object] | None = None
    for summary_path in summary_files:
        summary = _read_run_summary(summary_path)
        if run_id is not None and not _summary_matches_run_id(summary, run_id):
            continue
        if selected_summary_path is not None and run_id is None:
            raise ValueError(
                "multiple demo runs found under output_root; "
                "pass run_id to disambiguate"
            )
        selected_summary_path = summary_path
        selected_summary = summary
        if run_id is not None:
            break

    if selected_summary_path is None or selected_summary is None:
        raise FileNotFoundError(
            f"no demo run summary found under {output_root} for run_id {run_id!r}"
        )

    request_id = selected_summary.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        request_id = selected_summary_path.parent.name
    return (
        _demo_paths(
            request=None,
            output_root=output_root,
            request_id=request_id,
        ),
        selected_summary,
    )


def _summary_matches_run_id(summary: Mapping[str, object], run_id: str) -> bool:
    request_id = summary.get("request_id")
    if isinstance(request_id, str) and request_id == run_id:
        return True
    run_result_ref = summary.get("run_result_ref")
    if not isinstance(run_result_ref, Mapping):
        return False
    return run_result_ref.get("object_id") == run_id


def _resolve_root_ref(*, run_id: str | None, summary: Mapping[str, object], registry):
    default_root = _coerce_typed_ref(summary["run_result_ref"])
    effective_run_id = run_id or default_root.object_id

    candidates = registry.list_manifests_for_run(effective_run_id)
    if len(candidates) == 1:
        return effective_run_id, candidates[0].manifest.ref
    if len(candidates) > 1:
        for candidate in candidates:
            if candidate.manifest.ref == default_root:
                return effective_run_id, candidate.manifest.ref
        return effective_run_id, candidates[0].manifest.ref

    if run_id is not None:
        request_id = summary.get("request_id")
        if isinstance(request_id, str) and request_id == run_id:
            return default_root.object_id, default_root
        fallback = TypedRef(
            schema_name=default_root.schema_name,
            object_id=run_id,
        )
        registry.resolve(fallback)
        return run_id, fallback

    return default_root.object_id, default_root


def _summary_string(summary: Mapping[str, object], key: str) -> str:
    value = summary.get(key)
    if isinstance(value, str):
        return value
    raise ValueError(f"demo run summary missing string field {key!r}")


def _summary_typed_ref(summary: Mapping[str, object], key: str) -> TypedRef:
    value = summary.get(key)
    if isinstance(value, Mapping):
        return _coerce_typed_ref(value)
    raise ValueError(f"demo run summary missing typed ref field {key!r}")


def _ordered_refs(
    *,
    root_ref: TypedRef,
    refs: list[TypedRef],
) -> tuple[TypedRef, ...]:
    remainder = sorted(
        (ref for ref in refs if ref != root_ref),
        key=lambda ref: (ref.schema_name, ref.object_id),
    )
    return (root_ref, *remainder)


def _coerce_typed_ref(value: str | TypedRef | Mapping[str, object]) -> TypedRef:
    if isinstance(value, TypedRef):
        return value
    if isinstance(value, str):
        schema_name, object_id = value.split(":", 1)
        return TypedRef(schema_name=schema_name, object_id=object_id)
    schema_name = value.get("schema_name")
    object_id = value.get("object_id")
    if isinstance(schema_name, str) and isinstance(object_id, str):
        return TypedRef(schema_name=schema_name, object_id=object_id)
    raise ValueError("typed refs must be schema:object strings or mappings")


def _format_typed_ref(ref: TypedRef) -> str:
    return f"{ref.schema_name}:{ref.object_id}"


def _format_ref_block(refs: tuple[TypedRef, ...]) -> list[str]:
    if not refs:
        return ["- none"]
    return [f"- {_format_typed_ref(ref)}" for ref in refs]


def _format_reference_block(
    refs: tuple[ManifestReferenceRecord, ...],
    *,
    prefix: str,
) -> list[str]:
    if not refs:
        return ["- none"]
    if prefix == "source":
        return [
            f"- {reference.field_path} -> {_format_typed_ref(reference.target_ref)}"
            for reference in refs
        ]
    return [
        f"- {_format_typed_ref(reference.source_ref)} via {reference.field_path}"
        for reference in refs
    ]


def _format_lineage_branch(
    *,
    graph: RunArtifactGraph,
    ref: TypedRef,
    direction: str,
    prefix: str,
    seen: set[tuple[str, str]],
) -> list[str]:
    neighbors = (
        graph.parents_for(ref) if direction == "parents" else graph.children_for(ref)
    )
    lines: list[str] = []
    for index, neighbor in enumerate(neighbors):
        is_last = index == len(neighbors) - 1
        branch = "└── " if is_last else "├── "
        neighbor_key = (neighbor.schema_name, neighbor.object_id)
        suffix = " (seen)" if neighbor_key in seen else ""
        lines.append(f"{prefix}{branch}{_format_typed_ref(neighbor)}{suffix}")
        if neighbor_key in seen:
            continue
        seen.add(neighbor_key)
        extension = f"{prefix}{'    ' if is_last else '│   '}"
        lines.extend(
            _format_lineage_branch(
                graph=graph,
                ref=neighbor,
                direction=direction,
                prefix=extension,
                seen=seen,
            )
        )
    if not neighbors:
        lines.append(f"{prefix}└── none")
    return lines


def _single_prediction_artifact_ref(run_result_body: Mapping[str, Any]) -> TypedRef:
    prediction_refs = tuple(run_result_body.get("prediction_artifact_refs", ()))
    if len(prediction_refs) != 1:
        raise ValueError(
            "demo point inspection expects exactly one prediction artifact"
        )
    return _coerce_typed_ref(prediction_refs[0])


def _resolve_point_score_result_artifact(
    *,
    graph: RunArtifactGraph,
    registry,
    run_artifact: ArtifactInspection,
    prediction_artifact_ref: TypedRef,
) -> ArtifactInspection:
    scorecard = _resolve_scorecard_artifact(graph=graph, run_artifact=run_artifact)
    if scorecard is not None:
        point_score_result_ref = _coerce_typed_ref(
            scorecard.manifest.body["point_score_result_ref"]
        )
        return graph.inspect(point_score_result_ref)

    candidates = [
        manifest
        for manifest in registry.list_manifests_for_run(graph.run_id)
        if manifest.manifest.schema_name == "point_score_result_manifest@1.0.0"
    ]
    for candidate in candidates:
        body = candidate.manifest.body
        if (
            _coerce_typed_ref(body["prediction_artifact_ref"])
            == prediction_artifact_ref
        ):
            return _inspect_artifact(registry=registry, ref=candidate.manifest.ref)
    raise ValueError("demo run does not expose a point score result artifact")


def _resolve_scorecard_artifact(
    *,
    graph: RunArtifactGraph,
    run_artifact: ArtifactInspection,
) -> ArtifactInspection | None:
    scorecard_ref_payload = run_artifact.manifest.body.get("primary_scorecard_ref")
    if isinstance(scorecard_ref_payload, Mapping):
        return graph.inspect(scorecard_ref_payload)

    abstention_ref_payload = run_artifact.manifest.body.get("primary_abstention_ref")
    if isinstance(abstention_ref_payload, Mapping):
        abstention = graph.inspect(abstention_ref_payload)
        governing_refs = abstention.manifest.body.get("governing_refs", ())
        if isinstance(governing_refs, list) and governing_refs:
            return graph.inspect(governing_refs[0])
    return None


def _comparison_key_from_prediction_body(
    prediction_body: Mapping[str, Any],
    *,
    fallback: Mapping[str, Any],
) -> Mapping[str, Any]:
    comparison_key = prediction_body.get("comparison_key")
    if isinstance(comparison_key, Mapping):
        return dict(comparison_key)
    return dict(fallback)


def _horizon_set_from_prediction(
    *,
    prediction_body: Mapping[str, Any],
    comparison_key: Mapping[str, Any],
) -> tuple[int, ...]:
    horizon_set = comparison_key.get("horizon_set")
    if isinstance(horizon_set, list):
        return tuple(int(value) for value in horizon_set)
    rows = tuple(_dict_items(prediction_body.get("rows", ())))
    return tuple(sorted({int(row["horizon"]) for row in rows}))


def _dict_items(payload: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(payload, list | tuple):
        return ()
    return tuple(dict(item) for item in payload if isinstance(item, Mapping))


def _string_or_none(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _float_or_none(value: object) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _format_horizon_set(horizon_set: tuple[int, ...]) -> str:
    if not horizon_set:
        return "none"
    return ", ".join(str(horizon) for horizon in horizon_set)
