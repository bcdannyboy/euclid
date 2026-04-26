---
title: Modeling Pipeline
status: active_reference
related:
- system.md
- search-core.md
- contracts-manifests.md
- workbench.md
- testing-truthfulness.md
- ../../schemas/contracts/module-registry.yaml
---
# Modeling Pipeline

Euclid's modeling layer lives in `src/euclid/modules`. It turns ordered observations into descriptive equations, evaluates whether any candidate earns predictive support, and resolves publication status only after scoring, calibration, robustness, replay, and any required evidence gates.

The goal is still to discover legible mathematical structure in an ordered sequence, but to do it under explicit comparison geometry and publication discipline.

When a publishable point law and a publishable probabilistic lane share validation scope and publication record, downstream surfaces can expose that joined result as one unified deterministic-plus-stochastic equation.

## Stage map

### 1. Ingestion

`modules/ingestion.py` (`src/euclid/modules/ingestion.py`) canonicalizes raw tabular inputs into ordered observation records.

Core invariants:

- targets may be missing in raw lineage
- coded targets exclude missing values
- present numeric values must be finite
- records are ordered and typed before downstream use

### 2. Snapshotting and timeguard

`modules/snapshotting.py` (`src/euclid/modules/snapshotting.py`) freezes visible observations at a cutoff and chooses the latest visible revision per `(entity, event_time)`.

`modules/timeguard.py` (`src/euclid/modules/timeguard.py`) audits that:

- `available_at <= cutoff`
- `event_time <= cutoff`

This is the first hard time-safety boundary in the pipeline.

### 3. Feature materialization

`modules/features.py` (`src/euclid/modules/features.py`) builds retained-scope feature views from legal prefix history only.

Centered, cached, or otherwise leaky transforms are rejected at this layer.

### 4. Evaluation geometry

`modules/split_planning.py` (`src/euclid/modules/split_planning.py`) defines walk-forward folds, development and confirmatory segments, scored-origin panels, entity weighting, and horizon weights.

The result is a frozen comparison geometry that later scoring code and publication rules must respect.

### 5. Search planning

`modules/search_planning.py` (`src/euclid/modules/search_planning.py`) freezes:

- search class
- canonicalization policy
- budgets and proposal limits
- frontier axes
- shortlist and freeze artifacts
- descriptive scope versus law-eligible scope

Confirmatory, robustness, and other non-search axes are intentionally excluded from search surfaces.

### 6. Candidate fitting

`modules/candidate_fitting.py` (`src/euclid/modules/candidate_fitting.py`) fits shortlisted CIR candidates on declared windows and emits fitted artifacts plus optimizer diagnostics.

Fitting geometry is explicit through `FitStrategySpec`. The default `legacy_one_step` keeps old one-step behavior, while `recursive_rollout`, `direct_analytic`, `joint_analytic`, and `rectify_analytic` use legal training-origin panels and rollout objective rows when configured. Non-contiguous horizon sets are legal when the panel declares the corresponding origin/horizon rows.

`modules/shared_plus_local_decomposition.py` handles panel-specific shared-plus-local fitting and unseen-entity constraints. `src/euclid/modules/shared_plus_local_decomposition.py` handles panel-specific shared-plus-local fitting and unseen-entity constraints.

### 7. Evaluation and scoring

`modules/evaluation.py` (`src/euclid/modules/evaluation.py`) produces point prediction artifacts.

`modules/probabilistic_evaluation.py` (`src/euclid/modules/probabilistic_evaluation.py`) produces non-point artifacts for:

- `distribution`
- `interval`
- `quantile`
- `event_probability`

Production non-point artifacts are residual-history-backed and cite stochastic model manifest evidence through `residual_history_refs` and `stochastic_model_refs`. Family-aware scoring and calibration keep Gaussian, Student-t, Laplace, interval, quantile, and event-probability evidence from being treated as interchangeable. A heuristic Gaussian compatibility artifact remains readable, but it downgrades when those production refs are missing.

`modules/scoring.py` (`src/euclid/modules/scoring.py`) then scores those artifacts under explicitly compatible comparison geometry. Cross-object comparisons are invalid by design.

### 8. Calibration, decision rules, and gates

`modules/calibration.py` applies object-type-specific calibration policies and results. The resolving source path is `src/euclid/modules/calibration.py`.

`modules/evaluation_governance.py` (`src/euclid/modules/evaluation_governance.py`) builds:

- comparison keys
- comparison universes
- baseline registry
- evaluation event logs
- predictive gate policy
- confirmatory promotion rules

`modules/gate_lifecycle.py` (`src/euclid/modules/gate_lifecycle.py`) resolves scorecard status from admissibility, robustness, comparator, time-safety, calibration, and mechanistic inputs.

### 9. Claims, replay, and publication

`modules/claims.py` maps scorecards into claim cards or abstentions and caps interpretation scope. The resolving source path is `src/euclid/modules/claims.py`.

`modules/replay.py` builds reproducibility bundles and verifies replay. The resolving source path is `src/euclid/modules/replay.py`.

`modules/catalog_publishing.py` assembles run results, publication records, and local catalog entries, subject to replay and readiness constraints. The resolving source path is `src/euclid/modules/catalog_publishing.py`.

## Forecast object semantics

Point and non-point outputs are separate publication lanes.

- Point outputs can support predictive-within-scope publication if the surrounding scorecard, comparator, and replay requirements pass.
- Non-point outputs require forecast-object-specific evaluation and, for predictive promotion, successful calibration.
- When a publishable point law and a publishable probabilistic lane share the same validation scope and publication record, downstream surfaces can present the internal `holistic_equation` payload as a unified deterministic-plus-stochastic equation for the series.

The main forecast object types are declared in code and schemas as:

- `point`
- `distribution`
- `interval`
- `quantile`
- `event_probability`

## Key invariants

- All timestamps are normalized to ISO-8601 UTC.
- Snapshots are deterministic under input permutation.
- Comparison requires matching forecast object type, score policy, horizon set, scored-origin set, and panel geometry.
- Publication requires replay-verifiable bundles.
- Public catalog publication additionally requires a readiness judgment.

## Useful source anchors

- `modules/ingestion.py` (`src/euclid/modules/ingestion.py`)
- `modules/snapshotting.py` (`src/euclid/modules/snapshotting.py`)
- `modules/timeguard.py` (`src/euclid/modules/timeguard.py`)
- `modules/features.py` (`src/euclid/modules/features.py`)
- `modules/split_planning.py` (`src/euclid/modules/split_planning.py`)
- `modules/search_planning.py` (`src/euclid/modules/search_planning.py`)
- `modules/candidate_fitting.py` (`src/euclid/modules/candidate_fitting.py`)
- `modules/shared_plus_local_decomposition.py` (`src/euclid/modules/shared_plus_local_decomposition.py`)
- `modules/evaluation.py` (`src/euclid/modules/evaluation.py`)
- `modules/probabilistic_evaluation.py` (`src/euclid/modules/probabilistic_evaluation.py`)
- `modules/scoring.py` (`src/euclid/modules/scoring.py`)
- `modules/calibration.py` (`src/euclid/modules/calibration.py`)
- `modules/evaluation_governance.py` (`src/euclid/modules/evaluation_governance.py`)
- `modules/gate_lifecycle.py` (`src/euclid/modules/gate_lifecycle.py`)
- `modules/claims.py` (`src/euclid/modules/claims.py`)
- `modules/replay.py` (`src/euclid/modules/replay.py`)
- `modules/catalog_publishing.py` (`src/euclid/modules/catalog_publishing.py`)


## Claim lanes and evidence ceilings

The production claim-lane ids are `descriptive_structure`, `predictive_within_declared_scope`, and `mechanistically_compatible_law`. Legacy fixture aliases remain compatibility labels inside older bundles, but the live documentation and contract layer use the production names. `modules/claims.py` maps scorecards into claim cards or abstentions and caps interpretation scope. Publication is gated by scorecards, replay, readiness, calibration where required, robustness, and bound evidence contracts.

`modules/replay.py` builds reproducibility bundles and verifies replay. `modules/catalog_publishing.py` assembles run results, publication records, and local catalog entries, subject to replay and readiness constraints.
