# Modeling Pipeline

Euclid's modeling layer lives in `src/euclid/modules`. It turns ordered observations into descriptive equations, evaluates whether any candidate earns predictive support, and resolves publication status only after scoring, calibration, robustness, replay, and any required evidence gates.

The goal is still to discover legible mathematical structure in an ordered sequence, but to do it under explicit comparison geometry and publication discipline.

When a publishable point law and a publishable probabilistic lane share validation scope and publication record, downstream surfaces can expose that joined result as one unified deterministic-plus-stochastic equation.

## Stage map

### 1. Ingestion

`modules/ingestion.py` canonicalizes raw tabular inputs into ordered observation records.

Core invariants:

- targets may be missing in raw lineage
- coded targets exclude missing values
- present numeric values must be finite
- records are ordered and typed before downstream use

### 2. Snapshotting and timeguard

`modules/snapshotting.py` freezes visible observations at a cutoff and chooses the latest visible revision per `(entity, event_time)`.

`modules/timeguard.py` audits that:

- `available_at <= cutoff`
- `event_time <= cutoff`

This is the first hard time-safety boundary in the pipeline.

### 3. Feature materialization

`modules/features.py` builds retained-scope feature views from legal prefix history only.

Centered, cached, or otherwise leaky transforms are rejected at this layer.

### 4. Evaluation geometry

`modules/split_planning.py` defines walk-forward folds, development and confirmatory segments, scored-origin panels, entity weighting, and horizon weights.

The result is a frozen comparison geometry that later scoring code and publication rules must respect.

### 5. Search planning

`modules/search_planning.py` freezes:

- search class
- canonicalization policy
- budgets and proposal limits
- frontier axes
- shortlist and freeze artifacts
- descriptive scope versus law-eligible scope

Confirmatory, robustness, and other non-search axes are intentionally excluded from search surfaces.

### 6. Candidate fitting

`modules/candidate_fitting.py` fits shortlisted CIR candidates on declared windows and emits fitted artifacts plus optimizer diagnostics.

`modules/shared_plus_local_decomposition.py` handles panel-specific shared-plus-local fitting and unseen-entity constraints.

### 7. Evaluation and scoring

`modules/evaluation.py` produces point prediction artifacts.

`modules/probabilistic_evaluation.py` produces non-point artifacts for:

- `distribution`
- `interval`
- `quantile`
- `event_probability`

`modules/scoring.py` then scores those artifacts under explicitly compatible comparison geometry. Cross-object comparisons are invalid by design.

### 8. Calibration, decision rules, and gates

`modules/calibration.py` applies object-type-specific calibration policies and results.

`modules/evaluation_governance.py` builds:

- comparison keys
- comparison universes
- baseline registry
- evaluation event logs
- predictive gate policy
- confirmatory promotion rules

`modules/gate_lifecycle.py` resolves scorecard status from admissibility, robustness, comparator, time-safety, calibration, and mechanistic inputs.

### 9. Claims, replay, and publication

`modules/claims.py` maps scorecards into claim cards or abstentions and caps interpretation scope.

`modules/replay.py` builds reproducibility bundles and verifies replay.

`modules/catalog_publishing.py` assembles run results, publication records, and local catalog entries, subject to replay and readiness constraints.

## Forecast object semantics

Point and non-point outputs are separate publication lanes.

- Point outputs can support predictive-law publication if the surrounding scorecard, comparator, and replay requirements pass.
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

- `src/euclid/modules/ingestion.py`
- `src/euclid/modules/snapshotting.py`
- `src/euclid/modules/timeguard.py`
- `src/euclid/modules/features.py`
- `src/euclid/modules/split_planning.py`
- `src/euclid/modules/search_planning.py`
- `src/euclid/modules/candidate_fitting.py`
- `src/euclid/modules/evaluation.py`
- `src/euclid/modules/probabilistic_evaluation.py`
- `src/euclid/modules/scoring.py`
- `src/euclid/modules/calibration.py`
- `src/euclid/modules/evaluation_governance.py`
- `src/euclid/modules/gate_lifecycle.py`
- `src/euclid/modules/claims.py`
- `src/euclid/modules/replay.py`
- `src/euclid/modules/catalog_publishing.py`
