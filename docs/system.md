# System

Euclid is a forecasting system for ordered observations. It derives compact equations that describe a time series, tests whether those equations earn predictive support under replayable evidence, and publishes only the claim surface the run can defend.

The project is grounded in replayable experiments rather than post hoc interpretation.

When a publishable point law and a publishable probabilistic lane clear the same validation scope and publication record, downstream surfaces can show one unified deterministic-plus-stochastic equation for the series.

The package surface in `src/euclid/__init__.py` exports three broad classes of functionality:

- main operator entrypoints such as `run_operator`, `replay_operator`, benchmark profiling, and release verification
- compatibility-only demo and inspection helpers
- packaging and smoke helpers for local workflows

## Subsystem map

### 1. Certified runtime and control plane

Primary code:

- `src/euclid/cli`
- `src/euclid/operator_runtime`
- `src/euclid/control_plane`
- `src/euclid/storage`
- `src/euclid/artifacts`
- `src/euclid/runtime`

This surface accepts run requests, materializes runtime workspaces, records execution-state events, stores manifests and artifacts content-addressably, runs replay, and emits run summaries and publication artifacts.

### 2. Modeling pipeline

Primary code:

- `src/euclid/modules`

This surface handles time-safe ingestion, snapshots, feature construction, split geometry, search planning, candidate fitting, evaluation, calibration, robustness, decision gates, claims, replay bundles, and catalog publication.

### 3. Search and model semantics

Primary code:

- `src/euclid/search`
- `src/euclid/cir`
- `src/euclid/reducers`
- `src/euclid/adapters`
- `src/euclid/math`
- `src/euclid/prototype`

This surface defines reducer families, search classes, the candidate intermediate representation, descriptive coding, backends, and adapter normalization for multi-backend comparison.

### 4. Formal specification and release evidence

Primary code and data:

- `src/euclid/contracts`
- `src/euclid/manifests`
- `src/euclid/benchmarks`
- `src/euclid/readiness`
- `schemas`
- `benchmarks`
- `tools/spec_compiler`

This surface defines the formal schemas, runtime artifact families, benchmark task and suite manifests, readiness matrices, and the compiler that checks whether the published story matches the code and fixtures.

### 5. Local analysis workbench

Primary code:

- `src/euclid/workbench`
- `src/euclid/_assets/workbench`

This surface serves a local UI for exploring saved or freshly computed analyses, with explicit distinctions between descriptive equations, predictive laws, the internal `holistic_equation` payload that backs the public unified-equation surface, and uncertainty attachments. It is also where the repo's unified-equation story becomes visible: a published deterministic law joined to a publishable probabilistic lane only when both sides clear the same scope.

## Execution planes

Euclid’s behavior falls into three practical planes:

- Data plane: observation ingestion, snapshotting, timeguard audits, and feature materialization
- Search and evaluation plane: search planning, candidate fitting, scoring, calibration, robustness, decision gates, and symbolic law discovery
- Publication and release plane: scorecards, claim or abstention resolution, replay bundles, run results, publication records, readiness judgments, benchmark reports, and release evidence

## End-to-end lifecycle

1. Load a run request and resolve the packaged assets and formal specs it needs.
2. Create or resolve a runtime workspace with active and sealed run roots.
3. Ingest observations and freeze a time-safe snapshot at the declared cutoff.
4. Build features, evaluation geometry, and search constraints.
5. Search bounded candidate spaces and freeze a shortlist or accepted candidate.
6. Fit candidate artifacts and emit point or probabilistic prediction artifacts.
7. Score, calibrate, and enrich those artifacts with robustness or mechanistic evidence.
8. Resolve a scorecard into a claim card or abstention.
9. Build and verify a reproducibility bundle.
10. Persist a run result, publication record, and optional readiness or benchmark evidence.

## Code anchors

The most useful high-level anchors are:

- `src/euclid/cli/__init__.py`
- `src/euclid/operator_runtime/workflow.py`
- `src/euclid/operator_runtime/_compat_runtime.py`
- `src/euclid/modules/search_planning.py`
- `src/euclid/modules/scoring.py`
- `src/euclid/search/backends.py`
- `src/euclid/manifests/runtime_models.py`
- `src/euclid/benchmarks/runtime.py`
- `src/euclid/release.py`
- `src/euclid/workbench/service.py`
