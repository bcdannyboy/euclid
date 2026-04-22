# Runtime And CLI

The main way to run Euclid is through `euclid` and `python -m euclid`.

The certified runtime is the governed path from ordered observations to replayable symbolic outputs: descriptive equations, predictive-within-scope symbolic claims, and, when a publishable point-lane claim and a publishable probabilistic lane clear the same scope, the unified deterministic-plus-stochastic equation surface shown downstream.

The root CLI is defined in `src/euclid/cli/__init__.py`. It exposes the main operator workflow and labels `demo` as compatibility-only tooling.

## Certified commands

Root commands and command groups include:

- `smoke`
- `run`
- `replay`
- `benchmarks`
- `benchmark`
- `release`
- `workbench`

The most important certified commands are:

- `euclid run --config examples/current_release_run.yaml`
- `euclid replay --run-id <run-id>`
- `euclid benchmarks run --suite current-release.yaml`
- `euclid release status`
- `euclid release verify-completion`
- `euclid release certify-clean-install`
- `euclid release certify-research-readiness`
- `euclid workbench serve`

## Compatibility-only commands

The demo tree remains packaged because the repo still ships retained local demos and inspection helpers, but the CLI help text explicitly treats them as non-certified surfaces. Those commands live in `src/euclid/demo.py` and the parallel orchestration code in `src/euclid/operator_runtime/_compat_runtime.py`.

## Canonical Assets

Runtime assets resolve through `euclid.operator_runtime.resources`. Packaged
assets under `src/euclid/_assets` are canonical for examples, fixtures,
contracts, readiness policies, notebooks, and workbench files. When
`project_root` or `EUCLID_PROJECT_ROOT` points at a checkout, the resolver
prefers `src/euclid/_assets` and then the checkout mirror. Direct CLI paths such
as `examples/current_release_run.yaml` remain runnable in the checkout; packaged
asset tests should use the resolver when they are proving wheel/runtime
resource behavior.

Missing packaged assets raise `EuclidAssetError` with code
`euclid_asset_missing`. This replaces generic file-existence behavior so release
and replay failures carry typed evidence.

## Numerical Runtime

The declared runtime and dev/test dependency foundation includes NumPy, pandas,
SciPy, SymPy, Pint, statsmodels, scikit-learn, PySINDy, PySR, egglog, joblib,
SQLAlchemy, Pydantic, PyYAML, Typer, PyArrow, httpx, python-dotenv, vcrpy,
responses/respx, pytest-timeout, pytest-xdist, and Hypothesis. The pytest
commands in this repository require the dev/test dependencies because
`pyproject.toml` enables strict config and `pytest-timeout`. Ray is
intentionally optional via the `distributed` extra because distributed execution
is a policy decision, not a default local requirement.

Replay bundles record numerical environment metadata through
`euclid.runtime.numerical_environment`: Python version, package versions, Julia
availability for PySR, BLAS/LAPACK status, platform, and CPU metadata. PySR
package metadata is captured even when Julia is unavailable; Julia absence is a
typed runtime diagnostic rather than a silent import failure.

## Operator request model

The certified request model is `OperatorRequest` in `src/euclid/operator_runtime/models.py`.

Important fields:

- `request_id`
- `manifest_path`
- `dataset_csv`
- `cutoff_available_at`
- `quantization_step`
- `minimum_description_gain_bits`
- `min_train_size`
- `horizon`
- `search_family_ids`
- `search_class`
- `search_seed`
- `proposal_limit`
- `seasonal_period`
- `forecast_object_type`
- `calibration_thresholds`
- `external_evidence_payload`
- `mechanistic_evidence_payload`
- `robustness_payload`
- `declared_entity_panel`

`extension_lane_ids` are derived deterministically from the chosen forecast object type, families, and optional evidence lanes.

## Workspace layout

The runtime uses `RuntimeWorkspace` from `src/euclid/control_plane/workspace.py`.

For operator runs, the materialized surfaces are:

- `active-runs/<run_id>/control-plane/execution-state.sqlite3`
- `sealed-runs/<run_id>/artifacts`
- `sealed-runs/<run_id>/registry.sqlite3`
- `sealed-runs/<run_id>/run-summary.json`

The control plane uses:

- `FileLock` in `src/euclid/control_plane/locking.py`
- `SQLiteExecutionStateStore` in `src/euclid/control_plane/execution_state.py`
- SQLite-backed metadata and lineage storage in `src/euclid/control_plane/sqlite_store.py`

## Artifact and metadata invariants

Artifact storage is content-addressed and fail-closed:

- JSON canonicalization rejects non-finite floats and non-string keys
- hashes are `sha256:...`
- writes are atomic
- reads re-verify integrity

Metadata storage rejects a repeated `schema_name/object_id` pair if the content hash changes. It also keeps lineage edges and typed-reference edges for later replay and inspection.

## Replay and reproducibility

Replay is part of the product, not an afterthought. The runtime builds a reproducibility bundle with:

- required manifest refs
- artifact hash records
- seed records
- stage order

Replay is then executed and must verify before publication completes. The main replay helpers live in:

- `src/euclid/operator_runtime/replay.py`
- `src/euclid/modules/replay.py`

That matters because Euclid only publishes symbolic claims that close under replay. Predictive-within-scope symbolic claims and unified equations are governed runtime artifacts with seed, manifest, and stage-order provenance, not post hoc renderings layered on after the run.

## Extension lanes

The runtime can extend beyond plain point evaluation:

- probabilistic forecast objects: `distribution`, `interval`, `quantile`, `event_probability`
- external evidence
- mechanistic evidence
- shared-plus-local decomposition
- robustness override lanes

These lanes change required artifacts, admissibility rules, and publication behavior. Shared-plus-local runs require declared entity panel equality, mechanistic publication requires external evidence, and residual structure can block predictive-within-scope publication even when descriptive status passes. A unified deterministic-plus-stochastic equation only appears when the probabilistic lane also reaches publishable status on the same validation scope and publication record.

## Smoke and packaging surfaces

Operational scripts in `scripts/` cover smoke and packaging paths:

- `scripts/benchmark_smoke.sh`
- `scripts/benchmark_suite.sh`
- `scripts/demo.sh`
- `scripts/install_smoke.sh`
- `scripts/release_smoke.sh`
- `scripts/fmp_euclid_smoke.py`
- `scripts/workbench_ui_smoke.py`

They exercise public entrypoints, but they do not replace the release and readiness gates in `src/euclid/release.py`.
