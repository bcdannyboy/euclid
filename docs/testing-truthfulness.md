# Testing And Truthfulness

Euclid's test suite is an executable specification for runtime behavior, public claims, and abstention rules. It shows what the repo can promise, what it must refuse to claim, and which parts of the equation story are pinned down in code.

## Core runtime tests

Important suites:

- `tests/unit/test_cli_smoke.py`
- `tests/integration/test_cli_end_to_end.py`
- `tests/unit/control_plane/*`
- `tests/unit/storage/*`
- `tests/unit/operator_runtime/*`
- `tests/integration/test_operator_run_pipeline.py`
- `tests/integration/test_operator_replay_pipeline.py`

These tests pin:

- public CLI claims
- sealed-run workspace layout
- execution-state recording
- content-addressed storage
- replay verification
- extension-lane runtime behavior

## Modeling pipeline tests

Important suites:

- `tests/unit/modules/*`
- `tests/integration/test_phase03_planning_governance.py`
- `tests/integration/test_phase04_search_fixtures.py`
- `tests/integration/test_phase05_*`
- `tests/integration/test_phase06_*`
- `tests/integration/test_phase07_*`
- `tests/integration/test_phase08_*`

These tests pin:

- time-safety and revision semantics
- feature leakage rejection
- search planning constraints
- point and probabilistic evaluation behavior
- calibration and comparator logic
- claim, abstention, and replay publication rules

## Search and CIR tests

Important suites:

- `tests/unit/search/*`
- `tests/unit/search/dsl/*`
- `tests/unit/cir/*`
- `tests/unit/reducers/*`
- `tests/unit/math/*`
- `tests/unit/prototype/*`
- integration tests for exact, bounded, equality-saturation, stochastic, and multi-backend search

These tests pin:

- search-class disclosures
- deterministic replay hooks and seeds
- canonical CIR identity
- reducer composition semantics
- descriptive coding math
- algorithmic DSL parsing and enumeration

## Benchmark and authority tests

Important suites:

- `tests/unit/contracts/*`
- `tests/unit/manifests/*`
- `tests/unit/benchmarks/*`
- `tests/benchmarks/*`
- `tests/unit/test_release.py`
- `tests/unit/test_release_policy_loading.py`
- `tests/integration/test_clean_install_operator_runtime.py`
- `tests/integration/test_release_candidate_workflow.py`
- `tests/fixtures/test_full_vision_certification_fixtures.py`

These tests pin:

- schema ownership and typed refs
- runtime manifest round-trips
- benchmark task and suite semantics
- readiness scope separation
- release policy truthfulness
- packaging, readiness, fixture, and release-surface agreement

## Workbench tests

Important suites:

- `tests/unit/workbench/*`
- `tests/frontend/workbench/*.js`
- `tests/integration/test_workbench_analysis.py`
- `scripts/workbench_ui_smoke.py`

These tests pin:

- API contract and saved-analysis normalization
- conservative claim gating
- UI busy, failure, and no-winner states
- shared atlas selectors and URL synchronization
- explainer fallback behavior

## Practical reading strategy

When changing behavior:

1. read the subsystem source file
2. read the strongest unit test for that file
3. read the integration test that proves the subsystem in context
4. read the benchmark, fixture, or release test if the change affects public claims
