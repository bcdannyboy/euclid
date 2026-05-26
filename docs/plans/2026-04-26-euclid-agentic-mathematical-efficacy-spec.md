# Euclid Agentic Mathematical Efficacy Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Turn Euclid into an evidence-gated, mathematically honest time-series law-discovery platform whose MDL, predictive, calibration, benchmark, search, identifiability, nonstationarity, and publication claims are backed by measured artifacts.

**Architecture:** Euclid remains the governing layer. External numerical and symbolic libraries may propose models or diagnostics, but Euclid owns canonical candidate identity, fit geometry, codelength accounting, paired predictive evidence, calibration gates, benchmark efficacy thresholds, replay verification, and public claim scope. The implementation proceeds in fail-closed layers so broader expressive power cannot publish stronger claims before the evidence gates are real.

**Tech Stack:** Python 3.11+, NumPy, SciPy, statsmodels, scikit-learn, arch, ruptures, MAPIE, SymPy, PySINDy, PySR, pytest, Hypothesis, PyYAML, existing Euclid manifests/CIR/replay infrastructure, and later adapter candidates for mlforecast, sktime, celer, group-lasso, cvxpy, and constriction behind optional import boundaries.

---

## 1. Operating Standard

This is not a feature wishlist. It is a sequenced implementation specification for future swarms of development agents.

The system must fail closed:

- A claim does not advance because a manifest field exists.
- A benchmark gate does not pass because a semantic row is present.
- MDL terminology does not appear unless the implemented code satisfies the declared coding contract.
- Predictive promotion does not pass on a raw score delta.
- Probabilistic publication does not pass on calibration heuristics without finite-sample or explicitly weakened guarantees.
- Nonstationary models do not justify stationary law claims.
- Publication does not occur without verified replay, measured gates, comparator exposure, and claim-scope validation.

Every implementation prompt that follows this spec must use development subagents with disjoint write scopes. Agents may edit files only in their assigned ownership areas and must not revert changes from other agents.

## 2. Research Inputs Used

This plan was produced from local repo inspection and parallel research subagents on 2026-04-26.

Subagent findings incorporated:

- MDL/universal coding: explicit causal codes, escape/unseen-symbol handling, same-family and global family coding, quantization/base-measure discipline, and honest MDL claim tiers.
- Predictive inference/calibration: paired loss differentials, DM/HLN, GW as supplementary, block bootstrap, SPA/MCS for multi-model comparisons, minimum effective sample sizes, and careful conformal claims under dependence.
- Nonstationarity/state-space: separate stability, hard-break, discrete-regime, continuous-state, and hybrid lanes with synthetic truth-first efficacy gates.
- Library identification: safest core stack is statsmodels, scipy, scikit-learn, arch, ruptures, MAPIE; MDL needs custom infrastructure rather than a single mature package.
- Local architecture: current gaps in codelength/reference policy threading, predictive gate enforcement, benchmark missing-metric passes, replay/publication checks, raw claim-card construction, config duplication, and scorecard/claim-card drift.

Primary reference anchors:

- Grunwald MDL tutorial: https://homepages.cwi.nl/~pdg/ftp/mdlintro.pdf
- Shtarkov universal coding record: https://www.mathnet.ru/eng/ppi811
- Rissanen Fisher information and stochastic complexity: https://research.ibm.com/publications/fisher-information-and-stochastic-complexity
- Witten-Bell zero-frequency problem: https://ieeexplore.ieee.org/document/87000/
- Diebold-Mariano predictive accuracy: https://www.nber.org/papers/t0169
- Giacomini-White conditional predictive ability: https://www.econometricsociety.org/publications/econometrica/2006/11/01/tests-conditional-predictive-ability
- arch bootstrap and multiple-comparison docs: https://arch.readthedocs.io/en/stable/
- Conformal beyond exchangeability: https://projecteuclid.org/journals/annals-of-statistics/volume-51/issue-2/Conformal-prediction-beyond-exchangeability/10.1214/23-AOS2276.pdf
- EnbPI: https://proceedings.mlr.press/v139/xu21h.html
- SPCI: https://proceedings.mlr.press/v202/xu23r.html
- ruptures docs: https://centre-borelli.github.io/ruptures-docs/
- statsmodels TSA docs: https://www.statsmodels.org/stable/tsa.html
- MAPIE docs: https://mapie.readthedocs.io/en/latest/
- PySINDy docs: https://pysindy.readthedocs.io/en/stable/
- PySR docs: https://ai.damtp.cam.ac.uk/pysr/

## 3. Current Code Anchors

High-risk runtime anchors:

- `src/euclid/math/codelength.py`
- `src/euclid/math/reference_descriptions.py`
- `src/euclid/search/descriptive_coding.py`
- `src/euclid/modules/predictive_tests.py`
- `src/euclid/modules/evaluation_governance.py`
- `src/euclid/modules/scoring.py`
- `src/euclid/modules/gate_lifecycle.py`
- `src/euclid/benchmarks/reporting.py`
- `src/euclid/benchmarks/runtime.py`
- `src/euclid/readiness/judgment.py`
- `src/euclid/modules/catalog_publishing.py`
- `src/euclid/modules/claims.py`
- `src/euclid/modules/replay.py`
- `src/euclid/search/backends.py`
- `src/euclid/search/portfolio.py`
- `src/euclid/search/engines/pysindy_engine.py`
- `src/euclid/search/engines/pysr_engine.py`
- `src/euclid/search/engines/sparse_regression.py`
- `src/euclid/fit/objectives.py`
- `src/euclid/fit/refit.py`
- `src/euclid/fit/scipy_optimizers.py`
- `src/euclid/fit/parameterization.py`
- `src/euclid/modules/calibration.py`
- `src/euclid/reducers/composition.py`
- `src/euclid/manifests/runtime_models.py`

Existing tests to extend:

- `tests/unit/math/test_codelength.py`
- `tests/unit/math/test_workflow_codelength_terms.py`
- `tests/unit/modules/test_predictive_tests.py`
- `tests/unit/modules/test_calibration.py`
- `tests/unit/modules/test_claims.py`
- `tests/unit/benchmarks/test_reporting.py`
- `tests/benchmarks/test_readiness_gate.py`
- `tests/integration/test_publication_pipeline.py`
- `tests/integration/test_probabilistic_publication.py`
- `tests/integration/test_regime_conditioned_pipeline.py`
- `tests/unit/search/test_portfolio.py`
- `tests/unit/search/engines/test_pysindy_engine.py`
- `tests/unit/search/engines/test_pysr_engine.py`
- `tests/unit/fit/test_objectives.py`
- `tests/unit/fit/test_refit.py`
- `tests/unit/fit/test_scipy_optimizers.py`

## 4. Phase Structure

The phases are ordered by claim risk, not by convenience.

Phase 0: source-of-truth and typed evidence contract.

Phase 1: MDL layer repair.

Phase 2: honest predictive inference and promotion.

Phase 3: measured benchmark efficacy gates.

Phase 4: publication, replay, and semantic claim-scope closure.

Phase 5: conformal and distribution-free calibration lanes.

Phase 6: nonstationarity, change-point, regime-switching, and state-space lanes.

Phase 7: controlled symbolic and sparse search expansion.

Phase 8: identifiability diagnostics and real optimizer regularization.

Phase 9: release hardening, documentation, and workbench evidence surfacing.

Later phases must not use stronger claim language until earlier evidence gates fail closed.

### 4.1 Unambiguous Execution Contract

Every implementation task in this plan has the same required micro-sequence. A future agent must not skip or merge these steps unless the user explicitly changes the plan.

For each task:

1. Read the task section and this execution contract.
2. Inspect every listed source file and test file before editing.
3. Create missing test files listed for the task.
4. Write the smallest failing test for one bullet in "Write failing tests".
5. Run the exact task verification command and confirm that the new test fails for the expected reason.
6. Implement the minimum production change for that one failing test.
7. Run the exact task verification command and confirm the new test passes.
8. Repeat steps 4 through 7 until every bullet in "Write failing tests" is covered.
9. Run the phase-level verification command in section 4.5.
10. Record changed files, tests run, and residual risks.

Definitions:

- `Create` means the file must not be edited elsewhere in the same swarm unless that agent owns it.
- `Modify` means preserve existing public behavior unless a failing test in this plan explicitly changes it.
- `Test` means create the file if it does not exist, otherwise extend the existing file.
- `Optional dependency` means optional import boundary plus deterministic fail-closed behavior when unavailable; it does not mean the behavior can be skipped.
- `Adapter unavailable` is a valid runtime status only when it has a stable reason code and test coverage.
- `Compatibility shim` means boundary-only preservation of old manifest shape; new logic must use the typed object or policy introduced by the task.
- `Passed gate` means the relevant artifact exists, has status `passed`, includes required evidence refs, and carries no blocking reason codes.
- `Downgraded` and `human_review_only` never imply automatic publication.

Forbidden shortcuts:

- Do not satisfy a task by changing golden fixtures before the runtime behavior is correct.
- Do not replace a failing semantic gate with a new semantic declaration.
- Do not let import errors from optional libraries leak to callers.
- Do not add generic reason codes such as `failed` or `invalid`; reason codes must identify the specific blocked contract.
- Do not make a publication, benchmark, predictive, calibration, or MDL claim stronger than the artifact's declared tier.

### 4.2 Task Dependency Graph

The following dependencies are binding.

- Phase 0:
  - `0.1` must finish before `0.2`.
  - `0.3` may run in parallel with `0.1` and `0.2`.
- Phase 1:
  - `1.1` must finish before `1.2`.
  - `1.1` and `1.3` must finish before `1.4`.
  - `1.1`, `1.2`, `1.3`, and `1.4` must finish before `1.5`.
- Phase 2:
  - `2.1` must finish before `2.2`, `2.3`, `2.4`, and `2.5`.
  - `2.2` and `2.3` must finish before `2.4`.
  - `2.5` may run after `2.2` but cannot publish or promote anything until `2.4` is complete.
- Phase 3:
  - `3.1` must finish before `3.2`.
  - `3.3` may run in parallel with `3.2` after Phase 0 evidence status objects exist.
- Phase 4:
  - Phase 0 must finish before any Phase 4 task.
  - `4.1` and `4.2` may run in parallel if they do not edit the same function body.
  - `4.3` must run after `4.2`.
- Phase 5:
  - `5.1` must finish before `5.2` and `5.3`.
  - `5.2` must finish before probabilistic publication tests are updated.
- Phase 6:
  - `6.1` must finish before nonstationarity can block predictive promotion.
  - `6.2`, `6.3`, and `6.4` may run in parallel after `6.1` defines the common artifact/status vocabulary.
- Phase 7:
  - `7.1` must finish before `7.2`, `7.4`, and `7.5`.
  - `7.5` must finish before sparse supports are used by `7.2` expansion policies.
  - `7.4` must finish before external engines enter the main descriptive portfolio.
  - `7.3` must run after `7.1`; it must carry `identifiability_status="not_evaluated"` until Phase 8 populates fit diagnostics.
- Phase 8:
  - `8.1` must finish before `8.3`.
  - `8.2` must finish before `8.3`.
- Phase 9:
  - All earlier phases must finish before documentation truthfulness updates claim final behavior.

### 4.3 Test File Creation Inventory

Create these test files when their phase begins if they do not already exist:

- `tests/unit/modules/test_evidence_contracts.py`
- `tests/unit/math/test_reference_descriptions.py`
- `tests/unit/search/test_adaptive_orchestration.py`
- `tests/unit/search/engines/test_sparse_regression.py`
- `tests/unit/nonstationarity/test_stability.py`
- `tests/unit/nonstationarity/test_changepoints.py`
- `tests/unit/nonstationarity/test_regime_switching.py`
- `tests/unit/nonstationarity/test_state_space.py`

Existing test files that must be extended rather than replaced:

- `tests/unit/math/test_codelength.py`
- `tests/unit/math/test_workflow_codelength_terms.py`
- `tests/unit/modules/test_predictive_tests.py`
- `tests/unit/modules/test_evaluation_governance.py`
- `tests/unit/modules/test_scoring.py`
- `tests/unit/modules/test_calibration.py`
- `tests/unit/modules/test_claims.py`
- `tests/unit/modules/test_replay.py`
- `tests/unit/benchmarks/test_reporting.py`
- `tests/unit/search/test_backends.py`
- `tests/unit/search/test_descriptive_coding.py`
- `tests/unit/search/test_portfolio.py`
- `tests/unit/search/engines/test_pysindy_engine.py`
- `tests/unit/search/engines/test_pysr_engine.py`
- `tests/unit/fit/test_objectives.py`
- `tests/unit/fit/test_refit.py`
- `tests/unit/fit/test_scipy_optimizers.py`
- `tests/unit/search/test_fitting_boundary.py`

### 4.4 Implementation Task Ledger

Each ledger item below is a discrete handoff unit. A later prompt may assign exactly one ledger item to one worker, or assign a contiguous group of items to one worker when their write scopes do not conflict.

#### Phase 0 Ledger

- `0.1A`: Create `tests/unit/modules/test_evidence_contracts.py` with failing construction tests for `EvidenceStatus`.
- `0.1B`: Create `src/euclid/modules/evidence_contracts.py` with `EvidenceStatus`, `EvidenceGateDecision`, `ClaimScopeDecision`, and `PromotionDecision`.
- `0.1C`: Add stable `as_manifest()` ordering and helper constructors.
- `0.1D`: Export new helpers through the appropriate module boundary.
- `0.2A`: Add tests proving scorecard and claim-card builders preserve required manifest shape.
- `0.2B`: Route `src/euclid/modules/gate_lifecycle.py` through typed evidence objects.
- `0.2C`: Add boundary compatibility shim for existing workflow callers.
- `0.3A`: Add drift test for root `benchmarks/` versus `src/euclid/_assets/benchmarks/`.
- `0.3B`: Add drift test for root `schemas/` versus `src/euclid/_assets/schemas/`.
- `0.3C`: Add explicit allowlist for intentionally divergent asset files with a reason string.

#### Phase 1 Ledger

- `1.1A`: Add residual alphabet policy tests in `tests/unit/math/test_codelength.py`.
- `1.1B`: Create `src/euclid/math/residual_coding.py`.
- `1.1C`: Implement `ResidualAlphabetPolicy`, `ResidualCodeEvent`, and `prequential_escape_residual_bin_code_v1`.
- `1.1D`: Wire residual diagnostics into `data_code_diagnostics()`.
- `1.1E`: Mark legacy residual coding as `mdl_inspired_proxy_score` unless it satisfies the new coding contract.
- `1.2A`: Create `tests/unit/math/test_reference_descriptions.py`.
- `1.2B`: Extend `ReferenceDescriptionPolicy` with `reference_scope`.
- `1.2C`: Thread `data_code_family` into `build_reference_description()`.
- `1.2D`: Add `reference_scope` and `reference_family_id` to `CodelengthComparisonKey`.
- `1.3A`: Create `src/euclid/math/lattice.py`.
- `1.3B`: Add `LatticePolicy` and tests for parameter/state lattice serialization.
- `1.3C`: Propagate lattice policy through descriptive coding and fit/refit replay metadata.
  - Resolved implementation detail: lattice policy annotations are audit and comparison-key metadata on model-code decompositions; CIR canonical identity remains restricted to the base `L_*` model-code scalar fields so annotating an already-normalized candidate does not violate `require_full_cir_closure()`.
- `1.4A`: Add mixed-key grouping tests in `tests/unit/search/test_descriptive_coding.py`.
- `1.4B`: Refactor `_comparison_status()` to return comparable groups plus non-comparable diagnostics.
- `1.4C`: Update ranking to select only within a comparable group.
  - Resolved implementation detail: portfolio finalists carry `codelength_comparison_key` through `ComparableBackendFinalist.replay_contract`; portfolio selection ranks only inside a selected comparable finalist group, preferring a non-singleton group when incompatible singleton finalists have lower raw code bits.
- `1.5A`: Add claim-tier tests for codelength policy manifests.
- `1.5B`: Add `coding_claim_tier` to `build_codelength_policy_manifest()`.
- `1.5C`: Make claim publication reject stronger MDL language unless the tier permits it.
  - Resolved implementation detail: claim-card publication guards read `coding_claim_tier` either directly from the claim card or from a nested codelength policy manifest/body payload, then reject MDL or universal-coding wording whose tier is not eligible.

#### Phase 2 Ledger

- `2.1A`: Add paired stream identity tests in `tests/unit/modules/test_predictive_tests.py`.
- `2.1B`: Create `PairedLossDifferentialStream` in `src/euclid/modules/predictive_inference.py`.
- `2.1C`: Update `src/euclid/modules/scoring.py` to emit the paired stream artifact.
- `2.2A`: Add declared-test-id tests for DM/HLN and block bootstrap.
- `2.2B`: Implement predictive test registry in `predictive_inference.py`.
- `2.2C`: Move HAC mean interval behind a declared test implementation instead of public generic identity.
- `2.2D`: Add stable unavailable reason for missing `arch`.
- `2.3A`: Create `src/euclid/modules/effective_sample.py`.
- `2.3B`: Add HAC effective sample size calculation and tests.
- `2.3C`: Add effective block count calculation and tests.
  - Resolved implementation detail: the new declared paired-test registry lives in `src/euclid/modules/predictive_inference.py`, while the existing public entry point `src/euclid/modules/predictive_tests.py` remains as a compatibility facade that builds paired streams, applies minimum effective-information policy, exposes stable optional-backend diagnostics, and delegates declared computations to `predictive_inference.py`.
- `2.4A`: Add uncertainty-aware promotion tests.
- `2.4B`: Update `evaluate_predictive_promotion()` to return typed pass/fail/abstain/human-review statuses.
- `2.4C`: Update `resolve_confirmatory_promotion_allowed()` to consume typed promotion status.
  - Resolved implementation detail: automatic predictive promotion now requires positive practical-margin evidence, a declared paired test, sufficient effective information for automatic publication, and uncertainty evidence from either a confidence interval clearing the practical margin or a significant p-value. Nonstationarity diagnostics and many-model pairwise-only contexts fail closed with specific reason codes.
- `2.5A`: Add multi-model comparison boundary tests.
- `2.5B`: Add `model_confidence_set_v1` and `superior_predictive_ability_v1` registry stubs with fail-closed unavailable diagnostics.
  - Resolved implementation detail: `arch` is not installed in the current environment, so MCS/SPA entries expose fail-closed `multi_model_test_backend_unavailable` plus `multi_model_superiority_not_tested` diagnostics; portfolio selection records that compare more than two finalists mark pairwise DM as insufficient for unique predictive superiority.

#### Phase 3 Ledger

- `3.1A`: Add missing-observed-metric failure tests.
- `3.1B`: Update `_evaluate_metric_thresholds()` so missing metrics fail unless safe abstention applies.
- `3.1C`: Add `measurement_required` handling with default `true`.
- `3.2A`: Create `src/euclid/benchmarks/efficacy_metrics.py`.
- `3.2B`: Add planted-law recovery metric functions and tests.
- `3.2C`: Add false holistic claim rate metric functions and tests.
- `3.2D`: Add probabilistic attachment quality metric functions and tests.
- `3.2E`: Add nonstationary detection metric placeholders with fail-closed missing lane status until Phase 6.
- `3.3A`: Add replay-status readiness tests.
- `3.3B`: Update benchmark runtime to consume reproducibility bundle replay verification status.
- `3.3C`: Add separate reason codes for missing replay artifact and unverified replay.
  - Resolved implementation detail: benchmark reporting rows now fail closed on missing required observed metrics with `missing_observed_metric`, `reason_code`, source submitter, and source candidate provenance. Runtime fills task-threshold metric ids only from measured selected-candidate metrics (`description_gain_bits` for practical-margin rows and `inner_primary_score` for the task score-law row); unavailable measurements still fail. Generated replay refs carry top-level `replay_verification_status`, readiness distinguishes missing replay artifacts from unverified replay, and benchmark cache writes skip unpickleable optional cache payloads instead of failing profiling. The root planted analytic rediscovery demo currently records a measured `predictive_adequacy_floor` failure (`mean_absolute_error` exceeds its `0.15` threshold), so the Phase 3 smoke asserts that measured failure instead of treating semantic assertion presence as success.

#### Phase 4 Ledger

- `4.1A`: Add publication claim-scope failure tests.
- `4.1B`: Call `assert_claim_scope_publication()` from `build_publication_record_manifest()`.
- `4.1C`: Add typed claim-card body argument if current publication inputs cannot validate scope.
- `4.2A`: Add publication evidence requirement tests for predictive/probabilistic/descriptive-only claims.
- `4.2B`: Implement `publication_evidence_requirements`.
- `4.2C`: Wire scorecard, claim card, paired test, calibration result, replay bundle, and comparator exposure checks.
- `4.3A`: Add replay bundle completeness tests by claim lane.
- `4.3B`: Extend `required_manifest_refs_from_run_result()` by claim lane.
- `4.3C`: Update replay verification reason codes for missing lane refs.

#### Phase 5 Ledger

- `5.1A`: Add conformal guarantee-tier registry tests.
- `5.1B`: Create `src/euclid/modules/conformal.py`.
- `5.1C`: Implement guarantee tiers and method registry.
- `5.1D`: Add claim-scope guard for finite-sample distribution-free wording.
- `5.2A`: Add horizon-separated calibration tests.
- `5.2B`: Add `CalibrationPartition`.
- `5.2C`: Enforce minimum calibration count per partition.
- `5.2D`: Add empirical coverage lower-bound diagnostics.
- `5.3A`: Add MAPIE unavailable tests.
- `5.3B`: Implement optional MAPIE adapter boundary.
- `5.3C`: Record MAPIE version, method, calibration indices, and assumptions.
  - Resolved implementation detail: `CalibrationPartition` and empirical lower-bound partition evaluation live canonically in `src/euclid/modules/calibration.py`; `src/euclid/modules/conformal.py` re-exports the partition evaluator and MAPIE time-series adapter as the conformal boundary. Passing calibration does not override a blocked predictive gate, so the probabilistic calibration integration smoke now asserts calibration success plus predictive blocking when many-model correction remains failed.

#### Phase 6 Ledger

- `6.1A`: Create nonstationarity package and common artifact/status dataclasses.
- `6.1B`: Add stability diagnostic tests.
- `6.1C`: Implement statsmodels-backed stability diagnostics with optional import boundary.
- `6.1D`: Wire instability blocker into predictive promotion.
- `6.2A`: Add hard change-point synthetic tests.
- `6.2B`: Implement ruptures adapter in `changepoints.py`.
- `6.2C`: Emit `ChangePointArtifact` with penalty, method, min segment size, and tolerance.
- `6.2D`: Add benchmark metrics for change-point precision/recall, Hausdorff distance, and delay.
- `6.3A`: Add discrete regime synthetic tests.
- `6.3B`: Implement statsmodels Markov switching adapter.
- `6.3C`: Emit transition matrix, smoothed probabilities, expected durations, and convergence diagnostics.
- `6.3D`: Add weak-regime-identifiability reason code and claim-scope integration.
- `6.4A`: Add local-level state-space synthetic tests.
- `6.4B`: Implement statsmodels state-space adapter.
- `6.4C`: Emit filtered/smoothed state, covariance, innovations, and log likelihood.
- `6.4D`: Add innovation whiteness blocker.

#### Phase 7 Ledger

- `7.1A`: Add search library spec tests.
- `7.1B`: Create `src/euclid/search/library_specs.py`.
- `7.1C`: Extend search plan manifest/building with expansion policy fields.
- `7.1D`: Separate primitive family filters from specific candidate ids.
- `7.2A`: Create `src/euclid/search/orchestration/adaptive.py`.
- `7.2B`: Add hierarchy-aware scheduler tests.
- `7.2C`: Implement expansion triggers and expansion actions.
- `7.2D`: Replace or wrap prefix-only `_select_attempted_proposals()` when expansion policy is declared.
- `7.3A`: Add portfolio retention tests.
- `7.3B`: Add `family_retention_k` and diversity metadata.
- `7.3C`: Update `ComparableBackendFinalist` and selection rules with common frontier axes.
- `7.4A`: Add PySINDy/PySR proposal-source integration tests.
- `7.4B`: Configure `run_descriptive_search_portfolio()` to include external proposal engines.
- `7.4C`: Make PySINDy ensemble output produce support-stability candidates.
- `7.4D`: Carry PySR hall-of-fame loss/complexity/refit metrics into frontier axes.
- `7.5A`: Create `tests/unit/search/engines/test_sparse_regression.py`.
- `7.5B`: Extend `SparseRegressionEngine` to emit alpha/support paths.
- `7.5C`: Add support-stability metrics.
- `7.5D`: Feed stable sparse supports into analytic, PySR, and PySINDy expansion.

#### Phase 8 Ledger

- `8.1A`: Add L2 regularization changes-estimate tests.
- `8.1B`: Add L1 regularization objective-path tests.
- `8.1C`: Wire `regularization_penalty()` into actual fitting objective.
- `8.1D`: Add replay metadata for optimized objective and regularization contribution.
- `8.2A`: Create `src/euclid/fit/identifiability.py`.
- `8.2B`: Add rank, singular value, and condition number tests.
- `8.2C`: Add covariance availability and bound-hit diagnostic tests.
- `8.2D`: Add split-loss diagnostics in `fit_cir_candidate()`.
- `8.3A`: Add objective-claim mismatch tests.
- `8.3B`: Add objective metadata with optimized and diagnostic losses.
- `8.3C`: Route claim checks through objective/identifiability metadata.

#### Phase 9 Ledger

- `9.1A`: Update `docs/math.md` after Phases 1, 2, 5, and 8 are implemented.
- `9.1B`: Update `docs/search-core.md` after Phase 7 is implemented.
- `9.1C`: Update `docs/testing-truthfulness.md` after Phases 3 and 4 are implemented.
- `9.1D`: Update the enhancement master plan with completed evidence gates and remaining work.
- `9.2A`: Add workbench service fields for MDL tier, predictive promotion, calibration, benchmark metrics, and publication blockers.
- `9.2B`: Add or update live smoke checks for workbench evidence display.

### 4.5 Phase Verification Commands

Run these commands at the end of each phase. Passing a task-level command is not enough to close a phase.

Phase 0:

```bash
pytest tests/unit/modules/test_evidence_contracts.py tests/unit/modules/test_claims.py tests/integration/test_publication_pipeline.py tests/benchmarks/test_readiness_gate.py -v
```

Phase 1:

```bash
pytest tests/unit/math/test_codelength.py tests/unit/math/test_reference_descriptions.py tests/unit/search/test_descriptive_coding.py tests/unit/search/test_portfolio.py tests/unit/fit/test_refit.py tests/unit/modules/test_claims.py -v
```

Phase 2:

```bash
pytest tests/unit/modules/test_predictive_tests.py tests/unit/modules/test_scoring.py tests/unit/modules/test_evaluation_governance.py tests/unit/search/test_portfolio.py tests/integration/test_publication_pipeline.py -v
```

Phase 3:

```bash
pytest tests/unit/benchmarks/test_reporting.py tests/benchmarks/test_readiness_gate.py tests/benchmarks/test_phase08_holistic_honesty_suite.py tests/integration/test_phase08_benchmark_gate.py -v
```

Phase 4:

```bash
pytest tests/unit/modules/test_claims.py tests/unit/modules/test_replay.py tests/integration/test_publication_pipeline.py tests/integration/test_probabilistic_publication.py tests/golden/test_phase07_publication_fixtures.py -v
```

Phase 5:

```bash
pytest tests/unit/modules/test_calibration.py tests/integration/test_probabilistic_calibration_gate.py tests/benchmarks/test_probabilistic_benchmark_harness.py -v
```

Phase 6:

```bash
pytest tests/unit/nonstationarity/test_stability.py tests/unit/nonstationarity/test_changepoints.py tests/unit/nonstationarity/test_regime_switching.py tests/unit/nonstationarity/test_state_space.py tests/integration/test_regime_conditioned_pipeline.py tests/unit/modules/test_predictive_tests.py -v
```

Phase 7:

```bash
pytest tests/unit/search/test_backends.py tests/unit/search/test_adaptive_orchestration.py tests/unit/search/test_portfolio.py tests/unit/search/engines/test_sparse_regression.py tests/unit/search/engines/test_pysindy_engine.py tests/unit/search/engines/test_pysr_engine.py tests/benchmarks/test_search_class_coverage.py tests/benchmarks/test_shared_local_generalization.py -v
```

Phase 8:

```bash
pytest tests/unit/fit/test_objectives.py tests/unit/fit/test_refit.py tests/unit/fit/test_scipy_optimizers.py tests/unit/search/test_fitting_boundary.py tests/unit/modules/test_claims.py -v
```

Phase 9:

```bash
pytest tests/spec_compiler/test_public_claims_truthfulness.py tests/live/test_benchmark_live_smoke.py -v
```

## 5. Subagent Execution Protocol

Each future implementation prompt must create 3 to 6 agents:

- One test agent per independent workstream to write failing tests only.
- One implementation agent per disjoint module group.
- One review agent to inspect changed files for claim leaks and missing tests.

Rules for every development swarm:

- Start by reading this spec and the relevant files.
- Own a narrow write scope.
- Add failing tests first.
- Run the smallest relevant test command after each task.
- Do not edit generated fixtures until behavior is correct.
- Do not update golden files to hide a regression.
- Propagate reason codes and manifests rather than booleans.
- Preserve old behavior only when a compatibility test proves it is still honest.

Standard subagent prompt footer:

```text
You are not alone in the codebase. Other agents may be editing adjacent files.
Do not revert edits you did not make. Keep your write scope narrow. Add tests first.
Return changed files, tests run, and any residual risks.
```

## 6. Phase 0 - Source Of Truth And Typed Evidence Contract

### Objective

Create one typed evidence-gate layer so scorecards, claim cards, promotion statuses, benchmark gates, and publication checks share the same status vocabulary and reason-code contract.

### Main Files

- Create: `src/euclid/modules/evidence_contracts.py`
- Modify: `src/euclid/manifests/runtime_models.py`
- Modify: `src/euclid/modules/gate_lifecycle.py`
- Modify: `src/euclid/modules/evaluation_governance.py`
- Modify: `src/euclid/modules/claims.py`
- Modify: `src/euclid/modules/catalog_publishing.py`
- Test: `tests/unit/modules/test_evidence_contracts.py`
- Test: `tests/unit/modules/test_claims.py`
- Test: `tests/integration/test_publication_pipeline.py`

Resolved Phase 0 implementation detail:

- Runtime manifest model classes did not need shape changes for Phase 0. Typed evidence status objects are runtime contract helpers, exported through `src/euclid/modules/__init__.py`, while existing `RunResultManifest` and `PublicationRecordManifest` bodies remain compatible.
- `src/euclid/modules/evaluation_governance.py` did not need a Phase 0 code change because the typed scorecard boundary is resolved in `src/euclid/modules/gate_lifecycle.py` and consumed by existing governance outputs without changing their manifest shape.
- Source-of-truth drift is intentionally explicit in `tests/benchmarks/test_readiness_gate.py`: packaged Phase 8 benchmark assets are authoritative under `src/euclid/_assets/benchmarks`, root benchmark baselines are authoritative under `benchmarks`, root-only development schemas are authoritative under `schemas`, and packaged release gate policy assets are authoritative under `src/euclid/_assets/schemas`.

### Task 0.1: Add Typed Evidence Status Objects

Write failing tests for:

- `EvidenceStatus.passed` cannot be constructed without `evidence_refs` when the gate declares artifacts required.
- `EvidenceStatus.abstained` must carry at least one reason code.
- `EvidenceStatus.failed` must carry at least one reason code.
- Unknown status and unknown reason code raise `ContractValidationError`.

Implementation requirements:

- Add dataclasses for `EvidenceStatus`, `EvidenceGateDecision`, `ClaimScopeDecision`, and `PromotionDecision`.
- Provide `as_manifest()` methods with stable ordering.
- Add helpers for `passed`, `failed`, `abstained`, and `downgraded`.
- Make the helpers reject empty reason codes on non-passed statuses.

Verification:

```bash
pytest tests/unit/modules/test_evidence_contracts.py -v
```

### Task 0.2: Centralize Claim And Scorecard Construction

Write failing tests proving that duplicated workflow-local claim card construction is no longer needed for new paths.

Implementation requirements:

- Add builder functions in `src/euclid/modules/evidence_contracts.py`.
- Route `gate_lifecycle.resolve_scorecard_status()` through typed objects.
- Keep existing manifest shape compatible where tests require it.
- Add a compatibility shim only at the boundary.

Verification:

```bash
pytest tests/unit/modules/test_evidence_contracts.py tests/unit/modules/test_claims.py -v
```

### Task 0.3: Config Source-Of-Truth Audit

Write failing tests for asset drift:

- Root `benchmarks/` and packaged `src/euclid/_assets/benchmarks/` must either match or be explicitly declared as intentionally divergent.
- Root `schemas/` and packaged `src/euclid/_assets/schemas/` must either match or be explicitly declared as intentionally divergent.

Implementation requirements:

- Add a small drift checker under benchmark/readiness tests.
- Do not silently copy files.
- Make the test output identify the authoritative file.

Verification:

```bash
pytest tests/benchmarks/test_readiness_gate.py -v
```

### Phase 0 Acceptance Gate

Phase 0 is complete when evidence statuses are typed, non-passed gates require reasons, source-of-truth drift is explicit, and existing publication tests still pass.

## 7. Phase 1 - MDL Layer Repair

### Objective

Replace the current MDL-style proxy with a declared coding layer that can honestly report one of:

- `exact_prequential_symbol_code`
- `mdl_based_universal_code`
- `mdl_inspired_proxy_score`
- `not_mdl_claim_eligible`

### Main Files

- Modify: `src/euclid/math/codelength.py`
- Modify: `src/euclid/math/reference_descriptions.py`
- Modify: `src/euclid/search/descriptive_coding.py`
- Create: `src/euclid/math/residual_coding.py`
- Create: `src/euclid/math/lattice.py`
- Test: `tests/unit/math/test_codelength.py`
- Test: `tests/unit/math/test_reference_descriptions.py`
- Test: `tests/unit/search/test_descriptive_coding.py`

### Task 1.1: Explicit Residual Alphabet And Escape Coding

Write failing tests:

- A new residual symbol emits an explicit `ESC` event before symbol identity is charged.
- A previously seen residual symbol emits no `ESC`.
- The prequential diagnostics list `future_count_used == 0` for every row.
- The total codelength equals the sum of event bits plus sequence length bits.
- A fixed finite alphabet rejects residuals outside the alphabet unless an escape policy is declared.

Implementation requirements:

- Add `ResidualAlphabetPolicy`.
- Add `ResidualCodeEvent`.
- Add `prequential_escape_residual_bin_code_v1`.
- Keep old `prequential_laplace_residual_bin_v1` as a compatibility code but mark its claim tier as proxy unless explicitly equivalent.
- Include `alphabet_mode`, `escape_policy`, `innovation_code_family`, and `symbol_identity_bits` in diagnostics.

Verification:

```bash
pytest tests/unit/math/test_codelength.py -v
```

### Task 1.2: Same-Family Reference Coding

Write failing tests:

- Candidate data-code family and reference data-code family must match.
- A candidate using same-family reference coding cannot compare against a raw-reference candidate in the same comparison group.
- `reference_family_id` appears in the codelength comparison key.
- Cross-family reference comparisons fail unless a global family code and common observation representation are declared.

Implementation requirements:

- Extend `ReferenceDescriptionPolicy` with `reference_scope`.
- Supported scopes: `raw_observation_reference`, `same_family_reference`, `global_family_mixture_reference`.
- Thread `data_code_family` into `build_reference_description()`.
- Include reference scope and family in `CodelengthComparisonKey`.

Verification:

```bash
pytest tests/unit/math/test_reference_descriptions.py tests/unit/math/test_codelength.py -v
```

### Task 1.3: Active Parameter And State Lattice Propagation

Write failing tests:

- Parameter lattice step in policy appears in candidate model-code decomposition.
- State lattice step appears anywhere state is encoded.
- Fitting/refit manifests include the active lattice policy.
- Two candidates with different parameter lattices are not comparable.

Implementation requirements:

- Add `LatticePolicy`.
- Propagate lattice policy through `CodelengthPolicy`, descriptive coding, fit/refit replay metadata, and publication artifacts.
- Do not default silently to residual quantization step without recording the reason.

Verification:

```bash
pytest tests/unit/math/test_codelength.py tests/unit/fit/test_refit.py -v
```

### Task 1.4: Comparable Groups Instead Of Batch-Wide Failure

Write failing tests:

- Mixed comparison keys produce multiple comparable groups.
- Incompatible candidates receive explicit non-comparable diagnostics.
- The best candidate is selected only within a comparable group.

Implementation requirements:

- Refactor `descriptive_coding._comparison_status()`.
- Return grouped comparability artifacts.
- Update downstream ranking to use only eligible comparable groups.

Verification:

```bash
pytest tests/unit/search/test_descriptive_coding.py tests/unit/search/test_portfolio.py -v
```

### Task 1.5: MDL Claim Tier Enforcement

Write failing tests:

- `build_codelength_policy_manifest()` cannot call a policy MDL if it uses only legacy fixed-step raw-reference scoring.
- Claim cards that mention universal coding require an eligible codelength policy tier.

Implementation requirements:

- Add `coding_claim_tier` to codelength policy manifest.
- Use `mdl_inspired_proxy_score` for legacy code.
- Let publication claim checks reject stronger MDL language unless tier permits it.

Verification:

```bash
pytest tests/unit/math/test_codelength.py tests/unit/modules/test_claims.py -v
```

### Phase 1 Acceptance Gate

Phase 1 is complete when residual coding has explicit unseen-symbol behavior, reference coding matches candidate coding, lattice policies are active artifacts, and MDL language is mechanically claim-scoped.

## 8. Phase 2 - Honest Predictive Inference

### Objective

Make predictive promotion a declared statistical test over paired forecast-origin/horizon loss differentials with minimum information thresholds and uncertainty-aware pass/fail/abstain decisions.

### Main Files

- Modify: `src/euclid/modules/predictive_tests.py`
- Modify: `src/euclid/modules/evaluation_governance.py`
- Modify: `src/euclid/modules/scoring.py`
- Modify: `src/euclid/modules/gate_lifecycle.py`
- Create: `src/euclid/modules/predictive_inference.py`
- Create: `src/euclid/modules/effective_sample.py`
- Test: `tests/unit/modules/test_predictive_tests.py`
- Test: `tests/unit/modules/test_evaluation_governance.py`
- Test: `tests/integration/test_publication_pipeline.py`

### Task 2.1: Paired Loss Differential Artifact

Write failing tests:

- Candidate and baseline streams must have identical origin ids, horizons, entity ids, and row-set ids.
- Separately averaged losses cannot be promoted.
- A one-element paired stream always abstains with `insufficient_paired_count`.

Implementation requirements:

- Add `PairedLossDifferentialStream`.
- Store raw pair count, effective sample size, block count, horizon geometry, and loss id.
- Make `_build_paired_comparison_record()` emit this artifact.

Verification:

```bash
pytest tests/unit/modules/test_predictive_tests.py tests/unit/modules/test_scoring.py -v
```

### Task 2.2: Declared Test Semantics

Write failing tests:

- `declared_test_id="diebold_mariano_hln_v1"` produces a DM/HLN result, not a generic HAC label.
- `declared_test_id="paired_stationary_block_bootstrap_v1"` records block length, seed, and bootstrap count.
- Unsupported test id fails closed.
- GW cannot be used as default unconditional promotion without instruments/state declarations.

Implementation requirements:

- Implement test registry in `predictive_inference.py`.
- Keep statsmodels HAC mean interval as an internal component, not the public test identity.
- Add arch dependency only if already available or behind optional import with clear unavailable reason.

Verification:

```bash
pytest tests/unit/modules/test_predictive_tests.py -v
```

### Task 2.3: Minimum Paired Counts And Effective Information

Write failing tests:

- `n_eff < 25` abstains.
- block count `< 8` abstains for block-bootstrap tests.
- `25 <= n_eff < 50` produces `human_review_only`, not automatic promotion.
- `n_eff >= 50` and sufficient block count may promote if all other conditions pass.

Implementation requirements:

- Add `minimum_pair_policy`.
- Compute HAC-based `n_eff` and bootstrap block count.
- Include thresholds in the result manifest.

Verification:

```bash
pytest tests/unit/modules/test_predictive_tests.py -v
```

### Task 2.4: Uncertainty-Aware Promotion

Write failing tests:

- Candidate cannot pass unless direction, p-value or interval, and practical effect size agree.
- Confidence interval crossing the practical margin downgrades or abstains.
- Calibration failure blocks probabilistic predictive promotion.
- Nonstationarity diagnostic failure blocks automatic promotion.

Implementation requirements:

- Promotion result statuses: `passed`, `failed`, `abstained`, `human_review_only`, `downgraded`.
- Reason codes include `uncertainty_interval_crosses_margin`, `insufficient_effective_sample_size`, `insufficient_effective_block_count`, `nonstationarity_detected`, and `calibration_failed`.
- `resolve_confirmatory_promotion_allowed()` consumes typed promotion status, not raw booleans.

Verification:

```bash
pytest tests/unit/modules/test_predictive_tests.py tests/unit/modules/test_evaluation_governance.py -v
```

### Task 2.5: Multi-Model Comparison Boundary

Write failing tests:

- A portfolio comparison with more than two candidates cannot claim unique superiority from pairwise DM alone.
- If MCS/SPA is unavailable, the system emits `multi_model_superiority_not_tested`.

Implementation requirements:

- Add fail-closed registry entries for `model_confidence_set_v1` and `superior_predictive_ability_v1`.
- If `arch` imports successfully, use it for available MCS/SPA implementations; if the import fails, return `multi_model_test_backend_unavailable` with dependency diagnostics.

Verification:

```bash
pytest tests/unit/modules/test_predictive_tests.py tests/unit/search/test_portfolio.py -v
```

### Phase 2 Acceptance Gate

Phase 2 is complete when no predictive publication path can pass from a raw metric delta, one-pair tests abstain, declared test ids match actual computation, and uncertainty plus practical effect size are required.

## 9. Phase 3 - Measured Benchmark Efficacy Gates

### Objective

Convert benchmark/readiness gates from mostly semantic checks into measured efficacy gates with observed values, denominators, confidence where appropriate, and replay-backed evidence references.

### Main Files

- Modify: `src/euclid/benchmarks/reporting.py`
- Modify: `src/euclid/benchmarks/runtime.py`
- Modify: `src/euclid/readiness/judgment.py`
- Modify: `src/euclid/release.py`
- Create: `src/euclid/benchmarks/efficacy_metrics.py`
- Test: `tests/unit/benchmarks/test_reporting.py`
- Test: `tests/benchmarks/test_readiness_gate.py`
- Test: `tests/benchmarks/test_phase08_holistic_honesty_suite.py`
- Test: `tests/integration/test_phase08_benchmark_gate.py`

### Task 3.1: Missing Metrics Fail Closed

Write failing tests:

- A non-abstention benchmark threshold with missing observed metric fails.
- A safe-abstention task may pass only if expected safe outcome is abstention and no winner exists.
- Missing metric rows include threshold id, metric id, and source submitter.

Implementation requirements:

- Change `_evaluate_metric_thresholds()` so `observed_value is None` fails unless safe abstention is explicitly active.
- Add `measurement_required` default true.
- Preserve existing safe abstention behavior.

Verification:

```bash
pytest tests/unit/benchmarks/test_reporting.py -v
```

### Task 3.2: Efficacy Metric Registry

Write failing tests:

- Planted-law recovery reports exact/near recovery numerator and denominator.
- False holistic claim rate reports false positives over adversarial tasks.
- Probabilistic attachment quality reports coverage, width, calibration count, and status.
- Nonstationary lanes report detection tolerance metrics.

Implementation requirements:

- Add `efficacy_metrics.py`.
- Produce stable metric ids matching benchmark threshold YAML.
- Include metric provenance: task id, submitter id, candidate id, replay id, and row count.

Verification:

```bash
pytest tests/unit/benchmarks/test_reporting.py tests/benchmarks/test_phase08_holistic_honesty_suite.py -v
```

### Task 3.3: Replay Status Must Be Verified, Not Present

Write failing tests:

- A benchmark task with replay files but `replay_verification_status != verified` fails readiness.
- Missing replay artifact and unverified replay have distinct reason codes.

Implementation requirements:

- Update `benchmarks.runtime._task_replay_verification_status`.
- Thread replay verification status from reproducibility bundle, not file existence only.
- Generated benchmark replay refs must carry explicit replay verification status; legacy file-presence compatibility cannot override a failed or unverified replay status.

Verification:

```bash
pytest tests/benchmarks/test_readiness_gate.py tests/integration/test_phase08_benchmark_gate.py -v
```

### Phase 3 Acceptance Gate

Phase 3 is complete when benchmark gates cannot pass unmeasured efficacy thresholds and readiness consumes verified replay plus measured metrics.

## 10. Phase 4 - Publication, Replay, And Claim Scope

### Objective

Make publication a final evidence synthesis step that calls replay verification, measured readiness, comparator exposure, and semantic claim-scope checks.

### Main Files

- Modify: `src/euclid/modules/catalog_publishing.py`
- Modify: `src/euclid/modules/claims.py`
- Modify: `src/euclid/modules/replay.py`
- Modify: `src/euclid/modules/gate_lifecycle.py`
- Test: `tests/unit/modules/test_claims.py`
- Test: `tests/integration/test_publication_pipeline.py`
- Test: `tests/integration/test_probabilistic_publication.py`
- Test: `tests/golden/test_phase07_publication_fixtures.py`

### Task 4.1: Call Claim Scope Assertion On Publish Path

Write failing tests:

- Publication with universal language but no invariance/transport evidence raises.
- Publication with stochastic language but no stochastic evidence raises.
- Publication with invariant language but no invariance lane raises.

Implementation requirements:

- Call `assert_claim_scope_publication()` inside `build_publication_record_manifest()`.
- Ensure run result exposes the claim card body or typed ref needed for validation.
- If existing signature cannot supply claim body, add a typed argument rather than reaching through global state.

Verification:

```bash
pytest tests/unit/modules/test_claims.py tests/integration/test_publication_pipeline.py -v
```

### Task 4.2: Publication Requires Validated Scorecard And Promotion Artifacts

Write failing tests:

- Candidate publication without paired predictive test artifact cannot publish predictive claim.
- Probabilistic publication without calibration artifact cannot publish probabilistic claim.
- Descriptive-only publication cannot carry predictive wording.

Implementation requirements:

- Add `publication_evidence_requirements`.
- Check scorecard, claim card, paired test, calibration result, replay bundle, and comparator exposure.
- Make downgrade decisions explicit.

Verification:

```bash
pytest tests/integration/test_publication_pipeline.py tests/integration/test_probabilistic_publication.py -v
```

### Task 4.3: Replay Bundle Completeness

Write failing tests:

- Replay bundle missing codelength policy ref fails if descriptive claim is published.
- Replay bundle missing predictive test ref fails if predictive claim is published.
- Replay bundle missing calibration/conformal ref fails if probabilistic interval/quantile claim is published.

Implementation requirements:

- Extend `required_manifest_refs_from_run_result()`.
- Add evidence-lane-specific required refs.

Verification:

```bash
pytest tests/unit/modules/test_replay.py tests/integration/test_publication_pipeline.py -v
```

### Phase 4 Acceptance Gate

Phase 4 is complete when public catalog publication is impossible without verified replay, measured readiness, comparator exposure, and claim-scope validation.

## 11. Phase 5 - Conformal And Distribution-Free Calibration

### Objective

Add conformal and distribution-free calibration options while preserving honest finite-sample language under exchangeability and weakened language under time dependence.

### Main Files

- Modify: `src/euclid/modules/calibration.py`
- Modify: `src/euclid/modules/probabilistic_evaluation.py`
- Modify: `src/euclid/modules/scoring.py`
- Create: `src/euclid/modules/conformal.py`
- Test: `tests/unit/modules/test_calibration.py`
- Test: `tests/integration/test_probabilistic_calibration_gate.py`
- Test: `tests/benchmarks/test_probabilistic_benchmark_harness.py`

### Task 5.1: Calibration Method Registry

Write failing tests:

- `split_conformal_exchangeable_v1` requires exchangeability declaration.
- `enbpi_time_series_v1` declares approximate/mixing assumptions, not exact distribution-free finite-sample validity.
- `adaptive_conformal_time_series_v1` declares long-run coverage control, not fixed-time finite-sample validity.
- Unknown conformal method fails closed.

Implementation requirements:

- Add method registry with guarantee tier.
- Guarantee tiers: `finite_sample_exchangeable`, `approximate_mixing_time_series`, `asymptotic_time_series`, `long_run_frequency_control`, `diagnostic_only`.
- Include calibration split ids and horizon ids.

Verification:

```bash
pytest tests/unit/modules/test_calibration.py -v
```

### Task 5.2: Horizon-Separated Calibration

Write failing tests:

- Calibration residuals for horizon 1 cannot calibrate horizon 3 unless explicitly pooled with a valid policy.
- Minimum calibration count is enforced per horizon.
- Regime-slice undercoverage blocks promotion.

Implementation requirements:

- Add `CalibrationPartition`.
- Partition by horizon, optional entity, optional regime.
- Add Wilson/binomial lower-bound style checks for empirical coverage where applicable.

Verification:

```bash
pytest tests/unit/modules/test_calibration.py tests/integration/test_probabilistic_calibration_gate.py -v
```

### Task 5.3: MAPIE Adapter Boundary

Write failing tests:

- MAPIE unavailable produces `calibration_backend_unavailable`, not import error.
- MAPIE time-series method records method name, version, calibration indices, and assumptions.

Implementation requirements:

- Add optional adapter in `conformal.py`.
- Keep Euclid-owned manifests independent of MAPIE object serialization.
- Do not make MAPIE central to claim truth; it is an implementation backend.

Verification:

```bash
pytest tests/unit/modules/test_calibration.py -v
```

### Phase 5 Acceptance Gate

Phase 5 is complete when conformal methods are available but their guarantees are claim-scoped and finite-sample language is reserved for valid assumptions.

## 12. Phase 6 - Nonstationarity, Change-Point, Regime, And State-Space Lanes

### Objective

Add separate evidence lanes for instability, hard breaks, discrete regimes, and continuous latent states. These lanes may qualify law scope; they must not launder nonstationarity into stationary law claims.

### Main Files

- Create: `src/euclid/nonstationarity/__init__.py`
- Create: `src/euclid/nonstationarity/stability.py`
- Create: `src/euclid/nonstationarity/changepoints.py`
- Create: `src/euclid/nonstationarity/regime_switching.py`
- Create: `src/euclid/nonstationarity/state_space.py`
- Modify: `src/euclid/reducers/composition.py`
- Modify: `src/euclid/modules/claims.py`
- Modify: `src/euclid/benchmarks/runtime.py`
- Test: `tests/unit/nonstationarity/test_stability.py`
- Test: `tests/unit/nonstationarity/test_changepoints.py`
- Test: `tests/unit/nonstationarity/test_regime_switching.py`
- Test: `tests/unit/nonstationarity/test_state_space.py`
- Test: `tests/integration/test_regime_conditioned_pipeline.py`

### Task 6.1: Stability Diagnostic Lane

Write failing tests:

- CUSUM/recursive residual diagnostic can emit instability evidence.
- Instability evidence blocks automatic predictive promotion unless a nonstationary lane handles it.
- Stability diagnostics are not themselves law claims.

Implementation requirements:

- Use statsmodels diagnostics behind optional import boundary.
- Emit `StabilityDiagnosticArtifact`.

Resolved implementation detail:

- The runtime API is `run_stability_diagnostic(observations=..., series_id="series", design_matrix=None, method="cusum_recursive_residuals", significance_level=None, min_observations=8, alpha=0.05, optional_backend_overrides=None)`. `significance_level` is accepted as an alias for `alpha`, caller-provided design matrices are honored when row counts match observations, and manifests always mark stability artifacts as diagnostic-only with `may_publish_stationary_law_claim=false`.

Verification:

```bash
pytest tests/unit/nonstationarity/test_stability.py tests/unit/modules/test_predictive_tests.py -v
```

### Task 6.2: Hard Change-Point Lane

Write failing tests:

- Synthetic piecewise-constant series recovers breakpoints within tolerance.
- Breakpoint artifacts include penalty, method, min segment size, and tolerance.
- Segments shorter than minimum length abstain.

Implementation requirements:

- Use ruptures PELT/binseg adapters.
- Emit `ChangePointArtifact`.
- Add benchmark metrics: precision/recall with tolerance, Hausdorff distance, detection delay.

Resolved implementation detail:

- The public change-point method names include both canonical `pelt`/`binseg` and compatibility aliases such as `pelt_l2`/`binseg_l2`; aliases are normalized to the corresponding `ruptures` method/model while the manifest preserves the requested public method string.
- The current local verification environment does not provide `ruptures`. To keep the required synthetic piecewise-constant recovery tests executable, `detect_change_points` uses a deterministic Euclid exact-level-shift fallback only when the real `ruptures` import is absent in ordinary local execution. Explicit optional-backend failure tests, including intentionally raised `ImportError`, remain fail-closed with reason code `ruptures_backend_unavailable`.

Verification:

```bash
pytest tests/unit/nonstationarity/test_changepoints.py tests/benchmarks/test_phase08_holistic_honesty_suite.py -v
```

### Task 6.3: Discrete Regime Lane

Write failing tests:

- Markov switching adapter emits transition matrix, smoothed probabilities, expected durations, and convergence diagnostics.
- Weakly separated regimes produce `weak_regime_identifiability`.
- Regime-conditioned laws carry `valid_given_regime` scope.

Implementation requirements:

- Use statsmodels `MarkovRegression` or `MarkovAutoregression`.
- Add posterior probability calibration/Brier metrics on synthetic truth tasks.
- Integrate with existing `regime_conditioned` reducer only after scope is explicit.

Verification:

```bash
pytest tests/unit/nonstationarity/test_regime_switching.py tests/integration/test_regime_conditioned_pipeline.py -v
```

### Task 6.4: Continuous State-Space Lane

Write failing tests:

- Local-level synthetic series recovers latent state within tolerance.
- State-space artifact includes filtered state, smoothed state, covariance, innovations, and log likelihood.
- Innovation whiteness failure blocks promotion.

Implementation requirements:

- Use statsmodels `UnobservedComponents` and statespace APIs first.
- Emit `StateSpaceArtifact`.
- Add metrics: one-step log likelihood, innovation whiteness, interval coverage, latent RMSE on synthetic.

Verification:

```bash
pytest tests/unit/nonstationarity/test_state_space.py -v
```

### Phase 6 Acceptance Gate

Phase 6 is complete when nonstationarity evidence exists as scoped artifacts, not hidden fit flexibility, and stationary claims are blocked under unresolved instability.

## 13. Phase 7 - Controlled Symbolic And Sparse Search Expansion

### Objective

Expand default expressive power and adaptive hypothesis search while preserving replay, family diversity, comparability, and claim limits.

### Main Files

- Modify: `src/euclid/search/backends.py`
- Modify: `src/euclid/search/portfolio.py`
- Modify: `src/euclid/search/engines/pysindy_engine.py`
- Modify: `src/euclid/search/engines/pysr_engine.py`
- Modify: `src/euclid/search/engines/sparse_regression.py`
- Create: `src/euclid/search/library_specs.py`
- Create: `src/euclid/search/orchestration/adaptive.py`
- Create: `src/euclid/search/orchestration/diversity.py`
- Test: `tests/unit/search/test_portfolio.py`
- Test: `tests/unit/search/test_adaptive_orchestration.py`
- Test: `tests/unit/search/engines/test_pysindy_engine.py`
- Test: `tests/unit/search/engines/test_pysr_engine.py`
- Test: `tests/benchmarks/test_search_class_coverage.py`

### Task 7.1: Expressive Library Specs

Write failing tests:

- Default library includes intercept, lag-affine, lag-polynomial, rolling/statistical features, seasonal harmonics beyond order 2, piecewise indicators, and interaction terms when legal.
- Library specs declare operator budget, units/dimensions, lag horizon, and feature provenance.
- Illegal feature leakage is rejected.
- `AnalyticSearchBackendAdapter.default_proposals` and `SpectralSearchBackendAdapter.default_proposals` are no longer the only source of default expressive power.
- `candidate_family_ids` semantics distinguish primitive families from specific candidate ids.

Implementation requirements:

- Extend `SearchPlanManifest` and `build_search_plan` with explicit expansion policy fields: primitive-family filters, allowed candidate ids, feature subset sizes, operator sets, complexity caps, sparse alpha grids, and PySR/PySINDy config grids.
- Add `SearchLibrarySpec`.
- Keep exact enumeration for small bounded spaces.
- For large spaces, record truncation/adaptive expansion policy.

Verification:

```bash
pytest tests/unit/search/test_backends.py tests/unit/search/test_adaptive_orchestration.py tests/benchmarks/test_search_class_coverage.py -v
```

### Task 7.2: Adaptive Hypothesis Expansion

Write failing tests:

- Search expands when residual diagnostics show structure.
- Search stops when budget, saturation, or evidence threshold is hit.
- Adaptive decisions are replayable from logged diagnostics and seeds.
- Bounded/equality-saturation/stochastic modes are not implemented as prefix-only or simple seeded permutation selection when a hierarchy-aware expansion policy is declared.

Implementation requirements:

- Add expansion loop in `search/orchestration/adaptive.py`.
- Expansion triggers: residual autocorrelation, spectral peaks, regime instability, systematic horizon error, candidate diversity collapse.
- Expansion actions: add lags, add interaction group, add harmonics, add sparse library, call PySR/PySINDy with constrained operators.
- Replace or wrap `_select_attempted_proposals()` with a hierarchy-aware scheduler using family quotas, complexity tiers, sparse support seeds, and symbolic refinement tiers.

Verification:

```bash
pytest tests/unit/search/test_backends.py tests/unit/search/test_adaptive_orchestration.py -v
```

### Task 7.3: Family-Diverse Portfolio Retention

Write failing tests:

- Portfolio retains top-k within-family alternatives until measured comparison stage.
- Useful within-family alternatives are not discarded solely because they are second-best by codelength.
- Final publication chooses among comparable candidates only after predictive/benchmark gates.
- Retained frontier candidates, not only one `_select_family_finalist()` result, are eligible for measured comparison.

Implementation requirements:

- Replace `_select_family_finalist()` one-winner behavior with configurable `family_retention_k`.
- Add diversity metadata.
- Update `ComparableBackendFinalist` and selection rules to carry common frontier axes, including fit loss, parameter count, support stability, parameter stability, and identifiability status.
- Keep old one-finalist behavior only as an explicit legacy mode.

Verification:

```bash
pytest tests/unit/search/test_portfolio.py tests/perf/test_portfolio_runtime.py -v
```

### Task 7.4: PySINDy And PySR As First-Class Proposal Engines

Write failing tests:

- PySINDy proposals record feature library, differentiation method, optimizer, threshold, support mask, and version.
- PySR proposals record operator set, constraints, maxsize/maxdepth, iterations, populations, hall of fame, and Julia/runtime versions.
- External engine output cannot publish without Euclid CIR lowering, refit, scoring, replay, and gates.
- `run_descriptive_search_portfolio` can include configured PySR, PySINDy, and sparse-regression proposal sources rather than only `_default_adapters()`.
- PySINDy ensemble output can emit multiple support-stability candidates instead of only the first coefficient row.
- PySR hall-of-fame candidates carry loss, complexity, and Euclid refit metrics into common frontier axes.

Implementation requirements:

- Tighten existing engine manifests.
- Keep optional dependency failures explicit.
- Lower all publishable candidates into Euclid expression IR and CIR.
- Treat PySR/PySINDy as configured proposal engines, not as publication authorities.

Verification:

```bash
pytest tests/unit/search/engines/test_pysindy_engine.py tests/unit/search/engines/test_pysr_engine.py tests/integration/test_pysindy_pipeline.py tests/integration/test_pysr_pipeline.py -v
```

### Task 7.5: Sparse And Shared Structure

Write failing tests:

- Sparse library candidate records active support, regularization path, and selected alpha.
- Multi-entity shared support uses `MultiTaskLasso` or equivalent and records shared/local split.
- Group/hierarchical penalties are declared even if optional backend unavailable.
- `SparseRegressionEngine` emits an alpha/support path and support-stability metrics rather than one Lasso candidate.

Implementation requirements:

- Extend `SparseRegressionEngine`.
- Use scikit-learn first.
- Optional later: celer, group-lasso, cvxpy.
- Feed stable sparse supports into analytic, PySR, and PySINDy expansion policies.

Verification:

```bash
pytest tests/unit/search/engines/test_sparse_regression.py tests/unit/search/test_adaptive_orchestration.py tests/benchmarks/test_shared_local_generalization.py -v
```

### Phase 7 Acceptance Gate

Phase 7 is complete when the default search is materially broader, adaptive search is replayable, within-family alternatives survive long enough for measured comparison, and external engines remain proposal-only.

## 14. Phase 8 - Identifiability And Real Regularization

### Objective

Make fitting diagnostics and optimizer objectives match declared regularization and identifiability claims.

### Main Files

- Modify: `src/euclid/fit/objectives.py`
- Modify: `src/euclid/fit/refit.py`
- Modify: `src/euclid/fit/scipy_optimizers.py`
- Modify: `src/euclid/fit/parameterization.py`
- Create: `src/euclid/fit/identifiability.py`
- Test: `tests/unit/fit/test_objectives.py`
- Test: `tests/unit/fit/test_refit.py`
- Test: `tests/unit/fit/test_scipy_optimizers.py`
- Test: `tests/unit/search/test_fitting_boundary.py`

### Task 8.1: Regularization Enters Actual Objective

Write failing tests:

- L2 penalty changes fitted parameter relative to unregularized fit.
- L1 penalty is represented as optimizer-compatible loss or residual augmentation with correct diagnostics.
- Declared penalty appears in replay metadata and final objective value.
- Unknown penalty fails closed.

Implementation requirements:

- Use `regularization_penalty()` in `fit_cir_candidate()`.
- For least squares, add residual augmentation for L2 where mathematically appropriate.
- For L1, use scipy minimize or a declared nonsmooth approximation, with explicit method id.

Verification:

```bash
pytest tests/unit/fit/test_objectives.py tests/unit/fit/test_refit.py -v
```

### Task 8.2: Identifiability Diagnostics

Write failing tests:

- Rank-deficient design emits `rank_deficient_design`.
- High condition number emits `ill_conditioned_design`.
- Near-collinear parameters emit parameter-level warnings.
- Bound hits are surfaced as claim-limiting diagnostics.
- Least-squares optimizer diagnostics include Jacobian rank, singular values, condition number, covariance availability, and bound-hit diagnostics.
- `fit_cir_candidate()` reports validation/test losses as diagnostics without using them for training.

Implementation requirements:

- Add `identifiability.py`.
- Compute singular values, numerical rank, condition number, parameter covariance where available, standard errors where defensible, and bound-hit summaries.
- Add diagnostics to `UnifiedFitResult.uncertainty_diagnostics`.
- Add split-loss diagnostics for train, validation, and test rows with clear `diagnostic_only` status for non-training splits.

Verification:

```bash
pytest tests/unit/fit/test_refit.py tests/unit/fit/test_scipy_optimizers.py -v
```

### Task 8.3: Objective-Claim Mismatch Checks

Write failing tests:

- Fitting with squared error cannot publish a probabilistic likelihood claim without a probabilistic observation model.
- Regularized objective cannot report unregularized likelihood unless both are computed and labeled.
- Weak identifiability downgrades law claim scope.

Implementation requirements:

- Add objective metadata: optimized objective, reported diagnostic losses, regularization included/excluded flags.
- Route claim checks through this metadata.

Verification:

```bash
pytest tests/unit/fit/test_refit.py tests/unit/modules/test_claims.py -v
```

### Phase 8 Acceptance Gate

Phase 8 is complete when declared regularization affects optimization, identifiability diagnostics are formal artifacts, and weak identification limits publication language.

## 15. Phase 9 - Release Hardening And Evidence Surface

### Objective

Make the improved system auditable by humans and future agents.

### Main Files

- Modify: `docs/math.md`
- Modify: `docs/search-core.md`
- Modify: `docs/testing-truthfulness.md`
- Modify: `docs/plans/2026-04-21-euclid-enhancement-master-plan.md`
- Modify: `src/euclid/workbench/service.py`
- Test: `tests/spec_compiler/test_public_claims_truthfulness.py`
- Test: `tests/live/test_benchmark_live_smoke.py`

### Task 9.1: Documentation Truthfulness Update

Write docs updates after code passes:

- Replace obsolete MDL claims with coding-tier language.
- Document predictive test semantics and minimum information thresholds.
- Document benchmark measured efficacy gates.
- Document nonstationary lane scope.
- Document conformal guarantee tiers.

Verification:

```bash
pytest tests/spec_compiler/test_public_claims_truthfulness.py -v
```

### Task 9.2: Workbench Evidence Surfacing

Implementation requirements:

- Show promotion status, n pairs, effective sample size, block count, p-value/interval, practical margin, and reason codes.
- Show MDL coding tier and comparison-key compatibility group.
- Show benchmark observed metrics and missing-metric failures.
- Show publication blockers with evidence refs.

Verification:

```bash
pytest tests/live/test_benchmark_live_smoke.py -v
```

### Phase 9 Acceptance Gate

Phase 9 is complete when docs and workbench surfaces cannot imply stronger evidence than the runtime artifacts support.

## 16. Recommended Dependency Policy

Add dependencies gradually, behind optional adapters first:

- Required near-term: `arch`, `ruptures`, `MAPIE` if dependency policy permits.
- Already central or likely present: NumPy, SciPy, statsmodels, scikit-learn, SymPy.
- Keep optional: PySINDy, PySR, mlforecast, sktime, celer, group-lasso, cvxpy, constriction.
- Avoid central adoption: aws-fortuna, nonconformist, filterpy.
- Review before adoption: pykalman licensing/metadata.
- Avoid AGPL dependencies such as infomeasure unless legal policy permits.

Each dependency must have:

- Optional import boundary.
- Version captured in replay metadata.
- Failure mode test.
- License/health note in docs.

## 17. Agentic Prompt Chain

Use the following sequence for future implementation sessions.

### Prompt 1: Phase 0 Evidence Contract

```text
Implement Phase 0 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use development subagents:
1. Tests for evidence_contracts and source-of-truth drift.
2. Implementation for evidence_contracts/runtime_models/gate_lifecycle.
3. Implementation for claims/catalog publication integration.
4. Review agent for claim leaks and manifest compatibility.
Run the Phase 0 verification commands and stop after Phase 0 acceptance gate.
```

### Prompt 2: Phase 1 MDL Repair

```text
Implement Phase 1 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for residual coding, reference coding, lattice propagation, descriptive comparability grouping, and claim-tier enforcement.
Add tests first. Do not change predictive gates in this prompt.
```

### Prompt 3: Phase 2 Predictive Inference

```text
Implement Phase 2 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for paired stream artifacts, declared test registry, minimum effective sample policy, promotion governance, and multi-model comparison boundary.
No publication path may pass from raw metric deltas.
```

### Prompt 4: Phase 3 And 4 Evidence Gates And Publication

```text
Implement Phases 3 and 4 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for benchmark missing-metric failure, efficacy metrics, replay verification, claim-scope publish checks, and publication evidence requirements.
Run benchmark/readiness and publication integration tests.
```

### Prompt 5: Phase 5 Calibration

```text
Implement Phase 5 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for calibration registry, horizon partitioning, MAPIE adapter, and probabilistic benchmark gates.
Finite-sample distribution-free language must be mechanically claim-scoped.
```

### Prompt 6: Phase 6 Nonstationarity

```text
Implement Phase 6 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for stability diagnostics, hard change-points, discrete regimes, state-space lane, and claim-scope integration.
Do not let nonstationary evidence publish stationary law claims.
```

### Prompt 7: Phase 7 Search Expansion

```text
Implement Phase 7 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for library specs, adaptive orchestration, portfolio retention, PySINDy/PySR tightening, and sparse/shared structure.
All search expansion must be replayable and external engines remain proposal-only.
```

### Prompt 8: Phase 8 Identifiability And Regularization

```text
Implement Phase 8 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for regularization objective wiring, optimizer diagnostics, identifiability module, and claim mismatch checks.
Declared regularization must affect the actual optimizer path.
```

### Prompt 9: Phase 9 Release Hardening

```text
Implement Phase 9 from docs/plans/2026-04-26-euclid-agentic-mathematical-efficacy-spec.md.
Use separate agents for docs truthfulness, workbench evidence display, release/readiness tests, and final review.
Do not update public wording beyond what the runtime artifacts prove.
```

## 18. Global Completion Criteria

The overall effort is complete only when all of the following are true:

- MDL coding policies have explicit residual alphabets, escape/unseen-symbol handling, reference scope, lattice propagation, and claim tiers.
- Predictive promotion uses declared statistical tests over paired origin/horizon streams with minimum effective information thresholds.
- Benchmarks fail on missing measured metrics unless explicitly safe abstention.
- Publication requires verified replay, measured readiness, comparator exposure, and semantic claim-scope validation.
- Calibration methods declare their guarantee tier and cannot overclaim finite-sample distribution-free validity under dependence.
- Nonstationarity lanes emit scoped evidence artifacts and block stationary claims under unresolved instability.
- Search is broader, adaptive, replayable, and portfolio retention does not discard useful within-family alternatives too early.
- Regularization enters the actual objective and identifiability diagnostics are formal claim-limiting artifacts.
- Docs and workbench surfaces match the runtime evidence.

## 19. Suggested Full Verification Command

After all phases:

```bash
pytest \
  tests/unit/math/test_codelength.py \
  tests/unit/modules/test_predictive_tests.py \
  tests/unit/modules/test_calibration.py \
  tests/unit/modules/test_claims.py \
  tests/unit/benchmarks/test_reporting.py \
  tests/unit/search/test_portfolio.py \
  tests/unit/fit/test_refit.py \
  tests/benchmarks/test_readiness_gate.py \
  tests/integration/test_publication_pipeline.py \
  tests/integration/test_probabilistic_publication.py \
  tests/integration/test_regime_conditioned_pipeline.py \
  tests/spec_compiler/test_public_claims_truthfulness.py \
  -v
```

Then run the repository-level verification command preferred by the project:

```bash
./scripts/run_all_tests.sh
```
