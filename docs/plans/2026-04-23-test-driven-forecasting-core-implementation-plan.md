# Test-Driven Forecasting Core Implementation Plan

**Date:** 2026-04-23
**Status:** Proposed implementation plan
**Scope:** Residual-history stochastic support, probabilistic scoring and calibration, multi-horizon fitting, MDL comparability, fixtures, docs, and release gates.

## 1. Goal

Replace Euclid's current heuristic probabilistic and one-step forecasting core with a test-driven implementation that:

1. Builds probabilistic support from real, replayable forecast residual histories.
2. Uses the declared observation or residual family consistently for distributions, intervals, quantiles, event probabilities, scoring, and calibration.
3. Fits candidates against the same multi-horizon geometry used for scoring.
4. Strengthens MDL and descriptive coding so codelength comparisons are unit-aware, row-set-aware, and fail closed when incomparable.
5. Preserves Euclid's existing claim discipline: descriptive admission remains separate from predictive publication.

## 2. Non-Negotiable Invariants

- External search or math engines may propose candidates; they cannot publish claims.
- Confirmatory data must not train search, fitting, stochastic models, recalibration, conformal adjustments, prequential MDL, or hyperparameters.
- Probabilistic publication must require explicit stochastic model evidence.
- Legacy Gaussian proxy support may remain readable only as compatibility evidence.
- Replay identity must include residual history, stochastic family, parameters, horizon law, score policy, seed, library versions, and relevant schema versions.
- Calibration gates must be object-type-aware, family-aware, horizon-aware, and sample-size-aware.
- Codelength comparisons must fail closed across different quantizers, reference policies, row sets, horizon geometry, support semantics, residual-history construction, or composition runtime signatures.
- Multi-horizon fitting must use only legal training-origin panels.

## 3. Delivery Strategy

Use a strict test-first sequence:

1. Add failing tests for the desired behavior.
2. Implement the smallest runtime, manifest, or schema change needed to satisfy the tests.
3. Keep compatibility behavior explicitly tested until migration is complete.
4. Add new fixtures in parallel before migrating existing goldens.
5. Update docs after runtime shapes and replay identities stabilize.

Recommended PR sequence:

1. PR 1: Contract scaffolding and legacy characterization tests.
2. PR 2: Residual history capture and validation.
3. PR 3: Residual-backed stochastic model manifests and proxy downgrade.
4. PR 4: Family-aware distribution rows, scoring, and calibration.
5. PR 5: Multi-horizon fitting strategy and rollout objective.
6. PR 6: MDL quantization, reference bank, and comparability keys.
7. PR 7: Fixtures, docs, workbench, and release hardening.
8. PR 8+: Richer primitive forecasting families.

## 4. Phase 0 - Contract And Legacy Safety Net

### 4.1 Tests To Write First

Add `tests/spec_compiler/test_residual_history_contract.py`:

- Validates `residual_history_manifest@1.0.0`.
- Requires row keys for candidate, fit window, origin, horizon, entity, realized value, point forecast, residual, split role, and replay identity.
- Requires residual-history digest fields.
- Rejects missing split-role metadata.
- Rejects rows without origin and target availability metadata.

Add `tests/spec_compiler/test_stochastic_model_contract.py`:

- Validates `stochastic_model_manifest@1.0.0`.
- Requires a residual-history ref for production stochastic evidence.
- Requires observation or residual family, support kind, horizon law, fitted parameters, residual count, and min-count policy.
- Rejects `heuristic_gaussian_support: true` as production evidence.
- Accepts legacy heuristic support only under a compatibility status.

Extend replay and reference tests:

- Stochastic model refs are allowed in reproducibility bundles.
- Residual history refs are allowed in reproducibility bundles.
- Residual history and stochastic model artifacts are hash-covered.
- Missing stochastic refs fail production probabilistic replay.

Add legacy characterization tests:

- The current Gaussian proxy path remains readable as compatibility behavior.
- Existing default interval level `0.8` and quantile levels `0.1`, `0.5`, `0.9` remain default compatibility config.
- Current one-step fitting remains available as `legacy_one_step`.
- Current fixed-step/raw-reference MDL remains available as an explicit legacy coding policy.

### 4.2 Implementation Tasks

Update schemas:

- Add `schemas/contracts/residual-history.yaml`.
- Add or extend `schemas/contracts/stochastic-law.yaml` to reference residual-history-backed evidence.
- Add `stochastic_model_manifest@1.0.0`.
- Add residual history and stochastic model entries to `schemas/contracts/reference-types.yaml`.
- Update schema registry and module registry.
- Mirror schema changes under `src/euclid/_assets/schemas/contracts/`.

Update manifest support:

- Add runtime dataclasses for residual history and stochastic model manifests in `src/euclid/manifests/runtime_models.py` or the repo's established manifest module.
- Add optional `residual_history_refs` and `stochastic_model_refs` where prediction artifacts, run results, publication evidence, and reproducibility bundles enumerate refs.

### 4.3 Acceptance Criteria

- Existing artifacts remain readable.
- New manifest types validate through spec compiler tests.
- Production stochastic evidence tests fail until residual-history-backed runtime support is implemented.
- Legacy Gaussian proxy is explicitly labeled compatibility-only.

### 4.4 Prompt 1 Progress

Status: implemented for contract scaffolding and legacy safety.

- Added `residual_history_manifest@1.0.0` contract scaffolding with row geometry, split-role, availability, digest, and replay identity requirements.
- Strengthened `stochastic_model_manifest@1.0.0` contract scaffolding so production evidence requires residual-history evidence, while heuristic Gaussian support is compatibility-only.
- Updated schema registry, reference profiles, module registry, and packaged schema mirrors for residual-history and stochastic-model refs.
- Added runtime manifest models for residual histories and stochastic models, plus optional residual-history and stochastic-model refs on prediction artifacts, run results, and reproducibility bundles.
- Preserved the existing Gaussian proxy forecast behavior and labeled emitted probabilistic artifacts with compatibility stochastic support status.
- Added legacy characterization coverage for Gaussian proxy defaults, interval coverage `0.8`, quantile levels `[0.1, 0.5, 0.9]`, `legacy_one_step` fitting, and fixed-step/raw-reference MDL compatibility labels.

### 4.5 Handoff To Prompt 2

Prompt 2 can build residual-history capture against the new `ResidualHistoryManifest` shape and should replace the compatibility-only Gaussian proxy path with residual-history-backed stochastic evidence. The current probabilistic artifact refs are intentionally empty for the proxy path; production refs should be populated only after residual histories and stochastic model manifests are emitted as registered artifacts.

## 5. Phase 1 - Residual History Capture

### 5.1 Tests To Write First

Add `tests/unit/modules/test_candidate_fitting_residual_history.py`:

- Analytic fitter emits residual records with origin, target, horizon, forecast, realized, and residual.
- Weighted analytic fitter records weights and weighted residual summary.
- Recursive fitter records residuals from legal per-origin state, not final state.
- Spectral fitter records phase/state used at origin.
- Algorithmic fitter records the replay state used for each origin.
- Additive residual composition records base prediction, correction prediction, final prediction, and aligned residual rows.
- Shared/local panel fitting records entity identifiers and local/global components.
- Residual-history digest is deterministic across repeated runs.
- Confirmatory rows cannot appear as stochastic training evidence.

Extend `tests/unit/modules/test_split_planning.py`:

- Training-origin panels include only complete origin/horizon targets inside the training slice.
- Non-contiguous horizons such as `(1, 3)` are represented correctly.
- A missing entity target causes a clear incomplete-panel diagnostic.
- Target rows outside the training slice are excluded.
- Origin availability and target availability are both represented or derivable.

Add a leakage regression fixture:

- Include a late outlier target unavailable at an earlier origin.
- Assert that the late outlier cannot affect earlier-origin residual support.

### 5.2 Implementation Tasks

Add immutable runtime data models:

- `ForecastResidualRecord`
- `ResidualHistorySummary`
- `ResidualHistoryManifest`
- Optional `ResidualHistoryValidationResult`

Suggested record fields:

- `candidate_id`
- `fit_window_id`
- `entity`
- `origin_index`
- `origin_time`
- `origin_available_at`
- `target_index`
- `target_event_time`
- `target_available_at`
- `horizon`
- `point_forecast`
- `realized_observation`
- `residual`
- `weight`
- `split_role`
- `residual_basis`
- `time_safety_status`
- `component_id`
- `replay_identity`

Update fitting code:

- Capture residuals inside `src/euclid/modules/candidate_fitting.py`.
- Do not reconstruct earlier-origin residuals from `fit_result.final_state`.
- For recursive, spectral, and algorithmic candidates, record the legal state snapshot or the pre-update prediction used at the origin.
- Extend fit result objects to carry residual history or a residual-history ref.
- Add residual-history summary diagnostics to candidate transient diagnostics.

Update split planning:

- Add a helper for legal training-origin panels near the existing split planning helpers.
- Make origin availability explicit or consistently derivable from the origin feature row.
- Preserve target availability for time-safety validation.

### 5.3 Acceptance Criteria

- Every production-eligible fitted candidate exposes deterministic residual history.
- Residual histories are hashable and replay-addressable.
- Residual records identify their split role.
- Confirmatory residuals are never used as stochastic training evidence.
- Time-leak regression passes.

### 5.4 Prompt 2 Progress

Status: Phase 1 foundation complete; candidate fitters have not yet been refactored
to emit residual histories.

- Added immutable residual-history runtime models:
  `ForecastResidualRecord`, `ResidualHistorySummary`,
  `ResidualHistoryValidationIssue`, and `ResidualHistoryValidationResult`.
- Added deterministic residual-history digest and source-row-set digest helpers.
- Added residual-history summary and validation helpers covering split role, origin
  availability, target availability, horizon geometry, entity, and replay identity.
- Added legal training-origin panel planning in `split_planning` for complete
  origin/horizon rows inside the training slice.
- Added support for non-contiguous training residual horizons such as `(1, 3)`.
- Added clear split-planning diagnostics for targets outside the training slice
  and missing entity horizon targets.
- Existing forecasting behavior remains unchanged; this prompt only added
  support models/helpers and characterization tests.

### 5.5 Prompt 3 Handoff

Prompt 3 should capture fitter residuals by calling
`build_legal_training_origin_panel(...)` for each candidate fit window, then
emitting one `ForecastResidualRecord` per returned panel row. The record should
use the panel's `entity`, `origin_index`, `origin_time`, `origin_available_at`,
`horizon`, `target_index`, `target_event_time`, `target_available_at`, and
`split_role`; the fitter should fill `point_forecast`, the target row's realized
observation, `residual = realized_observation - point_forecast`, any weight or
component id, `time_safety_status`, and a deterministic replay identity that
includes the candidate id, fit window id, entity, origin, horizon, and the legal
state/pre-update prediction used at that origin.

After capture, Prompt 3 should call `summarize_residual_history(...)` and
`validate_residual_history(...)`, then package the rows with the existing
`ResidualHistoryManifest` scaffold. Recursive, spectral, algorithmic,
shared/local, and additive residual fitters must record the legal state snapshot
or pre-update prediction used for each origin rather than reconstructing
residuals from the final fitted state.

### 5.6 Prompt 3 Progress

Status: residual capture in candidate fitting complete; publication semantics are
still unchanged.

- `fit_candidate_window(...)` now emits `residual_history`,
  `residual_history_summary`, and `residual_history_validation` on
  `CandidateWindowFitResult`.
- Candidate residual rows are built from `build_legal_training_origin_panel(...)`
  and include origin/target availability, entity, horizon, split role, row weight,
  point forecast, realized observation, residual, and deterministic replay
  identity.
- Analytic residual rows carry fit weights and weighted residual summary fields.
- Recursive, spectral, and algorithmic residual rows use the legal per-origin
  state or phase snapshot for replay identity and point forecast, not the final
  fitted state.
- Additive residual capture records component alignment diagnostics with base
  prediction, residual prediction, final prediction, origin, target, and horizon.
- Shared/local panel residual capture preserves entity identifiers in every
  residual row.
- `build_candidate_fit_artifacts(...)` now emits a residual-history manifest
  alongside candidate spec, candidate state, and reducer artifact, but downstream
  publication/replay references are not yet changed.
- Leakage regression coverage verifies a late outlier outside the earlier
  development fit window cannot affect that window's residual evidence.

### 5.7 Prompt 4 Handoff

Prompt 4 should consume residual evidence from:

- `fit_result.residual_history`
- `fit_result.residual_history_summary`
- `fit_result.residual_history_validation`
- `build_candidate_fit_artifacts(...).residual_history`

The emitted residual-history manifest ref is the natural source for later
stochastic model evidence. Prompt 4 should wire residual-history artifacts into
the next artifact-registration or stochastic-evidence step without treating
legacy shared/local adapter status or compatibility Gaussian proxy support as
publication-grade stochastic evidence. Shared/local fitting still reports
`legacy_non_claim_adapter` semantics even though residual rows are captured.

## 6. Phase 2 - Residual-Backed Stochastic Models

### 6.1 Tests To Write First

Extend `tests/unit/stochastic/test_process_models.py`:

- Reject bare synthetic residual lists in production mode.
- Accept validated residual histories.
- Enforce minimum residual count.
- Fit Gaussian parameter maps.
- Fit Student-t parameter maps, including estimated or recorded degrees of freedom.
- Fit Laplace parameter maps.
- Record residual count, residual-history digest, residual source kind, horizon coverage, family id, support kind, and horizon law.
- Reject unsupported residual family/support combinations with a clear reason.

Extend `tests/unit/modules/test_probabilistic_evaluation.py`:

- Production probabilistic artifact cannot be emitted without stochastic model evidence.
- Emitted distribution rows reference a stochastic model.
- Distribution scale no longer depends on optimizer-loss proxy.
- Interval and quantile rows are derived from the fitted family.
- Event probability binds the declared family.
- Heuristic Gaussian support emits a downgrade or compatibility status.

Extend replay/publication tests:

- Residual-history and stochastic-model refs are included in replay hash records.
- Missing stochastic-model artifact fails production probabilistic publication.
- Missing residual-history artifact fails stochastic-model replay.

### 6.2 Implementation Tasks

Refactor `src/euclid/stochastic/process_models.py`:

- Change `fit_residual_stochastic_model` to accept a validated residual-history object.
- Add production mode validation that rejects synthetic proxy residual sources.
- Add min-count and coverage checks.
- Add family parameter maps.
- Add support semantics for Gaussian, Student-t, and Laplace as the first production residual families.
- Distinguish residual families from observation-space families where needed.

Update probabilistic evaluation:

- Remove `_stochastic_residual_proxy`.
- Remove `_base_scale`.
- Replace hardcoded `family_id="gaussian"` and `horizon_scale_law="sqrt_horizon"` with model/config-derived values.
- Build stochastic predictive support from fitted stochastic model manifests.
- Serialize family parameters into distribution rows.

Update manifests and replay:

- Add `StochasticModelManifest`.
- Add `stochastic_model_ref` or `stochastic_model_refs` to prediction artifacts.
- Include stochastic model refs in run result manifests, publication evidence, reproducibility bundles, demo runtime, and compatibility runtime.
- Add stochastic model artifact hashing to replay.

### 6.3 Acceptance Criteria

- Production probabilistic artifacts always cite stochastic model evidence.
- Stochastic model evidence always cites residual history.
- Legacy Gaussian proxy remains readable but cannot satisfy production stochastic-law support.
- Replay determinism includes stochastic model identity and residual evidence.

### 6.4 Prompt 4 Progress

Status: residual-backed stochastic process fitting is implemented; probabilistic
evaluation still uses the legacy compatibility proxy until Prompt 5 wires real
stochastic model artifacts into prediction emission.

- `fit_residual_stochastic_model(...)` now accepts `residual_history=...`
  records from candidate fitting and validates them with
  `validate_residual_history(...)`.
- Production stochastic evidence rejects bare synthetic residual lists with
  `synthetic_residual_source_not_production`.
- Production stochastic evidence still requires `residual_history_ref`; validated
  histories without a manifest ref do not satisfy production support.
- Stochastic model output records `residual_history_digest`,
  `residual_source_kind`, residual count, horizon coverage, residual family,
  support kind, horizon scale law, fitted parameter maps, and replay identity.
- Gaussian parameter maps use residual mean and population residual scale.
- Student-t parameter maps record explicit degrees of freedom when provided and
  validate `df > 2`.
- Laplace parameter maps use median location and mean absolute deviation scale.
- Minimum residual count and required horizon coverage fail closed with explicit
  diagnostics.
- Compatibility synthetic residual lists remain readable and continue to power
  the legacy Gaussian proxy, but they default to compatibility evidence and
  `heuristic_gaussian_support: true` only for Gaussian synthetic support.

### 6.5 Prompt 5 Handoff

Prompt 5 should wire probabilistic evaluation to the residual-backed stochastic
model path:

- Use `fit_result.residual_history` as `residual_history=...`.
- Use `build_candidate_fit_artifacts(...).residual_history.ref` as
  `residual_history_ref`.
- Pass `evidence_status="production"` only when the residual-history artifact ref
  is available and the residual history validation passed.
- Populate prediction artifact `residual_history_refs` and
  `stochastic_model_refs` from the residual-history manifest and stochastic-model
  manifest emitted from `FittedResidualStochasticModel.as_manifest()` or the
  runtime `StochasticModelManifest`.
- Build distribution/interval/quantile/event-probability rows from
  `model.support_path()` and `model.residual_parameter_summary` instead of
  `_stochastic_residual_proxy`, `_base_scale`, optimizer-loss proxy scale, or
  hardcoded Gaussian support.
- Keep the legacy Gaussian proxy as compatibility-only for old/readable paths
  until the later removal prompt; it must not satisfy production stochastic-law
  evidence.

### 6.6 Prompt 5 Progress

Status: production probabilistic prediction artifacts now use
residual-history-backed stochastic evidence; the legacy Gaussian proxy remains
compatibility-only for readable old paths.

- `emit_probabilistic_prediction_artifact(...)` accepts
  `stochastic_evidence_mode="production"`, `residual_history_ref`, and
  stochastic family options.
- Production mode fits stochastic support from `fit_result.residual_history`,
  requires a residual-history artifact ref, validates the residual history, and
  emits `stochastic_model_refs` from a `StochasticModelManifest`.
- Distribution rows carry `distribution_parameters`; interval, quantile, and
  event-probability rows now carry the declared distribution family and fitted
  parameter map.
- Interval and quantile construction binds the declared Gaussian, Student-t, or
  Laplace family instead of using fixed Gaussian z-scores in production mode.
- Event probabilities bind the declared family through the stochastic
  observation model rather than hardcoding Gaussian support.
- Optimizer-loss proxy scale is no longer used by the production path; it
  remains only behind the compatibility Gaussian proxy.
- Run-result manifests, publication helpers, demo runtime, operator runtime,
  compatibility runtime, and replay required-ref checks now carry stochastic
  support status plus residual-history and stochastic-model refs when the
  prediction artifact provides them.
- Production probabilistic replay fails closed when residual-history or
  stochastic-model refs are missing.

### 6.7 Prompt 6 Handoff

Prompt 6 should build on the family-aware rows emitted here:

- Prefer row `distribution_parameters` over legacy scalar `location`/`scale`
  fields in scoring, calibration, and observation-model binding.
- Continue accepting legacy Gaussian rows that only have scalar fields as
  compatibility artifacts.
- Add explicit family-aware `cdf`, `ppf`, interval, and PIT helpers/tests in
  `stochastic.observation_models` before replacing local evaluation quantile
  construction with shared APIs.
- Treat `stochastic_support_status="compatibility"` and
  `heuristic_gaussian_support_not_production` as downgrade evidence only.
- If production workflows begin registering stochastic-model manifests as
  standalone artifacts, include those registered artifacts in replay hash
  records alongside their residual-history artifacts.

## 7. Phase 3 - Family-Aware Artifacts, Scoring, And Calibration

### 7.1 Tests To Write First

Extend `tests/unit/stochastic/test_observation_models.py`:

- `cdf`, `ppf`, `interval`, and PIT for Gaussian.
- `cdf`, `ppf`, `interval`, and PIT for Student-t.
- `cdf`, `ppf`, `interval`, and PIT for Laplace.
- Deterministic randomized PIT for discrete families, or explicit unsupported failure.
- Parameter validation for each family.
- Mixture PIT either works or fails closed with `unsupported_pit_family`.

Extend `tests/unit/stochastic/test_scoring_rules.py`:

- Gaussian CRPS remains supported.
- Non-Gaussian CRPS fails early unless a tested implementation is added.
- Distribution log score works with parameter maps.

Extend `tests/unit/modules/test_scoring.py`:

- `distribution_parameters` are preferred over legacy scalar fields.
- Legacy `location` and `scale` fallback still works.
- Required quantile levels must be present exactly once.
- Required interval levels must be present exactly once.
- Crossed quantiles fail with a clear reason.
- Score result records effective probabilistic score configuration.

Extend `tests/unit/modules/test_calibration.py`:

- PIT uses the row's declared family.
- Gaussian fallback is not used for Student-t or Laplace rows.
- Equal-width reliability bins work.
- Equal-mass reliability bins work.
- Adaptive minimum-count bins work.
- Interval coverage diagnostics are per selected coverage level.
- Quantile hit diagnostics are per selected quantile level.
- Insufficient sample count by family/object/horizon abstains or fails.
- Recalibration/conformal lanes cannot fit on confirmatory rows.

Add runtime manifest roundtrip tests:

- Distribution rows round-trip with `distribution_parameters`.
- Interval rows round-trip with plural `intervals`.
- Prediction artifacts round-trip with effective probabilistic configuration.
- Score and calibration result manifests round-trip with effective configuration and lane diagnostics.

### 7.2 Implementation Tasks

Update observation model registry:

- Add family-neutral helpers: `cdf`, `ppf`, `interval`, `pit`, `parameter_names`, `distribution_family_id`.
- Implement deterministic randomized PIT using a row key and configured seed for discrete families if enabled.
- Fail closed for unsupported family/diagnostic combinations.

Update prediction artifact rows:

- Add optional `distribution_parameters`.
- Keep `location`, `scale`, and `support_kind` for compatibility.
- Add plural interval row support.
- Preserve legacy single interval fields.
- Add effective prediction configuration to prediction artifacts.

Update probabilistic configuration:

- `probabilistic.distribution_family`
- `probabilistic.interval_levels`
- `probabilistic.quantile_levels`
- `probabilistic.event_definitions`
- `probabilistic.reliability_bins`
- `probabilistic.pit`
- `probabilistic.recalibration_lanes`

Update scoring:

- Add a single helper to bind distribution rows to observation models.
- Validate score/family compatibility before aggregation.
- Use configured levels for quantile and interval scoring.
- Fail if required levels are absent, duplicated, or incoherent.
- Keep CRPS Gaussian-only until non-Gaussian CRPS is explicitly implemented.

Update calibration:

- Replace Gaussian-only PIT with family-bound PIT.
- Add configurable reliability bin strategies.
- Add per-level interval and quantile diagnostics.
- Include calibration identity fields: object type, observation family, score policy, horizon set, origin set, entity panel, and lane id where relevant.
- Add lane-aware output for raw, recalibrated, and conformal diagnostics.

### 7.3 Acceptance Criteria

- Student-t and Laplace artifacts produce family-correct intervals, quantiles, log scores, and PIT.
- Unsupported family/score combinations fail before score aggregation.
- Calibration gates are family-aware and sample-size-aware.
- Recalibration and conformal lanes are time-safe.

### 7.4 Prompt 6 Progress

Status: family-aware observation helpers, probabilistic row shapes, scoring
bindings, calibration diagnostics, and lane safety are implemented for the
current production/compatibility probabilistic artifacts.

- `BoundObservationModel` now exposes `parameter_names`,
  `distribution_family_id`, `ppf`, central `interval`, and PIT helpers.
- Discrete PIT fails closed unless deterministic randomized PIT is requested
  with a stable row key.
- Prediction artifacts can carry `effective_probabilistic_config`; interval
  rows can carry plural `intervals`; score and calibration result manifests can
  carry effective config and lane metadata.
- Probabilistic scoring now prefers row `distribution_parameters` over legacy
  scalar fields while preserving scalar fallback for readable Gaussian
  compatibility artifacts.
- Distribution scoring binds rows through a single observation-model helper,
  keeps CRPS Gaussian-only, and fails non-Gaussian CRPS before aggregation.
- Interval and quantile scoring validate required levels exactly once and fail
  crossed quantiles clearly.
- Calibration PIT now uses the declared row family, including Student-t and
  Laplace parameter maps.
- Calibration diagnostics now include equal-width, equal-mass, and
  adaptive-min-count reliability bins plus per-level interval/quantile
  diagnostics.
- Calibration enforces minimum sample count failures and blocks recalibration
  or conformal fit lanes on confirmatory rows.

### 7.5 Thread 2 / Prompt 7 Handoff

Prompt 7 should focus on multi-horizon fitting alignment, not another
probabilistic evidence migration:

- Use the existing legal training-origin panel helpers from split planning as
  the source of fit/scoring geometry.
- Introduce `FitStrategySpec` with identity covering horizon set, horizon
  weights, point loss, entity aggregation mode, and strategy id.
- Keep `legacy_one_step` as the default compatibility strategy and preserve
  current single-horizon behavior for `horizon_set=(1,)`.
- Add rollout objective helpers that produce the same per-origin/horizon rows
  consumed by `score_point_prediction_artifact(...)`; the training objective
  should cross-check against scoring aggregation on the identical panel.
- Implement non-contiguous horizon support such as `(1, 3)` using legal target
  availability diagnostics from split planning.
- Thread fit strategy metadata, training-scored-origin-set id, horizon weights,
  and objective geometry into candidate-fit diagnostics and replay identity.
- Do not let recursive, direct, joint, or rectify strategies fit on
  confirmatory rows; Prompt 6 lane safety covers calibration/conformal lanes,
  but Prompt 7 must apply the same time-safety principle to multi-horizon
  fitting.

## 8. Phase 4 - Multi-Horizon Fitting Alignment

### 8.1 Tests To Write First

Add `tests/unit/search/test_fitting_boundary.py`:

- `FitStrategySpec` defaults to `legacy_one_step`.
- Strategy identity includes horizon set, horizon weights, point loss, and entity aggregation mode.
- Non-contiguous horizon sets are valid.
- Invalid horizon weights fail validation.

Extend `tests/unit/modules/test_candidate_fitting.py`:

- Recursive rollout objective differs from one-step objective when horizon weights matter.
- Direct strategy emits horizon-specific parameters.
- Joint strategy changes fitted parameters when horizon weights change.
- Rectify strategy trains correction residuals only on legal training-origin panels.
- Incompatible candidate/strategy combinations fail closed with `incompatible_fit_strategy`.

Add scoring cross-check tests:

- Build a training prediction artifact for a known candidate.
- Assert rollout training objective equals scoring aggregation over the same origin/horizon panel.

Add integration fixture tests:

- Non-contiguous horizon set such as `(1, 3)`.
- Search `inner_primary_score` uses rollout objective when rollout fitting is configured.
- Legacy one-step objective remains available when explicitly configured.

### 8.2 Implementation Tasks

Add `src/euclid/fit/multi_horizon.py`:

- `FitStrategySpec`
- `resolve_fit_strategy`
- `TrainingOriginPanel`
- `RolloutObjectiveResult`

Add split planning helpers:

- Build legal complete training-origin panels.
- Include stable `training_scored_origin_set_id`.
- Reject panels with missing legal targets.

Add shared forecast path module:

- Move forecast path logic from evaluation into `src/euclid/modules/forecast_paths.py`.
- Use the shared path engine for fitting, evaluation, and probabilistic evaluation.
- Preserve legacy private behavior through wrappers during migration.

Extend candidate fitting:

- Add optional `fit_strategy`.
- Default to `legacy_one_step`.
- Store strategy metadata, objective geometry, horizon weights, origin-panel id, and optimizer diagnostics.
- Add rollout objective computation that mirrors scoring.

Implement strategies in stages:

1. `recursive`: one parameterization, full recursive rollout path.
2. `direct`: horizon-specific models and parameters.
3. `joint`: one shared parameter vector optimized against all horizons.
4. `rectify`: recursive base plus horizon-specific correction.
5. Later: spectral, shared/local, additive residual, and composition-aware rollout optimizers.

### 8.3 Acceptance Criteria

- `horizon_set=(1,)` preserves legacy behavior where expected.
- Multi-horizon training uses the same geometry as scoring.
- Horizon weights can affect fitted parameters.
- Search diagnostics expose objective geometry and fit strategy.
- No target outside the training slice participates in multi-horizon fitting.

## 9. Phase 5 - MDL And Descriptive Coding Upgrade

### 9.1 Tests To Write First

Add `tests/unit/math/test_quantization.py`:

- Fixed-step compatibility.
- Explicit measurement-resolution quantization.
- Scale-adaptive quantization.
- Deterministic zero-scale fallback.
- Scale-equivariance property under positive rescaling where intended.
- Invalid adaptive config fails clearly.

Add `tests/unit/math/test_codelength.py`:

- Natural integer code.
- Zigzag signed integer code.
- Float lattice scalar code.
- Literal code policy.
- Parameter code policy.
- State code policy.
- String or program literal cost.
- Monotonicity under finer precision.

Extend `tests/unit/search/test_descriptive_coding.py`:

- Reference-bank selection includes family-selection cost.
- Raw reference remains available as legacy policy.
- Naive residual reference works.
- Seasonal naive reference is unavailable without legal seasonal period.
- Differenced/local-linear reference works.
- Strict comparability fails across different quantizers.
- Strict comparability fails across different row sets.
- Strict comparability fails across different data-code families.
- Prequential MDL uses prefix-only residual evidence.

Add operator/prototype regression tests:

- Same deterministic candidate, row set, and policy yields identical bits across backend, prototype, and operator workflows.

### 9.2 Implementation Tasks

Update quantization:

- Add `QuantizationPolicy`.
- Add `ResolvedQuantization`.
- Add `resolve_quantizer(values, policy, measurement_resolution=None)`.
- Keep fixed-step mid-tread as explicit legacy/default policy.
- Add measurement-resolution and scale-adaptive modes.

Update codelength helpers:

- Move integer and scalar code laws into `src/euclid/math/codelength.py`.
- Add shared functions for literal, parameter, state, and residual coding.
- Stop importing private helpers from reference-description modules.

Update reference descriptions:

- Add reference family bank:
  - `raw_quantized_sequence_v1`
  - `naive_last_observation_residuals_v1`
  - `seasonal_naive_residuals_v1`
  - `differenced_local_linear_v1`
- Record unavailable-family diagnostics.
- Record selected reference family and family-selection bits.

Update descriptive search:

- Thread quantization policy, codelength policy, reference policy, seasonal period, data-code family, and coding row-set id through search.
- Replace count-based backend model coding with shared codelength helpers.
- Add `CodelengthComparisonKey`.
- Use `strict_single_class` for law eligibility.
- Allow diagnostic-only ranking inside explicitly comparable classes.

Add prequential MDL:

- Keep current data code as `residual_integer_sequence_v1`.
- Add `prequential_laplace_residual_bin_v1` as opt-in.
- Estimate residual scale only from prefix information.
- Record warmup policy, scale floor, distribution family, quantizer, and row-set identity.

### 9.3 Acceptance Criteria

- Incomparable codelength policies cannot be ranked together for law eligibility.
- Reference bits include family-selection cost.
- Prequential MDL uses only prefix/development evidence.
- Backend, prototype, and operator workflows agree for the same candidate, row set, and policy.

## 10. Phase 6 - Fixtures, Docs, Workbench, And Release Migration

### 10.1 Tests To Write First

Add or extend fixture coverage tests for:

- Residual-history-backed probabilistic publication.
- Heuristic Gaussian downgrade.
- Student-t or Laplace calibrated distribution lane.
- Non-contiguous horizon set.
- Additive residual multi-horizon fitting.
- MDL comparability failure.
- Conformal or recalibration no-leak failure.

Update documentation truthfulness tests:

- Gaussian/sqrt-horizon support is documented as compatibility-only.
- Production stochastic support requires residual-history-backed evidence.
- Multi-horizon fitting docs match configured strategy behavior.
- MDL docs describe quantization, reference bank, and comparability keys.

### 10.2 Implementation Tasks

Update docs:

- `docs/math.md`
- `docs/reference/search-core.md`
- `docs/search-core.md`
- `docs/reference/modeling-pipeline.md`
- `docs/reference/contracts-manifests.md`

Update fixtures:

- Add parallel canonical fixtures for the new evidence path.
- Do not immediately overwrite old goldens.
- Regenerate runtime phase fixtures only after replay identity is deterministic.
- Update `fixtures/canonical/fixture-coverage.yaml` atomically with new canonical publication bundles.

Update workbench:

- Display configured interval and quantile levels.
- Stop assuming `location +/- scale`.
- Surface distribution family, stochastic model ref, residual-history ref, calibration bins, and lane status.
- Make downgrade reasons visible for heuristic Gaussian compatibility evidence.

### 10.3 Acceptance Criteria

- Spec compiler, fixtures, docs, workbench surfaces, and replay agree.
- Release smoke passes with both legacy compatibility evidence and new production evidence.
- Production probabilistic publication requires residual-history-backed stochastic model evidence.

## 11. Phase 7 - Richer Primitive Forecasting Families

Do this only after Phases 0 through 6 are stable.

### 11.1 Tests To Write First

Add candidate family tests for:

- Sparse AR(p) or selected lag models.
- Seasonal lag models.
- Trend terms.
- Multi-harmonic Fourier models.
- Shared/local shrinkage diagnostics.
- Observation-aware GLM reducers.
- Regime mixture predictive distributions.

For every new family, require tests for:

- CIR identity.
- Fit strategy compatibility.
- Support semantics.
- Residual-history generation.
- Stochastic support integration.
- Scoring and calibration compatibility.
- Codelength comparability key.
- Replay identity.

### 11.2 Implementation Tasks

Add primitive families:

- Sparse multi-lag analytic laws.
- Seasonal lags and trend terms.
- Multi-harmonic spectral laws with harmonic-group selection.
- Hierarchical shared/local panel laws with shrinkage.
- GLM-style observation-aware reducers for count, binary, bounded, and positive targets.
- Regime-conditioned mixture predictive distributions.

### 11.3 Acceptance Criteria

- New primitive families cannot bypass residual-history-backed evidence.
- Search disclosures remain bounded by backend class.
- Richer families publish only the claim lane justified by evidence.

## 12. Cross-Cutting Risk Gates

No leakage:

- Stochastic models, conformal lanes, recalibration lanes, and prequential MDL must not fit on confirmatory residuals.

Replay determinism:

- Residual-history id, stochastic family, horizon law, parameters, score policy, seed, library versions, and schema versions must be hash-covered.

Comparability:

- No cross-family, cross-horizon, cross-weight, cross-quantizer, cross-reference-policy, cross-row-set, or ambiguous composition-signature comparison may publish as comparable.

Calibration power:

- Enforce minimum sample counts by object type, family, horizon bucket, and lane.

Performance:

- Multi-horizon fitting must use cached or vectorized path rollout before release smoke.

Publication safety:

- The old Gaussian proxy path may score numerically, but it must not publish as production stochastic evidence.

## 13. Final Readiness Chain - Prompt 1

Status: completed for residual-backed probabilistic publication/replay
stabilization.

- Reproduced the four failing publication goldens with verbose diffs.
- Diagnosed the phase07 point publication failures as stale deterministic
  manifest hash/ref changes.
- Diagnosed the probabilistic publication failures as real release-path
  omissions: stochastic model manifests were not exposed for registry
  publication, probabilistic demo publication was still using compatibility
  evidence, and the first production wiring attempted to use confirmatory
  residuals as stochastic training evidence.
- Fixed the probabilistic publication path so production stochastic support uses
  development residual history while confirmatory prediction/scoring remains the
  publication target.
- Registered residual-history and stochastic-model manifests, threaded their
  refs into prediction artifacts, run results, scorecards, claim cards, local
  publication catalog entries, reproducibility required refs, and replay hash
  records.
- Strengthened replay validation so heuristic Gaussian compatibility evidence
  cannot satisfy a production stochastic-evidence body.
- Regenerated the four affected publication goldens from the test snapshot
  helpers after the runtime identity stabilized.

### 13.1 Handoff To Prompt 2

Prompt 2 should start from the passing Prompt 1 golden and targeted
unit/contract gates, then broaden release confidence around the same evidence
surface:

- Run the wider contract/fixture gate and inspect any remaining publication
  fixture drift outside the four Prompt 1 goldens.
- Confirm operator/prototype probabilistic publication surfaces match the demo
  path for residual-history refs, stochastic-model refs, scorecard/claim-card
  disclosure, catalog disclosure, and replay hash records.
- Add stress coverage for publication bundles with more than one stochastic
  model ref if multi-origin probabilistic publications begin emitting multiple
  distinct stochastic support manifests.
- Keep compatibility Gaussian artifacts readable, but preserve the production
  fail-closed behavior for `heuristic_gaussian_support_not_production`.

### 13.2 Final Readiness Chain - Prompt 2

Status: completed for Phase 4 scaffolding without enabling new optimizer
behavior.

- Added `FitStrategySpec` and `resolve_fit_strategy(...)` with deterministic
  identity over strategy id, horizon set, horizon weights, point loss, and
  entity aggregation mode.
- Reused split-planning legal training-origin panels as fitting geometry and
  added rollout objective rows/results over those panels.
- Cross-checked rollout objective aggregation against
  `score_point_prediction_artifact(...)` on the identical origin/horizon panel.
- Added the shared `modules.forecast_paths` import surface and routed point and
  probabilistic evaluation through it while preserving the existing legacy
  forecast-path implementation behind the wrapper.
- Wired optional `fit_strategy` into `fit_candidate_window(...)`. The standalone
  spec defaults to `legacy_one_step` with `horizon_set=(1,)`; candidate fitting
  keeps the existing default fit-window horizon geometry under the
  `legacy_one_step` strategy id.
- Threaded fit strategy metadata, objective horizon weights, training scored
  origin-set identity, and fit strategy identity into candidate fit diagnostics
  and residual replay identities.
- Added regression coverage proving explicit `horizon_set=(1,)` preserves the
  legacy one-step fit surface.

### 13.3 Handoff To Prompt 3

Prompt 3 can build on this scaffolding by implementing behavior, not more
metadata plumbing:

- Add compatibility checks and fail-closed diagnostics for unsupported
  candidate/strategy pairs.
- Implement `recursive` rollout fitting with a single parameterization and full
  recursive objective rows.
- Implement `direct` fitting with horizon-specific parameters and manifests.
- Implement `joint` fitting with one shared parameter vector optimized over all
  declared horizons.
- Implement `rectify` fitting as recursive base plus legal-panel correction
  residuals.
- Keep all four strategy families constrained to legal development/training
  panels; no confirmatory rows should participate in strategy fitting.

### 13.4 Final Readiness Chain - Prompt 3

Status: completed for staged multi-horizon fitting behavior while preserving
`legacy_one_step` as the default.

- Implemented recursive rollout fitting for analytic affine candidates and
  recursive level-smoother candidates, with rollout objective diagnostics over
  the declared legal training-origin panel.
- Implemented direct analytic fitting with horizon-specific parameter names
  such as `horizon_3__intercept` and `horizon_3__lag_coefficient`, including
  non-contiguous horizon sets like `(1, 3)`.
- Implemented joint analytic fitting with one shared analytic parameter vector
  optimized against declared horizon weights.
- Implemented rectify analytic fitting as a legacy analytic base plus
  horizon-specific correction residuals trained only on legal training-origin
  rows.
- Added fail-closed `incompatible_fit_strategy` behavior for unsupported
  strategy/family combinations and for composed or expression candidates that
  do not yet have rollout-aware fitting.
- Made descriptive search use `rollout_primary_objective` for
  `inner_primary_score` when a rollout fit strategy is configured.
- Added strategy diagnostics for fit strategy id, identity, horizon set,
  horizon weights, point loss, entity aggregation mode, training origin-set id,
  and rollout objective geometry.

### 13.5 Handoff To Prompt 4

Prompt 4 should treat Phase 4 fit-strategy behavior as available for analytic
and recursive smoke coverage, but should keep the following constraints visible:

- `legacy_one_step` remains the default and the compatibility baseline.
- `direct`, `joint`, and `rectify` are analytic-only.
- Recursive rollout support is limited to analytic affine rollout and recursive
  level smoothing/running-mean candidates.
- Spectral, algorithmic, expression, and composed candidates intentionally fail
  closed for rollout strategy ids until family-specific multi-horizon fitting is
  specified.
- Publication/replay work in later prompts should surface the new fit strategy
  diagnostics without weakening residual-history and stochastic-model evidence
  closure.

### 13.6 Final Readiness Chain - Prompt 4

Status: completed for Phase 5 MDL/codelength alignment.

- Added explicit `QuantizationPolicy`, `ResolvedQuantization`, and
  `resolve_quantizer(...)` support for fixed-step, measurement-resolution, and
  scale-adaptive mid-tread quantization.
- Centralized natural-integer, zigzag signed-integer, float-lattice, literal,
  parameter, state, residual-data, and prequential Laplace residual-bin
  codelength laws in `math.codelength`.
- Added a reference-code family bank covering raw quantized sequences,
  naive-last-observation residuals, seasonal-naive residuals, and
  differenced/local-linear residuals, with explicit family-selection cost.
- Added `CodelengthComparisonKey` and strict single-class law eligibility over
  quantizer, reference policy, data-code family, support, horizon geometry,
  coding row set, residual-history construction, parameter lattice, and state
  lattice.
- Threaded data-code family, reference policy, quantization policy, seasonal
  period, coding row-set id, and codelength comparison keys through descriptive
  coding and descriptive search artifacts.
- Added opt-in `prequential_laplace_residual_bin_v1` data coding with
  prefix-only evidence rows.
- Replaced the backend proposal count-based model-code surface with shared
  precision-aware codelength helpers.
- Collapsed duplicated prototype/operator `_description_components(...)` logic
  onto the shared codelength helper and added parity coverage.

### 13.7 Handoff To Prompt 5

Prompt 5 can assume the MDL comparison surface is strict and metadata-backed:

- Default compatibility behavior still uses fixed-step mid-tread quantization
  and the raw reference family, but the cost now includes explicit reference
  family selection.
- Prequential residual-bin coding is opt-in through `data_code_family`; the
  default remains residual signed-integer Elias-delta coding.
- Search artifacts now carry row-set and comparison-key metadata that
  publication/replay prompts should preserve when surfacing descriptive
  evidence.
- Family-specific stochastic/probabilistic publication work should not loosen
  `strict_single_class_v1`; incompatible codelength keys must remain
  diagnostic-only.

### 13.8 Final Readiness Chain - Prompt 5

Status: completed for Phase 6 fixture, docs, workbench, and golden migration
coverage.

- Added a parallel Phase 6 canonical readiness fixture set for
  residual-history-backed probabilistic publication, heuristic Gaussian
  downgrade, Student-t calibrated distribution support, non-contiguous
  horizon panels, additive residual multi-horizon fitting, MDL comparability
  failure, and conformal recalibration no-leak failure.
- Updated docs truthfulness coverage and reference docs so Gaussian
  `sqrt_horizon` support is documented as readable compatibility behavior, not
  the whole production probabilistic story.
- Documented residual-history-backed stochastic support, family-aware scoring
  and calibration, multi-horizon `FitStrategySpec` strategies, strict
  `CodelengthComparisonKey` comparability, and opt-in
  `prequential_laplace_residual_bin_v1`.
- Updated workbench normalization and frontend display to prefer
  family-derived/configured bands over location-plus-scale fallbacks and to
  surface stochastic model refs, residual-history refs, family, calibration
  bins, lane status, and downgrade reasons.
- Regenerated deterministic publication goldens for the current production and
  legacy downgrade paths after Prompt 1-4 identity changes.

### 13.9 Handoff To Prompt 6

Prompt 6 should treat production-vs-compatibility fixture coverage and
workbench evidence display as closed. Remaining release-hardening work should
focus on final end-to-end gates, packaging parity, and any broad-suite drift
outside the Prompt 5 surfaces:

- Keep heuristic Gaussian artifacts readable but compatibility-only unless
  production `residual_history_refs` and `stochastic_model_refs` are present.
- Preserve strict MDL comparability diagnostics in publication/replay surfaces;
  do not collapse row-set or quantizer mismatches into ordinary ranking ties.
- Re-run final full release, benchmark, performance, spec, fixture, golden,
  workbench, and frontend gates after any remaining Prompt 6 packaging changes.

### 13.10 Final Readiness Chain - Prompt 6

Status: completed for Phase 7 primitive-family expansion and full release
hardening.

- Added release-stable selected-feature analytic laws for sparse multi-lag
  (`lag_1`, `lag_2`) and seasonal-lag-plus-trend (`seasonal_lag`,
  `trend_anchor`) candidates. These preserve CIR identity, legal feature
  declarations, residual-history capture, stochastic support, score/calibration
  compatibility, codelength comparability, and replay identity for the
  supported legacy one-step strategy.
- Added multi-harmonic spectral laws with harmonic-group selection and fitted
  sine/cosine coefficient identity. Evaluation, descriptive coding, fitting,
  residual histories, and replay identity now understand harmonic groups.
- Kept richer but incomplete surfaces fail-closed: selected-feature analytic
  rollout/direct/joint/rectify fitting reports `incompatible_fit_strategy`;
  GLM-style observation-aware reducers fail without a bound supported
  observation family; regime-conditioned mixture predictive distributions fail
  scoring/calibration compatibility until mixture scoring and calibration are
  complete.
- Fixed release benchmark search-frontier fitting on ragged entity panels by
  using the common entity-local training span instead of total panel row count.
- Updated stale demo, publication, Phase 4 fixture, and packaged-asset
  expectations for the current deterministic `seasonal_naive` demo winner,
  stricter MDL admissibility, and predictive claim-card migration.
- Installed missing declared local dependencies needed by this checkout's
  release gates: `pytest-xdist` and `Pillow>=11.3.0,<12`.
- Verified final gates: focused Prompt 6 unit gate, spec/golden/fixture gate,
  `./scripts/test.sh`, benchmark smoke, performance smoke via `bash
  scripts/perf_smoke.sh` because the script lacks executable mode, and
  `./scripts/release_smoke.sh`.

### 13.11 Handoff To Final Audit Thread

Final audit should treat the Prompt 1-6 readiness chain as release-green in
this workspace and focus on independent audit rather than feature expansion:

- Confirm no production probabilistic publication path can be satisfied by
  heuristic Gaussian compatibility artifacts without residual-history-backed
  stochastic model refs.
- Confirm unsupported Phase 7 surfaces remain explicit, tested, and
  fail-closed rather than silently publishing weaker evidence.
- Review generated build reports under `build/reports/` and certification
  artifacts under `build/certification/` for packaging parity and stale-asset
  drift.
- Consider setting executable mode on `scripts/perf_smoke.sh`; the smoke gate
  itself passes when invoked through `bash`.

## 14. Test Commands

Focused unit gate:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/modules/test_probabilistic_evaluation.py \
  tests/unit/modules/test_probabilistic_scoring.py \
  tests/unit/modules/test_calibration.py \
  tests/unit/modules/test_candidate_fitting.py \
  tests/unit/search/test_descriptive_coding.py \
  tests/unit/search/test_fitting_boundary.py
```

Contract and fixture gate:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler tests/golden tests/fixtures
```

Release gate:

```bash
./scripts/test.sh
./scripts/benchmark_smoke.sh
./scripts/perf_smoke.sh
./scripts/release_smoke.sh
```

## 15. Definition Of Done

The implementation is complete when:

- Every production probabilistic artifact references a stochastic model.
- Every production stochastic model references a legal residual history.
- Distribution, interval, quantile, event, score, and calibration lanes use the declared family.
- Multi-horizon fit strategies train against the same objective geometry used by scoring.
- MDL/descriptive comparisons fail closed across incompatible coding geometry.
- Replay bundles hash every new evidence artifact.
- Fixture coverage includes the new production path and the legacy downgrade path.
- Documentation truthfulness tests describe the implemented math, not the old Gaussian proxy.
- Release, benchmark, performance, spec, fixture, and golden gates pass.
