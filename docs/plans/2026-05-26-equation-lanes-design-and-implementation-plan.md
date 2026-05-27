# Euclid Equation Lanes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an additive `equation_lanes` API/UI contract that always separates sample-exact descriptive reconstruction from predictive law search while preserving the existing `descriptive_reconstruction`, `predictive_law`, and `holistic_equation` fields.

**Architecture:** The workbench service remains the authority for normalized claim taxonomy. A new normalized `equation_lanes` object is built from the same source artifacts as the legacy fields, with `descriptive_exact` carrying sample-exact, full-sample, non-law reconstruction and `predictive_law_search` carrying the evidence-gated predictive search outcome. The frontend renders lanes from `equation_lanes` when present, falls back to legacy fields for saved analyses, and never promotes exact descriptive closure into `claim_class`, `publishable`, `predictive_law`, or `holistic_equation`.

**Tech Stack:** Python 3.11, NumPy FFT, pytest, vanilla JavaScript workbench assets, frontend tests under `tests/frontend/workbench/app.test.js`, Markdown reference docs.

---

## Status

- Date: 2026-05-26
- Scope: design spec and implementation plan only
- Approved approach: Approach A, add `equation_lanes` while preserving legacy fields
- Planning swarm: requested 7 planners; local agent thread cap allowed 6 spawns; one was closed to keep an odd 5-planner wave
- Jury target: 5 voting jurors because the local cap blocked a 7-juror wave; unanimity required
- Jury status: final readiness pass is complete. Revision 6 carries unanimous implementation readiness: backend-contract, exact-math, predictive-gate, frontend/API-semantics, and TDD/docs verdicts are all `APPROVE`.

## Current Repo Anchors

- `src/euclid/workbench/service.py:312` builds the current descriptive reconstruction from a retained-harmonic Fourier ladder.
- `src/euclid/workbench/service.py:1074` builds `predictive_law` only after operator publication, scorecard, claim-card, evidence, and banned-path gates pass.
- `src/euclid/workbench/service.py:1450` applies top-level claim taxonomy and publication status.
- `src/euclid/_assets/workbench/app.js:1811` assembles deterministic overlays for holistic, predictive, descriptive, benchmark-local, and point-lane paths.
- `tests/unit/workbench/test_service.py:372` currently asserts descriptive reconstruction is non-exact.
- `tests/unit/workbench/test_service.py:4880` verifies normalization can build a descriptive reconstruction claim.
- `tests/frontend/workbench/app.test.js:1673` verifies predictive law rendering when no holistic claim passes.
- `docs/workbench.md:47` documents normalized workbench surfaces and conservative interpretation.
- `docs/reference/search-core.md:70` documents that search breadth and publication eligibility are separate policies.

## Revision History

- Revision 0: initial Approach A spec and TDD task breakdown.
- Jury pass 1: five of five jurors returned `REVISE`.
- Revision 1: addressed scaled FFT tolerance, exactness edge tests, replay metadata, stale lane rebuilding, persisted normalized analyses, blocked predictive-search semantics, new exact-marker banned paths, stable frontend hierarchy, URL-state tests, docs assertions, and legacy fallback tests.
- Jury pass 2: exact-math and predictive-gate jurors returned `APPROVE`; backend-contract, frontend/API-semantics, and TDD/docs jurors returned `REVISE`.
- Revision 2: addressed invalid and legacy overlay URL tests, hero/ribbon selector coverage, no-law non-promotion, exact-lane copy restrictions, stale malicious `descriptive_exact` rebuilding, operator-point-missing blocked semantics, holistic-only gap-code filtering from predictive-search reasons, single-row exact-reconstruction blocking, sequential test-command ordering, fixture row-count validity, and unit placement for saved-analysis normalization tests.
- Jury pass 3: predictive-gate juror returned `APPROVE`; backend-contract, exact-math, frontend/API-semantics, and TDD/docs jurors returned `REVISE`.
- Revision 3: corrected executable snippets for import requirements, JSON-serialized FFT literal replay, `event_time` timestamp assertions, nonfinite parametrization, blocked-lane invariant fields, dataset-row availability semantics, long-history compatibility fixtures, stale-field absence assertions, publishable predictive frontend authority state, and holistic frontend authority refs.
- Jury pass 4: backend-contract, exact-math, frontend/API-semantics, and TDD/docs jurors returned `APPROVE`; predictive-gate juror returned `REVISE`.
- Revision 4: isolated the new exact descriptive marker bans by parametrizing predictive-law rejection for `exactness`, `candidate_id`, and `source` independently, and by replacing the holistic rejection snippet with an otherwise eligible backend joint-claim fixture whose only holistic blocker is the exact descriptive marker.
- Jury pass 5: backend-contract, exact-math, and predictive-gate jurors returned `APPROVE`; frontend/API-semantics juror returned `REVISE`; TDD/docs verdict was still pending when Revision 5 was applied.
- Revision 5: corrected the legacy overlay hydration fixture by adding a completed legacy `descriptive_reconstruction` payload before deleting `equation_lanes`, and identified the overlay-control binding/selector ambiguity that Revision 6 resolves with a both-attributes requirement.
- Revision 6: added final hardening requested by prior jurors: stale-lane assertions now reject malicious notes and `law_eligible` state, exact reconstruction must write the reconstructed chart/equation curve directly from the RFFT replay, failed operator artifacts remain blocked even when partial result metadata exists, holistic frontend coexistence explicitly clears default blockers and proves the predictive lane remains present but unselected, and holistic exact-marker rejection now asserts the exact marker is the isolated blocker rather than an incidental gate failure.
- Jury pass 6: a fresh five-member jury reran the full review against Revision 5. Backend-contract, exact-math, predictive-gate, frontend/API-semantics, and TDD/docs jurors all returned `APPROVE`, so Revision 6 records final implementation readiness.

## Problem Statement

The screenshot-driven user need is legitimate: a descriptive equation should be able to represent the observed data exactly when the stated job is in-sample description. The current workbench generated a dashed curve from a non-exact retained-harmonic ladder, so it showed the general shape but missed local turns.

The dangerous failure mode is also clear: a perfect in-sample curve can be mistaken for a predictive law. Euclid must support both:

1. An exact descriptive equation over the observed rows.
2. A separate predictive law search lane that only publishes a law when the existing evidence gates pass.

The implementation must make both visible without collapsing them into the same claim.

## Non-Goals

- Do not make exact reconstruction publishable.
- Do not weaken `predictive_law` or `holistic_equation` gates.
- Do not remove existing `descriptive_reconstruction`, `predictive_law`, or `holistic_equation` fields.
- Do not treat full-sample exactness as causal, mechanistic, market-valid, or forecast-valid evidence.
- Do not redesign the whole search portfolio.
- Do not use exact reconstruction as a hidden fallback when predictive law search fails.

## Lane Contract

`analysis["equation_lanes"]` is a normalized, UI-facing contract. It is additive and versioned.

```json
{
  "schema_version": "1.0.0",
  "source": "workbench_normalization",
  "lane_order": [
    "predictive_law_search",
    "descriptive_exact",
    "descriptive_fit"
  ],
  "descriptive_exact": {
    "status": "completed",
    "claim_class": "descriptive_reconstruction",
    "lane_kind": "descriptive_exact",
    "source": "workbench_descriptive_exact_reconstruction",
    "candidate_id": "descriptive_exact_fourier_reconstruction",
    "family_id": "spectral",
    "exactness": "sample_exact_reconstruction",
    "access_scope": "full_sample",
    "time_basis": "observed_row_index",
    "is_law_claim": false,
    "law_eligible": false,
    "publishable": false,
    "law_rejection_reason_codes": [
      "sample_exact_reconstruction_descriptive_only"
    ],
    "honesty_note": "Sample-exact reconstruction of observed rows only. It is descriptive, non-publishable, and not evidence of future behavior.",
    "equation": {
      "label": "y(t)=mean+full DFT reconstruction over observed row index",
      "candidate_id": "descriptive_exact_fourier_reconstruction",
      "family_id": "spectral",
      "render_status": "formula_supported",
      "curve": []
    },
    "chart": {
      "actual_series": [],
      "equation_curve": []
    },
    "reconstruction_metrics": {
      "sample_size": 19,
      "max_abs_error": 0.0,
      "normalized_max_abs_error": 0.0,
      "mae": 0.0,
      "normalized_mae": 0.0,
      "r2_vs_mean_baseline": 1.0,
      "exact_abs_floor": 1e-10,
      "exact_relative_factor": 1.4210854715202004e-14,
      "exact_scale": 1.45,
      "effective_exact_tolerance": 1e-10,
      "exact_tolerance_cleared": true
    }
  },
  "predictive_law_search": {
    "status": "no_publishable_law",
    "lane_kind": "predictive_law_search",
    "source": "operator_point_publication_gate",
    "publishable": false,
    "predictive_law": null,
    "candidate_id": null,
    "reason_codes": [],
    "evidence_summary": null,
    "honesty_note": "Predictive law search did not produce a publishable law under the declared validation scope."
  },
  "descriptive_fit": null
}
```

### Lane Key Semantics

- `equation_lanes.schema_version`: version of this display contract. Initial value is `1.0.0`.
- `lane_order`: display order, not claim rank. The UI may show predictive search first to answer "did we find a law?", but exact reconstruction must remain visibly descriptive-only.
- `descriptive_exact`: always built when target rows support numeric path reconstruction.
- `predictive_law_search`: always present and explicit. It can be `publishable_law`, `no_publishable_law`, or `blocked`.
- `descriptive_fit`: optional bridge to the existing benchmark-local descriptive fit when available.

### Predictive Search Status Values

- `publishable_law`: `predictive_law` exists and passed current gates.
- `no_publishable_law`: operator/search completed or abstained without a publishable law.
- `blocked`: predictive law state cannot be resolved because required artifacts are missing, malformed, stale, or failed.

`predictive_law_search.predictive_law` must either be the exact normalized `analysis["predictive_law"]` object or `null`. It must never contain `descriptive_exact`.

## Exact Descriptive Reconstruction

### Algorithm

Use the full discrete Fourier reconstruction over observed row index for numeric path targets.

For `N` observations `y[0], ..., y[N-1]`:

1. Compute `spectrum = np.fft.rfft(y)`.
2. Store all real FFT coefficients required by `np.fft.irfft(spectrum, n=N)`.
3. Build `equation_curve` from `irfft`.
4. Preserve observed timestamps by copying each source row time into the matching chart point.
5. Compute errors against the original observations.
6. Compute `exact_scale = max(1.0, max(abs(y_i)))`.
7. Compute `effective_exact_tolerance = max(1e-10, exact_scale * np.finfo(float).eps * max(64.0, 8.0 * N))`.
8. Mark exactness cleared only when `max_abs_error <= effective_exact_tolerance`.

This is exact over the observed row index, not exact over continuous time and not exact for unobserved future rows.

### Odd And Even Sample Handling

- Odd `N`: `np.fft.rfft` returns `(N // 2) + 1` complex bins.
- Even `N`: `np.fft.rfft` includes the Nyquist bin at index `N // 2`.
- The implementation must store `sample_size`, `fft_library`, `transform`, `inverse_transform`, `normalization`, `coefficient_order`, `rfft_real`, `rfft_imag`, `row_index_semantics`, and Nyquist metadata instead of relying on a truncated sine/cosine ladder.
- For even `N`, store `nyquist_bin_index = N // 2` and `nyquist_imag_abs`; tests must assert the Nyquist imaginary component is within the same effective tolerance.
- Tests must include odd `N=19` jagged data, even `N=20` or `N=48` data, `N=2`, a constant series, a large-magnitude finite series, nonfinite rows, irregular timestamps, and replay from stored literals.

### Target Eligibility

Return a completed `descriptive_exact` only when:

- at least two rows exist,
- every row has finite `observed_value`,
- the target is a numeric path target such as `price_close`, `daily_return`, or `log_return`.

Unsupported cases return a lane object with `status: "blocked"` and a concrete reason code:

- no dataset CSV or loadable rows: `dataset_rows_unavailable`
- fewer than two rows: `insufficient_rows_for_sample_exact_reconstruction`
- missing, null, `NaN`, or infinite observations: `nonfinite_observed_value`
- non-path target such as `next_day_up`: `target_not_numeric_path`
- FFT round-trip error above scaled tolerance: `sample_exact_tolerance_not_cleared`

### Required Invariants

- `descriptive_exact.law_eligible is False`.
- `descriptive_exact.publishable is False`.
- `descriptive_exact.exactness == "sample_exact_reconstruction"`.
- `sample_exact_reconstruction_descriptive_only` appears in law rejection reasons.
- `descriptive_exact` never changes `analysis["claim_class"]` above `descriptive_reconstruction`.
- `descriptive_exact` never appears inside `predictive_law`, `holistic_equation`, or `uncertainty_attachment`.
- If exactness tolerance does not clear, the lane must be `blocked` and must not be displayed as exact.
- Any incoming saved `equation_lanes` payload is discarded during normalization and rebuilt from normalized top-level truth.

## Predictive Law Search Lane

`predictive_law_search` is a display wrapper around existing operator publication truth, not a new law publisher.

### Publishable Case

When `_build_predictive_law(...)` returns an object:

- `predictive_law_search.status = "publishable_law"`
- `predictive_law_search.publishable = true`
- `predictive_law_search.predictive_law = analysis["predictive_law"]`
- `predictive_law_search.evidence_summary = analysis["predictive_law"]["evidence_summary"]`
- `predictive_law_search.reason_codes = []`

### No Publishable Law Case

When operator/search artifacts exist and the operator reached a resolved abstained, candidate-only, or non-publishable state:

- `predictive_law_search.status = "no_publishable_law"`
- `predictive_law_search.publishable = false`
- `predictive_law_search.predictive_law = null`
- `predictive_law_search.reason_codes` uses dedicated predictive-law reasons: operator abstention reason codes, predictive-law banned-path reason codes, non-publishable publication status, scorecard/claim-card predictive support failures, and `no_publishable_predictive_law`.
- Holistic-only blockers such as `no_backend_joint_claim` must not be copied into `predictive_law_search.reason_codes` unless they are explicitly mapped to a predictive-law reason.

### Blocked Case

When predictive state is missing or malformed:

- `predictive_law_search.status = "blocked"`
- `predictive_law_search.publishable = false`
- `predictive_law_search.predictive_law = null`
- `predictive_law_search.reason_codes` includes concrete missing-artifact codes.
- Missing or malformed `operator_point` must produce `blocked`, not `no_publishable_law`.

### Required Invariants

- Exact descriptive reconstruction cannot satisfy or improve predictive law search.
- Existing banned paths still block predictive law and holistic equation promotion:
  - exact closure,
  - sample-exact reconstruction,
  - `descriptive_exact_fourier_reconstruction`,
  - `workbench_descriptive_exact_reconstruction`,
  - posthoc symbolic synthesis,
  - residual wrappers where banned,
  - stale saved predictive payloads without current operator gates,
  - benchmark-local descriptive fit promotion.
- Search-core language remains intact: exactness over a declared finite space is not global predictive truth, and heuristic search requires omission disclosure.

## Legacy Compatibility

`equation_lanes` does not replace existing fields in the first implementation wave.

- `analysis["descriptive_reconstruction"]` remains the current legacy non-exact retained-harmonic reconstruction for legacy clients.
- `analysis["descriptive_reconstruction"]` must not be repointed to the exact helper in this wave.
- `analysis["predictive_law"]` remains present or `None` exactly as current gates decide.
- `analysis["holistic_equation"]` remains present only when current joint-gate logic permits it.
- Saved analyses without `equation_lanes` continue to render through existing fallback paths.
- New saved analyses include both top-level legacy fields and `equation_lanes`.
- `create_workbench_analysis(...)` must persist the normalized payload containing `equation_lanes` to `analysis.json`, or the server must rewrite the normalized payload before returning. The implementation plan chooses persistence from `create_workbench_analysis(...)` so direct callers and HTTP callers see the same saved contract.
- `normalize_analysis_payload(...)` must discard stale incoming `equation_lanes` before rebuilding lanes.

## Frontend Behavior

### Display Rules

- The UI reads `analysis.equation_lanes` first when present.
- If missing, it falls back to legacy `analysis.predictive_law`, `analysis.descriptive_reconstruction`, and `analysis.descriptive_fit`.
- The overview must show both:
  - exact descriptive representation of observed rows,
  - predictive law search result.
- The exact lane label must avoid "law" and "perfect" unless bounded by "observed samples".
- The predictive lane must show an explicit no-law state instead of disappearing.
- Lane cards must have stable selectors, for example `data-equation-lane-card="descriptive_exact"` and `data-equation-lane-card="predictive_law_search"`.

### Overlay Rules

- `descriptive_exact` may be selectable as an overlay.
- `predictive_law_search` is a lane/card wrapper, not a standalone overlay id. When it contains a publishable predictive law with a curve, the UI must expose that curve through the existing `predictive_law` overlay id so URL state and legacy overlay behavior do not fork.
- Default overlay precedence is: valid `holistic_equation`, publishable `predictive_law`, completed `descriptive_exact`, legacy `descriptive_reconstruction`, legacy `descriptive_fit`, point path, probabilistic mean. Existing holistic-default behavior remains intact.
- URL state for selected overlay must remain stable.
- `?overlay=descriptive_exact` must hydrate to the exact descriptive overlay when available.
- Clicking the exact descriptive overlay must write `overlay=descriptive_exact` to query state.
- Invalid stale overlay query values must fall back to default precedence.
- Legacy `overlay=descriptive_reconstruction` must continue to work when `equation_lanes` is absent.

### Copy Rules

Allowed phrases:

- "Sample-exact descriptive reconstruction"
- "Observed rows only"
- "Predictive law search found no publishable law"
- "Publishable within declared validation scope"

Forbidden or unsafe phrases unless immediately bounded:

- "perfect law"
- "perfect equation" without "for observed samples"
- "predictive" attached to `descriptive_exact`
- "operator publication" attached to `descriptive_exact`

## Failure Modes And Required Mitigations

| Failure mode | Mitigation |
| --- | --- |
| Exact fit is mistaken for prediction | `law_eligible: false`, `publishable: false`, UI honesty note, docs language, tests |
| Odd/even Fourier off-by-one | Odd 19-row and even-row unit tests |
| Legacy clients break | Preserve top-level fields and fallback frontend render path |
| `equation_lanes` drifts from legacy fields | Build lanes after legacy normalization from normalized fields |
| No predictive law is hidden | First-class `predictive_law_search.status = "no_publishable_law"` |
| Stale saved predictive payload outranks normalized truth | Build lanes only after `_build_predictive_law` and `_build_holistic_equation` gates |
| Tests cover smooth data only | Add jagged 19-row fixture from screenshot shape |
| UI overlay implies strongest claim | Frontend tests for card order, labels, and absence of predictive copy |
| Payload grows too much | Store coefficient arrays but keep labels summarized; future wave may compress large arrays |

## Test Strategy

Every behavior change starts with a failing test. Existing failing tests count as red tests only after they are re-run and the failure proves the intended missing behavior.

Targeted commands:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py
npm run test:frontend -- --run tests/frontend/workbench/app.test.js
```

Broader verification after targeted green:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_workbench_analysis.py tests/unit/workbench/test_service.py
npm run test:frontend -- --run tests/frontend/workbench/app.test.js tests/frontend/workbench-ui.test.js
```

Final release-adjacent smoke only after implementation is stable:

```bash
./scripts/release_smoke.sh
```

## Implementation Ledger

This section is the durable progress record for implementation. Update it before
and after each task batch so the spec remains the source of truth.

| Batch | Scope | Status | Files changed | Commands run | Swarm verdict | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | Progress ledger and implementation cadence | Completed | `docs/plans/2026-05-26-equation-lanes-design-and-implementation-plan.md` | Planning swarm read-only review | Planning swarm consensus: backend first, single integrator, validation swarms after task pairs | Added ledger before code edits. |
| 1 | Task 1-2 exact descriptive reconstruction helper | Completed | `tests/unit/workbench/test_service.py`, `src/euclid/workbench/service.py` | RED: `PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_matches_jagged_odd_rows` failed on missing `_build_descriptive_exact_reconstruction`; GREEN: `PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_matches_jagged_odd_rows tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_records_even_sample_rfft_metadata tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_supports_minimum_two_rows tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_handles_constant_rows tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_uses_scaled_tolerance_for_large_values tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_blocks_nonfinite_rows tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_preserves_irregular_row_times tests/unit/workbench/test_service.py::test_descriptive_exact_reconstruction_literals_replay_curve` passed `11 passed` | `APPROVE` from exact-math, backend-test, and claim-boundary jurors | Implemented direct RFFT replay helper; legacy retained-harmonic helper unchanged. |
| 2 | Task 3-4 equation lane normalization | Completed | `tests/unit/workbench/test_service.py`, `src/euclid/workbench/service.py` | RED: first Task 3 test failed with `KeyError: 'equation_lanes'`; RED: remaining seven Task 3 tests failed on missing lanes/stale incoming lane preservation; GREEN: `PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py::test_normalize_analysis_payload_adds_equation_lanes_with_exact_descriptive_and_no_law tests/unit/workbench/test_service.py::test_equation_lanes_wrap_publishable_predictive_law_without_replacing_legacy_fields tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rebuilds_stale_incoming_equation_lanes tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_when_operator_point_missing tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_for_malformed_operator_point tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_for_failed_operator_even_with_result_mode tests/unit/workbench/test_service.py::test_descriptive_exact_does_not_change_claim_surface_or_joint_artifacts tests/unit/workbench/test_service.py::test_predictive_law_search_reasons_exclude_holistic_only_gap_codes` passed `8 passed` | `APPROVE` from backend-contract, claim-boundary, and test-executability validators | Added rebuilt `equation_lanes`, exact descriptive lane wrapper, predictive law search lane states, stale lane discard, and holistic-only gap filtering. |
| 3 | Task 7 target eligibility and exact-marker rejection | Completed | `tests/unit/workbench/test_service.py`, `src/euclid/workbench/service.py` | RED: `PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py::test_equation_lanes_mark_exact_reconstruction_not_applicable_for_next_day_up tests/unit/workbench/test_service.py::test_equation_lanes_mark_exact_reconstruction_blocked_without_dataset_rows tests/unit/workbench/test_service.py::test_equation_lanes_mark_exact_reconstruction_blocked_for_single_row tests/unit/workbench/test_service.py::test_equation_lanes_do_not_turn_exact_descriptive_lane_into_predictive_law tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_publishable_exact_descriptive_operator_point_for_predictive_law tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_exact_descriptive_holistic_equation` failed `6 failed, 4 passed` on exact-marker smuggling; GREEN: same command passed `10 passed` | `APPROVE` from target-eligibility, gap-semantics, and regression-risk validators | Added target eligibility tests and independent exact-descriptive marker rejection for operator and holistic equations. |
| 4 | Task 5-6 frontend lane rendering | Completed | `tests/frontend/workbench/app.test.js`, `src/euclid/_assets/workbench/app.js` | RED: `npm run test:frontend -- --run tests/frontend/workbench/app.test.js` initially failed on same-origin URL harness regression, then failed `10 failed | 34 passed` on missing lane cards/selectors; GREEN: same command passed `44 passed`; copy fix targeted run passed `3 passed | 41 skipped`; final full run passed `44 passed` | `APPROVE` from lane-rendering, claim-boundary-copy, and regression-risk validators after one copy revision | Added lane fixtures/tests, exact descriptive overlay, lane cards, stable overlay selectors, hero/ribbon source ids, query-state hydration, and bounded exact-lane UI copy. |
| 5 | Task 8 docs truthfulness | Completed | `tests/spec_compiler/test_math_documentation_truthfulness.py`, `docs/workbench.md`, `docs/reference/workbench.md` | RED: `PYTEST_ADDOPTS="" PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_math_documentation_truthfulness.py::test_workbench_docs_define_equation_lanes_without_predictive_promotion` failed on missing `equation_lanes`; GREEN: same command passed `1 passed`; `PYTEST_ADDOPTS="" PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_math_documentation_truthfulness.py` passed `8 passed`; `PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py` passed `96 passed` | `APPROVE` from docs-truthfulness, doc/test-scope, and implementation-alignment validators | Documented lane contract, exact descriptive boundary, predictive law search outcomes, overlay behavior, and legacy fallback. |
| 6 | Task 9 persistence compatibility | Completed | `tests/integration/test_workbench_analysis.py`, `tests/unit/workbench/test_service.py`, `src/euclid/workbench/service.py` | RED: `PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_workbench_analysis.py::test_create_workbench_analysis_persists_equation_lanes tests/unit/workbench/test_service.py::test_saved_analysis_without_equation_lanes_rebuilds_lanes_from_legacy_fields` failed `2 failed` on missing persisted lanes and legacy fixture mismatch; GREEN: same command passed `2 passed`; post-review regression run `PYTHONPATH=src python3.11 -m pytest -q tests/integration/test_workbench_analysis.py::test_create_workbench_analysis_persists_equation_lanes tests/integration/test_workbench_analysis.py::test_saved_analysis_reload_preserves_no_winner_daily_return_fixture tests/integration/test_workbench_analysis.py::test_create_workbench_analysis_runs_real_runtime_with_fixture_history tests/unit/workbench/test_service.py::test_saved_analysis_without_equation_lanes_rebuilds_lanes_from_legacy_fields tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rebuilds_stale_incoming_equation_lanes tests/unit/workbench/test_service.py::test_descriptive_exact_does_not_change_claim_surface_or_joint_artifacts tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_publishable_exact_descriptive_operator_point_for_predictive_law tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_exact_descriptive_holistic_equation` passed `15 passed` | `APPROVE` from persistence-contract, legacy-compatibility, and regression-risk validators after two revise fixes | Persisted normalized lane payloads from `create_workbench_analysis`, added server reload lane assertions, restored saved-analysis gap ordering, rejected exact-descriptive markers at both top-level and nested holistic payload positions, and updated runtime integration expectations for normalized descriptive-fit floor failures. |
| 7 | Task 10 final verification | Completed | `docs/plans/2026-05-26-equation-lanes-design-and-implementation-plan.md`, rendered evidence `euclid-equation-lanes-rendered-smoke-2026-05-27.png` | `PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py tests/integration/test_workbench_analysis.py` passed `112 passed`; `npm run test:frontend -- --run tests/frontend/workbench/app.test.js tests/frontend/workbench-ui.test.js` passed `48 passed`; `PYTEST_ADDOPTS="" PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_math_documentation_truthfulness.py` passed `8 passed`; `PYTHONPATH=src python3.11 scripts/workbench_ui_smoke.py --output-root /private/tmp/euclid-workbench-render-smoke --port 8765 --live-delay-seconds 0` passed API smoke; Playwright rendered `http://127.0.0.1:8765` with exact descriptive hero/overlay, separate predictive law search lane card, and only favicon 404 console noise; `git diff --check` passed; optional `./scripts/release_smoke.sh` was attempted but stopped after over one hour without producing terminal output. Existing `build/reports/repo_test_matrix.json` reports `1356 passed, 11 warnings in 1944.78s (0:32:24)` for source digest `repo_checkout_digest:3c9bcef0e9c0ac97cc6215f55a9800709563081e635cd86c1285d1ea3e03c8f2`; the current source digest after this patch set is `repo_checkout_digest:db54fc7a1dc4a689b838fd9b3aef720811d2c04bdedef3c7aebf642a1628ab13`, so that matrix is historical, not current final evidence. | Post-implementation final audit: backend/API `APPROVE`, frontend/docs `APPROVE`, completion-ledger `APPROVE` after one ledger-only revision | Required backend, frontend, docs, API smoke, rendered UI, and diff hygiene gates are green; full release-adjacent smoke remains explicitly not counted as current evidence because the only available repo matrix is stale to the current source digest. |

## Implementation Tasks

### Task 1: Backend Red Tests For Exact Descriptive Lane

**Files:**
- Modify: `tests/unit/workbench/test_service.py`
- Read: `src/euclid/workbench/service.py:312-512`

**Step 0: Add test imports required by the new snippets**

In `tests/unit/workbench/test_service.py`, add:

```python
import json

import numpy as np
```

and add `_build_descriptive_exact_reconstruction` to the existing
`from euclid.workbench.service import (...)` list. These imports are part of the
red-test setup, not implementation.

**Step 1: Write the failing odd-sample exactness test**

Add a test near `test_build_descriptive_reconstruction_is_descriptive_structure_and_non_exact`:

```python
def test_build_descriptive_exact_reconstruction_matches_jagged_odd_rows() -> None:
    values = [
        0.20, 1.10, 0.72, 0.28, -0.08, 0.55, 0.95, 1.05, -1.45,
        -0.52, -0.10, -0.72, 1.35, 0.30, 0.46, 0.62, -0.18, -0.35, 0.08,
    ]
    dataset_rows = _workbench_dataset_rows(values)

    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=dataset_rows,
    )

    assert reconstruction is not None
    assert reconstruction["status"] == "completed"
    assert reconstruction["exactness"] == "sample_exact_reconstruction"
    assert reconstruction["law_eligible"] is False
    assert reconstruction["publishable"] is False
    assert reconstruction["reconstruction_metrics"]["sample_size"] == 19
    assert reconstruction["reconstruction_metrics"]["exact_tolerance_cleared"] is True
    metrics = reconstruction["reconstruction_metrics"]
    assert metrics["max_abs_error"] <= metrics["effective_exact_tolerance"]
    fitted = [
        point["fitted_value"]
        for point in reconstruction["chart"]["equation_curve"]
    ]
    assert fitted == pytest.approx(values, abs=metrics["effective_exact_tolerance"])
```

**Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_matches_jagged_odd_rows
```

Expected: FAIL because `_build_descriptive_exact_reconstruction` does not exist.

**Step 3: Write the failing even-sample coefficient metadata test**

```python
def test_build_descriptive_exact_reconstruction_records_even_sample_rfft_metadata() -> None:
    values = [
        math.sin((2.0 * math.pi * index) / 20.0) + (0.1 * index)
        for index in range(20)
    ]
    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=_workbench_dataset_rows(values),
    )

    literals = reconstruction["equation"]["literals"]

    assert literals["sample_size"] == 20
    assert literals["basis"] == "rfft_full_sample"
    assert literals["fft_library"] == "numpy.fft"
    assert literals["transform"] == "rfft"
    assert literals["inverse_transform"] == "irfft"
    assert literals["normalization"] == "numpy_default_backward"
    assert literals["coefficient_order"] == "rfft_frequency_bin_order"
    assert len(literals["rfft_real"]) == 11
    assert len(literals["rfft_imag"]) == 11
    assert literals["nyquist_bin_index"] == 10
    assert literals["nyquist_imag_abs"] <= reconstruction["reconstruction_metrics"]["effective_exact_tolerance"]
    assert reconstruction["reconstruction_metrics"]["max_abs_error"] <= reconstruction["reconstruction_metrics"]["effective_exact_tolerance"]
```

**Step 4: Write edge-case exactness tests required by the math jury**

Add these named red tests in the same file:

```python
def test_build_descriptive_exact_reconstruction_supports_minimum_two_rows() -> None:
    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=_workbench_dataset_rows([2.0, -3.0]),
    )

    assert reconstruction["status"] == "completed"
    assert reconstruction["reconstruction_metrics"]["sample_size"] == 2
    assert reconstruction["reconstruction_metrics"]["max_abs_error"] <= reconstruction["reconstruction_metrics"]["effective_exact_tolerance"]


def test_build_descriptive_exact_reconstruction_handles_constant_rows() -> None:
    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=_workbench_dataset_rows([4.25] * 12),
    )

    assert reconstruction["status"] == "completed"
    assert reconstruction["reconstruction_metrics"]["r2_vs_mean_baseline"] == 1.0
    assert reconstruction["reconstruction_metrics"]["max_abs_error"] <= reconstruction["reconstruction_metrics"]["effective_exact_tolerance"]


def test_build_descriptive_exact_reconstruction_uses_scaled_tolerance_for_large_values() -> None:
    values = [1.0e12 + ((-1) ** index) * (index * 123.0) for index in range(19)]

    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=_workbench_dataset_rows(values),
    )

    metrics = reconstruction["reconstruction_metrics"]
    assert metrics["max_abs_error"] > 1e-10
    assert metrics["max_abs_error"] <= metrics["effective_exact_tolerance"]


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf"), float("-inf")])
def test_build_descriptive_exact_reconstruction_blocks_nonfinite_rows(
    bad_value: float | None,
) -> None:
    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=_workbench_dataset_rows([1.0, bad_value, 3.0]),
    )

    assert reconstruction["status"] == "blocked"
    assert "nonfinite_observed_value" in reconstruction["reason_codes"]
    assert reconstruction["law_eligible"] is False
    assert reconstruction["publishable"] is False


def test_build_descriptive_exact_reconstruction_preserves_irregular_row_times() -> None:
    rows = _workbench_dataset_rows([1.0, -0.5, 2.25, 0.0])
    rows[1]["event_time"] = "2026-05-03T00:00:00Z"
    rows[2]["event_time"] = "2026-05-20T00:00:00Z"

    reconstruction = _build_descriptive_exact_reconstruction(dataset_rows=rows)

    assert [
        point["event_time"]
        for point in reconstruction["chart"]["equation_curve"]
    ] == [
        point["event_time"]
        for point in reconstruction["chart"]["actual_series"]
    ]
```

**Step 5: Write stored-literal replay test**

```python
def test_descriptive_exact_reconstruction_literals_replay_curve() -> None:
    values = [0.2, -1.0, 0.4, 1.5, -0.7, 0.3, 0.0]
    reconstruction = _build_descriptive_exact_reconstruction(
        dataset_rows=_workbench_dataset_rows(values),
    )
    literals = json.loads(json.dumps(reconstruction["equation"]["literals"]))

    spectrum = np.asarray(literals["rfft_real"]) + (
        1j * np.asarray(literals["rfft_imag"])
    )
    replayed = np.fft.irfft(spectrum, n=literals["sample_size"])

    tolerance = reconstruction["reconstruction_metrics"]["effective_exact_tolerance"]
    curve_values = [
        point["fitted_value"]
        for point in reconstruction["chart"]["equation_curve"]
    ]
    observed_values = [
        point["observed_value"]
        for point in reconstruction["chart"]["actual_series"]
    ]

    assert replayed.tolist() == pytest.approx(curve_values, abs=tolerance)
    assert replayed.tolist() == pytest.approx(observed_values, abs=tolerance)
```

**Step 6: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_records_even_sample_rfft_metadata \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_supports_minimum_two_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_handles_constant_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_uses_scaled_tolerance_for_large_values \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_blocks_nonfinite_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_preserves_irregular_row_times \
  tests/unit/workbench/test_service.py::test_descriptive_exact_reconstruction_literals_replay_curve
```

Expected: FAIL because the helper, blocked statuses, scaled tolerance, and replay metadata do not exist.

**Step 7: Commit after green**

After implementation in Task 2 and green tests:

```bash
git add tests/unit/workbench/test_service.py src/euclid/workbench/service.py
git commit -m "feat: add exact descriptive reconstruction lane"
```

### Task 2: Implement Exact Descriptive Reconstruction Helper

**Files:**
- Modify: `src/euclid/workbench/service.py:312-512`
- Test: `tests/unit/workbench/test_service.py`

**Step 1: Implement the minimal helper**

Add `_build_descriptive_exact_reconstruction(...)` beside `_build_descriptive_reconstruction(...)`.

```python
def _build_descriptive_exact_reconstruction(
    *,
    dataset_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    if len(dataset_rows) < 2:
        return {
            "status": "blocked",
            "claim_class": "descriptive_reconstruction",
            "lane_kind": "descriptive_exact",
            "exactness": "sample_exact_reconstruction",
            "law_eligible": False,
            "publishable": False,
            "law_rejection_reason_codes": [
                "sample_exact_reconstruction_descriptive_only"
            ],
            "reason_codes": ["insufficient_rows_for_sample_exact_reconstruction"],
        }
    observed_values = [
        _float_or_none(row.get("observed_value"))
        for row in dataset_rows
    ]
    if any(value is None or not math.isfinite(float(value)) for value in observed_values):
        return {
            "status": "blocked",
            "claim_class": "descriptive_reconstruction",
            "lane_kind": "descriptive_exact",
            "exactness": "sample_exact_reconstruction",
            "law_eligible": False,
            "publishable": False,
            "law_rejection_reason_codes": [
                "sample_exact_reconstruction_descriptive_only"
            ],
            "reason_codes": ["nonfinite_observed_value"],
        }
    values = np.asarray([float(value) for value in observed_values], dtype=float)
    sample_size = int(values.size)
    spectrum = np.fft.rfft(values)
    reconstructed = np.fft.irfft(spectrum, n=sample_size)
    errors = reconstructed - values
    max_abs_error = float(np.max(np.abs(errors))) if sample_size else 0.0
    mae = float(np.mean(np.abs(errors))) if sample_size else 0.0
    exact_scale = float(max(1.0, np.max(np.abs(values))))
    exact_abs_floor = 1e-10
    exact_relative_factor = float(np.finfo(float).eps * max(64.0, 8.0 * sample_size))
    effective_exact_tolerance = max(
        exact_abs_floor,
        exact_scale * exact_relative_factor,
    )
    exact_cleared = max_abs_error <= effective_exact_tolerance
    if not exact_cleared:
        return {
            "status": "blocked",
            "claim_class": "descriptive_reconstruction",
            "lane_kind": "descriptive_exact",
            "exactness": "sample_exact_reconstruction",
            "law_eligible": False,
            "publishable": False,
            "law_rejection_reason_codes": [
                "sample_exact_reconstruction_descriptive_only"
            ],
            "reason_codes": ["sample_exact_tolerance_not_cleared"],
            "reconstruction_metrics": {
                "sample_size": sample_size,
                "max_abs_error": max_abs_error,
                "mae": mae,
                "exact_abs_floor": exact_abs_floor,
                "exact_relative_factor": exact_relative_factor,
                "exact_scale": exact_scale,
                "effective_exact_tolerance": effective_exact_tolerance,
                "exact_tolerance_cleared": False,
            },
        }
    # Build equation and chart using existing helper conventions.
```

Complete the implementation with:

- `candidate_id = "descriptive_exact_fourier_reconstruction"`
- `family_id = "spectral"`
- `literals["basis"] = "rfft_full_sample"`
- `literals["fft_library"] = "numpy.fft"`
- `literals["transform"] = "rfft"`
- `literals["inverse_transform"] = "irfft"`
- `literals["normalization"] = "numpy_default_backward"`
- `literals["coefficient_order"] = "rfft_frequency_bin_order"`
- `literals["sample_size"] = sample_size`
- `literals["rfft_real"] = [float(value) for value in spectrum.real]`
- `literals["rfft_imag"] = [float(value) for value in spectrum.imag]`
- `literals["nyquist_bin_index"] = sample_size // 2 if sample_size % 2 == 0 else None`
- `literals["nyquist_imag_abs"] = abs(float(spectrum[sample_size // 2].imag)) if sample_size % 2 == 0 else None`
- `literals["row_index_semantics"] = "observed_row_order_no_calendar_interpolation"`
- `equation["curve"]` containing the reconstructed values with the same date/time labels as `_actual_series(...)`
- The exact helper must write `equation["curve"]` and `chart["equation_curve"]` directly from the `np.fft.irfft(...)` replay. Do not route this path through any existing truncated harmonic ladder or generic curve renderer that can drop bins, interpolate calendar time, or recompute a non-exact curve from the formula label.
- metrics from `_descriptive_fit_reconstruction_metrics(...)` plus exact fields, or direct exact metrics if that helper is too coupled

**Step 2: Preserve the old helper temporarily**

Do not delete `_build_descriptive_reconstruction(...)` in this task. It remains the legacy compatibility field until lane normalization is complete.

**Step 3: Run targeted backend tests**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_matches_jagged_odd_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_records_even_sample_rfft_metadata \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_supports_minimum_two_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_handles_constant_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_uses_scaled_tolerance_for_large_values \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_blocks_nonfinite_rows \
  tests/unit/workbench/test_service.py::test_build_descriptive_exact_reconstruction_preserves_irregular_row_times \
  tests/unit/workbench/test_service.py::test_descriptive_exact_reconstruction_literals_replay_curve
```

Expected: PASS.

**Step 4: Refactor only if green**

If exact curve construction duplicates existing chart code, extract a tiny private helper. Do not refactor predictive law gates.

### Task 3: Backend Red Tests For `equation_lanes`

**Files:**
- Modify: `tests/unit/workbench/test_service.py`
- Read: `src/euclid/workbench/service.py:637-804`
- Read: `src/euclid/workbench/service.py:1450-1565`

**Step 1: Write failing normalization test for no publishable law**

```python
def test_normalize_analysis_payload_adds_equation_lanes_with_exact_descriptive_and_no_law(
    tmp_path: Path,
) -> None:
    values = [
        0.20, 1.10, 0.72, 0.28, -0.08, 0.55, 0.95, 1.05, -1.45,
        -0.52, -0.10, -0.72, 1.35, 0.30, 0.46, 0.62, -0.18, -0.35, 0.08,
    ]
    dataset_csv = _write_dataset_csv(
        tmp_path / "jagged-descriptive-exact.csv",
        _workbench_dataset_rows(values),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "abstained",
            "abstention": {"reason_codes": ["predictive_support_failed"]},
        },
    }

    normalized = normalize_analysis_payload(payload)
    lanes = normalized["equation_lanes"]

    assert lanes["schema_version"] == "1.0.0"
    assert lanes["source"] == "workbench_normalization"
    assert lanes["lane_order"] == [
        "predictive_law_search",
        "descriptive_exact",
        "descriptive_fit",
    ]
    assert lanes["descriptive_exact"]["lane_kind"] == "descriptive_exact"
    assert lanes["predictive_law_search"]["lane_kind"] == "predictive_law_search"
    assert lanes["descriptive_exact"]["status"] == "completed"
    assert lanes["descriptive_exact"]["law_eligible"] is False
    assert lanes["descriptive_exact"]["publishable"] is False
    assert (
        lanes["descriptive_exact"]["reconstruction_metrics"]["max_abs_error"]
        <= lanes["descriptive_exact"]["reconstruction_metrics"]["effective_exact_tolerance"]
    )
    assert lanes["predictive_law_search"]["status"] == "no_publishable_law"
    assert lanes["predictive_law_search"]["publishable"] is False
    assert lanes["predictive_law_search"]["predictive_law"] is None
    assert normalized["predictive_law"] is None
    assert normalized["publishable"] is False
```

**Step 2: Run test to verify it fails**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py::test_normalize_analysis_payload_adds_equation_lanes_with_exact_descriptive_and_no_law
```

Expected: FAIL because `equation_lanes` is missing.

**Step 3: Write failing publishable-law coexistence test**

Use `_predictive_law_candidate_publication_payload(...)` already present in `tests/unit/workbench/test_service.py`.

```python
def test_equation_lanes_wrap_publishable_predictive_law_without_replacing_legacy_fields(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path=tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "curve": [
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.2},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 10.4},
            ],
        },
    )
    dataset_csv = _write_dataset_csv(
        tmp_path / "spy-price-close-long-history.csv",
        _workbench_dataset_rows(
            [10.0, 11.2, 10.7, 11.4, 11.9, 12.2, 11.8, 12.6, 12.1]
        ),
    )
    payload["dataset"]["dataset_csv"] = str(dataset_csv)

    normalized = normalize_analysis_payload(payload)
    lanes = normalized["equation_lanes"]

    assert normalized["predictive_law"] is not None
    assert lanes["source"] == "workbench_normalization"
    assert lanes["predictive_law_search"]["status"] == "publishable_law"
    assert lanes["predictive_law_search"]["publishable"] is True
    assert lanes["predictive_law_search"]["lane_kind"] == "predictive_law_search"
    assert lanes["predictive_law_search"]["predictive_law"] == normalized["predictive_law"]
    assert lanes["descriptive_exact"]["law_eligible"] is False
    assert lanes["descriptive_exact"]["lane_kind"] == "descriptive_exact"
    assert normalized["descriptive_reconstruction"]["equation"]["candidate_id"] == "descriptive_fourier_reconstruction"
    assert lanes["descriptive_exact"]["equation"]["candidate_id"] == "descriptive_exact_fourier_reconstruction"
    assert normalized["claim_class"] == "predictive_law"
```

**Step 4: Write stale-lane and blocked-search red tests**

```python
def test_normalize_analysis_payload_rebuilds_stale_incoming_equation_lanes(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "stale-lanes.csv",
        _workbench_dataset_rows([1.0, -0.5, 0.25, 0.75]),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {"status": "failed", "error": "fixture"},
        "predictive_law": None,
        "equation_lanes": {
            "descriptive_exact": {
                "status": "completed",
                "candidate_id": "stale_exact_publishable",
                "publishable": True,
                "law_eligible": True,
                "honesty_note": "stale malicious exact lane",
            },
            "predictive_law_search": {
                "status": "publishable_law",
                "publishable": True,
                "predictive_law": {"claim_class": "predictive_law"},
            }
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["equation_lanes"]["predictive_law_search"]["status"] == "blocked"
    assert normalized["equation_lanes"]["predictive_law_search"]["publishable"] is False
    assert (
        normalized["equation_lanes"]["descriptive_exact"]["equation"]["candidate_id"]
        == "descriptive_exact_fourier_reconstruction"
    )
    assert "stale_exact_publishable" not in str(normalized["equation_lanes"])
    assert "stale malicious exact lane" not in str(normalized["equation_lanes"])
    assert normalized["equation_lanes"]["descriptive_exact"]["law_eligible"] is False
    assert normalized["equation_lanes"]["descriptive_exact"]["publishable"] is False


def test_equation_lanes_mark_predictive_search_blocked_when_operator_point_missing(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "missing-operator.csv",
        _workbench_dataset_rows([1.0, 2.0, 3.0]),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["equation_lanes"]["predictive_law_search"]["status"] == "blocked"
    assert "operator_point_missing" in normalized["equation_lanes"]["predictive_law_search"]["reason_codes"]


def test_equation_lanes_mark_predictive_search_blocked_for_malformed_operator_point(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "malformed-operator.csv",
        _workbench_dataset_rows([1.0, 2.0, 3.0]),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {"status": "completed"},
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["equation_lanes"]["predictive_law_search"]["status"] == "blocked"
    assert "operator_point_result_mode_missing" in normalized["equation_lanes"]["predictive_law_search"]["reason_codes"]


def test_equation_lanes_mark_predictive_search_blocked_for_failed_operator_even_with_result_mode(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "failed-operator-with-result-mode.csv",
        _workbench_dataset_rows([1.0, 2.0, 3.0]),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {
            "status": "failed",
            "result_mode": "candidate_publication",
            "error": "fixture failure after partial metadata",
        },
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["equation_lanes"]["predictive_law_search"]["status"] == "blocked"
    assert "operator_point_failed" in normalized["equation_lanes"]["predictive_law_search"]["reason_codes"]


def test_descriptive_exact_does_not_change_claim_surface_or_joint_artifacts(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "descriptive-only.csv",
        _workbench_dataset_rows([1.0, 0.5, -0.25, 0.75, 1.25, 0.0, -0.5, 0.4, 0.9]),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {"status": "failed", "error": "fixture"},
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["publishable"] is False
    assert normalized["holistic_equation"] is None
    assert normalized.get("uncertainty_attachment") is None
    assert normalized["evidence_studio"]["claim_surface"]["claim_lane"] == "descriptive"
    assert normalized["descriptive_reconstruction"]["equation"]["candidate_id"] == "descriptive_fourier_reconstruction"
    assert normalized["equation_lanes"]["descriptive_exact"]["equation"]["candidate_id"] == "descriptive_exact_fourier_reconstruction"
    assert normalized["equation_lanes"]["descriptive_exact"]["publishable"] is False


def test_predictive_law_search_reasons_exclude_holistic_only_gap_codes(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "holistic-gap.csv",
        _workbench_dataset_rows([1.0, -0.5, 0.25, 0.75]),
    )
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {
            "status": "completed",
            "result_mode": "abstained",
            "abstention": {"reason_codes": ["predictive_support_failed"]},
        },
        "gap_report": ["no_backend_joint_claim", "predictive_support_failed"],
    }

    normalized = normalize_analysis_payload(payload)

    reasons = normalized["equation_lanes"]["predictive_law_search"]["reason_codes"]
    assert "predictive_support_failed" in reasons
    assert "no_backend_joint_claim" not in reasons
```

**Step 5: Run tests to verify they fail**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/workbench/test_service.py::test_equation_lanes_wrap_publishable_predictive_law_without_replacing_legacy_fields \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rebuilds_stale_incoming_equation_lanes \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_when_operator_point_missing \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_for_malformed_operator_point \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_for_failed_operator_even_with_result_mode \
  tests/unit/workbench/test_service.py::test_descriptive_exact_does_not_change_claim_surface_or_joint_artifacts \
  tests/unit/workbench/test_service.py::test_predictive_law_search_reasons_exclude_holistic_only_gap_codes
```

Expected: FAIL because `equation_lanes`, stale-lane rebuilding, and blocked predictive-search statuses are missing.

### Task 4: Implement `equation_lanes` Normalization

**Files:**
- Modify: `src/euclid/workbench/service.py`
- Test: `tests/unit/workbench/test_service.py`

**Step 1: Discard stale lane payloads and add lane builder after legacy normalization**

At the start of `normalize_analysis_payload(...)`, remove any incoming lane payload:

```python
analysis.pop("equation_lanes", None)
```

Create `_build_equation_lanes(analysis=analysis)` and call it after:

- dataset rows have been loaded,
- `descriptive_reconstruction` is assigned,
- `predictive_law` is assigned,
- `holistic_equation` is assigned,
- `_apply_claim_taxonomy(...)` has run or immediately before evidence studio if taxonomy values are needed.

Preferred call site: after `_apply_claim_taxonomy(...)` and before `_build_evidence_studio(...)`, then ensure evidence studio sees unchanged top-level claim truth.

**Step 2: Build `descriptive_exact` from rows**

Use `_build_descriptive_exact_reconstruction(...)`. Keep `analysis["descriptive_reconstruction"]` for compatibility:

- `analysis["descriptive_reconstruction"]` remains populated for legacy clients,
- `equation_lanes["descriptive_exact"]` carries the exact lane.
- Do not point `analysis["descriptive_reconstruction"]` at the exact helper in this wave.
- The lane builder must distinguish row availability before calling the exact helper:
  missing, absent, or unloadable dataset rows produce a blocked lane with
  `reason_codes = ["dataset_rows_unavailable"]`; an available one-row dataset
  reaches `_build_descriptive_exact_reconstruction(...)` and returns
  `insufficient_rows_for_sample_exact_reconstruction`.

**Step 3: Build `predictive_law_search` from normalized predictive law**

```python
def _build_predictive_law_search_lane(*, analysis: Mapping[str, Any]) -> dict[str, Any]:
    predictive_law = analysis.get("predictive_law")
    if isinstance(predictive_law, Mapping):
        return {
            "status": "publishable_law",
            "lane_kind": "predictive_law_search",
            "source": "operator_point_publication_gate",
            "publishable": True,
            "predictive_law": dict(_jsonable(predictive_law)),
            "candidate_id": _string_or_none(
                ((predictive_law.get("equation") or {}).get("candidate_id"))
            ),
            "reason_codes": [],
            "evidence_summary": dict(predictive_law.get("evidence_summary") or {}),
            "honesty_note": predictive_law.get("honesty_note"),
        }
    operator_point = analysis.get("operator_point")
    if not isinstance(operator_point, Mapping):
        return {
            "status": "blocked",
            "lane_kind": "predictive_law_search",
            "source": "operator_point_publication_gate",
            "publishable": False,
            "predictive_law": None,
            "candidate_id": None,
            "reason_codes": ["operator_point_missing"],
            "evidence_summary": None,
            "honesty_note": "Predictive law search state is blocked because operator point artifacts are unavailable.",
        }
    result_mode = _string_or_none(operator_point.get("result_mode"))
    status = _string_or_none(operator_point.get("status"))
    if status == "failed":
        return {
            "status": "blocked",
            "lane_kind": "predictive_law_search",
            "source": "operator_point_publication_gate",
            "publishable": False,
            "predictive_law": None,
            "candidate_id": None,
            "reason_codes": ["operator_point_failed"],
            "evidence_summary": None,
            "honesty_note": "Predictive law search state is blocked because operator point execution failed.",
        }
    if status != "completed" or result_mode is None:
        return {
            "status": "blocked",
            "lane_kind": "predictive_law_search",
            "source": "operator_point_publication_gate",
            "publishable": False,
            "predictive_law": None,
            "candidate_id": None,
            "reason_codes": ["operator_point_result_mode_missing"],
            "evidence_summary": None,
            "honesty_note": "Predictive law search state is blocked because operator point result metadata is incomplete.",
        }
    reason_codes = _predictive_law_search_reason_codes(analysis=analysis)
    return {
        "status": "no_publishable_law",
        "lane_kind": "predictive_law_search",
        "source": "operator_point_publication_gate",
        "publishable": False,
        "predictive_law": None,
        "candidate_id": None,
        "reason_codes": reason_codes,
        "evidence_summary": None,
        "honesty_note": (
            "Predictive law search did not produce a publishable law under "
            "the declared validation scope."
        ),
    }
```

Add `_predictive_law_search_reason_codes(...)` so predictive search reasons are not copied blindly from holistic `gap_report`. It may use operator abstention reason codes, predictive-law banned-path reason codes, publication status, scorecard/claim-card predictive support failures, or `no_publishable_predictive_law`.

**Step 4: Run targeted normalization tests**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_adds_equation_lanes_with_exact_descriptive_and_no_law \
  tests/unit/workbench/test_service.py::test_equation_lanes_wrap_publishable_predictive_law_without_replacing_legacy_fields \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rebuilds_stale_incoming_equation_lanes \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_when_operator_point_missing \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_for_malformed_operator_point \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_predictive_search_blocked_for_failed_operator_even_with_result_mode \
  tests/unit/workbench/test_service.py::test_descriptive_exact_does_not_change_claim_surface_or_joint_artifacts \
  tests/unit/workbench/test_service.py::test_predictive_law_search_reasons_exclude_holistic_only_gap_codes
```

Expected: PASS.

**Step 5: Run adjacent predictive rejection tests**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_exact_closure_operator_point_for_predictive_law \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_stale_saved_predictive_law_without_operator_gates \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_synthetic_operator_point_for_predictive_law
```

Expected: PASS.

**Step 6: Commit**

```bash
git add src/euclid/workbench/service.py tests/unit/workbench/test_service.py
git commit -m "feat: normalize equation lanes"
```

### Task 5: Frontend Red Tests For Lane Rendering

**Files:**
- Modify: `tests/frontend/workbench/app.test.js`
- Read: `src/euclid/_assets/workbench/app.js:1811-1905`

**Step 1: Add fixture helper for `equation_lanes`**

In `tests/frontend/workbench/app.test.js`, add executable helpers before the lane tests:

```javascript
function routeAnalysis(analysis) {
  return (url) => {
    if (url.pathname === "/api/config") {
      return jsonResponse(
        buildConfig({
          recentAnalyses: [buildRecentEntry(analysis)],
        }),
      );
    }
    if (url.pathname === "/api/analysis") {
      return jsonResponse(analysis);
    }
    throw new Error(`Unhandled request: ${url.pathname}`);
  };
}

function attachEquationLanesNoLaw(analysis) {
  analysis.equation_lanes = {
    schema_version: "1.0.0",
    source: "workbench_normalization",
    lane_order: ["predictive_law_search", "descriptive_exact", "descriptive_fit"],
    descriptive_exact: {
      status: "completed",
      lane_kind: "descriptive_exact",
      exactness: "sample_exact_reconstruction",
      law_eligible: false,
      publishable: false,
      honesty_note:
      "Sample-exact reconstruction of observed rows only. It is descriptive, non-publishable, and not evidence of future behavior.",
      equation: {
        label: String.raw`y(t)=\operatorname{DFTExact}_{N}(t)`,
        curve: analysis.operator_point.equation.curve,
      },
      chart: {
        equation_curve: analysis.operator_point.equation.curve,
      },
      reconstruction_metrics: {
        max_abs_error: 0,
        effective_exact_tolerance: 1e-10,
        exact_tolerance_cleared: true,
      },
    },
    predictive_law_search: {
      status: "no_publishable_law",
      lane_kind: "predictive_law_search",
      publishable: false,
      predictive_law: null,
      reason_codes: ["predictive_support_failed"],
      honesty_note:
        "Predictive law search did not produce a publishable law under the declared validation scope.",
    },
  };
}

function markOperatorPublicationPublishable(analysis) {
  analysis.operator_point.publication = {
    status: "publishable",
    headline: "Operator point publication cleared the declared validation scope.",
  };
  analysis.operator_point.abstention = null;
  analysis.would_have_abstained_because = [];
}

function attachValidPredictiveLaw(analysis) {
  markOperatorPublicationPublishable(analysis);
  analysis.claim_class = "predictive_law";
  analysis.predictive_law = {
    status: "completed",
    claim_class: "predictive_law",
    honesty_note:
      "Predictive symbolic law reflects the publishable point-lane claim inside the declared validation scope.",
    equation: analysis.operator_point.equation,
    claim_card_ref: "artifacts/claim-card.json",
    scorecard_ref: "artifacts/scorecard.json",
    validation_scope_ref: "artifacts/validation-scope.json",
    publication_record_ref: "artifacts/publication-record.json",
    evidence_summary: {},
  };
}

function attachEquationLanesPublishableLaw(analysis) {
  attachEquationLanesNoLaw(analysis);
  analysis.equation_lanes.predictive_law_search = {
    status: "publishable_law",
    lane_kind: "predictive_law_search",
    publishable: true,
    predictive_law: analysis.predictive_law,
    reason_codes: [],
    evidence_summary: analysis.predictive_law?.evidence_summary || {},
  };
}
```

**Step 2: Write failing UI test for both lane cards**

```javascript
test("renders exact descriptive lane separately from no-publishable predictive search", async () => {
  const analysis = buildAtlasFixture();
  analysis.claim_class = "descriptive_reconstruction";
  analysis.publishable = false;
  analysis.predictive_law = null;
  analysis.holistic_equation = null;
  attachEquationLanesNoLaw(analysis);

  await mountWorkbench({ route: routeAnalysis(analysis) });

  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(textContent("#tab-overview")).toContain(
      "Sample-exact descriptive reconstruction",
    );
  });

  const overviewText = textContent("#tab-overview");
  expect(overviewText).toContain("Predictive law search");
  expect(overviewText).toContain("no publishable law");
  expect(overviewText).toContain("observed rows only");
  expect(overviewText).not.toContain("perfect law");
  expect(
    document.querySelector('[data-equation-lane-card="descriptive_exact"]'),
  ).not.toBeNull();
  expect(
    document.querySelector('[data-equation-lane-card="predictive_law_search"]'),
  ).not.toBeNull();
  expect(
    document.querySelector('[data-overlay-option="predictive_law_search"]'),
  ).toBeNull();
});
```

**Step 3: Run test to verify it fails**

Run:

```bash
npm run test:frontend -- --run tests/frontend/workbench/app.test.js
```

Expected: FAIL because frontend does not yet render `equation_lanes` lane cards.

**Step 4: Add failing overlay test**

```javascript
test("uses exact descriptive lane as descriptive overlay without marking it predictive", async () => {
  const analysis = buildAtlasFixture();
  analysis.predictive_law = null;
  analysis.holistic_equation = null;
  attachEquationLanesNoLaw(analysis);

  await mountWorkbench({ route: routeAnalysis(analysis) });
  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(textContent("#tab-point")).toContain("Sample-exact descriptive reconstruction");
  });

  const pointText = textContent("#tab-point");
  expect(pointText).toContain("active deterministic overlay");
  expect(pointText).not.toContain("Predictive symbolic law reflects");
  document
    .querySelector('[data-overlay-option="descriptive_exact"]')
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));
  expect(window.location.search).toContain("overlay=descriptive_exact");
});
```

**Step 5: Add failing frontend hierarchy and compatibility tests**

```javascript
test("keeps publishable predictive law ahead of exact descriptive overlay when both lanes exist", async () => {
  const analysis = buildAtlasFixture();
  attachValidPredictiveLaw(analysis);
  attachEquationLanesPublishableLaw(analysis);

  await mountWorkbench({ route: routeAnalysis(analysis) });
  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(textContent("#tab-overview")).toContain("Predictive symbolic law");
  });

  expect(document.querySelector('[data-equation-hero="predictive_law"]')).not.toBeNull();
  expect(document.querySelector('[data-equation-ribbon-item="predictive_law"]')).not.toBeNull();
  expect(
    document.querySelector('[data-overlay-option="predictive_law"][aria-pressed="true"]'),
  ).not.toBeNull();
  expect(textContent("#tab-overview")).toContain("Sample-exact descriptive reconstruction");
  expect(textContent("#tab-overview")).not.toContain("perfect law");
});


test("keeps holistic equation as default overlay when valid lanes coexist", async () => {
  const analysis = buildAtlasFixture();
  attachValidPredictiveLaw(analysis);
  analysis.claim_class = "holistic_equation";
  analysis.holistic_equation = {
    status: "completed",
    claim_class: "holistic_equation",
    honesty_note: "Backend-backed holistic equation.",
    equation: analysis.operator_point.equation,
    deterministic_source: "operator_point_publication",
    probabilistic_source: "probabilistic_distribution",
    validation_scope_ref: "artifacts/validation-scope.json",
    publication_record_ref: "artifacts/publication-record.json",
  };
  analysis.gap_report = [];
  analysis.not_holistic_because = [];
  attachEquationLanesPublishableLaw(analysis);

  await mountWorkbench({ route: routeAnalysis(analysis) });
  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(textContent("#tab-overview")).toContain("Holistic equation");
  });

  expect(document.querySelector('[data-equation-hero="holistic"]')).not.toBeNull();
  expect(document.querySelector('[data-equation-ribbon-item="holistic"]')).not.toBeNull();
  expect(document.querySelector('[data-equation-ribbon-item="predictive_law"]')).not.toBeNull();
  expect(document.querySelector('[data-overlay-option="predictive_law"]')).not.toBeNull();
  expect(
    document.querySelector('[data-overlay-option="holistic"][aria-pressed="true"]'),
  ).not.toBeNull();
  expect(
    document.querySelector('[data-overlay-option="predictive_law"][aria-pressed="true"]'),
  ).toBeNull();
});


test("hydrates and syncs descriptive exact overlay query state", async () => {
  const analysis = buildAtlasFixture();
  attachEquationLanesNoLaw(analysis);

  await mountWorkbench({
    route: routeAnalysis(analysis),
    locationSearch: "?overlay=descriptive_exact",
  });

  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(
      document.querySelector('[data-overlay-option="descriptive_exact"][aria-pressed="true"]'),
    ).not.toBeNull();
  });

  expect(window.location.search).toContain("overlay=descriptive_exact");
});


test("falls back from stale overlay query to default precedence with lanes present", async () => {
  const analysis = buildAtlasFixture();
  attachEquationLanesNoLaw(analysis);

  await mountWorkbench({
    route: routeAnalysis(analysis),
    locationSearch: "?overlay=bogus",
  });

  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(
      document.querySelector('[data-overlay-option="descriptive_exact"][aria-pressed="true"]'),
    ).not.toBeNull();
  });

  expect(window.location.search).not.toContain("overlay=bogus");
});


test("hydrates legacy descriptive reconstruction overlay when lanes are absent", async () => {
  const analysis = buildAtlasFixture();
  analysis.descriptive_reconstruction = {
    status: "completed",
    claim_class: "descriptive_reconstruction",
    honesty_note:
      "Legacy saved descriptive reconstruction remains available for old analyses.",
    equation: {
      candidate_id: "descriptive_fourier_reconstruction",
      family_id: "analytic",
      label: String.raw`y(t)=\operatorname{LegacyFourier}_{k}(t)`,
      curve: analysis.operator_point.equation.curve,
    },
    chart: {
      equation_curve: analysis.operator_point.equation.curve,
    },
  };
  delete analysis.equation_lanes;

  await mountWorkbench({
    route: routeAnalysis(analysis),
    locationSearch: "?overlay=descriptive_reconstruction",
  });

  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(
      document.querySelector('[data-overlay-option="descriptive_reconstruction"][aria-pressed="true"]'),
    ).not.toBeNull();
  });
});


test("does not promote no-law predictive search into hero ribbon or overlays", async () => {
  const analysis = buildAtlasFixture();
  analysis.predictive_law = null;
  analysis.holistic_equation = null;
  attachEquationLanesNoLaw(analysis);

  await mountWorkbench({ route: routeAnalysis(analysis) });
  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(
      document.querySelector('[data-equation-lane-card="predictive_law_search"]'),
    ).not.toBeNull();
  });

  expect(document.querySelector('[data-overlay-option="predictive_law_search"]')).toBeNull();
  expect(document.querySelector('[data-equation-hero="predictive_law_search"]')).toBeNull();
  expect(document.querySelector('[data-equation-ribbon-item="predictive_law_search"]')).toBeNull();
  expect(textContent("#hero")).not.toContain("Predictive symbolic law");
  expect(textContent("#hero")).not.toContain("publishable law");
  expect(textContent("#tab-overview")).toContain("no publishable law");
});


test("keeps exact descriptive card copy bounded to observed samples", async () => {
  const analysis = buildAtlasFixture();
  attachEquationLanesNoLaw(analysis);

  await mountWorkbench({ route: routeAnalysis(analysis) });
  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(
      document.querySelector('[data-equation-lane-card="descriptive_exact"]'),
    ).not.toBeNull();
  });

  const exactText = textContent('[data-equation-lane-card="descriptive_exact"]');
  expect(exactText).toContain("observed rows");
  expect(exactText).not.toMatch(/\blaw\b/i);
  expect(exactText).not.toMatch(/\bpublication\b/i);
  expect(exactText).not.toMatch(/\bperfect\b/i);
  expect(exactText).not.toMatch(/\bpredictive\b/i);
});


test("renders legacy saved analysis without equation lanes through old fields", async () => {
  const analysis = buildAtlasFixture();
  delete analysis.equation_lanes;

  await mountWorkbench({ route: routeAnalysis(analysis) });
  document
    .querySelector("button[data-analysis-path]")
    .dispatchEvent(new MouseEvent("click", { bubbles: true }));

  await waitFor(() => {
    expect(textContent("#tab-overview")).toContain("Benchmark-local descriptive fit");
  });

  expect(textContent("#tab-overview")).not.toContain("Sample-exact descriptive reconstruction");
});
```

Update `mountWorkbench({ route })` in the test harness to accept optional `locationSearch = ""` and call `window.history.replaceState(null, "", "http://localhost/" + locationSearch)` before importing the app module. This is a test-harness red step for URL hydration.

**Step 6: Commit after green**

After Task 6 passes:

```bash
git add tests/frontend/workbench/app.test.js src/euclid/_assets/workbench/app.js
git commit -m "feat: render equation lanes in workbench"
```

### Task 6: Implement Frontend Lane Rendering

**Files:**
- Modify: `src/euclid/_assets/workbench/app.js`
- Test: `tests/frontend/workbench/app.test.js`

**Step 1: Add lane access helpers**

Add helpers near existing claim helpers:

```javascript
function equationLanes(analysis) {
  return analysis?.equation_lanes && typeof analysis.equation_lanes === "object"
    ? analysis.equation_lanes
    : null;
}

function descriptiveExactLane(analysis) {
  const lane = equationLanes(analysis)?.descriptive_exact;
  return lane && lane.status === "completed" ? lane : null;
}

function predictiveLawSearchLane(analysis) {
  const lane = equationLanes(analysis)?.predictive_law_search;
  return lane && typeof lane === "object" ? lane : null;
}
```

**Step 2: Update overlay assembly**

Modify `availableDeterministicOverlays(analysis)`:

- Keep holistic and publishable predictive law overlays ahead of `descriptiveExactLane(analysis)`.
- Add `descriptiveExactLane(analysis)` before legacy `descriptive_reconstruction`.
- Use label `Sample-exact descriptive reconstruction`.
- Use honesty note from lane.
- When `predictiveLawSearchLane(analysis).status === "publishable_law"` and it has a curve, expose it through the existing `predictive_law` overlay id. Do not introduce a selectable `predictive_law_search` overlay id.
- Keep legacy fallback when no lanes exist.
- Add stable data attributes to overlay controls. Overlay buttons must render both the existing `data-overlay="${overlay.id}"` binding and the stable selector `data-overlay-option="${overlay.id}"`, with `aria-pressed="true"` on the selected overlay and `aria-pressed="false"` on inactive overlays. Keep `bindDynamicControls()` wired to one of those rendered attributes; do not strand the overlay buttons as inert markup.

**Step 3: Update overview/ribbon cards**

Where the overview/ribbon currently chooses cards from top-level fields, add lane cards:

- Predictive law search card always appears when lanes exist.
- Descriptive exact card appears when completed.
- No-law status appears as a lower claim, not as an error.
- Exact descriptive lane cards use `data-equation-lane-card="descriptive_exact"`.
- Predictive search cards use `data-equation-lane-card="predictive_law_search"`.
- Replace the current hardcoded overview hero marker with the selected source id: the overview hero uses `data-equation-hero="<source id>"`.
- Equation ribbon entries use `data-equation-ribbon-item="<source id>"`.
- Hero and ribbon precedence remains: holistic, predictive law, descriptive exact, legacy descriptive reconstruction, legacy descriptive fit. A no-law predictive search card never becomes the hero claim.

**Step 4: Keep copy fail-closed**

Use "sample-exact", "observed rows", "descriptive", and "no publishable law". Do not use "perfect" or "predictive" for the exact lane.

**Step 5: Run frontend tests**

Run:

```bash
npm run test:frontend -- --run tests/frontend/workbench/app.test.js
```

Expected: PASS.

### Task 7: Backend Tests For Target Eligibility And Exact-Closure Rejection

**Files:**
- Modify: `tests/unit/workbench/test_service.py`
- Modify: `src/euclid/workbench/service.py`

**Step 1: Write failing non-path target test**

```python
def test_equation_lanes_mark_exact_reconstruction_not_applicable_for_next_day_up(
    tmp_path: Path,
) -> None:
    rows = _workbench_dataset_rows([0.0, 1.0, 0.0, 1.0])
    dataset_csv = _write_dataset_csv(tmp_path / "next-day-up.csv", rows)
    payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "next_day_up"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {"status": "failed", "error": "fixture"},
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["equation_lanes"]["descriptive_exact"]["status"] == "blocked"
    assert "target_not_numeric_path" in normalized["equation_lanes"]["descriptive_exact"]["reason_codes"]
    assert normalized["equation_lanes"]["descriptive_exact"]["law_eligible"] is False
    assert normalized["equation_lanes"]["descriptive_exact"]["publishable"] is False


def test_equation_lanes_mark_exact_reconstruction_blocked_without_dataset_rows() -> None:
    normalized = normalize_analysis_payload(
        {
            "dataset": {
                "symbol": "SPY",
                "target": {"id": "daily_return"},
            },
            "operator_point": {"status": "failed", "error": "fixture"},
        }
    )

    assert normalized["equation_lanes"]["descriptive_exact"]["status"] == "blocked"
    assert "dataset_rows_unavailable" in normalized["equation_lanes"]["descriptive_exact"]["reason_codes"]
    assert normalized["equation_lanes"]["descriptive_exact"]["law_eligible"] is False
    assert normalized["equation_lanes"]["descriptive_exact"]["publishable"] is False


def test_equation_lanes_mark_exact_reconstruction_blocked_for_single_row(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "single-row.csv",
        _workbench_dataset_rows([1.0]),
    )
    normalized = normalize_analysis_payload(
        {
            "dataset": {
                "symbol": "SPY",
                "target": {"id": "daily_return"},
                "dataset_csv": str(dataset_csv),
            },
            "operator_point": {"status": "failed", "error": "fixture"},
        }
    )

    assert normalized["equation_lanes"]["descriptive_exact"]["status"] == "blocked"
    assert "insufficient_rows_for_sample_exact_reconstruction" in normalized["equation_lanes"]["descriptive_exact"]["reason_codes"]
    assert normalized["equation_lanes"]["descriptive_exact"]["law_eligible"] is False
    assert normalized["equation_lanes"]["descriptive_exact"]["publishable"] is False
```

**Step 2: Write failing exact-closure isolation test**

```python
def test_equation_lanes_do_not_turn_exact_descriptive_lane_into_predictive_law(
    tmp_path: Path,
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path=tmp_path,
        operator_equation={
            "candidate_id": "analytic_exact_closure_surface",
            "exactness": "sample_exact_closure",
            "label": "y(t) = exact_closure(sample)",
            "curve": [{"event_time": "2025-01-01T00:00:00Z", "fitted_value": 1.0}],
        },
    )

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is None
    assert normalized["equation_lanes"]["predictive_law_search"]["status"] != "publishable_law"
    assert normalized["equation_lanes"]["descriptive_exact"]["law_eligible"] is False
```

**Step 3: Write failing tests for new exact-lane marker smuggling**

```python
@pytest.mark.parametrize(
    "marker_patch",
    [
        {"exactness": "sample_exact_reconstruction"},
        {"candidate_id": "descriptive_exact_fourier_reconstruction"},
        {"source": "workbench_descriptive_exact_reconstruction"},
    ],
)
def test_normalize_analysis_payload_rejects_publishable_exact_descriptive_operator_point_for_predictive_law(
    tmp_path: Path,
    marker_patch: dict[str, Any],
) -> None:
    operator_equation = {
        "candidate_id": "analytic_lag1_affine",
        "family_id": "analytic",
        "label": "y(t) = 1.8 + 0.92*y(t-1)",
        "curve": [
            {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
            {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
            {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
            {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
        ],
    }
    operator_equation.update(marker_patch)
    payload = _predictive_law_candidate_publication_payload(
        tmp_path=tmp_path,
        operator_equation=operator_equation,
    )

    normalized = normalize_analysis_payload(payload)

    assert normalized["operator_point"]["publication"]["status"] == "publishable"
    assert normalized["predictive_law"] is None
    assert normalized["equation_lanes"]["predictive_law_search"]["status"] != "publishable_law"
    assert "requires_exact_sample_reconstruction" in normalized["gap_report"]
    assert (
        "requires_exact_sample_reconstruction"
        in normalized["equation_lanes"]["predictive_law_search"]["reason_codes"]
    )


@pytest.mark.parametrize(
    "marker_patch",
    [
        {"exactness": "sample_exact_reconstruction"},
        {"candidate_id": "descriptive_exact_fourier_reconstruction"},
        {"source": "workbench_descriptive_exact_reconstruction"},
    ],
)
def test_normalize_analysis_payload_rejects_exact_descriptive_holistic_equation(
    tmp_path: Path,
    marker_patch: dict[str, Any],
) -> None:
    payload = _predictive_law_candidate_publication_payload(
        tmp_path=tmp_path,
        operator_equation={
            "candidate_id": "analytic_lag1_affine",
            "label": "y(t) = 1.8 + 0.92*y(t-1)",
            "curve": [
                {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
                {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
                {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
                {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
            ],
        },
    )
    payload["probabilistic"] = {
        "distribution": {
            "status": "completed",
            "selected_family": "analytic",
            "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
            "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
            "rows": [
                {
                    "origin_time": "2026-04-16T00:00:00Z",
                    "available_at": "2026-04-17T21:00:00Z",
                    "horizon": 1,
                    "location": 11.3,
                    "scale": 0.4,
                    "realized_observation": 11.4,
                }
            ],
            "latest_row": {
                "origin_time": "2026-04-16T00:00:00Z",
                "available_at": "2026-04-17T21:00:00Z",
                "horizon": 1,
                "location": 11.3,
                "scale": 0.4,
                "realized_observation": 11.4,
            },
            "calibration": {
                "status": "passed",
                "passed": True,
                "gate_effect": "publishable",
                "diagnostics": [
                    {
                        "diagnostic_id": "pit_or_randomized_pit_uniformity",
                        "sample_size": 1,
                        "status": "passed",
                    }
                ],
            },
            "chart": {"forecast_bands": []},
        }
    }
    holistic_equation_payload = {
        "candidate_id": "analytic_lag1_affine",
        "family_id": "analytic",
        "label": "y(t)=otherwise_valid_backend_joint_claim",
        "curve": [
            {"event_time": "2026-04-14T00:00:00Z", "fitted_value": 10.0},
            {"event_time": "2026-04-15T00:00:00Z", "fitted_value": 11.0},
            {"event_time": "2026-04-16T00:00:00Z", "fitted_value": 10.8},
            {"event_time": "2026-04-17T00:00:00Z", "fitted_value": 11.3},
        ],
    }
    holistic_equation_payload.update(marker_patch)
    payload["holistic_equation"] = {
        "status": "completed",
        "claim_class": "holistic_equation",
        "joint_claim_gate": {
            "backend_authored": True,
            "status": "accepted",
        },
        "deterministic_source": "predictive_law",
        "probabilistic_source": "distribution",
        "validation_scope_ref": "validation_scope_manifest@1.0.0:scope-1",
        "publication_record_ref": "publication_record_manifest@1.1.0:publication-1",
        "honesty_note": "Exact descriptive reconstruction must not become a holistic law.",
        "equation": holistic_equation_payload,
    }

    normalized = normalize_analysis_payload(payload)

    assert normalized["predictive_law"] is not None
    assert normalized["holistic_equation"] is None
    assert "requires_exact_sample_reconstruction" in normalized["gap_report"]
    assert normalized["not_holistic_because"] == [
        "requires_exact_sample_reconstruction"
    ]
    assert "no_backend_joint_claim" not in normalized["gap_report"]
    assert "probabilistic_evidence_thin" not in normalized["gap_report"]
```

**Step 4: Run tests to verify failure**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_exact_reconstruction_not_applicable_for_next_day_up \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_exact_reconstruction_blocked_without_dataset_rows \
  tests/unit/workbench/test_service.py::test_equation_lanes_mark_exact_reconstruction_blocked_for_single_row \
  tests/unit/workbench/test_service.py::test_equation_lanes_do_not_turn_exact_descriptive_lane_into_predictive_law \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_publishable_exact_descriptive_operator_point_for_predictive_law \
  tests/unit/workbench/test_service.py::test_normalize_analysis_payload_rejects_exact_descriptive_holistic_equation
```

Expected: FAIL until target eligibility and search-lane reason handling are implemented.

**Step 5: Implement minimal target and rejection handling**

- Add target-id check before building exact reconstruction.
- Add explicit dataset-row availability handling before the exact helper so
  missing/unloadable rows use `dataset_rows_unavailable` while available
  one-row datasets use `insufficient_rows_for_sample_exact_reconstruction`.
- Add blocked exact lane for `next_day_up`.
- Ensure predictive search lane uses normalized `analysis["predictive_law"]`, not raw operator point.
- Extend `_predictive_law_gap_reason_codes(...)` and holistic gate checks to reject each new exact descriptive marker independently with `requires_exact_sample_reconstruction`: `exactness == "sample_exact_reconstruction"`, `candidate_id == "descriptive_exact_fourier_reconstruction"` or equivalent candidate-id/source-candidate fields, and `source == "workbench_descriptive_exact_reconstruction"` or equivalent source fields.

**Step 6: Run targeted tests**

Expected: PASS.

### Task 8: Documentation Red Checks And Updates

**Files:**
- Modify: `docs/workbench.md`
- Modify: `docs/reference/workbench.md`
- Optional modify: `docs/reference/search-core.md` only if wording needs the new display contract
- Test: `tests/spec_compiler/test_math_documentation_truthfulness.py`

**Step 1: Add failing documentation assertions**

Add to `tests/spec_compiler/test_math_documentation_truthfulness.py`:

```python
WORKBENCH_DOCS = (
    REPO_ROOT / "docs" / "workbench.md",
    REPO_ROOT / "docs" / "reference" / "workbench.md",
)


def test_workbench_docs_define_equation_lanes_without_predictive_promotion() -> None:
    combined = "\n".join(_text(path) for path in WORKBENCH_DOCS)

    for required in (
        "equation_lanes",
        "descriptive_exact",
        "predictive_law_search",
        "sample-exact",
        "observed rows",
        "not a predictive law",
        "no publishable law",
    ):
        assert required in combined

    unsafe_fragments = (
        "perfect law",
        "sample-exact predictive",
        "descriptive_exact publishable",
    )
    for fragment in unsafe_fragments:
        assert fragment not in combined
```

**Step 2: Run docs test to verify failure**

Run:

```bash
PYTEST_ADDOPTS="" PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_math_documentation_truthfulness.py::test_workbench_docs_define_equation_lanes_without_predictive_promotion
```

Expected: FAIL because the docs do not yet define `equation_lanes`.

**Step 3: Update docs**

Update `docs/workbench.md` and `docs/reference/workbench.md` around the normalized surface list:

- Add `equation_lanes`.
- Define `descriptive_exact`.
- Define `predictive_law_search`.
- Keep exact closure out of predictive claims.
- State that "perfect" means "within tolerance over observed rows only" if the word appears at all.

**Step 4: Run docs and targeted tests**

Run:

```bash
PYTEST_ADDOPTS="" PYTHONPATH=src python3.11 -m pytest -q tests/spec_compiler/test_math_documentation_truthfulness.py
PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add docs/workbench.md docs/reference/workbench.md tests/spec_compiler/test_math_documentation_truthfulness.py
git commit -m "docs: define equation lane semantics"
```

### Task 9: Integration And Saved-Analysis Compatibility

**Files:**
- Modify: `tests/integration/test_workbench_analysis.py`
- Read: `src/euclid/workbench/server.py`
- Read: `src/euclid/workbench/service.py`

**Step 1: Write failing persisted normalized analysis test**

Add a test named `test_create_workbench_analysis_persists_equation_lanes`:

```python
def test_create_workbench_analysis_persists_equation_lanes(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        "euclid.workbench.service.fetch_fmp_eod_history",
        lambda **_: [
            {"date": f"2026-05-{day:02d}", "close": 100.0 + ((-1) ** day) * day}
            for day in range(1, 12)
        ],
    )

    analysis = create_workbench_analysis(
        symbol="SPY",
        api_key="test-key",
        target_id="daily_return",
        output_root=tmp_path / "out",
        project_root=tmp_path,
        start_date="2026-05-01",
        end_date="2026-05-11",
        include_probabilistic=False,
        include_benchmark=False,
    )
    persisted = json.loads(Path(analysis["analysis_path"]).read_text(encoding="utf-8"))

    assert persisted["equation_lanes"]["schema_version"] == "1.0.0"
    assert persisted["descriptive_reconstruction"] is not None
    assert "descriptive_exact" in persisted["equation_lanes"]
    assert "predictive_law_search" in persisted["equation_lanes"]
```

**Step 2: Write failing legacy saved-payload normalization unit test**

In `tests/unit/workbench/test_service.py`, add a test named `test_saved_analysis_without_equation_lanes_rebuilds_lanes_from_legacy_fields`:

```python
def test_saved_analysis_without_equation_lanes_rebuilds_lanes_from_legacy_fields(
    tmp_path: Path,
) -> None:
    dataset_csv = _write_dataset_csv(
        tmp_path / "legacy.csv",
        _workbench_dataset_rows([1.0, -0.5, 0.25]),
    )
    legacy_payload = {
        "dataset": {
            "symbol": "SPY",
            "target": {"id": "daily_return"},
            "dataset_csv": str(dataset_csv),
        },
        "operator_point": {"status": "failed", "error": "legacy"},
        "descriptive_reconstruction": {"status": "completed"},
    }

    normalized = normalize_analysis_payload(legacy_payload)

    assert normalized["equation_lanes"]["schema_version"] == "1.0.0"
    assert normalized["equation_lanes"]["descriptive_exact"]["status"] == "completed"
    assert normalized["predictive_law"] is None
```

**Step 3: Run persistence and legacy-normalization tests to verify failure**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/integration/test_workbench_analysis.py::test_create_workbench_analysis_persists_equation_lanes \
  tests/unit/workbench/test_service.py::test_saved_analysis_without_equation_lanes_rebuilds_lanes_from_legacy_fields
```

Expected: FAIL until saved-analysis normalization and persistence include lanes.

**Step 4: Implement minimal compatibility fix**

- Update `create_workbench_analysis(...)` so it persists the normalized payload containing `equation_lanes` to `analysis.json`.
- Ensure server reload endpoints still call `normalize_analysis_payload(...)` before returning saved payloads.
- Ensure normalization discards stale incoming `equation_lanes` before rebuilding.

**Step 5: Run integration tests**

Expected: PASS.

### Task 10: Final Verification And Release Notes

**Files:**
- Modify: `docs/plans/2026-05-26-equation-lanes-design-and-implementation-plan.md` if implementation notes are recorded there
- Optional modify: `docs/progress/2026-05-26-equation-lanes-implementation.md` if implementation starts in a later wave

**Step 1: Run targeted backend tests**

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/unit/workbench/test_service.py tests/integration/test_workbench_analysis.py
```

Expected: PASS.

**Step 2: Run frontend tests**

```bash
npm run test:frontend -- --run tests/frontend/workbench/app.test.js tests/frontend/workbench-ui.test.js
```

Expected: PASS.

**Step 3: Run release smoke if time and environment allow**

```bash
./scripts/release_smoke.sh
```

Expected: PASS or fail-closed with exact blocker notes.

**Step 4: Manual workbench check**

Run a local workbench analysis with a short jagged or real SPY daily-return range and verify:

- Exact descriptive lane overlays the observed points.
- Predictive law search shows publishable or no-law state.
- The page never calls the exact descriptive lane predictive.

**Step 5: Commit final verification notes**

```bash
git add docs/plans/2026-05-26-equation-lanes-design-and-implementation-plan.md
git commit -m "docs: approve equation lanes implementation plan"
```

## Jury Protocol

The validation jury reviews this file after the first draft is written. The local synthesizer provides the same file path and current repo anchors to every juror.

### Actual Jury Size

Use 5 voting jurors in this run because the subagent backend hit a thread cap while attempting a 7-agent planning wave. This is an honest capacity normalization. The five roles are a merged form of the planning swarm recommendations.

### Juror Roles

1. Backend Contract Juror
2. Exact Reconstruction Math Juror
3. Predictive Law Gate Juror
4. Frontend/API Semantics Juror
5. TDD/Docs/Verification Juror

### Common Jury Prompt

```text
Review `docs/plans/2026-05-26-equation-lanes-design-and-implementation-plan.md` for Approach A: add an additive `equation_lanes` API/UI contract while preserving `descriptive_reconstruction`, `predictive_law`, and `holistic_equation`. Work read-only. Use current repo state as authority, especially `src/euclid/workbench/service.py`, `src/euclid/_assets/workbench/app.js`, `tests/unit/workbench/test_service.py`, `tests/frontend/workbench/app.test.js`, `docs/workbench.md`, and `docs/reference/search-core.md`.

Return one of `APPROVE`, `REVISE`, or `BLOCK`. Include blockers, missing tests, strongest dissenting concern, confidence, and whether the spec is ready for implementation. Do not approve unless no exact descriptive equation can be promoted into predictive law, the API is backward-compatible, every implementation task is test-first, and docs/UI language preserves the descriptive-vs-predictive boundary.
```

### Role-Specific Review Focus

- Backend Contract Juror: payload shape, lane versioning, normalized source of truth, saved-analysis compatibility, top-level field preservation.
- Exact Reconstruction Math Juror: full DFT reconstruction, odd/even sample sizes, tolerance, coefficient storage, finite-row requirements, target applicability.
- Predictive Law Gate Juror: exact reconstruction cannot enter predictive/holistic promotion, existing publication gates stay fail-closed, search-core policy remains intact.
- Frontend/API Semantics Juror: rendering order, overlay selection, no-law state, copy, URL state, legacy fallback, visual claim hierarchy.
- TDD/Docs/Verification Juror: task granularity, failing-test-first steps, exact commands, docs updates, final verification adequacy.

### Approval Rule

Unanimity is required. Any `REVISE` or `BLOCK` sends this file back for revision and a full jury rerun. Majority cannot override claim-boundary, mathematical exactness, or test-coverage objections.

### Jury Verdict Ledger

Final verdict: `APPROVE`.

| Role | Final Juror | Verdict | Confidence | Blocking Findings |
| --- | --- | --- | ---: | --- |
| Backend Contract Juror | Locke (`019e66a1-6e12-7783-ade1-cf58245819ba`) | `APPROVE` | 0.91 | None |
| Exact Math Juror | Noether (`019e66a1-7061-7972-a73f-f6fa48e23928`) | `APPROVE` | 0.92 | None |
| Predictive Gate Juror | Confucius (`019e66a1-72df-7ac1-ba2b-38bfb2b190e7`) | `APPROVE` | 0.91 | None |
| Frontend/API Semantics Juror | Kant (`019e66a8-c8bc-7331-b9c2-78961f36c2ef`) | `APPROVE` | 0.91 | None |
| TDD/Docs Juror | Boole (`019e66a1-7980-7850-a273-37009b760b4b`) | `APPROVE` | 0.91 | None |

Readiness conclusion: unanimous approval. Revision 6 is ready for implementation under the Approval Rule above.

## Post-Implementation Audit Ledger

Final implementation verdict: `APPROVE` for the equation-lanes plan scope.

| Role | Final Auditor | Verdict | Blocking Findings |
| --- | --- | --- | --- |
| Backend/API Contract Audit | Erdos (`019e672d-81c0-7451-8d08-2336236224e1`) | `APPROVE` | None |
| Frontend/Docs Semantics Audit | Jason (`019e672d-b70b-7e13-aed3-bf0635145747`) | `APPROVE` | None |
| Completion Ledger Audit | Lorentz (`019e672d-fa39-7b22-b557-c86c8230affa`) | `APPROVE` after one ledger-only revision | Initial finding was ledger-only: Batch 7 was still in progress, final audit was pending, and the repo matrix artifact status was ambiguous. Recheck approved the terminal Batch 7 row, post-implementation audit ledger, non-terminal release-smoke note, and stale historical repo-matrix clarification. |

Post-implementation audit conclusion: the implementation is complete for the approved Approach A scope. Exact descriptive reconstruction is sample-exact over observed rows only, predictive law search remains evidence-gated, legacy fields remain additive and backward-compatible, and final verification is recorded with explicit limits around the non-terminal release-smoke attempt.

## Completion Criteria For This Planning Objective

This planning objective is complete only when:

- this spec exists in `docs/plans/`,
- it decomposes implementation into test-driven tasks and subtasks,
- it records the planning swarm and jury protocol,
- a five-member independent jury has reviewed it,
- every juror returns `APPROVE`,
- any required revisions have been applied,
- this file records the final unanimous verdict.
