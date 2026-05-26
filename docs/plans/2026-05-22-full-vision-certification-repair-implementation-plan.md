# Full-Vision Certification Repair Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Repair Euclid's full-vision certification blockers until release, benchmark, clean-install, replay, and readiness gates pass without hidden skips.

**Architecture:** Use the existing claim/evidence/readiness architecture and make each readiness row close through executable artifacts. Repair upstream semantic claim guards first, then publication/replay bundles, shared-local lifecycle evidence, benchmark semantic aggregation, and compatibility serialization.

**Tech Stack:** Python 3.11, pytest, Euclid CLI, JSON/YAML readiness contracts, golden artifact fixtures.

---

## Coordination Rules

- Preserve all pre-existing dirty work.
- Use `apply_patch` for manual edits.
- Do not weaken tests or readiness gates to make failures disappear.
- Every behavior change starts from a re-run failing test.
- Record every wave in `docs/progress/2026-05-22-full-vision-completion-ledger.md`.
- After each wave, run an odd expert jury before declaring the wave complete.

## Task 1: Nonstationarity And Claim-Scope Guards

**Files:**
- Modify: `src/euclid/modules/claims.py`
- Inspect/possibly modify: `src/euclid/nonstationarity/*`
- Test: `tests/unit/modules/test_phase6_stationarity_review.py`
- Test: `tests/unit/nonstationarity/test_stability.py`
- Test: `tests/unit/nonstationarity/test_stability_phase61_worker.py`

**Step 1: Verify RED**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/unit/modules/test_phase6_stationarity_review.py \
  tests/unit/nonstationarity/test_stability.py \
  tests/unit/nonstationarity/test_stability_phase61_worker.py
```

Expected: failures showing unresolved instability and scoped evidence can still
support invariant/stationary law publication.

**Step 2: Implement minimal claim-scope checks**

Ensure claim publication rejects:

- unresolved `regime_switching_artifact` instability for invariant stationary law claims
- unresolved `state_space_artifact` instability for invariant stationary law claims
- generic `instability_evidence` artifacts used as law support
- regime/state-scoped evidence without explicit valid-given wording or manifest scope
- regime/state-scoped evidence laundering into unscoped stationary law claims

**Step 3: Verify GREEN**

Run the same command. Expected: all selected tests pass.

## Task 2: Mechanistic Dossier Publication Semantics

**Files:**
- Inspect/modify: `src/euclid/modules/*mechanistic*`
- Inspect/modify: `src/euclid/modules/claims.py`
- Inspect/modify: `src/euclid/modules/catalog_publishing.py`
- Test: `tests/integration/test_mechanistic_lane_publication.py`
- Test: `tests/integration/test_mechanistic_operator_pipeline.py`
- Test: `tests/benchmarks/test_mechanistic_track.py`

**Step 1: Verify RED**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/integration/test_mechanistic_lane_publication.py \
  tests/integration/test_mechanistic_operator_pipeline.py \
  tests/benchmarks/test_mechanistic_track.py
```

Expected: positive mechanistic support does not raise the claim type to
`mechanistically_compatible_law`, while contradictory/insufficient cases use the
wrong dossier downgrade status.

**Step 2: Implement minimal dossier status alignment**

Align positive, contradictory, and insufficient support cases with the benchmark
contract without allowing weak mechanistic evidence to publish stronger claims.

**Step 3: Verify GREEN**

Run the same command. Expected: selected tests pass.

## Task 3: Publication, Replay, And Golden Bundles

**Files:**
- Modify: `src/euclid/modules/catalog_publishing.py`
- Modify as needed: `src/euclid/modules/probabilistic_evaluation.py`
- Modify as needed: `src/euclid/modules/claims.py`
- Test: `tests/integration/test_phase07_publication_paths.py`
- Test: `tests/integration/test_probabilistic_publication.py`
- Test: `tests/integration/test_probabilistic_replay_and_publication.py`
- Test: `tests/golden/test_phase07_publication_fixtures.py`
- Test: `tests/golden/test_probabilistic_publication_bundles.py`

**Step 1: Verify RED**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/integration/test_phase07_publication_paths.py \
  tests/integration/test_probabilistic_publication.py \
  tests/integration/test_probabilistic_replay_and_publication.py \
  tests/golden/test_phase07_publication_fixtures.py \
  tests/golden/test_probabilistic_publication_bundles.py
```

Expected: publication bundle and golden fixture mismatches expose the current
contract drift.

**Step 2: Implement minimal bundle contract alignment**

Make publication bundles expose replay/catalog support and the updated claim
scope/evidence semantics consistently.

**Step 3: Verify GREEN**

Run the same command. Expected: selected tests pass.

## Task 4: Shared-Plus-Local Lifecycle Evidence

**Files:**
- Inspect/modify: shared-local module paths discovered by `rg "shared_local|shared-plus-local|decomposition_policy|aggregation_table" src tests`
- Test: `tests/integration/test_shared_local_operator_pipeline.py`
- Test: `tests/golden/test_shared_local_bundles.py`

**Step 1: Verify RED**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/integration/test_shared_local_operator_pipeline.py \
  tests/golden/test_shared_local_bundles.py
```

Expected: lifecycle artifacts do not satisfy publication/golden expectations.

**Step 2: Implement minimal lifecycle artifact evidence**

Ensure decomposition policy and aggregation table artifacts are emitted with
semantic runtime and replay visibility.

**Step 3: Verify GREEN**

Run the same command. Expected: selected tests pass.

## Task 5: Benchmark Semantic Readiness

**Files:**
- Modify: `src/euclid/benchmarks/runtime.py`
- Modify: `src/euclid/benchmarks/reporting.py`
- Modify as needed: `src/euclid/readiness/judgment.py`
- Test: `tests/benchmarks/test_current_release_readiness.py`
- Test: `tests/benchmarks/test_full_vision_suite.py`
- Test: `tests/benchmarks/test_p13_benchmark_universe.py`
- Test: `tests/benchmarks/test_suite_runner.py`

**Step 1: Verify RED**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q \
  tests/benchmarks/test_current_release_readiness.py \
  tests/benchmarks/test_full_vision_suite.py \
  tests/benchmarks/test_p13_benchmark_universe.py \
  tests/benchmarks/test_suite_runner.py
```

Expected: benchmark surfaces complete execution but fail semantic assertions or
canonical active-scope readiness.

**Step 2: Implement minimal semantic assertion closure**

Require benchmark surface pass status to include measured semantic assertions
for every declared surface and make those assertions available to readiness
judgment.

**Step 3: Verify GREEN**

Run the same command. Expected: selected tests pass.

## Task 6: Lattice Policy Serialization Compatibility

**Files:**
- Modify: `src/euclid/math/lattice.py`
- Test: `tests/unit/math/test_lattice_worker.py`

**Step 1: Verify RED**

Run:

```bash
PYTHONPATH=src python3.11 -m pytest -q tests/unit/math/test_lattice_worker.py
```

Expected: `LatticePolicy.as_dict()` includes artifact metadata not expected by
the compact policy contract.

**Step 2: Implement minimal serialization split**

Keep artifact metadata available where required, but preserve the compact
`as_dict()` contract expected by policy serialization tests.

**Step 3: Verify GREEN**

Run the same command. Expected: selected test passes.

## Task 7: Full Verification Wave

**Files:**
- Update: `docs/progress/2026-05-22-full-vision-completion-ledger.md`

**Step 1: Run matrix**

Run:

```bash
PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .
```

Expected: zero failures and zero hidden skips for required gates.

**Step 2: Run certification commands**

Run the commands in `scripts/release_smoke.sh` or run:

```bash
./scripts/release_smoke.sh
```

Expected: all certification commands pass.

**Step 3: Run expert jury**

Launch an odd independent jury over mathematics, statistics, forecasting,
symbolic regression, software architecture, reproducibility, security, UX, and
release engineering.

Expected: no unresolved correctness, realism, efficacy, reproducibility, or
release-readiness issue.
