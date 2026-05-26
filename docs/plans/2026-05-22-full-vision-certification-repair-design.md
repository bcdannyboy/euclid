# Euclid Full-Vision Certification Repair Design

Date: 2026-05-22
Status: Approved for implementation by `/goal implement continuously`

## Goal

Close Euclid's full-vision readiness gap by repairing executable evidence
paths, not by weakening gates. A public claim is releasable only when its
descriptive, predictive, probabilistic, mechanistic, robustness, benchmark,
clean-install, and replay evidence survive the existing certification commands.

## Recommended Approach

Use an evidence-first certification repair loop:

1. Start from the failing executable gates.
2. Group failures by semantic surface.
3. For each surface, write or run the narrow failing tests first.
4. Implement the smallest contract-aligned fix.
5. Rerun the cluster, then the broader repo matrix.
6. Run an odd independent jury after each wave.
7. Update `docs/progress/2026-05-22-full-vision-completion-ledger.md` with
   commands, results, failures, and residual risks.

This is preferred over a broad redesign because the repo already has dirty,
partially implemented full-vision surfaces and a rich test/readiness matrix.
The fastest safe route is to make those gates true and replayable.

## Alternatives Considered

### Matrix-Only Repair

Fix the 36 current test failures directly and stop when the matrix is green.

Trade-off: fastest path to a green test run, but it can miss readiness rows that
need semantic artifacts and replay proof. Rejected as insufficient for the
full-vision completion standard.

### Full Contract Redesign

Redesign runtime, claim, benchmark, and publication contracts together.

Trade-off: conceptually clean, but too disruptive in the current dirty tree and
likely to invalidate golden/replay artifacts. Rejected unless the focused repair
loop proves the existing contracts cannot be made truthful.

## Architecture

The repaired system keeps the existing separation of responsibilities:

- Claim scope validation stays in `src/euclid/modules/claims.py` and related
  evidence contract helpers.
- Nonstationarity diagnostics emit diagnostic or instability evidence, not law
  claims.
- Mechanistic, probabilistic, robustness, and shared-local modules produce
  artifacts with explicit publication ceilings.
- Benchmark runtime/reporting converts task evidence into semantic readiness
  assertions without hidden fixture shortcuts.
- Readiness judgment consumes artifacts, benchmark summaries, repo matrix
  results, and replay evidence to produce release verdicts.

## Data Flow

Ordered observations enter run pipelines, produce candidate descriptive
structures, and may produce predictive claims only after time-safe evaluation.
Evidence artifacts are attached to publication bundles and benchmark summaries.
Readiness rows close only when those artifacts are available through runtime,
replay, benchmark, and clean-install surfaces.

## First Repair Order

1. Claim-scope and nonstationarity guards.
2. Mechanistic publication ceilings and dossier statuses.
3. Publication/replay/golden bundle contract alignment.
4. Shared-plus-local lifecycle artifact evidence.
5. Benchmark semantic readiness aggregation.
6. Lattice policy serialization compatibility.

This order targets upstream semantic blockers before downstream golden and
readiness failures.

## Error Handling Rules

- Instability evidence must block invariant stationary law claims unless the
  publication explicitly narrows scope with valid-given wording and manifest
  scope.
- Scoped regime/state evidence must not support unscoped law claims.
- Mechanistic evidence may raise or lower the claim ceiling, but weak or
  contradictory evidence must never be presented as mechanistic support.
- Benchmark pass status must require measured semantic assertions, not only task
  completion.
- Replay proof must point to inspectable artifacts, not inferred state.

## Testing Strategy

Use TDD for behavior changes. Existing failing tests count as red tests only
when the failure has been re-run in the current implementation wave and the
failure reason is understood. Each wave runs:

1. Narrow failing tests for the cluster.
2. Adjacent integration/golden tests affected by the cluster.
3. Relevant benchmark tests.
4. `PYTHONPATH=src python3.11 -m euclid release repo-test-matrix --project-root .`
   after the cluster set is green enough to justify a full matrix.

Full completion still requires all commands in `scripts/release_smoke.sh`.
