# Euclid Reference

This directory is the public technical reference for Euclid and the shortest path from the GitHub release pitch to the implementation.

Each document is anchored to a live subsystem and written to do two jobs at once: explain what Euclid actually claims, and show where those claims cash out in code.

The through-line is consistent across the reference set: Euclid derives descriptive equations for ordered data, tests whether those equations earn predictive support, and only presents a unified deterministic-plus-stochastic equation when both lanes clear the same publication scope. If you are evaluating the project, this directory is where the headline story stops being a pitch and becomes a technical contract.

## Documents

- `system.md`: the full subsystem map and execution planes
- `runtime-cli.md`: the main command-line surface, workspace layout, replay model, and operational behavior
- `modeling-pipeline.md`: the stage-by-stage path from ordered observations to scored forecasts and publishable outcomes
- `search-core.md`: Euclid’s mathematical core, including CIR normalization, reducer semantics, description gain, and adapters
- `contracts-manifests.md`: the formal specification layer for schemas, manifests, object identity, and registries
- `benchmarks-readiness.md`: benchmark semantics, release scopes, readiness rules, and compiler outputs
- `workbench.md`: the local analysis studio, saved-analysis model, and explainer behavior
- `testing-truthfulness.md`: the tests that keep code, docs, and release claims aligned

## Reading paths

If you are new to the repo:

1. Read `../README.md`
2. Read `system.md`
3. Pick the subsystem doc that matches the surface you are changing

If you are changing runtime or release behavior:

1. `runtime-cli.md`
2. `contracts-manifests.md`
3. `benchmarks-readiness.md`

If you are changing search, fitting, or publication logic:

1. `modeling-pipeline.md`
2. `search-core.md`
3. `testing-truthfulness.md`

If you want the equation story end to end:

1. `modeling-pipeline.md`
2. `search-core.md`
3. `workbench.md`

If you are changing the local UI:

1. `workbench.md`
2. `testing-truthfulness.md`
