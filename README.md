---
title: Euclid
status: active_reference
related:
- docs/reference/README.md
- docs/reference/system.md
- schemas/core/euclid-system.yaml
- schemas/core/source-map.yaml
---
# Euclid

In Darren Aronofsky's 1998 film [_Pi_](https://en.wikipedia.org/wiki/Pi_%28film%29), Max Cohen builds a homemade computer named Euclid to hunt patterns in the stock market. Movie Euclid does what any good paranoid math thriller needs it to do: it spits out a mysterious 216-digit number, crashes, and sends everyone looking for either market domination or the name of God. Great cinema. Terrible release process.

This Euclid is a realistic effort at time-series pattern derivation. It asks the fun question from the movie in a boringly useful way: if ordered data really does contain a compact pattern, can we derive it, write it down as an equation, and make it survive evidence checks before anyone gets too excited?

Most forecasting stacks give you a score, a chart, and a request for trust. Euclid tries to give you an equation and then make that equation earn the right to be believed.

Euclid is a forecasting system for ordered observations. It derives compact equations that describe a time series, then tests whether those equations have predictive support under replay, comparator, calibration, and publication gates.

Euclid keeps a descriptive equation separate from predictive claims, so a curve that summarizes the sample does not automatically get to masquerade as a forecast. When the deterministic and probabilistic lanes both clear the same publication scope, Euclid can surface one unified equation: a readable deterministic backbone joined to an explicit stochastic rule. When the evidence falls short, Euclid downgrades the claim or abstains instead of bluffing.

## The movie version vs. this repo

Movie Euclid chases total hidden order, attracts mystics and financiers, and has a very poor relationship with system reliability.

Repo Euclid does something less cinematic and more useful:

- accepts that a time series may contain structure, noise, both, or neither
- searches for compact symbolic descriptions and reducer laws
- scores those candidates against ordered holdout data
- keeps stochastic uncertainty explicit instead of pretending the equation knows everything
- records replayable artifacts so a result can be inspected later
- refuses to promote a pattern into a stronger claim unless the current implementation and tests support it

## System surfaces

The codebase is organized around five cooperating surfaces:

- Main CLI and run engine in `src/euclid/cli`, `src/euclid/operator_runtime`, `src/euclid/control_plane`, `src/euclid/storage`, and `src/euclid/artifacts`
- Modeling pipeline in `src/euclid/modules`
- Search, CIR, reducer, and descriptive-coding core in `src/euclid/search`, `src/euclid/cir`, `src/euclid/reducers`, `src/euclid/adapters`, and `src/euclid/math`
- Formal specs, manifests, benchmarks, readiness, and packaging in `src/euclid/contracts`, `src/euclid/manifests`, `src/euclid/benchmarks`, `src/euclid/readiness`, and `schemas`
- Local workbench UI in `src/euclid/workbench` and `src/euclid/_assets/workbench`

Runtime assets are resolved from packaged resources under `src/euclid/_assets`
through `euclid.operator_runtime.resources`. Checkout-root asset mirrors are not
part of the runtime contract unless explicitly documented.

## Scope model

Euclid distinguishes three release scopes in code and packaged policies:

- `current_release`: the certified operator surface that the root CLI exposes
- `full_vision`: the broader capability matrix used by readiness and benchmark closure
- `shipped_releasable`: the packaging and clean-install view of the certified surface

The CLI labels `demo` and related notebook-friendly flows as compatibility-only tooling. They remain available for inspection and retained demos, but they are not the certified runtime path.

## Run the certified surface

Certified entrypoints:

- `python -m euclid smoke`
- `euclid run --config examples/current_release_run.yaml`
- `euclid replay --run-id <run-id>`
- `euclid benchmarks run --suite current-release.yaml`
- `euclid release status`
- `euclid release verify-completion`
- `euclid release certify-research-readiness`
- `euclid workbench serve`

Compatibility-only entrypoints:

- `euclid demo run`
- `euclid demo replay`
- `euclid demo point run`
- `euclid demo probabilistic run`
- `euclid demo algorithmic run`

## Start here

- `docs/reference/README.md`: the release-to-implementation bridge and recommended reading paths through the docs
- `docs/reference/system.md`: the big picture, the main subsystems, and the end-to-end lifecycle
- `docs/reference/runtime-cli.md`: the main CLI surface, replay model, workspace layout, and extension lanes
- `docs/reference/modeling-pipeline.md`: how Euclid turns ordered data into candidate laws, scores, and publication outcomes
- `docs/math.md`: the implemented formulas for coding, fitting, scoring, stochastic support, rewrites, invariance, and promotion boundaries
- `docs/reference/search-core.md`: the mathematical search layer, CIR, reducer families, description gain, and algorithmic DSL
- `docs/reference/contracts-manifests.md`: the formal specification layer for schemas, manifests, object identity, and registries
- `docs/reference/benchmarks-readiness.md`: benchmarks, release scopes, readiness rules, and packaging outputs
- `docs/reference/workbench.md`: the local analysis studio, saved-analysis model, and explainer flow
- `docs/reference/testing-truthfulness.md`: executable documentation anchored in tests
- `schemas/core/euclid-system.yaml`: canonical document spine, vocabulary roots, runtime module map, and artifact classes
- `schemas/core/source-map.yaml`: code-to-document routing table for the live reference workspace

## Repository map

- `src/euclid`: package source
- `tests`: unit, integration, benchmark, frontend, and spec-compiler assertions
- `schemas`: formal vocabularies, artifact schemas, and readiness policies
- `benchmarks`: benchmark task, suite, and baseline manifests
- `examples`: runnable manifests and example datasets
- `tools/spec_compiler`: optional consistency tooling for schema and fixture audits
- `scripts`: smoke, release, and UI verification entrypoints

## What Euclid emits

Euclid emits typed, replayable research artifacts rather than one opaque prediction blob. The main families include:

- symbolic descriptive fits, predictive-within-scope symbolic claims, and, when scope-aligned deterministic and probabilistic evidence clears, unified equations for a time series
- dataset snapshots and feature views
- search and evaluation plans
- fitted candidate specs and prediction artifacts
- point or probabilistic score results
- calibration, robustness, and mechanistic evidence artifacts
- scorecards, claim cards, and abstentions
- reproducibility bundles and replay verification outputs
- run results, publication records, and readiness judgments

## Operating principles

- Time only moves forward. The code audits `event_time`, `available_at`, and revision visibility before model fitting.
- Search classes say exactly what they have and have not explored. Exact, bounded, equality-saturation, and stochastic backends do not make the same promise.
- Publication is gated. Replay, comparator validity, calibration, robustness, and readiness all affect what can be claimed. A symbolic result still has to earn the stronger claim.
- The workbench stays conservative. It separates descriptive structure from publishable predictive statements and rejects stale or synthetic claim payloads.
