---
title: Benchmarks And Readiness
status: active_reference
related:
- ../../README.md
- runtime-cli.md
- contracts-manifests.md
- testing-truthfulness.md
- ../../benchmarks/suites/current-release.yaml
- ../../benchmarks/suites/full-vision.yaml
---
# Benchmarks And Readiness

Benchmarks and readiness are the evidence surfaces that state what Euclid can publish today, what remains aspirational, and which release claims are backed by runnable artifacts.

## Benchmark tasks and suites

The benchmark layer lives in:

- `src/euclid/benchmarks`
- `benchmarks/tasks`
- `benchmarks/suites`
- `benchmarks/baselines`

Each task manifest declares a frozen protocol that includes dataset references, forecast object type, split policy, quantization, scoring, calibration, budget, replay rules, and submitter selection.

Each suite groups tasks into capability surfaces and binds the fixture and scope metadata that define the exact public claim under test.

## Benchmark runtime flow

`src/euclid/benchmarks/runtime.py` performs the main flow:

1. load the benchmark task manifest
2. build the runtime intake context from the live specification layer
3. ingest the dataset
4. materialize operator and search context
5. run submitters
6. write report and replay artifacts
7. emit task and suite summaries with semantic coverage details

`src/euclid/benchmarks/reporting.py` writes the machine-readable report surfaces, including task result JSON, submitter result JSON, replay refs, and portfolio-selection evidence where applicable.

## Readiness scopes

The readiness layer lives in:

- `src/euclid/readiness/judgment.py`
- `schemas/readiness/current-release-v1.yaml`
- `schemas/readiness/full-vision-v1.yaml`
- `schemas/readiness/shipped-releasable-v1.yaml`
- `schemas/readiness/full-vision-matrix.yaml`
- `schemas/readiness/evidence-strength-policy.yaml`

The current codebase uses one full matrix and multiple policies that choose different required row sets.

- `current_release` is the certified operator subset
- `full_vision` is the broader capability target
- `shipped_releasable` follows the packaged and clean-install view of the certified subset

These scopes let the repo describe ambition without blurring what is already certified and shipped.

The evidence-strength policy controls what kinds of proof are strong enough to close a row.

## Release and packaging

Release verification lives in `src/euclid/release.py`.

It supports:

- release status reporting
- contract validation
- benchmark smoke execution
- determinism and performance smoke
- clean-install certification
- completion report generation and verification
- research-readiness certification

The release layer also loads packaged release-support assets from `src/euclid/_assets/docs/implementation/*.yaml`, so those packaged contracts remain part of the code's closure model even though the public docs now live as ordinary markdown files.

## Consistency tooling

`tools/spec_compiler/compiler.py` remains as optional consistency tooling for schema and fixture audits.

It can still build:

- canonical pack
- contract graph
- fixture closure pack
- readiness pack

The public documentation itself is no longer generated from that toolchain. The authoritative reader-facing docs are the markdown files in `README.md`, `docs/*.md`, and `docs/examples/*.md`.

## Source anchors

- `src/euclid/benchmarks/manifests.py`
- `src/euclid/benchmarks/runtime.py`
- `src/euclid/benchmarks/reporting.py`
- `src/euclid/benchmarks/submitters.py`
- `src/euclid/readiness/judgment.py`
- `src/euclid/release.py`
- `tools/spec_compiler/compiler.py`


## Benchmark semantics

Benchmarks are semantic evidence, not claim evidence by themselves. A passing benchmark can close a benchmark surface only when its report includes declared tracks, support objects, replay identity, engine provenance, score policies, threshold outcomes, and claim-scope boundaries. Scientific claim promotion still requires the modeling, contract, replay, and evidence gates for the specific claim lane.

The certified suite ids are `current_release` and `full_vision`; `shipped_releasable` is the packaging and clean-install scope. The release contracts that bind those scopes are `docs/implementation/*.yaml`, and the consistency compiler entrypoint remains `tools/spec_compiler/compiler.py`.
