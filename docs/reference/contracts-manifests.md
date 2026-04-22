---
title: Contracts And Manifests
status: active_reference
related:
- system.md
- modeling-pipeline.md
- search-core.md
- benchmarks-readiness.md
- ../../schemas/contracts/module-registry.yaml
- ../../schemas/contracts/schema-registry.yaml
- ../../schemas/contracts/reference-types.yaml
- ../../schemas/contracts/enum-registry.yaml
---
# Contracts And Manifests

Euclid's contract and manifest layer keeps schemas, runtime artifacts, and release claims on one typed specification spine. It is the formal specification layer that lets public documentation, machine-readable artifacts, and packaged release evidence refer to the same objects.

## Formal specification layer

The contract layer lives in `src/euclid/contracts`.

Important pieces:

- `loader.py`: loads YAML contracts into a `ContractCatalog`
- `refs.py`: typed-ref validation and allowed target-family/version checks
- `errors.py`: contract validation error types

The source specifications live in `schemas/`:

- `schemas/core`: closed vocabularies and source maps
- `schemas/contracts`: module, schema, ref, lifecycle, and domain rules
- `schemas/readiness`: readiness matrix and policy surfaces

## Manifest envelope

The manifest base layer lives in `src/euclid/manifests/base.py`.

`ManifestEnvelope.build(...)` verifies that:

- the schema exists
- the declaring module owns that schema
- typed refs in the body conform to the contract catalog

Only then does it canonicalize the payload and assign a stable object id and hash.

## Runtime manifest families

`src/euclid/manifests/runtime_models.py` defines the concrete artifact families used by the runtime.

The main classes cover:

- dataset and feature artifacts
- search and evaluation plans
- candidate specs and prediction artifacts
- point score and calibration results
- robustness reports
- scorecards, claims, and abstentions
- reproducibility bundles
- run results and publication records
- readiness judgments and lifecycle closures

## Registry and lineage

`src/euclid/manifests/registry.py` persists manifests and records:

- content hashes
- lineage edges
- typed-reference edges

This makes the registry a typed provenance and reproducibility record, not just a log.

## Formal spec assets

Two kinds of machine-readable assets matter here:

- schemas and vocabularies under `schemas/`
- packaged release-support assets under `src/euclid/_assets/docs/implementation/*.yaml`

The packaged release-support assets are consumed by release code in `src/euclid/release.py`. They cover authority snapshot, closure map, traceability, fixture spec, evidence policy, and command policy.

## Source anchors

- `src/euclid/contracts/loader.py`
- `src/euclid/contracts/refs.py`
- `src/euclid/manifests/base.py`
- `src/euclid/manifests/runtime_models.py`
- `src/euclid/manifests/registry.py`
- `schemas/core/euclid-system.yaml`
- `schemas/contracts/module-registry.yaml`
- `schemas/contracts/schema-registry.yaml`
- `schemas/contracts/reference-types.yaml`
- `schemas/contracts/enum-registry.yaml`


## Registry contracts

The live machine-readable registries are `schemas/contracts/module-registry.yaml`, `schemas/contracts/schema-registry.yaml`, `schemas/contracts/reference-types.yaml`, and `schemas/contracts/enum-registry.yaml`. `ManifestEnvelope.build` verifies schema ownership, typed refs, and canonical payload identity before a manifest becomes a runtime artifact.

The registry covers scorecards, claims, and abstentions; reproducibility bundles; run results and publication records; lifecycle contracts; expression IR; rewrite traces; optimizer diagnostics; PySINDy and PySR engine traces; stochastic-law evidence; falsification dossiers; invariance evaluation; event definitions; prequential score streams; and paired predictive test results.

Packaged release-support contracts live under `src/euclid/_assets/docs/implementation/*.yaml`, and the source copies live under `docs/implementation/*.yaml`. They bind authority snapshots, command contracts, fixture specs, evidence policy, closure mapping, lifecycle proof families, and subtask traceability to the same release story.
