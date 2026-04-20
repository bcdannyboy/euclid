# Euclid Canonical Pack

## Project
- Name: Fixture Euclid
- Scope: Fixture scope statement for the Euclid canonical-pack compiler.

## Canonical Docs
- Fixture Euclid Project (`README.md`) | Related: [[Fixture System Map]]
- Fixture System Map (`docs/system.md`) | Related: [[Fixture Euclid Project]]

## Vocabularies
- abstention_types: `schemas/core/abstention-types.yaml` -> no_admissible_reducer
- claim_lanes: `schemas/core/claim-lanes.yaml` -> descriptive_only
- composition_operators: `schemas/core/composition-operators.yaml` -> piecewise
- evidence_classes: `schemas/core/evidence-classes.yaml` -> descriptive_compression
- forecast_object_types: `schemas/core/forecast-object-types.yaml` -> point
- reducer_families: `schemas/core/reducer-families.yaml` -> analytic
- scope_axes: `schemas/core/scope-axes.yaml` -> entity_scope

## Runtime Modules
- core: manifest_registry, evaluation

## Artifact Classes
- publication: claim_card, publication_record

## Contract Artifacts
- `schemas/contracts/module-ownership.json` (module_ownership) | Owners: control_plane, evaluation_team | Modules: manifest_registry
- `schemas/readiness/readiness-charter.yaml` (readiness_charter) | Owners: readiness_council | Modules: None

## Math Fixtures
- None

## Validation Summary
- required_refs_checked: 9
- closed_vocabularies_loaded: 7
- contract_artifacts_loaded: 2
- math_fixtures_loaded: 0
- owners_declared: 3
