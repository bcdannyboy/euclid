---
title: Blocked And Invalid Fixtures
status: active_reference
scenario_ids:
- shared-plus-local-contract-failure
- illegal-time-access
- illegal-ref-shape
- unresolved-scope-violation
fixture_bundles:
- fixtures/canonical/negative/shared-plus-local-contract-failure.yaml
- fixtures/canonical/negative/illegal-time-access.yaml
- fixtures/canonical/invalid/illegal-ref-shape.yaml
- fixtures/canonical/invalid/unresolved-scope-violation.yaml
related:
- fixture-walkthroughs.md
- ../modeling-pipeline.md
- ../search-core.md
- ../contracts-manifests.md
---
# Blocked And Invalid Fixtures

This walkthrough page binds the listed scenarios to their executable fixture bundles and keeps prose subordinate to the manifests.

## Scenario Bundles

  - shared-plus-local-contract-failure: [fixtures/canonical/negative/shared-plus-local-contract-failure.yaml](../../../fixtures/canonical/negative/shared-plus-local-contract-failure.yaml)
  - illegal-time-access: [fixtures/canonical/negative/illegal-time-access.yaml](../../../fixtures/canonical/negative/illegal-time-access.yaml)
  - illegal-ref-shape: [fixtures/canonical/invalid/illegal-ref-shape.yaml](../../../fixtures/canonical/invalid/illegal-ref-shape.yaml)
  - unresolved-scope-violation: [fixtures/canonical/invalid/unresolved-scope-violation.yaml](../../../fixtures/canonical/invalid/unresolved-scope-violation.yaml)

## Evidence Boundary

Fixture success is executable coverage for the scenario, not standalone scientific claim evidence. Claim scope still comes from the scorecard, claim or abstention artifact, replay bundle, publication record, and readiness policy.
