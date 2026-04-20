# Fixture Walkthroughs

These walkthroughs are the readable layer over Euclid’s canonical fixture corpus. Each page explains what a fixture bundle is trying to prove, which claim lane or stop condition it exercises, and how the lifecycle should close when the bundle is valid.

The docs in this directory are plain markdown. The fixture inventory still lives in `fixtures/canonical/fixture-coverage.yaml`, but these pages are written for humans first: they explain what each bundle proves and where to inspect the evidence.

## Reading Order

1. [Publication foundations](./publication-foundations.md) for the baseline distinction between descriptive publication and predictively supported point publication.
2. [Probabilistic publication fixtures](./probabilistic-publication-fixtures.md) for the distribution, interval, quantile, and event-probability lanes.
3. [Extension-lane publication fixtures](./extension-lane-publication-fixtures.md) for mechanistic, shared-plus-local, and algorithmic publication paths.
4. [Negative publication fixtures](./negative-publication-fixtures.md) for abstentions, downgrades, and rejected upgrades that still end as valid bundles.
5. [Blocked and invalid fixtures](./blocked-and-invalid-fixtures.md) for runs that must stop before normal publication or fail validation outright.

## How To Read A Fixture Bundle

- Start with `expected_outcome`. It tells you the claim lane, forecast object type, publication mode, terminal lifecycle state, and catalog scope the fixture is supposed to realize.
- Follow `lifecycle_trace` next. That is the shortest proof that the bundle enters the right states, produces the right typed objects, and stops at the right place.
- Use the `artifacts` section to inspect why the verdict was allowed. Policies, scorecards, abstentions, run results, and publication records are the binding evidence, not the prose summary.
- Treat negative and blocked bundles carefully. Some are valid fixtures with `publication_completed`; others are valid fixtures with `publication_blocked`; invalid bundles never enter the lifecycle at all.

## Reading Guidance

- Use [Modeling pipeline](../modeling-pipeline.md) for lane, gate, calibration, and abstention semantics.
- Use [Search core](../search-core.md) for search backends, reducer families, CIR normalization, and adapter roles.
- Use [Contracts and manifests](../contracts-manifests.md) for schema families, typed refs, object identity, and ownership boundaries.
- Use [Fixture coverage plan](../../fixtures/canonical/fixture-coverage.yaml) for the closed scenario inventory behind these walkthroughs.
