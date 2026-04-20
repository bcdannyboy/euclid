# Extension-Lane Publication Fixtures

These fixtures cover the publication paths that depend on extension evidence rather than baseline point or probabilistic support alone. They are useful when you want to see how Euclid carries a stronger interpretation, a decomposition-specific surface, or an algorithmic discovery result through the same publication machinery without flattening everything into the default lane.

## Fixture Bundles

- [mechanistic-publication](../../fixtures/canonical/publication/mechanistic-publication.yaml): point forecast published as `mechanistically_compatible_hypothesis`
- [shared-plus-local-publication](../../fixtures/canonical/publication/shared-plus-local-publication.yaml): point forecast published with shared-plus-local decomposition surfaces intact
- [algorithmic-discovery-publication](../../fixtures/canonical/publication/algorithmic-discovery-publication.yaml): distribution forecast published through the algorithmic discovery lane

## Walkthrough

### `mechanistic-publication`

This is the strongest interpretive example in the set. The bundle publishes a point forecast into the `mechanistically_compatible_hypothesis` lane, not merely `predictively_supported`, while preserving the lower claim lane for comparison. Its scoring still uses `absolute_error`, calibration remains `forbidden_for_gate`, and the predictive gate requires `clean` confirmatory status. The extra lift comes from mechanistic evidence, not from a looser predictive policy.

### `shared-plus-local-publication`

This bundle keeps the claim lane at `predictively_supported`, but it proves that shared-plus-local decomposition surfaces can remain first-class publication objects instead of being stripped away before cataloging. It is still a point forecast with `absolute_error`, calibration forbidden for the gate, and `replicated` confirmatory status. What changes is the specification surface that has to survive into the published bundle.

### `algorithmic-discovery-publication`

This fixture shows that an algorithmic CIR path can still end as a normal publishable result rather than staying trapped in exploratory mode. The published object is `distribution`, the primary score is `log_score`, calibration is required, and confirmatory status must be `replicated`. The result is a predictively supported internal publication whose origin is algorithmic discovery rather than a hand-labeled extension.

## What To Check

- Mechanistic publication includes the extra evidence needed to publish the stronger interpretive lane without relaxing the underlying predictive checks.
- Shared-plus-local publication preserves the decomposition-specific specification surface that the search core and modeling pipeline describe.
- Algorithmic discovery publication exercises the algorithmic CIR lane while still producing a normal publishable bundle.
- Read the catalog scopes carefully: all three fixtures publish internally, even when the lifecycle closes successfully.
- [Search core](../search-core.md) and [Contracts and manifests](../contracts-manifests.md) define the composition rules and artifact requirements behind these fixtures.
