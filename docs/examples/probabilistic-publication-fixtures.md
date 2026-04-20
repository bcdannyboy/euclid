# Probabilistic Publication Fixtures

These fixtures show that Euclid’s probabilistic publication path is not one generic “probability mode.” Each forecast object type carries its own score family, calibration policy, and gate semantics. What stays fixed is the publication discipline: each bundle still has to earn `predictively_supported`, survive replay and robustness, and close as `candidate_publication`.

## Fixture Bundles

- [probabilistic-publication-distribution](../../fixtures/canonical/publication/probabilistic-publication-distribution.yaml): `distribution`, `distribution_score`, required calibration, internal publication
- [probabilistic-publication-interval](../../fixtures/canonical/publication/probabilistic-publication-interval.yaml): `interval`, `interval_score`, required calibration, internal publication
- [probabilistic-publication-quantile](../../fixtures/canonical/publication/probabilistic-publication-quantile.yaml): `quantile`, `quantile_score`, required calibration, internal publication
- [probabilistic-publication-event-probability](../../fixtures/canonical/publication/probabilistic-publication-event-probability.yaml): `event_probability`, `event_probability_score`, required calibration, internal publication

## Walkthrough

All four bundles share the same outer shape. They finish as `predictively_supported`, publish through `candidate_publication`, terminate at `publication_completed`, and project into the internal catalog. The difference is the typed policy surface each forecast object must satisfy:

- `distribution` uses `distribution_score`, requires a distribution calibration policy, and demands `replicated` confirmatory status before publication.
- `interval` swaps in `interval_score` with interval-specific calibration while keeping the same confirmatory and lifecycle expectations.
- `quantile` does the same for `quantile_score`, proving that quantile forecasts are evaluated through their own manifest family rather than through a borrowed point or distribution policy.
- `event_probability` closes the set by using `event_probability_score` and required calibration for directional event forecasts.

Read together, the bundles show a consistent rule: probabilistic publication is type-specific at evaluation time and uniform at publication time. Euclid will only publish the object the bundle actually earned.

## What To Check

- Distribution, interval, quantile, and event-probability bundles each pair the forecast object type with the right probabilistic score and calibration policy family.
- Each bundle reaches `publication_completed` without downgrading out of the predictively supported lane.
- Each bundle uses internal catalog scope; none of them silently inherit the public point-publication examples’ catalog settings.
- Each bundle requires calibration as a gate, unlike the point-publication foundation where calibration is recorded but not gate-forming.
- [Modeling pipeline](../modeling-pipeline.md) defines the predictive-gate and calibration rules that these bundles encode.
