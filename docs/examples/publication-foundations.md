# Publication Foundations

Start here for the minimum publication story Euclid supports. These two fixtures establish the baseline distinction between a symbolic result that is worth publishing descriptively and a point-valued forecast that has earned predictive support.

## Fixture Bundles

- [descriptive-publication](../../fixtures/canonical/publication/descriptive-publication.yaml): `descriptive_only`, `point`, public `candidate_publication`
- [predictive-publication-point](../../fixtures/canonical/publication/predictive-publication-point.yaml): `predictively_supported`, `point`, public `candidate_publication`

## Walkthrough

### `descriptive-publication`

This bundle is the clean proof that publication does not imply predictive support. The fixture reaches `publication_completed`, writes the normal publication record, and lands in the public catalog, but its claim lane remains `descriptive_only`.

The evidence profile is deliberate. The scorecard is driven by `description_length_delta`, the robustness report carries descriptive compression and robustness evidence, replay verification closes cleanly, and the claim card summary says only that descriptive compression survives replay and robustness checks. Nothing in the bundle upgrades that result into a predictive claim.

### `predictive-publication-point`

This bundle adds the extra machinery a point forecast must survive before Euclid will treat it as predictively supported. It introduces split planning and timeguard, materializes a prediction artifact, scores the forecast with `squared_error`, records calibration as `forbidden_for_gate`, and still requires `verified` time safety plus a `clean` confirmatory status.

When those typed surfaces line up, the claim lane advances to `predictively_supported` and the bundle still closes as ordinary public `candidate_publication`. This is the smallest closed example of a point-valued forecast that earns predictive publication rather than merely descriptive publication.

## What To Check

- Confirm the descriptive fixture never overclaims predictive support and still reaches `publication_completed`.
- Confirm the descriptive fixture’s publication record is public even though the strongest claim remains `descriptive_only`.
- Confirm the point-publication fixture materializes evaluation, calibration, scorecard, claim-card, replay, and publication artifacts as one closed bundle.
- Read the two bundles side by side: the difference is not “publication versus no publication,” but “descriptive publication versus predictively supported publication.”
- Cross-read [Modeling pipeline](../modeling-pipeline.md) for lane semantics and [Search core](../search-core.md) for reducer and CIR fields referenced by the manifests.
