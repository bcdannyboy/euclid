# Negative Publication Fixtures

These fixtures are where Euclid’s publication discipline is easiest to see. The bundles remain valid object graphs, and most of them still close a lifecycle, but the strongest candidate does not get to tell the strongest possible story. Some runs abstain. Some are downgraded. Some preserve typed evidence for a failed upgrade instead of pretending the upgrade never happened.

## Fixture Bundles

- [abstention-only-publication](../../fixtures/canonical/negative/abstention-only-publication.yaml): typed abstention with `publication_completed`
- [inadmissible-candidate](../../fixtures/canonical/negative/inadmissible-candidate.yaml): rejected candidate removed before a descriptive-only publication
- [failed-predictive-promotion](../../fixtures/canonical/negative/failed-predictive-promotion.yaml): predictive upgrade attempt downgraded to descriptive-only publication
- [probabilistic-gate-failure](../../fixtures/canonical/negative/probabilistic-gate-failure.yaml): probabilistic evaluation evidence preserved, but the upgrade is blocked
- [mechanistic-evidence-insufficiency](../../fixtures/canonical/negative/mechanistic-evidence-insufficiency.yaml): mechanistic ambition falls back to plain predictive support

## Walkthrough

### `abstention-only-publication`

This is the canonical typed abstention case. The bundle is accepted, replay still verifies, the publication record is still written, and the lifecycle still reaches `publication_completed`, but the result mode is `abstention_only_publication`. The abstention manifest carries `no_admissible_reducer`, and the scorecard records that the descriptive gate failed because no candidate survived search.

### `inadmissible-candidate`

This fixture proves that a bundle can recover from a rejected candidate without collapsing the whole publication path. The validator accepts the bundle, the lifecycle still completes, and the surviving publication is public and `descriptive_only`. What disappears is the inadmissible candidate, not the typed publication machinery around it.

### `failed-predictive-promotion`

This bundle is a downgrade, not an abstention. The point forecast still publishes, but only as `descriptive_only`. Read it as the negative mirror of the point-publication foundation: the predictive story was attempted and rejected, and the bundle keeps the lower descriptive result instead of overclaiming.

### `probabilistic-gate-failure`

This fixture does the same kind of downgrade on the probabilistic side. The forecast object stays `distribution`, typed evaluation evidence remains in the bundle, and the lifecycle still reaches `publication_completed`, but the claim lane settles at `descriptive_only` because the probabilistic predictive gate did not clear.

### `mechanistic-evidence-insufficiency`

This bundle shows how Euclid handles an ambitious interpretation that lacks enough extra evidence. The run does not fail outright. Instead, it falls back to the strongest supported lower claim: `predictively_supported`. That makes it the negative companion to `mechanistic-publication`.

## What To Check

- Abstention-only publication still ends in `publication_completed`, but only with the abstention publication mode.
- Downgraded scenarios encode their failure through typed artifacts and lifecycle outcomes, not prose-only notes.
- Negative fixtures are not all the same shape: some end with no claim lane at all, some publish descriptively, and some preserve predictive support while dropping a stronger interpretive lane.
- Distinguish “bundle accepted but strongest claim reduced” from “publication blocked” and from “fixture rejected.” Those are different pages in this directory for a reason.
- [Modeling pipeline](../modeling-pipeline.md) and [Contracts and manifests](../contracts-manifests.md) define the allowed abstention, rejection, and downgrade rules.
