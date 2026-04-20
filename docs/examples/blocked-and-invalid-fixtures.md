# Blocked And Invalid Fixtures

These fixtures separate two failure shapes that are easy to conflate in prose and should never be conflated in the manifests.

- A blocked publication bundle is structurally valid, accepted by the validator, and far enough along to produce typed stop evidence, but the runtime rules require the lifecycle to halt.
- An invalid bundle is rejected as a fixture before publication semantics even begin.

## Fixture Bundles

- [shared-plus-local-contract-failure](../../fixtures/canonical/negative/shared-plus-local-contract-failure.yaml): valid bundle, blocked publication
- [illegal-time-access](../../fixtures/canonical/negative/illegal-time-access.yaml): valid bundle, blocked publication
- [illegal-ref-shape](../../fixtures/canonical/invalid/illegal-ref-shape.yaml): invalid fixture, rejected before lifecycle entry
- [unresolved-scope-violation](../../fixtures/canonical/invalid/unresolved-scope-violation.yaml): invalid fixture, rejected because declared scope structure is incoherent

## Walkthrough

### `shared-plus-local-contract-failure`

This blocked bundle is the clearest shared-plus-local stop condition. The decomposition policy is `deferred_non_binding`, publication is `forbidden_in_v1`, unseen-entity handling is also forbidden, and the request resolution is already marked `blocked`. The validator still accepts the bundle, but the lifecycle halts at `publication_blocked` with `shared_plus_local_decomposition_deferred_from_scope` carried forward as typed evidence.

### `illegal-time-access`

This is the time-safety stop case. The bundle is valid enough to enter the lifecycle, but the predictive path stops before claim binding because time safety fails. The result mode is `publication_blocked`, not abstention and not descriptive fallback.

### `illegal-ref-shape`

This fixture never becomes a runtime publication case. Typed-ref validation fails before lifecycle entry, so the expected outcome is `publication_mode: none` and `terminal_lifecycle_state: not_applicable_fixture_rejected`. It is useful as the smallest proof that malformed object references are a fixture-construction error, not a publishable negative result.

### `unresolved-scope-violation`

This fixture is the scope-level companion to `illegal-ref-shape`. The declared scope axes and deferred scope refs do not cohere, so the validator rejects the bundle before publication semantics apply. Like `illegal-ref-shape`, it is invalid rather than merely blocked.

## What To Check

- Blocked publication scenarios terminate in `publication_blocked` with typed evidence for the stop condition.
- Invalid fixtures never enter a publication lifecycle; they fail because refs or scope declarations violate the declared manifest rules.
- `shared-plus-local-contract-failure` and `illegal-time-access` are valid negative runtime outcomes, not malformed fixtures.
- `illegal-ref-shape` and `unresolved-scope-violation` should never be described as abstentions or downgrades; they are fixture rejections.
- [Search core](../search-core.md) and [Contracts and manifests](../contracts-manifests.md) describe the specification and typed-ref boundaries these fixtures are meant to exercise.
