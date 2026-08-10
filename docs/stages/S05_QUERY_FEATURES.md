# S05 — Query features and request state

## Goal

Implement prompt-only route features and request-specific route storage without a learned router.

## Tasks

- Use incoming prompt hidden states during prefill as the route input.
- Mean-pool only non-padding prompt positions as the initial feature per D007.
- Define request state for selected attention and FFN routes.
- Store routes during prefill and reuse them during decoding.
- Validate batch-size-one request lifecycle without cross-request caching.

## Tests

- Padding does not contribute to the initial mean pool.
- Empty or invalid prompt masks have explicit documented behavior.
- Routes are stored per request and reused during decoding.
- A second request cannot observe route state from the first.
- Manual route execution remains deterministic.

## Required outputs

- Feature and request-state interface.
- Prompt masking and request isolation tests.
- Route reuse report with exact commands and seed.
- Updated decisions and status.

## Known uncertainties

- Feature dimensionality, normalization, and storage schema beyond D007 remain unspecified.
- Exact interaction between model hooks and request lifetime remains to be verified.

## CONTINUE condition

Prompt-only features and request-specific route reuse work deterministically for batch size one without a learned router or cross-request cache.

## PAUSE condition

Model execution or request lifecycle integration cannot be exercised yet.

## REVISE condition

A feature or request-state assumption needs correction and is recorded before retesting.

## STOP condition

The route cannot be tied to prompt prefill and request-scoped decoding without inventing unrecorded behavior.
