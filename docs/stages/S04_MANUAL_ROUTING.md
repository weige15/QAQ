# S04 — Manual attention/FFN precision plans

## Goal

Prove individual attention and FFN units can independently consume selected precisions without a learned router.

## Tasks

- Define attention and FFN unit boundaries from the inspected target model.
- Route attention and FFN units independently using explicit manual plans.
- Keep all projections inside one selected unit at the unit's selected precision.
- Exercise mixed 4-bit/8-bit plans without introducing a learned router.
- Compare manual routed execution against the static baselines and reference paths.

## Tests

- Attention-only and FFN-only route changes affect the intended units only.
- Mixed manual plans execute deterministically.
- Unit projection precision is consistent within each selected unit.
- Outputs and resource measurements use packed planes.

## Required outputs

- Manual route-plan format and examples.
- Unit isolation and mixed-plan tests.
- Correctness and resource report.
- Updated decisions and status.

## Known uncertainties

- Exact module boundaries and execution hooks may expose model-specific complications.
- The effect of independent routes on numerical behavior is unknown.

## CONTINUE condition

Independent manual attention and FFN plans execute reproducibly with unit-level precision control and no learned router.

## PAUSE condition

A model integration dependency or required hardware capability is unavailable.

## REVISE condition

The unit boundary or route-plan interface needs correction without changing the separate-routing objective.

## STOP condition

Independent unit routing cannot be isolated or would require silently changing the stated scope.
