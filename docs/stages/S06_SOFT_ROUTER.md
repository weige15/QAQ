# S06 — Trainable soft router

## Goal

Implement the router network and differentiable mixture of the 4-bit and 8-bit operation outputs.
Trainable parameters must be router parameters only.

## Tasks

- Build the smallest router consistent with the S05 feature and request-state contracts.
- Produce soft 4-bit/8-bit weights for attention and FFN decisions separately.
- Mix the two packed operation outputs differentiably for training.
- Freeze all quantized model weights and all non-router parameters.
- Record the router parameterization, initialization, temperature, and any other assumptions.

## Tests

- Only router parameters receive gradients and updates.
- Attention and FFN router outputs are separate.
- Soft mixtures are differentiable and numerically stable.
- Frozen quantized operations produce no parameter updates.
- Deterministic initialization and seed reproduce the same route probabilities.

## Required outputs

- Router module and focused tests.
- Parameter-freeze verification.
- Soft-mixture correctness and stability report.
- Updated decisions and status.

## Known uncertainties

- Router architecture, feature projection, temperature, and optimization details are unspecified until chosen and recorded.
- The source papers' exact differentiable routing formulation remains to be established by source review.

## CONTINUE condition

A deterministic soft router and differentiable 4/8-bit mixture work while only router parameters are trainable.

## PAUSE condition

Training data, teacher outputs, or required model execution is unavailable.

## REVISE condition

The router interface or parameterization requires a recorded correction.

## STOP condition

The implementation updates quantized weights, cannot produce separate routes, or requires an unrecorded objective or mechanism.
