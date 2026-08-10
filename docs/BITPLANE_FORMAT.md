# Bit-plane format contract

This document is a specification placeholder for S02.
It must be completed from measured backend behavior and explicit implementation decisions, not guessed from terminology.

## Required decisions for S02

- plane order and bit significance;
- physical packing layout and word or byte boundaries;
- reconstruction equations and signed-value handling;
- padding rules;
- lookup representation;
- nested 4-bit and 8-bit relationship;
- serialized metadata and versioning;
- measured bytes for representative tensors.

Production representations must be physically bit-packed.
A byte-per-bit representation is permitted only as a test/reference oracle and must never support memory, transfer, or latency claims.
Packed planes, not unpacked weights, are the unit transferred by the baseline loader.

## Current uncertainty

No layout, sign convention, padding rule, lookup representation, or byte-count claim is established by this scaffold.
S02 must cite the pinned backend and record measurements before this contract is normative.
