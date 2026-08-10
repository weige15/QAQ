# S00 — Lock environment and specification

## Goal

Lock the execution environment and the baseline specification before implementation.
Capture hardware and software versions, pin dependencies, validate CUDA prerequisites, inspect the target model structure, pin source revisions, and determine whether the proposed backend/model combination is viable.
Do not quantize the full model.

## Tasks

- Activate and verify `~/.venv` using the mandatory project command.
- Capture hardware, driver, CUDA, framework, compiler, and tool versions with exact commands.
- Define the target model, revision, tokenizer, evaluation inputs, and reproducibility identifiers.
- Inspect the target model's attention and FFN unit structure without modifying model weights.
- Review the source papers and record source-supported claims separately from unknowns.
- Locate the official Any-Precision implementation, record its provenance, and pin the exact commit before any modification or wrapper work.
- Validate the proposed backend/model combination with a minimal viability check; do not quantize the full model.

## Tests

- Environment activation and `which python` resolves within `~/.venv`.
- Version and prerequisite probes complete with captured output.
- Target model structure inspection is deterministic and reproducible.
- The exact upstream revision is independently recoverable.
- No full-model quantization or implementation-stage experiment is run.

## Required outputs

- Environment and hardware report with exact commands.
- Source-review notes with citations, unknowns, and assumptions.
- Target model structure summary.
- Exact Any-Precision revision and provenance record.
- S00 viability report and a proposed next-stage command set.
- Updated `docs/DECISIONS.md` and `docs/STATUS.md`.

## Known uncertainties

- The target model and revision are not established by this scaffold.
- Backend/model compatibility, CUDA prerequisites, and source revision remain unverified.
- The papers' precise packing, routing, training, and loading details remain to be extracted.

## CONTINUE condition

The environment, target model, source revision, and backend/model viability are documented with reproducible evidence and no blocking unknown remains for S01.

## PAUSE condition

An external dependency, credential, hardware allocation, or source clarification is needed before the remaining S00 evidence can be collected.

## REVISE condition

The target specification, backend choice, or an initial implementation assumption must change based on evidence; record the change and return S00 to its affected checks.

## STOP condition

The mandatory environment cannot be activated or verified, the proposed backend/model combination is infeasible, or the specification cannot be made reproducible without inventing unresolved details.
