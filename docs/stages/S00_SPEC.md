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

## Current evidence audit

| Requirement | Status | Evidence location | Unresolved issue |
| --- | --- | --- | --- |
| Mandatory `~/.venv` activation, Python path, and Python version | COMPLETE | `docs/environment.json`; `scripts/inspect_environment.py` command record | None; Python 3.12.3 is the observed interpreter. |
| Operating system, kernel, CPU RAM, disk availability, GPU model/count/VRAM, CUDA toolkit, compiler, PyTorch, CUDA availability, and Transformers evidence | COMPLETE | `docs/environment.json`; D013 | The snapshot is dated by D013 (2026-08-11); Python 3.11 remains an upstream-documented prerequisite comparison that is not met. |
| Minimal PyTorch CUDA operation | COMPLETE | `docs/environment.json` (`result_values`: 4.0 and 6.0) | None. |
| Python 3.12 compatibility wording | COMPLETE | D002; `docs/SOURCE_NOTES.md` | Local empirical result only; upstream Any-Precision README still lists Python 3.11 and is not rewritten. |
| Any-Precision upstream URL, full SHA, commit date, representation, branch state, and clean dependency state | COMPLETE | D001-D002; `docs/SOURCE_NOTES.md`; `.gitmodules`; submodule gitlink | None; the exact SHA is pinned even though the checkout currently has branch `main`. |
| Any-Precision package import and CUDA/backend compatibility check | COMPLETE | D002; `docs/SOURCE_NOTES.md` exact command block | None; no model quantization or full benchmark was run. |
| Clean-checkout reproducibility without untracked current-directory files | COMPLETE | `docs/SOURCE_NOTES.md`; clean recursive clone check recorded below | None once the documented recursive-clone check passes. |
| Source-paper review separating supported claims from assumptions | COMPLETE | `docs/SOURCE_NOTES.md`, `S00 source review`; D001-D012 | The local `QAQ.pdf` remains a local artifact without independently verified public provenance. |
| D001-D012 decision ledger and source-supported-versus-implementation distinction | COMPLETE | `docs/DECISIONS.md` | None; D003-D012 remain implementation choices. |
| S00 proposed next-stage command set | COMPLETE | `S01_BACKEND.md`; command set below | Commands are a proposal only and must not be run in this task. |
| Target model repository and exact target-model revision | COMPLETE | `configs/model.yaml`; D014; `docs/SOURCE_NOTES.md` | The exact revision used by QAQ authors is not identified by the local paper; the pinned revision is our implementation choice. |
| Target tokenizer, evaluation inputs, and reproducibility identifiers | PARTIAL | `configs/model.yaml`; D014; `docs/SOURCE_NOTES.md` | Tokenizer repository/revision and files are recorded; evaluation inputs remain for a later S00 step. |
| Target architecture class and Transformer layer count | DEFERRED_BY_INSTRUCTION | S00 task list and known uncertainties | Architecture inspection is explicitly deferred; no model was instantiated. |
| Target hidden size, attention projection names, and FFN projection names | DEFERRED_BY_INSTRUCTION | S00 task list and known uncertainties | Architecture inspection is explicitly deferred. |
| Target tied-weight, bias, embedding, and output-head behavior | DEFERRED_BY_INSTRUCTION | S00 task list and known uncertainties | Architecture inspection is explicitly deferred. |
| Backend/model combination viability and target-model structure summary | DEFERRED_BY_INSTRUCTION | S00 required outputs and CONTINUE condition | Any-Precision/Qwen compatibility remains unproven until architecture mapping. |
| No full-model quantization, implementation-stage experiment, target-model artifact, or S01 code | COMPLETE | `docs/environment.json`; tracked-file audit; current tree | None. |
| Tracked repository cleanliness for generated caches, build artifacts, temporary files, and papers | COMPLETE | `.gitignore`; `git ls-files`; paper hashes; current Git status | Ignored build/cache files are not tracked; no cleanup was needed. |

The proposed S01 command set is documentation only:

```bash
source ~/.venv/bin/activate
which python
python --version
CUDA_VISIBLE_DEVICES=0 python -m pytest tests/unit tests/integration -q
```

The final command is intentionally not executable yet because S01 must first add the focused packed 4-bit and 8-bit tests and their reference contract. No S01 command was run during this audit.

Clean recursive-clone verification command:

```bash
rm -rf /tmp/qaq-s00-clean
git clone --recurse-submodules /nfs/home/s314511048/firstmate/projects/QAQ /tmp/qaq-s00-clean
test -z "$(git -C /tmp/qaq-s00-clean status --porcelain=v1)"
test -z "$(git -C /tmp/qaq-s00-clean/third_party/any-precision-llm status --porcelain=v1)"
test "$(git -C /tmp/qaq-s00-clean/third_party/any-precision-llm rev-parse HEAD)" = a3257d02740cc5757c78673da534b0630ff3a4ea
rm -rf /tmp/qaq-s00-clean
```

Result: PASS after evidence commit `279ae2137f8a2c6017feeb2cda8660b5ed79214c`; the clean clone and initialized submodule were both clean, and the submodule resolved to `a3257d02740cc5757c78673da534b0630ff3a4ea`.

## Known uncertainties

- The target repository revision and tokenizer identity are established, but the exact revision used by QAQ authors remains unknown.
- Evaluation inputs, target architecture behavior, and model-specific backend viability remain unverified until the next authorized S00 step.
- The papers do not establish several QAQ implementation details, including exact route features, hard-route timing, loader lifetime, and the initial 4/8-bit scope.

## CONTINUE condition

The environment, target-model identity, and source revision are documented with reproducible evidence; full S00 completion still requires the separately authorized architecture and backend/model viability checks.

## PAUSE condition

An external dependency, credential, hardware allocation, or source clarification is needed before the remaining S00 evidence can be collected.

## REVISE condition

The target specification, backend choice, or an initial implementation assumption must change based on evidence; record the change and return S00 to its affected checks.

## STOP condition

The mandatory environment cannot be activated or verified, the proposed backend/model combination is infeasible, or the specification cannot be made reproducible without inventing unresolved details.
