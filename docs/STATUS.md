Current stage: S00
Status: COMPLETE
Last passing commit: to be recorded immediately after this evidence commit.
Completed in this pass:
- Environment evidence was re-audited and corrected to the active Python 3.12.3 / PyTorch 2.2.2+cu121 / Transformers 4.39.3 snapshot in `docs/environment.json`.
- Any-Precision provenance, compatibility, and exact-revision evidence remain internally consistent.
- Source-paper evidence was reviewed and separated from implementation choices.
- The local QAQ paper was verified to report Qwen3-4B among its evaluated models.
- The initial target was selected as the official `Qwen/Qwen3-4B` repository and `main` was resolved to immutable revision `1cfa9a7208912126459214e8b04321603b3df60c`.
- Model and tokenizer identity were recorded in `configs/model.yaml`; no model weights were downloaded.
- The pinned Qwen3 configuration and official Transformers source establish the exact class hierarchy, dimensions, projections, normalization, rotary, cache, and tied-weight behavior without model instantiation.
- The complete 252-module target list was generated and cross-checked as 144 attention plus 108 FFN projections with no duplicates.
- Any-Precision support was separated into explicit support (not present for Qwen3) and structural mappability (present for the seven standard linear targets).
- Repository cleanliness and absence of target-model artifacts were checked.
- A fresh recursive clone resolved the exact Any-Precision submodule revision with clean status.

Observed prerequisite comparison: Python 3.11 FAIL (3.12.3 installed), CUDA Toolkit 12+ PASS (12.4), GCC 9+ PASS (12.4.0), and PyTorch CUDA smoke check PASS. Python 3.12 compatibility remains an empirical local result, not an upstream support claim. The installed Transformers 4.39.3 source lacks Qwen3; later S01 work must validate the runtime under a Transformers version containing Qwen3.
Next action: Begin S01: validate the pinned Any-Precision packed linear backend at 4-bit and 8-bit precision.

Do not execute the next action automatically.
