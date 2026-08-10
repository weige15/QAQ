# Source notes

This file records the preserved source inventory and the S00 source review.
Any source-supported claim is kept separate from QAQ implementation assumptions.

## Preserved source material

- `papers/QAQ.pdf`
- `papers/Any_Precision_LLM.pdf`
- `papers/dp_llm.pdf`
- `papers/PMPD.pdf`
- `papers/README.md`

The PDFs under `papers/` are project source material and are unchanged by this scaffold.
No claim about an unresolved paper detail is established here.

## Any-Precision dependency provenance (S00 source-pinning pass)

- **Dependency:** Any-Precision LLM.
- **Upstream URL:** `https://github.com/SNU-ARC/any-precision-llm.git`.
- **Exact commit:** `a3257d02740cc5757c78673da534b0630ff3a4ea`.
- **Commit date:** `2025-07-04T16:00:35+09:00`.
- **Local path:** `third_party/any-precision-llm`.
- **Representation:** Git submodule; `.gitmodules` stores the upstream URL and the gitlink stores the exact commit rather than a floating branch.
- **Checkout condition:** The identified source checkout was on `main` at the exact commit and clean before preservation; the preserved submodule is also clean after the build and smoke test.
- **Compatibility-test result:** PASS. Under `~/.venv` with Python `3.12.3`, `import any_precision` passed and a CUDA smoke test using the real `any_precision_ext.dequant_kbit` and `matmul_kbit` functions passed on an NVIDIA GeForce RTX 3090. No model quantization or full benchmark was run.
- **Environment record:** `docs/environment.json`; the Python 3.12.3 compatibility result is empirical on this machine, not an upstream support claim.
- **Exact verification commands:**
  ```bash
  source ~/.venv/bin/activate
  which python
  python --version
  cd /nfs/home/s314511048/firstmate/projects/QAQ/third_party/any-precision-llm/any_precision/modules/kernels
  python -m pip install --no-deps --no-build-isolation .
  cd /nfs/home/s314511048/firstmate/projects/QAQ
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD/third_party/any-precision-llm${PYTHONPATH:+:$PYTHONPATH}" python - <<'PY'
  import torch
  import any_precision
  import any_precision_ext
  from any_precision_ext import dequant_kbit, matmul_kbit
  qweight = torch.zeros((3, 4, 1), dtype=torch.int32, device="cuda")
  lut = torch.ones((4, 8), dtype=torch.float16, device="cuda")
  inputs = torch.ones((1, 32), dtype=torch.float16, device="cuda")
  assert torch.allclose(dequant_kbit(qweight, lut, 3), torch.ones((4, 32), device="cuda", dtype=torch.float16))
  assert torch.allclose(matmul_kbit(inputs, qweight, lut, 3), torch.full((1, 4), 32.0, device="cuda", dtype=torch.float16))
  torch.cuda.synchronize()
  PY
  ```

## S00 source review

The following claims are limited to the preserved local PDFs and their cited public metadata. They are source-supported observations, not endorsements of every implementation choice in `docs/DECISIONS.md`.

| Source | Source-supported behavior relevant to QAQ | Not established by the source and therefore not assumed here |
| --- | --- | --- |
| `papers/Any_Precision_LLM.pdf`, pp. 1, 3–5; https://arxiv.org/abs/2402.10517 | Any-Precision LLM describes PTQ incremental upscaling from a seed bit-width to a parent bit-width, overlaying multiple bit-width variants, and a bitplane-oriented software engine for reduced-bit memory access. | The paper does not select QAQ's target model, 4/8-bit-only scope, route granularity, loader lifetime, or Python 3.12 support. |
| `papers/dp_llm.pdf`, pp. 1–4; https://arxiv.org/abs/2508.06041 | DP-LLM describes changing layer sensitivity across decoding steps, uses relative error as a precision-selection proxy, and discusses lightweight runtime selectors over candidate precision pairs. | It does not establish QAQ's query feature, teacher-student objective, hard argmax policy, or separate attention/FFN route contract. |
| `papers/QAQ.pdf`, pp. 1–4; local workshop artifact | The local artifact describes query-conditioned routing, bit-plane storage, block-level MHA/FFN organization, teacher-student router training, CPU-to-GPU on-demand loading, and reports Qwen3-4B, Qwen3-8B, and Llama3.1-8B evaluations. | This local PDF has no independently verified public record in this repository and does not identify an exact Hugging Face repository revision. Its reported metrics and system details are not treated as validated evidence for S00. |
| `papers/PMPD.pdf`, pp. 2–3; local paper artifact | PMPD describes phase-aware and progressively lowering precision during decoding, motivated by differing error resilience across prefill and decoding. | PMPD does not authorize adding phase schedulers or progressive precision to the QAQ baseline before the documented freeze boundary. |

The source review therefore supports the choice to investigate Any-Precision as a storage/backend substrate and to keep paper claims separate from QAQ implementation choices. D003–D012 remain implementation choices unless later evidence explicitly changes them.

## Target-model provenance (S00 identity pass)

- **Source-supported model fact:** `papers/QAQ.pdf` reports Qwen3-4B as one of the evaluated models. The QAQ implementation plan chooses this smaller reported model for the initial baseline.
- **Selected repository:** `Qwen/Qwen3-4B`, the official Qwen repository. This exact repository identity is an implementation selection informed by the paper's `Qwen3-4B` name; the paper does not state the Hugging Face repository ID or revision.
- **Repository owner:** `Qwen`.
- **License:** Apache-2.0 (`license: apache-2.0` in official Hugging Face model metadata and the repository `LICENSE`).
- **Immutable revision:** `1cfa9a7208912126459214e8b04321603b3df60c`.
- **Revision date:** `2025-07-26T03:46:39Z`, from the first entry of the official Hugging Face commits API for `main`.
- **Tokenizer identity:** `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`; relevant files are `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, and `merges.txt`.
- **Other relevant repository files:** `config.json`, `generation_config.json`, `README.md`, and `LICENSE`.
- **Approximate repository size:** `8,060,926,626` bytes (`8.06 GB`, approximately `7.51 GiB`) according to the official Hugging Face tree API at the pinned revision. This includes three `.safetensors` weight shards; none were downloaded.
- **Revision-resolution commands:**
  ```bash
  source ~/.venv/bin/activate
  which python
  python --version
  curl --fail --silent --show-error --location \
    'https://huggingface.co/api/models/Qwen/Qwen3-4B/commits/main'
  curl --fail --silent --show-error --location \
    'https://huggingface.co/api/models/Qwen/Qwen3-4B/revision/main'
  ```
- **Exact-file verification:** The pinned `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, `README.md`, and `LICENSE` endpoints returned HTTP 200 at the immutable revision. Weight-shard endpoints were not downloaded; HEAD metadata only reported their sizes.
- **Task download accounting:** Ten temporary API/metadata files were downloaded under `/tmp` for this identity check, totaling `11,498,399` bytes. No weight shard or full snapshot was downloaded, and no model artifact was written into the project.
- **Resolution date:** `2026-08-10` UTC.
- **Implementation choices:** Pin the current `main` resolution as our reproducibility revision, use the same revision for the tokenizer, set `trust_remote_code: false`, and leave `local_weight_path` null. The exact revision used by QAQ authors remains unknown unless future source evidence states it.
- **Compatibility boundary:** Any-Precision/Qwen3 compatibility remains unproven until architecture inspection and backend mapping; repository accessibility does not establish compatibility.

## Qwen3 structure and backend-mapping evidence (S00 final pass)

- **Pinned model-derived source:** `Qwen/Qwen3-4B` configuration at revision `1cfa9a7208912126459214e8b04321603b3df60c`; exact URL and captured output are represented in `docs/model_structure.json`.
- **Configuration facts:** `Qwen3ForCausalLM`, `model_type: qwen3`, 36 layers, hidden size 2560, intermediate size 9728, vocabulary 151936, 32 attention heads, 8 key/value heads, head dimension 128, maximum position embeddings 40960, BF16 configured dtype, no attention bias, and tied word embeddings.
- **Transformers source:** Official Transformers `4.51.0` Qwen3 implementation at commit `0720e206c6ba28887e4d60ef60a6a089f6c1cc76`, with `configuration_qwen3.py` and `modeling_qwen3.py` inspected without importing or instantiating the model.
- **Runtime comparison:** The project environment's recorded Transformers version is `4.39.3`; its installed source has no `transformers.models.qwen3` package. This is an explicit runtime compatibility limitation, not an architecture ambiguity.
- **Observed classes and paths:** `Qwen3ForCausalLM` → `model: Qwen3Model` → `layers: ModuleList[Qwen3DecoderLayer]`; each layer contains `self_attn: Qwen3Attention`, `mlp: Qwen3MLP`, two `Qwen3RMSNorm` layer norms, and the attention's Q/K RMS norms. The base model also contains `embed_tokens`, `rotary_emb`, and final `norm`; the wrapper contains `lm_head`.
- **Target enumeration:** Four attention projections and three FFN projections per layer produce 144 attention targets, 108 FFN targets, and 252 unique total targets. All seven are bias-free standard linear modules and all input dimensions are divisible by 32.
- **Non-target behavior:** Q/K normalization, rotary position processing, activation/gating, all RMS norms, embeddings, tied output head, and KV-cache state remain outside packed linear replacement.
- **Any-Precision source facts:** The pinned revision has explicit architecture YAMLs for `LlamaForCausalLM`, `MistralForCausalLM`, `OPTForCausalLM`, and `PhiForCausalLM`, but no Qwen3 YAML. Its fallback analyzer scans first-layer `torch.nn.Linear` modules and warns that automatic detection may be incorrect. Therefore Qwen3 is structurally mappable but not explicitly supported.
- **Inspection artifacts:** `docs/model_structure.json`, `docs/QWEN3_MAPPING.md`, `scripts/inspect_model.py`, and `tests/unit/test_model_inspection.py`.
- **Weight safety:** The inspection fetched only the pinned configuration and small source files. No model object, full-model tensor allocation, `.safetensors`, `.bin`, or snapshot was downloaded.

## S00 reproducibility command record

The environment snapshot was regenerated with:

```bash
source ~/.venv/bin/activate
which python
python --version
python scripts/inspect_environment.py > /tmp/qaq-environment-audit.json
```

The exact dependency revision was verified with:

```bash
git -C third_party/any-precision-llm rev-parse HEAD
git -C third_party/any-precision-llm show -s --format='%H%n%aI%n%s' HEAD
git -C third_party/any-precision-llm status --porcelain=v1
git submodule status --recursive
```

A clean QAQ clone was then initialized recursively and checked for a clean superproject and dependency checkout. The command and result are recorded in `docs/stages/S00_SPEC.md` under `Current evidence audit`.
