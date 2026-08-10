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
| `papers/QAQ.pdf`, pp. 1–4; local workshop artifact | The local artifact describes query-conditioned routing, bit-plane storage, block-level MHA/FFN organization, teacher-student router training, and CPU-to-GPU on-demand loading. | This local PDF has no independently verified public record in this repository; its reported target models, metrics, and system details are not treated as validated evidence for S00. |
| `papers/PMPD.pdf`, pp. 2–3; local paper artifact | PMPD describes phase-aware and progressively lowering precision during decoding, motivated by differing error resilience across prefill and decoding. | PMPD does not authorize adding phase schedulers or progressive precision to the QAQ baseline before the documented freeze boundary. |

The source review therefore supports the choice to investigate Any-Precision as a storage/backend substrate and to keep paper claims separate from QAQ implementation choices. D003–D012 remain implementation choices unless later evidence explicitly changes them.

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
