# S10-A — Enable static 6-bit execution

## Gate result

**CONTINUE.** S10-A enables public static precision `6` against the existing
nested Qwen3 4→8 artifact. The implementation change is limited to
`qaq.model.static`: its public static precision set is now exactly `(4, 6, 8)`.
The pinned Any-Precision source, artifact, router source, request loading,
hard routing, router checkpoints, and S09 evidence were not changed.

The artifact-backed results below are the recorded implementation-gate
evidence. The exact identity-matched S03-B artifact was authorized read-only
from the original QAQ worktree through `QAQ_S03_ARTIFACT`, and its
`pytorch_model.bin` SHA-256 was verified. The focused integration and
preservation reruns passed; no artifact was regenerated or substituted.

## Sources, identity, and artifact inventory

The authoritative S09 closeout establishes S01–S08 complete and S09 frozen and
closed. The artifact used for this gate is:

```text
quantized/s03b_qwen3_4b/backend_cache/packed/
  anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64
```

Its `pytorch_model.bin` SHA-256 is
`29d9bc526b3da0bd39daf2f82afd141f82d005ca1232cabc75cfe9d9ecc1cfee`,
matching `docs/quantized_model_manifest.json`. The pinned Any-Precision
submodule is `a3257d02740cc5757c78673da534b0630ff3a4ea` and was clean.

The recorded artifact inventory was loaded without regenerating or
requantizing it. All 252 expected Qwen3 targets had exactly one parent
`<target>.qweight` plus `lut4`, `lut6`, and `lut8`; the state dictionary also
retains the natural pinned-backend `lut5` and `lut7` buffers. There were no
missing or extra quantized keys.

| inventory | result |
| --- | ---: |
| target / qweight count | 252 |
| qweight shape and dtype | `[8, out_features, in_features // 32]`, `torch.int32` |
| qweight parent payload | 3,633,315,840 bytes |
| selected six-plane payload | 2,724,986,880 bytes (`qweight[:6]`) |
| LUT6 count | 252 |
| LUT6 shapes | 72 × `[2560,64]`; 72 × `[9728,64]`; 72 × `[1024,64]`; 36 × `[4096,64]` |
| LUT6 dtype / values | `torch.float16`; all finite |
| LUT6 total | 141,557,760 bytes |

Representative tensor evidence (SHA-256 is over the contiguous tensor bytes):

| target | qweight shape / bytes / digest | LUT6 shape / bytes / digest |
| --- | --- | --- |
| `model.layers.0.mlp.down_proj` | `[8,2560,304]` / 24,903,680 / `0c9b602586778d37a29cbb9ae90b9d6506de24d31e6161ae9e4cc1884c1b2c7d` | `[2560,64]` / 327,680 / `4af0512d26f8339b7be02ca0cb3bfa1f51263b784876d15415351829fa0b6610` |
| `model.layers.0.mlp.gate_proj` | `[8,9728,80]` / 24,903,680 / `a694efd7223cace865825e893b1ce8607877bd5b6e7bd10a93ac73929cd212d2` | `[9728,64]` / 1,245,184 / `d9c5a8e2fdbeafeada1b23549d254f47d1f4b84d47c0c12a7353f5e7df88d920` |
| `model.layers.0.self_attn.o_proj` | `[8,2560,128]` / 10,485,760 / `c45f3b7214f051e976dc8bb75b2a257688d341e94a14353b299b98306c7ab163` | `[2560,64]` / 327,680 / `6814e0fac3a9b145cce9db57d6dd4caa5898aeef1ec9593509a2c16176deb412` |
| `model.layers.0.self_attn.q_proj` | `[8,4096,80]` / 10,485,760 / `e7b3060bc87b050e7eb54fef50930a0ca7972c1c34d844411d6249861d965bf3` | `[4096,64]` / 524,288 / `12052f80149d0fe9bb75e41afba7ff02ec318c346c960bb6ebbaa0600de80c5e` |

## Six-bit semantics and backend evidence

Precision 6 uses the existing nested parent and matching lookup table exactly:

```text
4 bits: qweight[:4] + lut4
6 bits: qweight[:6] + lut6
8 bits: qweight[:8] + lut8
```

The parent is physically packed; no byte-per-bit representation, separate
six-bit qweight, persistent dequantized weight, or duplicate model is added.
The pinned backend continues to load the inclusive 4–8 buffers, including its
internal `lut5` and `lut7` buffers, while QAQ's public static gate exposes only
4, 6, and 8.

The real representative `model.layers.0.mlp.down_proj` was executed on
`cuda:0` (NVIDIA GeForce RTX 3090) with deterministic CPU-seeded FP16 input
`[4,9728]`, using the pinned backend at precision 6. The output was finite
FP16 `[4,2560]` on `cuda:0`, and the independent pinned `dequant_kbit` plus
FP16 `torch.matmul` reference was finite and shape-equal. With the established
S01 tolerance `atol=0.05`, `rtol=0.01`, `max_abs_error=0.015625`,
`mean_abs_error=0.0017089294`, and `allclose=True`. Repeated backend output
was bitwise equal with digest
`e25d9c8d531d22269c18b1ef27ebe17dc6ceb34223326a6f25824f154e02778f`.
The module had no parameters and no registered dense weight; the temporary
reference reconstruction was not part of persistent module state.

## Full-model static smoke and audit

The deterministic full Qwen3 static loader was run at precision 6 on the same
RTX 3090. It loaded the recorded artifact normally with no state-dict errors
and returned finite FP16 logits of shape `[1,8,151936]`. The first and repeated
logits digest was
`4e0856454ebab64588183a1e72acc2fc34ffea68d82c590526624edd804e3390`.

The loaded graph contained 252 `AnyPrecisionLinear` modules, 252 qweight
buffers, 252 LUT6 buffers, and zero `qweight6` buffers. Quantized modules had
zero dense parameters. The existing loader graph reported 389,152,256 total
parameters, all currently marked trainable by that loader; S07's established
router-training freeze machinery remains unchanged and was not re-run here.
The precision-6 smoke was repeated bitwise deterministically. Existing static
4-bit and 8-bit forward, checkpoint, target-inventory, duplicate-model, and
physical-byte regressions also passed.

## Recorded validation commands

Every command was preceded by the mandated `~/.venv` activation, interpreter
check, and successful `nvidia-smi` check. `PYTHONPATH=src:.` was used where
needed to avoid an ambient unrelated `scripts` package on `PYTHONPATH`.

```text
PYTHONPATH=src:. pytest -q tests/unit
110 passed

PYTHONPATH=src:third_party/any-precision-llm:. \
  QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q tests/integration/test_s10a_static6.py
3 passed in 221.42s

PYTHONPATH=src:. QAQ_S03_ARTIFACT=<identity-matched S03-B artifact> \
  QAQ_MODEL_DEVICE=cuda:0 pytest -q \
  tests/integration/test_static4_forward.py \
  tests/integration/test_static8_forward.py \
  tests/integration/test_expected_modules_quantized.py \
  tests/integration/test_no_duplicate_precision_models.py \
  tests/integration/test_manifest_byte_count.py \
  tests/integration/test_checkpoint_roundtrip.py
10 passed in 640.26s

PYTHONPATH=src:. pytest -q \
  tests/integration/test_s06_soft_routing.py \
  tests/integration/test_s07_distillation_smoke.py
3 passed in 9.37s

ruff check src/qaq/model/static.py \
  tests/unit/test_static_precision_validation.py \
  tests/integration/test_s10a_static6.py
All checks passed
```

## Limitations and boundary

S10-A establishes static 6-bit execution only. It does not add 6-bit routing,
change `CANDIDATE_BITS`, change hard-route choices, alter request loading or
state ownership, retrain or reload router checkpoints, or assess quality,
latency, transfer, or memory claims. Router semantics remain exactly 4/8
after S10-A. A separate later stage and decision is required for any 6-bit
routing behavior.
