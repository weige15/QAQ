# S05 — Query features and request state

## Goal and scope

S05 implements prompt-only query features and request-owned fixed routes using
the existing S04 manual policy. It remains batch-size one. There is no learned
router, probability output, soft routing, training loss, completion-token
feature update, or CPU-to-GPU loading.

## Feature sources and timing

For layer `i` during prefill, the attention feature is computed from the
hidden states entering that layer's attention unit: after `input_layernorm`
and before `q_proj`, `k_proj`, `v_proj`, or any other attention-unit output.
The FFN feature is computed from the hidden states entering the FFN unit:
after the real attention operation, residual addition, and
`post_attention_layernorm`, but before `gate_proj`, `up_proj`, or `down_proj`.
Thus neither routed unit's output can influence its own feature or route.

The feature is a detached tensor of shape `[hidden_size]` (2560 for the pinned
Qwen3-4B model). Pooling is explicitly
`sum(hidden_states * valid_mask) / valid_token_count`, accumulated in float32.
The mask must be a batch-size-one `[1, sequence_length]` 0/1 or boolean tensor;
padding contributes neither values nor denominator. All-padding, missing,
malformed, and mismatched masks fail clearly. Right-padding invariance is
covered by a documented `atol=1e-6`, `rtol=0` test.

## Request lifecycle

`QaqRequestState` owns `request_id`, `prompt_length` (valid prompt-token
count), per-layer attention/FFN route lists, and per-layer attention/FFN
feature lists. It validates layer counts, route values, feature dimensions,
and ownership. Feature tensors are detached clones. A concrete state binds to
one model object; `request_id` is not a global registry key, so duplicate IDs
are safe only when the state objects are independent.

Prefill requires a state, an explicit prompt mask, and `phase="prefill"`.
Each attention and FFN unit computes and stores its feature, invokes the
deterministic policy exactly once, and stores the returned 4 or 8 route before
executing the unit. The route trace records request ID, layer, unit, phase,
feature-computed flag, policy-invoked flag, and selected precision. Auxiliary
events record `incoming_hidden -> feature_computed -> route_available ->
unit_execute`.

Decode requires the completed same state and `phase="decode"`. It does not
pool hidden states, mutate stored features/routes, or invoke the policy. The
route trace marks each precision as reused. Completion hidden states therefore
cannot retroactively affect prompt-derived state.

The policy is either an S04 `PrecisionPlan` adapter or a deterministic callback
`(layer_index, unit_type, feature) -> 4|8`. The callback is intentionally
non-adaptive in the baseline; features are still computed and stored.

## Evidence and exact checks

Focused S05 checks:

```text
source ~/.venv/bin/activate
which python                         # /nfs/home/s314511048/.venv/bin/python
python --version                     # Python 3.12.3
pytest -q tests/unit/test_prompt_mask_pooling.py tests/unit/test_padding_invariance.py tests/unit/test_request_state.py tests/integration/test_no_completion_token_leakage.py tests/integration/test_attention_feature_timing.py tests/integration/test_ffn_feature_timing.py tests/integration/test_route_fixed_during_decode.py tests/integration/test_request_state_isolation.py
                                     # PASS: 20 passed
```

The S05 artifact-dependent parity test is
`tests/integration/test_s05_manual_routing.py`; it compares request-prefill
all-4/all-8 logits with the verified S03 static outputs at the S04 tolerance
`atol=1e-3`, `rtol=1e-3`. It is skipped when the ignored S03 artifact is not
available in the disposable worktree. With the supplied read-only packed
artifact, the exact combined command

```text
QAQ_S03_ARTIFACT='/nfs/home/s314511048/firstmate/projects/QAQ/quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64' QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_s04_manual_routing.py tests/integration/test_s05_manual_routing.py
  PASS: 10 passed in 422.84s (0:07:02)
```

This is 8 unchanged S04 tests plus 2 S05 all-4/all-8 request-prefill parity
tests. `tests/integration/test_s05_tiny_qwen3_execution.py`
exercises the real Transformers Qwen3 wrapper with a deterministic tiny
36-layer CPU configuration and passes prefill plus a different completion
decode step; it is a timing/lifecycle check, not a quantized parity claim.

The validation record for this worktree is:

```text
pytest -q tests/unit/test_prompt_mask_pooling.py tests/unit/test_padding_invariance.py tests/unit/test_request_state.py tests/integration/test_no_completion_token_leakage.py tests/integration/test_attention_feature_timing.py tests/integration/test_ffn_feature_timing.py tests/integration/test_route_fixed_during_decode.py tests/integration/test_request_state_isolation.py tests/integration/test_s05_tiny_qwen3_execution.py
  PASS: 23 passed
pytest -q tests/unit
  PASS: 58 passed
pytest -q tests/unit/test_backend_import.py tests/unit/test_single_linear_precision4.py tests/unit/test_single_linear_precision8.py tests/unit/test_cuda_vs_dequantized_reference.py tests/unit/test_deterministic_output.py tests/unit/test_pack_unpack_known_pattern.py tests/unit/test_backend_known_patterns.py tests/unit/test_pack_unpack_random.py tests/unit/test_plane_order.py tests/unit/test_prefix_precision.py tests/unit/test_padding.py tests/unit/test_packed_byte_count.py tests/unit/test_no_byte_per_bit_production_storage.py tests/unit/test_reference_backend_agreement.py tests/unit/test_nested_quantization_metadata.py tests/unit/test_serialization_order.py
  PASS: 27 passed in 8.94s
QAQ_S03_ARTIFACT='/nfs/home/s314511048/firstmate/projects/QAQ/quantized/s03b_qwen3_4b/backend_cache/packed/anyprec-(1cfa9a7208912126459214e8b04321603b3df60c)-w8_orig4-gc1-c4_s1_blk64' QAQ_MODEL_DEVICE=cuda:3 pytest -q tests/integration/test_static4_forward.py tests/integration/test_static8_forward.py
  PASS: 2 passed in 218.23s (0:03:38)
pytest -q tests/integration/test_s04_manual_routing.py tests/integration/test_s05_manual_routing.py tests/integration/test_static4_forward.py tests/integration/test_static8_forward.py tests/integration/test_static_generation.py tests/integration/test_checkpoint_roundtrip.py tests/integration/test_manifest_byte_count.py tests/integration/test_expected_modules_quantized.py tests/integration/test_no_duplicate_precision_models.py
  21 skipped; all artifact-dependent tests report the absent S03-B artifact
ruff check src/qaq tests/unit tests/integration/test_no_completion_token_leakage.py tests/integration/test_attention_feature_timing.py tests/integration/test_ffn_feature_timing.py tests/integration/test_route_fixed_during_decode.py tests/integration/test_request_state_isolation.py tests/integration/test_s05_manual_routing.py tests/integration/test_s05_tiny_qwen3_execution.py
  PASS: All checks passed
python -m compileall -q src tests
  PASS: exit 0
```

The S04 historical record remains **8 passed** for all-4/all-8 parity and
route isolation at commit `a5802358acd756751d4006705ebea961a27b0f8c`; the
same 8 tests passed again above against the supplied artifact.

## Limitations and gate

The wrapper is specialized to the verified Qwen3-4B 36-layer S04 graph. The
artifact-backed parity and static regressions require the supplied read-only
artifact path. No S06 functionality is implemented or executed.

**CONTINUE condition:** prompt-only timing, padding invariance, no completion
leakage, request isolation, route reuse without decode policy calls, shape and
ownership validation, deterministic manual policy behavior, S04 parity, and
S01-S04 regressions all pass.

**REVISE condition:** circular/output-dependent timing, padding dependence,
leakage, route recomputation, state leakage, or parity failure.

**PAUSE condition:** timing instrumentation or request ownership cannot be
trusted, including unavailable artifact-backed execution evidence.

**STOP condition:** batch-size expansion is required or request isolation
fails.
