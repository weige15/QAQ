# Qwen3-4B module mapping specification

This is the target-model architecture specification from the initial specification objective (legacy work item S00), not an adapter implementation.

## Locked evidence

- Model repository: `Qwen/Qwen3-4B`.
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`.
- Configuration URL: `https://huggingface.co/Qwen/Qwen3-4B/resolve/1cfa9a7208912126459214e8b04321603b3df60c/config.json`.
- Configuration-declared Transformers version: `4.51.0`.
- Transformers implementation source commit: `0720e206c6ba28887e4d60ef60a6a089f6c1cc76`, the dereferenced `v4.51.0` tag.
- Modeling source: `src/transformers/models/qwen3/modeling_qwen3.py` at the implementation source commit.
- Configuration source: `src/transformers/models/qwen3/configuration_qwen3.py` at the implementation source commit.
- Any-Precision revision: `a3257d02740cc5757c78673da534b0630ff3a4ea`.

The active project environment contains Transformers `4.39.3`, which has no `transformers.models.qwen3` package.
Therefore the class hierarchy below is established from the pinned model configuration and the official Transformers `4.51.0` source, while execution under the current environment remains unverified.

## Class hierarchy and paths

- Causal-LM wrapper: `Qwen3ForCausalLM`.
- Base model: `Qwen3ForCausalLM.model` → `Qwen3Model`.
- Decoder layers: `Qwen3ForCausalLM.model.layers` → `ModuleList[Qwen3DecoderLayer]`.
- Attention unit in layer `i`: `model.layers.i.self_attn` → `Qwen3Attention`.
- FFN unit in layer `i`: `model.layers.i.mlp` → `Qwen3MLP`.
- Normalization class: `Qwen3RMSNorm`.
- Token embedding: `model.embed_tokens`.
- Final normalization: `model.norm`.
- Rotary component: `model.rotary_emb` → `Qwen3RotaryEmbedding`.
- Output head: `lm_head` on `Qwen3ForCausalLM`.

## Dimensions

The exact pinned `config.json` establishes:

- Decoder layers: `36`.
- Hidden size: `2560`.
- Intermediate FFN size: `9728`.
- Vocabulary size: `151936`.
- Attention heads: `32`.
- Key/value heads: `8`.
- Attention head dimension: `128`.
- Maximum configured sequence length: `40960`.
- Configured dtype: `bfloat16`.
- Activation: `silu`.
- Attention bias: `false`.
- `use_cache`: `true`.

## Attention mapping

The exact representative-layer paths are:

| Relative path | Full path pattern | Input | Output | Bias | Target |
| --- | --- | ---: | ---: | --- | --- |
| `self_attn.q_proj` | `model.layers.<i>.self_attn.q_proj` | 2560 | 4096 | false | yes |
| `self_attn.k_proj` | `model.layers.<i>.self_attn.k_proj` | 2560 | 1024 | false | yes |
| `self_attn.v_proj` | `model.layers.<i>.self_attn.v_proj` | 2560 | 1024 | false | yes |
| `self_attn.o_proj` | `model.layers.<i>.self_attn.o_proj` | 4096 | 2560 | false | yes |

The four projections are standard `nn.Linear` modules in `Qwen3Attention`.
Q and K are reshaped to head dimension, passed through `q_norm` and `k_norm`, transposed, and then transformed by rotary position embeddings.
V is reshaped and transposed without Q/K normalization.
The attention output is reshaped before `o_proj`.

The following remain outside the packed linear replacement:

- `model.layers.<i>.self_attn.q_norm`, `Qwen3RMSNorm(128)`.
- `model.layers.<i>.self_attn.k_norm`, `Qwen3RMSNorm(128)`.
- `model.rotary_emb`, `Qwen3RotaryEmbedding`.
- `model.layers.<i>.input_layernorm`, `Qwen3RMSNorm(2560)`.
- `model.layers.<i>.post_attention_layernorm`, `Qwen3RMSNorm(2560)`.

## FFN mapping

The exact representative-layer paths are:

| Relative path | Full path pattern | Input | Output | Bias | Target |
| --- | --- | ---: | ---: | --- | --- |
| `mlp.gate_proj` | `model.layers.<i>.mlp.gate_proj` | 2560 | 9728 | false | yes |
| `mlp.up_proj` | `model.layers.<i>.mlp.up_proj` | 2560 | 9728 | false | yes |
| `mlp.down_proj` | `model.layers.<i>.mlp.down_proj` | 9728 | 2560 | false | yes |

The exact activation relationship is:

```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

All three projections are standard `nn.Linear` modules in `Qwen3MLP`.

## Complete target list and counts

The later adapter must generate these seven relative names for every `i` from `0` through `35`:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
```

The complete full-name pattern is:

```text
model.layers.<i>.self_attn.q_proj
model.layers.<i>.self_attn.k_proj
model.layers.<i>.self_attn.v_proj
model.layers.<i>.self_attn.o_proj
model.layers.<i>.mlp.gate_proj
model.layers.<i>.mlp.up_proj
model.layers.<i>.mlp.down_proj
```

Counts are derived without reading weight tensors:

- Attention: `36 * 4 = 144`.
- FFN: `36 * 3 = 108`.
- Total: `36 * 7 = 252`.
- The generated list contains 252 unique names.

The input dimensions of all seven targets are divisible by 32, as required by the pinned `AnyPrecisionLinear` packed-weight shape.

## Non-target components

The following are deliberately excluded from the packed target list and retained in the configured BF16/FP16 policy unless later execution evidence changes the policy:

- `model.embed_tokens`: input embedding, shape `[151936, 2560]`.
- `model.norm`: final RMS normalization, shape `[2560]`.
- Each layer's `input_layernorm` and `post_attention_layernorm`: RMS normalization, shape `[2560]`.
- Each layer's `self_attn.q_norm` and `self_attn.k_norm`: RMS normalization, shape `[128]`.
- `model.rotary_emb`: rotary-position component, not a packed linear weight.
- `lm_head`: output projection, shape `[2560, 151936]`, bias false, retained outside the packed target list.
- `past_key_values`: runtime cache. The configuration enables caching and the implementation updates a `DynamicCache`; cache entries are not quantized model weights.

## Tied weights

The exact configuration sets `tie_word_embeddings: true`.
The official `Qwen3ForCausalLM` source declares `_tied_weights_keys = ["lm_head.weight"]` and exposes `model.embed_tokens` through `get_input_embeddings()` and `lm_head` through `get_output_embeddings()`.
Transformers' inherited `tie_weights()` mechanism therefore represents `lm_head.weight` as tied to `model.embed_tokens.weight`.
The embedding and output head must remain excluded together unless a later implementation explicitly preserves this relationship.

## QAQ routing decisions

The observed paths make separate attention and FFN routing units structurally unambiguous.
All four attention projections can share one attention-unit route even though Q/K/V and O have different matrix shapes, because each replacement retains its own dimensions.
All three FFN projections can share one FFN-unit route for the same reason.
The structure does not prove numerical correctness, memory behavior, or backend execution; those require later tests.

## Any-Precision compatibility

The pinned Any-Precision source explicitly provides architecture YAMLs for `LlamaForCausalLM`, `MistralForCausalLM`, `OPTForCausalLM`, and `PhiForCausalLM`.
It does not provide a Qwen3 YAML, so Qwen3 is **not explicitly supported** by this revision.

When no matching YAML exists, `get_analyzer()` falls back to `AutoArchConfig`, which scans the first layer for `torch.nn.Linear` modules and warns that automatic detection may be incorrect.
The generic analyzer can therefore discover the seven Qwen3 projections in principle, but that fallback is not an official support claim and is not a sufficient reproducibility contract for QAQ.

The Qwen3 target is structurally compatible with the pinned `AnyPrecisionLinear` interface:

- Every target is a standard linear module.
- All target input dimensions are divisible by 32.
- All target biases are absent and the backend supports `bias=False`.
- Q/K normalization, rotary processing, gating, normalization, embeddings, output head, and KV cache remain outside the packed linear replacement.

Later work must add an explicit Qwen3 mapping without modifying the pinned upstream source, then validate it under a Transformers version that contains Qwen3.
That work must verify module replacement, tied-weight preservation, packed dimensions, 4-bit and 8-bit execution, and numerical behavior.

## Reproducibility tool

`scripts/inspect_model.py` reads the repository and immutable revision from `configs/model.yaml`, fetches only the pinned `config.json` and small official Transformers source files, verifies the class markers, generates the complete target list, and writes `docs/model_structure.json`.
It never calls `from_pretrained`, instantiates a full model, allocates full-model tensors, or downloads weight shards.
