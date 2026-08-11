"""S04 explicit manual attention/FFN routing over the S03 packed model.

The route is an input to each forward call.  No packed linear's mutable
``precision`` attribute is changed: every packed call receives its selected
bit-width explicitly and can be audited through a per-call trace.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import torch
from torch import nn

from .s03_static import assert_target_invariant, load_static_model

LAYER_COUNT = 36
SUPPORTED_BITS = (4, 8)
ATTENTION_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
FFN_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def _validate_route_field(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple[int, ...]; got {type(value).__name__}")
    if len(value) != LAYER_COUNT:
        raise ValueError(
            f"{name} must contain exactly {LAYER_COUNT} entries, one per layer; got {len(value)}"
        )
    for layer_index, bits in enumerate(value):
        if isinstance(bits, bool) or not isinstance(bits, int):
            raise TypeError(f"{name}[{layer_index}] must be an integer precision of 4 or 8")
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"{name}[{layer_index}] must be 4 or 8; got {bits}")
    return value


@dataclass(frozen=True, slots=True)
class PrecisionPlan:
    """Immutable per-layer attention and FFN precision selections."""

    attention_bits: tuple[int, ...]
    ffn_bits: tuple[int, ...]
    layer_count: ClassVar[int] = LAYER_COUNT

    def __post_init__(self) -> None:
        _validate_route_field("attention_bits", self.attention_bits)
        _validate_route_field("ffn_bits", self.ffn_bits)

    @classmethod
    def uniform(cls, bits: int) -> PrecisionPlan:
        """Build an explicit all-4 or all-8 plan."""

        return cls(attention_bits=(bits,) * LAYER_COUNT, ffn_bits=(bits,) * LAYER_COUNT)

    def validate(self) -> None:
        """Revalidate immediately before execution as a defensive boundary."""

        _validate_route_field("attention_bits", self.attention_bits)
        _validate_route_field("ffn_bits", self.ffn_bits)

    def to_dict(self) -> dict[str, list[int]]:
        """Return a JSON-compatible copy with stable field names."""

        return {
            "attention_bits": list(self.attention_bits),
            "ffn_bits": list(self.ffn_bits),
        }

    def to_json(self) -> str:
        """Serialize deterministically for experiment records and fixtures."""

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PrecisionPlan:
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"precision plan payload must be an object; got {type(payload).__name__}"
            )
        expected = {"attention_bits", "ffn_bits"}
        keys = set(payload)
        missing = sorted(expected - keys)
        extra = sorted(keys - expected, key=str)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing fields {missing}")
            if extra:
                details.append(f"extra fields {extra}")
            raise ValueError("invalid precision plan: " + ", ".join(details))

        fields: dict[str, tuple[int, ...]] = {}
        for name in ("attention_bits", "ffn_bits"):
            value = payload[name]
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"{name} must be a JSON array of integer precisions; got {type(value).__name__}"
                )
            fields[name] = tuple(value)
        return cls(**fields)

    @classmethod
    def from_json(cls, serialized: str) -> PrecisionPlan:
        try:
            payload = json.loads(serialized)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("precision plan JSON is invalid") from error
        if not isinstance(payload, Mapping):
            raise TypeError(
                f"precision plan JSON must contain an object; got {type(payload).__name__}"
            )
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class PrecisionCall:
    """One selected packed projection call captured by ``PrecisionTrace``."""

    layer_index: int
    unit_type: str
    module_path: str
    selected_bits: int

    def to_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "unit_type": self.unit_type,
            "module_path": self.module_path,
            "selected_bits": self.selected_bits,
        }


class PrecisionTrace:
    """Per-forward mutable collector; the model never stores it."""

    __slots__ = ("_records",)

    def __init__(self) -> None:
        self._records: list[PrecisionCall] = []

    @property
    def records(self) -> tuple[PrecisionCall, ...]:
        return tuple(self._records)

    def record(
        self, *, layer_index: int, unit_type: str, module_path: str, selected_bits: int
    ) -> None:
        self._records.append(
            PrecisionCall(
                layer_index=layer_index,
                unit_type=unit_type,
                module_path=module_path,
                selected_bits=selected_bits,
            )
        )

    def to_list(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self._records]


def expected_trace(plan: PrecisionPlan) -> tuple[PrecisionCall, ...]:
    """Return the exact projection call sequence required by the Qwen3 mapping."""

    plan.validate()
    records: list[PrecisionCall] = []
    for layer_index in range(LAYER_COUNT):
        records.extend(
            PrecisionCall(
                layer_index=layer_index,
                unit_type="attention",
                module_path=f"model.layers.{layer_index}.self_attn.{projection}",
                selected_bits=plan.attention_bits[layer_index],
            )
            for projection in ATTENTION_PROJECTIONS
        )
        records.extend(
            PrecisionCall(
                layer_index=layer_index,
                unit_type="ffn",
                module_path=f"model.layers.{layer_index}.mlp.{projection}",
                selected_bits=plan.ffn_bits[layer_index],
            )
            for projection in FFN_PROJECTIONS
        )
    return tuple(records)


class _RoutedPackedLinear(nn.Module):
    """Require an explicit precision argument for one verified packed target."""

    def __init__(self, packed: nn.Module, *, layer_index: int, unit_type: str, module_path: str):
        super().__init__()
        self.packed = packed
        self.layer_index = layer_index
        self.unit_type = unit_type
        self.module_path = module_path

    def forward(
        self, inputs: torch.Tensor, *, precision: int, trace: PrecisionTrace
    ) -> torch.Tensor:
        if isinstance(precision, bool) or not isinstance(precision, int):
            raise TypeError("selected packed precision must be an integer")
        if precision not in SUPPORTED_BITS:
            raise ValueError(f"selected packed precision must be 4 or 8; got {precision}")
        trace.record(
            layer_index=self.layer_index,
            unit_type=self.unit_type,
            module_path=self.module_path,
            selected_bits=precision,
        )
        return self.packed(inputs, precision=precision)


class _ManualAttention(nn.Module):
    def __init__(self, base: nn.Module, layer_index: int):
        super().__init__()
        self.config = base.config
        self.layer_idx = base.layer_idx
        self.head_dim = base.head_dim
        self.num_key_value_groups = base.num_key_value_groups
        self.scaling = base.scaling
        self.attention_dropout = base.attention_dropout
        self.is_causal = base.is_causal
        self.sliding_window = base.sliding_window
        self.q_proj = base.q_proj
        self.k_proj = base.k_proj
        self.v_proj = base.v_proj
        self.o_proj = base.o_proj
        self.q_norm = base.q_norm
        self.k_norm = base.k_norm
        self.layer_index = layer_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_value: Any = None,
        cache_position: torch.LongTensor | None = None,
        *,
        selected_bits: int,
        trace: PrecisionTrace,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(
            self.q_proj(hidden_states, precision=selected_bits, trace=trace).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states, precision=selected_bits, trace=trace).view(hidden_shape)
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states, precision=selected_bits, trace=trace)
            .view(hidden_shape)
            .transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation == "sdpa" and kwargs.get(
                "output_attentions", False
            ):
                attention_interface = eager_attention_forward
            else:
                attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output, precision=selected_bits, trace=trace)
        return attn_output, attn_weights


class _ManualMLP(nn.Module):
    def __init__(self, base: nn.Module, layer_index: int):
        super().__init__()
        self.config = base.config
        self.hidden_size = base.hidden_size
        self.intermediate_size = base.intermediate_size
        self.gate_proj = base.gate_proj
        self.up_proj = base.up_proj
        self.down_proj = base.down_proj
        self.act_fn = base.act_fn
        self.layer_index = layer_index

    def forward(
        self, inputs: torch.Tensor, *, selected_bits: int, trace: PrecisionTrace
    ) -> torch.Tensor:
        gate = self.gate_proj(inputs, precision=selected_bits, trace=trace)
        up = self.up_proj(inputs, precision=selected_bits, trace=trace)
        return self.down_proj(self.act_fn(gate) * up, precision=selected_bits, trace=trace)


class _ManualDecoderLayer(nn.Module):
    def __init__(self, base: nn.Module, layer_index: int):
        super().__init__()
        self.hidden_size = base.hidden_size
        self.layer_index = layer_index
        self.self_attn = _ManualAttention(base.self_attn, layer_index)
        self.mlp = _ManualMLP(base.mlp, layer_index)
        self.input_layernorm = base.input_layernorm
        self.post_attention_layernorm = base.post_attention_layernorm

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        *,
        precision_plan: PrecisionPlan,
        trace: PrecisionTrace,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        attention_bits = precision_plan.attention_bits[self.layer_index]
        ffn_bits = precision_plan.ffn_bits[self.layer_index]
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            selected_bits=attention_bits,
            trace=trace,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, selected_bits=ffn_bits, trace=trace)
        hidden_states = residual + hidden_states

        outputs: tuple[torch.Tensor, ...] = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class _ManualBaseModel(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.padding_idx = base.padding_idx
        self.vocab_size = base.vocab_size
        self.config = base.config
        self.embed_tokens = base.embed_tokens
        self.layers = nn.ModuleList(
            [_ManualDecoderLayer(layer, index) for index, layer in enumerate(base.layers)]
        )
        self.norm = base.norm
        self.rotary_emb = base.rotary_emb
        # The helper is deliberately not registered: its mask implementation is
        # read-only and the routed module tree must contain one execution graph.
        object.__setattr__(self, "_mask_helper", base)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        *,
        precision_plan: PrecisionPlan,
        trace: PrecisionTrace,
        **flash_attn_kwargs: Any,
    ) -> Any:
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._mask_helper._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                precision_plan=precision_plan,
                trace=trace,
                **flash_attn_kwargs,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ManualRoutedQwen3ForCausalLM(nn.Module):
    """Qwen3 Causal-LM wrapper requiring one explicit ``PrecisionPlan`` per call."""

    def __init__(self, static_model: nn.Module):
        super().__init__()
        self.config = static_model.config
        self.vocab_size = static_model.vocab_size
        self.model = _ManualBaseModel(static_model.model)
        self.lm_head = static_model.lm_head

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Any = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        *,
        precision_plan: PrecisionPlan,
        trace: PrecisionTrace | None = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(precision_plan, PrecisionPlan):
            raise TypeError("precision_plan must be a PrecisionPlan")
        precision_plan.validate()
        active_trace = trace if trace is not None else PrecisionTrace()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            precision_plan=precision_plan,
            trace=active_trace,
            **kwargs,
        )
        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = None
        if labels is not None:
            loss = self._loss(logits, labels)
        from transformers.modeling_outputs import CausalLMOutputWithPast

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @staticmethod
    def _loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != labels.shape[1]:
            raise ValueError(
                "labels require logits for the complete sequence in S04 manual execution"
            )
        return nn.functional.cross_entropy(
            logits[..., :-1, :].contiguous().view(-1, logits.shape[-1]),
            labels[..., 1:].contiguous().view(-1),
            ignore_index=-100,
        )


def _replace_module(root: nn.Module, name: str, replacement: nn.Module) -> None:
    parts = name.split(".")
    parent: Any = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    setattr(parent, parts[-1], replacement)


def _wrap_verified_targets(static_model: nn.Module) -> None:
    target_names = assert_target_invariant()
    modules = dict(static_model.named_modules())
    for module_path in target_names:
        packed = modules.get(module_path)
        if packed is None or packed.__class__.__name__ != "AnyPrecisionLinear":
            raise ValueError(
                f"verified packed target is missing or has the wrong type: {module_path}"
            )
        layer_index = int(module_path.split(".")[2])
        if ".self_attn." in module_path:
            unit_type = "attention"
        elif ".mlp." in module_path:
            unit_type = "ffn"
        else:
            raise ValueError(f"target is outside the S04 route scopes: {module_path}")
        _replace_module(
            static_model,
            module_path,
            _RoutedPackedLinear(
                packed,
                layer_index=layer_index,
                unit_type=unit_type,
                module_path=module_path,
            ),
        )


def load_manual_model(
    artifact: str | os.PathLike[str], device: str
) -> ManualRoutedQwen3ForCausalLM:
    """Load the verified S03 nested checkpoint and expose explicit S04 routing."""

    static_model = load_static_model(artifact, device)
    _wrap_verified_targets(static_model)
    manual_model = ManualRoutedQwen3ForCausalLM(static_model)
    manual_model.eval()
    return manual_model
