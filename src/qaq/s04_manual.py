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

from .model.request_state import QaqRequestState
from .router.features import (
    coerce_manual_policy,
    masked_mean_pool,
    validate_policy_result,
    validate_prompt_mask,
)
from .router.soft_linear import mix_packed_outputs
from .s03_static import assert_target_invariant, load_static_model
from .s08_loader import PackedLinearSource, SynchronousPackedRequest

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


@dataclass(frozen=True, slots=True)
class SoftPrecisionCall:
    """One packed projection call carrying the shared probability tensor."""

    layer_index: int
    unit_type: str
    module_path: str
    probabilities: torch.Tensor


@dataclass(frozen=True, slots=True)
class RouteTraceRecord:
    """One request-level route decision, captured before its unit executes."""

    request_id: str
    layer_index: int
    unit_type: str
    phase: str
    feature_computed: bool
    policy_invoked: bool
    selected_precision: int | None
    reused_precision: int | None

    @property
    def precision(self) -> int:
        value = (
            self.selected_precision
            if self.selected_precision is not None
            else self.reused_precision
        )
        if value is None:  # pragma: no cover - construction is validated by record_route
            raise RuntimeError("route trace record has no selected or reused precision")
        return value

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "layer_index": self.layer_index,
            "unit_type": self.unit_type,
            "phase": self.phase,
            "feature_computed": self.feature_computed,
            "policy_invoked": self.policy_invoked,
            "selected_precision": self.selected_precision,
            "reused_precision": self.reused_precision,
        }


@dataclass(frozen=True, slots=True)
class RouteTraceEvent:
    """Timing evidence for the route-before-unit execution ordering."""

    request_id: str
    layer_index: int
    unit_type: str
    phase: str
    event: str
    precision: int | None = None


class PrecisionTrace:
    """Per-forward mutable collector; the model never stores it."""

    __slots__ = ("_events", "_records", "_route_records", "_soft_records")

    def __init__(self) -> None:
        self._records: list[PrecisionCall] = []
        self._soft_records: list[SoftPrecisionCall] = []
        self._route_records: list[RouteTraceRecord] = []
        self._events: list[RouteTraceEvent] = []

    @property
    def records(self) -> tuple[PrecisionCall, ...]:
        return tuple(self._records)

    @property
    def route_records(self) -> tuple[RouteTraceRecord, ...]:
        return tuple(self._route_records)

    @property
    def soft_records(self) -> tuple[SoftPrecisionCall, ...]:
        return tuple(self._soft_records)

    @property
    def events(self) -> tuple[RouteTraceEvent, ...]:
        return tuple(self._events)

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

    def record_soft(
        self,
        *,
        layer_index: int,
        unit_type: str,
        module_path: str,
        probabilities: torch.Tensor,
    ) -> None:
        self._soft_records.append(
            SoftPrecisionCall(
                layer_index=layer_index,
                unit_type=unit_type,
                module_path=module_path,
                probabilities=probabilities,
            )
        )

    def record_route(
        self,
        *,
        request_id: str,
        layer_index: int,
        unit_type: str,
        phase: str,
        feature_computed: bool,
        policy_invoked: bool,
        selected_precision: int | None = None,
        reused_precision: int | None = None,
    ) -> None:
        if (selected_precision is None) == (reused_precision is None):
            raise ValueError("route trace requires exactly one selected or reused precision")
        self._route_records.append(
            RouteTraceRecord(
                request_id=request_id,
                layer_index=layer_index,
                unit_type=unit_type,
                phase=phase,
                feature_computed=feature_computed,
                policy_invoked=policy_invoked,
                selected_precision=selected_precision,
                reused_precision=reused_precision,
            )
        )

    def record_event(
        self,
        *,
        request_id: str,
        layer_index: int,
        unit_type: str,
        phase: str,
        event: str,
        precision: int | None = None,
    ) -> None:
        self._events.append(
            RouteTraceEvent(
                request_id=request_id,
                layer_index=layer_index,
                unit_type=unit_type,
                phase=phase,
                event=event,
                precision=precision,
            )
        )

    def to_list(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self._records]

    def route_to_list(self) -> list[dict[str, object]]:
        return [record.to_dict() for record in self._route_records]


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


def _select_soft_request_route(
    *,
    request_state: QaqRequestState,
    layer_index: int,
    unit_type: str,
    incoming_hidden: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    soft_router: Any,
    trace: PrecisionTrace,
) -> torch.Tensor:
    trace.record_event(
        request_id=request_state.request_id,
        layer_index=layer_index,
        unit_type=unit_type,
        phase="prefill",
        event="incoming_hidden",
    )
    feature = masked_mean_pool(incoming_hidden, prompt_attention_mask)
    request_state.store_feature(unit_type, layer_index, feature)
    trace.record_event(
        request_id=request_state.request_id,
        layer_index=layer_index,
        unit_type=unit_type,
        phase="prefill",
        event="feature_computed",
    )
    probabilities = soft_router(layer_index, unit_type, feature.detach())
    if not isinstance(probabilities, torch.Tensor) or probabilities.shape != (2,):
        raise ValueError("soft router must return exactly two probabilities with shape [2]")
    if not torch.isfinite(probabilities).all() or not torch.all(probabilities >= 0):
        raise ValueError("soft router probabilities must be finite and non-negative")
    if not torch.allclose(probabilities.sum(), probabilities.new_tensor(1), atol=1e-5, rtol=0):
        raise ValueError("soft router probabilities must sum to one")
    request_state.store_probability(unit_type, layer_index, probabilities)
    trace.record_event(
        request_id=request_state.request_id,
        layer_index=layer_index,
        unit_type=unit_type,
        phase="prefill",
        event="route_available",
    )
    return probabilities


def _select_request_route(
    *,
    request_state: QaqRequestState,
    layer_index: int,
    unit_type: str,
    incoming_hidden: torch.Tensor,
    prompt_attention_mask: torch.Tensor | None,
    phase: str,
    routing_policy: Any,
    trace: PrecisionTrace,
) -> int:
    """Compute a prompt feature and select a route, or reuse a stored route."""

    request_id = request_state.request_id
    if phase == "prefill":
        if prompt_attention_mask is None:
            raise ValueError("S05 prefill requires an explicit prompt attention mask")
        trace.record_event(
            request_id=request_id,
            layer_index=layer_index,
            unit_type=unit_type,
            phase=phase,
            event="incoming_hidden",
        )
        feature = masked_mean_pool(incoming_hidden, prompt_attention_mask)
        request_state.store_feature(unit_type, layer_index, feature)
        trace.record_event(
            request_id=request_id,
            layer_index=layer_index,
            unit_type=unit_type,
            phase=phase,
            event="feature_computed",
        )
        if routing_policy is None:
            raise ValueError("S05 prefill requires a deterministic routing_policy")
        policy = coerce_manual_policy(routing_policy)
        precision = validate_policy_result(policy(layer_index, unit_type, feature))
        request_state.store_route(unit_type, layer_index, precision)
        trace.record_route(
            request_id=request_id,
            layer_index=layer_index,
            unit_type=unit_type,
            phase=phase,
            feature_computed=True,
            policy_invoked=True,
            selected_precision=precision,
        )
        trace.record_event(
            request_id=request_id,
            layer_index=layer_index,
            unit_type=unit_type,
            phase=phase,
            event="route_available",
            precision=precision,
        )
        return precision

    if phase == "decode":
        precision = request_state.route_for_decode(unit_type, layer_index)
        trace.record_route(
            request_id=request_id,
            layer_index=layer_index,
            unit_type=unit_type,
            phase=phase,
            feature_computed=False,
            policy_invoked=False,
            reused_precision=precision,
        )
        trace.record_event(
            request_id=request_id,
            layer_index=layer_index,
            unit_type=unit_type,
            phase=phase,
            event="route_available",
            precision=precision,
        )
        return precision

    raise ValueError(f"S05 phase must be 'prefill' or 'decode'; got {phase!r}")


class _OnDemandRoutedPackedLinear(nn.Module):
    """Execute one CPU-authoritative packed source through a request context."""

    def __init__(
        self,
        source: PackedLinearSource,
        *,
        layer_index: int,
        unit_type: str,
        module_path: str,
    ) -> None:
        super().__init__()
        self.source = source
        self.layer_index = layer_index
        self.unit_type = unit_type
        self.module_path = module_path

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        trace: PrecisionTrace,
        precision: int | None = None,
        request_state: QaqRequestState | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
    ) -> torch.Tensor:
        if precision is None:
            raise ValueError("on-demand packed execution requires a hard precision")
        if request_state is None or on_demand_context is None:
            raise ValueError("on-demand packed execution requires its request context")
        if on_demand_context.request_state is not request_state:
            raise RuntimeError("on-demand request context belongs to a different request state")
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
        return on_demand_context.execute(self.module_path, inputs, precision=precision)


class _RoutedPackedLinear(nn.Module):
    """Require an explicit hard precision or one S06 soft probability pair."""

    def __init__(self, packed: nn.Module, *, layer_index: int, unit_type: str, module_path: str):
        super().__init__()
        self.packed = packed
        self.layer_index = layer_index
        self.unit_type = unit_type
        self.module_path = module_path
        for parameter in self.packed.parameters():
            parameter.requires_grad_(False)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        trace: PrecisionTrace,
        precision: int | None = None,
        probabilities: torch.Tensor | None = None,
        request_state: QaqRequestState | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
    ) -> torch.Tensor:
        if (precision is None) == (probabilities is None):
            raise ValueError("packed execution requires exactly one of precision or probabilities")
        if probabilities is not None:
            if probabilities.shape != (2,):
                raise ValueError("soft packed probabilities must have shape [2]")
            if not torch.isfinite(probabilities).all() or not torch.all(probabilities >= 0):
                raise ValueError("soft packed probabilities must be finite and non-negative")
            if not torch.allclose(
                probabilities.sum(), probabilities.new_tensor(1), atol=1e-5, rtol=0
            ):
                raise ValueError("soft packed probabilities must sum to one")
            trace.record_soft(
                layer_index=self.layer_index,
                unit_type=self.unit_type,
                module_path=self.module_path,
                probabilities=probabilities,
            )
            return mix_packed_outputs(self.packed, inputs, probabilities)
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
        selected_bits: int | None = None,
        routing_probabilities: torch.Tensor | None = None,
        trace: PrecisionTrace,
        request_state: QaqRequestState | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        from transformers.models.qwen3.modeling_qwen3 import (
            ALL_ATTENTION_FUNCTIONS,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        packed_kwargs = (
            {"probabilities": routing_probabilities}
            if routing_probabilities is not None
            else {"precision": selected_bits}
        )
        packed_kwargs.update(
            request_state=request_state,
            on_demand_context=on_demand_context,
        )
        query_states = self.q_norm(
            self.q_proj(hidden_states, trace=trace, **packed_kwargs).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states, trace=trace, **packed_kwargs).view(hidden_shape)
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states, trace=trace, **packed_kwargs)
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
        attn_output = self.o_proj(attn_output, trace=trace, **packed_kwargs)
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
        self,
        inputs: torch.Tensor,
        *,
        selected_bits: int | None = None,
        routing_probabilities: torch.Tensor | None = None,
        trace: PrecisionTrace,
        request_state: QaqRequestState | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
    ) -> torch.Tensor:
        packed_kwargs = (
            {"probabilities": routing_probabilities}
            if routing_probabilities is not None
            else {"precision": selected_bits}
        )
        packed_kwargs.update(
            request_state=request_state,
            on_demand_context=on_demand_context,
        )
        gate = self.gate_proj(inputs, trace=trace, **packed_kwargs)
        up = self.up_proj(inputs, trace=trace, **packed_kwargs)
        return self.down_proj(self.act_fn(gate) * up, trace=trace, **packed_kwargs)


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
        precision_plan: PrecisionPlan | None,
        trace: PrecisionTrace,
        phase: str | None = None,
        request_state: QaqRequestState | None = None,
        routing_policy: Any = None,
        soft_router: Any = None,
        prompt_attention_mask: torch.Tensor | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, ...]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_probabilities = None
        if request_state is not None:
            if phase is None:
                raise ValueError("S05 request execution requires an explicit phase")
            if soft_router is not None:
                if phase != "prefill" or prompt_attention_mask is None:
                    raise ValueError("S06 soft routing supports prefill only")
                attention_probabilities = _select_soft_request_route(
                    request_state=request_state,
                    layer_index=self.layer_index,
                    unit_type="attention",
                    incoming_hidden=hidden_states,
                    prompt_attention_mask=prompt_attention_mask,
                    soft_router=soft_router,
                    trace=trace,
                )
                attention_bits = None
            else:
                attention_bits = _select_request_route(
                    request_state=request_state,
                    layer_index=self.layer_index,
                    unit_type="attention",
                    incoming_hidden=hidden_states,
                    prompt_attention_mask=prompt_attention_mask,
                    phase=phase,
                    routing_policy=routing_policy,
                    trace=trace,
                )
        else:
            if precision_plan is None:
                raise ValueError("S04 execution requires precision_plan")
            attention_bits = precision_plan.attention_bits[self.layer_index]
        if request_state is not None:
            trace.record_event(
                request_id=request_state.request_id,
                layer_index=self.layer_index,
                unit_type="attention",
                phase=phase,
                event="unit_execute",
                precision=attention_bits,
            )
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
            routing_probabilities=attention_probabilities,
            trace=trace,
            request_state=request_state,
            on_demand_context=on_demand_context,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        ffn_probabilities = None
        if request_state is not None:
            if soft_router is not None:
                if phase != "prefill" or prompt_attention_mask is None:
                    raise ValueError("S06 soft routing supports prefill only")
                ffn_probabilities = _select_soft_request_route(
                    request_state=request_state,
                    layer_index=self.layer_index,
                    unit_type="ffn",
                    incoming_hidden=hidden_states,
                    prompt_attention_mask=prompt_attention_mask,
                    soft_router=soft_router,
                    trace=trace,
                )
                ffn_bits = None
            else:
                ffn_bits = _select_request_route(
                    request_state=request_state,
                    layer_index=self.layer_index,
                    unit_type="ffn",
                    incoming_hidden=hidden_states,
                    prompt_attention_mask=prompt_attention_mask,
                    phase=phase,
                    routing_policy=routing_policy,
                    trace=trace,
                )
        else:
            if precision_plan is None:
                raise ValueError("S04 execution requires precision_plan")
            ffn_bits = precision_plan.ffn_bits[self.layer_index]
        if request_state is not None:
            trace.record_event(
                request_id=request_state.request_id,
                layer_index=self.layer_index,
                unit_type="ffn",
                phase=phase,
                event="unit_execute",
                precision=ffn_bits,
            )
        hidden_states = self.mlp(
            hidden_states,
            selected_bits=ffn_bits,
            routing_probabilities=ffn_probabilities,
            trace=trace,
            request_state=request_state,
            on_demand_context=on_demand_context,
        )
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
        precision_plan: PrecisionPlan | None,
        trace: PrecisionTrace,
        phase: str | None = None,
        request_state: QaqRequestState | None = None,
        routing_policy: Any = None,
        soft_router: Any = None,
        prompt_attention_mask: torch.Tensor | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
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
                phase=phase,
                request_state=request_state,
                routing_policy=routing_policy,
                soft_router=soft_router,
                prompt_attention_mask=prompt_attention_mask,
                on_demand_context=on_demand_context,
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
    """S04 explicit-plan wrapper plus the request-owned S05 lifecycle."""

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

    def create_on_demand_request(
        self, request_state: QaqRequestState
    ) -> SynchronousPackedRequest:
        """Create a fresh request-local context for this on-demand model."""

        sources = {
            module.module_path: module.source
            for module in self.modules()
            if isinstance(module, _OnDemandRoutedPackedLinear)
        }
        if not sources:
            raise RuntimeError("this model has no CPU-authoritative on-demand targets")
        return SynchronousPackedRequest(
            sources,
            request_state,
            self.model.embed_tokens.weight.device,
        )

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
        precision_plan: PrecisionPlan | None = None,
        trace: PrecisionTrace | None = None,
        phase: str | None = None,
        request_state: QaqRequestState | None = None,
        routing_policy: Any = None,
        soft_router: Any = None,
        prompt_attention_mask: torch.Tensor | None = None,
        on_demand_context: SynchronousPackedRequest | None = None,
        **kwargs: Any,
    ) -> Any:
        if request_state is None:
            if (
                phase is not None
                or routing_policy is not None
                or soft_router is not None
                or on_demand_context is not None
            ):
                raise ValueError("routing controls require an S05 request_state")
            if not isinstance(precision_plan, PrecisionPlan):
                raise TypeError("precision_plan must be a PrecisionPlan")
            precision_plan.validate()
            active_phase = None
            prompt_attention_mask = None
            on_demand_context = None
        else:
            if not isinstance(request_state, QaqRequestState):
                raise TypeError("request_state must be a QaqRequestState")
            if phase not in ("prefill", "decode"):
                raise ValueError("S05 request execution requires phase='prefill' or phase='decode'")
            if soft_router is not None and (phase != "prefill" or routing_policy is not None):
                raise ValueError("S06 soft routing requires prefill without a hard routing policy")
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if input_ids is None and inputs_embeds is None:
                raise ValueError("S05 request execution requires input_ids or inputs_embeds")
            if input_ids is not None:
                if input_ids.ndim != 2 or input_ids.shape[0] != 1:
                    raise ValueError("S05 request execution supports only batch-size-one input_ids")
            elif inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
                raise ValueError("S05 request execution supports only batch-size-one inputs_embeds")
            request_state.bind_owner(self)
            sequence_length = int(
                input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
            )
            feature_dim = int(
                inputs_embeds.shape[-1]
                if inputs_embeds is not None
                else self.model.embed_tokens.embedding_dim
            )
            request_state.validate_for_model(layer_count=LAYER_COUNT, feature_dim=feature_dim)
            if phase == "prefill":
                prompt_attention_mask = validate_prompt_mask(
                    prompt_attention_mask if prompt_attention_mask is not None else attention_mask,
                    sequence_length=sequence_length,
                )
                request_state.begin_prefill(prompt_length=int(prompt_attention_mask.sum().item()))
                if routing_policy is None and soft_router is None:
                    routing_policy = precision_plan
            else:
                if soft_router is not None:
                    raise ValueError("S06 soft routing supports prefill only")
                request_state.assert_complete()
                prompt_attention_mask = None
            active_phase = phase
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
            phase=active_phase,
            request_state=request_state,
            routing_policy=routing_policy,
            soft_router=soft_router,
            prompt_attention_mask=prompt_attention_mask,
            on_demand_context=on_demand_context,
            **kwargs,
        )
        if request_state is not None and phase == "prefill":
            if soft_router is not None:
                request_state.assert_soft_complete()
            else:
                request_state.assert_complete()
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


def _wrap_on_demand_targets(static_model: nn.Module) -> None:
    """Replace packed modules before moving the graph so no GPU copy survives."""

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
            raise ValueError(f"target is outside the S08 route scopes: {module_path}")
        _replace_module(
            static_model,
            module_path,
            _OnDemandRoutedPackedLinear(
                PackedLinearSource.from_module(packed, module_path),
                layer_index=layer_index,
                unit_type=unit_type,
                module_path=module_path,
            ),
        )


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


def load_on_demand_model(
    artifact: str | os.PathLike[str], device: str
) -> ManualRoutedQwen3ForCausalLM:
    """Load Qwen3 with CPU-authoritative packed sources and no resident copy."""

    static_model = load_static_model(artifact, "cpu")
    _wrap_on_demand_targets(static_model)
    model = ManualRoutedQwen3ForCausalLM(static_model)
    model.to(device)
    model.eval()
    return model
