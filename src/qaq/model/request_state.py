"""Request-owned prompt features, routing decisions, and provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch

from ..router.network import CANDIDATE_BITS, validate_candidate_bits, validate_probabilities

SUPPORTED_BITS = CANDIDATE_BITS
DEFAULT_LAYER_COUNT = 36
SAME_UNIT = "same_unit"
LOOKAHEAD_ATTENTION_ONE_UNIT = "lookahead_attention_one_unit"
ROUTING_TIMINGS = (SAME_UNIT, LOOKAHEAD_ATTENTION_ONE_UNIT)
POST_ATTENTION_PRE_FFN = "post_attention_pre_ffn"


def validate_routing_timing(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("routing_timing must be a string")
    if value not in ROUTING_TIMINGS:
        raise ValueError(f"routing_timing must be one of {ROUTING_TIMINGS}; got {value!r}")
    return value


@dataclass(frozen=True, slots=True)
class RoutingProvenance:
    """Source evidence for one target-owned routing feature and decision."""

    source_layer: int
    target_layer: int
    target_unit_type: str
    source_point: str
    routing_timing: str

    def __post_init__(self) -> None:
        for name in ("source_layer", "target_layer"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.target_unit_type not in ("attention", "ffn"):
            raise ValueError("target_unit_type must be attention or ffn")
        validate_routing_timing(self.routing_timing)
        if not isinstance(self.source_point, str) or not self.source_point:
            raise ValueError("source_point must be a non-empty string")
        if self.routing_timing == LOOKAHEAD_ATTENTION_ONE_UNIT:
            if self.target_unit_type != "attention":
                raise ValueError("one-unit lookahead provenance is attention-only")
            if self.target_layer != self.source_layer + 1:
                raise ValueError("one-unit lookahead target_layer must equal source_layer + 1")
            if self.source_point != POST_ATTENTION_PRE_FFN:
                raise ValueError(
                    f"one-unit lookahead source_point must be {POST_ATTENTION_PRE_FFN}"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "target_unit_type": self.target_unit_type,
            "source_point": self.source_point,
            "routing_timing": self.routing_timing,
        }


def _validate_route_list(
    name: str,
    values: list[int | None],
    layer_count: int,
    candidate_bits: tuple[int, ...],
) -> None:
    if len(values) != layer_count:
        raise ValueError(f"{name} must contain exactly {layer_count} entries; got {len(values)}")
    for index, value in enumerate(values):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value not in candidate_bits
        ):
            allowed = "None, 4, or 8" if candidate_bits == (4, 8) else f"one of {candidate_bits}"
            raise ValueError(f"{name}[{index}] must be {allowed}; got {value!r}")


def _validate_features(
    name: str, values: list[torch.Tensor | None], layer_count: int, feature_dim: int | None
) -> int | None:
    if len(values) != layer_count:
        raise ValueError(f"{name} must contain exactly {layer_count} entries; got {len(values)}")
    inferred = feature_dim
    for index, value in enumerate(values):
        if value is None:
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name}[{index}] must be a torch.Tensor or None")
        if value.ndim != 1:
            raise ValueError(
                f"{name}[{index}] must have shape [feature_dim]; got {tuple(value.shape)}"
            )
        if inferred is None:
            inferred = int(value.shape[0])
        if value.shape[0] != inferred:
            raise ValueError(
                f"{name}[{index}] feature dimension must be {inferred}; got {value.shape[0]}"
            )
    return inferred


@dataclass(slots=True)
class QaqRequestState:
    """Mutable state owned by exactly one request execution.

    ``request_id`` is descriptive metadata, not a lookup key.  Independent
    state objects may use the same identifier, but they never share storage;
    a model binds to the concrete state object and rejects a different owner.
    This avoids a process-global request registry and prevents route leakage.
    """

    request_id: str
    prompt_length: int
    attention_routes: list[int | None] | None = None
    ffn_routes: list[int | None] | None = None
    attention_features: list[torch.Tensor | None] | None = None
    ffn_features: list[torch.Tensor | None] | None = None
    attention_probabilities: list[torch.Tensor | None] | None = None
    ffn_probabilities: list[torch.Tensor | None] | None = None
    attention_provenance: list[RoutingProvenance | None] | None = None
    ffn_provenance: list[RoutingProvenance | None] | None = None
    feature_dim: int | None = None
    layer_count: int = DEFAULT_LAYER_COUNT
    candidate_bits: tuple[int, ...] = CANDIDATE_BITS
    routing_timing: str = SAME_UNIT
    _owner: object | None = field(default=None, init=False, repr=False)
    _cleanup_callbacks: list[object] = field(default_factory=list, init=False, repr=False)
    _attention_route_consumed: list[bool] = field(default_factory=list, init=False, repr=False)
    _attention_probability_consumed: list[bool] = field(
        default_factory=list, init=False, repr=False
    )
    _ended: bool = field(default=False, init=False, repr=False)
    _DEFAULT_LAYER_COUNT: ClassVar[int] = DEFAULT_LAYER_COUNT

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if isinstance(self.prompt_length, bool) or not isinstance(self.prompt_length, int):
            raise TypeError("prompt_length must be an integer")
        if self.prompt_length <= 0:
            raise ValueError("prompt_length must be positive")
        if isinstance(self.layer_count, bool) or not isinstance(self.layer_count, int):
            raise TypeError("layer_count must be an integer")
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        self.candidate_bits = validate_candidate_bits(self.candidate_bits)
        self.routing_timing = validate_routing_timing(self.routing_timing)
        if self.feature_dim is not None:
            if isinstance(self.feature_dim, bool) or not isinstance(self.feature_dim, int):
                raise TypeError("feature_dim must be an integer or None")
            if self.feature_dim <= 0:
                raise ValueError("feature_dim must be positive")
        self.attention_routes = list(
            [None] * self.layer_count if self.attention_routes is None else self.attention_routes
        )
        self.ffn_routes = list(
            [None] * self.layer_count if self.ffn_routes is None else self.ffn_routes
        )
        self.attention_features = list(
            [None] * self.layer_count
            if self.attention_features is None
            else self.attention_features
        )
        self.ffn_features = list(
            [None] * self.layer_count if self.ffn_features is None else self.ffn_features
        )
        self.attention_probabilities = list(
            [None] * self.layer_count
            if self.attention_probabilities is None
            else self.attention_probabilities
        )
        self.ffn_probabilities = list(
            [None] * self.layer_count if self.ffn_probabilities is None else self.ffn_probabilities
        )
        self.attention_provenance = list(
            [None] * self.layer_count
            if self.attention_provenance is None
            else self.attention_provenance
        )
        self.ffn_provenance = list(
            [None] * self.layer_count if self.ffn_provenance is None else self.ffn_provenance
        )
        self._attention_route_consumed = [False] * self.layer_count
        self._attention_probability_consumed = [False] * self.layer_count
        _validate_route_list(
            "attention_routes", self.attention_routes, self.layer_count, self.candidate_bits
        )
        _validate_route_list("ffn_routes", self.ffn_routes, self.layer_count, self.candidate_bits)
        for name, probabilities in (
            ("attention_probabilities", self.attention_probabilities),
            ("ffn_probabilities", self.ffn_probabilities),
        ):
            if len(probabilities) != self.layer_count:
                raise ValueError(
                    f"{name} must contain exactly {self.layer_count} entries; got {len(probabilities)}"
                )
            for layer_index, probability in enumerate(probabilities):
                if probability is not None:
                    validate_probabilities(
                        probability,
                        self.candidate_bits,
                        context=f"{name}[{layer_index}]",
                        require_vector=True,
                    )
        for name, values in (
            ("attention_provenance", self.attention_provenance),
            ("ffn_provenance", self.ffn_provenance),
        ):
            if len(values) != self.layer_count:
                raise ValueError(
                    f"{name} must contain exactly {self.layer_count} entries; got {len(values)}"
                )
            for layer_index, provenance in enumerate(values):
                if provenance is not None:
                    self._validate_provenance(
                        "attention" if name == "attention_provenance" else "ffn",
                        layer_index,
                        provenance,
                    )
        attention_dim = _validate_features(
            "attention_features", self.attention_features, self.layer_count, self.feature_dim
        )
        ffn_dim = _validate_features(
            "ffn_features", self.ffn_features, self.layer_count, attention_dim
        )
        if attention_dim is not None and ffn_dim is not None and attention_dim != ffn_dim:
            raise ValueError("attention and FFN feature dimensions must match")
        self.feature_dim = ffn_dim or attention_dim

    @property
    def ended(self) -> bool:
        """Whether explicit request-end cleanup has completed."""

        return self._ended

    @property
    def early_attention_route_consumed(self) -> tuple[bool, ...]:
        return tuple(self._attention_route_consumed)

    @property
    def early_attention_probability_consumed(self) -> tuple[bool, ...]:
        return tuple(self._attention_probability_consumed)

    def _validate_layer_index(self, layer_index: int) -> None:
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.layer_count
        ):
            raise ValueError(f"layer_index must be in [0, {self.layer_count}); got {layer_index}")

    def _validate_provenance(
        self, unit_type: str, layer_index: int, provenance: RoutingProvenance
    ) -> None:
        if not isinstance(provenance, RoutingProvenance):
            raise TypeError("routing provenance must be a RoutingProvenance")
        if provenance.target_layer != layer_index or provenance.target_unit_type != unit_type:
            raise ValueError("routing provenance must name its target-owned layer and unit")
        if provenance.routing_timing != self.routing_timing:
            raise ValueError("routing provenance timing must match the request state")
        if provenance.source_layer >= self.layer_count:
            raise ValueError("routing provenance source_layer is outside the request state")
        if (
            self.routing_timing == LOOKAHEAD_ATTENTION_ONE_UNIT
            and unit_type == "attention"
            and layer_index > 0
            and provenance.source_layer != layer_index - 1
        ):
            raise ValueError("lookahead attention provenance must map source s to target s+1")

    def register_cleanup(self, callback: object) -> None:
        """Register one request-owned cleanup callback for explicit request end."""

        if self._ended:
            raise RuntimeError("cannot register cleanup after request end")
        if not callable(callback):
            raise TypeError("request cleanup callback must be callable")
        self._cleanup_callbacks.append(callback)

    def end_request(self) -> None:
        """Run and clear request-owned cleanup callbacks exactly once."""

        if self._ended:
            return
        self._ended = True
        callbacks = tuple(self._cleanup_callbacks)
        self._cleanup_callbacks.clear()
        for callback in callbacks:
            callback()
        for values in (
            self.attention_routes,
            self.ffn_routes,
            self.attention_features,
            self.ffn_features,
            self.attention_probabilities,
            self.ffn_probabilities,
            self.attention_provenance,
            self.ffn_provenance,
        ):
            values[:] = [None] * self.layer_count
        self._attention_route_consumed[:] = [False] * self.layer_count
        self._attention_probability_consumed[:] = [False] * self.layer_count
        self._owner = None

    def bind_owner(self, owner: object) -> None:
        if self._ended:
            raise RuntimeError(f"request state {self.request_id!r} has already ended")
        if self._owner is None:
            self._owner = owner
        elif self._owner is not owner:
            raise RuntimeError(
                f"request state {self.request_id!r} is already owned by another model execution"
            )

    def validate_for_model(self, *, layer_count: int, feature_dim: int) -> None:
        if self.layer_count != layer_count:
            raise ValueError(
                f"request state layer_count must be {layer_count}; got {self.layer_count}"
            )
        if self.feature_dim is not None and self.feature_dim != feature_dim:
            raise ValueError(
                f"request state feature_dim must be {feature_dim}; got {self.feature_dim}"
            )
        self.feature_dim = feature_dim
        _validate_route_list(
            "attention_routes", self.attention_routes, layer_count, self.candidate_bits
        )
        _validate_route_list("ffn_routes", self.ffn_routes, layer_count, self.candidate_bits)

    def begin_prefill(self, *, prompt_length: int) -> None:
        if prompt_length != self.prompt_length:
            raise ValueError(
                f"request state prompt_length must be {self.prompt_length}; got {prompt_length}"
            )
        if any(route is not None for route in self.attention_routes + self.ffn_routes):
            raise RuntimeError("prefill cannot overwrite routes already stored in request state")
        if any(feature is not None for feature in self.attention_features + self.ffn_features):
            raise RuntimeError("prefill cannot overwrite features already stored in request state")
        if any(
            probability is not None
            for probability in self.attention_probabilities + self.ffn_probabilities
        ):
            raise RuntimeError(
                "prefill cannot overwrite probabilities already stored in request state"
            )
        if any(
            provenance is not None for provenance in self.attention_provenance + self.ffn_provenance
        ):
            raise RuntimeError(
                "prefill cannot overwrite provenance already stored in request state"
            )
        if any(self._attention_route_consumed) or any(self._attention_probability_consumed):
            raise RuntimeError("prefill cannot reuse consumed early attention decisions")

    def store_feature(
        self,
        unit_type: str,
        layer_index: int,
        feature: torch.Tensor,
        *,
        provenance: RoutingProvenance | None = None,
    ) -> None:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported routing unit: {unit_type}")
        self._validate_layer_index(layer_index)
        if not isinstance(feature, torch.Tensor) or feature.ndim != 1:
            raise ValueError("stored request feature must be a one-dimensional torch.Tensor")
        if self.feature_dim is None:
            self.feature_dim = int(feature.shape[0])
        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"stored feature dimension must be {self.feature_dim}; got {feature.shape[0]}"
            )
        features = self.attention_features if unit_type == "attention" else self.ffn_features
        provenance_by_unit = (
            self.attention_provenance if unit_type == "attention" else self.ffn_provenance
        )
        if features[layer_index] is not None or provenance_by_unit[layer_index] is not None:
            raise RuntimeError(f"{unit_type} feature for layer {layer_index} is already stored")
        if (
            self.routing_timing == LOOKAHEAD_ATTENTION_ONE_UNIT
            and unit_type == "attention"
            and layer_index > 0
            and provenance is None
        ):
            raise RuntimeError("lookahead attention features require source provenance")
        if provenance is not None:
            self._validate_provenance(unit_type, layer_index, provenance)
        features[layer_index] = feature.detach().clone()
        provenance_by_unit[layer_index] = provenance

    def store_probability(
        self, unit_type: str, layer_index: int, probabilities: torch.Tensor
    ) -> None:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported routing unit: {unit_type}")
        self._validate_layer_index(layer_index)
        validate_probabilities(
            probabilities,
            self.candidate_bits,
            context="stored routing probabilities",
            require_vector=True,
        )
        features = self.attention_features if unit_type == "attention" else self.ffn_features
        probabilities_by_unit = (
            self.attention_probabilities if unit_type == "attention" else self.ffn_probabilities
        )
        if features[layer_index] is None:
            raise RuntimeError(f"cannot store {unit_type} probability before its feature")
        if probabilities_by_unit[layer_index] is not None:
            raise RuntimeError(f"{unit_type} probability for layer {layer_index} is already stored")
        # Keep the clone connected to the router graph so request-level
        # auxiliary objectives can backpropagate through stored decisions.
        probabilities_by_unit[layer_index] = probabilities.clone()

    def store_route(self, unit_type: str, layer_index: int, precision: int) -> None:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported routing unit: {unit_type}")
        self._validate_layer_index(layer_index)
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision not in self.candidate_bits
        ):
            allowed = "4 or 8" if self.candidate_bits == (4, 8) else f"one of {self.candidate_bits}"
            raise ValueError(f"stored route must be {allowed}; got {precision!r}")
        features = self.attention_features if unit_type == "attention" else self.ffn_features
        routes = self.attention_routes if unit_type == "attention" else self.ffn_routes
        if features[layer_index] is None:
            raise RuntimeError(f"cannot store {unit_type} route before its feature")
        if routes[layer_index] is not None:
            raise RuntimeError(f"{unit_type} route for layer {layer_index} is already stored")
        routes[layer_index] = precision

    def route_for_decode(self, unit_type: str, layer_index: int) -> int:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported routing unit: {unit_type}")
        self._validate_layer_index(layer_index)
        routes = self.attention_routes if unit_type == "attention" else self.ffn_routes
        precision = routes[layer_index]
        if precision is None:
            raise RuntimeError(
                f"decode requested before {unit_type} route for layer {layer_index} was stored"
            )
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision not in self.candidate_bits
        ):
            allowed = (
                "integer 4 or 8"
                if self.candidate_bits == (4, 8)
                else f"integer from {self.candidate_bits}"
            )
            raise ValueError(
                f"{unit_type} route for layer {layer_index} must be an {allowed}; got {precision!r}"
            )
        return precision

    def _consume_early_attention(
        self, target_layer: int, *, probability: bool
    ) -> torch.Tensor | int:
        self._validate_layer_index(target_layer)
        if self.routing_timing != LOOKAHEAD_ATTENTION_ONE_UNIT or target_layer == 0:
            raise RuntimeError("early attention consumption requires a lookahead target layer")
        provenance = self.attention_provenance[target_layer]
        if provenance is None:
            raise RuntimeError(
                f"lookahead attention target layer {target_layer} is missing early provenance"
            )
        self._validate_provenance("attention", target_layer, provenance)
        consumed = (
            self._attention_probability_consumed if probability else self._attention_route_consumed
        )
        if consumed[target_layer]:
            raise RuntimeError(
                f"lookahead attention target layer {target_layer} was already consumed"
            )
        values = self.attention_probabilities if probability else self.attention_routes
        value = values[target_layer]
        if value is None:
            kind = "probability" if probability else "route"
            raise RuntimeError(
                f"lookahead attention target layer {target_layer} is missing its early {kind}"
            )
        consumed[target_layer] = True
        return value

    def consume_early_attention_route(self, target_layer: int) -> int:
        value = self._consume_early_attention(target_layer, probability=False)
        if not isinstance(value, int):  # pragma: no cover - route storage validates this
            raise TypeError("stored early attention route is invalid")
        return value

    def consume_early_attention_probability(self, target_layer: int) -> torch.Tensor:
        value = self._consume_early_attention(target_layer, probability=True)
        if not isinstance(value, torch.Tensor):  # pragma: no cover - storage validates this
            raise TypeError("stored early attention probability is invalid")
        return value

    def _assert_lookahead_coverage(self, *, soft: bool) -> None:
        if self.routing_timing != LOOKAHEAD_ATTENTION_ONE_UNIT:
            return
        for target_layer in range(1, self.layer_count):
            provenance = self.attention_provenance[target_layer]
            if provenance is None:
                raise RuntimeError("request state is missing lookahead attention provenance")
            self._validate_provenance("attention", target_layer, provenance)
        consumed = self._attention_probability_consumed if soft else self._attention_route_consumed
        if any(not consumed[layer] for layer in range(1, self.layer_count)):
            raise RuntimeError("request state has an unconsumed lookahead attention decision")

    def assert_complete(self) -> None:
        if any(feature is None for feature in self.attention_features + self.ffn_features):
            raise RuntimeError("request state is missing one or more prefill features")
        if any(route is None for route in self.attention_routes + self.ffn_routes):
            raise RuntimeError("request state is missing one or more prefill routes")
        self._assert_lookahead_coverage(soft=False)

    def assert_soft_complete(self) -> None:
        if any(feature is None for feature in self.attention_features + self.ffn_features):
            raise RuntimeError("request state is missing one or more prefill features")
        if any(
            probability is None
            for probability in self.attention_probabilities + self.ffn_probabilities
        ):
            raise RuntimeError("request state is missing one or more soft routing probabilities")
        self._assert_lookahead_coverage(soft=True)


RequestState = QaqRequestState
