"""Request-owned S05 features and fixed route selections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import torch

SUPPORTED_BITS = (4, 8)
DEFAULT_LAYER_COUNT = 36


def _validate_route_list(name: str, values: list[int | None], layer_count: int) -> None:
    if len(values) != layer_count:
        raise ValueError(f"{name} must contain exactly {layer_count} entries; got {len(values)}")
    for index, value in enumerate(values):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value not in SUPPORTED_BITS
        ):
            raise ValueError(f"{name}[{index}] must be None, 4, or 8; got {value!r}")


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
    feature_dim: int | None = None
    layer_count: int = DEFAULT_LAYER_COUNT
    _owner: object | None = field(default=None, init=False, repr=False)
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
            [None] * self.layer_count
            if self.ffn_probabilities is None
            else self.ffn_probabilities
        )
        _validate_route_list("attention_routes", self.attention_routes, self.layer_count)
        _validate_route_list("ffn_routes", self.ffn_routes, self.layer_count)
        for name, probabilities in (
            ("attention_probabilities", self.attention_probabilities),
            ("ffn_probabilities", self.ffn_probabilities),
        ):
            if len(probabilities) != self.layer_count:
                raise ValueError(
                    f"{name} must contain exactly {self.layer_count} entries; got {len(probabilities)}"
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

    def bind_owner(self, owner: object) -> None:
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
        _validate_route_list("attention_routes", self.attention_routes, layer_count)
        _validate_route_list("ffn_routes", self.ffn_routes, layer_count)

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
            raise RuntimeError("prefill cannot overwrite probabilities already stored in request state")

    def store_feature(self, unit_type: str, layer_index: int, feature: torch.Tensor) -> None:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported routing unit: {unit_type}")
        if not isinstance(feature, torch.Tensor) or feature.ndim != 1:
            raise ValueError("stored request feature must be a one-dimensional torch.Tensor")
        if self.feature_dim is None:
            self.feature_dim = int(feature.shape[0])
        if feature.shape[0] != self.feature_dim:
            raise ValueError(
                f"stored feature dimension must be {self.feature_dim}; got {feature.shape[0]}"
            )
        features = self.attention_features if unit_type == "attention" else self.ffn_features
        if features[layer_index] is not None:
            raise RuntimeError(f"{unit_type} feature for layer {layer_index} is already stored")
        features[layer_index] = feature.detach().clone()

    def store_probability(
        self, unit_type: str, layer_index: int, probabilities: torch.Tensor
    ) -> None:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported routing unit: {unit_type}")
        if not isinstance(probabilities, torch.Tensor) or probabilities.shape != (2,):
            raise ValueError("stored routing probabilities must have shape [2]")
        if not torch.isfinite(probabilities).all() or not torch.all(probabilities >= 0):
            raise ValueError("stored routing probabilities must be finite and non-negative")
        if not torch.allclose(probabilities.sum(), probabilities.new_tensor(1), atol=1e-6, rtol=0):
            raise ValueError("stored routing probabilities must sum to one")
        features = self.attention_features if unit_type == "attention" else self.ffn_features
        probabilities_by_unit = (
            self.attention_probabilities
            if unit_type == "attention"
            else self.ffn_probabilities
        )
        if features[layer_index] is None:
            raise RuntimeError(f"cannot store {unit_type} probability before its feature")
        if probabilities_by_unit[layer_index] is not None:
            raise RuntimeError(f"{unit_type} probability for layer {layer_index} is already stored")
        probabilities_by_unit[layer_index] = probabilities.detach().clone()

    def store_route(self, unit_type: str, layer_index: int, precision: int) -> None:
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision not in SUPPORTED_BITS
        ):
            raise ValueError(f"stored route must be 4 or 8; got {precision!r}")
        features = self.attention_features if unit_type == "attention" else self.ffn_features
        routes = self.attention_routes if unit_type == "attention" else self.ffn_routes
        if features[layer_index] is None:
            raise RuntimeError(f"cannot store {unit_type} route before its feature")
        if routes[layer_index] is not None:
            raise RuntimeError(f"{unit_type} route for layer {layer_index} is already stored")
        routes[layer_index] = precision

    def route_for_decode(self, unit_type: str, layer_index: int) -> int:
        routes = self.attention_routes if unit_type == "attention" else self.ffn_routes
        precision = routes[layer_index]
        if precision is None:
            raise RuntimeError(
                f"decode requested before {unit_type} route for layer {layer_index} was stored"
            )
        if (
            isinstance(precision, bool)
            or not isinstance(precision, int)
            or precision not in SUPPORTED_BITS
        ):
            raise ValueError(
                f"{unit_type} route for layer {layer_index} must be an integer 4 or 8; got {precision!r}"
            )
        return precision

    def assert_complete(self) -> None:
        if any(feature is None for feature in self.attention_features + self.ffn_features):
            raise RuntimeError("request state is missing one or more prefill features")
        if any(route is None for route in self.attention_routes + self.ffn_routes):
            raise RuntimeError("request state is missing one or more prefill routes")

    def assert_soft_complete(self) -> None:
        if any(feature is None for feature in self.attention_features + self.ffn_features):
            raise RuntimeError("request state is missing one or more prefill features")
        if any(
            probability is None
            for probability in self.attention_probabilities + self.ffn_probabilities
        ):
            raise RuntimeError("request state is missing one or more soft routing probabilities")


RequestState = QaqRequestState
