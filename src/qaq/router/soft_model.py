"""S06 trainable soft routing over the frozen S04 packed execution graph."""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from ..model.manual import LAYER_COUNT, PrecisionTrace, load_manual_model
from ..model.request_state import QaqRequestState
from .network import (
    CANDIDATE_BITS,
    ROUTER_HIDDEN_WIDTH,
    ROUTER_TEMPERATURE,
    SoftPrecisionRouter,
    router_parameter_count,
    trainable_parameter_audit,
    validate_candidate_bits,
)


class SoftRoutedQwen3ForCausalLM(nn.Module):
    """S06 model: 72 distinct routers drive both packed paths per projection."""

    def __init__(
        self,
        manual_model: nn.Module,
        *,
        hidden_width: int = ROUTER_HIDDEN_WIDTH,
        temperature: float = ROUTER_TEMPERATURE,
        candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
    ) -> None:
        super().__init__()
        self.base = manual_model
        self.candidate_bits = validate_candidate_bits(candidate_bits)
        self.feature_dim = int(self.base.model.embed_tokens.embedding_dim)
        self.routers = nn.ModuleDict(
            {
                f"{unit_type}_{layer_index}": SoftPrecisionRouter(
                    self.feature_dim,
                    hidden_width=hidden_width,
                    temperature=temperature,
                    candidate_bits=self.candidate_bits,
                )
                for unit_type in ("attention", "ffn")
                for layer_index in range(LAYER_COUNT)
            }
        )
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.base.eval()

    def train(self, mode: bool = True) -> SoftRoutedQwen3ForCausalLM:
        super().train(mode)
        # The baseline freezes the packed model and keeps its execution deterministic.
        self.base.eval()
        return self

    @property
    def router_count(self) -> int:
        return len(self.routers)

    @property
    def router_parameter_count(self) -> int:
        return sum(router_parameter_count(router) for router in self.routers.values())

    def route(self, layer_index: int, unit_type: str, feature: torch.Tensor) -> torch.Tensor:
        if unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported S06 routing unit: {unit_type}")
        if not 0 <= layer_index < LAYER_COUNT:
            raise ValueError(f"layer_index must be in [0, {LAYER_COUNT}); got {layer_index}")
        return self.routers[f"{unit_type}_{layer_index}"](feature.detach())

    def parameter_audit(self) -> dict[str, object]:
        return trainable_parameter_audit(self)

    def forward(
        self,
        *,
        request_state: QaqRequestState,
        phase: str = "prefill",
        trace: PrecisionTrace | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if not isinstance(request_state, QaqRequestState):
            raise TypeError("S06 soft execution requires a QaqRequestState")
        if request_state.candidate_bits != self.candidate_bits:
            raise ValueError(
                "request state candidate_bits do not match the soft router: "
                f"{request_state.candidate_bits} != {self.candidate_bits}"
            )
        if phase != "prefill":
            raise ValueError("S06 soft execution supports phase='prefill' only")
        if trace is None:
            trace = PrecisionTrace()
        return self.base(
            request_state=request_state,
            phase=phase,
            trace=trace,
            precision_plan=None,
            routing_policy=None,
            soft_router=self.route,
            prompt_attention_mask=prompt_attention_mask,
            **kwargs,
        )


def load_soft_model(
    artifact: str | os.PathLike[str],
    device: str,
    *,
    hidden_width: int = ROUTER_HIDDEN_WIDTH,
    temperature: float = ROUTER_TEMPERATURE,
    candidate_bits: tuple[int, ...] = CANDIDATE_BITS,
) -> SoftRoutedQwen3ForCausalLM:
    """Load the verified S03 artifact and add only S06 router parameters."""

    return SoftRoutedQwen3ForCausalLM(
        load_manual_model(artifact, device),
        hidden_width=hidden_width,
        temperature=temperature,
        candidate_bits=candidate_bits,
    )


__all__ = ["SoftRoutedQwen3ForCausalLM", "load_soft_model"]
