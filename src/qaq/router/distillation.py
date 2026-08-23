"""S07 teacher-student router distillation and cost-composition primitives.

The module does not select routes by sampling or own teacher/packed model
weights. S07's masked KL remains unchanged; S10-C's optional normalized
bit-plane-count term composes with it without claiming hardware cost. The S06
model remains the execution owner; this module supplies the explicit data,
loss, freeze, optimizer, checkpoint, objective, and observation seams.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .network import (
    CANDIDATE_BITS,
    THREE_WAY_CANDIDATE_BITS,
    validate_candidate_bits,
    validate_probabilities,
)

SUPPORTED_BITS = (4, 8)
CANDIDATE_ORDERING = CANDIDATE_BITS
CHECKPOINT_FORMAT_VERSION = 1
CAUSAL_TARGET_IGNORE_INDEX = -100


def _require_finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise TypeError(f"{name} must be a finite positive number")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number; got {value}")
    return value


def _validate_1d_mask(mask: torch.Tensor, *, name: str, length: int) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 1 or mask.shape[0] != length:
        raise ValueError(f"{name} must have shape [{length}]")
    if mask.dtype == torch.bool:
        result = mask
    elif mask.is_floating_point() or mask.dtype in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        if not torch.isfinite(mask).all() or not torch.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} must contain only finite 0/1 values")
        result = mask != 0
    else:
        raise TypeError(f"{name} has unsupported dtype {mask.dtype}")
    return result.to(dtype=torch.bool)


def causal_target_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Build target IDs aligned with causal logits at the same sequence positions."""

    if not isinstance(input_ids, torch.Tensor) or input_ids.ndim not in (1, 2):
        raise ValueError("input_ids must have shape [sequence] or [batch, sequence]")
    if input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise TypeError("input_ids must be an integer tensor")
    if not isinstance(attention_mask, torch.Tensor) or attention_mask.shape != input_ids.shape:
        raise ValueError("attention_mask must have the same shape as input_ids")
    if input_ids.ndim == 1:
        mask = _validate_1d_mask(
            attention_mask, name="attention_mask", length=int(input_ids.shape[0])
        )
    else:
        mask = _validate_batch_mask(attention_mask, "attention_mask")
    mask = mask.to(device=input_ids.device)
    expected = torch.full_like(input_ids, CAUSAL_TARGET_IGNORE_INDEX)
    if input_ids.shape[-1] > 1:
        linked = mask[..., :-1] & mask[..., 1:]
        expected[..., :-1] = torch.where(linked, input_ids[..., 1:], expected[..., :-1])
    return expected


def _validate_causal_target_alignment(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_ids: torch.Tensor,
    completion_loss_mask: torch.Tensor,
) -> None:
    expected = causal_target_ids(input_ids, attention_mask)
    if target_ids.device != expected.device or not torch.equal(target_ids, expected):
        raise ValueError(
            "target_ids must align causally: target_ids[t] must equal input_ids[t+1] "
            "for linked valid tokens and use -100 otherwise"
        )
    completion = completion_loss_mask.to(device=target_ids.device, dtype=torch.bool)
    if bool((completion & (target_ids == CAUSAL_TARGET_IGNORE_INDEX)).any()):
        raise ValueError(
            "completion_loss_mask must select causal target positions with valid target_ids"
        )


@dataclass(frozen=True, slots=True)
class TokenRange:
    """Half-open token range in one aligned sequence."""

    start: int
    end: int

    def validate(self, sequence_length: int, name: str) -> None:
        if (
            isinstance(self.start, bool)
            or isinstance(self.end, bool)
            or not isinstance(self.start, int)
            or not isinstance(self.end, int)
            or not 0 <= self.start < self.end <= sequence_length
        ):
            raise ValueError(
                f"{name} must be a non-empty half-open range within [0, {sequence_length}): {self}"
            )


@dataclass(frozen=True, slots=True)
class DistillationExample:
    """One explicit prompt/completion example for aligned KD.

    ``completion_loss_mask`` selects causal logit positions whose target token
    is part of the completion.  It is never inferred from ``attention_mask``.
    ``target_ids[t]`` is the token predicted by the logit at position ``t``;
    the final position and links into padding use ``-100``.
    """

    example_id: str
    tokenizer_revision: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_loss_mask: torch.Tensor
    target_ids: torch.Tensor
    prompt_text: str | None = None
    completion_text: str | None = None
    prompt_token_range: TokenRange | None = None
    completion_token_range: TokenRange | None = None
    prompt_attention_mask: torch.Tensor | None = None
    sequence_positions: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.example_id, str) or not self.example_id.strip():
            raise ValueError("example_id must be a non-empty string")
        if not isinstance(self.tokenizer_revision, str) or not self.tokenizer_revision.strip():
            raise ValueError("tokenizer_revision must be a non-empty string")
        if self.prompt_text is None and self.prompt_token_range is None:
            raise ValueError("provide prompt_text or prompt_token_range explicitly")
        if self.completion_text is None and self.completion_token_range is None:
            raise ValueError("provide completion_text or completion_token_range explicitly")
        if self.prompt_text is not None and not isinstance(self.prompt_text, str):
            raise TypeError("prompt_text must be a string or None")
        if self.completion_text is not None and not isinstance(self.completion_text, str):
            raise TypeError("completion_text must be a string or None")
        if not isinstance(self.input_ids, torch.Tensor) or self.input_ids.ndim != 1:
            raise ValueError("input_ids must have shape [sequence]")
        if self.input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError("input_ids must be an integer tensor")
        sequence_length = int(self.input_ids.shape[0])
        attention = _validate_1d_mask(
            self.attention_mask, name="attention_mask", length=sequence_length
        )
        completion_mask = _validate_1d_mask(
            self.completion_loss_mask,
            name="completion_loss_mask",
            length=sequence_length,
        )
        if not bool(attention.any()):
            raise ValueError("attention_mask must contain at least one valid token")
        if bool((completion_mask & ~attention).any()):
            raise ValueError("completion_loss_mask cannot include padding positions")
        if not bool(completion_mask.any()):
            raise ValueError("completion_loss_mask has zero valid completion targets")
        if not isinstance(self.target_ids, torch.Tensor) or self.target_ids.shape != (
            sequence_length,
        ):
            raise ValueError("target_ids must have shape [sequence] aligned with input_ids")
        if self.target_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError("target_ids must be an integer tensor")
        _validate_causal_target_alignment(
            self.input_ids, attention, self.target_ids, completion_mask
        )
        attention_for_input = attention.to(device=self.input_ids.device)
        prompt_positions: torch.Tensor | None = None
        if self.prompt_token_range is not None:
            self.prompt_token_range.validate(sequence_length, "prompt_token_range")
            prompt_positions = torch.zeros(
                sequence_length, dtype=torch.bool, device=self.input_ids.device
            )
            prompt_positions[self.prompt_token_range.start : self.prompt_token_range.end] = True
            if bool((prompt_positions & ~attention_for_input).any()):
                raise ValueError("prompt_token_range cannot include padding positions")
        elif self.prompt_attention_mask is None:
            raise ValueError("prompt_text examples require an explicit prompt_attention_mask")
        prompt_mask: torch.Tensor | None = None
        if self.prompt_attention_mask is not None:
            prompt_mask = _validate_1d_mask(
                self.prompt_attention_mask,
                name="prompt_attention_mask",
                length=sequence_length,
            )
            if bool((prompt_mask & ~attention).any()):
                raise ValueError("prompt_attention_mask cannot include padding positions")
            if prompt_positions is not None and not torch.equal(
                prompt_mask.to(device=self.input_ids.device), prompt_positions
            ):
                raise ValueError(
                    "prompt_token_range and prompt_attention_mask must represent the same tokens"
                )
        if self.completion_token_range is not None:
            self.completion_token_range.validate(sequence_length, "completion_token_range")
            if self.completion_token_range.start == 0:
                raise ValueError("completion_token_range must start after a causal context token")
            if prompt_mask is not None:
                positions_at_or_after_completion = (
                    torch.arange(sequence_length, device=prompt_mask.device)
                    >= self.completion_token_range.start
                )
                if bool((prompt_mask & positions_at_or_after_completion).any()):
                    raise ValueError(
                        "prompt_attention_mask must select only tokens before completion_token_range"
                    )
            completion_positions = torch.zeros(
                sequence_length, dtype=torch.bool, device=self.input_ids.device
            )
            completion_positions[
                self.completion_token_range.start : self.completion_token_range.end
            ] = True
            if bool((completion_positions & ~attention_for_input).any()):
                raise ValueError("completion_token_range cannot include padding positions")
            expected_completion_mask = torch.zeros(
                sequence_length, dtype=torch.bool, device=completion_mask.device
            )
            expected_completion_mask[
                self.completion_token_range.start - 1 : self.completion_token_range.end - 1
            ] = True
            if not torch.equal(completion_mask, expected_completion_mask):
                raise ValueError(
                    "completion_loss_mask must mark the causal logits for the completion-token range"
                )
        if (
            self.prompt_token_range is not None
            and self.completion_token_range is not None
            and self.prompt_token_range.end > self.completion_token_range.start
        ):
            raise ValueError("prompt_token_range must end before completion_token_range starts")
        if self.sequence_positions is not None:
            if self.sequence_positions.shape != (sequence_length,):
                raise ValueError("sequence_positions must have shape [sequence]")
            if self.sequence_positions.dtype not in (torch.int64, torch.int32):
                raise TypeError("sequence_positions must be an integer tensor")

    @property
    def sequence_length(self) -> int:
        return int(self.input_ids.shape[0])

    def prompt_mask(self) -> torch.Tensor:
        if self.prompt_token_range is not None:
            mask = torch.zeros(self.sequence_length, dtype=torch.bool, device=self.input_ids.device)
            mask[self.prompt_token_range.start : self.prompt_token_range.end] = True
            return mask
        if self.prompt_attention_mask is None:  # pragma: no cover - guarded in __post_init__
            raise ValueError("an explicit prompt_attention_mask is required")
        return self.prompt_attention_mask.to(device=self.input_ids.device, dtype=torch.bool)

    def execution_inputs(self) -> ExecutionInputs:
        positions = self.sequence_positions
        if positions is None:
            positions = torch.arange(self.sequence_length, device=self.input_ids.device)
        return ExecutionInputs(
            tokenizer_revision=self.tokenizer_revision,
            input_ids=self.input_ids.unsqueeze(0),
            attention_mask=self.attention_mask.unsqueeze(0),
            completion_loss_mask=self.completion_loss_mask.unsqueeze(0),
            sequence_positions=positions.unsqueeze(0),
        )


@dataclass(frozen=True, slots=True)
class DistillationBatch:
    """A batch of examples with every alignment field explicitly batched."""

    example_ids: tuple[str, ...]
    tokenizer_revision: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_loss_mask: torch.Tensor
    target_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    sequence_positions: torch.Tensor

    @classmethod
    def from_examples(cls, examples: Sequence[DistillationExample]) -> DistillationBatch:
        if not examples:
            raise ValueError("distillation batch cannot be empty")
        revision = examples[0].tokenizer_revision
        if any(example.tokenizer_revision != revision for example in examples):
            raise ValueError("all examples in one batch must use one tokenizer revision")
        lengths = {example.sequence_length for example in examples}
        if len(lengths) != 1:
            raise ValueError("S07 batch collation requires aligned equal sequence lengths")
        inputs = [example.execution_inputs() for example in examples]
        return cls(
            example_ids=tuple(example.example_id for example in examples),
            tokenizer_revision=revision,
            input_ids=torch.cat([item.input_ids for item in inputs]),
            attention_mask=torch.cat([item.attention_mask for item in inputs]),
            completion_loss_mask=torch.cat([item.completion_loss_mask for item in inputs]),
            target_ids=torch.stack([example.target_ids for example in examples]),
            prompt_attention_mask=torch.stack([example.prompt_mask() for example in examples]),
            sequence_positions=torch.cat([item.sequence_positions for item in inputs]),
        )

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        if not self.example_ids or len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("example_ids must be non-empty and unique within a batch")
        if not isinstance(self.tokenizer_revision, str) or not self.tokenizer_revision.strip():
            raise ValueError("tokenizer_revision must be a non-empty string")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if self.input_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError("input_ids must be an integer tensor")
        batch, sequence = self.input_ids.shape
        if len(self.example_ids) != batch:
            raise ValueError("example_ids count must equal input_ids batch dimension")
        for name, value in (
            ("attention_mask", self.attention_mask),
            ("completion_loss_mask", self.completion_loss_mask),
            ("prompt_attention_mask", self.prompt_attention_mask),
            ("sequence_positions", self.sequence_positions),
            ("target_ids", self.target_ids),
        ):
            if value.shape != (batch, sequence):
                raise ValueError(f"{name} must have shape [{batch}, {sequence}]")
        if self.target_ids.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
            raise TypeError("target_ids must be an integer tensor")
        attention = _validate_batch_mask(self.attention_mask, "attention_mask")
        completion = _validate_batch_mask(self.completion_loss_mask, "completion_loss_mask")
        prompt = _validate_batch_mask(self.prompt_attention_mask, "prompt_attention_mask")
        _validate_causal_target_alignment(self.input_ids, attention, self.target_ids, completion)
        if bool((completion & ~attention).any()):
            raise ValueError("completion_loss_mask cannot include padding positions")
        if bool((prompt & ~attention).any()):
            raise ValueError("prompt_attention_mask cannot include padding positions")
        if bool((completion.sum(dim=1) == 0).any()):
            raise ValueError("completion_loss_mask has zero valid completion targets")

    def execution_inputs(self) -> ExecutionInputs:
        return ExecutionInputs(
            tokenizer_revision=self.tokenizer_revision,
            input_ids=self.input_ids,
            attention_mask=self.attention_mask,
            completion_loss_mask=self.completion_loss_mask,
            sequence_positions=self.sequence_positions,
        )


def _validate_batch_mask(mask: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(mask, torch.Tensor) or mask.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, sequence]")
    rows = [
        _validate_1d_mask(mask[index], name=name, length=int(mask.shape[1]))
        for index in range(mask.shape[0])
    ]
    return torch.stack(rows)


@dataclass(frozen=True, slots=True)
class ExecutionInputs:
    tokenizer_revision: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    completion_loss_mask: torch.Tensor
    sequence_positions: torch.Tensor

    def validate(self) -> None:
        if not isinstance(self.tokenizer_revision, str) or not self.tokenizer_revision.strip():
            raise ValueError("tokenizer_revision must be a non-empty string")
        if self.input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        expected = self.input_ids.shape
        for name, value in (
            ("attention_mask", self.attention_mask),
            ("completion_loss_mask", self.completion_loss_mask),
            ("sequence_positions", self.sequence_positions),
        ):
            if value.shape != expected:
                raise ValueError(
                    f"{name} shape {tuple(value.shape)} does not match {tuple(expected)}"
                )
        _validate_batch_mask(self.attention_mask, "attention_mask")
        completion = _validate_batch_mask(self.completion_loss_mask, "completion_loss_mask")
        attention = _validate_batch_mask(self.attention_mask, "attention_mask")
        if bool((completion & ~attention).any()):
            raise ValueError("completion_loss_mask cannot include padding positions")
        if bool((completion.sum(dim=1) == 0).any()):
            raise ValueError("completion_loss_mask has zero valid completion targets")


def validate_execution_alignment(teacher: ExecutionInputs, student: ExecutionInputs) -> None:
    """Reject any teacher/student tokenizer, token, mask, or position drift."""

    teacher.validate()
    student.validate()
    if teacher.tokenizer_revision != student.tokenizer_revision:
        raise ValueError("teacher/student tokenizer revisions are not aligned")
    for field_name in (
        "input_ids",
        "attention_mask",
        "completion_loss_mask",
        "sequence_positions",
    ):
        left = getattr(teacher, field_name)
        right = getattr(student, field_name)
        if left.shape != right.shape or not torch.equal(left, right):
            raise ValueError(f"teacher/student {field_name} are not aligned")


def masked_kl_distillation_loss(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    completion_loss_mask: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Return ``T**2 * mean_valid(KL(teacher || student))``.

    Logits have shape ``[batch, sequence, vocabulary]`` and the vocabulary is
    the final axis.  KL is summed over that axis, then masked over explicit
    completion target positions only.  The denominator is the number of true
    mask entries, not the attention-mask count or sequence length.  Teacher
    log-probabilities are passed to PyTorch's ``log_target=True`` KL API.
    """

    temperature = _require_finite_positive(temperature, "temperature")
    if not isinstance(teacher_logits, torch.Tensor) or not isinstance(student_logits, torch.Tensor):
        raise TypeError("teacher_logits and student_logits must be tensors")
    if teacher_logits.ndim != 3 or student_logits.ndim != 3:
        raise ValueError("teacher and student logits must have shape [batch, sequence, vocabulary]")
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("teacher and student logits must have identical shapes")
    if completion_loss_mask.shape != teacher_logits.shape[:2]:
        raise ValueError("completion_loss_mask must have shape [batch, sequence]")
    mask = _validate_batch_mask(completion_loss_mask, "completion_loss_mask")
    valid = int(mask.sum().item())
    if valid == 0:
        raise ValueError("completion_loss_mask has zero valid completion targets")
    if not torch.isfinite(teacher_logits).all() or not torch.isfinite(student_logits).all():
        raise ValueError("teacher and student logits must be finite")
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    per_token = F.kl_div(
        student_log_probs,
        teacher_log_probs,
        reduction="none",
        log_target=True,
    ).sum(dim=-1)
    weights = mask.to(device=per_token.device, dtype=per_token.dtype)
    return per_token.mul(weights).sum() / valid * (temperature**2)


def expected_bit_cost(
    probabilities: torch.Tensor,
    candidate_bits: tuple[int, ...] = THREE_WAY_CANDIDATE_BITS,
) -> torch.Tensor:
    """Return normalized expected bit-plane cost in the explicit candidate order.

    The normalized cost is ``(bit - 4) / 4``: 4-bit is zero, 6-bit is one
    half, and 8-bit is one.  The candidate tuple is always used to construct
    the cost vector, so vector length never assigns bit meanings.
    """

    candidate_bits = validate_candidate_bits(candidate_bits)
    probabilities = validate_probabilities(probabilities, candidate_bits)
    dtype = probabilities.dtype if probabilities.is_floating_point() else torch.get_default_dtype()
    costs = torch.tensor(
        [(bit - 4) / 4 for bit in candidate_bits],
        device=probabilities.device,
        dtype=dtype,
    )
    return (probabilities.to(dtype=dtype) * costs).sum(dim=-1)


def mean_expected_bit_cost(
    probabilities: torch.Tensor,
    candidate_bits: tuple[int, ...] = THREE_WAY_CANDIDATE_BITS,
) -> torch.Tensor:
    """Return the unweighted arithmetic mean cost over routing decisions."""

    costs = expected_bit_cost(probabilities, candidate_bits)
    if costs.numel() == 0:
        raise ValueError("probabilities must contain at least one routing decision")
    return costs.mean()


@dataclass(frozen=True, slots=True)
class RequestStateCostDiagnostics:
    expected_bit_cost: torch.Tensor
    expected_bit_width: torch.Tensor | None


def request_state_expected_bit_cost(
    state: Any, *, return_diagnostics: bool = False
) -> torch.Tensor | RequestStateCostDiagnostics:
    """Aggregate every stored attention and FFN probability exactly once.

    The stored probability clones remain connected to the router graph.  A
    request contributes one decision per attention and FFN layer, with equal
    weight regardless of unit type or layer.
    """

    candidate_bits = validate_candidate_bits(state.candidate_bits)
    layer_count = state.layer_count
    decisions: list[torch.Tensor] = []
    for name in ("attention_probabilities", "ffn_probabilities"):
        values = getattr(state, name, None)
        if not isinstance(values, list) or len(values) != layer_count:
            raise ValueError(f"{name} must contain exactly {layer_count} decisions")
        if any(value is None for value in values):
            raise ValueError(f"{name} is missing one or more routing decisions")
        for value in values:
            validate_probabilities(
                value,
                candidate_bits,
                context=f"{name} entries",
                require_vector=True,
            )
        decisions.extend(values)
    if len(decisions) != 2 * layer_count:
        raise ValueError("request state must contain one attention and one FFN decision per layer")
    stacked = torch.stack(decisions, dim=0)
    bit_cost = mean_expected_bit_cost(stacked, candidate_bits)
    if not return_diagnostics:
        return bit_cost
    expected_width = 4 + 4 * bit_cost if candidate_bits == THREE_WAY_CANDIDATE_BITS else None
    return RequestStateCostDiagnostics(
        expected_bit_cost=bit_cost,
        expected_bit_width=expected_width,
    )


def _validate_cost_weight(cost_weight: object) -> float:
    if isinstance(cost_weight, bool) or not isinstance(cost_weight, Real):
        raise TypeError("cost_weight must be a finite non-negative number")
    value = float(cost_weight)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"cost_weight must be a finite non-negative number; got {cost_weight}")
    return value


def cost_aware_distillation_loss(
    kd_loss: torch.Tensor,
    expected_cost: torch.Tensor,
    cost_weight: float = 0.0,
) -> torch.Tensor:
    """Compose unchanged KD with an optional normalized bit-cost objective."""

    if not isinstance(kd_loss, torch.Tensor) or kd_loss.ndim != 0:
        raise ValueError("kd_loss must be a scalar tensor")
    if not isinstance(expected_cost, torch.Tensor) or expected_cost.ndim != 0:
        raise ValueError("expected_cost must be a scalar tensor")
    if not torch.isfinite(kd_loss) or not torch.isfinite(expected_cost):
        raise ValueError("kd_loss and expected_cost must be finite")
    weight = _validate_cost_weight(cost_weight)
    if weight == 0.0:
        return kd_loss
    return kd_loss + weight * expected_cost


@dataclass(frozen=True, slots=True)
class FrozenParameterSnapshot:
    values: dict[str, torch.Tensor]
    requires_grad: dict[str, bool]

    def assert_unchanged(self, modules: Mapping[str, nn.Module]) -> None:
        current: dict[str, tuple[nn.Parameter, str]] = {}
        for label, module in modules.items():
            for name, parameter in module.named_parameters():
                current[f"{label}.{name}"] = (parameter, name)
        if set(current) != set(self.values):
            raise AssertionError("frozen parameter set changed")
        for key, (parameter, _) in current.items():
            if parameter.requires_grad or parameter.grad is not None:
                raise AssertionError(f"frozen parameter became trainable or has a gradient: {key}")
            if not torch.equal(parameter.detach().cpu(), self.values[key]):
                raise AssertionError(f"frozen parameter changed: {key}")


def snapshot_frozen_parameters(
    teacher: nn.Module, student: nn.Module, *, router_prefix: str = "routers."
) -> FrozenParameterSnapshot:
    values: dict[str, torch.Tensor] = {}
    requires_grad: dict[str, bool] = {}
    for name, parameter in teacher.named_parameters():
        values[f"teacher.{name}"] = parameter.detach().cpu().clone()
        requires_grad[f"teacher.{name}"] = parameter.requires_grad
    for name, parameter in student.named_parameters():
        if not name.startswith(router_prefix):
            values[f"student.{name}"] = parameter.detach().cpu().clone()
            requires_grad[f"student.{name}"] = parameter.requires_grad
    return FrozenParameterSnapshot(values=values, requires_grad=requires_grad)


def freeze_teacher_and_packed_student(
    teacher: nn.Module, student: nn.Module, *, router_prefix: str = "routers."
) -> FrozenParameterSnapshot:
    """Freeze teacher plus every S06 student parameter outside ``routers.``."""

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for name, parameter in student.named_parameters():
        if name.startswith(router_prefix):
            parameter.requires_grad_(True)
        else:
            parameter.requires_grad_(False)
            parameter.grad = None
    snapshot = snapshot_frozen_parameters(teacher, student, router_prefix=router_prefix)
    snapshot.assert_unchanged(
        {
            "teacher": teacher,
            "student": _FrozenStudentView(student, router_prefix),
        }
    )
    return snapshot


class _FrozenStudentView(nn.Module):
    """Non-registered view used only to expose the S06 non-router parameters."""

    def __init__(self, student: nn.Module, router_prefix: str) -> None:
        super().__init__()
        self._parameters_view = tuple(
            (name, parameter)
            for name, parameter in student.named_parameters()
            if not name.startswith(router_prefix)
        )

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[no-untyped-def]
        del prefix, recurse
        yield from self._parameters_view


@dataclass(frozen=True, slots=True)
class RouterOptimizerAudit:
    tensor_count: int
    scalar_count: int
    included_names: tuple[str, ...]
    included_name_prefixes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _router_items(
    model: nn.Module, router_prefixes: tuple[str, ...]
) -> list[tuple[str, nn.Parameter]]:
    items = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(router_prefixes)
    ]
    if not items:
        raise ValueError(f"no router parameters found under prefixes {router_prefixes}")
    return items


def build_router_optimizer(
    model: nn.Module,
    *,
    lr: float,
    optimizer_cls: type[torch.optim.Optimizer] = torch.optim.AdamW,
    router_prefixes: Sequence[str] = ("routers.",),
    **optimizer_kwargs: Any,
) -> tuple[torch.optim.Optimizer, RouterOptimizerAudit]:
    """Construct an optimizer from explicit router names, never a grad filter."""

    lr = _require_finite_positive(lr, "learning rate")
    prefixes = tuple(router_prefixes)
    if not prefixes or any(not prefix for prefix in prefixes):
        raise ValueError("router_prefixes must contain non-empty prefixes")
    items = _router_items(model, prefixes)
    optimizer = optimizer_cls([parameter for _, parameter in items], lr=lr, **optimizer_kwargs)
    audit = audit_router_optimizer(model, optimizer, router_prefixes=prefixes)
    return optimizer, audit


def audit_router_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    router_prefixes: Sequence[str] = ("routers.",),
) -> RouterOptimizerAudit:
    prefixes = tuple(router_prefixes)
    intended = _router_items(model, prefixes)
    intended_ids = {id(parameter): name for name, parameter in intended}
    included_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    if len(included_ids) != len(set(included_ids)):
        raise AssertionError("router optimizer contains duplicate parameter tensors")
    if set(included_ids) != set(intended_ids):
        missing = sorted(set(intended_ids) - set(included_ids))
        extra = sorted(set(included_ids) - set(intended_ids))
        raise AssertionError(f"router optimizer mismatch; missing={missing}, extra={extra}")
    names = tuple(sorted(intended_ids[parameter_id] for parameter_id in included_ids))
    return RouterOptimizerAudit(
        tensor_count=len(names),
        scalar_count=sum(model.get_parameter(name).numel() for name in names),
        included_names=names,
        included_name_prefixes=tuple(sorted(prefixes)),
    )


def hard_route(
    probabilities: torch.Tensor,
    *,
    candidate_bits: tuple[int, ...] = CANDIDATE_ORDERING,
) -> int | torch.Tensor:
    """Map ordinary first-maximum argmax indices through explicit candidates."""

    candidate_bits = validate_candidate_bits(candidate_bits)
    validate_probabilities(probabilities, candidate_bits, context="route probabilities")
    index = torch.argmax(probabilities, dim=-1)
    values = torch.tensor(candidate_bits, device=index.device, dtype=torch.long)
    bits = values[index]
    return int(bits.item()) if probabilities.ndim == 1 else bits


@dataclass(frozen=True, slots=True)
class RouteLogRecord:
    request_id: str
    layer: int
    unit_type: str
    p4: float
    p8: float
    hard_bit: int
    entropy: float
    soft_average_width: float | None = None
    p6: float | None = None
    candidate_bits: tuple[int, ...] = CANDIDATE_ORDERING

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id.strip():
            raise ValueError("route log request_id must be non-empty")
        if self.unit_type not in ("attention", "ffn"):
            raise ValueError("route log unit_type must be attention or ffn")
        candidate_bits = validate_candidate_bits(self.candidate_bits)
        if self.layer < 0 or self.hard_bit not in candidate_bits:
            raise ValueError("route log layer and hard_bit are invalid")
        if candidate_bits == THREE_WAY_CANDIDATE_BITS and self.p6 is None:
            raise ValueError("three-way route logs require p6")
        values = self.probability_values()
        if not all(math.isfinite(value) for value in (*values, self.entropy)):
            raise ValueError("route log values must be finite")
        if any(value < 0 for value in values) or not math.isclose(sum(values), 1.0, abs_tol=1e-5):
            raise ValueError("route log probabilities must be non-negative and sum to one")
        if candidate_bits == CANDIDATE_ORDERING and self.p6 is not None:
            raise ValueError("historical [4, 8] route logs cannot carry p6")
        if self.soft_average_width is not None and not math.isfinite(self.soft_average_width):
            raise ValueError("route log soft average width must be finite")

    def probability_values(self) -> tuple[float, ...]:
        values = {4: self.p4, 6: self.p6, 8: self.p8}
        return tuple(float(values[bit]) for bit in self.candidate_bits)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_bits"] = list(self.candidate_bits)
        return payload

    @classmethod
    def from_probabilities(
        cls,
        request_id: str,
        layer: int,
        unit_type: str,
        probabilities: torch.Tensor,
        *,
        log_base: float = 2.0,
        include_soft_average_width: bool = True,
        candidate_bits: tuple[int, ...] = CANDIDATE_ORDERING,
    ) -> RouteLogRecord:
        candidate_bits = validate_candidate_bits(candidate_bits)
        validate_probabilities(probabilities, candidate_bits, context="route log probabilities")
        log_base = _require_finite_positive(log_base, "entropy log base")
        bits = hard_route(probabilities, candidate_bits=candidate_bits)
        values = probabilities.detach().float().tolist()
        by_bit = dict(zip(candidate_bits, values, strict=True))
        entropy = -sum(value * math.log(value, log_base) for value in values if value > 0)
        return cls(
            request_id=request_id,
            layer=int(layer),
            unit_type=unit_type,
            p4=float(by_bit[4]),
            p8=float(by_bit[8]),
            p6=None if 6 not in by_bit else float(by_bit[6]),
            hard_bit=int(bits),
            entropy=float(entropy),
            soft_average_width=(sum(bit * by_bit[bit] for bit in candidate_bits))
            if include_soft_average_width
            else None,
            candidate_bits=candidate_bits,
        )


class RouteLogCollector:
    """Collect one compact record per attention/FFN unit and validate coverage."""

    def __init__(self, *, layer_count: int, request_ids: Iterable[str] | None = None) -> None:
        if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
            raise ValueError("layer_count must be a positive integer")
        self.layer_count = layer_count
        self.request_ids = None if request_ids is None else tuple(request_ids)
        self._records: list[RouteLogRecord] = []

    def add(self, record: RouteLogRecord) -> None:
        if record.unit_type not in ("attention", "ffn"):
            raise ValueError(f"unsupported route log unit type: {record.unit_type}")
        if not 0 <= record.layer < self.layer_count:
            raise ValueError(f"route log layer must be in [0, {self.layer_count})")
        self._records.append(record)

    @property
    def records(self) -> tuple[RouteLogRecord, ...]:
        return tuple(self._records)

    def finalize(self) -> tuple[RouteLogRecord, ...]:
        requests = self.request_ids or tuple(
            dict.fromkeys(record.request_id for record in self._records)
        )
        expected = {
            (request_id, layer, unit)
            for request_id in requests
            for layer in range(self.layer_count)
            for unit in ("attention", "ffn")
        }
        actual = [(record.request_id, record.layer, record.unit_type) for record in self._records]
        if len(actual) != len(set(actual)):
            raise AssertionError("route logs contain duplicate request/layer/unit records")
        actual_set = set(actual)
        if actual_set != expected:
            raise AssertionError(
                f"route log coverage mismatch; missing={sorted(expected - actual_set)}, unexpected={sorted(actual_set - expected)}"
            )
        return self.records


def route_records_from_request_state(
    request_id: str, state: Any, *, log_base: float = 2.0
) -> tuple[RouteLogRecord, ...]:
    records: list[RouteLogRecord] = []
    for layer, probabilities in enumerate(state.attention_probabilities):
        if probabilities is None:
            raise ValueError(f"missing attention probability for layer {layer}")
        records.append(
            RouteLogRecord.from_probabilities(
                request_id,
                layer,
                "attention",
                probabilities,
                log_base=log_base,
                candidate_bits=state.candidate_bits,
            )
        )
    for layer, probabilities in enumerate(state.ffn_probabilities):
        if probabilities is None:
            raise ValueError(f"missing FFN probability for layer {layer}")
        records.append(
            RouteLogRecord.from_probabilities(
                request_id,
                layer,
                "ffn",
                probabilities,
                log_base=log_base,
                candidate_bits=state.candidate_bits,
            )
        )
    return tuple(records)


def route_statistics(
    records: Iterable[RouteLogRecord],
    *,
    distillation_loss: float | None = None,
    entropy_log_base: float = 2.0,
) -> dict[str, object]:
    """Compute observational route summaries; no value here enters the loss."""

    log_base = _require_finite_positive(entropy_log_base, "entropy log base")
    values = tuple(records)
    if not values:
        raise ValueError("route statistics require at least one route record")
    if distillation_loss is not None and not math.isfinite(float(distillation_loss)):
        raise ValueError("distillation_loss must be finite")
    count = len(values)
    candidates = tuple(
        bit for bit in THREE_WAY_CANDIDATE_BITS if any(bit in record.candidate_bits for record in values)
    )
    hard_counts = {bit: sum(record.hard_bit == bit for record in values) for bit in candidates}
    by_layer: dict[str, dict[str, float]] = {}
    for layer in sorted({record.layer for record in values}):
        subset = [record for record in values if record.layer == layer]
        by_layer[str(layer)] = {
            str(bit): sum(record.hard_bit == bit for record in subset) / len(subset)
            for bit in candidates
        }
    by_unit: dict[str, dict[str, float]] = {}
    for unit in ("attention", "ffn"):
        subset = [record for record in values if record.unit_type == unit]
        if subset:
            by_unit[unit] = {
                str(bit): sum(record.hard_bit == bit for record in subset) / len(subset)
                for bit in candidates
            }
    grouped: dict[str, dict[tuple[int, str], int]] = defaultdict(dict)
    for record in values:
        grouped[record.request_id][(record.layer, record.unit_type)] = record.hard_bit
    prompt_routes = tuple(grouped.values())
    all_keys = set().union(*(routes.keys() for routes in prompt_routes))
    changed = sum(
        len({routes[key] for routes in prompt_routes if key in routes}) > 1 for key in all_keys
    )
    return {
        "distillation_loss": None if distillation_loss is None else float(distillation_loss),
        "entropy_log_base": log_base,
        "mean_entropy": sum(
            -sum(
                probability * math.log(probability, log_base)
                for probability in record.probability_values()
                if probability > 0
            )
            for record in values
        )
        / count,
        "mean_soft_average_width": (
            sum(
                record.soft_average_width
                for record in values
                if record.soft_average_width is not None
            )
            / sum(record.soft_average_width is not None for record in values)
            if any(record.soft_average_width is not None for record in values)
            else None
        ),
        "hard_fraction_4": hard_counts.get(4, 0) / count,
        "hard_fraction_6": hard_counts.get(6, 0) / count,
        "hard_fraction_8": hard_counts.get(8, 0) / count,
        "precision_distribution_by_layer": by_layer,
        "attention_vs_ffn_distribution": by_unit,
        "route_variation_across_prompts": {
            "prompt_count": len(prompt_routes),
            "unit_count": len(all_keys),
            "changed_unit_count": changed,
            "changed_fraction": changed / len(all_keys) if all_keys else 0.0,
        },
    }


@dataclass(frozen=True, slots=True)
class RouterCheckpointMetadata:
    model_repository: str
    model_revision: str
    quantized_checkpoint_id: str
    quantized_checkpoint_hash: str
    any_precision_revision: str
    router_architecture: Mapping[str, object]
    candidate_ordering: tuple[int, ...] = CANDIDATE_ORDERING
    training_step: int = 0
    training_step_metadata: Mapping[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_ordering"] = list(self.candidate_ordering)
        payload["router_architecture"] = dict(self.router_architecture)
        payload["training_step_metadata"] = dict(self.training_step_metadata)
        return payload

    def validate(self) -> None:
        for name in (
            "model_repository",
            "model_revision",
            "quantized_checkpoint_id",
            "quantized_checkpoint_hash",
            "any_precision_revision",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"checkpoint metadata {name} must be non-empty")
        validate_candidate_bits(tuple(self.candidate_ordering))
        if (
            isinstance(self.training_step, bool)
            or not isinstance(self.training_step, int)
            or self.training_step < 0
        ):
            raise ValueError("training_step must be a non-negative integer")


def _router_state_only(router: nn.Module) -> dict[str, torch.Tensor]:
    state = router.state_dict()
    forbidden = ("base", "packed", "teacher", "student", "lm_head", "embed_tokens")
    if any(any(part in key.lower() for part in forbidden) for key in state):
        raise ValueError("router checkpoint input appears to contain teacher or packed model state")
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def save_router_checkpoint(
    path: str | Path,
    router: nn.Module,
    metadata: RouterCheckpointMetadata,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """Save router state and metadata only; never teacher or packed weights."""

    metadata.validate()
    router_candidates = getattr(router, "candidate_bits", None)
    if router_candidates is not None and tuple(router_candidates) != tuple(
        metadata.candidate_ordering
    ):
        raise ValueError("router candidate_bits do not match checkpoint metadata")
    payload: dict[str, object] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": metadata.to_dict(),
        "router_state": _router_state_only(router),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, Path(path))


def load_router_checkpoint(
    path: str | Path,
    router: nn.Module,
    expected_metadata: RouterCheckpointMetadata,
    *,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, object]:
    expected_metadata.validate()
    router_candidates = getattr(router, "candidate_bits", None)
    if router_candidates is not None and tuple(router_candidates) != tuple(
        expected_metadata.candidate_ordering
    ):
        raise ValueError("router candidate_bits do not match checkpoint metadata")
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("unsupported router checkpoint format")
    if payload.get("metadata") != expected_metadata.to_dict():
        raise ValueError("router checkpoint metadata is incompatible")
    state = payload.get("router_state")
    if not isinstance(state, dict):
        raise TypeError("router checkpoint is missing router_state")
    router.load_state_dict(state)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload["metadata"]


def _extract_logits(output: Any) -> torch.Tensor:
    logits = output if isinstance(output, torch.Tensor) else getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError(
            "teacher/student execution must return logits with shape [batch, sequence, vocabulary]"
        )
    return logits


@dataclass(frozen=True, slots=True)
class DistillationStepResult:
    loss: float
    router_gradient_norm: float
    router_parameter_changed: bool


class RouterDistillationTrainer:
    """One-step S07-A trainer around a frozen teacher and frozen S06 base."""

    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        temperature: float,
        router_prefix: str = "routers.",
        request_state_factory: Callable[[str, int], Any] | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.optimizer = optimizer
        self.temperature = _require_finite_positive(temperature, "temperature")
        self.router_prefix = router_prefix
        self.request_state_factory = request_state_factory
        self.optimizer_audit = audit_router_optimizer(
            student, optimizer, router_prefixes=(router_prefix,)
        )
        self.frozen_snapshot = freeze_teacher_and_packed_student(
            teacher, student, router_prefix=router_prefix
        )

    def step(self, batch: DistillationBatch, *, trace: Any = None) -> DistillationStepResult:
        batch.validate()
        execution = batch.execution_inputs()
        execution.validate()
        validate_execution_alignment(execution, execution)
        self.optimizer.zero_grad(set_to_none=True)
        self.teacher.eval()
        self.student.train()
        model_kwargs = {
            "input_ids": execution.input_ids,
            "attention_mask": execution.attention_mask,
            "position_ids": execution.sequence_positions,
            "use_cache": False,
        }
        with torch.no_grad():
            teacher_logits = _extract_logits(self.teacher(**model_kwargs))
        request_state = (
            self.request_state_factory(
                batch.example_ids[0], int(batch.prompt_attention_mask[0].sum())
            )
            if self.request_state_factory is not None
            else None
        )
        if request_state is None:
            raise ValueError("S07 S06 execution requires request_state_factory")
        if execution.input_ids.shape[0] != 1:
            raise ValueError("the established S06 request execution supports batch size one")
        student_logits = _extract_logits(
            self.student(
                **model_kwargs,
                request_state=request_state,
                phase="prefill",
                prompt_attention_mask=batch.prompt_attention_mask,
                trace=trace,
            )
        )
        if teacher_logits.shape != student_logits.shape:
            raise ValueError("teacher/student logits are not sequence-aligned")
        loss = masked_kl_distillation_loss(
            teacher_logits,
            student_logits,
            batch.completion_loss_mask,
            temperature=self.temperature,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("distillation loss is non-finite")
        loss.backward()
        self.frozen_snapshot.assert_unchanged(
            {
                "teacher": self.teacher,
                "student": _FrozenStudentView(self.student, self.router_prefix),
            }
        )
        router_items = _router_items(self.student, (self.router_prefix,))
        gradients = [parameter.grad for _, parameter in router_items]
        if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("router gradient is missing or non-finite")
        grad_norm = math.sqrt(
            sum(float(gradient.detach().float().square().sum()) for gradient in gradients)
        )
        before = {name: parameter.detach().clone() for name, parameter in router_items}
        self.optimizer.step()
        self.frozen_snapshot.assert_unchanged(
            {
                "teacher": self.teacher,
                "student": _FrozenStudentView(self.student, self.router_prefix),
            }
        )
        changed = any(not torch.equal(before[name], parameter) for name, parameter in router_items)
        return DistillationStepResult(float(loss.detach()), grad_norm, changed)


__all__ = [
    "CANDIDATE_ORDERING",
    "CAUSAL_TARGET_IGNORE_INDEX",
    "DistillationBatch",
    "DistillationExample",
    "DistillationStepResult",
    "ExecutionInputs",
    "FrozenParameterSnapshot",
    "RequestStateCostDiagnostics",
    "RouteLogCollector",
    "RouteLogRecord",
    "RouterCheckpointMetadata",
    "RouterDistillationTrainer",
    "RouterOptimizerAudit",
    "TokenRange",
    "audit_router_optimizer",
    "build_router_optimizer",
    "causal_target_ids",
    "cost_aware_distillation_loss",
    "expected_bit_cost",
    "freeze_teacher_and_packed_student",
    "hard_route",
    "load_router_checkpoint",
    "masked_kl_distillation_loss",
    "mean_expected_bit_cost",
    "request_state_expected_bit_cost",
    "route_records_from_request_state",
    "route_statistics",
    "save_router_checkpoint",
    "snapshot_frozen_parameters",
    "validate_execution_alignment",
]
