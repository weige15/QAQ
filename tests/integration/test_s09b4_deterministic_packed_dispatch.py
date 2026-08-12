from __future__ import annotations

import hashlib
import os

import pytest
import torch
from torch import nn

from qaq.model.request_state import QaqRequestState
from qaq.s08_loader import (
    PackedLinearSource,
    SynchronousPackedPlaneLoader,
    execute_packed_linear,
    pinned_backend,
    uses_atomic_k_split,
)

DEVICE = torch.device(os.environ.get("QAQ_S09B4_DEVICE", "cuda:3"))


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="S09-B4 requires the pinned CUDA backend"
)


def _digest(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _affected_source() -> PackedLinearSource:
    generator = torch.Generator(device="cpu").manual_seed(1729)
    qweight = torch.randint(
        -(2**31), 2**31 - 1, (8, 2560, 9728 // 32), generator=generator, dtype=torch.int64
    ).to(torch.int32)
    lut4 = torch.randn((2560, 16), generator=generator, dtype=torch.float32).to(torch.float16)
    lut8 = torch.randn((2560, 256), generator=generator, dtype=torch.float32).to(torch.float16)
    return PackedLinearSource("model.layers.0.mlp.down_proj", qweight, lut4, lut8)


def test_pinned_atomic_k_split_family_is_exact_and_unaffected_path_is_preserved():
    source = _affected_source()
    inputs = torch.randn((1, 1, 9728), device=DEVICE, dtype=torch.float16)
    qweight = source.qweight.to(DEVICE)
    lut = source.lut8.to(DEVICE)
    dequant_kbit, matmul_kbit = pinned_backend()

    assert uses_atomic_k_split(inputs, qweight, 8)
    affected = [
        execute_packed_linear(
            inputs,
            qweight,
            lut,
            8,
            dequant_kbit=dequant_kbit,
            matmul_kbit=matmul_kbit,
        )
        for _ in range(5)
    ]
    assert all(torch.isfinite(value).all().item() for value in affected)
    assert len({_digest(value) for value in affected}) == 1
    assert all(torch.equal(affected[0], value) for value in affected[1:])

    small_inputs = torch.randn((1, 1, 1024), device=DEVICE, dtype=torch.float16)
    small_qweight = source.qweight[:, :, : 1024 // 32]
    small_lut = source.lut8
    assert not uses_atomic_k_split(small_inputs, small_qweight.to(DEVICE), 8)
    expected = matmul_kbit(small_inputs, small_qweight.to(DEVICE), small_lut.to(DEVICE), 8)
    actual = execute_packed_linear(
        small_inputs,
        small_qweight.to(DEVICE),
        small_lut.to(DEVICE),
        8,
        dequant_kbit=dequant_kbit,
        matmul_kbit=matmul_kbit,
    )
    assert torch.equal(actual, expected)

    wide_inputs = torch.randn((9, 1, 1024), device=DEVICE, dtype=torch.float16)
    wide_expected = dequant_kbit(small_qweight.to(DEVICE), small_lut.to(DEVICE), 8)
    wide_expected = torch.matmul(wide_inputs, wide_expected.transpose(0, 1))
    wide_actual = execute_packed_linear(
        wide_inputs,
        small_qweight.to(DEVICE),
        small_lut.to(DEVICE),
        8,
        dequant_kbit=dequant_kbit,
        matmul_kbit=matmul_kbit,
    )
    assert torch.equal(wide_actual, wide_expected)


def test_resident_and_request_owned_sources_share_fallback_and_leave_no_dense_state():
    source = _affected_source()
    inputs = torch.randn((1, 1, 9728), device=DEVICE, dtype=torch.float16)
    dequant_kbit, matmul_kbit = pinned_backend()
    resident_qweight = source.qweight.to(DEVICE)
    resident_lut = source.lut8.to(DEVICE)
    holder = nn.Module()
    holder.register_buffer("qweight", resident_qweight)
    holder.register_buffer("lut8", resident_lut)
    registered_before = tuple(name for name, _ in holder.named_parameters()) + tuple(
        name for name, _ in holder.named_buffers()
    )

    resident = [
        execute_packed_linear(
            inputs,
            holder.qweight,
            holder.lut8,
            8,
            dequant_kbit=dequant_kbit,
            matmul_kbit=matmul_kbit,
        )
        for _ in range(5)
    ]
    state = QaqRequestState("s09b4-request", prompt_length=1, layer_count=1)
    loader = SynchronousPackedPlaneLoader(source, state, DEVICE)
    on_demand = [loader(inputs, precision=8) for _ in range(5)]

    assert all(torch.equal(resident[0], value) for value in resident[1:])
    assert all(torch.equal(on_demand[0], value) for value in on_demand[1:])
    assert torch.equal(resident[0], on_demand[0])
    assert len({_digest(value) for value in resident}) == 1
    assert len({_digest(value) for value in on_demand}) == 1
    assert all(torch.isfinite(value).all().item() for value in resident + on_demand)
    assert tuple(name for name, _ in holder.named_parameters()) + tuple(
        name for name, _ in holder.named_buffers()
    ) == registered_before
    assert not hasattr(holder, "weight")
    assert not hasattr(holder, "dequantized_weight")
    state.end_request()
    assert loader.retained_entry_count == 0
    assert loader.retained_gpu_buffer_count == 0
