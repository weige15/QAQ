from __future__ import annotations

import pytest
import torch

from qaq.model.request_state import QaqRequestState
from qaq.quantization.backend import build_case
from qaq.loading.loader import PackedLinearSource, SynchronousPackedPlaneLoader


def test_loader_rejects_gpu_authoritative_source_and_unsupported_precision():
    case = build_case()
    with pytest.raises(ValueError, match="CPU-authoritative"):
        PackedLinearSource(
            module_id="invalid",
            qweight=case.linear.qweight,
            lut4=case.linear.lut4,
            lut8=case.linear.lut8,
        )

    source = PackedLinearSource.from_module(case.linear, "fixture.projection")
    state = QaqRequestState("s08-invalid", prompt_length=1, layer_count=1)
    loader = SynchronousPackedPlaneLoader(source, state, case.device)
    with pytest.raises(ValueError, match="precisions 4 and 8"):
        loader(case.inputs, precision=6)
    assert loader.records == ()
    state.end_request()


def test_loader_source_is_real_cpu_packed_storage_before_first_use():
    case = build_case()
    source = PackedLinearSource.from_module(case.linear, "fixture.projection")
    assert source.qweight.device.type == "cpu"
    assert source.qweight.dtype == torch.int32
    assert tuple(source.qweight.shape) == (8, 64, 32)
    assert source.qweight.is_contiguous()
    assert source.lut4.device.type == "cpu"
    assert source.lut8.device.type == "cpu"

    state = QaqRequestState("s08-cpu-authority", prompt_length=1, layer_count=1)
    loader = SynchronousPackedPlaneLoader(source, state, case.device)
    assert loader.retained_entry_count == 0
    assert loader.retained_gpu_buffer_count == 0
    state.end_request()
