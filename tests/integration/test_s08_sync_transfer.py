from __future__ import annotations

import pytest
import torch

from qaq.model.request_state import QaqRequestState
from qaq.quantization.backend import build_case, packed_output
from qaq.loading.loader import PackedLinearSource, SynchronousPackedPlaneLoader


def _loader(case, request_id: str):
    source = PackedLinearSource.from_module(case.linear, "fixture.projection")
    state = QaqRequestState(request_id, prompt_length=1, layer_count=1)
    return source, state, SynchronousPackedPlaneLoader(source, state, case.device)


def _bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


@pytest.mark.parametrize("precision", [4, 8])
def test_first_use_transfers_exact_real_packed_buffers_and_matches_resident(precision):
    case = build_case()
    source, state, loader = _loader(case, f"s08-first-{precision}")
    assert loader.retained_entry_count == 0

    on_demand = loader(case.inputs, precision=precision)
    resident = packed_output(case, precision)
    record = loader.records[-1]
    expected_names = {f"qweight[:{precision}]", f"lut{precision}"}
    expected_bytes = _bytes(source.qweight[:precision]) + _bytes(
        getattr(source, f"lut{precision}")
    )

    assert record.event == "first_use"
    assert record.source_device == "cpu"
    assert record.destination_device == str(case.device)
    assert {item["name"] for item in record.buffers} == expected_names
    assert record.transferred_bytes == expected_bytes
    assert record.transferred_bytes == sum(int(item["bytes"]) for item in record.buffers)
    assert all(item["dtype"] in {"torch.int32", "torch.float16"} for item in record.buffers)
    assert torch.isfinite(on_demand).all()
    assert torch.equal(on_demand, resident)
    state.end_request()


def test_repeated_use_reuses_request_retained_data_without_transfer():
    case = build_case()
    _, state, loader = _loader(case, "s08-reuse")
    first = loader(case.inputs, precision=4)
    second = loader(case.inputs, precision=4)

    assert torch.equal(first, second)
    assert len(loader.records) == 2
    assert loader.records[0].event == "first_use"
    assert loader.records[1].event == "reuse"
    assert loader.records[1].transferred_bytes == 0
    assert loader.records[1].buffers == ()
    state.end_request()


def test_precision_upgrade_transfers_only_missing_planes_and_lut():
    case = build_case()
    source, state, loader = _loader(case, "s08-upgrade")
    output4 = loader(case.inputs, precision=4)
    output8 = loader(case.inputs, precision=8)

    assert torch.equal(output4, packed_output(case, 4))
    assert torch.equal(output8, packed_output(case, 8))
    upgrade = loader.records[1]
    assert upgrade.event == "first_use"
    assert {item["name"] for item in upgrade.buffers} == {"qweight[4:8]", "lut8"}
    assert upgrade.transferred_bytes == _bytes(source.qweight[4:8]) + _bytes(source.lut8)
    assert loader.retained_plane_count == 8

    output4_again = loader(case.inputs, precision=4)
    assert torch.equal(output4_again, packed_output(case, 4))
    assert loader.records[-1].event == "reuse"
    assert loader.records[-1].transferred_bytes == 0
    state.end_request()
