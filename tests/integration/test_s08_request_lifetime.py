from __future__ import annotations

import pytest

from qaq.model.request_state import QaqRequestState
from qaq.quantization.backend import build_case
from qaq.loading.loader import PackedLinearSource, SynchronousPackedPlaneLoader


def test_request_end_releases_gpu_references_and_use_after_end_fails():
    case = build_case()
    source = PackedLinearSource.from_module(case.linear, "fixture.projection")
    state = QaqRequestState("s08-lifetime", prompt_length=1, layer_count=1)
    loader = SynchronousPackedPlaneLoader(source, state, case.device)
    loader(case.inputs, precision=4)
    assert loader.retained_entry_count == 1
    assert loader.retained_gpu_buffer_count == 2

    state.end_request()
    assert state.ended
    assert loader.retained_entry_count == 0
    assert loader.retained_gpu_buffer_count == 0
    assert loader.retained_plane_count == 0
    with pytest.raises(RuntimeError, match="after request cleanup"):
        loader(case.inputs, precision=4)


def test_same_textual_request_id_does_not_share_retained_planes():
    case = build_case()
    source = PackedLinearSource.from_module(case.linear, "fixture.projection")
    state_a = QaqRequestState("same-request-id", prompt_length=1, layer_count=1)
    state_b = QaqRequestState("same-request-id", prompt_length=1, layer_count=1)
    loader_a = SynchronousPackedPlaneLoader(source, state_a, case.device)
    loader_b = SynchronousPackedPlaneLoader(source, state_b, case.device)

    loader_a(case.inputs, precision=4)
    loader_b(case.inputs, precision=4)
    record_a = loader_a.records[0]
    record_b = loader_b.records[0]

    assert record_a.event == "first_use"
    assert record_b.event == "first_use"
    assert record_a.transferred_bytes > 0
    assert record_b.transferred_bytes == record_a.transferred_bytes
    assert record_a.request_id == record_b.request_id
    assert record_a.request_state_identity != record_b.request_state_identity
    assert loader_a.retained_entry_count == loader_b.retained_entry_count == 1

    state_a.end_request()
    assert loader_a.retained_entry_count == 0
    assert loader_b.retained_entry_count == 1
    state_b.end_request()
