import inspect

from qaq.quantization.backend import load_pinned_backend, require_cuda


def test_pinned_backend_import_and_constructor_contract():
    require_cuda()
    any_precision_linear, dequant_kbit, matmul_kbit = load_pinned_backend()
    parameters = inspect.signature(any_precision_linear).parameters
    assert ["in_features", "out_features", "supported_bits"] == list(parameters)[:3]
    assert {"bias", "precisions", "device", "dtype"}.issubset(parameters)
    assert callable(dequant_kbit)
    assert callable(matmul_kbit)
