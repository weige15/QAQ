import pytest
from torch import nn

from qaq.model.static import TARGET_PRECISIONS, set_static_precision


class AnyPrecisionLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.selected = None

    def set_precision(self, precision):
        self.selected = precision


class StaticFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.packed = AnyPrecisionLinear()


@pytest.mark.parametrize("precision", [4, 6, 8])
def test_static_precision_accepts_only_public_4_6_8_values(precision):
    model = StaticFixture()

    assert TARGET_PRECISIONS == (4, 6, 8)
    set_static_precision(model, precision)
    assert model.packed.selected == precision


@pytest.mark.parametrize("precision", [3, 5, 7, 9, 4.0, True, "6", None])
def test_static_precision_rejects_invalid_values(precision):
    with pytest.raises(ValueError, match="unsupported static precision"):
        set_static_precision(StaticFixture(), precision)
