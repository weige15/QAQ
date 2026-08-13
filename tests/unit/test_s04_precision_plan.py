import json

import pytest

from qaq.model.manual import LAYER_COUNT, PrecisionPlan


def test_precision_plan_is_immutable_and_round_trips_deterministically():
    plan = PrecisionPlan(
        attention_bits=tuple(4 if layer % 2 == 0 else 8 for layer in range(LAYER_COUNT)),
        ffn_bits=tuple(8 if layer % 2 == 0 else 4 for layer in range(LAYER_COUNT)),
    )

    assert plan.attention_bits[0] == 4
    assert plan.ffn_bits[-1] == 4
    with pytest.raises((AttributeError, TypeError)):
        plan.attention_bits = (8,) * LAYER_COUNT

    serialized = plan.to_json()
    assert serialized == plan.to_json()
    assert json.loads(serialized) == plan.to_dict()
    assert PrecisionPlan.from_json(serialized) == plan


@pytest.mark.parametrize(
    ("field", "value", "exception", "message"),
    [
        ("attention_bits", [4] * LAYER_COUNT, TypeError, "tuple"),
        ("ffn_bits", (4,) * (LAYER_COUNT - 1), ValueError, "exactly 36"),
        ("attention_bits", (4,) * LAYER_COUNT + (8,), ValueError, "exactly 36"),
        ("ffn_bits", (4,) * 35 + (6,), ValueError, "must be 4 or 8"),
        ("attention_bits", (4,) * 35 + (True,), TypeError, "integer"),
    ],
)
def test_precision_plan_rejects_invalid_route_fields(field, value, exception, message):
    fields = {
        "attention_bits": (4,) * LAYER_COUNT,
        "ffn_bits": (4,) * LAYER_COUNT,
    }
    fields[field] = value

    with pytest.raises(exception, match=message):
        PrecisionPlan(**fields)


def test_precision_plan_deserialization_rejects_missing_and_extra_routes():
    payload = {
        "attention_bits": [4] * LAYER_COUNT,
        "ffn_bits": [4] * LAYER_COUNT,
    }

    missing = dict(payload)
    del missing["ffn_bits"]
    with pytest.raises(ValueError, match="missing"):
        PrecisionPlan.from_dict(missing)

    extra = dict(payload)
    extra["other_bits"] = [4] * LAYER_COUNT
    with pytest.raises(ValueError, match="extra"):
        PrecisionPlan.from_dict(extra)
