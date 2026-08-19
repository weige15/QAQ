from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import torch
from torch import nn

from qaq.model.request_state import QaqRequestState
from qaq.router import s10h_executor as executor
from qaq.router.distillation import (
    DistillationBatch,
    DistillationExample,
    TokenRange,
    cost_aware_distillation_loss,
    masked_kl_distillation_loss,
    request_state_expected_bit_cost,
)
from qaq.router.network import S10_CANDIDATE_BITS, SoftPrecisionRouter
from qaq.router.s10h_executor import (
    ExecutionOutcome,
    ExecutorError,
    QwenRuntime,
    _ordered_example_ids,
    _validate_example_order,
    _validate_selected_artifact,
    execute_with_runtime,
    write_validated_result,
)
from scripts import run_s07b
from scripts import run_s10h as protocol

ROOT = Path(__file__).parents[2]


class _TinyModel(nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.routers = nn.ModuleDict(
            {
                f"{unit_type}_{layer}": SoftPrecisionRouter(
                    1, hidden_width=1, candidate_bits=S10_CANDIDATE_BITS
                )
                for unit_type in ("attention", "ffn")
                for layer in range(36)
            }
        )
        self.base = nn.Parameter(torch.tensor([1.0]), requires_grad=False)

    def route(self, layer: int, unit_type: str, feature: torch.Tensor) -> torch.Tensor:
        return self.routers[f"{unit_type}_{layer}"](feature)


class _TinyRuntime:
    """Injected test runtime; it is never selected by production dispatch."""

    enforce_frozen_router_scalar_count = False

    def __init__(self):
        self.train_examples = [self._example(f"train-{index}") for index in range(24)]
        self.validation_examples = [
            self._example(request_id, validation=True) for request_id in protocol.VALIDATION_IDS
        ]
        self.teacher = nn.Parameter(torch.tensor([0.25]), requires_grad=False)
        self.router_runtime_audit = {
            "source": "injected deterministic runtime",
            "expected_router_tensor_count": 288,
            "expected_router_scalar_count": 23630040,
            "actual_router_tensor_count": 288,
            "actual_router_scalar_count": 504,
        }
        self.loss_calls = {"kd": 0, "cost": 0}

    @staticmethod
    def _example(example_id: str, validation: bool = False) -> DistillationExample:
        input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        attention = torch.ones(4, dtype=torch.bool)
        completion = torch.tensor([False, True, True, False])
        target_ids = torch.tensor([2, 3, 4, -100], dtype=torch.long)
        return DistillationExample(
            example_id=example_id,
            tokenizer_revision=protocol.MODEL_REVISION,
            input_ids=input_ids,
            attention_mask=attention,
            completion_loss_mask=completion,
            target_ids=target_ids,
            prompt_token_range=TokenRange(0, 2),
            completion_token_range=TokenRange(2, 4),
        )

    def prepare(self, config, device):
        del config, device

    def dataset_evidence(self, config):
        data = config["protocol"]["dataset"]
        return {
            "repository": data["repository"],
            "config": data["config"],
            "train_split": data["train_split"],
            "validation_split": data["validation_split"],
            "tokenizer_revision": data["tokenizer_revision"],
            "revision": data["revision"],
            "train_example_count": 24,
            "validation_example_count": 12,
            "train_manifest": [
                {"example_id": f"train-{row}", "source_row": row, "source_offset": offset}
                for row, offset in zip(protocol.TRAIN_ROWS, protocol.TRAIN_OFFSETS, strict=True)
            ],
            "validation_manifest": [
                {"example_id": request_id, "source_row": row, "source_offset": offset}
                for request_id, row, offset in zip(
                    protocol.VALIDATION_IDS,
                    protocol.VALIDATION_ROWS,
                    protocol.VALIDATION_OFFSETS,
                    strict=True,
                )
            ],
        }

    def identity_evidence(self):
        return {
            "model_repository": "Qwen/Qwen3-4B",
            "model_revision": protocol.MODEL_REVISION,
            "tokenizer_revision": protocol.MODEL_REVISION,
            "dataset_revision": protocol.DATASET_REVISION,
            "any_precision_revision": protocol.ANY_PRECISION_REVISION,
            "packed_artifact": protocol.PACKED_ARTIFACT,
            "manifest_sha256": protocol.LOCKED_MANIFEST_SHA256,
            "packed_artifact_pytorch_model_sha256": protocol.ARTIFACT_SHA256,
            "historical_s07_checkpoint_used": False,
            "historical_s07_checkpoint_sha256": protocol.HISTORICAL_S07_CHECKPOINT_SHA256,
        }

    def inherited_regressions_audit(self):
        return {
            "status": "passed",
            "test_selection": "S10-D/S10-E/S10-F predecessor regression selection",
            "passed": True,
        }

    def prohibited_work_audit(self):
        return {
            "forbidden_actions_observed": [],
            "forbidden_measurements_observed": [],
            "passed": True,
        }

    def build_seed_model(self, seed, device):
        del device
        return _TinyModel(seed)

    def router_state(self, model):
        return {
            name: value.detach().cpu().clone() for name, value in model.routers.state_dict().items()
        }

    def restore_router_state(self, model, state):
        model.routers.load_state_dict(copy.deepcopy(state), strict=True)
        for parameter in model.routers.parameters():
            parameter.grad = None

    @staticmethod
    def _hash_tensor(value):
        return hashlib.sha256(
            value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        ).hexdigest()

    def frozen_snapshot(self, model):
        return {
            "teacher_hash": self._hash_tensor(self.teacher),
            "base_hash": self._hash_tensor(model.base),
            "teacher_requires_grad_false": self.teacher.requires_grad is False,
            "base_requires_grad_false": model.base.requires_grad is False,
            "teacher_gradients_absent": self.teacher.grad is None,
            "base_gradients_absent": model.base.grad is None,
        }

    def frozen_audit(self, model, before):
        after = self.frozen_snapshot(model)
        return {"passed": before == after, "before": before, "after": after}

    def _routing_state(self, model, example):
        state = QaqRequestState(
            example.example_id,
            2,
            layer_count=36,
            feature_dim=1,
            candidate_bits=S10_CANDIDATE_BITS,
        )
        values = []
        offset = 0.01 if example.example_id.startswith("validation") else 0.02
        for unit_type in ("attention", "ffn"):
            for layer in range(36):
                feature = torch.tensor([1.0 + offset + layer / 1000])
                state.store_feature(unit_type, layer, feature)
                probability = model.route(layer, unit_type, feature)
                state.store_probability(unit_type, layer, probability)
                values.append(probability)
        return state, torch.stack(values).mean(dim=0)

    def train_step(self, model, example, optimizer, lambda_bit, step, device):
        del device
        batch = DistillationBatch.from_examples([example])
        optimizer.zero_grad(set_to_none=True)
        state, mean_probability = self._routing_state(model, example)
        student_values = torch.stack(
            (mean_probability[0], mean_probability[1], mean_probability[2], mean_probability.sum())
        )
        student_logits = student_values.view(1, 1, 4).expand(1, 4, 4)
        teacher_logits = torch.zeros_like(student_logits)
        teacher_logits[..., 0] = 1.0
        kd = masked_kl_distillation_loss(
            teacher_logits,
            student_logits,
            batch.completion_loss_mask,
            temperature=2.0,
        )
        self.loss_calls["kd"] += 1
        bit_cost = request_state_expected_bit_cost(state)
        self.loss_calls["cost"] += 1
        total = cost_aware_distillation_loss(kd, bit_cost, lambda_bit)
        router_parameters = list(model.routers.parameters())
        kd_gradients = torch.autograd.grad(kd, router_parameters, retain_graph=True)
        cost_gradients = torch.autograd.grad(bit_cost, router_parameters, retain_graph=True)
        kd_norm = float(torch.sqrt(sum(value.square().sum() for value in kd_gradients)).item())
        cost_norm = float(torch.sqrt(sum(value.square().sum() for value in cost_gradients)).item())
        total.backward()
        gradients = [parameter.grad for parameter in router_parameters]
        optimizer.step()
        finite = all(torch.isfinite(value).all() for value in (kd, bit_cost, total)) and all(
            gradient is not None and torch.isfinite(gradient).all() for gradient in gradients
        )
        nonzero = any(
            torch.count_nonzero(gradient).item() for gradient in gradients if gradient is not None
        )
        return {
            "finite_loss": bool(finite),
            "finite_kd_loss": bool(torch.isfinite(kd).item()),
            "finite_bit_cost": bool(torch.isfinite(bit_cost).item()),
            "finite_weighted_cost": True,
            "finite_total_loss": bool(torch.isfinite(total).item()),
            "finite_gradient": bool(finite),
            "router_gradients_present": all(gradient is not None for gradient in gradients),
            "router_gradients_nonzero": bool(nonzero),
            "router_gradient_norm": float(
                torch.sqrt(sum(gradient.square().sum() for gradient in gradients)).item()
            ),
            "initial_kd_gradient_norm": kd_norm,
            "initial_bit_cost_gradient_norm": cost_norm,
            "lambda_weighted_gradient_ratio": lambda_bit * cost_norm / kd_norm,
            "kd_loss": float(kd.item()),
            "expected_bit_cost": float(bit_cost.item()),
            "weighted_cost": float(lambda_bit * bit_cost.item()),
            "total_loss": float(total.item()),
        }

    def validate(self, model, mode, device):
        del device
        records = []
        maps = {}
        per_example = []
        for example in self.validation_examples:
            state, mean_probability = self._routing_state(model, example)
            if mode == "hard":
                for unit_type in ("attention", "ffn"):
                    for layer in range(36):
                        probability = getattr(state, f"{unit_type}_probabilities")[layer]
                        state.store_route(
                            unit_type,
                            layer,
                            int(torch.tensor(protocol.CANDIDATE_BITS)[torch.argmax(probability)]),
                        )
            student_values = torch.stack(
                (
                    mean_probability[0],
                    mean_probability[1],
                    mean_probability[2],
                    mean_probability.sum(),
                )
            )
            logits = student_values.view(1, 1, 4).expand(1, 4, 4)
            teacher_logits = torch.zeros_like(logits)
            teacher_logits[..., 0] = 1.0
            kd = masked_kl_distillation_loss(
                teacher_logits,
                logits,
                example.completion_loss_mask.unsqueeze(0),
                temperature=2.0,
            )
            from qaq.router.distillation import route_records_from_request_state

            route_records = route_records_from_request_state(example.example_id, state)
            route = sorted(route_records, key=lambda item: (item.layer, item.unit_type))
            maps[example.example_id] = [record.hard_bit for record in route]
            records.extend(route_records)
            per_example.append(
                {
                    "kd": float(kd.item()),
                    "mean_error": float((logits - teacher_logits).abs().mean().item()),
                    "max_error": float((logits - teacher_logits).abs().max().item()),
                }
            )
        values = records
        p4 = sum(record.p4 for record in values) / len(values)
        p6 = sum(record.p6 for record in values) / len(values)
        p8 = sum(record.p8 for record in values) / len(values)
        hard_width, fractions, unique = protocol._route_stats(maps)
        variation_count = sum(
            len({maps[request_id][index] for request_id in maps}) > 1 for index in range(72)
        )
        result = {
            "validation_kd": sum(item["kd"] for item in per_example) / len(per_example),
            "mean_absolute_logit_error": sum(item["mean_error"] for item in per_example)
            / len(per_example),
            "maximum_absolute_logit_error": max(item["max_error"] for item in per_example),
            "finite_outputs": True,
            "route_variation": {
                "prompt_count": 12,
                "unit_count": 72,
                "changed_unit_count": variation_count,
                "changed_fraction": variation_count / 72,
            },
            "distinct_hard_route_map_count": unique,
            "hard_validation_route_maps": maps,
            "hard_validation_mean_selected_bit_width": hard_width,
            "hard_validation_fraction_4": fractions["4"],
            "hard_validation_fraction_6": fractions["6"],
            "hard_validation_fraction_8": fractions["8"],
            "collapse_audit": {
                "classification": "OTHER",
                "invalid_or_degenerate": False,
                "passed": True,
            },
        }
        if mode == "soft":
            result.update(
                {
                    "mean_expected_bit_width": 4 * p4 + 6 * p6 + 8 * p8,
                    "mean_p4": p4,
                    "mean_p6": p6,
                    "mean_p8": p8,
                    "mean_entropy": sum(record.entropy for record in values) / len(values),
                }
            )
        return result

    @staticmethod
    def close_model(model):
        del model


def _result_fixture() -> dict[str, object]:
    return protocol.synthetic_structural_fixture()


def test_injected_runtime_executes_all_nine_trials_and_writes_only_temp(tmp_path):
    runtime = _TinyRuntime()
    output = tmp_path / "h2-test-result.json"
    config = protocol._load_frozen_config()
    outcome = execute_with_runtime(
        runtime,
        config=config,
        device="cpu",
        output=output,
    )
    assert isinstance(outcome, ExecutionOutcome)
    assert outcome.classification in {"CONTINUE", "REFINE"}
    assert outcome.written is True
    assert output.is_file()
    assert len(outcome.result["trials"]) == 9
    assert [(trial["seed"], trial["lambda_bit"]) for trial in outcome.result["trials"]] == list(
        protocol.TRIAL_PAIRS
    )
    assert all(len(trial["training_history"]) == 24 for trial in outcome.result["trials"])
    assert runtime.loss_calls == {"kd": 216, "cost": 216}
    assert outcome.validation["errors"] == []


def test_executor_rejects_canonical_output_without_runtime_dispatch(tmp_path):
    runtime = _TinyRuntime()
    outcome = execute_with_runtime(
        runtime,
        config=protocol._load_frozen_config(),
        device="cuda:0",
        output=protocol.RESULT_PATH,
    )
    assert outcome.classification == "PAUSE"
    assert outcome.written is False


def test_atomic_writer_validates_before_write_and_refuses_overwrite(tmp_path):
    output = tmp_path / "result.json"
    fixture = _result_fixture()
    write_validated_result(fixture, output)
    original = output.read_bytes()
    with pytest.raises(protocol.CanonicalResultExists):
        write_validated_result(fixture, output)
    assert output.read_bytes() == original
    broken = copy.deepcopy(fixture)
    broken["trials"][0]["finite_loss_audit"] = False
    with pytest.raises(ExecutorError):
        write_validated_result(broken, tmp_path / "broken.json")
    assert not list(tmp_path.glob(".broken.json.*.tmp"))


def test_runtime_audit_failure_does_not_leave_output(tmp_path):
    runtime = _TinyRuntime()
    original = runtime.train_step

    def nonfinite(*args, **kwargs):
        result = original(*args, **kwargs)
        result["finite_gradient"] = False
        return result

    runtime.train_step = nonfinite
    output = tmp_path / "failed.json"
    outcome = execute_with_runtime(
        runtime,
        config=protocol._load_frozen_config(),
        device="cpu",
        output=output,
    )
    assert outcome.classification == "REVISE"
    assert not output.exists()
    assert not list(tmp_path.glob(".failed.json.*.tmp"))


class _SelectionTokenizer:
    def __call__(self, text, *, add_special_tokens):
        assert text
        assert add_special_tokens is False
        return {"input_ids": [11, 12, 13, 14]}

    def decode(self, token_ids, *, skip_special_tokens):
        assert skip_special_tokens is False
        return " ".join(map(str, token_ids))


def _selected_examples(split):
    return run_s07b._select_examples(
        [{"text": "first row"}, {"text": "second row"}],
        _SelectionTokenizer(),
        [0, 1],
        split=split,
        config={
            "dataset": {"sequence_length": 4, "prompt_tokens": 2, "completion_tokens": 2}
        },
        torch=torch,
    )[0]


def test_selected_examples_accept_exact_train_and_validation_order_without_subscription():
    train = _selected_examples("train")
    validation = _selected_examples("validation")
    assert all(isinstance(example, DistillationExample) for example in train + validation)
    _validate_example_order(train, ["train-0", "train-1"], split="train")
    _validate_example_order(
        validation, ["validation-0", "validation-1"], split="validation"
    )


@pytest.mark.parametrize("split", ["train", "validation"])
def test_selected_examples_reject_reordered_frozen_ids(split):
    examples = _selected_examples(split)
    with pytest.raises(ExecutorError, match="order") as error:
        _validate_example_order(
            list(reversed(examples)),
            [f"{split}-0", f"{split}-1"],
            split=split,
        )
    assert error.value.outcome == "REVISE"


@pytest.mark.parametrize(
    ("example", "message"),
    [
        (object(), "no example_id"),
        (type("EmptyId", (), {"example_id": ""})(), "invalid example_id"),
        (type("NonStringId", (), {"example_id": 7})(), "invalid example_id"),
        ({"example_id": "train-0"}, "dictionary substitute"),
    ],
)
def test_ordered_example_ids_reject_invalid_attribute_contract(example, message):
    with pytest.raises(ExecutorError, match=message) as error:
        _ordered_example_ids([example], split="train")
    assert error.value.outcome == "REVISE"


def test_qwen_prepare_retains_real_selection_order_and_distillation_objects(monkeypatch):
    from qaq.evaluation import quality
    from scripts import run_s10d

    events = []
    converted = []

    class FakeTeacher(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor([1.0]))

        def cpu(self):
            events.append("teacher.cpu")
            return self

    def load_dataset(_repository, _config, *, split, **_kwargs):
        events.append(f"dataset:{split}")
        return [{"text": f"one qualifying {split} row"}]

    def device_example(example, device, torch_module):
        assert isinstance(example, DistillationExample)
        assert device == "cpu"
        assert torch_module is torch
        converted.append(example)
        events.append(f"device:{example.example_id}")
        return example

    def load_teacher(_snapshot, device):
        assert device == "cpu"
        events.append("teacher:double")
        return FakeTeacher()

    def precompute(teacher, examples, torch_module):
        assert isinstance(teacher, FakeTeacher)
        assert torch_module is torch
        assert [example.example_id for example in examples] == ["train-0", "validation-0"]
        events.append("teacher_logits:double")
        return {example.example_id: torch.zeros(1) for example in examples}

    data = {
        "repository": "fake/wikitext",
        "config": "fake-config",
        "train_split": "train",
        "validation_split": "validation",
        "sequence_length": 4,
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "train_offsets": [0],
        "validation_offsets": [0],
        "train_example_ids": ["train-0"],
        "validation_example_ids": ["validation-0"],
    }
    config = {"protocol": {"dataset": data}}
    monkeypatch.setattr(
        executor,
        "_validate_selected_artifact",
        lambda *_args: events.append("artifact:double"),
    )
    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *_args, **_kwargs: events.append("tokenizer:double") or _SelectionTokenizer(),
    )
    monkeypatch.setattr(run_s07b, "_device_example", device_example)
    monkeypatch.setattr(run_s07b, "_precompute_teacher_logits", precompute)
    monkeypatch.setattr(quality, "load_full_precision_model", load_teacher)
    monkeypatch.setattr(
        run_s10d,
        "install_memory_saving_packed_backward",
        lambda: events.append("packed_backward:double"),
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("cuda_cache:double"))

    runtime = QwenRuntime(config, preflight={"identities": {}})
    try:
        runtime.prepare(config, "cpu")
        assert [example.example_id for example in converted] == ["train-0", "validation-0"]
        assert runtime.train_examples == [converted[0]]
        assert runtime.validation_examples == [converted[1]]
        assert all(isinstance(example, DistillationExample) for example in converted)
        assert all(not parameter.requires_grad for parameter in runtime.teacher.parameters())
        assert events == [
            "artifact:double",
            "tokenizer:double",
            "dataset:train",
            "dataset:validation",
            "device:train-0",
            "device:validation-0",
            "teacher:double",
            "teacher_logits:double",
            "teacher.cpu",
            "cuda_cache:double",
            "packed_backward:double",
        ]
    finally:
        runtime.teacher = None
        runtime.teacher_targets.clear()
        runtime.train_examples.clear()
        runtime.validation_examples.clear()
    assert runtime.teacher is None
    assert runtime.teacher_targets == {}
    assert runtime.train_examples == []
    assert runtime.validation_examples == []


def test_selected_artifact_identity_is_verified_before_loading(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    checkpoint = artifact / "pytorch_model.bin"
    checkpoint.write_bytes(b"frozen checkpoint")
    config = artifact / "config.json"
    config.write_bytes(b"frozen config")
    manifest = {
        "artifact": {
            "artifact_file_list": [
                {
                    "path": "pytorch_model.bin",
                    "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                },
                {
                    "path": "config.json",
                    "sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
                },
            ]
        }
    }
    _validate_selected_artifact(artifact, manifest)
    checkpoint.write_bytes(b"tampered checkpoint")
    with pytest.raises(ExecutorError, match="identity changed"):
        _validate_selected_artifact(artifact, manifest)
