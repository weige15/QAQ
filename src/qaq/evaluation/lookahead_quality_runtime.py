"""Lazy production runtime shared by the frozen S11-B and S11-C quality checks.

Importing this module does not itself import an ML runtime.  ``prepare`` is the
only path that imports Torch/Transformers and it is reachable only after the
standard-library dispatcher validates one exact mode, CUDA device, and output.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from qaq.evaluation import lookahead_quality_runner as contract


class ProductionRuntime:
    """Inference-only resident Qwen3 teacher/student implementation."""

    evidence_label = "production pilot evidence"

    def __init__(self) -> None:
        self.torch: Any | None = None
        self.student: Any | None = None
        self.device = ""
        self.config: Mapping[str, Any] | None = None
        self.mode: Mapping[str, Any] | None = None
        self.teacher_logits: dict[tuple[int, str], Any] = {}
        self._hardware: dict[str, Any] = {}
        self._identities: dict[str, Any] = {}
        self._teacher_audit: dict[str, Any] | None = None
        self._student_before: dict[str, list[dict[str, Any]]] = {}
        self._freeze_result: dict[str, Any] | None = None

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _require_resource(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"PAUSE: {message}")

    @staticmethod
    def _require_identity(condition: bool, message: str) -> None:
        if not condition:
            raise RuntimeError(f"INVALID_EVIDENCE: {message}")

    def _snapshot_path(self, revision: str) -> Path:
        default = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots" / revision
        path = Path(os.environ.get("QAQ_MODEL_SNAPSHOT", str(default))).expanduser().resolve()
        self._require_resource(path.is_dir(), f"pinned Qwen3 snapshot unavailable: {path}")
        self._require_identity(path.name == revision, "model snapshot revision path drifted")
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            self._require_resource(
                (path / name).is_file(), f"snapshot file unavailable: {path / name}"
            )
        self._snapshot_architecture(path)
        return path

    def _snapshot_architecture(self, path: Path) -> None:
        try:
            payload = json.loads((path / "config.json").read_text())
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"INVALID_EVIDENCE: model snapshot config is malformed: {exc}"
            ) from exc
        self._require_identity(
            isinstance(payload, dict)
            and payload.get("model_type") == "qwen3"
            and payload.get("architectures") == ["Qwen3ForCausalLM"],
            "model snapshot architecture drifted",
        )

    def _verified_file(self, path: Path, expected_sha256: str, label: str) -> Path:
        self._require_resource(path.is_file(), f"{label} unavailable: {path}")
        self._require_identity(
            self._sha256_file(path) == expected_sha256,
            f"{label} SHA-256 drifted",
        )
        return path

    def _backend_preflight(self, path: Path, expected: str) -> None:
        self._require_resource(path.is_dir(), f"pinned Any-Precision checkout unavailable: {path}")
        revision = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        self._require_resource(revision.returncode == 0, "cannot inspect Any-Precision revision")
        self._require_identity(
            revision.stdout.strip() == expected, "Any-Precision revision drifted"
        )
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain=v1"],
            capture_output=True,
            text=True,
            check=False,
        )
        self._require_resource(dirty.returncode == 0, "cannot inspect Any-Precision status")
        self._require_identity(not dirty.stdout.strip(), "Any-Precision checkout is dirty")

    def _external_preflight(self, protocol: Mapping[str, Any]) -> tuple[Path, Path, Path]:
        """Validate filesystem identities before importing Torch or Transformers."""

        identities = protocol["identities"]
        snapshot = self._snapshot_path(identities["model"]["revision"])
        artifact = (contract.ROOT / identities["packed_artifact"]["relative_path"]).resolve()
        self._require_resource(artifact.is_dir(), f"packed artifact unavailable: {artifact}")
        self._verified_file(
            artifact / identities["packed_artifact"]["checkpoint_file"],
            identities["packed_artifact"]["sha256"],
            "packed checkpoint",
        )
        self._backend_preflight(
            contract.ROOT / identities["any_precision"]["submodule_path"],
            identities["any_precision"]["commit"],
        )
        router_identity = identities["router_checkpoint"]
        router_path = (
            Path(
                os.environ.get(
                    router_identity["path_env_override"],
                    router_identity["recorded_external_path"],
                )
            )
            .expanduser()
            .resolve()
        )
        self._verified_file(
            router_path,
            router_identity["sha256"],
            "S07 router checkpoint",
        )
        return snapshot, artifact, router_path

    def _validate_comparable_gpu(self, gpu_model: str, device: str) -> None:
        self._require_resource(
            gpu_model == "NVIDIA GeForce RTX 3090",
            f"frozen comparable GPU is unavailable on {device}: {gpu_model}",
        )

    def _validate_student_representation(self, student: Any, torch: Any) -> None:
        """Validate the frozen resident packed target and router representation."""

        packed_modules = [
            module
            for module in student.modules()
            if module.__class__.__name__ == "AnyPrecisionLinear"
        ]
        self._require_identity(
            len(packed_modules) == 252,
            "resident student does not contain exactly 252 physical packed targets",
        )
        self._require_identity(
            all(
                module.qweight.dtype == torch.int32
                and module.qweight.ndim == 3
                and module.qweight.shape[0] == 8
                and module.qweight.is_cuda
                and module._buffers.get("lut4") is not None
                and module._buffers.get("lut8") is not None
                for module in packed_modules
            ),
            "resident student packed-plane/LUT representation drifted",
        )
        self._require_identity(student.router_count == 72, "router count drifted")
        self._require_identity(
            student.router_parameter_count == 23620752,
            "historical router scalar count drifted",
        )

    def prepare(
        self,
        protocol: Mapping[str, Any],
        mode: Mapping[str, Any],
        device: str,
        requests: Sequence[Mapping[str, Any]],
    ) -> None:
        active = os.environ.get("VIRTUAL_ENV", "")
        self._require_resource(
            bool(active) and Path(active).resolve() == (Path.home() / ".venv").resolve(),
            "exact ~/.venv environment is not active",
        )
        contract._validate_device(device)
        self.config = protocol
        self.mode = mode
        self.device = device
        snapshot, artifact, router_path = self._external_preflight(protocol)

        import torch
        import transformers

        self.torch = torch
        self._require_resource(torch.cuda.is_available(), "CUDA is unavailable")
        target = torch.device(device)
        self._require_resource(
            target.index is not None and target.index < torch.cuda.device_count(),
            f"explicit CUDA device is unavailable: {device}",
        )
        torch.cuda.set_device(target)
        gpu_model = torch.cuda.get_device_name(target)
        self._validate_comparable_gpu(gpu_model, device)
        driver = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
                "-i",
                str(target.index),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self._require_resource(driver.returncode == 0, "NVIDIA driver query failed")
        self._hardware = {
            "cuda_device": device,
            "device_index": target.index,
            "gpu_model": gpu_model,
            "driver_version": driver.stdout.strip().splitlines()[0],
            "cuda_runtime_version": str(torch.version.cuda),
            "pytorch_version": str(torch.__version__),
            "transformers_version": str(transformers.__version__),
            "python_version": platform.python_version(),
        }

        identities = protocol["identities"]
        revision = identities["model"]["revision"]

        import random

        seed = int(protocol["execution_contract"]["seed"])
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        from qaq.evaluation.quality import load_full_precision_model

        teacher = load_full_precision_model(snapshot, device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        teacher_before = self._entries_for_module(teacher)
        with torch.inference_mode():
            for repeat_index in range(2):
                for request in requests:
                    input_ids = torch.tensor(
                        request["full_input_ids"], dtype=torch.long, device=device
                    ).unsqueeze(0)
                    attention = torch.ones_like(input_ids, dtype=torch.bool)
                    output = teacher(
                        input_ids=input_ids,
                        attention_mask=attention,
                        use_cache=False,
                    ).logits.detach()
                    self._require_identity(
                        tuple(output.shape[:2]) == (1, 64)
                        and bool(torch.isfinite(output).all().item()),
                        "teacher logits are malformed or non-finite",
                    )
                    self.teacher_logits[(repeat_index, request["request_id"])] = (
                        output.cpu().clone()
                    )
        teacher_after = self._entries_for_module(teacher)
        self._teacher_audit = self._component_audit(teacher_before, teacher_after)
        self._require_identity(
            self._teacher_audit["hashes_equal"], "teacher state changed during inference"
        )
        teacher.cpu()
        del teacher
        torch.cuda.empty_cache()

        from qaq.router.distillation import RouterCheckpointMetadata, load_router_checkpoint
        from qaq.router.soft_model import load_soft_model

        student = load_soft_model(artifact, device, candidate_bits=(4, 8))
        manifest = json.loads(
            (contract.ROOT / identities["packed_artifact"]["manifest_path"]).read_text()
        )
        metadata = RouterCheckpointMetadata(
            model_repository=identities["model"]["repository"],
            model_revision=revision,
            quantized_checkpoint_id=identities["packed_artifact"]["relative_path"],
            quantized_checkpoint_hash=f"sha256:{identities['packed_artifact']['sha256']}",
            any_precision_revision=identities["any_precision"]["commit"],
            router_architecture={
                "feature_dim": int(student.feature_dim),
                "hidden_width": 128,
                "activation": "GELU",
                "normalization": "parameter-free RMS",
                "normalization_epsilon": 1e-6,
                "temperature": 1.0,
                "router_count": int(student.router_count),
            },
            candidate_ordering=(4, 8),
            training_step=4,
            training_step_metadata={"seed": 1729, "format": "qaq-s07b-router-training-v1"},
        )
        load_router_checkpoint(router_path, student.routers, metadata)
        student.to(device).eval()
        self._validate_student_representation(student, torch)
        for parameter in student.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self._require_identity(
            manifest["artifact"]["checkpoint_hashes"]["pytorch_model.bin"]
            == identities["packed_artifact"]["sha256"],
            "manifest packed identity drifted",
        )
        self.student = student
        self._student_before = self._student_components(student)
        if protocol.get("schema") == "qaq-s11c-broader-quality-v1":
            from qaq.evaluation import lookahead_broader_quality

            self._identities = lookahead_broader_quality._expected_identities(protocol)
        else:
            self._identities = contract._expected_identities(protocol)

    def hardware_evidence(self) -> dict[str, Any]:
        return dict(self._hardware)

    def identity_evidence(self) -> dict[str, Any]:
        return dict(self._identities)

    def _tensor_digest(self, value: Any) -> str:
        tensor = value.detach().cpu().contiguous()
        digest = hashlib.sha256()
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.view(self.torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def _state_entry(self, name: str, kind: str, value: Any) -> dict[str, Any]:
        return {
            "name": name,
            "kind": kind,
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "requires_grad": bool(value.requires_grad),
            "gradient_absent": getattr(value, "grad", None) is None,
            "value_sha256": self._tensor_digest(value),
        }

    def _entries_for_module(self, module: Any) -> list[dict[str, Any]]:
        entries = [
            self._state_entry(name, "parameter", value) for name, value in module.named_parameters()
        ]
        state_names = set(module.state_dict())
        entries.extend(
            self._state_entry(name, "buffer", value)
            for name, value in module.named_buffers()
            if name in state_names
        )
        return sorted(entries, key=lambda item: (item["name"], item["kind"]))

    def _student_components(self, student: Any) -> dict[str, list[dict[str, Any]]]:
        result = {
            "packed_weights_and_buffers": [],
            "non_router_base": [],
            "router": [],
        }
        for item in self._entries_for_module(student):
            name = item["name"]
            if name.startswith("routers."):
                result["router"].append(item)
            elif ".packed." in name:
                result["packed_weights_and_buffers"].append(item)
            else:
                result["non_router_base"].append(item)
        for name, entries in result.items():
            self._require_identity(bool(entries), f"state audit component is empty: {name}")
        return result

    @staticmethod
    def _provenance(mode_id: str, request_id: str) -> list[dict[str, Any]]:
        return [
            contract._expected_provenance(mode_id, request_id, layer, unit)
            for layer in range(36)
            for unit in contract.UNIT_TYPES
        ]

    def run_request(
        self,
        *,
        mode: Mapping[str, Any],
        request: Mapping[str, Any],
        repeat_index: int,
        device: str,
    ) -> dict[str, Any]:
        if self.student is None or self.torch is None:
            raise RuntimeError("INVALID_EVIDENCE: production runtime is not prepared")
        torch = self.torch
        from qaq.model.manual import PrecisionTrace
        from qaq.model.request_state import QaqRequestState
        from qaq.router.distillation import hard_route, masked_kl_distillation_loss

        state = QaqRequestState(
            request["request_id"],
            prompt_length=32,
            layer_count=36,
            candidate_bits=(4, 8),
            routing_timing=mode["routing_timing"],
        )
        trace = PrecisionTrace()

        def policy(layer: int, unit_type: str, feature: Any) -> int:
            probabilities = self.student.route(layer, unit_type, feature)
            return int(hard_route(probabilities, candidate_bits=(4, 8)))

        input_ids = torch.tensor(
            request["full_input_ids"], dtype=torch.long, device=device
        ).unsqueeze(0)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        prompt_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        prompt_mask[:, :32] = True
        with torch.inference_mode():
            student_logits = self.student.base(
                input_ids=input_ids,
                attention_mask=attention,
                use_cache=False,
                request_state=state,
                phase="prefill",
                prompt_attention_mask=prompt_mask,
                routing_policy=policy,
                trace=trace,
            ).logits.detach()
        teacher_logits = self.teacher_logits[(repeat_index, request["request_id"])].to(device)
        self._require_identity(
            tuple(student_logits.shape) == tuple(teacher_logits.shape),
            "teacher/student logits shape mismatch",
        )
        self._require_identity(
            bool(torch.isfinite(student_logits).all().item()), "student logits are non-finite"
        )
        self._require_identity(
            len(trace.records) == 252 and len(trace.route_records) == 72,
            "packed projection or route trace coverage drifted",
        )
        routes = [
            {
                "request_id": request["request_id"],
                "target_layer": layer,
                "unit_type": unit,
                "selected_bits": int(
                    state.attention_routes[layer]
                    if unit == "attention"
                    else state.ffn_routes[layer]
                ),
            }
            for layer in range(36)
            for unit in contract.UNIT_TYPES
        ]
        if mode["id"] == contract.MODE_IDS[1]:
            for layer in range(1, 36):
                observed = state.attention_provenance[layer]
                expected = contract._expected_provenance(
                    mode["id"], request["request_id"], layer, "attention"
                )
                self._require_identity(
                    observed is not None
                    and observed.to_dict()
                    == {
                        key: expected[key]
                        for key in (
                            "source_layer",
                            "target_layer",
                            "target_unit_type",
                            "source_point",
                            "routing_timing",
                        )
                    },
                    f"lookahead provenance drifted at target layer {layer}",
                )
        provenance = self._provenance(mode["id"], request["request_id"])
        completion_mask = torch.zeros((1, 64), dtype=torch.bool, device=device)
        completion_mask[:, 31:63] = True
        kl = masked_kl_distillation_loss(
            teacher_logits,
            student_logits,
            completion_mask,
            temperature=2.0,
        )
        absolute_error = (student_logits.float() - teacher_logits.float()).abs()
        raw = {
            "request_id": request["request_id"],
            "full_input_ids_sha256": request["token_digest_sha256"],
            "teacher_logits_digest": self._tensor_digest(teacher_logits),
            "student_logits_digest": self._tensor_digest(student_logits),
            "teacher_logits_shape": list(teacher_logits.shape),
            "student_logits_shape": list(student_logits.shape),
            "finite_teacher_logits": bool(torch.isfinite(teacher_logits).all().item()),
            "finite_student_logits": bool(torch.isfinite(student_logits).all().item()),
            "kl": float(kl.item()),
            "mean_absolute_logit_error": float(absolute_error.mean().item()),
            "maximum_absolute_logit_error": float(absolute_error.max().item()),
            "routes": routes,
            "provenance": provenance,
            "request_cleanup": {},
        }
        state.end_request()
        cleanup = {
            "state_ended": state.ended,
            "routes_released": all(
                item is None for item in state.attention_routes + state.ffn_routes
            ),
            "features_released": all(
                item is None for item in state.attention_features + state.ffn_features
            ),
            "probabilities_released": all(
                item is None for item in state.attention_probabilities + state.ffn_probabilities
            ),
            "provenance_released": all(
                item is None for item in state.attention_provenance + state.ffn_provenance
            ),
        }
        cleanup["passed"] = all(cleanup.values())
        raw["request_cleanup"] = cleanup
        return raw

    @staticmethod
    def _component_audit(
        before: list[dict[str, Any]], after: list[dict[str, Any]]
    ) -> dict[str, Any]:
        before_hash = contract._digest(before)
        after_hash = contract._digest(after)
        return {
            "before_entries": before,
            "after_entries": after,
            "parameter_count": sum(item["kind"] == "parameter" for item in before),
            "buffer_count": sum(item["kind"] == "buffer" for item in before),
            "before_aggregate_sha256": before_hash,
            "after_aggregate_sha256": after_hash,
            "hashes_equal": before == after,
        }

    def freeze_audit(self) -> dict[str, Any]:
        if self.student is None or self._teacher_audit is None:
            raise RuntimeError("INVALID_EVIDENCE: runtime state audit is unavailable")
        after = self._student_components(self.student)
        components = {
            "teacher": self._teacher_audit,
            "packed_weights_and_buffers": self._component_audit(
                self._student_before["packed_weights_and_buffers"],
                after["packed_weights_and_buffers"],
            ),
            "non_router_base": self._component_audit(
                self._student_before["non_router_base"], after["non_router_base"]
            ),
            "router": self._component_audit(self._student_before["router"], after["router"]),
        }
        before_hashes = {
            name: audit["before_aggregate_sha256"] for name, audit in components.items()
        }
        after_hashes = {name: audit["after_aggregate_sha256"] for name, audit in components.items()}
        self._freeze_result = {
            "components": components,
            "before_hashes": before_hashes,
            "after_hashes": after_hashes,
            "hashes_equal": before_hashes == after_hashes
            and all(audit["hashes_equal"] for audit in components.values()),
            "optimizer_absent": True,
            "gradients_absent": all(
                item["gradient_absent"]
                for audit in components.values()
                for item in audit["after_entries"]
            ),
        }
        return self._freeze_result

    def close(self) -> None:
        if self.student is not None:
            try:
                self.student.cpu()
            finally:
                self.student = None
        self.teacher_logits.clear()
        if self.torch is not None:
            self.torch.cuda.empty_cache()


__all__ = ["ProductionRuntime"]
