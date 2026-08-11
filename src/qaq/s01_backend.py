"""Small S01 adapter and deterministic evidence helpers for Any-Precision.

The adapter intentionally delegates packing, dequantization, and CUDA matmul to
the pinned Any-Precision source.  Synthetic packing below is test-only: it
constructs the same parent-label input consumed by the pinned pack helper and
never represents a production quantization pipeline.
"""

from __future__ import annotations

import hashlib
import importlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANY_PRECISION_ROOT = PROJECT_ROOT / "third_party" / "any-precision-llm"
PINNED_ANY_PRECISION_COMMIT = "a3257d02740cc5757c78673da534b0630ff3a4ea"

SEED = 1729
SUPPORTED_PRECISIONS = (4, 8)
M = 4
N = 64
K = 1024
INPUT_DTYPE = torch.float16
LUT_DTYPE = torch.float16
QWEIGHT_DTYPE = torch.int32
ATOL = 5e-2
RTOL = 1e-2
RELATIVE_FLOOR = 1e-2


@dataclass
class BackendCase:
    """One deterministic CUDA operation shared by tests and validation."""

    linear: torch.nn.Module
    inputs: torch.Tensor
    source_weights: torch.Tensor
    labels8: torch.Tensor
    device: torch.device
    dequant_kbit: Any


def _source_commit() -> str:
    if not ANY_PRECISION_ROOT.is_dir():
        raise RuntimeError(f"Pinned Any-Precision source is missing: {ANY_PRECISION_ROOT}")
    repository = subprocess.run(
        ["git", "-C", str(ANY_PRECISION_ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    )
    if repository.returncode != 0:
        raise RuntimeError(
            f"Could not inspect pinned Any-Precision repository: {repository.stderr.strip()}"
        )
    if Path(repository.stdout.strip()).resolve() != ANY_PRECISION_ROOT.resolve():
        raise RuntimeError(f"Pinned Any-Precision source is not initialized: {ANY_PRECISION_ROOT}")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(ANY_PRECISION_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if status.returncode != 0:
        raise RuntimeError(
            f"Could not inspect pinned Any-Precision source status: {status.stderr.strip()}"
        )
    if status.stdout.strip():
        raise RuntimeError(f"Pinned Any-Precision source is dirty: {status.stdout.strip()}")
    completed = subprocess.run(
        ["git", "-C", str(ANY_PRECISION_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Could not inspect pinned Any-Precision source: {completed.stderr.strip()}"
        )
    commit = completed.stdout.strip()
    if commit != PINNED_ANY_PRECISION_COMMIT:
        raise RuntimeError(
            f"Any-Precision source mismatch: expected {PINNED_ANY_PRECISION_COMMIT}, got {commit}"
        )
    return commit


def require_cuda() -> torch.device:
    """Return the target CUDA device or fail explicitly; CUDA absence is not a pass."""

    if not torch.cuda.is_available():
        raise RuntimeError("S01 requires CUDA; torch.cuda.is_available() returned False")
    return torch.device("cuda:0")


def load_pinned_backend() -> tuple[Any, Any, Any]:
    """Load the pinned class and its two compiled CUDA entry points."""

    _source_commit()
    source_path = str(ANY_PRECISION_ROOT)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)

    linear_module = importlib.import_module("any_precision.modules.AnyPrecisionLinear")
    extension = importlib.import_module("any_precision_ext")
    return linear_module.AnyPrecisionLinear, extension.dequant_kbit, extension.matmul_kbit


def _pack_parent_labels(labels8: np.ndarray) -> torch.Tensor:
    """Use the pinned pack helper for test-only physical parent-label packing."""

    if labels8.shape != (N, K) or labels8.dtype != np.uint8:
        raise ValueError(
            f"expected uint8 labels with shape {(N, K)}, got {labels8.shape} {labels8.dtype}"
        )

    # These operations mirror the pinned pack.py caller: one MSB-first bitmap
    # per parent bit, followed by the source's warp/thread byte permutation.
    pack_module = importlib.import_module("any_precision.quantization.pack")
    bitarray = np.empty((8, labels8.size // 8), dtype=np.uint8)
    mask = 1 << 7
    flattened = labels8.reshape(-1)
    for bit in range(8):
        bitarray[bit] = np.packbits((flattened & mask).astype(bool))
        mask >>= 1
    bitarray = bitarray.reshape((8, N, K // 8))
    packed = pack_module._permute_bitmaps_int32(bitarray)
    return torch.from_numpy(np.asarray(packed).copy())


def _synthetic_source() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    source_weights = torch.empty((N, K), dtype=torch.float32).uniform_(
        -0.75, 0.75, generator=generator
    )
    inputs = torch.empty((M, K), dtype=torch.float32).uniform_(-0.5, 0.5, generator=generator)

    row_min = source_weights.amin(dim=1, keepdim=True)
    row_span = source_weights.amax(dim=1, keepdim=True) - row_min
    normalized = ((source_weights - row_min) / row_span).clamp(0.0, 1.0)
    labels8 = torch.round(normalized * 255.0).to(torch.uint8)

    levels8 = torch.arange(256, dtype=torch.float32).unsqueeze(0)
    lut8 = row_min + row_span * (levels8 / 255.0)
    levels4 = (torch.arange(16, dtype=torch.float32).unsqueeze(0) * 16.0 + 7.5) / 255.0
    lut4 = row_min + row_span * levels4
    return source_weights, inputs, labels8, torch.cat((lut4, lut8), dim=1)


def build_case() -> BackendCase:
    """Build one deterministic, physically packed synthetic CUDA linear."""

    device = require_cuda()
    AnyPrecisionLinear, dequant_kbit, _ = load_pinned_backend()
    source_weights, inputs_cpu, labels8, luts = _synthetic_source()

    linear = AnyPrecisionLinear(
        K,
        N,
        list(SUPPORTED_PRECISIONS),
        bias=False,
        precisions=list(SUPPORTED_PRECISIONS),
        device=device,
        dtype=LUT_DTYPE,
    )
    packed_cpu = _pack_parent_labels(labels8.numpy())
    with torch.no_grad():
        linear.qweight.copy_(packed_cpu.to(device=device, dtype=QWEIGHT_DTYPE))
        linear.lut4.copy_(luts[:, :16].to(device=device, dtype=LUT_DTYPE))
        linear.lut8.copy_(luts[:, 16:].to(device=device, dtype=LUT_DTYPE))

    return BackendCase(
        linear=linear,
        inputs=inputs_cpu.to(device=device, dtype=INPUT_DTYPE),
        source_weights=source_weights,
        labels8=labels8,
        device=device,
        dequant_kbit=dequant_kbit,
    )


def packed_output(case: BackendCase, precision: int) -> torch.Tensor:
    if precision not in SUPPORTED_PRECISIONS:
        raise ValueError(f"S01 only exercises precisions {SUPPORTED_PRECISIONS}, got {precision}")
    with torch.no_grad():
        output = case.linear(case.inputs, precision=precision)
    torch.cuda.synchronize(case.device)
    return output.detach()


def dequantized_reference(case: BackendCase, precision: int) -> tuple[torch.Tensor, torch.Tensor]:
    lut = case.linear._buffers[f"lut{precision}"]
    with torch.no_grad():
        weight = case.dequant_kbit(case.linear.qweight, lut, precision)
        reference = torch.matmul(case.inputs, weight.transpose(0, 1))
    torch.cuda.synchronize(case.device)
    return weight.detach(), reference.detach()


def tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def comparison_metrics(output: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    error = (output.float() - reference.float()).abs()
    reference_abs = reference.float().abs()
    meaningful = reference_abs >= RELATIVE_FLOOR
    relative = error[meaningful] / reference_abs[meaningful]
    return {
        "max_absolute_error": float(error.max().item()),
        "mean_absolute_error": float(error.mean().item()),
        "meaningful_max_relative_error": float(relative.max().item()) if relative.numel() else 0.0,
        "meaningful_relative_floor": RELATIVE_FLOOR,
        "allclose": bool(torch.allclose(output, reference, atol=ATOL, rtol=RTOL)),
        "atol": ATOL,
        "rtol": RTOL,
    }


def assert_matches_reference(case: BackendCase, precision: int) -> dict[str, Any]:
    output = packed_output(case, precision)
    weight, reference = dequantized_reference(case, precision)
    metrics = comparison_metrics(output, reference)
    if not metrics["allclose"]:
        raise AssertionError(f"{precision}-bit packed output missed reference tolerance: {metrics}")
    metrics.update(
        {
            "precision": precision,
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "output_device": str(output.device),
            "output_digest": tensor_digest(output),
            "reference_digest": tensor_digest(reference),
            "dequantized_weight_shape": list(weight.shape),
            "dequantized_weight_dtype": str(weight.dtype),
            "output_sample": output.flatten()[:8].float().cpu().tolist(),
            "reference_sample": reference.flatten()[:8].float().cpu().tolist(),
        }
    )
    return metrics


def storage_observations(case: BackendCase) -> dict[str, Any]:
    qweight = case.linear.qweight
    return {
        "qweight_shape": list(qweight.shape),
        "qweight_dtype": str(qweight.dtype),
        "qweight_device": str(qweight.device),
        "qweight_contiguous": bool(qweight.is_contiguous()),
        "qweight_bytes": int(qweight.numel() * qweight.element_size()),
        "selected_packed_bytes": {
            str(bit): int(bit * qweight.shape[1] * qweight.shape[2] * qweight.element_size())
            for bit in SUPPORTED_PRECISIONS
        },
        "lookup": {
            str(bit): {
                "shape": list(case.linear._buffers[f"lut{bit}"].shape),
                "dtype": str(case.linear._buffers[f"lut{bit}"].dtype),
                "device": str(case.linear._buffers[f"lut{bit}"].device),
                "bytes": int(
                    case.linear._buffers[f"lut{bit}"].numel()
                    * case.linear._buffers[f"lut{bit}"].element_size()
                ),
            }
            for bit in SUPPORTED_PRECISIONS
        },
        "bias": case.linear.bias is not None,
        "supported_bits": list(case.linear.supported_bits),
        "precisions": list(case.linear.precisions),
        "default_precision": int(case.linear.precision),
    }


def full_validation_report() -> dict[str, Any]:
    """Run the complete S01 evidence path and return JSON-serializable observations."""

    case = build_case()
    results = {str(bit): assert_matches_reference(case, bit) for bit in SUPPORTED_PRECISIONS}

    deterministic = {}
    for bit in SUPPORTED_PRECISIONS:
        first = packed_output(case, bit)
        second = packed_output(case, bit)
        if not torch.equal(first, second):
            raise AssertionError(
                f"repeated {bit}-bit packed executions are not bitwise deterministic"
            )
        deterministic[str(bit)] = {
            "bitwise_equal": True,
            "digest_first": tensor_digest(first),
            "digest_second": tensor_digest(second),
        }

    weight4, _ = dequantized_reference(case, 4)
    weight8, _ = dequantized_reference(case, 8)
    weight_delta = (weight4.float() - weight8.float()).abs()
    prefix4 = case.linear.qweight[:4]
    suffix4 = case.linear.qweight[4:]
    distinct = {
        "supported_precisions": list(case.linear.precisions),
        "lut4_shape": list(case.linear.lut4.shape),
        "lut8_shape": list(case.linear.lut8.shape),
        "qweight_prefix4_digest": tensor_digest(prefix4),
        "qweight_suffix4_digest": tensor_digest(suffix4),
        "qweight_suffix4_nonzero": bool(torch.count_nonzero(suffix4).item()),
        "effective_weight4_digest": tensor_digest(weight4),
        "effective_weight8_digest": tensor_digest(weight8),
        "effective_weight_max_absolute_delta": float(weight_delta.max().item()),
        "effective_weights_distinct": bool(torch.any(weight_delta > 0).item()),
    }
    if not distinct["qweight_suffix4_nonzero"] or not distinct["effective_weights_distinct"]:
        raise AssertionError(f"4-bit and 8-bit precision paths were not distinct: {distinct}")

    return {
        "pinned_commit": PINNED_ANY_PRECISION_COMMIT,
        "seed": SEED,
        "dimensions": {"M": M, "N": N, "K": K},
        "dtypes": {"input": str(INPUT_DTYPE), "lut": str(LUT_DTYPE), "qweight": str(QWEIGHT_DTYPE)},
        "synthetic": {
            "source_weights_shape": list(case.source_weights.shape),
            "source_weights_dtype": str(case.source_weights.dtype),
            "source_weights_digest": tensor_digest(case.source_weights),
            "inputs_shape": list(case.inputs.shape),
            "inputs_dtype": str(case.inputs.dtype),
            "inputs_digest": tensor_digest(case.inputs),
            "parent_labels_shape": list(case.labels8.shape),
            "parent_labels_dtype": str(case.labels8.dtype),
            "parent_labels_digest": tensor_digest(case.labels8),
        },
        "device": {"index": case.device.index, "name": torch.cuda.get_device_name(case.device)},
        "storage": storage_observations(case),
        "precision_results": results,
        "determinism": deterministic,
        "distinct_precision_paths": distinct,
        "tolerance_rationale": (
            "ATOL=0.05 and RTOL=0.01 were selected before execution for fp16 inputs, fp16 LUT/dequantized "
            "weights, fp16 accumulation behavior in the pinned CUDA kernel, and the independent fp16 matmul."
        ),
        "no_model_or_dataset": True,
    }
