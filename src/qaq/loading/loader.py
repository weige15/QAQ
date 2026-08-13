"""S08-A synchronous request-scoped loading of nested packed planes.

The CPU ``PackedLinearSource`` is authoritative.  A loader owns only the
GPU buffers retained for one concrete ``QaqRequestState`` and copies missing
packed planes and lookup tables synchronously on first use.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import lru_cache

import torch

from ..model.request_state import QaqRequestState
from ..model.static import ANY_PRECISION_ROOT, source_commit

SUPPORTED_BITS = (4, 8)
PARENT_PLANES = 8
ATOMIC_K_SPLIT_MIN_K = 4096
ATOMIC_K_SPLIT_MIN_BITS = 7


@lru_cache(maxsize=1)
def pinned_backend() -> tuple[Callable[..., torch.Tensor], Callable[..., torch.Tensor]]:
    """Return the pinned CUDA dequantization and packed-matmul functions."""

    source_commit()
    if str(ANY_PRECISION_ROOT) not in sys.path:
        sys.path.insert(0, str(ANY_PRECISION_ROOT))
    from any_precision_ext import dequant_kbit, matmul_kbit

    return dequant_kbit, matmul_kbit


def uses_atomic_k_split(
    inputs: torch.Tensor, qweight: torch.Tensor, precision: int
) -> bool:
    """Mirror the pinned ``matmul_kbit`` dispatch that uses atomic accumulation."""

    if inputs.ndim == 0 or qweight.ndim != 3:
        return False
    rows = int(inputs.numel() // inputs.shape[-1])
    in_features = int(qweight.shape[2] * 32)
    device_name = torch.cuda.get_device_name(qweight.device)
    return (
        device_name != "Orin"
        and rows == 1
        and in_features > ATOMIC_K_SPLIT_MIN_K
        and precision >= ATOMIC_K_SPLIT_MIN_BITS
    )


def execute_packed_linear(
    inputs: torch.Tensor,
    qweight: torch.Tensor,
    lut: torch.Tensor,
    precision: int,
    *,
    dequant_kbit: Callable[..., torch.Tensor],
    matmul_kbit: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Use deterministic dequantized arithmetic only for pinned atomic k-split calls."""

    rows = int(inputs.numel() // inputs.shape[-1])
    if uses_atomic_k_split(inputs, qweight, precision) or rows > 8:
        weight = dequant_kbit(qweight, lut, precision)
        return torch.matmul(inputs, weight.transpose(0, 1))
    return matmul_kbit(inputs, qweight, lut, precision)


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel() * tensor.element_size())


def _buffer_description(name: str, tensor: torch.Tensor) -> dict[str, object]:
    return {
        "name": name,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "bytes": _tensor_bytes(tensor),
    }


@dataclass(frozen=True, slots=True)
class PackedLinearSource:
    """CPU-authoritative nested packed buffers for one projection."""

    module_id: str
    qweight: torch.Tensor
    lut4: torch.Tensor
    lut8: torch.Tensor
    bias: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.module_id, str) or not self.module_id:
            raise ValueError("module_id must be a non-empty string")
        for name, tensor in (
            ("qweight", self.qweight),
            ("lut4", self.lut4),
            ("lut8", self.lut8),
        ):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.device.type != "cpu":
                raise ValueError(f"{name} must remain CPU-authoritative; got {tensor.device}")
            if not tensor.is_contiguous():
                raise ValueError(f"{name} must be contiguous")
        if self.qweight.dtype != torch.int32:
            raise ValueError(f"qweight must use torch.int32 packed storage; got {self.qweight.dtype}")
        if self.qweight.ndim != 3 or self.qweight.shape[0] != PARENT_PLANES:
            raise ValueError(
                f"qweight must have shape [8, N, K//32]; got {tuple(self.qweight.shape)}"
            )
        if self.qweight.shape[2] <= 0:
            raise ValueError("qweight must contain at least one packed input word")
        if self.lut4.dtype != torch.float16 or self.lut8.dtype != torch.float16:
            raise ValueError("Any-Precision lookup tables must use torch.float16")
        if self.lut4.ndim != 2 or self.lut4.shape[1] != 16:
            raise ValueError(f"lut4 must have shape [N, 16]; got {tuple(self.lut4.shape)}")
        if self.lut8.ndim != 2 or self.lut8.shape[1] != 256:
            raise ValueError(f"lut8 must have shape [N, 256]; got {tuple(self.lut8.shape)}")
        if self.lut4.shape[0] != self.qweight.shape[1] or self.lut8.shape[0] != self.qweight.shape[1]:
            raise ValueError("lookup-table row counts must match qweight output rows")
        if self.bias is not None:
            if self.bias.device.type != "cpu" or not self.bias.is_contiguous():
                raise ValueError("bias must be a contiguous CPU tensor")
            if self.bias.shape != (self.qweight.shape[1],):
                raise ValueError("bias shape must match qweight output rows")

    @classmethod
    def from_module(cls, module: torch.nn.Module, module_id: str) -> PackedLinearSource:
        """Copy an existing verified packed module into CPU-authoritative storage."""

        bias = getattr(module, "bias", None)
        return cls(
            module_id=module_id,
            qweight=module.qweight.detach().to(device="cpu").contiguous(),
            lut4=module.lut4.detach().to(device="cpu").contiguous(),
            lut8=module.lut8.detach().to(device="cpu").contiguous(),
            bias=None if bias is None else bias.detach().to(device="cpu").contiguous(),
        )

    @property
    def in_features(self) -> int:
        return int(self.qweight.shape[2] * 32)

    @property
    def out_features(self) -> int:
        return int(self.qweight.shape[1])


@dataclass(frozen=True, slots=True)
class TransferRecord:
    """Compact evidence for one loader call; it contains no weight contents."""

    request_id: str
    request_state_identity: int
    module_id: str
    precision: int
    source_device: str
    destination_device: str
    event: str
    transferred_bytes: int
    buffers: tuple[dict[str, object], ...]


class SynchronousPackedPlaneLoader:
    """Load one packed projection for exactly one request-state object."""

    def __init__(
        self,
        source: PackedLinearSource,
        request_state: QaqRequestState,
        device: str | torch.device,
    ) -> None:
        if not isinstance(source, PackedLinearSource):
            raise TypeError("source must be a PackedLinearSource")
        if not isinstance(request_state, QaqRequestState):
            raise TypeError("request_state must be a QaqRequestState")
        self.source = source
        self.request_state = request_state
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("S08-A synchronous packed loading requires a CUDA destination")
        if not torch.cuda.is_available():
            raise RuntimeError("S08-A requires CUDA for the packed integration proof")
        source_commit()
        if str(ANY_PRECISION_ROOT) not in sys.path:
            sys.path.insert(0, str(ANY_PRECISION_ROOT))
        from any_precision_ext import dequant_kbit, matmul_kbit

        self._dequant_kbit = dequant_kbit
        self._matmul_kbit = matmul_kbit
        self._qweight: torch.Tensor | None = None
        self._retained_planes = 0
        self._lut4: torch.Tensor | None = None
        self._lut8: torch.Tensor | None = None
        self._bias: torch.Tensor | None = None
        self._used_precisions: set[int] = set()
        self._records: list[TransferRecord] = []
        self._closed = False
        request_state.register_cleanup(self.close)

    @property
    def records(self) -> tuple[TransferRecord, ...]:
        return tuple(self._records)

    @property
    def retained_entry_count(self) -> int:
        return int(
            self._qweight is not None
            or self._lut4 is not None
            or self._lut8 is not None
            or self._bias is not None
        )

    @property
    def retained_gpu_buffer_count(self) -> int:
        return sum(
            tensor is not None
            for tensor in (self._qweight, self._lut4, self._lut8, self._bias)
        )

    @property
    def retained_plane_count(self) -> int:
        return self._retained_planes

    @property
    def retained_packed_bytes(self) -> int:
        """Return bytes held by this request for packed GPU buffers only."""

        return sum(
            _tensor_bytes(tensor)
            for tensor in (self._qweight, self._lut4, self._lut8)
            if tensor is not None
        )

    def _ensure_open(self) -> None:
        if self._closed or self.request_state.ended:
            raise RuntimeError("S08-A loader cannot be used after request cleanup")

    def _copy_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        # Do not pass non_blocking=True: this is the synchronous baseline path.
        copied = tensor.to(device=self.device)
        torch.cuda.synchronize(self.device)
        return copied

    def _ensure_precision(self, precision: int) -> tuple[tuple[dict[str, object], ...], int]:
        transferred: list[dict[str, object]] = []
        if self._qweight is None:
            self._qweight = self._copy_to_device(self.source.qweight[:precision])
            self._retained_planes = precision
            transferred.append(_buffer_description(f"qweight[:{precision}]", self._qweight))
        elif self._retained_planes < precision:
            start = self._retained_planes
            missing = self._copy_to_device(self.source.qweight[start:precision])
            self._qweight = torch.cat((self._qweight, missing), dim=0).contiguous()
            self._retained_planes = precision
            transferred.append(_buffer_description(f"qweight[{start}:{precision}]", missing))
        lut_name = f"lut{precision}"
        if precision == 4 and self._lut4 is None:
            self._lut4 = self._copy_to_device(self.source.lut4)
            transferred.append(_buffer_description(lut_name, self._lut4))
        elif precision == 8 and self._lut8 is None:
            self._lut8 = self._copy_to_device(self.source.lut8)
            transferred.append(_buffer_description(lut_name, self._lut8))
        if self.source.bias is not None and self._bias is None:
            self._bias = self._copy_to_device(self.source.bias)
            transferred.append(_buffer_description("bias", self._bias))
        return tuple(transferred), sum(int(item["bytes"]) for item in transferred)

    def __call__(self, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        self._ensure_open()
        if isinstance(precision, bool) or precision not in SUPPORTED_BITS:
            raise ValueError(f"S08-A supports only precisions 4 and 8; got {precision!r}")
        if not isinstance(inputs, torch.Tensor) or inputs.device != self.device:
            raise ValueError(f"inputs must be on the loader destination {self.device}")
        if inputs.shape[-1] != self.source.in_features:
            raise ValueError(
                f"inputs last dimension must be {self.source.in_features}; got {inputs.shape[-1]}"
            )
        buffers, transferred_bytes = self._ensure_precision(precision)
        event = "first_use" if precision not in self._used_precisions else "reuse"
        self._used_precisions.add(precision)
        self._records.append(
            TransferRecord(
                request_id=self.request_state.request_id,
                request_state_identity=id(self.request_state),
                module_id=self.source.module_id,
                precision=precision,
                source_device="cpu",
                destination_device=str(self.device),
                event=event,
                transferred_bytes=transferred_bytes,
                buffers=buffers,
            )
        )
        lut = self._lut4 if precision == 4 else self._lut8
        if self._qweight is None or lut is None:  # pragma: no cover - guarded by _ensure_precision
            raise RuntimeError("S08-A packed execution buffers were not retained")
        output = execute_packed_linear(
            inputs,
            self._qweight,
            lut,
            precision,
            dequant_kbit=self._dequant_kbit,
            matmul_kbit=self._matmul_kbit,
        )
        if self._bias is not None:
            output = output + self._bias
        return output

    def close(self) -> None:
        """Release all GPU buffers owned by this concrete request loader."""

        if self._closed:
            return
        self._qweight = None
        self._lut4 = None
        self._lut8 = None
        self._bias = None
        self._retained_planes = 0
        self._used_precisions.clear()
        self._closed = True


class SynchronousPackedRequest:
    """Request-local loader collection with no model- or process-global cache."""

    def __init__(
        self,
        sources: Mapping[str, PackedLinearSource],
        request_state: QaqRequestState,
        device: str | torch.device,
    ) -> None:
        if not isinstance(request_state, QaqRequestState):
            raise TypeError("request_state must be a QaqRequestState")
        if request_state.ended:
            raise RuntimeError("cannot create a packed request after request cleanup")
        self.sources = dict(sources)
        if any(not isinstance(key, str) or not isinstance(value, PackedLinearSource)
               for key, value in self.sources.items()):
            raise TypeError("sources must map module IDs to PackedLinearSource values")
        self.request_state = request_state
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("S08-B synchronous packed loading requires a CUDA destination")
        self._loaders: dict[str, SynchronousPackedPlaneLoader] = {}
        self._records: list[TransferRecord] = []

    def loader_for(self, module_id: str) -> SynchronousPackedPlaneLoader:
        if self.request_state.ended:
            raise RuntimeError("cannot load packed data after request cleanup")
        try:
            source = self.sources[module_id]
        except KeyError as error:
            raise KeyError(f"no on-demand packed source for {module_id}") from error
        loader = self._loaders.get(module_id)
        if loader is None:
            loader = SynchronousPackedPlaneLoader(source, self.request_state, self.device)
            self._loaders[module_id] = loader
        return loader

    def execute(self, module_id: str, inputs: torch.Tensor, *, precision: int) -> torch.Tensor:
        """Run one projection and preserve global execution order for evidence."""

        loader = self.loader_for(module_id)
        start = len(loader.records)
        output = loader(inputs, precision=precision)
        self._records.extend(loader.records[start:])
        return output

    @property
    def loaders(self) -> tuple[SynchronousPackedPlaneLoader, ...]:
        return tuple(self._loaders.values())

    @property
    def records(self) -> tuple[TransferRecord, ...]:
        return tuple(self._records)

    @property
    def retained_entry_count(self) -> int:
        return sum(loader.retained_entry_count for loader in self.loaders)

    @property
    def retained_gpu_buffer_count(self) -> int:
        return sum(loader.retained_gpu_buffer_count for loader in self.loaders)

    @property
    def retained_packed_bytes(self) -> int:
        return sum(loader.retained_packed_bytes for loader in self.loaders)


__all__ = [
    "PackedLinearSource",
    "SynchronousPackedPlaneLoader",
    "SynchronousPackedRequest",
    "TransferRecord",
    "execute_packed_linear",
    "pinned_backend",
    "uses_atomic_k_split",
]
