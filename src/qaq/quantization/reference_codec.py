"""Slow, independent reference codec for the pinned Any-Precision layout.

This module is a correctness oracle only.  Its intermediate bitmaps and code
arrays are intentionally easy to inspect and must not be used as production
inference storage or as evidence for production memory/transfer claims.

The pinned backend stores one ``int32`` word per plane, output row, and group
of 32 logical input positions.  Before the ``int32`` view, its packer applies
a warp-oriented byte permutation.  The implementation below reproduces that
permutation with ordinary NumPy indexing instead of importing the pinned
packer, so tests can compare the two implementations independently.
"""

from __future__ import annotations

import sys
from typing import Final

import numpy as np

_MIN_BITS: Final = 3
_MAX_BITS: Final = 8
_WORD_BITS: Final = 32
_WORD_BYTES: Final = 4
_WARP_BYTES: Final = 32 * _WORD_BYTES


def _validate_bits(bits: int, *, name: str) -> int:
    if not isinstance(bits, (int, np.integer)) or isinstance(bits, bool):
        raise TypeError(f"{name} must be an integer")
    bits = int(bits)
    if not _MIN_BITS <= bits <= _MAX_BITS:
        raise ValueError(f"{name} must be between {_MIN_BITS} and {_MAX_BITS}")
    return bits


def _validate_logical_codes(codes: np.ndarray, parent_bits: int) -> np.ndarray:
    codes = np.asarray(codes)
    if codes.ndim != 2:
        raise ValueError(f"logical codes must have shape (N, K), got {codes.shape}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise TypeError(f"logical codes must use an integer dtype, got {codes.dtype}")
    if codes.shape[0] <= 0 or codes.shape[1] <= 0:
        raise ValueError(f"logical codes must be non-empty, got {codes.shape}")
    if codes.shape[1] % _WORD_BITS:
        raise ValueError(
            "the pinned packer requires K divisible by 32; "
            f"got K={codes.shape[1]} (no implicit padding is applied)"
        )
    if np.any(codes < 0) or np.any(codes >= (1 << parent_bits)):
        raise ValueError(f"codes must be in [0, {1 << parent_bits})")
    return codes.astype(np.uint8, copy=False)


def _permutation_indices(total_bytes: int) -> np.ndarray:
    """Return the pinned packer's storage index for each logical byte.

    For a logical byte index ``b``, the returned value is the byte position
    where ``b`` is stored.  The source handles complete 128-byte warps and a
    final complete-word remainder separately; this mirrors that behavior.
    """

    if total_bytes <= 0 or total_bytes % _WORD_BYTES:
        raise ValueError(f"packed byte count must be a positive multiple of 4, got {total_bytes}")

    def calculate(byte_indices: np.ndarray, threads_per_warp: int, offset: int = 0) -> np.ndarray:
        bytes_per_warp = threads_per_warp * _WORD_BYTES
        warp_idx, byte_offsets = np.divmod(byte_indices, bytes_per_warp)
        warp_offsets = warp_idx * bytes_per_warp
        thread_indices = byte_indices % threads_per_warp
        byte_offsets_within_thread = (byte_offsets // threads_per_warp) ^ 3
        return warp_offsets + thread_indices * _WORD_BYTES + byte_offsets_within_thread + offset

    full_warps_bytes = (total_bytes // _WARP_BYTES) * _WARP_BYTES
    indices = np.empty(total_bytes, dtype=np.int64)
    if full_warps_bytes:
        indices[:full_warps_bytes] = calculate(
            np.arange(full_warps_bytes, dtype=np.int64), 32
        )
    remaining_bytes = total_bytes - full_warps_bytes
    if remaining_bytes:
        indices[full_warps_bytes:] = calculate(
            np.arange(remaining_bytes, dtype=np.int64),
            remaining_bytes // _WORD_BYTES,
            offset=full_warps_bytes,
        )
    return indices


def _permuted_bytes(bitmaps: np.ndarray) -> np.ndarray:
    total_bytes = bitmaps.shape[2]
    storage_indices = _permutation_indices(total_bytes)
    return bitmaps[:, :, np.argsort(storage_indices)]


def _unpermuted_bytes(stored: np.ndarray) -> np.ndarray:
    total_bytes = stored.shape[2]
    storage_indices = _permutation_indices(total_bytes)
    return stored[:, :, storage_indices]


def pack(codes: np.ndarray, parent_bits: int = 8) -> np.ndarray:
    """Pack logical unsigned LUT codes into pinned-backend ``int32`` planes.

    Args:
        codes: Integer array with shape ``(N, K)`` and values in
            ``[0, 2**parent_bits)``.
        parent_bits: Number of stored planes.  The pinned backend supports
            3 through 8; QAQ uses 8 as the nested parent representation.

    Returns:
        A C-contiguous little-endian ``int32`` array with shape
        ``(parent_bits, N, K // 32)``.  It is a test/reference value, not a
        production storage object.

    Raises:
        ValueError: If ``K`` is not divisible by 32.  The pinned packer does
            not pad non-aligned widths, and this codec does not invent padding.
    """

    parent_bits = _validate_bits(parent_bits, name="parent_bits")
    codes = _validate_logical_codes(codes, parent_bits)
    n_rows, logical_k = codes.shape
    total_bytes = logical_k // 8

    bitmaps = np.empty((parent_bits, n_rows, total_bytes), dtype=np.uint8)
    flat_codes = codes.reshape(-1)
    for plane in range(parent_bits):
        shift = parent_bits - 1 - plane
        logical_bits = ((flat_codes >> shift) & 1).astype(np.uint8, copy=False)
        bitmaps[plane] = np.packbits(
            logical_bits.reshape(n_rows, logical_k), axis=1, bitorder="big"
        )

    # The pinned source views the permuted bytes as native int32.  The tested
    # environment is little-endian; reject another host rather than silently
    # creating a serialization contract that was not established here.
    if sys.byteorder != "little":
        raise RuntimeError("the pinned serialized int32 representation is only verified on little-endian hosts")
    packed = _permuted_bytes(bitmaps).reshape(parent_bits, n_rows, -1, _WORD_BYTES)
    return np.ascontiguousarray(packed).view(np.dtype("<i4")).reshape(
        parent_bits, n_rows, logical_k // _WORD_BITS
    )


def _validate_packed(packed: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed)
    if packed.ndim != 3:
        raise ValueError(f"packed planes must have shape (P, N, K/32), got {packed.shape}")
    parent_bits, n_rows, words = packed.shape
    _validate_bits(parent_bits, name="packed plane count")
    if n_rows <= 0 or words <= 0:
        raise ValueError(f"packed planes must be non-empty, got {packed.shape}")
    if packed.dtype.kind != "i" or packed.dtype.itemsize != _WORD_BYTES:
        raise TypeError(f"packed planes must use int32 storage, got {packed.dtype}")
    if packed.dtype.byteorder == ">":
        raise ValueError("big-endian int32 packed storage is not supported by the pinned backend contract")
    if sys.byteorder != "little":
        raise RuntimeError("the pinned serialized int32 representation is only verified on little-endian hosts")
    return np.ascontiguousarray(packed)


def unpack(
    packed: np.ndarray,
    precision: int | None = None,
    logical_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Unpack physical planes into unsigned logical LUT codes.

    ``precision`` selects the leading plane prefix.  For an 8-plane parent,
    ``precision=4`` returns the code represented by planes 0..3, namely the
    high four bits of the parent code.  ``logical_shape`` can trim a physically
    padded payload for inspection, but :func:`pack` never creates such padding.
    """

    packed = _validate_packed(packed)
    parent_bits, n_rows, words = packed.shape
    if precision is None:
        precision = parent_bits
    precision = _validate_bits(precision, name="precision")
    if precision > parent_bits:
        raise ValueError(f"precision={precision} needs {precision} planes, only {parent_bits} are present")

    physical_k = words * _WORD_BITS
    if logical_shape is None:
        logical_n, logical_k = n_rows, physical_k
    else:
        if len(logical_shape) != 2:
            raise ValueError(f"logical_shape must be (N, K), got {logical_shape}")
        logical_n, logical_k = (int(logical_shape[0]), int(logical_shape[1]))
        if logical_n != n_rows or not 0 < logical_k <= physical_k:
            raise ValueError(
                f"logical_shape {logical_shape} is incompatible with packed shape {packed.shape}"
            )

    stored_bytes = packed.view(np.uint8).reshape(parent_bits, n_rows, words * _WORD_BYTES)
    bitmaps = _unpermuted_bytes(stored_bytes)
    unpacked_bits = np.unpackbits(bitmaps, axis=2, bitorder="big")
    unpacked_bits = unpacked_bits.reshape(parent_bits, n_rows, physical_k)

    codes = np.zeros((logical_n, logical_k), dtype=np.uint8)
    for plane in range(precision):
        codes |= unpacked_bits[plane, :, :logical_k].astype(np.uint8) << (precision - 1 - plane)
    return codes


def reconstruct(
    packed: np.ndarray,
    lut: np.ndarray,
    precision: int,
    *,
    scales: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct signed/real values by explicit row-wise LUT lookup.

    The pinned backend has no separate scale or sign plane: its quantized code
    indexes a row-specific LUT whose floating values may themselves be
    negative.  ``lut`` is therefore required and must have shape
    ``(N, 2**precision)``.  ``scales`` is an explicit guard: passing one is
    rejected because a separate scale representation is not part of the
    pinned format and must not be silently invented by this reference codec.
    """

    if scales is not None:
        raise ValueError("the pinned backend stores direct LUT values; separate scales are unsupported")
    precision = _validate_bits(precision, name="precision")
    lut = np.asarray(lut)
    if lut.ndim != 2 or lut.shape[1] != (1 << precision):
        raise ValueError(f"lut must have shape (N, {1 << precision}), got {lut.shape}")
    if not np.issubdtype(lut.dtype, np.floating):
        raise TypeError(f"lut must use a floating dtype, got {lut.dtype}")
    codes = unpack(packed, precision=precision)
    if lut.shape[0] != codes.shape[0]:
        raise ValueError(f"lut rows {lut.shape[0]} do not match packed rows {codes.shape[0]}")
    return np.take_along_axis(lut, codes, axis=1)


def packed_byte_count(n_rows: int, logical_k: int, parent_bits: int = 8) -> int:
    """Return physical qweight payload bytes for a backend-aligned tensor."""

    parent_bits = _validate_bits(parent_bits, name="parent_bits")
    if not isinstance(n_rows, (int, np.integer)) or n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if not isinstance(logical_k, (int, np.integer)) or logical_k <= 0 or logical_k % _WORD_BITS:
        raise ValueError(f"logical_k must be a positive multiple of 32, got {logical_k}")
    return parent_bits * int(n_rows) * (int(logical_k) // _WORD_BITS) * _WORD_BYTES


def lut_byte_count(n_rows: int, precision: int, *, element_size: int = 2) -> int:
    """Return bytes for one row-wise LUT tensor, excluding qweight payload."""

    precision = _validate_bits(precision, name="precision")
    if not isinstance(n_rows, (int, np.integer)) or n_rows <= 0:
        raise ValueError(f"n_rows must be positive, got {n_rows}")
    if element_size <= 0:
        raise ValueError(f"element_size must be positive, got {element_size}")
    return int(n_rows) * (1 << precision) * int(element_size)
