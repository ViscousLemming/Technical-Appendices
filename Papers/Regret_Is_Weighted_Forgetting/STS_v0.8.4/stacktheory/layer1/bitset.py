# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Packed bitset utilities.

Layer 1 uses a packed bitset as the canonical internal representation for any
extensional set of environment states.

A packed bitset is a non negative integer whose binary representation encodes
membership. Bit k is 1 if and only if the element with index k is in the set.

This is exactly the same meaning as a boolean tensor membership vector.
A boolean tensor stores one byte or one word per element.
A packed bitset stores one bit per element.

Why this matters

- Bitwise AND is set intersection.
- Bitwise OR is set union.
- Bitwise NOT, when masked to the universe size, is set complement.

The mapping is defined relative to a fixed finite enumeration of the
environment states.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import torch


def bitset_full(n_bits: int) -> int:
    """Return the bitset with the lowest n_bits set to 1."""
    if n_bits < 0:
        raise ValueError("n_bits must be non negative")
    if n_bits == 0:
        return 0
    return (1 << n_bits) - 1


def bitset_popcount(bitset: int) -> int:
    """Return the number of 1 bits in the bitset."""
    if bitset < 0:
        raise ValueError("bitset must be non negative")
    return bitset.bit_count()


def pack_bool_tensor(x: torch.Tensor) -> int:
    """Pack a 1D boolean tensor into an integer bitset.

    The tensor is flattened in row major order.
    Index 0 becomes the least significant bit.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    x_flat = x.to(dtype=torch.bool, device="cpu").flatten()
    n_bits = int(x_flat.numel())
    if n_bits == 0:
        return 0

    # Fast path.
    # We pack with little bit order so index 0 becomes the least significant bit.
    as_u8 = x_flat.numpy().astype(np.uint8, copy=False)
    packed = np.packbits(as_u8, bitorder="little")
    return int.from_bytes(packed.tobytes(), byteorder="little", signed=False)


def unpack_bool_tensor(bitset: int, n_bits: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Unpack an integer bitset into a 1D boolean tensor of length n_bits."""
    if n_bits < 0:
        raise ValueError("n_bits must be non negative")
    if bitset < 0:
        raise ValueError("bitset must be non negative")
    if n_bits == 0:
        return torch.empty((0,), dtype=torch.bool, device=device)

    n_bytes = (n_bits + 7) // 8
    raw = int(bitset).to_bytes(n_bytes, byteorder="little", signed=False)
    byte_arr = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(byte_arr, bitorder="little")[:n_bits]
    return torch.from_numpy(bits.astype(np.bool_, copy=False)).to(device=device)


def bitset_from_indices(indices: Iterable[int]) -> int:
    """Build a bitset from an iterable of non negative indices."""
    bits = 0
    for i in indices:
        if i < 0:
            raise ValueError("indices must be non negative")
        bits |= 1 << int(i)
    return bits


def bitset_to_indices(bitset: int) -> list[int]:
    """Return the sorted list of indices whose bits are 1."""
    if bitset < 0:
        raise ValueError("bitset must be non negative")

    out: list[int] = []
    b = int(bitset)
    while b:
        lsb = b & -b
        out.append(lsb.bit_length() - 1)
        b ^= lsb
    return out


def bitset_iter_indices(bitset: int) -> Iterator[int]:
    """Yield indices whose bits are 1 in ascending order."""
    if bitset < 0:
        raise ValueError("bitset must be non negative")
    b = int(bitset)
    while b:
        lsb = b & -b
        yield lsb.bit_length() - 1
        b ^= lsb
