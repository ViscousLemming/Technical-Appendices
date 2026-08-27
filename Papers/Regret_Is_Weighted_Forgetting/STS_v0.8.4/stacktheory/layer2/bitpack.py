# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Packed uint64 bitset representation for large extensional sets.

Layer 1 represents an extensional set as a Python integer.
That is simple and correct.

For larger environments it can be useful to store bits in a torch.uint64 vector.
This is a step toward GPU capable model set operations.

Bit numbering convention

- bit 0 is the least significant bit of word 0
- word index is floor(bit / 64)
- bit position inside a word is bit % 64
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch


def pack_bool_tensor_uint64(x: torch.Tensor) -> torch.Tensor:
    """Pack a 1D boolean tensor into a 64 bit word vector.

    PyTorch CPU does not implement bitwise operations for torch.uint64.
    This function therefore uses torch.int64 as the storage dtype.
    Bits are interpreted using two's complement.
    """
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    flat = x.to(dtype=torch.bool, device="cpu").flatten()
    n_bits = int(flat.numel())
    if n_bits == 0:
        return torch.empty((0,), dtype=torch.int64)

    # Pack to bytes with little bit order so index 0 becomes the least significant bit.
    as_u8 = flat.numpy().astype(np.uint8, copy=False)
    packed_bytes = np.packbits(as_u8, bitorder="little")

    # Pad bytes to a multiple of 8 for uint64 words.
    pad = (-int(packed_bytes.size)) % 8
    if pad:
        packed_bytes = np.pad(packed_bytes, (0, pad), constant_values=0)

    words_u64 = packed_bytes.view(np.uint64)
    words_i64 = words_u64.view(np.int64)
    return torch.from_numpy(words_i64.copy())


def unpack_bool_tensor_uint64(bits: torch.Tensor, *, n_bits: int, device: str | torch.device = "cpu") -> torch.Tensor:
    """Unpack a 64 bit word vector into a 1D boolean tensor of length n_bits."""
    if not isinstance(bits, torch.Tensor):
        raise TypeError("bits must be a torch.Tensor")
    if n_bits < 0:
        raise ValueError("n_bits must be non negative")

    b = bits.to(dtype=torch.int64, device="cpu").flatten().contiguous()
    if b.numel() == 0:
        return torch.zeros((n_bits,), dtype=torch.bool, device=device)

    as_np = b.numpy()
    u64 = as_np.view(np.uint64)
    byte_view = u64.view(np.uint8)
    unpacked = np.unpackbits(byte_view, bitorder="little")
    unpacked = unpacked[:n_bits].astype(np.bool_, copy=False)
    return torch.from_numpy(unpacked).to(device=device)


@lru_cache(maxsize=1)
def _byte_popcount_table() -> torch.Tensor:
    table = torch.zeros((256,), dtype=torch.uint8)
    for i in range(256):
        table[i] = int(i).bit_count()
    return table


def popcount_uint64(bits: torch.Tensor) -> int:
    """Return the number of 1 bits in a packed 64 bit word vector."""
    if not isinstance(bits, torch.Tensor):
        raise TypeError("bits must be a torch.Tensor")

    b = bits.to(dtype=torch.int64, device="cpu").flatten()
    if b.numel() == 0:
        return 0

    table = _byte_popcount_table()
    byte_view = b.view(torch.uint8)
    byte_idx = byte_view.to(dtype=torch.int64)
    return int(table[byte_idx].to(dtype=torch.int64).sum().item())
