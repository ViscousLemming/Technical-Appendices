# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Word packed bitset backend that supports CPU and GPU tensors.

Layer 1 represents extensional sets as Python integers.
That is exact and fast for many single set operations.

For large environments and batched operations, it is useful to store the same
extensional set as a vector of 64 bit words in a torch tensor.
This representation maps cleanly to GPU devices.

The representation is logically equivalent to the integer bitset.
Each bit position refers to the same environment index.

Bit numbering convention

- bit 0 is the least significant bit of word 0
- word index is floor(bit / 64)
- bit position inside a word is bit % 64

This matches Layer 1 packing and unpacking.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from .bitpack import pack_bool_tensor_uint64, unpack_bool_tensor_uint64


def _n_words_for_bits(n_bits: int) -> int:
    if n_bits < 0:
        raise ValueError("n_bits must be non negative")
    return (int(n_bits) + 63) // 64


def _tail_mask_int64(n_bits: int) -> int:
    """Mask for the last word so bits above n_bits are 0.

    The mask is returned as a Python int that fits in signed 64 bit.
    When n_bits is a multiple of 64, the mask is all ones.
    """

    r = int(n_bits) % 64
    if r == 0:
        return -1
    return (1 << r) - 1


@lru_cache(maxsize=8)
def _byte_popcount_table(device_key: str) -> torch.Tensor:
    table = torch.zeros((256,), dtype=torch.uint8)
    for i in range(256):
        table[i] = int(i).bit_count()
    device = torch.device(device_key)
    return table.to(device=device)


def popcount_words(words: torch.Tensor) -> torch.Tensor:
    """Return popcount for a tensor of int64 words.

    Input shape can be (n_words,) or (..., n_words).
    Output shape is (...) with integer counts.

    This implementation is defined in terms of raw bytes.
    It is correct for negative int64 values because it interprets the tensor
    as two's complement bit patterns.
    """

    if not isinstance(words, torch.Tensor):
        raise TypeError("words must be a torch.Tensor")
    w = words.to(dtype=torch.int64)
    if w.numel() == 0:
        out_shape = w.shape[:-1] if w.ndim >= 1 else ()
        return torch.zeros(out_shape, dtype=torch.int64, device=w.device)
    if w.ndim == 0:
        raise ValueError("words must have at least one dimension")

    b = w.contiguous().view(torch.uint8)
    b = b.view(*w.shape[:-1], w.shape[-1] * 8)

    table = _byte_popcount_table(str(w.device))
    counts = table[b.to(dtype=torch.int64)].to(dtype=torch.int64)
    return counts.sum(dim=-1)


@dataclass(frozen=True)
class WordBitset:
    """Packed bitset as a vector of 64 bit words.

    Parameters
    ----------
    words
        1D tensor of dtype int64.
    n_bits
        Number of meaningful bits.
        Bits above n_bits are treated as 0.
    """

    words: torch.Tensor
    n_bits: int

    def __post_init__(self) -> None:
        if self.n_bits < 0:
            raise ValueError("n_bits must be non negative")
        if not isinstance(self.words, torch.Tensor):
            raise TypeError("words must be a torch.Tensor")
        if self.words.ndim != 1:
            raise ValueError("words must be a 1D tensor")

        if self.words.dtype != torch.int64:
            w = self.words.to(dtype=torch.int64)
        else:
            w = self.words

        n_words = _n_words_for_bits(self.n_bits)
        if int(w.numel()) != n_words:
            raise ValueError("words length does not match n_bits")

        # Canonicalise storage.
        #
        # We want.
        # - contiguous words for predictable performance
        # - tail bits above n_bits forced to 0
        #
        # This code avoids an unconditional clone, because cloning large
        # word vectors dominates runtime and memory in batched workloads.

        if n_words == 0:
            object.__setattr__(self, "words", w.contiguous() if not w.is_contiguous() else w)
            return

        w_contig = w.contiguous() if not w.is_contiguous() else w

        tail = _tail_mask_int64(self.n_bits)
        if tail == -1:
            object.__setattr__(self, "words", w_contig)
            return

        # If the tail bits are already masked, keep the tensor as is.
        last = w_contig[-1]
        if bool(((last & int(tail)) == last).item()):
            object.__setattr__(self, "words", w_contig)
            return

        # Otherwise, copy and fix the last word.
        ww = w_contig.clone()
        ww[-1] = ww[-1] & int(tail)
        object.__setattr__(self, "words", ww)

    @property
    def device(self) -> torch.device:
        return self.words.device

    @property
    def n_words(self) -> int:
        return int(self.words.numel())

    def to(self, device: str | torch.device) -> "WordBitset":
        return WordBitset(self.words.to(device=device), n_bits=self.n_bits)

    def clone(self) -> "WordBitset":
        return WordBitset(self.words.clone(), n_bits=self.n_bits)

    def is_empty(self) -> bool:
        if self.n_words == 0:
            return True
        return bool(torch.all(self.words == 0).item())

    def cardinality(self) -> int:
        if self.n_words == 0:
            return 0
        return int(popcount_words(self.words).item())

    def issubset(self, other: "WordBitset") -> bool:
        self._check_compatible(other)
        return bool(torch.all((self.words & (~other.words)) == 0).item())

    def to_bool_tensor(self, *, device: str | torch.device | None = None) -> torch.Tensor:
        if device is None:
            device = self.device
        return unpack_bool_tensor_uint64(self.words, n_bits=self.n_bits, device=device)

    def to_bitset_int(self) -> int:
        """Convert to the Layer 1 integer bitset.

        This is intended for tests and interoperability.
        """

        if self.n_bits == 0:
            return 0
        w_cpu = self.words.to(dtype=torch.int64, device="cpu").contiguous().numpy()
        u64 = w_cpu.view(np.uint64)
        raw = u64.tobytes()
        bitset = int.from_bytes(raw, byteorder="little", signed=False)
        if self.n_bits % 64:
            bitset &= (1 << self.n_bits) - 1
        return int(bitset)

    @classmethod
    def zeros(cls, n_bits: int, *, device: str | torch.device = "cpu") -> "WordBitset":
        n_words = _n_words_for_bits(n_bits)
        return cls(torch.zeros((n_words,), dtype=torch.int64, device=device), n_bits=n_bits)

    @classmethod
    def full(cls, n_bits: int, *, device: str | torch.device = "cpu") -> "WordBitset":
        n_words = _n_words_for_bits(n_bits)
        if n_words == 0:
            return cls(torch.empty((0,), dtype=torch.int64, device=device), n_bits=0)
        words = torch.full((n_words,), fill_value=-1, dtype=torch.int64, device=device)
        return cls(words, n_bits=n_bits)

    @classmethod
    def from_bool_tensor(cls, x: torch.Tensor, *, device: str | torch.device | None = None) -> "WordBitset":
        """Create a WordBitset from a boolean tensor.

        Packing is performed on CPU using numpy for deterministic bit order.
        The packed words are then moved to the requested device.
        """

        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if device is None:
            device = x.device
        packed = pack_bool_tensor_uint64(x)
        return cls(packed.to(device=device), n_bits=int(x.numel()))

    @classmethod
    def from_bitset_int(cls, bitset: int, *, n_bits: int, device: str | torch.device = "cpu") -> "WordBitset":
        """Create a WordBitset from the Layer 1 integer bitset."""

        if n_bits < 0:
            raise ValueError("n_bits must be non negative")
        if bitset < 0:
            raise ValueError("bitset must be non negative")

        n_words = _n_words_for_bits(n_bits)
        if n_words == 0:
            return cls(torch.empty((0,), dtype=torch.int64, device=device), n_bits=0)

        n_bytes = n_words * 8
        raw = int(bitset).to_bytes(n_bytes, byteorder="little", signed=False)
        arr = np.frombuffer(raw, dtype=np.uint8)
        u64 = arr.view(np.uint64)
        i64 = u64.view(np.int64)
        t = torch.from_numpy(i64.copy()).to(dtype=torch.int64, device=device)
        return cls(t, n_bits=n_bits)

    def __and__(self, other: "WordBitset") -> "WordBitset":
        self._check_compatible(other)
        return WordBitset(self.words & other.words, n_bits=self.n_bits)

    def __or__(self, other: "WordBitset") -> "WordBitset":
        self._check_compatible(other)
        return WordBitset(self.words | other.words, n_bits=self.n_bits)

    def __xor__(self, other: "WordBitset") -> "WordBitset":
        self._check_compatible(other)
        return WordBitset(self.words ^ other.words, n_bits=self.n_bits)

    def __invert__(self) -> "WordBitset":
        if self.n_words == 0:
            return self
        inv = ~self.words
        tail = _tail_mask_int64(self.n_bits)
        inv = inv.clone()
        inv[-1] = inv[-1] & int(tail)
        return WordBitset(inv, n_bits=self.n_bits)

    def __sub__(self, other: "WordBitset") -> "WordBitset":
        self._check_compatible(other)
        return WordBitset(self.words & (~other.words), n_bits=self.n_bits)

    def _check_compatible(self, other: "WordBitset") -> None:
        if not isinstance(other, WordBitset):
            raise TypeError("other must be a WordBitset")
        if self.n_bits != other.n_bits:
            raise ValueError("WordBitsets must have the same n_bits")
