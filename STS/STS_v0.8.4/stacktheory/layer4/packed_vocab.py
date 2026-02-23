# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Packed vocabulary backend for large extensional environments.

This module provides a GPU friendly representation of a Layer 1 Vocabulary.
Each program is stored as a vector of 64 bit words in a torch.int64 tensor.

The semantics are extensional.
Each bit position refers to the same environment index as in Layer 1.

This is designed for two common workflows.

Incremental truth set updates
Search algorithms often move from a statement to a neighbouring statement by
adding or removing one vocabulary element.
For extensional programs, the truth set update is one bitwise AND.

Batched candidate scoring
When expanding a node, you often want to score many candidate additions.
If you represent the current truth set as words of length W and you have K
candidate programs, you can compute K new truth sets with one broadcast AND.

The environment in this repository is CPU only.
This module still supports CUDA devices when used on a GPU enabled machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Vocabulary
from stacktheory.layer2.wordbitset import WordBitset, popcount_words


@dataclass
class PackedVocabulary:
    """A vocabulary with programs stored as packed word tensors."""

    vocab: Vocabulary
    program_words: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.program_words, torch.Tensor):
            raise TypeError("program_words must be a torch.Tensor")
        if self.program_words.dtype != torch.int64:
            raise TypeError("program_words must have dtype torch.int64")
        if self.program_words.ndim != 2:
            raise ValueError("program_words must have shape (m, n_words)")
        if self.program_words.shape[0] != self.vocab.size:
            raise ValueError("program_words first dimension must equal vocab.size")

        expected_words = (self.vocab.env.size + 63) // 64
        if int(self.program_words.shape[1]) != expected_words:
            raise ValueError("program_words second dimension does not match env size")

        # Cache Phi words for repeated truth set computations.
        self._phi_words = WordBitset.full(self.n_bits, device=self.device).words

    @property
    def device(self) -> torch.device:
        return self.program_words.device

    @property
    def n_bits(self) -> int:
        return int(self.vocab.env.size)

    @property
    def n_words(self) -> int:
        return int(self.program_words.shape[1])

    def to(self, device: str | torch.device) -> "PackedVocabulary":
        return PackedVocabulary(self.vocab, self.program_words.to(device=device))

    @classmethod
    def from_vocabulary(cls, vocab: Vocabulary, *, device: str | torch.device = "cpu") -> "PackedVocabulary":
        """Create a PackedVocabulary from a Layer 1 Vocabulary."""
        m = vocab.size
        n_bits = int(vocab.env.size)
        n_words = (n_bits + 63) // 64

        words = torch.zeros((m, n_words), dtype=torch.int64)
        for i, p in enumerate(vocab.programs):
            if not isinstance(p, Program):
                raise TypeError("vocab.programs must be Program instances")
            wb = WordBitset.from_bitset_int(p.bitset, n_bits=n_bits, device="cpu")
            words[i] = wb.words

        return cls(vocab=vocab, program_words=words.to(device=device))

    def phi_words(self) -> torch.Tensor:
        """Return the word tensor for Phi, meaning all environment states."""
        return self._phi_words

    def truth_words_of_mask(self, mask: int) -> torch.Tensor:
        """Return T(mask) as a word tensor.

        This mirrors Vocabulary.truth_set_of_mask.
        """
        if mask < 0 or mask >= (1 << self.vocab.size):
            raise ValueError("mask contains bits above vocabulary size")

        words = self.phi_words()
        for i in self.vocab_statement_indices(mask):
            words = words & self.program_words[i]
            if self.is_empty_words(words):
                break
        return words

    def vocab_statement_indices(self, mask: int) -> List[int]:
        """Return indices of set bits in a statement mask."""
        out: List[int] = []
        b = int(mask)
        while b:
            lsb = b & -b
            out.append(lsb.bit_length() - 1)
            b ^= lsb
        return out

    def is_empty_words(self, words: torch.Tensor) -> bool:
        """Return True if and only if the packed word tensor is empty."""
        if words.numel() == 0:
            return True
        return bool(torch.all(words == 0).item())

    def cardinality_words(self, words: torch.Tensor) -> int:
        """Return |words| as a state count."""
        if words.numel() == 0:
            return 0
        return int(popcount_words(words).item())

    def apply_program(self, base_words: torch.Tensor, program_index: int) -> torch.Tensor:
        """Return base_words AND vocab.programs[program_index]."""
        if program_index < 0 or program_index >= self.vocab.size:
            raise ValueError("program_index out of range")
        return base_words & self.program_words[int(program_index)]

    def apply_programs_batch(self, base_words: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Apply multiple program intersections in one batched operation.

        Parameters
        ----------
        base_words
            Word tensor of shape (n_words,).
        indices
            Integer tensor of shape (k,) with program indices.

        Returns
        -------
        torch.Tensor
            Word tensor of shape (k, n_words).
        """
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if indices.ndim != 1:
            raise ValueError("indices must be a 1D tensor")
        idx = indices.to(device=self.device, dtype=torch.int64)
        return base_words.unsqueeze(0) & self.program_words.index_select(0, idx)
