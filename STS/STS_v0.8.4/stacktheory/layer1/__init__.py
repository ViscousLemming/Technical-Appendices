# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

r"""Layer 1.

Layer 1 is the semantic core of the Stack Theory Suite.

It implements the Stack Theory primitives for finite environments.
Everything in this layer is exact.

Math mapping

- An environment is a finite state set \Phi.
- A program is a subset p \subseteq \Phi.
- A vocabulary is a finite set of programs.
- A statement is a subset of a vocabulary.
  It is valid when its truth set is not empty.

Representation mapping

Programs are stored as packed bitsets.
This is exactly the same as a boolean tensor membership vector.
It is just much smaller and much faster for set ops.
"""

from .environment import FiniteEnvironment, BooleanCubeEnvironment, IndexEnvironment
from .program import Program
from .vocabulary import Vocabulary, Statement
from .logic import Literal, Clause, Term, CNF, DNF
from .bitset import pack_bool_tensor, unpack_bool_tensor, bitset_popcount

__all__ = [
    "FiniteEnvironment",
    "BooleanCubeEnvironment",
    "IndexEnvironment",
    "Program",
    "Vocabulary",
    "Statement",
    "Literal",
    "Clause",
    "Term",
    "CNF",
    "DNF",
    "pack_bool_tensor",
    "unpack_bool_tensor",
    "bitset_popcount",
]
