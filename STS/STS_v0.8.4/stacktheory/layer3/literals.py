# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Literal vocabulary fast paths.

When the environment is a Boolean cube and the vocabulary consists only of
literal programs, many Stack Theory quantities have closed forms.

This module detects that special case and exposes fast exact computations.

Definitions

- A literal program is the set of assignments where x_var = value.
- A literal vocabulary is a vocabulary where every program is one of these literals.

In this setting, a statement is satisfiable if and only if it does not contain a contradiction.
This lets us compute weakness without enumerating extensions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple

from stacktheory.layer1.environment import BooleanCubeEnvironment
from stacktheory.layer1.vocabulary import Vocabulary


@dataclass(frozen=True)
class LiteralVocabularyInfo:
    """Structural information about a literal vocabulary."""

    n_vars: int
    # Map (var, value) -> program index.
    literal_to_index: Dict[Tuple[int, int], int]
    # Map var -> set of available values in the vocabulary.
    available_values: Dict[int, Set[int]]

    def weakness_of_mask(self, mask: int) -> int:
        """Closed form weakness for a statement mask.

        This equals the number of consistent supersets inside the induced language.
        """
        fixed: Set[int] = set()
        for (var, value), idx in self.literal_to_index.items():
            if (mask >> idx) & 1:
                fixed.add(var)

        w = 1
        for var in range(self.n_vars):
            if var in fixed:
                continue
            w *= 1 + len(self.available_values.get(var, set()))
        return int(w)


def analyze_literal_vocabulary(vocab: Vocabulary) -> Optional[LiteralVocabularyInfo]:
    """Return LiteralVocabularyInfo if vocab is a literal vocabulary.

    Returns None otherwise.
    """
    env = vocab.env
    if not isinstance(env, BooleanCubeEnvironment):
        return None

    n = env.n
    literal_to_index: Dict[Tuple[int, int], int] = {}
    available: Dict[int, Set[int]] = {i: set() for i in range(n)}

    # Build a lookup table from literal bitset to (var, value).
    # This is only feasible for modest n because the bitsets have size 2^n.
    bitset_to_lit: Dict[int, Tuple[int, int]] = {}
    for var in range(n):
        for value in (0, 1):
            bitset_to_lit[env.literal_program_mask(var=var, value=value)] = (var, value)

    for idx, p in enumerate(vocab.programs):
        lit = bitset_to_lit.get(p.bitset)
        if lit is None:
            return None
        if lit in literal_to_index:
            # Duplicate literal program.
            return None
        literal_to_index[lit] = idx
        available[lit[0]].add(lit[1])

    return LiteralVocabularyInfo(n_vars=n, literal_to_index=literal_to_index, available_values=available)
