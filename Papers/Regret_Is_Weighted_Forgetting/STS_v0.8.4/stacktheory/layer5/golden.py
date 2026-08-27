# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Golden fixtures for regression testing.

Golden fixtures are tiny finite objects with known answers.
They are meant to be read by humans.

The main fixture in this module is a 4 state environment with a 3 program vocabulary.
It is small enough that you can compute truth sets and extensions by hand.

Doctest

>>> case = golden_toy_vocabulary()
>>> case.vocab.size
3
>>> case.vocab.weakness(case.statements["empty"])
6
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from stacktheory.layer1.environment import FiniteEnvironment
from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Statement, Vocabulary


@dataclass(frozen=True)
class GoldenToyCase:
    """A small hand checkable case.

    Attributes
    env
        FiniteEnvironment with 4 states.
    vocab
        Vocabulary with programs A, B, C.
    statements
        Convenience statements keyed by name.
        Keys include empty, A, B, C, AB, BC.
    expected_language_masks
        Sorted list of masks in the induced language.
    expected_weakness
        Expected weakness values w(l) for the named statements.
    expected_truth_bitsets
        Expected truth set membership bitsets for the named statements.
    """

    env: FiniteEnvironment
    vocab: Vocabulary
    statements: Dict[str, Statement]
    expected_language_masks: List[int]
    expected_weakness: Dict[str, int]
    expected_truth_bitsets: Dict[str, int]


def golden_toy_vocabulary() -> GoldenToyCase:
    """Return a small vocabulary with known induced language and weakness values.

    Construction

    Environment Phi has four states.
    They are enumerated as 0, 1, 2, 3.

    Programs are.

    - A = {0, 1}
    - B = {1, 2}
    - C = {2, 3}

    The induced language contains exactly six statements.

    - empty
    - A
    - B
    - C
    - AB
    - BC

    Statement AC is not allowed because A and C have empty intersection.
    ABC is not allowed for the same reason.
    """

    env = FiniteEnvironment(states=[0, 1, 2, 3])

    A = Program.from_state_indices(env, [0, 1])
    B = Program.from_state_indices(env, [1, 2])
    C = Program.from_state_indices(env, [2, 3])

    vocab = Vocabulary(env, programs=[A, B, C], names=["A", "B", "C"])

    s_empty = vocab.empty_statement()
    s_A = vocab.statement(["A"])
    s_B = vocab.statement(["B"])
    s_C = vocab.statement(["C"])
    s_AB = vocab.statement(["A", "B"])
    s_BC = vocab.statement(["B", "C"])

    statements: Dict[str, Statement] = {
        "empty": s_empty,
        "A": s_A,
        "B": s_B,
        "C": s_C,
        "AB": s_AB,
        "BC": s_BC,
    }

    expected_language_masks = sorted([s.mask for s in statements.values()])

    expected_weakness = {
        "empty": 6,
        "A": 2,
        "B": 3,
        "C": 2,
        "AB": 1,
        "BC": 1,
    }

    expected_truth_bitsets = {
        "empty": 0b1111,
        "A": 0b0011,
        "B": 0b0110,
        "C": 0b1100,
        "AB": 0b0010,
        "BC": 0b0100,
    }

    return GoldenToyCase(
        env=env,
        vocab=vocab,
        statements=statements,
        expected_language_masks=expected_language_masks,
        expected_weakness=expected_weakness,
        expected_truth_bitsets=expected_truth_bitsets,
    )
