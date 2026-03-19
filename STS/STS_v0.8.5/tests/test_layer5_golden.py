# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Golden fixture tests.

Golden fixtures are small hand checkable examples.
They act as a regression suite for core Stack Theory behaviours.
"""

import pytest

from stacktheory.layer5.golden import golden_toy_vocabulary


def test_golden_induced_language_and_truth_sets():
    case = golden_toy_vocabulary()

    L = case.vocab.induced_language_masks()
    assert L == case.expected_language_masks

    for name, stmt in case.statements.items():
        truth = stmt.truth_set().bitset
        assert truth == case.expected_truth_bitsets[name]


def test_golden_weakness_values():
    case = golden_toy_vocabulary()
    for name, expected in case.expected_weakness.items():
        stmt = case.statements[name]
        assert case.vocab.weakness(stmt) == expected


def test_golden_abstractor_programs_for_A():
    case = golden_toy_vocabulary()
    A = case.statements["A"]
    abs_vocab = case.vocab.abstractor(A)

    # Expected truth sets are T(A) and T(AB).
    bitsets = sorted([p.bitset for p in abs_vocab.programs])
    assert bitsets == sorted([case.expected_truth_bitsets["A"], case.expected_truth_bitsets["AB"]])

    # Abstractor programs must belong to the same environment.
    assert all(p.env is case.env for p in abs_vocab.programs)
