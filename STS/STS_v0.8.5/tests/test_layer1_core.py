# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Tests for Layer 1 semantics.

These tests check the exact Stack Theory primitives implemented in Layer 1.
They cover packed bitset equivalence, program set operations, induced language
membership, truth sets, completions, and weakness.

If one of these tests fails, the code is no longer aligned with the Layer 1
math definitions.
"""

import math
import random

import torch

from stacktheory.layer1 import (
    FiniteEnvironment,
    BooleanCubeEnvironment,
    Program,
    Vocabulary,
    Literal,
    Clause,
    Term,
    CNF,
    DNF,
    pack_bool_tensor,
    unpack_bool_tensor,
    bitset_popcount,
)
from stacktheory.layer1.logic import (
    assignment_tensor_from_state,
    cnf_tensor_satisfied,
    dnf_tensor_satisfied,
)


def test_pack_unpack_round_trip():
    rng = random.Random(0)
    n = 137
    x = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)
    b = pack_bool_tensor(x)
    y = unpack_bool_tensor(b, n_bits=n)
    assert torch.equal(x, y)


def test_pack_boolean_algebra():
    rng = random.Random(1)
    n = 80
    a = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)
    b = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)
    pa = pack_bool_tensor(a)
    pb = pack_bool_tensor(b)
    assert pack_bool_tensor(a & b) == (pa & pb)
    assert pack_bool_tensor(a | b) == (pa | pb)

    # Complement is relative to the n-bit universe.
    full = (1 << n) - 1
    assert pack_bool_tensor(~a) == ((~pa) & full)


def test_program_set_operations():
    env = FiniteEnvironment(states=["a", "b", "c", "d"])
    p = Program.from_state_indices(env, [0, 2])
    q = Program.from_state_indices(env, [2, 3])

    assert (p & q).bitset == (1 << 2)
    assert (p | q).cardinality() == 3
    assert (~p).cardinality() == env.size - p.cardinality()
    assert p.issubset(p | q)
    assert (p - q).bitset == (1 << 0)


def test_vocabulary_induced_language_truth_sets_and_weakness():
    # Environment has 3 states.
    env = FiniteEnvironment(states=[0, 1, 2])
    p0 = Program.from_state_indices(env, [0, 1])
    p1 = Program.from_state_indices(env, [1, 2])
    p2 = Program.from_state_indices(env, [2])

    v = Vocabulary(env, programs=[p0, p1, p2], names=["p0", "p1", "p2"])

    L = v.induced_language()
    assert len(L) > 0
    for s in L:
        assert s.is_valid()
        assert not s.truth_set().is_empty()

    empty = v.empty_statement()
    assert empty.is_valid()
    assert empty.truth_set().cardinality() == env.size

    s = v.statement(["p0"])
    # Completions of {p0} are {p0} and {p0,p1}. {p0,p2} is inconsistent, {p0,p1,p2} is inconsistent.
    comp = v.completions(s)
    comp_masks = {c.mask for c in comp}
    assert comp_masks == {v.statement(["p0"]).mask, v.statement(["p0", "p1"]).mask}
    assert v.weakness(s) == 2

    # Monotonicity under inclusion. If l1 subset l2 then w(l1) >= w(l2).
    s12 = v.statement(["p0", "p1"])
    assert v.weakness(s) >= v.weakness(s12)


def test_abstractor_deduplicates_truth_sets_and_is_deterministic():
    env = FiniteEnvironment(states=[0, 1, 2])
    p0 = Program.from_state_indices(env, [0, 1])
    p1 = Program.from_state_indices(env, [1, 2])
    p2 = Program.from_state_indices(env, [1])

    v = Vocabulary(env, programs=[p0, p1, p2], names=["p0", "p1", "p2"])
    abs_vocab = v.abstractor(v.empty_statement())

    # Several distinct completions have the same truth set {1}.
    # The abstractor function is a set of programs so duplicates must collapse.
    bitsets = [p.bitset for p in abs_vocab.programs]
    assert bitsets == sorted(set(bitsets))
    assert set(bitsets) == {2, 3, 6, 7}


def test_vocabulary_rejects_duplicate_programs():
    env = FiniteEnvironment(states=["a", "b", "c"])
    p0 = Program.from_state_indices(env, [0])
    p0_dup = Program(env=env, bitset=p0.bitset)
    try:
        Vocabulary(env, programs=[p0, p0_dup], names=["p0", "p0_dup"])
    except ValueError:
        assert True
    else:
        assert False


def test_program_from_bool_tensor_checks_env_size():
    env = FiniteEnvironment(states=[0, 1, 2])
    x = torch.zeros((env.size + 1,), dtype=torch.bool)
    try:
        Program.from_bool_tensor(env, x)
    except ValueError:
        assert True
    else:
        assert False


def test_boolean_cube_literal_vocabulary_weakness_matches_closed_form():
    env = BooleanCubeEnvironment(n=3)
    programs = []
    names = []
    for var in range(env.n):
        for val in (0, 1):
            programs.append(env.literal_program(var=var, value=val))
            names.append(f"x{var}={val}")
    v = Vocabulary(env, programs=programs, names=names)

    s = v.statement(["x0=1", "x2=0"])
    # Statement fixes 2 variables, leaves 1 variable free.
    # In the literal vocabulary, each free variable has 3 completion choices.
    expected_weakness = 3 ** (env.n - 2)
    assert v.weakness(s) == expected_weakness

    # Truth set size is number of compatible complete assignments.
    expected_models = 2 ** (env.n - 2)
    assert s.truth_set().cardinality() == expected_models


def test_cnf_compilation_and_tensor_round_trip():
    env = BooleanCubeEnvironment(n=3)
    cnf = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1), Literal(1, 0)]),
            Clause.from_iterable([Literal(2, 1)]),
        ]
    )
    prog = cnf.to_program(env)

    # Check extensional semantics by brute force.
    for state in range(env.size):
        expected = cnf.satisfied_by(state)
        actual = bool((prog.bitset >> state) & 1)
        assert expected == actual

    t = cnf.to_tensor(n_vars=env.n)
    cnf2 = CNF.from_tensor(t)
    assert cnf2 == cnf


def test_cnf_tensor_evaluation_matches_syntactic_semantics():
    env = BooleanCubeEnvironment(n=4)
    cnf = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1), Literal(2, 0)]),
            Clause.from_iterable([Literal(1, 1)]),
        ]
    )
    cnf_t = cnf.to_tensor(n_vars=env.n)

    assignments = torch.stack([assignment_tensor_from_state(s, env.n) for s in range(env.size)], dim=0)
    sat_tensor = cnf_tensor_satisfied(assignments, cnf_t)

    for state in range(env.size):
        assert bool(sat_tensor[state].item()) == cnf.satisfied_by(state)


def test_dnf_tensor_evaluation_matches_syntactic_semantics():
    env = BooleanCubeEnvironment(n=4)
    dnf = DNF.from_iterable(
        [
            Term.from_iterable([Literal(0, 1), Literal(1, 0)]),
            Term.from_iterable([Literal(3, 1)]),
        ]
    )
    dnf_t = dnf.to_tensor(n_vars=env.n)

    assignments = torch.stack([assignment_tensor_from_state(s, env.n) for s in range(env.size)], dim=0)
    sat_tensor = dnf_tensor_satisfied(assignments, dnf_t)

    for state in range(env.size):
        assert bool(sat_tensor[state].item()) == dnf.satisfied_by(state)


def test_weakness_monotone_under_inclusion_randomised():
    # Weakness is always monotone under statement inclusion.
    # If l1 subset l2 then Ext(l2) subset Ext(l1) so w(l1) >= w(l2).
    rng = random.Random(2)

    for _trial in range(10):
        n_states = 8
        env = FiniteEnvironment(states=list(range(n_states)))

        # Build a small random vocabulary of unique programs.
        progs = []
        names = []
        seen = set()
        m = 10
        while len(progs) < m:
            bits = 0
            for i in range(n_states):
                if rng.random() < 0.5:
                    bits |= 1 << i
            # Allow empty programs sometimes, but keep uniqueness.
            if bits in seen:
                continue
            seen.add(bits)
            progs.append(Program(env, bits))
            names.append(f"p{len(progs)}")

        v = Vocabulary(env, programs=progs, names=names)
        L = v.induced_language_masks()
        assert 0 in L

        for _ in range(200):
            m2 = rng.choice(L)
            # Pick a random subset of m2 by clearing some bits.
            m1 = m2
            for j in range(v.size):
                if (m1 >> j) & 1 and rng.random() < 0.5:
                    m1 &= ~(1 << j)
            # m1 must be in the induced language by downward closure.
            assert v.is_in_language_mask(m1)
            assert v.weakness_of_mask(m1) >= v.weakness_of_mask(m2)
