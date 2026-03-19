# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Tests for Layer 2 encodings and adapters.

Layer 2 must not change semantics.
These tests check that tensor and DIMACS conversions round trip and that the
resulting objects agree with Layer 1 truth sets.
"""

import random

import torch

from stacktheory.layer1 import BooleanCubeEnvironment, FiniteEnvironment, Literal, Clause, Term, CNF, DNF, Program, Vocabulary
from stacktheory.layer2 import (
    statement_to_mask_tensor,
    statement_from_mask_tensor,
    vocabulary_to_tensor,
    vocabulary_from_tensor,
    cnf_to_str,
    dnf_to_str,
    parse_cnf,
    parse_dnf,
    cnf_to_dimacs,
    cnf_from_dimacs,
    cnf_to_sympy,
    dnf_to_sympy,
    sympy_to_cnf,
    sympy_to_dnf,
    pack_bool_tensor_uint64,
    unpack_bool_tensor_uint64,
    popcount_uint64,
    sat_solve_cnf,
    sat_is_satisfiable,
)


def test_statement_mask_tensor_round_trip():
    env = BooleanCubeEnvironment(n=3)
    progs = []
    names = []
    for var in range(env.n):
        for val in (0, 1):
            progs.append(env.literal_program(var=var, value=val))
            names.append(f"x{var}={val}")
    vocab = Vocabulary(env, programs=progs, names=names)

    s = vocab.statement(["x0=1", "x2=0"])
    t = statement_to_mask_tensor(s)
    s2 = statement_from_mask_tensor(vocab, t, validate=True)
    assert s2 == s


def test_vocabulary_tensor_round_trip_bitsets():
    env = FiniteEnvironment(states=[0, 1, 2, 3])
    p0 = Program.from_state_indices(env, [0, 1])
    p1 = Program.from_state_indices(env, [1, 2])
    p2 = Program.from_state_indices(env, [3])
    vocab = Vocabulary(env, programs=[p0, p1, p2], names=["p0", "p1", "p2"])

    t = vocabulary_to_tensor(vocab)
    vocab2 = vocabulary_from_tensor(env, t, names=["p0", "p1", "p2"], validate_uniqueness=True)

    assert [p.bitset for p in vocab2.programs] == [p.bitset for p in vocab.programs]


def test_cnf_string_round_trip():
    cnf = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1), Literal(1, 0)]),
            Clause.from_iterable([Literal(2, 1)]),
        ]
    )

    s = cnf_to_str(cnf)
    cnf2 = parse_cnf(s)
    assert cnf2 == cnf


def test_dnf_string_round_trip():
    dnf = DNF.from_iterable(
        [
            Term.from_iterable([Literal(0, 1), Literal(1, 0)]),
            Term.from_iterable([Literal(2, 1)]),
        ]
    )

    s = dnf_to_str(dnf)
    dnf2 = parse_dnf(s)
    assert dnf2 == dnf


def test_pretty_parses_constants_for_empty_cnf_and_dnf():
    # Empty CNF represents TRUE.
    cnf_true = CNF.from_iterable([])
    s_true = cnf_to_str(cnf_true)
    assert parse_cnf(s_true) == cnf_true

    # Empty DNF represents FALSE.
    dnf_false = DNF.from_iterable([])
    s_false = dnf_to_str(dnf_false)
    assert parse_dnf(s_false) == dnf_false


def test_dimacs_round_trip():
    cnf = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1), Literal(1, 0)]),
            Clause.from_iterable([Literal(2, 1)]),
        ]
    )

    dimacs = cnf_to_dimacs(cnf, n_vars=3)
    cnf2, n_vars2 = cnf_from_dimacs(dimacs)

    assert n_vars2 == 3
    assert cnf2 == cnf


def test_sympy_bridge_round_trip_cnf_and_dnf():
    import sympy as sp

    symbols = sp.symbols("x0:4")

    cnf = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1), Literal(2, 0)]),
            Clause.from_iterable([Literal(1, 1)]),
        ]
    )
    expr = cnf_to_sympy(cnf, n_vars=4, symbols=symbols)
    cnf2 = sympy_to_cnf(expr, symbols=symbols)
    assert cnf2 == cnf

    dnf = DNF.from_iterable(
        [
            Term.from_iterable([Literal(0, 1), Literal(1, 0)]),
            Term.from_iterable([Literal(3, 1)]),
        ]
    )
    expr2 = dnf_to_sympy(dnf, n_vars=4, symbols=symbols)
    dnf2 = sympy_to_dnf(expr2, symbols=symbols)
    assert dnf2 == dnf


def test_uint64_bitpack_round_trip_and_popcount():
    rng = random.Random(0)
    n = 137
    x = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)

    packed = pack_bool_tensor_uint64(x)
    y = unpack_bool_tensor_uint64(packed, n_bits=n)

    assert torch.equal(x, y)
    assert popcount_uint64(packed) == int(x.to(dtype=torch.int64).sum().item())


def test_sat_solver_finds_models_and_detects_unsat():
    # (x0) AND (~x0 OR x1) is satisfiable.
    cnf = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1)]),
            Clause.from_iterable([Literal(0, 0), Literal(1, 1)]),
        ]
    )
    res = sat_solve_cnf(cnf, n_vars=2, truth_table_max_vars=1)
    assert res.satisfiable
    assert res.model_state is not None
    # Check the model by extensional compilation.
    env = BooleanCubeEnvironment(n=2)
    prog = cnf.to_program(env)
    assert ((prog.bitset >> int(res.model_state)) & 1) == 1

    # (x0) AND (~x0) is unsatisfiable.
    cnf_unsat = CNF.from_iterable(
        [
            Clause.from_iterable([Literal(0, 1)]),
            Clause.from_iterable([Literal(0, 0)]),
        ]
    )
    assert not sat_is_satisfiable(cnf_unsat, n_vars=1, truth_table_max_vars=0)
