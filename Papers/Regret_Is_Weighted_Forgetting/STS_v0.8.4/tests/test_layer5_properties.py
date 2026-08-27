# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Property tests for algebraic laws.

These tests check basic algebraic identities and representation invariance.
They help ensure we did not break set logic laws while optimising.
"""

import random

import torch

from stacktheory.layer1.environment import BooleanCubeEnvironment, IndexEnvironment
from stacktheory.layer1.program import Program
from stacktheory.layer1.logic import Clause, CNF, Literal

from stacktheory.layer1.bitset import pack_bool_tensor, unpack_bool_tensor
from stacktheory.layer2.wordbitset import WordBitset
from stacktheory.layer2.sat import sat_solve_cnf, sat_solve_cnf_lns, LNSConfig


def test_program_algebraic_laws_randomised():
    rng = random.Random(0)
    env = IndexEnvironment(n_states=128)

    for _ in range(200):
        a = Program(env, rng.getrandbits(env.size))
        b = Program(env, rng.getrandbits(env.size))
        c = Program(env, rng.getrandbits(env.size))

        # Commutativity.
        assert (a & b) == (b & a)
        assert (a | b) == (b | a)

        # Associativity.
        assert ((a & b) & c) == (a & (b & c))
        assert ((a | b) | c) == (a | (b | c))

        # Identities.
        empty = Program(env, 0)
        full = Program(env, env.all_states_bitset())
        assert (a & full) == a
        assert (a | empty) == a

        # De Morgan.
        assert ~(a & b) == ((~a) | (~b))
        assert ~(a | b) == ((~a) & (~b))

        # Difference and symmetric difference.
        assert (a - b) == (a & ~b)
        assert (a ^ b) == ((a - b) | (b - a))

        # Subset characterisation.
        assert a.issubset(b) == ((a & ~b).is_empty())


def test_packing_unpacking_equivalence():
    rng = random.Random(1)
    n = 130
    x = torch.tensor([rng.randrange(2) for _ in range(n)], dtype=torch.bool)

    bitset = pack_bool_tensor(x)
    y = unpack_bool_tensor(bitset, n_bits=n)

    assert torch.equal(x.to(dtype=torch.bool), y.to(dtype=torch.bool))

    wb = WordBitset.from_bool_tensor(x)
    assert wb.to_bitset_int() == bitset
    assert torch.equal(wb.to_bool_tensor(), x)


def test_boolean_cube_literal_masks_agree_with_semantics():
    rng = random.Random(2)
    n = 8
    env = BooleanCubeEnvironment(n=n)

    for _ in range(50):
        var = rng.randrange(n)
        value = rng.randrange(2)
        bits = env.literal_program_mask(var=var, value=value)

        for state in range(env.size):
            in_set = ((bits >> state) & 1) == 1
            sem = ((state >> var) & 1) == value
            assert in_set == sem


def _truth_table_sat(cnf: CNF, n_vars: int) -> bool:
    env = BooleanCubeEnvironment(n=n_vars)
    prog = cnf.to_program(env)
    return not prog.is_empty()


def test_internal_dpll_solver_matches_truth_table_for_small_n():
    rng = random.Random(3)
    n_vars = 10

    for _ in range(25):
        n_clauses = rng.randrange(1, 30)
        clauses = []
        for _c in range(n_clauses):
            width = rng.randrange(1, 4)
            lits = []
            for _k in range(width):
                var = rng.randrange(n_vars)
                value = rng.randrange(2)
                lits.append(Literal(var=var, value=value))
            clauses.append(Clause.from_iterable(lits))
        cnf = CNF.from_iterable(clauses)

        # Force DPLL by setting truth_table_max_vars to 0.
        res = sat_solve_cnf(cnf, n_vars=n_vars, truth_table_max_vars=0)
        expected = _truth_table_sat(cnf, n_vars)

        assert res.satisfiable == expected
        if res.satisfiable:
            assert res.model_state is not None
            assert cnf.satisfied_by(int(res.model_state))


def test_lns_solver_never_lies_when_it_claims_sat():
    rng = random.Random(4)
    n_vars = 40

    clauses = []
    for _ in range(120):
        width = 3
        lits = []
        for _k in range(width):
            var = rng.randrange(n_vars)
            value = rng.randrange(2)
            lits.append(Literal(var=var, value=value))
        clauses.append(Clause.from_iterable(lits))
    cnf = CNF.from_iterable(clauses)

    h = sat_solve_cnf_lns(cnf, n_vars=n_vars, config=LNSConfig(free_vars=18, max_iters=200, restarts=8))
    if h.status == "sat":
        assert h.model_state is not None
        assert cnf.satisfied_by(int(h.model_state))
