# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Tests for Layer 4 scaling utilities.

Layer 4 adds faster backends and scaling helpers.
These tests check that the Layer 4 wrappers match Layer 1 semantics on small cases.
"""

from stacktheory.layer1 import BooleanCubeEnvironment, IndexEnvironment, Program, Vocabulary
from stacktheory.layer4 import Language, EmbodiedTask, EmbodiedPolicyEvaluator
from stacktheory.layer1.logic import Clause, Literal, CNF
from stacktheory.layer2 import LNSConfig, sat_solve_cnf_lns, sat_solve_cnf


def test_index_environment_basic_mapping():
    env = IndexEnvironment(n_states=5)
    assert env.size == 5
    assert list(env.iter_states()) == [0, 1, 2, 3, 4]
    for s in range(5):
        assert env.state_to_index(s) == s

    p = Program.from_state_indices(env, [0, 2, 4])
    assert p.cardinality() == 3


def test_language_induced_matches_vocabulary():
    env = BooleanCubeEnvironment(n=3)
    progs = [env.literal_program(var=0, value=1), env.literal_program(var=1, value=1)]
    vocab = Vocabulary(env, programs=progs, names=["x0=1", "x1=1"])

    L = Language(vocab)
    masks = [0, 1, 2, 3]
    for m in masks:
        assert L.is_in_language_mask(m) == vocab.is_in_language_mask(m)
        if L.is_in_language_mask(m):
            assert L.weakness_of_mask(m) == vocab.weakness_of_mask(m)
            assert set(L.extension_masks_of_mask(m)) == set(vocab.extension_masks_of_mask(m))


def test_completion_helpers_return_empty_on_unsatisfiable_subset():
    env = BooleanCubeEnvironment(n=1)
    p0 = env.literal_program(var=0, value=1)
    p1 = env.literal_program(var=0, value=0)
    vocab = Vocabulary(env, programs=[p0, p1], names=["x0=1", "x0=0"])
    L = Language(vocab)

    # The conjunction (x0=1) AND (x0=0) is unsatisfiable so it is not in L_v.
    assert vocab.is_in_language_mask(3) is False
    assert L.is_in_language_mask(3) is False

    # Convenience completion helpers must still behave sensibly.
    assert L.completion_masks_for_subset_mask(3) == []
    assert L.completion_count_for_subset_mask(3) == 0


def test_embodied_task_correct_policies():
    env = BooleanCubeEnvironment(n=2)
    p0 = env.literal_program(var=0, value=1)
    p1 = env.literal_program(var=1, value=1)
    vocab = Vocabulary(env, programs=[p0, p1], names=["x0=1", "x1=1"])
    L = Language(vocab)

    # Inputs: {x0=1}. Outputs: {x0=1 and x1=1}.
    task = EmbodiedTask(language=L, inputs=frozenset({1}), outputs=frozenset({3}))
    ev = EmbodiedPolicyEvaluator(task)

    # Policy {x0=1, x1=1} is correct.
    assert ev.is_correct(3)
    # Policy {x0=1} is not correct because it allows completion 1.
    assert not ev.is_correct(1)

    # Policy {x1=1} is also correct because it intersects Ext_I in exactly {3}.
    cps = task.correct_policies()
    assert cps == [2, 3]


def test_sat_lns_returns_model_or_unknown_but_never_unsat():
    # Empty CNF is satisfiable for any n_vars.
    cnf = CNF.from_iterable([])
    res = sat_solve_cnf_lns(cnf, n_vars=25, config=LNSConfig(free_vars=20, max_iters=1, restarts=1, seed=0))
    assert res.status in ("sat", "unknown")
    assert res.status == "sat"
    assert res.model_state is not None


def test_sat_solve_cnf_lns_first_is_sound():
    # (x0) AND (x1) AND ... AND (x21) is satisfiable.
    clauses = [Clause.from_iterable([Literal(i, 1)]) for i in range(22)]
    cnf = CNF.from_iterable(clauses)
    res = sat_solve_cnf(
        cnf,
        n_vars=22,
        truth_table_max_vars=20,
        lns_first=True,
        lns_config=LNSConfig(free_vars=20, max_iters=200, restarts=10, seed=0),
    )
    assert res.satisfiable
    assert res.model_state is not None
    # Verify by evaluating the clauses.
    state = int(res.model_state)
    for i in range(22):
        assert ((state >> i) & 1) == 1
