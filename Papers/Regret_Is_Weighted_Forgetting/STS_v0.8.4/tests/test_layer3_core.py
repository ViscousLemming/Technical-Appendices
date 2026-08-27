# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Tests for Layer 3 tasks, heuristics, and learners.

These tests check the task correctness definition and that search based learners
return policies that satisfy the Stack Theory correctness condition.
"""

import pytest

from stacktheory.layer1.environment import FiniteEnvironment
from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Vocabulary, Statement

from stacktheory.layer3 import (
    Task,
    is_child_task,
    is_correct_policy,
    learn,
    learn_min_description_length,
    Proxy,
    key_weakness_then_simplicity,
    make_lexicographic_key,
    learn_best_first,
    learn_branch_and_bound,
    learn_beam_search,
    learn_random_search,
    learn_genetic,
    GeneticConfig,
    learn_hill_climb,
    learn_simulated_annealing,
    LocalSearchConfig,
)


def test_task_validation_and_child_relation():
    env = FiniteEnvironment(list(range(3)))
    p0 = Program.from_state_indices(env, [0, 1])
    r = Program.from_state_indices(env, [1, 2])
    vocab = Vocabulary(env, [p0, r], names=["p0", "r"])

    i = vocab.statement(["p0"])
    o = vocab.statement(["p0", "r"])

    alpha = Task.from_statements([i], [o])
    omega = Task.from_statements([i, vocab.statement(["r"])], [o])

    assert is_child_task(alpha, omega)
    assert not is_child_task(omega, alpha)

    with pytest.raises(ValueError):
        # Input has no extending output.
        Task.from_statements([i], [vocab.statement(["r"])])


def test_correct_policy_and_learning_under_proxies():
    env = FiniteEnvironment(list(range(3)))
    p0 = Program.from_state_indices(env, [0, 1])
    r = Program.from_state_indices(env, [1, 2])
    vocab = Vocabulary(env, [p0, r], names=["p0", "r"])

    i = vocab.statement(["p0"])
    o = vocab.statement(["p0", "r"])
    task = Task.from_statements([i], [o])

    pi_strict = o
    pi_weaker = vocab.statement(["r"])
    pi_wrong = vocab.statement(["p0"])

    assert is_correct_policy(task, pi_strict)
    assert is_correct_policy(task, pi_weaker)
    assert not is_correct_policy(task, pi_wrong)

    learned_w = learn(task, proxy=Proxy.WEAKNESS)
    assert learned_w == pi_weaker

    learned_d = learn(task, proxy=Proxy.DESCRIPTION_LENGTH)
    assert learned_d == pi_weaker

    # Custom heuristic use.
    learned_custom = learn(task, proxy=Proxy.WEAKNESS, heuristic=key_weakness_then_simplicity)
    assert learned_custom == pi_weaker

    # Exact minimal description length search.
    learned_mdl = learn_min_description_length(task)
    assert learned_mdl == pi_weaker

    # User supplied lexicographic key builder.
    key = make_lexicographic_key([("weakness", "max"), ("description_length", "min")])
    learned_lex = learn(task, heuristic=key)
    assert learned_lex == pi_weaker


def test_additional_learners_match_bruteforce_on_small_task():
    env = FiniteEnvironment(list(range(3)))
    p0 = Program.from_state_indices(env, [0, 1])
    r = Program.from_state_indices(env, [1, 2])
    vocab = Vocabulary(env, [p0, r], names=["p0", "r"])

    i = vocab.statement(["p0"])
    o = vocab.statement(["p0", "r"])
    task = Task.from_statements([i], [o])

    reference = learn(task, proxy=Proxy.WEAKNESS)
    assert reference == vocab.statement(["r"])

    assert learn_best_first(task) == reference
    assert learn_branch_and_bound(task) == reference
    assert learn_beam_search(task, beam_width=4, max_depth=2) == reference

    # Random search should find a correct policy with enough samples.
    assert learn_random_search(task, n_samples=256, seed=0) == reference

    # Genetic search should also find it with a fixed seed.
    cfg = GeneticConfig(population_size=64, generations=60, seed=0)
    assert learn_genetic(task, config=cfg) == reference

    # Local search learners should find it with fixed seeds.
    lcfg = LocalSearchConfig(steps=2000, restarts=2, seed=0)
    assert learn_hill_climb(task, config=lcfg) == reference
    assert learn_simulated_annealing(task, config=lcfg) == reference

