# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Genetic and evolutionary search learners.

These learners are heuristic.
They do not guarantee optimality.

They are useful when.

- the induced language is too large to enumerate
- you can evaluate correctness and proxies but cannot brute force

The implementation is intentionally simple and dependency free.
Users can swap in more advanced evolutionary optimisers later.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from stacktheory.layer1.vocabulary import Statement

from .evaluator import PolicyEvaluator
from .heuristics import KeyFn, PolicyStats, key_weakness_then_simplicity
from .tasks import Task


@dataclass(frozen=True)
class GeneticConfig:
    """Configuration for the genetic learner."""

    population_size: int = 128
    generations: int = 200
    elite_fraction: float = 0.1
    mutation_rate: float = 0.02
    crossover_rate: float = 0.7
    seed: int = 0


def _repair_to_language(evaluator: PolicyEvaluator, mask: int, rng: random.Random) -> int:
    """Repair a mask so it is in the induced language.

    Repair strategy

    If the mask is inconsistent, repeatedly drop a random included bit until
    the truth set is nonempty.

    This always terminates because the empty statement is in the induced
    language.
    """
    vocab = evaluator.vocab
    m = mask & ((1 << vocab.size) - 1)
    if evaluator.is_valid_mask(m):
        return m

    # Drop random bits until valid.
    while m != 0 and not evaluator.is_valid_mask(m):
        # Pick a random set bit to clear.
        bits = [i for i in range(vocab.size) if (m >> i) & 1]
        if not bits:
            break
        i = rng.choice(bits)
        m &= ~(1 << i)

    # Empty statement is always valid.
    if not evaluator.is_valid_mask(m):
        return 0
    return m


def _fitness(
    evaluator: PolicyEvaluator,
    mask: int,
    heuristic: KeyFn,
) -> Tuple[int, int, Tuple[int, ...]]:
    """Return a fitness tuple.

    Fitness order

    - Prefer correct policies over incorrect ones.
    - Among incorrect policies, prefer smaller symmetric difference between
      Ext_I ∩ Ext_pi and O.
    - Among correct policies, use the heuristic key.
    """
    counts = evaluator.correctness_counts(mask)
    correct = 1 if counts.is_correct else 0
    dist = counts.symmetric_difference

    if correct:
        w = evaluator.weakness(mask)
        d = int(mask).bit_count()
        key = heuristic(PolicyStats(mask=mask, weakness=w, description_length=d))
        return correct, -dist, key

    # For incorrect policies, do not spend time computing weakness.
    # Use a cheap key that prefers smaller masks for determinism.
    return correct, -dist, (-int(mask),)


def learn_genetic(
    task: Task,
    *,
    heuristic: KeyFn = key_weakness_then_simplicity,
    config: GeneticConfig = GeneticConfig(),
) -> Statement:
    """Heuristic genetic learner.

    Returns the best correct policy found.
    If no correct policy is found, it raises ValueError.

    Notes
    
    This learner can be made much faster by.

    - caching more intermediate values
    - using vectorised operations for mutation and crossover
    - adding problem specific repair operators
    """
    if config.population_size <= 0:
        raise ValueError("population_size must be positive")
    if config.generations <= 0:
        raise ValueError("generations must be positive")
    if not (0.0 < config.elite_fraction <= 1.0):
        raise ValueError("elite_fraction must be in (0,1]")
    if not (0.0 <= config.mutation_rate <= 1.0):
        raise ValueError("mutation_rate must be in [0,1]")
    if not (0.0 <= config.crossover_rate <= 1.0):
        raise ValueError("crossover_rate must be in [0,1]")

    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    rng = random.Random(config.seed)

    def random_valid_mask() -> int:
        # Construct a random valid statement by incremental addition.
        bits = vocab.env.all_states_bitset()
        mask = 0
        order = list(range(vocab.size))
        rng.shuffle(order)
        for j in order:
            if rng.random() < 0.5:
                new_bits = bits & vocab.programs[j].bitset
                if new_bits != 0:
                    bits = new_bits
                    mask |= 1 << j
        return mask

    population: List[int] = [_repair_to_language(evaluator, random_valid_mask(), rng) for _ in range(config.population_size)]

    best_mask: Optional[int] = None
    best_fit: Optional[Tuple[int, int, Tuple[int, ...]]] = None

    elite_n = max(1, int(round(config.elite_fraction * config.population_size)))

    for _gen in range(config.generations):
        scored = [(m, _fitness(evaluator, m, heuristic)) for m in population]
        scored.sort(key=lambda x: x[1], reverse=True)

        m0, fit0 = scored[0]
        if best_fit is None or fit0 > best_fit:
            best_fit = fit0
            best_mask = m0

        # Build next generation.
        elites = [m for m, _ in scored[:elite_n]]
        next_pop: List[int] = list(elites)

        while len(next_pop) < config.population_size:
            if rng.random() < config.crossover_rate and len(elites) >= 2:
                a = rng.choice(elites)
                b = rng.choice(elites)
                cut = rng.randrange(vocab.size)
                low_mask = (1 << cut) - 1
                child = (a & low_mask) | (b & ~low_mask)
            else:
                child = rng.choice(elites)

            # Mutate.
            if config.mutation_rate > 0:
                for i in range(vocab.size):
                    if rng.random() < config.mutation_rate:
                        child ^= 1 << i

            child = _repair_to_language(evaluator, child, rng)
            next_pop.append(child)

        population = next_pop

    if best_mask is None or not evaluator.is_correct(best_mask):
        raise ValueError("Genetic learner did not find a correct policy")
    return Statement(vocab, best_mask)
