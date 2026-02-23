# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Local search learners.

These learners are heuristic.
They treat policy learning as an optimisation problem over statement masks.

They are designed for medium sized vocabularies where.

- brute force enumeration is too expensive
- you can evaluate correctness and weakness fast enough

The basic fitness signal is.

- correct policies beat incorrect policies
- among incorrect policies, smaller symmetric difference between Ext_I ∩ Ext_pi and O is better
- among correct policies, use the supplied heuristic key
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from stacktheory.layer1.vocabulary import Statement

from .evaluator import PolicyEvaluator
from .genetic import _repair_to_language
from .heuristics import KeyFn, PolicyStats, key_weakness_then_simplicity
from .tasks import Task


@dataclass(frozen=True)
class LocalSearchConfig:
    steps: int = 5000
    seed: int = 0
    restarts: int = 4
    initial_temperature: float = 1.0
    temperature_decay: float = 0.999


def _fitness_tuple(evaluator: PolicyEvaluator, mask: int, heuristic: KeyFn) -> Tuple[int, int, Tuple[int, ...]]:
    counts = evaluator.correctness_counts(mask)
    correct = 1 if counts.is_correct else 0
    dist = counts.symmetric_difference
    if correct:
        w = evaluator.weakness(mask)
        d = int(mask).bit_count()
        key = heuristic(PolicyStats(mask=mask, weakness=w, description_length=d))
        return correct, -dist, key
    return correct, -dist, (-int(mask),)


def learn_hill_climb(
    task: Task,
    *,
    heuristic: KeyFn = key_weakness_then_simplicity,
    config: LocalSearchConfig = LocalSearchConfig(),
) -> Statement:
    """Greedy hill climbing with random restarts.

    This learner tries to maximise the fitness tuple defined above.
    It returns the best correct policy found.
    """
    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    rng = random.Random(config.seed)

    best_mask: Optional[int] = None
    best_fit: Optional[Tuple[int, int, Tuple[int, ...]]] = None

    for _restart in range(max(1, int(config.restarts))):
        mask = _repair_to_language(evaluator, rng.randrange(1 << vocab.size), rng)
        fit = _fitness_tuple(evaluator, mask, heuristic)

        for _ in range(max(1, int(config.steps))):
            i = rng.randrange(vocab.size)
            cand = mask ^ (1 << i)
            cand = _repair_to_language(evaluator, cand, rng)
            cand_fit = _fitness_tuple(evaluator, cand, heuristic)
            if cand_fit > fit:
                mask = cand
                fit = cand_fit

        if best_fit is None or fit > best_fit:
            best_fit = fit
            best_mask = mask

    if best_mask is None or not evaluator.is_correct(best_mask):
        raise ValueError("Hill climb learner did not find a correct policy")
    return Statement(vocab, best_mask)


def learn_simulated_annealing(
    task: Task,
    *,
    heuristic: KeyFn = key_weakness_then_simplicity,
    config: LocalSearchConfig = LocalSearchConfig(),
) -> Statement:
    """Simulated annealing over masks.

    This is often more robust than pure hill climbing.
    It can accept worse moves early and cools over time.
    """
    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    rng = random.Random(config.seed)

    best_mask: Optional[int] = None
    best_fit: Optional[Tuple[int, int, Tuple[int, ...]]] = None

    for _restart in range(max(1, int(config.restarts))):
        mask = _repair_to_language(evaluator, rng.randrange(1 << vocab.size), rng)
        fit = _fitness_tuple(evaluator, mask, heuristic)

        temperature = float(config.initial_temperature)
        decay = float(config.temperature_decay)
        if decay <= 0 or decay >= 1:
            decay = 0.999

        for _ in range(max(1, int(config.steps))):
            i = rng.randrange(vocab.size)
            cand = mask ^ (1 << i)
            cand = _repair_to_language(evaluator, cand, rng)
            cand_fit = _fitness_tuple(evaluator, cand, heuristic)

            if cand_fit > fit:
                accept = True
            else:
                # Accept worse moves with probability exp(delta / T).
                # We collapse the lexicographic tuple to a scalar proxy.
                delta = (cand_fit[0] - fit[0]) * 10.0 + (cand_fit[1] - fit[1])
                accept = rng.random() < math.exp(delta / max(1e-9, temperature))

            if accept:
                mask = cand
                fit = cand_fit

            temperature *= decay

        if best_fit is None or fit > best_fit:
            best_fit = fit
            best_mask = mask

    if best_mask is None or not evaluator.is_correct(best_mask):
        raise ValueError("Simulated annealing learner did not find a correct policy")
    return Statement(vocab, best_mask)
