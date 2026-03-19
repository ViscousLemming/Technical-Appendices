# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Search based learning algorithms.

Layer 3 learning is defined as.

Pick a proxy maximal correct policy pi from a candidate set Q.

The brute force reference learner enumerates Q and is correct by inspection.
That is valuable for tests and small problems.

This module adds additional learners that are still aligned to the appendix
definitions but can cut work for many tasks.

All algorithms use the canonical Stack Theory semantics from Layer 1.
They never relax the definition of.

- induced language membership
- extensions
- weakness
- correctness
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from stacktheory.layer1.vocabulary import Statement, Vocabulary

from .heuristics import KeyFn, PolicyStats, key_weakness_then_simplicity
from .tasks import Task
from .evaluator import PolicyEvaluator


def _negate_key(key: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(-int(x) for x in key)


@dataclass(frozen=True)
class SearchStats:
    """Simple accounting information for search based learners."""

    expanded: int
    generated: int


def learn_best_first(
    task: Task,
    *,
    heuristic: KeyFn = key_weakness_then_simplicity,
    max_expansions: Optional[int] = None,
    assume_monotone_key: bool = True,
    rng: Optional[random.Random] = None,
) -> Statement:
    """Best first search over the induced language.

    The search space is the induced language L_v.
    Nodes are statements represented by masks.
    Children are formed by adding one more vocabulary element.

    The priority queue orders nodes by the heuristic key.
    If assume_monotone_key is True, the search stops as soon as the best
    remaining candidate cannot beat the best correct policy found so far.
    This stop rule is correct for monotone lexicographic keys built from.

    - weakness maximisation
    - description length minimisation

    For arbitrary custom heuristics, set assume_monotone_key False.
    """
    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    prog_bits = vocab.program_bitsets()
    full_bits = vocab.env.all_states_bitset()

    if rng is None:
        rng = random.Random(0)

    best_mask: Optional[int] = None
    best_key: Optional[Tuple[int, ...]] = None

    # Heap items are (neg_key, tie_break, mask, bits, last_index).
    heap: List[Tuple[Tuple[int, ...], int, int, int, int]] = []

    def push(mask: int, bits: int, last_index: int) -> None:
        nonlocal heap
        w = evaluator.weakness(mask)
        d = int(mask).bit_count()
        key = heuristic(PolicyStats(mask=mask, weakness=w, description_length=d))
        neg_key = _negate_key(key)
        # tie_break randomises among equal keys to avoid pathological expansion order.
        tie = rng.randrange(1 << 30)
        heapq.heappush(heap, (neg_key, tie, mask, bits, last_index))

    # Start from empty statement.
    push(0, full_bits, -1)

    expanded = 0
    generated = 1

    while heap:
        if max_expansions is not None and expanded >= max_expansions:
            break

        neg_key, _tie, mask, bits, last = heapq.heappop(heap)
        key = tuple(-x for x in neg_key)
        expanded += 1

        if best_key is not None and assume_monotone_key:
            # Since we pop in descending key order, no future candidate can beat best_key.
            if key <= best_key:
                break

        if evaluator.is_correct(mask):
            if best_key is None or key > best_key:
                best_key = key
                best_mask = mask

        # Expand children.
        for j in range(last + 1, vocab.size):
            new_bits = bits & prog_bits[j]
            if new_bits == 0:
                continue
            new_mask = mask | (1 << j)
            push(new_mask, new_bits, j)
            generated += 1

    if best_mask is None:
        raise ValueError("No correct policy found within the search budget")
    return Statement(vocab, best_mask)


def learn_branch_and_bound(
    task: Task,
    *,
    heuristic: KeyFn = key_weakness_then_simplicity,
    max_nodes: Optional[int] = None,
    assume_monotone_key: bool = True,
) -> Statement:
    """Depth first search with pruning.

    The prune rule is safe when assume_monotone_key is True and the key
    decreases when you add vocabulary elements.
    """
    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    prog_bits = vocab.program_bitsets()
    full_bits = vocab.env.all_states_bitset()

    best_mask: Optional[int] = None
    best_key: Optional[Tuple[int, ...]] = None
    visited = 0

    def node_key(mask: int) -> Tuple[int, ...]:
        w = evaluator.weakness(mask)
        d = int(mask).bit_count()
        return heuristic(PolicyStats(mask=mask, weakness=w, description_length=d))

    def rec(start: int, mask: int, bits: int) -> None:
        nonlocal best_mask, best_key, visited
        if max_nodes is not None and visited >= max_nodes:
            return
        visited += 1

        # Check correctness.
        if evaluator.is_correct(mask):
            key = node_key(mask)
            if best_key is None or key > best_key:
                best_key = key
                best_mask = mask

        # Prune if nothing below can beat best_key.
        if best_key is not None and assume_monotone_key:
            key = node_key(mask)
            if key <= best_key:
                return

        for j in range(start, vocab.size):
            new_bits = bits & prog_bits[j]
            if new_bits == 0:
                continue
            rec(j + 1, mask | (1 << j), new_bits)

    rec(0, 0, full_bits)

    if best_mask is None:
        raise ValueError("No correct policy found within the node budget")
    return Statement(vocab, best_mask)


def learn_beam_search(
    task: Task,
    *,
    beam_width: int = 32,
    heuristic: KeyFn = key_weakness_then_simplicity,
    max_depth: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Statement:
    """Beam search over induced language.

    Beam search is not guaranteed to find an optimal policy.
    It is intended as a scalable heuristic learner.
    """
    if beam_width <= 0:
        raise ValueError("beam_width must be positive")
    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    prog_bits = vocab.program_bitsets()
    full_bits = vocab.env.all_states_bitset()

    if rng is None:
        rng = random.Random(0)

    if max_depth is None:
        max_depth = vocab.size
    if max_depth < 0 or max_depth > vocab.size:
        raise ValueError("max_depth out of range")

    # Each element is (mask, bits, last_index).
    beam: List[Tuple[int, int, int]] = [(0, full_bits, -1)]

    best_mask: Optional[int] = None
    best_key: Optional[Tuple[int, ...]] = None

    def key_of(mask: int) -> Tuple[int, ...]:
        w = evaluator.weakness(mask)
        d = int(mask).bit_count()
        return heuristic(PolicyStats(mask=mask, weakness=w, description_length=d))

    for _depth in range(max_depth + 1):
        # Check current beam for correct policies.
        for mask, _bits, _last in beam:
            if evaluator.is_correct(mask):
                key = key_of(mask)
                if best_key is None or key > best_key:
                    best_key = key
                    best_mask = mask

        # Expand.
        candidates: List[Tuple[Tuple[int, ...], int, int, int, int]] = []
        for mask, bits, last in beam:
            for j in range(last + 1, vocab.size):
                new_bits = bits & prog_bits[j]
                if new_bits == 0:
                    continue
                new_mask = mask | (1 << j)
                key = key_of(new_mask)
                tie = rng.randrange(1 << 30)
                candidates.append((key, tie, new_mask, new_bits, j))

        if not candidates:
            break

        # Keep top beam_width by key.
        candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
        top = candidates[:beam_width]
        new_beam: List[Tuple[int, int, int]] = []
        for _key, _tie, new_mask, new_bits, last in top:
            # new_bits is already the truth set bitset for new_mask.
            new_beam.append((new_mask, new_bits, last))
        beam = new_beam

        if not beam:
            break

    if best_mask is None:
        raise ValueError("No correct policy found in the beam search budget")
    return Statement(vocab, best_mask)


def learn_random_search(
    task: Task,
    *,
    heuristic: KeyFn = key_weakness_then_simplicity,
    n_samples: int = 1000,
    seed: int = 0,
) -> Statement:
    """Random search baseline.

    Samples random statement masks and returns the best correct one found.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    rng = random.Random(seed)

    vocab = task.vocab
    evaluator = PolicyEvaluator(task)
    best_mask: Optional[int] = None
    best_key: Optional[Tuple[int, ...]] = None

    for _ in range(n_samples):
        mask = rng.randrange(1 << vocab.size)
        if not evaluator.is_valid_mask(mask):
            continue
        if not evaluator.is_correct(mask):
            continue
        w = evaluator.weakness(mask)
        d = int(mask).bit_count()
        key = heuristic(PolicyStats(mask=mask, weakness=w, description_length=d))
        if best_key is None or key > best_key:
            best_key = key
            best_mask = mask

    if best_mask is None:
        raise ValueError("No correct policy found in the random sample set")
    return Statement(vocab, best_mask)
