# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Heuristics and preference orderings for learning.

Stack Theory defines learning in terms of a proxy preference relation < on a
candidate set Q.

In code we often want a deterministic total order for selection.
This module standardises that in a way that is faithful to the appendix
definitions.

Core idea

- A policy is a statement pi in the induced language L_v.
- A proxy prefers some policies over others.
- We compute simple stats for a candidate policy.
- A heuristic maps those stats to a tuple key.
- Larger keys are preferred.

Lexicographic keys give a clean way to combine weakness and simplicity with a
priority order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Literal, Sequence, Tuple


@dataclass(frozen=True)
class PolicyStats:
    """Summary statistics for a policy statement.

    mask
        Integer mask representation of the statement over the vocabulary.
    weakness
        Weakness w(pi) which is |E_pi|.
    description_length
        Description length proxy |pi| which is the number of selected
        vocabulary elements.
    """

    mask: int
    weakness: int
    description_length: int


KeyFn = Callable[[PolicyStats], Tuple[int, ...]]

MetricName = Literal[
    "weakness",
    "description_length",
    "mask",
]

Direction = Literal[
    "max",
    "min",
]


def key_weakness_then_simplicity(stats: PolicyStats) -> Tuple[int, ...]:
    """Prefer larger weakness.

    Tie break by shorter description length.
    Tie break deterministically by smaller mask.
    """
    return (stats.weakness, -stats.description_length, -stats.mask)


def key_simplicity_then_weakness(stats: PolicyStats) -> Tuple[int, ...]:
    """Prefer shorter description length.

    Tie break by larger weakness.
    Tie break deterministically by smaller mask.
    """
    return (-stats.description_length, stats.weakness, -stats.mask)


def make_lexicographic_key(order: Sequence[Tuple[MetricName, Direction]]) -> KeyFn:
    """Build a lexicographic key function.

    Parameters
    ----------
    order
        A sequence of (metric_name, direction) pairs.
        direction is either "max" or "min".

    Returns
    -------
    A key function that maps PolicyStats to a tuple.
    Larger tuples are preferred.

    Notes
    -----
    If "mask" is not included, the key appends a deterministic tie break
    term that prefers smaller masks.
    """

    if len(order) == 0:
        raise ValueError("order must be nonempty")

    valid_metrics = {"weakness", "description_length", "mask"}
    for name, direction in order:
        if name not in valid_metrics:
            raise ValueError("Unknown metric name")
        if direction not in ("max", "min"):
            raise ValueError("Unknown direction")

    include_mask = any(name == "mask" for name, _ in order)

    def key(stats: PolicyStats) -> Tuple[int, ...]:
        out: List[int] = []
        for name, direction in order:
            val = getattr(stats, name)
            out.append(val if direction == "max" else -val)
        if not include_mask:
            out.append(-stats.mask)
        return tuple(out)

    return key
