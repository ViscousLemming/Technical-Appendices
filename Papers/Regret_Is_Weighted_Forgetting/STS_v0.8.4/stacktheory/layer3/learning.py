# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Policy correctness and reference learners.

This module implements the task and learning definitions from the Stack Theory
appendices in an executable form.

The focus is exactness for finite vocabularies.
These reference algorithms are intentionally simple.

They are good for.

- regression tests
- small scale experiments
- validating faster learners
"""

from __future__ import annotations

from enum import Enum
from typing import Iterable, Optional, Union

from stacktheory.layer1.vocabulary import Statement, Vocabulary

from .tasks import Task
from .heuristics import PolicyStats, KeyFn, key_weakness_then_simplicity, key_simplicity_then_weakness


class Proxy(str, Enum):
    """Preference proxies used by learners."""

    WEAKNESS = "weakness"
    DESCRIPTION_LENGTH = "description_length"


def _default_key(proxy: Proxy) -> KeyFn:
    if proxy == Proxy.WEAKNESS:
        return key_weakness_then_simplicity
    if proxy == Proxy.DESCRIPTION_LENGTH:
        return key_simplicity_then_weakness
    raise ValueError("Unknown proxy")


def _as_mask(vocab: Vocabulary, policy: Union[int, Statement]) -> int:
    if isinstance(policy, int):
        m = int(policy)
        if m < 0 or m >= (1 << vocab.size):
            raise ValueError("policy mask contains bits above vocabulary size")
        if not vocab.is_in_language_mask(m):
            raise ValueError("policy mask is not a statement in the induced language")
        return m
    if isinstance(policy, Statement):
        if policy.vocab is not vocab:
            raise ValueError("policy statement does not belong to the task vocabulary")
        if not policy.is_valid():
            raise ValueError("policy must be a statement in the induced language")
        return policy.mask
    raise TypeError("policy must be an int mask or a Statement")


def extension_set(vocab: Vocabulary, statement_mask: int) -> frozenset[int]:
    """Return E_l as a frozen set of statement masks."""
    return frozenset(vocab.extension_masks_of_mask(statement_mask))


def extension_of_inputs(task: Task) -> frozenset[int]:
    """Return Ext_{I_alpha} as a frozen set of masks."""
    vocab = task.vocab
    out: set[int] = set()
    for i in task.inputs:
        out.update(vocab.extension_masks_of_mask(i))
    return frozenset(out)


def is_correct_policy(
    task: Task,
    policy: Union[int, Statement],
    *,
    ext_inputs: Optional[frozenset[int]] = None,
) -> bool:
    """Return True if and only if policy is a correct policy for task.

    Implements the condition.
    Ext_{I_alpha} intersect Ext_pi equals O_alpha.
    """
    vocab = task.vocab
    pi_mask = _as_mask(vocab, policy)
    ext_I = extension_of_inputs(task) if ext_inputs is None else ext_inputs

    # Correctness is defined in terms of completions.
    # Ext_pi = {y in L_v : pi ⊆ y}.
    # Ext_I is a subset of L_v by construction.
    # Therefore Ext_I ∩ Ext_pi = {y in Ext_I : pi ⊆ y}.
    inter = frozenset(y for y in ext_I if (pi_mask & y) == pi_mask)
    return inter == task.outputs


def correct_policies(
    task: Task,
    *,
    candidates: Optional[Iterable[Union[int, Statement]]] = None,
) -> list[Statement]:
    """Return the list of correct policies among candidates.

    If candidates is None, this enumerates the full induced language.
    Exact enumeration is only supported for small vocabularies.
    """
    vocab = task.vocab
    if candidates is None:
        candidates = (Statement(vocab, m) for m in vocab.induced_language_masks())

    ext_I = extension_of_inputs(task)

    out: list[Statement] = []
    for c in candidates:
        m = _as_mask(vocab, c)
        if is_correct_policy(task, m, ext_inputs=ext_I):
            out.append(Statement(vocab, m))

    out.sort(key=lambda s: s.mask)
    return out


def learn(
    task: Task,
    *,
    candidates: Optional[Iterable[Union[int, Statement]]] = None,
    proxy: Proxy = Proxy.WEAKNESS,
    heuristic: Optional[KeyFn] = None,
) -> Statement:
    """Learn a task under a given proxy.

    This is an exact brute force implementation.
    It enumerates candidates, filters to correct policies, and selects a
    proxy maximal one.

    Ties are broken deterministically.
    """
    vocab = task.vocab
    if candidates is None:
        candidates = (Statement(vocab, m) for m in vocab.induced_language_masks())

    ext_I = extension_of_inputs(task)

    key_fn = _default_key(proxy) if heuristic is None else heuristic

    best_mask: Optional[int] = None
    best_key: Optional[tuple] = None

    outputs = task.outputs

    for c in candidates:
        m = _as_mask(vocab, c)

        # Fast correctness check.
        # Compute Ext_I ∩ Ext_pi by filtering Ext_I using subset inclusion.
        inter = frozenset(y for y in ext_I if (m & y) == m)
        if inter != outputs:
            continue

        # Only compute weakness once correctness holds.
        w = vocab.weakness_of_mask(m)
        dlen = m.bit_count()
        stats = PolicyStats(mask=m, weakness=w, description_length=dlen)
        key = key_fn(stats)
        if best_key is None or key > best_key:
            best_key = key
            best_mask = m

    if best_mask is None:
        raise ValueError("No correct policy found in the candidate set")

    return Statement(vocab, best_mask)


def learn_min_description_length(
    task: Task,
    *,
    candidates: Optional[Iterable[Union[int, Statement]]] = None,
    heuristic: Optional[KeyFn] = None,
    max_description_length: Optional[int] = None,
) -> Statement:
    """Learn with minimal description length as a hard primary objective.

    This is an exact search strategy for the <_d proxy.
    It searches increasing statement size and stops at the first size that has
    any correct policy.

    Within the minimal size, selection uses the supplied heuristic.
    If heuristic is None, it uses simplicity then weakness.

    Notes
    -----
    If candidates is provided, this function falls back to brute force
    filtering over that candidate set.
    """
    vocab = task.vocab

    if candidates is not None:
        return learn(task, candidates=candidates, proxy=Proxy.DESCRIPTION_LENGTH, heuristic=heuristic)

    key_fn = key_simplicity_then_weakness if heuristic is None else heuristic

    m = vocab.size
    if max_description_length is None:
        max_description_length = m
    if max_description_length < 0 or max_description_length > m:
        raise ValueError("max_description_length out of range")

    prog_bits = vocab.program_bitsets()
    full_bits = vocab.env.all_states_bitset()

    ext_I = extension_of_inputs(task)

    best_mask: Optional[int] = None
    best_key: Optional[tuple] = None

    def rec(start: int, chosen: int, bits: int, chosen_count: int, target: int) -> None:
        nonlocal best_mask, best_key

        if chosen_count == target:
            inter = frozenset(y for y in ext_I if (chosen & y) == chosen)
            if inter != task.outputs:
                return

            w = vocab.weakness_of_mask(chosen)
            stats = PolicyStats(mask=chosen, weakness=w, description_length=chosen_count)
            key = key_fn(stats)
            if best_key is None or key > best_key:
                best_key = key
                best_mask = chosen
            return

        for j in range(start, m):
            new_bits = bits & prog_bits[j]
            if new_bits == 0:
                continue
            rec(j + 1, chosen | (1 << j), new_bits, chosen_count + 1, target)

    for target in range(0, max_description_length + 1):
        best_mask = None
        best_key = None
        rec(0, 0, full_bits, 0, target)
        if best_mask is not None:
            return Statement(vocab, best_mask)

    raise ValueError("No correct policy found up to max_description_length")
