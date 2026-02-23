# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Tasks and task relations.

This module implements Stack Theory tasks for a fixed vocabulary.

A task is a pair alpha = <I_alpha, O_alpha> where I_alpha and O_alpha are
subsets of the induced language L_v.

Well formedness

The appendix definition of a v-task requires.
O_alpha is a subset of Ext(I_alpha).
Equivalently.
Every output statement must extend at least one admissible input statement.

Some results additionally assume the task is total.
Totality means every input has at least one output that extends it.
This implementation does not require totality, but provides a helper to test it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, Optional

from stacktheory.layer1.vocabulary import Statement, Vocabulary


def _mask_is_subset(a: int, b: int) -> bool:
    """Return True if and only if a is a subset of b as sets of vocabulary elements."""
    return (a & b) == a


@dataclass(frozen=True)
class Task:
    """A Stack Theory task for a fixed vocabulary.

    The internal representation stores statement masks.
    All masks are interpreted relative to the same vocabulary instance.
    """

    vocab: Vocabulary
    inputs: FrozenSet[int]
    outputs: FrozenSet[int]

    def __post_init__(self) -> None:
        # Validate all masks are within range and in the induced language.
        for m in self.inputs:
            self._validate_statement_mask(m, label="input")
        for m in self.outputs:
            self._validate_statement_mask(m, label="output")

        # Definition v-task requires O_alpha subset Ext(I_alpha).
        # If there are no inputs then Ext(I_alpha) is empty, so outputs must be empty.
        if len(self.inputs) == 0:
            if len(self.outputs) != 0:
                raise ValueError("Task has no inputs so outputs must be empty")
            return

        for o in self.outputs:
            ok = any(_mask_is_subset(i, o) for i in self.inputs)
            if not ok:
                raise ValueError("Each output must extend at least one input")

    def is_total(self) -> bool:
        """Return True if and only if the task is total.

        A task is total when every input i has at least one output o with i subset o.
        """
        if len(self.inputs) == 0:
            return True
        return all(any(_mask_is_subset(i, o) for o in self.outputs) for i in self.inputs)

    def _validate_statement_mask(self, m: int, *, label: str) -> None:
        if not isinstance(m, int):
            raise TypeError(f"{label} masks must be integers")
        if m < 0 or m >= (1 << self.vocab.size):
            raise ValueError(f"{label} mask contains bits above vocabulary size")
        if not self.vocab.is_in_language_mask(m):
            raise ValueError(f"{label} mask is not a statement in the induced language")

    @classmethod
    def from_statements(
        cls,
        inputs: Iterable[Statement],
        outputs: Iterable[Statement],
        *,
        vocab: Optional[Vocabulary] = None,
    ) -> "Task":
        """Create a Task from Statement objects.

        If vocab is not provided, it is inferred from the first input or output.
        """
        inputs_list = list(inputs)
        outputs_list = list(outputs)

        if vocab is None:
            if inputs_list:
                vocab = inputs_list[0].vocab
            elif outputs_list:
                vocab = outputs_list[0].vocab
            else:
                raise ValueError("Cannot infer vocab from empty inputs and outputs")

        for s in inputs_list + outputs_list:
            if s.vocab is not vocab:
                raise ValueError("All statements must belong to the same vocabulary instance")

        return cls(
            vocab=vocab,
            inputs=frozenset(s.mask for s in inputs_list),
            outputs=frozenset(s.mask for s in outputs_list),
        )

    def input_statements(self) -> list[Statement]:
        return [Statement(self.vocab, m) for m in sorted(self.inputs)]

    def output_statements(self) -> list[Statement]:
        return [Statement(self.vocab, m) for m in sorted(self.outputs)]


def is_child_task(alpha: Task, omega: Task) -> bool:
    """Return True if and only if alpha is a child of omega.

    alpha is a child of omega when.

    - I_alpha is a strict subset of I_omega
    - O_alpha is a subset of O_omega
    """
    if alpha.vocab is not omega.vocab:
        raise ValueError("Tasks must share the same vocabulary instance")
    return (alpha.inputs < omega.inputs) and (alpha.outputs <= omega.outputs)
