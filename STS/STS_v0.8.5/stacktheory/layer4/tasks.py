# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Tasks over the induced language.

Layer 3 implements Stack Theory tasks for a fixed vocabulary v and its induced
language L_v.

Earlier prototypes in this repository experimented with restricting the
statement set directly.
That does not match the appendix definition of an induced language.
It also breaks basic closure properties such as downward closure.

This layer therefore keeps the task semantics unchanged.
A task is always defined over the full induced language L_v.

If you want a restricted candidate set Q for learning, pass it to the learner
functions in layer3.
Do not redefine L_v.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Iterable, List, Optional

from .language import Language


def _is_subset(a: int, b: int) -> bool:
    return (a & b) == a


@dataclass(frozen=True)
class EmbodiedTask:
    """A Stack Theory task defined over a Language.

    The Language in this layer is always the induced language L_v of its
    vocabulary.

    Parameters
    ----------
    language
        The induced language wrapper.
    inputs
        I_alpha as a set of statement masks in L_v.
    outputs
        O_alpha as a set of statement masks in Ext(I_alpha).

    Notes
    -----
    The appendix definition of a v-task requires.
    Every output extends at least one input.

    This implementation also enforces totality.
    Every input has at least one output that extends it.

    Both restrictions are convenient for experiments.
    They ensure that the task admits at least one correct output per input and
    avoids degenerate tasks where no correct policy exists.
    """

    language: Language
    inputs: FrozenSet[int]
    outputs: FrozenSet[int]

    def __post_init__(self) -> None:
        if len(self.inputs) == 0:
            raise ValueError("inputs must be non empty")
        if len(self.outputs) == 0:
            raise ValueError("outputs must be non empty")

        L = self.language

        for i in self.inputs:
            if not L.is_in_language_mask(i):
                raise ValueError("input is not a statement in the induced language")
        for o in self.outputs:
            if not L.is_in_language_mask(o):
                raise ValueError("output is not a statement in the induced language")

        # Definition v-task requires O_alpha subset Ext(I_alpha).
        for o in self.outputs:
            ok = any(_is_subset(i, o) for i in self.inputs)
            if not ok:
                raise ValueError("Each output must extend at least one input")

        # This suite additionally assumes tasks are total.
        for i in self.inputs:
            ok = any(_is_subset(i, o) for o in self.outputs)
            if not ok:
                raise ValueError("Task is not total")

    @property
    def vocab(self):
        return self.language.vocab

    def extension_of_inputs(self) -> FrozenSet[int]:
        """Return Ext(I_alpha) as a frozen set of masks."""
        out: set[int] = set()
        for i in self.inputs:
            out.update(self.language.extension_masks_of_mask(i))
        return frozenset(out)

    def is_correct_policy(
        self,
        policy_mask: int,
        *,
        ext_inputs: Optional[FrozenSet[int]] = None,
        require_policy_in_language: bool = True,
    ) -> bool:
        """Return True if and only if policy_mask is a correct policy.

        Implements.
        Ext(I_alpha) ∩ Ext(pi) = O_alpha.
        """
        if require_policy_in_language and not self.language.is_in_language_mask(policy_mask):
            raise ValueError("policy_mask must be a statement in the induced language")

        ExtI = self.extension_of_inputs() if ext_inputs is None else ext_inputs
        inter = frozenset(y for y in ExtI if _is_subset(policy_mask, y))
        return inter == self.outputs

    def correct_policies(
        self,
        *,
        candidates: Optional[Iterable[int]] = None,
        require_policy_in_language: bool = True,
    ) -> List[int]:
        """Return the list of correct policy masks among candidates.

        If candidates is None, this enumerates the full induced language L_v.
        """
        if candidates is None:
            candidates = self.language.iter_masks()

        extI = self.extension_of_inputs()
        out: List[int] = []
        for c in candidates:
            if require_policy_in_language and not self.language.is_in_language_mask(c):
                continue
            if self.is_correct_policy(c, ext_inputs=extI, require_policy_in_language=False):
                out.append(int(c))
        out.sort()
        return out
