# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Policy evaluation helpers for learning algorithms.

Layer 3 defines learning in terms of.

- a task alpha = <I_alpha, O_alpha>
- a candidate set Q of policies pi in L_v
- a proxy preference relation <

Many learning algorithms need repeated evaluation of.

- policy correctness
- weakness w(pi) = |E_pi|
- description length |pi|

This module provides a cached evaluator that computes these quantities in a
way that matches the appendix definitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from stacktheory.layer1.vocabulary import Vocabulary

from .tasks import Task


@dataclass(frozen=True)
class CorrectnessCounts:
    """Counts derived from Ext_I ∩ Ext_pi compared to O.

    in_outputs
        Number of elements that are both in the intersection and in outputs.
    false_positives
        Number of elements in the intersection that are not in outputs.
    false_negatives
        Number of outputs not present in the intersection.
    """

    in_outputs: int
    false_positives: int
    false_negatives: int

    @property
    def symmetric_difference(self) -> int:
        return self.false_positives + self.false_negatives

    @property
    def is_correct(self) -> bool:
        return (self.false_positives == 0) and (self.false_negatives == 0)


@dataclass(frozen=True)
class PolicyEval:
    """Evaluation record for a policy mask."""

    mask: int
    is_valid: bool
    description_length: int
    weakness: Optional[int]
    correctness: Optional[CorrectnessCounts]

    @property
    def is_correct(self) -> bool:
        return bool(self.correctness is not None and self.correctness.is_correct)


class PolicyEvaluator:
    """Cacheable evaluator for a fixed task.

    The evaluator caches.

- correctness counts
- weakness

It is safe because vocab and task are treated as immutable.
"""

    def __init__(self, task: Task):
        self.task = task
        self.vocab: Vocabulary = task.vocab
        # Ext_I is used by every correctness check.
        from .learning import extension_of_inputs

        self.ext_I = extension_of_inputs(task)
        self.outputs = task.outputs

        self._correct_cache: Dict[int, CorrectnessCounts] = {}
        self._weakness_cache: Dict[int, int] = {}

    def is_valid_mask(self, mask: int) -> bool:
        if mask < 0 or mask >= (1 << self.vocab.size):
            return False
        return self.vocab.is_in_language_mask(mask)

    def description_length(self, mask: int) -> int:
        return int(mask).bit_count()

    def correctness_counts(self, mask: int) -> CorrectnessCounts:
        """Return correctness counts for a valid policy mask."""
        if mask in self._correct_cache:
            return self._correct_cache[mask]

        outputs = self.outputs
        in_outputs = 0
        false_positives = 0

        for y in self.ext_I:
            if (mask & y) == mask:
                if y in outputs:
                    in_outputs += 1
                else:
                    false_positives += 1

        false_negatives = len(outputs) - in_outputs
        counts = CorrectnessCounts(
            in_outputs=in_outputs,
            false_positives=false_positives,
            false_negatives=false_negatives,
        )
        self._correct_cache[mask] = counts
        return counts

    def is_correct(self, mask: int) -> bool:
        return self.correctness_counts(mask).is_correct

    def weakness(self, mask: int) -> int:
        """Return w(mask) for a valid statement mask."""
        if mask in self._weakness_cache:
            return self._weakness_cache[mask]
        w = self.vocab.weakness_of_mask(mask)
        self._weakness_cache[mask] = w
        return w

    def evaluate(self, mask: int, *, compute_weakness: bool = True) -> PolicyEval:
        """Evaluate a mask.

        If the mask is not a statement in L_v, the record has is_valid False.
        """
        dlen = self.description_length(mask)
        if not self.is_valid_mask(mask):
            return PolicyEval(
                mask=mask,
                is_valid=False,
                description_length=dlen,
                weakness=None,
                correctness=None,
            )

        counts = self.correctness_counts(mask)
        w = self.weakness(mask) if compute_weakness else None
        return PolicyEval(
            mask=mask,
            is_valid=True,
            description_length=dlen,
            weakness=w,
            correctness=counts,
        )
