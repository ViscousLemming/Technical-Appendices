# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Cached evaluation helpers for embodied tasks.

Layer 4 introduces embodied languages and embodied tasks.
The logic is the same as in Layer 3.
Only the backend that answers language queries changes.

This file mirrors layer3.evaluator.PolicyEvaluator.
It caches repeated computations of.

- validity of a policy statement
- weakness
- correctness against a fixed task

Math mapping

- A policy is a statement mask pi in the induced language of an embodied language.
- Ext(I) is the set of completions of every input statement.
- pi is correct when Ext(I) intersect Ext(pi) equals O.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from .tasks import EmbodiedTask


def _is_subset(a: int, b: int) -> bool:
    return (a & ~b) == 0


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


class EmbodiedPolicyEvaluator:
    """Cacheable evaluator for a fixed embodied task."""

    def __init__(self, task: EmbodiedTask):
        self.task = task
        self.language = task.language
        self.ext_I = task.extension_of_inputs()
        self.outputs = task.outputs

        self._correct_cache: Dict[int, CorrectnessCounts] = {}
        self._weakness_cache: Dict[int, int] = {}

    def is_valid_policy(self, mask: int) -> bool:
        return self.language.is_in_language_mask(mask)

    def weakness(self, mask: int) -> int:
        if mask in self._weakness_cache:
            return self._weakness_cache[mask]
        w = self.language.weakness_of_mask(mask)
        self._weakness_cache[mask] = w
        return w

    def correctness_counts(self, mask: int) -> CorrectnessCounts:
        if mask in self._correct_cache:
            return self._correct_cache[mask]

        outputs = self.outputs
        in_outputs = 0
        false_positives = 0
        for y in self.ext_I:
            if _is_subset(mask, y):
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

    def evaluate(self, mask: int, *, compute_weakness: bool = True) -> Dict[str, Optional[int]]:
        """Return a minimal evaluation record for notebooks."""
        if not self.is_valid_policy(mask):
            return {
                "mask": int(mask),
                "is_valid": 0,
                "weakness": None,
                "description_length": int(mask).bit_count(),
                "is_correct": None,
            }
        counts = self.correctness_counts(mask)
        w = self.weakness(mask) if compute_weakness else None
        return {
            "mask": int(mask),
            "is_valid": 1,
            "weakness": w,
            "description_length": int(mask).bit_count(),
            "is_correct": 1 if counts.is_correct else 0,
        }
