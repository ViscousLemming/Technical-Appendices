# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Induced language tooling.

Stack Theory Definition 2 defines the induced language L_v as the set of all
conjunctions of vocabulary programs that have a non empty truth set.

This module provides a small convenience wrapper around a Layer 1 Vocabulary.

Important definition alignment

- Extension is computed as completions in the induced language.
- Weakness is the cardinality of that extension.

If you want to restrict search to a finite candidate set Q, do not redefine
the induced language.
Pass Q into the learner functions in layer3 as the candidates argument.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

from stacktheory.layer1.vocabulary import Vocabulary


@dataclass
class Language:
    """A wrapper for the induced language L_v of a fixed finite vocabulary.

    The underlying language is always the induced language from the appendix
    definitions.

    Notes
    -----
    The induced language can be exponentially large in the vocabulary size.
    Methods that enumerate all statements are only intended for small vocabularies.
    """

    vocab: Vocabulary

    def __post_init__(self) -> None:
        m = self.vocab.size
        self._full_mask = (1 << m) - 1

    @property
    def is_induced(self) -> bool:
        """Return True.

        This class always represents the induced language L_v.
        """
        return True

    @property
    def size(self) -> int:
        """Return |L_v|.

        This enumerates L_v and is only viable for small vocabularies.
        """
        return len(self.vocab.induced_language_masks())

    def is_in_language_mask(self, mask: int) -> bool:
        """Return True if and only if mask is a statement in L_v."""
        if mask < 0 or mask > self._full_mask:
            return False
        return self.vocab.is_in_language_mask(mask)

    def iter_masks(self) -> Iterator[int]:
        """Iterate masks in L_v.

        This enumerates L_v and is only intended for small vocabularies.
        """
        yield from self.vocab.induced_language_masks()

    def masks(self) -> List[int]:
        return list(self.iter_masks())

    def extension_masks_of_mask(self, base_mask: int) -> List[int]:
        """Return Ext(base_mask) as a list of masks.

        This matches Definition 2 and Definition 3 in the appendices.
        Ext(l) is the set of completions l' in L_v such that l ⊆ l'.
        """
        if not self.is_in_language_mask(base_mask):
            raise ValueError("base_mask must be a statement in the induced language")
        return list(self.vocab.iter_extension_masks_of_mask(base_mask))

    def weakness_of_mask(self, base_mask: int) -> int:
        """Return w(base_mask) for base_mask in L_v.

        Weakness is the number of completions in the induced language.
        """
        if not self.is_in_language_mask(base_mask):
            raise ValueError("base_mask must be a statement in the induced language")
        return self.vocab.weakness_of_mask(base_mask)

    def completion_masks_for_subset_mask(self, subset_mask: int) -> List[int]:
        """Return completions for an arbitrary subset mask.

        This is a convenience method that accepts subset_mask even when it is not
        in the induced language.

        If subset_mask has empty truth set then it has no completions.
        If subset_mask is satisfiable then it is already in L_v and this is just
        Ext(subset_mask).
        """
        if subset_mask < 0 or subset_mask > self._full_mask:
            raise ValueError("subset_mask contains bits above vocabulary size")
        if not self.vocab.is_in_language_mask(subset_mask):
            return []
        return list(self.vocab.iter_extension_masks_of_mask(subset_mask))

    def completion_count_for_subset_mask(self, subset_mask: int) -> int:
        """Return the number of completions for subset_mask.

        This is 0 when subset_mask is unsatisfiable.
        Otherwise it equals weakness(subset_mask).
        """
        if subset_mask < 0 or subset_mask > self._full_mask:
            raise ValueError("subset_mask contains bits above vocabulary size")
        if not self.vocab.is_in_language_mask(subset_mask):
            return 0
        return self.vocab.weakness_of_mask(subset_mask)
