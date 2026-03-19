# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Layer 4 experiment and scaling utilities.

Layer 4 is where we move from reference correctness code to tooling that makes
large experimental suites practical.

This layer intentionally stays faithful to the Stack Theory definitions.
When it introduces optional convenience extensions that are not part of the
core definitions, those APIs are named to make the distinction clear.
"""

from .language import Language
from .tasks import EmbodiedTask
from .evaluator import EmbodiedPolicyEvaluator
from .sat_language import ClauseVocabulary, WeaknessEstimate
from .packed_vocab import PackedVocabulary

__all__ = [
    "Language",
    "EmbodiedTask",
    "EmbodiedPolicyEvaluator",
    "ClauseVocabulary",
    "WeaknessEstimate",
    "PackedVocabulary",
]
