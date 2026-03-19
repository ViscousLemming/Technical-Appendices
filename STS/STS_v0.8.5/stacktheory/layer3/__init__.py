# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Layer 3 tasks, policies, and learning.

Layer 1 defines Stack Theory semantics for finite environments.
Layer 2 adds interoperability and convenient representations.

Layer 3 introduces the task and learning objects from the appendices.
It provides exact reference implementations for.

- tasks
- policy correctness
- brute force learning under weakness or description length proxies

These implementations are intended to be correct and transparent.
Higher performance learners can be added later as long as they preserve the
same semantics in the finite setting.
"""

from .tasks import Task, is_child_task
from .heuristics import (
    PolicyStats,
    KeyFn,
    key_weakness_then_simplicity,
    key_simplicity_then_weakness,
    make_lexicographic_key,
)
from .learning import (
    is_correct_policy,
    correct_policies,
    learn,
    learn_min_description_length,
    Proxy,
)

from .evaluator import (
    PolicyEvaluator,
    PolicyEval,
    CorrectnessCounts,
)

from .search import (
    learn_best_first,
    learn_branch_and_bound,
    learn_beam_search,
    learn_random_search,
)

from .genetic import (
    learn_genetic,
    GeneticConfig,
)

from .rl_env import (
    PolicyConstructionEnv,
    StepResult,
)

from .literals import (
    analyze_literal_vocabulary,
    LiteralVocabularyInfo,
)

from .local_search import (
    learn_hill_climb,
    learn_simulated_annealing,
    LocalSearchConfig,
)

__all__ = [
    "Task",
    "is_child_task",
    "Proxy",
    "is_correct_policy",
    "correct_policies",
    "learn",
    "learn_min_description_length",
    "PolicyEvaluator",
    "PolicyEval",
    "CorrectnessCounts",
    "learn_best_first",
    "learn_branch_and_bound",
    "learn_beam_search",
    "learn_random_search",
    "learn_genetic",
    "GeneticConfig",
    "PolicyConstructionEnv",
    "StepResult",
    "analyze_literal_vocabulary",
    "LiteralVocabularyInfo",
    "learn_hill_climb",
    "learn_simulated_annealing",
    "LocalSearchConfig",
    "PolicyStats",
    "KeyFn",
    "key_weakness_then_simplicity",
    "key_simplicity_then_weakness",
    "make_lexicographic_key",
]
