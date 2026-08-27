# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Reinforcement learning scaffolding for Stack Theory policy search.

This module does not implement a full RL algorithm.
It provides a small environment like interface that lets an RL agent
construct a policy statement one vocabulary element at a time.

Motivation

In many practical settings the induced language is huge.
Search based learners and evolutionary learners can help.
RL is another option.

The core idea

- The current state is a partial statement mask.
- An action adds a program index, or terminates.
- The episode ends when the agent terminates or hits max_steps.
- The reward is shaped using a user supplied heuristic key.

The environment is dependency free.
It is compatible with Gym style wrappers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional, Tuple

from stacktheory.layer1.vocabulary import Statement

from .evaluator import PolicyEvaluator
from .heuristics import KeyFn, PolicyStats, key_weakness_then_simplicity
from .tasks import Task


@dataclass(frozen=True)
class StepResult:
    mask: int
    done: bool
    reward: float


class PolicyConstructionEnv:
    """A minimal environment for constructing policy statements.

Actions

- 0..|v|-1 adds that program index to the statement.
- |v| means terminate.

State

The state is the current mask.
The environment also tracks the current truth set intersection so it can
reject inconsistent additions.
"""

    def __init__(
        self,
        task: Task,
        *,
        heuristic: KeyFn = key_weakness_then_simplicity,
        max_steps: Optional[int] = None,
        seed: int = 0,
        invalid_action_penalty: float = -1.0,
        terminal_incorrect_penalty: float = -1.0,
        terminal_correct_reward: float = 1.0,
    ):
        self.task = task
        self.vocab = task.vocab
        self.evaluator = PolicyEvaluator(task)
        self.heuristic = heuristic
        self.max_steps = self.vocab.size if max_steps is None else int(max_steps)
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.rng = random.Random(seed)
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.terminal_incorrect_penalty = float(terminal_incorrect_penalty)
        self.terminal_correct_reward = float(terminal_correct_reward)

        self._mask = 0
        self._bits = self.vocab.env.all_states_bitset()
        self._steps = 0
        self._last_key: Optional[Tuple[int, ...]] = None

    @property
    def n_actions(self) -> int:
        return self.vocab.size + 1

    def reset(self) -> int:
        self._mask = 0
        self._bits = self.vocab.env.all_states_bitset()
        self._steps = 0
        self._last_key = None
        return self._mask

    def _current_key(self) -> Tuple[int, ...]:
        w = self.evaluator.weakness(self._mask)
        d = int(self._mask).bit_count()
        return self.heuristic(PolicyStats(mask=self._mask, weakness=w, description_length=d))

    def step(self, action: int) -> StepResult:
        if action < 0 or action >= self.n_actions:
            raise ValueError("action out of range")

        done = False
        reward = 0.0

        if action == self.vocab.size:
            done = True
        else:
            # Add program index if it keeps the statement in the language.
            bit = 1 << action
            if self._mask & bit:
                # No op. Penalise a bit to discourage wasting steps.
                reward += self.invalid_action_penalty * 0.1
            else:
                new_bits = self._bits & self.vocab.programs[action].bitset
                if new_bits == 0:
                    # Inconsistent addition.
                    reward += self.invalid_action_penalty
                else:
                    self._mask |= bit
                    self._bits = new_bits

        self._steps += 1
        if self._steps >= self.max_steps:
            done = True

        # Shaped reward based on heuristic improvement.
        key = self._current_key()
        if self._last_key is not None:
            if key > self._last_key:
                reward += 0.1
            elif key < self._last_key:
                reward -= 0.05
        self._last_key = key

        if done:
            if self.evaluator.is_correct(self._mask):
                reward += self.terminal_correct_reward
            else:
                reward += self.terminal_incorrect_penalty

        return StepResult(mask=self._mask, done=done, reward=reward)

    def current_statement(self) -> Statement:
        return Statement(self.vocab, self._mask)
