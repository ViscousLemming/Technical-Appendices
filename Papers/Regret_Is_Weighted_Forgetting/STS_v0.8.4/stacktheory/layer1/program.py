# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Programs as extensional sets of environment states.

In Stack Theory, a program is any subset of the environment state space Phi.
This module implements programs for finite environments, using a packed bitset.

The semantics are extensional.
Two programs are equal if and only if they contain the same states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from .bitset import bitset_iter_indices, bitset_popcount, pack_bool_tensor, unpack_bool_tensor
from .environment import FiniteEnv


@dataclass(frozen=True)
class Program:
    """A program p that is a subset of Phi for a finite environment.

    Parameters
    ----------
    env
        A finite environment with a fixed enumeration of states.
    bitset
        An integer whose bits encode membership relative to env.
    """

    env: FiniteEnv
    bitset: int

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Program):
            return False
        return (self.env is other.env) and (self.bitset == other.bitset)

    def __hash__(self) -> int:
        return hash((id(self.env), self.bitset))

    def __post_init__(self) -> None:
        if self.bitset < 0:
            raise ValueError("bitset must be non negative")
        # Mask out any bits above env.size to keep the representation canonical.
        masked = self.env.mask_to_size(self.bitset)
        object.__setattr__(self, "bitset", masked)

    def __and__(self, other: "Program") -> "Program":
        self._check_compatible(other)
        return Program(self.env, self.bitset & other.bitset)

    def __or__(self, other: "Program") -> "Program":
        self._check_compatible(other)
        return Program(self.env, self.bitset | other.bitset)

    def __sub__(self, other: "Program") -> "Program":
        self._check_compatible(other)
        return Program(self.env, self.bitset & (~other.bitset))

    def __xor__(self, other: "Program") -> "Program":
        """Return the symmetric difference of two programs."""
        self._check_compatible(other)
        return Program(self.env, self.bitset ^ other.bitset)

    def __invert__(self) -> "Program":
        return Program(self.env, (~self.bitset) & self.env.all_states_bitset())

    def issubset(self, other: "Program") -> bool:
        self._check_compatible(other)
        return (self.bitset & ~other.bitset) == 0

    def is_empty(self) -> bool:
        return self.bitset == 0

    def cardinality(self) -> int:
        """Return the number of states in the program."""
        return bitset_popcount(self.bitset)

    def to_bool_tensor(self, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return a boolean tensor of shape (env.size,) representing membership."""
        return unpack_bool_tensor(self.bitset, self.env.size, device=device)

    @classmethod
    def from_bool_tensor(cls, env: FiniteEnv, x: torch.Tensor) -> "Program":
        """Create a Program from a boolean tensor with env.size elements."""
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.numel() != env.size:
            raise ValueError("tensor must have exactly env.size elements")
        bitset = pack_bool_tensor(x)
        return cls(env=env, bitset=bitset)

    @classmethod
    def from_state_indices(cls, env: FiniteEnv, indices: Iterable[int]) -> "Program":
        """Create a Program from an iterable of environment indices."""
        bits = 0
        for i in indices:
            if i < 0 or i >= env.size:
                raise ValueError("index out of range")
            bits |= 1 << int(i)
        return cls(env=env, bitset=bits)

    def iter_state_indices(self) -> Iterator[int]:
        """Iterate over indices in ascending order."""
        yield from bitset_iter_indices(self.bitset)

    def symmetric_difference(self, other: "Program") -> "Program":
        """Return (self minus other) union (other minus self)."""
        return self ^ other

    def _check_compatible(self, other: "Program") -> None:
        if self.env is not other.env:
            # Environments are treated as identity objects.
            # Users can share env instances to compare programs.
            raise ValueError("Programs belong to different environments")
