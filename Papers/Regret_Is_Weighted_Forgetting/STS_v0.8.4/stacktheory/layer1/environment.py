# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Environment types for Layer 1.

An environment is a nonempty set of mutually exclusive states.

Layer 1 focuses on finite environments, because finiteness is what makes exact
extensional reasoning computable and testable.

The environment also supplies an enumeration of states.
That enumeration is what makes packed bitsets meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Hashable, Iterator, Protocol, Sequence, Tuple, runtime_checkable

from .bitset import bitset_full


@runtime_checkable
class FiniteEnv(Protocol):
    """A finite environment interface.

    Stack Theory treats an environment Phi as a nonempty set of mutually
    exclusive states.

    In the implementation we also need a fixed enumeration of those states.
    That enumeration is what makes extensional sets computable.
    """

    @property
    def size(self) -> int:
        """Return the number of states in Phi."""

    def all_states_bitset(self) -> int:
        """Return the bitset that contains every state in Phi."""

    def mask_to_size(self, bitset: int) -> int:
        """Mask out any bits above the environment size."""


@dataclass(frozen=True)
class FiniteEnvironment:
    """A finite environment with a fixed enumeration of states.

    Parameters
    ----------
    states
        A nonempty sequence of hashable states.
        The order defines the canonical enumeration.
    """

    states: Tuple[Hashable, ...]
    index: Dict[Hashable, int]

    def __init__(self, states: Sequence[Hashable]):
        if len(states) == 0:
            raise ValueError("Environment must be nonempty")
        unique = list(states)
        if len(set(unique)) != len(unique):
            raise ValueError("Environment states must be unique")
        object.__setattr__(self, "states", tuple(unique))
        object.__setattr__(self, "index", {s: i for i, s in enumerate(unique)})

    @property
    def size(self) -> int:
        return len(self.states)

    def iter_states(self) -> Iterator[Hashable]:
        return iter(self.states)

    def state_to_index(self, state: Hashable) -> int:
        return self.index[state]

    def all_states_bitset(self) -> int:
        return bitset_full(self.size)

    def empty_states_bitset(self) -> int:
        return 0

    def mask_to_size(self, bitset: int) -> int:
        """Mask out bits above the environment size."""
        return bitset & self.all_states_bitset()


@dataclass(frozen=True)
class BooleanCubeEnvironment:
    """Boolean cube environment Phi_n = {0,1}^n.

    The canonical enumeration is the integer encoding of assignments.
    State k corresponds to the n-bit vector given by the binary expansion of k.

    Variable indices use 0 based indexing.
    Variable 0 is the least significant bit of the integer state.
    """

    n: int

    def __post_init__(self) -> None:
        if self.n <= 0:
            raise ValueError("n must be positive")

    @property
    def size(self) -> int:
        return 1 << self.n

    def iter_states(self) -> Iterator[int]:
        return iter(range(self.size))

    def state_to_index(self, state: int) -> int:
        if state < 0 or state >= self.size:
            raise ValueError("state out of range")
        return int(state)

    def all_states_bitset(self) -> int:
        return bitset_full(self.size)

    def empty_states_bitset(self) -> int:
        return 0

    def mask_to_size(self, bitset: int) -> int:
        return bitset & self.all_states_bitset()

    def bit_value(self, state: int, var: int) -> int:
        if var < 0 or var >= self.n:
            raise ValueError("var out of range")
        return (state >> var) & 1

    def assignment_bits(self, state: int) -> Tuple[int, ...]:
        """Return the full assignment as a tuple of bits."""
        return tuple(self.bit_value(state, i) for i in range(self.n))

    def literal_program_mask(self, *, var: int, value: int) -> int:
        """Return the membership bitset for the literal (x_var = value)."""
        return self._literal_program_mask_cached(var=var, value=value)

    @lru_cache(maxsize=None)
    def _literal_program_mask_cached(self, *, var: int, value: int) -> int:
        """Cached helper for literal_program_mask.

        This uses a block construction rather than a loop over every state.
        """
        if value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        if var < 0 or var >= self.n:
            raise ValueError("var out of range")

        # The enumeration is the integer encoding.
        # For a fixed var, its value flips every 2^var states.
        #
        # We build the membership bitset as a repeated pattern.
        # This avoids an O(2^n) loop for small var values.
        #
        # Pattern over one period.
        # block is the run length where var is constant.
        block = 1 << var
        step = block << 1
        size = self.size

        half = (1 << block) - 1
        pat = (half << block) if value == 1 else half

        # Repeat the pattern to fill size bits.
        # size and step are powers of two so the repeat count is also a power of two.
        out = pat
        current = step
        while current < size:
            out |= out << current
            current <<= 1

        return int(out) & self.all_states_bitset()

    def literal_program(self, *, var: int, value: int):
        """Return the literal program as a Program object.

        This method imports Program lazily to avoid circular imports.
        """
        from .program import Program
        return Program(env=self, bitset=self.literal_program_mask(var=var, value=value))


@dataclass(frozen=True)
class IndexEnvironment:
    """Environment Phi = {0,1,...,n_states-1}.

    This is a memory efficient finite environment for large datasets.
    Unlike FiniteEnvironment, it does not store an explicit tuple of states.

    The canonical enumeration is the identity mapping.
    State k has index k.
    """

    n_states: int

    def __post_init__(self) -> None:
        if self.n_states <= 0:
            raise ValueError("n_states must be positive")

    @property
    def size(self) -> int:
        return int(self.n_states)

    def iter_states(self) -> Iterator[int]:
        return iter(range(self.size))

    def state_to_index(self, state: int) -> int:
        if state < 0 or state >= self.size:
            raise ValueError("state out of range")
        return int(state)

    def all_states_bitset(self) -> int:
        return bitset_full(self.size)

    def empty_states_bitset(self) -> int:
        return 0

    def mask_to_size(self, bitset: int) -> int:
        return bitset & self.all_states_bitset()
