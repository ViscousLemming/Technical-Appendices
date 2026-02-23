# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

r"""Vocabularies and statements for Layer 1.

This module implements the Stack Theory objects built from a fixed finite
environment and a finite vocabulary.

Math mapping

Vocabulary
A vocabulary is a finite set v of programs.
Each program is a set p \subseteq \Phi.
In code we store a vocabulary as an ordered list [p_0,...,p_{m-1}].

Statement
A statement is a set \ell \subseteq v.
In code a statement is represented by an integer mask m.
Bit j of m is 1 exactly when p_j \in \ell.

Truth set
The truth set is T(\ell) = \bigcap_{p \in \ell} p.
In code this is a bitwise AND over the selected program bitsets.

Induced language
The induced language L_v is the set of all statements with nonempty truth set.
Equivalently, m is in L_v when T(m) is not empty.

Completion, extension, weakness
A completion of \ell is any \ell' \in L_v with \ell \subseteq \ell'.
In code, y is a completion of m when (m & y) == m and y is in L_v.
The extension is Ext(\ell) = {\ell' \in L_v \mid \ell \subseteq \ell'}.
Weakness is w(\ell) = |Ext(\ell)|.

Implementation notes

- The empty statement is always valid because its truth set is \Phi.
- This file is the reference for computing truth sets, extensions, and weakness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Union

from .bitset import bitset_iter_indices, bitset_popcount
from .environment import FiniteEnv
from .program import Program


NameOrIndex = Union[str, int]


@dataclass(frozen=True)
class Statement:
    """A statement candidate represented as a bitmask over the vocabulary.

A Statement stores a vocabulary reference and an integer mask.
Not every mask is a valid statement in the induced language.
Use is_valid to check membership in L_v.
Use Vocabulary.statement when you want construction that rejects invalid masks.
"""

    vocab: "Vocabulary"
    mask: int

    def __post_init__(self) -> None:
        if self.mask < 0:
            raise ValueError("mask must be non negative")
        if self.mask >= (1 << self.vocab.size):
            raise ValueError("mask contains bits above vocabulary size")

    def __len__(self) -> int:
        """Return |l|, the description length proxy used in the learning definitions."""
        return bitset_popcount(self.mask)

    def is_valid(self) -> bool:
        """Return True if and only if this statement is in the induced language L_v."""
        return self.vocab.is_in_language_mask(self.mask)

    def truth_set(self) -> Program:
        """Return T(l) as a Program."""
        return self.vocab.truth_set_of_mask(self.mask)

    def programs(self) -> List[Program]:
        return [self.vocab.programs[i] for i in self.program_indices()]

    def program_indices(self) -> List[int]:
        return list(bitset_iter_indices(self.mask))

    def program_names(self) -> List[str]:
        return [self.vocab.names[i] for i in self.program_indices()]

    def conjoin(self, other: "Statement") -> "Statement":
        """Return the conjunction of two statements, if it is consistent in L_v."""
        if self.vocab is not other.vocab:
            raise ValueError("Statements belong to different vocabularies")
        new_mask = self.mask | other.mask
        s = Statement(self.vocab, new_mask)
        if not s.is_valid():
            raise ValueError("Conjunction is not a statement in the induced language")
        return s


class Vocabulary:
    """A finite vocabulary that is a set of programs.

    The vocabulary owns the environment.
    All programs must belong to the same environment instance.

    Statements are represented by integer masks over the program list.
    """

    def __init__(self, env: FiniteEnv, programs: Sequence[Program], names: Optional[Sequence[str]] = None):
        if names is None:
            names = [f"p{i}" for i in range(len(programs))]
        if len(names) != len(programs):
            raise ValueError("names length must match programs length")
        if len(set(names)) != len(names):
            raise ValueError("names must be unique")

        for p in programs:
            if p.env is not env:
                raise ValueError("All programs must belong to the same environment instance")

        if len(set(programs)) != len(programs):
            raise ValueError("Vocabulary programs must be unique as sets of states")

        self.env = env
        self.programs = list(programs)
        self.names = list(names)
        self.name_to_index: Dict[str, int] = {n: i for i, n in enumerate(self.names)}

        # Cache bitsets for hot loops.
        # Programs are treated as immutable after vocabulary construction.
        self._prog_bits = [p.bitset for p in self.programs]

        self._L_masks_cache: Optional[List[int]] = None
        self._truth_cache: Dict[int, Program] = {}
        # Cache for weakness values, keyed by statement mask.
        # This is safe because the vocabulary is treated as immutable after construction.
        self._weakness_cache: Dict[int, int] = {}

    @property
    def size(self) -> int:
        return len(self.programs)

    def program_bitsets(self) -> List[int]:
        """Return a copy of the program membership bitsets.

        This is a convenience method for performance sensitive code.
        """
        return list(self._prog_bits)

    def program_index(self, name_or_index: NameOrIndex) -> int:
        if isinstance(name_or_index, int):
            idx = int(name_or_index)
            if idx < 0 or idx >= self.size:
                raise ValueError("program index out of range")
            return idx
        if isinstance(name_or_index, str):
            if name_or_index not in self.name_to_index:
                raise KeyError(f"Unknown program name {name_or_index!r}")
            return self.name_to_index[name_or_index]
        raise TypeError("name_or_index must be str or int")

    def statement(self, items: Iterable[NameOrIndex]) -> Statement:
        mask = 0
        for it in items:
            idx = self.program_index(it)
            mask |= 1 << idx
        s = Statement(self, mask)
        if not s.is_valid():
            raise ValueError("The given subset is not a statement in the induced language")
        return s

    def empty_statement(self) -> Statement:
        return Statement(self, 0)

    def truth_set_of_mask(self, mask: int) -> Program:
        """Compute T(l) for the statement mask.

        For mask = 0, T(empty) = Phi.
        """
        if mask < 0 or mask >= (1 << self.size):
            raise ValueError("mask contains bits above vocabulary size")

        if mask in self._truth_cache:
            return self._truth_cache[mask]

        if mask == 0:
            t = Program(self.env, self.env.all_states_bitset())
            self._truth_cache[mask] = t
            return t

        # Start from Phi, then intersect each included program.
        # Iterate only over the set bits in mask for speed.
        bits = self.env.all_states_bitset()
        for i in bitset_iter_indices(mask):
            bits &= self.programs[i].bitset
            if bits == 0:
                break
        t = Program(self.env, bits)
        self._truth_cache[mask] = t
        return t

    def is_in_language_mask(self, mask: int) -> bool:
        return not self.truth_set_of_mask(mask).is_empty()

    def induced_language_masks(self) -> List[int]:
        """Enumerate L_v as a list of masks.

        This is exponential in vocabulary size.
        It is intended for small vocabularies and for tests.
        """
        if self._L_masks_cache is not None:
            return list(self._L_masks_cache)

        m = self.size
        if m > 20:
            raise ValueError("Exact induced language enumeration is too large for this vocabulary size")

        prog_bits = self._prog_bits
        full = self.env.all_states_bitset()
        out: List[int] = []

        def rec(start: int, mask: int, bits: int) -> None:
            # bits is the current truth set intersection.
            out.append(mask)
            self._truth_cache[mask] = Program(self.env, bits)
            for j in range(start, m):
                new_bits = bits & prog_bits[j]
                if new_bits != 0:
                    rec(j + 1, mask | (1 << j), new_bits)

        rec(0, 0, full)
        out.sort()
        self._L_masks_cache = out
        return list(out)

    def induced_language(self) -> List[Statement]:
        return [Statement(self, mask) for mask in self.induced_language_masks()]

    def completions(self, s: Statement) -> List[Statement]:
        """Return the list of completions of statement s in L_v."""
        if s.vocab is not self:
            raise ValueError("Statement does not belong to this vocabulary")
        if not s.is_valid():
            raise ValueError("s must be a statement in the induced language")
        return [Statement(self, mask) for mask in self.extension_masks_of_mask(s.mask)]

    def extension(self, s: Statement) -> List[Statement]:
        """Alias for completions."""
        return self.completions(s)

    def weakness(self, s: Statement) -> int:
        """Return w(s) = |E_s|."""
        if s.vocab is not self:
            raise ValueError("Statement does not belong to this vocabulary")
        if not s.is_valid():
            raise ValueError("s must be a statement in the induced language")
        return self.weakness_of_mask(s.mask)

    def equivalent(self, s1: Statement, s2: Statement) -> bool:
        """Return True if and only if E_{s1} = E_{s2}."""
        if s1.vocab is not self or s2.vocab is not self:
            raise ValueError("Statements must belong to this vocabulary")
        if not s1.is_valid() or not s2.is_valid():
            raise ValueError("Statements must be in the induced language")
        e1 = set(self.extension_masks_of_mask(s1.mask))
        e2 = set(self.extension_masks_of_mask(s2.mask))
        return e1 == e2

    def abstractor(self, s: Statement) -> "Vocabulary":
        """Return the abstractor vocabulary f(v,s).

        The mathematical definition is a set of programs.
        The implementation returns a Vocabulary instance whose programs are the distinct
        truth sets of completions.
        """
        progs = self.abstractor_programs(s)
        names = [f"T{i}" for i in range(len(progs))]
        return Vocabulary(self.env, programs=progs, names=names)

    def abstractor_programs(self, s: Statement) -> List[Program]:
        """Return the distinct truth sets {T(o) | o in E_s} as a deterministic list."""
        if s.vocab is not self:
            raise ValueError("Statement does not belong to this vocabulary")
        if not s.is_valid():
            raise ValueError("s must be a statement in the induced language")
        unique: Dict[int, Program] = {}
        for mask in self.iter_extension_masks_of_mask(s.mask):
            t = self.truth_set_of_mask(mask)
            unique[t.bitset] = t
        out = list(unique.values())
        out.sort(key=lambda p: p.bitset)
        return out

    def iter_extension_masks_of_mask(self, base_mask: int) -> Iterator[int]:
        """Yield completion masks y in L_v such that base_mask subset y.

        This is the extension E_x in Definition 3.
        The caller must ensure base_mask is a statement mask in L_v.
        """
        if base_mask < 0 or base_mask >= (1 << self.size):
            raise ValueError("base_mask contains bits above vocabulary size")

        base_truth = self.truth_set_of_mask(base_mask).bitset
        if base_truth == 0:
            raise ValueError("base_mask is not in the induced language")

        m = self.size
        prog_bits = self._prog_bits
        remaining = [i for i in range(m) if ((base_mask >> i) & 1) == 0]

        def rec(pos: int, mask: int, bits: int) -> Iterator[int]:
            yield mask
            for k in range(pos, len(remaining)):
                idx = remaining[k]
                new_bits = bits & prog_bits[idx]
                if new_bits != 0:
                    yield from rec(k + 1, mask | (1 << idx), new_bits)

        yield from rec(0, base_mask, base_truth)

    def extension_masks_of_mask(self, base_mask: int) -> List[int]:
        """Return the list of completion masks for base_mask."""
        return list(self.iter_extension_masks_of_mask(base_mask))

    def weakness_of_mask(self, base_mask: int) -> int:
        """Return w(l) for a statement mask base_mask in L_v."""
        if base_mask < 0 or base_mask >= (1 << self.size):
            raise ValueError("base_mask contains bits above vocabulary size")

        if base_mask in self._weakness_cache:
            return self._weakness_cache[base_mask]

        base_truth = self.truth_set_of_mask(base_mask).bitset
        if base_truth == 0:
            raise ValueError("base_mask is not in the induced language")

        m = self.size
        prog_bits = self._prog_bits
        remaining = [i for i in range(m) if ((base_mask >> i) & 1) == 0]

        def rec(pos: int, bits: int) -> int:
            total = 1
            for k in range(pos, len(remaining)):
                idx = remaining[k]
                new_bits = bits & prog_bits[idx]
                if new_bits != 0:
                    total += rec(k + 1, new_bits)
            return total

        w = rec(0, base_truth)
        self._weakness_cache[base_mask] = w
        return w

