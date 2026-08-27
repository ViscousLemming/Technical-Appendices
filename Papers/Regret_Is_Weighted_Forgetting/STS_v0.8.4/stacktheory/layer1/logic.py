# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Boolean cube logic helpers.

This module gives a small syntax for boolean formulas when the environment is
a Boolean cube.

These syntax objects do not add new semantics.
They are just a clean way to build programs p that are subsets of Phi_n.

Mapping to Stack Theory

- The environment Phi_n is the set of all n bit assignments.
- A literal x_i = 1 denotes the program of all states where bit i is 1.
- A clause is an OR of literals. It denotes the union of the literal programs.
- A CNF is an AND of clauses. It denotes the intersection of the clause programs.
- A term is an AND of literals. It denotes the intersection of the literal programs.
- A DNF is an OR of terms. It denotes the union of the term programs.

The to_program methods compile each object to an extensional Program bitset.
That compilation is exact.

Tensor encodings

CNF and DNF can also be encoded in the thesis ternary tensor form.
This is a boolean tensor of shape (n_vars, 2).
Index 0 means the negative literal is present.
Index 1 means the positive literal is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import torch

from .environment import BooleanCubeEnvironment
from .program import Program


@dataclass(frozen=True, order=True)
class Literal:
    """A boolean literal of the form x_var = value."""
    var: int
    value: int  # 0 or 1

    def __post_init__(self) -> None:
        if self.value not in (0, 1):
            raise ValueError("value must be 0 or 1")
        if self.var < 0:
            raise ValueError("var must be non negative")

    def matches(self, assignment_state: int) -> bool:
        return ((assignment_state >> self.var) & 1) == self.value

    def to_program(self, env: BooleanCubeEnvironment) -> Program:
        return env.literal_program(var=self.var, value=self.value)


@dataclass(frozen=True)
class Clause:
    """A disjunction of literals."""
    literals: Tuple[Literal, ...] = ()

    def __post_init__(self) -> None:
        # Canonicalise order and remove duplicates.
        unique = {(l.var, l.value): l for l in self.literals}
        ordered = tuple(unique[k] for k in sorted(unique.keys()))
        object.__setattr__(self, "literals", ordered)

    @classmethod
    def from_iterable(cls, literals: Iterable[Literal]) -> "Clause":
        return cls(tuple(literals))

    def satisfied_by(self, assignment_state: int) -> bool:
        return any(l.matches(assignment_state) for l in self.literals)

    def to_program(self, env: BooleanCubeEnvironment) -> Program:
        # Empty clause is false everywhere.
        bits = 0
        for lit in self.literals:
            bits |= env.literal_program_mask(var=lit.var, value=lit.value)
        return Program(env, bits)

    def to_tensor(self, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return the thesis style ternary tensor encoding for this clause.

        Shape is (n_vars, 2).
        Index 0 is the negative literal.
        Index 1 is the positive literal.
        """
        t = torch.zeros((n_vars, 2), dtype=torch.bool, device=device)
        for lit in self.literals:
            if lit.var >= n_vars:
                raise ValueError("literal var out of range for n_vars")
            # value 0 means ~x, value 1 means x
            t[lit.var, lit.value] = True
        return t

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "Clause":
        """Parse a clause from a tensor of shape (n_vars, 2)."""
        if t.ndim != 2 or t.shape[1] != 2:
            raise ValueError("tensor must have shape (n_vars, 2)")
        t = t.to(dtype=torch.bool, device="cpu")
        lits: List[Literal] = []
        for var in range(t.shape[0]):
            for value in (0, 1):
                if bool(t[var, value].item()):
                    lits.append(Literal(var=var, value=value))
        return cls.from_iterable(lits)


@dataclass(frozen=True)
class Term:
    """A conjunction of literals."""
    literals: Tuple[Literal, ...] = ()

    def __post_init__(self) -> None:
        unique = {(l.var, l.value): l for l in self.literals}
        ordered = tuple(unique[k] for k in sorted(unique.keys()))
        object.__setattr__(self, "literals", ordered)

    @classmethod
    def from_iterable(cls, literals: Iterable[Literal]) -> "Term":
        return cls(tuple(literals))

    def satisfied_by(self, assignment_state: int) -> bool:
        return all(l.matches(assignment_state) for l in self.literals)

    def to_program(self, env: BooleanCubeEnvironment) -> Program:
        # Empty term is true everywhere.
        bits = env.all_states_bitset()
        for lit in self.literals:
            bits &= env.literal_program_mask(var=lit.var, value=lit.value)
            if bits == 0:
                break
        return Program(env, bits)

    def to_tensor(self, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return the ternary tensor encoding for this term.

        The encoding is identical to Clause.
        The semantics differ.
        """
        t = torch.zeros((n_vars, 2), dtype=torch.bool, device=device)
        for lit in self.literals:
            if lit.var >= n_vars:
                raise ValueError("literal var out of range for n_vars")
            t[lit.var, lit.value] = True
        return t

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "Term":
        if t.ndim != 2 or t.shape[1] != 2:
            raise ValueError("tensor must have shape (n_vars, 2)")
        t = t.to(dtype=torch.bool, device="cpu")
        lits: List[Literal] = []
        for var in range(t.shape[0]):
            for value in (0, 1):
                if bool(t[var, value].item()):
                    lits.append(Literal(var=var, value=value))
        return cls.from_iterable(lits)


@dataclass(frozen=True)
class CNF:
    """A conjunction of clauses."""
    clauses: Tuple[Clause, ...] = ()

    def __post_init__(self) -> None:
        # Canonicalise clause order and remove duplicates.
        unique = {c.literals: c for c in self.clauses}
        ordered_keys = sorted(unique.keys())
        object.__setattr__(self, "clauses", tuple(unique[k] for k in ordered_keys))

    @classmethod
    def from_iterable(cls, clauses: Iterable[Clause]) -> "CNF":
        return cls(tuple(clauses))

    def satisfied_by(self, assignment_state: int) -> bool:
        return all(c.satisfied_by(assignment_state) for c in self.clauses)

    def to_program(self, env: BooleanCubeEnvironment) -> Program:
        bits = env.all_states_bitset()
        for c in self.clauses:
            bits &= c.to_program(env).bitset
            if bits == 0:
                break
        return Program(env, bits)

    def to_tensor(self, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return the thesis style tensor encoding for this CNF.

        Shape is (n_clauses, n_vars, 2).
        """
        t = torch.zeros((len(self.clauses), n_vars, 2), dtype=torch.bool, device=device)
        for i, c in enumerate(self.clauses):
            t[i, :, :] = c.to_tensor(n_vars, device=device)
        return t

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "CNF":
        if t.ndim != 3 or t.shape[2] != 2:
            raise ValueError("tensor must have shape (n_clauses, n_vars, 2)")
        t = t.to(dtype=torch.bool, device="cpu")
        clauses: List[Clause] = []
        for i in range(t.shape[0]):
            clauses.append(Clause.from_tensor(t[i]))
        return cls.from_iterable(clauses)


@dataclass(frozen=True)
class DNF:
    """A disjunction of terms."""
    terms: Tuple[Term, ...] = ()

    def __post_init__(self) -> None:
        unique = {t.literals: t for t in self.terms}
        ordered_keys = sorted(unique.keys())
        object.__setattr__(self, "terms", tuple(unique[k] for k in ordered_keys))

    @classmethod
    def from_iterable(cls, terms: Iterable[Term]) -> "DNF":
        return cls(tuple(terms))

    def satisfied_by(self, assignment_state: int) -> bool:
        return any(t.satisfied_by(assignment_state) for t in self.terms)

    def to_program(self, env: BooleanCubeEnvironment) -> Program:
        bits = 0
        for t in self.terms:
            bits |= t.to_program(env).bitset
        return Program(env, bits)

    def to_tensor(self, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
        """Return the thesis style tensor encoding for this DNF.

        Shape is (n_terms, n_vars, 2).
        """
        t = torch.zeros((len(self.terms), n_vars, 2), dtype=torch.bool, device=device)
        for i, term in enumerate(self.terms):
            t[i, :, :] = term.to_tensor(n_vars, device=device)
        return t

    @classmethod
    def from_tensor(cls, t: torch.Tensor) -> "DNF":
        if t.ndim != 3 or t.shape[2] != 2:
            raise ValueError("tensor must have shape (n_terms, n_vars, 2)")
        t = t.to(dtype=torch.bool, device="cpu")
        terms: List[Term] = []
        for i in range(t.shape[0]):
            terms.append(Term.from_tensor(t[i]))
        return cls.from_iterable(terms)


def assignment_tensor_from_state(state: int, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Encode a complete assignment state as a tensor of shape (n_vars, 2).

    The encoding is one hot per variable.
    Index 0 is false.
    Index 1 is true.
    """
    t = torch.zeros((n_vars, 2), dtype=torch.bool, device=device)
    for var in range(n_vars):
        val = (state >> var) & 1
        t[var, val] = True
    return t


def cnf_tensor_satisfied(
    assignments: torch.Tensor,
    cnf: torch.Tensor,
    *,
    validate_assignments: bool = True,
) -> torch.Tensor:
    """Evaluate a CNF tensor on a batch of assignment tensors.

    Parameters
    ----------
    assignments
        Tensor of shape (batch, n_vars, 2) with one hot values.
    cnf
        Tensor of shape (n_clauses, n_vars, 2) using the ternary literal encoding.

    Returns
    -------
    Tensor of shape (batch,) where True means the assignment satisfies the CNF.
    """
    if assignments.ndim != 3 or assignments.shape[2] != 2:
        raise ValueError("assignments must have shape (batch, n_vars, 2)")
    if cnf.ndim != 3 or cnf.shape[2] != 2:
        raise ValueError("cnf must have shape (n_clauses, n_vars, 2)")
    if assignments.shape[1] != cnf.shape[1]:
        raise ValueError("n_vars mismatch")

    # A complete assignment must be one hot per variable.
    if validate_assignments:
        if not torch.all(assignments.to(dtype=torch.int64).sum(dim=2) == 1):
            raise ValueError("assignments must be one hot on the last dimension")

    # clause_match[b, c, v, val] is True when clause contains literal and assignment matches it
    clause_match = (assignments.unsqueeze(1) & cnf.unsqueeze(0))
    # clause_satisfied[b, c] is True if any literal matches in that clause
    clause_satisfied = clause_match.flatten(start_dim=2).any(dim=2)
    # cnf satisfied if all clauses satisfied
    return clause_satisfied.all(dim=1)


def dnf_tensor_satisfied(
    assignments: torch.Tensor,
    dnf: torch.Tensor,
    *,
    validate_assignments: bool = True,
) -> torch.Tensor:
    """Evaluate a DNF tensor on a batch of assignment tensors.

    The DNF tensor is interpreted as a disjunction of conjunctive terms.
    A term is satisfied if all its specified literals match the assignment.
    """
    if assignments.ndim != 3 or assignments.shape[2] != 2:
        raise ValueError("assignments must have shape (batch, n_vars, 2)")
    if dnf.ndim != 3 or dnf.shape[2] != 2:
        raise ValueError("dnf must have shape (n_terms, n_vars, 2)")
    if assignments.shape[1] != dnf.shape[1]:
        raise ValueError("n_vars mismatch")

    if validate_assignments:
        if not torch.all(assignments.to(dtype=torch.int64).sum(dim=2) == 1):
            raise ValueError("assignments must be one hot on the last dimension")

    term_match = (assignments.unsqueeze(1) & dnf.unsqueeze(0))
    # Count per term, per assignment.
    matched = term_match.flatten(start_dim=2).sum(dim=2)
    required = dnf.unsqueeze(0).flatten(start_dim=2).sum(dim=2)
    term_satisfied = matched == required
    # If a term has no literals, required is 0 and it is satisfied by every assignment.
    return term_satisfied.any(dim=1)
