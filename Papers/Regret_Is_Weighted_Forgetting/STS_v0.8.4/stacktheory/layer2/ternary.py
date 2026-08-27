# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Standard tensor encodings for logic objects.

The thesis prototype used a ternary literal encoding.
Each variable position stores two booleans.
Index 0 means the negative literal is present.
Index 1 means the positive literal is present.
Both false means the variable is absent.
Both true is ambiguous and is rejected by default.

Layer 1 exposes a permissive parser.
Layer 2 provides a strict parser with an explicit allow_both_true flag.
"""

from __future__ import annotations

import torch

from stacktheory.layer1.logic import Clause, CNF, DNF, Literal, Term


def validate_ternary_matrix(
    t: torch.Tensor,
    *,
    allow_both_true: bool = False,
    name: str = "ternary",
) -> torch.Tensor:
    """Validate that t is a ternary literal matrix.

    Accepted shapes

    - (n_vars, 2)
    - (n_blocks, n_vars, 2)

    Returns a boolean tensor on CPU.
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")

    x = t.to(dtype=torch.bool, device="cpu")
    if x.ndim == 2:
        if x.shape[1] != 2:
            raise ValueError(f"{name} must have shape (n_vars, 2)")
    elif x.ndim == 3:
        if x.shape[2] != 2:
            raise ValueError(f"{name} must have shape (n_blocks, n_vars, 2)")
    else:
        raise ValueError(f"{name} must have 2 or 3 dimensions")

    if not allow_both_true:
        both = x[..., 0] & x[..., 1]
        if bool(both.any().item()):
            raise ValueError(
                f"{name} contains positions where both literals are True. "
                "Set allow_both_true=True to allow this."
            )

    return x


def clause_tensor_to_clause(t: torch.Tensor, *, allow_both_true: bool = False) -> Clause:
    """Convert a tensor of shape (n_vars, 2) to a Clause."""
    x = validate_ternary_matrix(t, allow_both_true=allow_both_true, name="clause")
    if x.ndim != 2:
        raise ValueError("clause tensor must have shape (n_vars, 2)")
    lits: list[Literal] = []
    for var in range(x.shape[0]):
        for value in (0, 1):
            if bool(x[var, value].item()):
                lits.append(Literal(var=var, value=value))
    return Clause.from_iterable(lits)


def clause_to_clause_tensor(clause: Clause, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a Clause to a tensor of shape (n_vars, 2)."""
    return clause.to_tensor(n_vars=n_vars, device=device)


def term_tensor_to_term(t: torch.Tensor, *, allow_both_true: bool = False) -> Term:
    """Convert a tensor of shape (n_vars, 2) to a Term."""
    x = validate_ternary_matrix(t, allow_both_true=allow_both_true, name="term")
    if x.ndim != 2:
        raise ValueError("term tensor must have shape (n_vars, 2)")
    lits: list[Literal] = []
    for var in range(x.shape[0]):
        for value in (0, 1):
            if bool(x[var, value].item()):
                lits.append(Literal(var=var, value=value))
    return Term.from_iterable(lits)


def term_to_term_tensor(term: Term, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a Term to a tensor of shape (n_vars, 2)."""
    return term.to_tensor(n_vars=n_vars, device=device)


def cnf_tensor_to_cnf(t: torch.Tensor, *, allow_both_true: bool = False) -> CNF:
    """Convert a tensor of shape (n_clauses, n_vars, 2) to a CNF."""
    x = validate_ternary_matrix(t, allow_both_true=allow_both_true, name="cnf")
    if x.ndim != 3:
        raise ValueError("cnf tensor must have shape (n_clauses, n_vars, 2)")
    clauses = [clause_tensor_to_clause(x[i, :, :], allow_both_true=allow_both_true) for i in range(x.shape[0])]
    return CNF.from_iterable(clauses)


def cnf_to_cnf_tensor(cnf: CNF, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a CNF to a tensor of shape (n_clauses, n_vars, 2)."""
    return cnf.to_tensor(n_vars=n_vars, device=device)


def dnf_tensor_to_dnf(t: torch.Tensor, *, allow_both_true: bool = False) -> DNF:
    """Convert a tensor of shape (n_terms, n_vars, 2) to a DNF."""
    x = validate_ternary_matrix(t, allow_both_true=allow_both_true, name="dnf")
    if x.ndim != 3:
        raise ValueError("dnf tensor must have shape (n_terms, n_vars, 2)")
    terms = [term_tensor_to_term(x[i, :, :], allow_both_true=allow_both_true) for i in range(x.shape[0])]
    return DNF.from_iterable(terms)


def dnf_to_dnf_tensor(dnf: DNF, n_vars: int, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a DNF to a tensor of shape (n_terms, n_vars, 2)."""
    return dnf.to_tensor(n_vars=n_vars, device=device)
