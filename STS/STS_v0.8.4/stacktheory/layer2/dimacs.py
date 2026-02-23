# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""DIMACS CNF import and export.

This is a common interchange format for SAT solvers.
Only CNF is supported.

This module does not call external solvers.
It only formats and parses.
"""

from __future__ import annotations

from typing import List, Tuple

from stacktheory.layer1.logic import Clause, CNF, Literal


def cnf_to_dimacs(cnf: CNF, *, n_vars: int) -> str:
    """Convert a CNF into a DIMACS string.

    A Literal(var, value=1) becomes (var+1).
    A Literal(var, value=0) becomes -(var+1).

    Tautological clauses are omitted because they do not constrain a CNF.
    """
    clauses: List[List[int]] = []

    for clause in cnf.clauses:
        lits: set[int] = set()
        tautology = False
        for lit in clause.literals:
            if lit.var < 0 or lit.var >= n_vars:
                raise ValueError("literal var out of range for n_vars")
            k = lit.var + 1
            dim = k if lit.value == 1 else -k
            if -dim in lits:
                tautology = True
                break
            lits.add(dim)

        if tautology:
            continue

        clause_list = sorted(lits, key=lambda x: (abs(x), x))
        clauses.append(clause_list)

    lines: List[str] = []
    lines.append(f"p cnf {n_vars} {len(clauses)}")
    for c in clauses:
        if len(c) == 0:
            lines.append("0")
        else:
            lines.append(" ".join(str(x) for x in c) + " 0")
    return "\n".join(lines) + "\n"


def cnf_from_dimacs(text: str) -> Tuple[CNF, int]:
    """Parse a DIMACS CNF string.

    Returns (cnf, n_vars).
    """
    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s == "":
            continue
        if s.startswith("c"):
            continue
        lines.append(s)

    if not lines:
        raise ValueError("no DIMACS content")

    header = None
    for i, s in enumerate(lines):
        if s.startswith("p "):
            header = s
            header_index = i
            break
    if header is None:
        raise ValueError("missing DIMACS header")

    parts = header.split()
    if len(parts) < 4 or parts[1] != "cnf":
        raise ValueError("unsupported DIMACS header")

    n_vars = int(parts[2])
    # n_clauses = int(parts[3])  # not trusted

    clause_lines = lines[header_index + 1 :]
    clauses: List[Clause] = []

    current: List[int] = []
    for line in clause_lines:
        for tok in line.split():
            lit = int(tok)
            if lit == 0:
                # Finish current clause.
                literals: List[Literal] = []
                for k in current:
                    var = abs(k) - 1
                    value = 1 if k > 0 else 0
                    literals.append(Literal(var=var, value=value))
                clauses.append(Clause.from_iterable(literals))
                current = []
            else:
                current.append(lit)

    if current:
        raise ValueError("DIMACS clause did not terminate with 0")

    return CNF.from_iterable(clauses), n_vars
