# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Human readable formatting and simple parsers for CNF and DNF.

This module is for lab work.
It is deterministic.
It avoids clever simplification.

These helper objects are not part of the Layer 1 core.
They are a convenience layer.
They let you read and write small formulas, then compile them into extensional
programs on a Boolean cube environment.

Supported grammar

Literals

- x0 means variable 0 is True
- ~x0 or !x0 means variable 0 is False

CNF

- (x0 | ~x1) & (x2)

DNF

- (x0 & ~x1) | (x2)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from stacktheory.layer1.logic import Clause, CNF, DNF, Literal, Term


def literal_to_str(lit: Literal, *, var_names: Optional[Sequence[str]] = None) -> str:
    """Render a Literal as a readable string.

    If var_names is given, it is used to name variables.
    Otherwise variables are shown as x0, x1, ...

    Examples

    - Literal(var=0,value=1) -> x0
    - Literal(var=0,value=0) -> ~x0
    """
    if var_names is None:
        name = f"x{lit.var}"
    else:
        if lit.var < 0 or lit.var >= len(var_names):
            raise ValueError("literal var out of range for var_names")
        name = str(var_names[lit.var])

    if lit.value == 1:
        return name
    return "~" + name


def clause_to_str(clause: Clause, *, var_names: Optional[Sequence[str]] = None) -> str:
    """Render a Clause as a readable string.

    This prints a disjunction of literals.
    Empty clause is printed as (FALSE).
    """
    if len(clause.literals) == 0:
        return "(FALSE)"
    return "(" + " | ".join(literal_to_str(l, var_names=var_names) for l in clause.literals) + ")"


def term_to_str(term: Term, *, var_names: Optional[Sequence[str]] = None) -> str:
    """Render a Term as a readable string.

    This prints a conjunction of literals.
    Empty term is printed as (TRUE).
    """
    if len(term.literals) == 0:
        return "(TRUE)"
    return "(" + " & ".join(literal_to_str(l, var_names=var_names) for l in term.literals) + ")"


def cnf_to_str(cnf: CNF, *, var_names: Optional[Sequence[str]] = None) -> str:
    """Render a CNF as a readable string."""
    if len(cnf.clauses) == 0:
        return "(TRUE)"
    return " & ".join(clause_to_str(c, var_names=var_names) for c in cnf.clauses)


def dnf_to_str(dnf: DNF, *, var_names: Optional[Sequence[str]] = None) -> str:
    """Render a DNF as a readable string."""
    if len(dnf.terms) == 0:
        return "(FALSE)"
    return " | ".join(term_to_str(t, var_names=var_names) for t in dnf.terms)


def _strip_outer_parens(s: str) -> str:
    """Strip one pair of outer parentheses if they wrap the whole string."""
    s = s.strip()
    if len(s) >= 2 and s[0] == "(" and s[-1] == ")":
        return s[1:-1].strip()
    return s


def _split_top_level(s: str, sep: str) -> List[str]:
    """Split a string by a separator that is not inside parentheses.

    This is a tiny parser helper.
    It lets us split CNF by & and clauses by | without getting confused by
    nested parentheses.
    """
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(s):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses")
        elif ch == sep and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    if depth != 0:
        raise ValueError("unbalanced parentheses")
    parts.append(s[start:].strip())
    return [p for p in parts if p != ""]


def parse_literal(token: str, *, var_to_index: Optional[dict[str, int]] = None) -> Literal:
    """Parse a literal token like x0 or ~x1 into a Literal."""
    tok = token.strip()
    if tok == "":
        raise ValueError("empty literal")

    neg = False
    if tok[0] in ("~", "!"):
        neg = True
        tok = tok[1:].strip()

    if tok == "":
        raise ValueError("literal missing variable name")

    if var_to_index is not None:
        if tok not in var_to_index:
            raise ValueError(f"unknown variable name {tok!r}")
        var = int(var_to_index[tok])
    else:
        if not tok.startswith("x"):
            raise ValueError("expected variable names like x0, x1, ...")
        suffix = tok[1:]
        if suffix == "" or not suffix.isdigit():
            raise ValueError("expected variable names like x0, x1, ...")
        var = int(suffix)

    value = 0 if neg else 1
    return Literal(var=var, value=value)


def parse_clause(s: str, *, var_to_index: Optional[dict[str, int]] = None) -> Clause:
    """Parse a clause string like (x0 | ~x1) into a Clause."""
    inner = _strip_outer_parens(s)
    if inner.strip().upper() == "FALSE":
        return Clause.from_iterable([])
    lits = [_strip_outer_parens(x) for x in _split_top_level(inner, "|")]
    if len(lits) == 1 and lits[0] == inner and "|" not in inner:
        lits = [inner]
    literals = [parse_literal(tok, var_to_index=var_to_index) for tok in lits if tok.strip() != ""]
    return Clause.from_iterable(literals)


def parse_term(s: str, *, var_to_index: Optional[dict[str, int]] = None) -> Term:
    """Parse a term string like (x0 & ~x1) into a Term."""
    inner = _strip_outer_parens(s)
    if inner.strip().upper() == "TRUE":
        return Term.from_iterable([])
    lits = [_strip_outer_parens(x) for x in _split_top_level(inner, "&")]
    if len(lits) == 1 and lits[0] == inner and "&" not in inner:
        lits = [inner]
    literals = [parse_literal(tok, var_to_index=var_to_index) for tok in lits if tok.strip() != ""]
    return Term.from_iterable(literals)


def parse_cnf(s: str, *, var_to_index: Optional[dict[str, int]] = None) -> CNF:
    """Parse a CNF string like (x0 | ~x1) & (x2) into a CNF."""
    text = s.strip()
    if text == "":
        raise ValueError("empty CNF string")
    norm = _strip_outer_parens(text)
    if norm.strip().upper() == "TRUE":
        return CNF.from_iterable([])
    if norm.strip().upper() == "FALSE":
        return CNF.from_iterable([Clause.from_iterable([])])

    clause_strs = _split_top_level(text, "&")
    clauses = [parse_clause(cs, var_to_index=var_to_index) for cs in clause_strs]
    return CNF.from_iterable(clauses)


def parse_dnf(s: str, *, var_to_index: Optional[dict[str, int]] = None) -> DNF:
    """Parse a DNF string like (x0 & ~x1) | (x2) into a DNF."""
    text = s.strip()
    if text == "":
        raise ValueError("empty DNF string")
    norm = _strip_outer_parens(text)
    if norm.strip().upper() == "FALSE":
        return DNF.from_iterable([])
    if norm.strip().upper() == "TRUE":
        return DNF.from_iterable([Term.from_iterable([])])

    term_strs = _split_top_level(text, "|")
    terms = [parse_term(ts, var_to_index=var_to_index) for ts in term_strs]
    return DNF.from_iterable(terms)
