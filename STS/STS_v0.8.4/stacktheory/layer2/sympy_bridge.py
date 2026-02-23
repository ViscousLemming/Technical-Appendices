# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Optional interoperability with SymPy.

SymPy is useful for validation and for quick research code.
The Stack Theory Suite does not depend on SymPy at runtime.

This module imports SymPy lazily inside the conversion functions.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

from stacktheory.layer1.logic import Clause, CNF, DNF, Literal, Term


def _symbols_for_n(n: int, *, base: str = "x"):
    import sympy as sp
    return sp.symbols(f"{base}0:{n}")


def cnf_to_sympy(cnf: CNF, *, n_vars: int, symbols: Optional[Sequence[object]] = None):
    """Convert a CNF object into a SymPy boolean expression."""
    import sympy as sp

    if symbols is None:
        symbols = _symbols_for_n(n_vars)
    if len(symbols) != n_vars:
        raise ValueError("symbols length must match n_vars")

    clauses = []
    for c in cnf.clauses:
        lits = []
        for lit in c.literals:
            if lit.var < 0 or lit.var >= n_vars:
                raise ValueError("literal var out of range")
            s = symbols[lit.var]
            lits.append(s if lit.value == 1 else sp.Not(s))
        if len(lits) == 0:
            clauses.append(sp.false)
        else:
            clauses.append(sp.Or(*lits))

    if len(clauses) == 0:
        return sp.true
    return sp.And(*clauses)


def dnf_to_sympy(dnf: DNF, *, n_vars: int, symbols: Optional[Sequence[object]] = None):
    """Convert a DNF object into a SymPy boolean expression."""
    import sympy as sp

    if symbols is None:
        symbols = _symbols_for_n(n_vars)
    if len(symbols) != n_vars:
        raise ValueError("symbols length must match n_vars")

    terms = []
    for t in dnf.terms:
        lits = []
        for lit in t.literals:
            if lit.var < 0 or lit.var >= n_vars:
                raise ValueError("literal var out of range")
            s = symbols[lit.var]
            lits.append(s if lit.value == 1 else sp.Not(s))
        if len(lits) == 0:
            terms.append(sp.true)
        else:
            terms.append(sp.And(*lits))

    if len(terms) == 0:
        return sp.false
    return sp.Or(*terms)


def _parse_sympy_literal(expr, symbol_to_var: Dict[object, int]) -> Literal:
    import sympy as sp

    if expr in symbol_to_var:
        return Literal(var=symbol_to_var[expr], value=1)
    if isinstance(expr, sp.Not) and len(expr.args) == 1 and expr.args[0] in symbol_to_var:
        return Literal(var=symbol_to_var[expr.args[0]], value=0)
    raise ValueError("expression is not a literal")


def sympy_to_cnf(expr, *, symbols: Optional[Sequence[object]] = None) -> CNF:
    """Parse a SymPy expression that is already in CNF.

    This function does not run logic minimisation.
    It only accepts a syntactic CNF form.
    """
    import sympy as sp

    if symbols is None:
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))

    symbol_to_var = {sym: i for i, sym in enumerate(symbols)}

    # Normalise trivial constants.
    if expr is sp.true:
        return CNF.from_iterable([])
    if expr is sp.false:
        return CNF.from_iterable([Clause.from_iterable([])])

    if isinstance(expr, sp.And):
        clause_exprs = list(expr.args)
    else:
        clause_exprs = [expr]

    clauses = []
    for ce in clause_exprs:
        if ce is sp.true:
            continue
        if ce is sp.false:
            clauses.append(Clause.from_iterable([]))
            continue
        if isinstance(ce, sp.Or):
            lit_exprs = list(ce.args)
        else:
            lit_exprs = [ce]

        raw_literals = [_parse_sympy_literal(le, symbol_to_var) for le in lit_exprs]

        # Canonicalise literal order and remove duplicates.
        unique = {(lit.var, lit.value): lit for lit in raw_literals}
        literals = sorted(unique.values(), key=lambda l: (l.var, l.value))
        clauses.append(Clause.from_iterable(literals))

    # Canonicalise clause order and remove duplicates.
    def clause_sig(c: Clause) -> Tuple[Tuple[int, int], ...]:
        return tuple((lit.var, lit.value) for lit in c.literals)

    unique_clauses = {clause_sig(c): c for c in clauses}
    ordered = [unique_clauses[k] for k in sorted(unique_clauses.keys())]
    return CNF.from_iterable(ordered)


def sympy_to_dnf(expr, *, symbols: Optional[Sequence[object]] = None) -> DNF:
    """Parse a SymPy expression that is already in DNF."""
    import sympy as sp

    if symbols is None:
        symbols = sorted(list(expr.free_symbols), key=lambda s: str(s))

    symbol_to_var = {sym: i for i, sym in enumerate(symbols)}

    if expr is sp.false:
        return DNF.from_iterable([])
    if expr is sp.true:
        return DNF.from_iterable([Term.from_iterable([])])

    if isinstance(expr, sp.Or):
        term_exprs = list(expr.args)
    else:
        term_exprs = [expr]

    terms = []
    for te in term_exprs:
        if te is sp.false:
            continue
        if te is sp.true:
            terms.append(Term.from_iterable([]))
            continue
        if isinstance(te, sp.And):
            lit_exprs = list(te.args)
        else:
            lit_exprs = [te]

        raw_literals = [_parse_sympy_literal(le, symbol_to_var) for le in lit_exprs]

        unique = {(lit.var, lit.value): lit for lit in raw_literals}
        literals = sorted(unique.values(), key=lambda l: (l.var, l.value))
        terms.append(Term.from_iterable(literals))

    def term_sig(t: Term) -> Tuple[Tuple[int, int], ...]:
        return tuple((lit.var, lit.value) for lit in t.literals)

    unique_terms = {term_sig(t): t for t in terms}
    ordered = [unique_terms[k] for k in sorted(unique_terms.keys())]
    return DNF.from_iterable(ordered)
