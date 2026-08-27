# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Internal SAT solving utilities.

This module implements a small SAT solver for CNF formulas.

Why SAT appears in this suite

In Stack Theory a statement is valid when its truth set is not empty.
When the environment is a Boolean cube with a huge number of variables,
we cannot store truth sets extensionally as bitsets.
A SAT solver gives an exact membership test for the induced language in
that large n regime.

So in code you will often see this module used for one precise job.
Given a set of clauses, decide if there exists any assignment that
satisfies all of them.
That is the same as checking that the truth set is not empty.

Design goals

- Correctness first.
- Very fast for small and medium n.
- Still useful for larger n via large neighbourhood search.

Strategy

- For small n we use an exact truth table bitset method.
  It checks all assignments in parallel using integer bit ops.
- For larger n we use a simple DPLL style backtracking solver.
- Optionally we run a large neighbourhood search loop.
  That can solve large instances by fixing most variables and solving
  the remaining subproblem many times.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import random
from typing import Dict, List, Optional, Sequence, Tuple

from stacktheory.layer1.environment import BooleanCubeEnvironment
from stacktheory.layer1.logic import CNF


@dataclass(frozen=True)
class SATResult:
    """Result of a SAT solve."""

    satisfiable: bool
    n_vars: int
    model_state: Optional[int] = None


@dataclass(frozen=True)
class SATHeuristicResult:
    """Result of a heuristic SAT attempt.

    status
        Either "sat" or "unknown".
        This object never claims "unsat" because the heuristic does not prove it.
    model_state
        A satisfying assignment encoded as an integer bit mask.
        Bit i is 1 when variable i is true.
    """

    status: str
    n_vars: int
    model_state: Optional[int] = None
    tries: int = 0


@dataclass(frozen=True)
class LNSConfig:
    """Large neighborhood search configuration.

    This is a pragmatic speed hack for large n_vars.
    It repeatedly fixes most variables and solves a smaller free subproblem using
    the fast truth table backend.

    Important caveat

    This method is not complete.
    If it fails to find a model, that does not imply unsatisfiability.
    """

    free_vars: int = 20
    max_iters: int = 2000
    restarts: int = 32
    seed: int = 0
    focus_on_unsatisfied_clause: bool = True


def _infer_n_vars_from_cnf(cnf: CNF) -> int:
    max_var = -1
    for clause in cnf.clauses:
        for lit in clause.literals:
            if lit.var > max_var:
                max_var = lit.var
    return max_var + 1


def _cnf_to_clause_masks(cnf: CNF, n_vars: int) -> List[Tuple[int, int]]:
    """Convert a CNF into a list of (pos_mask, neg_mask) clauses.

    pos_mask has bit i set when variable i appears positively.
    neg_mask has bit i set when variable i appears negatively.
    """
    clauses: List[Tuple[int, int]] = []
    for clause in cnf.clauses:
        pos = 0
        neg = 0
        for lit in clause.literals:
            if lit.var < 0 or lit.var >= n_vars:
                raise ValueError("literal var out of range for n_vars")
            bit = 1 << lit.var
            if lit.value == 1:
                pos |= bit
            else:
                neg |= bit
        # Tautology clause can be dropped.
        if (pos & neg) != 0:
            continue
        clauses.append((pos, neg))
    return clauses


def _truth_table_solve(cnf: CNF, n_vars: int) -> SATResult:
    env = BooleanCubeEnvironment(n=n_vars)
    program = cnf.to_program(env)
    if program.is_empty():
        return SATResult(satisfiable=False, n_vars=n_vars, model_state=None)

    # Find a witness state.
    bits = program.bitset
    lsb = bits & -bits
    state = lsb.bit_length() - 1
    return SATResult(satisfiable=True, n_vars=n_vars, model_state=state)


@lru_cache(maxsize=64)
def _literal_assignment_mask(n_free: int, var: int, value: int) -> int:
    """Return the bitset of satisfying assignments for a literal over n_free variables.

    Universe assignments are integers 0..2^n_free-1.
    Bit a is 1 if and only if assignment a makes the literal true.

    The bitset is represented as a python integer.
    """
    if n_free < 0:
        raise ValueError("n_free must be non negative")
    if n_free == 0:
        return 0
    if var < 0 or var >= n_free:
        raise ValueError("var out of range")
    if value not in (0, 1):
        raise ValueError("value must be 0 or 1")

    size = 1 << n_free

    # Base pattern over one period.
    # step is the period length in assignments.
    block = 1 << var
    step = block << 1
    half = (1 << block) - 1
    pat = (half << block) if value == 1 else half

    # Repeat the base pattern to reach size bits.
    # Because size and step are both powers of two, the repeat count is also a power of two.
    out = pat
    current = step
    while current < size:
        out |= out << current
        current <<= 1
    return out

@lru_cache(maxsize=16)
def _truth_table_universe_bits(n_free: int) -> int:
    """Return the all ones assignment universe bitset for n_free variables.

    Universe assignments are integers 0..2^n_free-1.
    The bitset is a Python int with 2^n_free bits.

    This is only used for the fast truth table backend when n_free is small.
    The result is cached because allocating this big integer repeatedly is expensive.
    """
    if n_free < 0:
        raise ValueError("n_free must be non negative")
    size = 1 << int(n_free)
    if size == 0:
        return 0
    return (1 << size) - 1



def _truth_table_solve_clause_masks(clauses: Sequence[Tuple[int, int]], n_free: int) -> SATResult:
    """Solve a CNF given as clause masks by truth table enumeration.

    clauses
        Sequence of (pos_mask, neg_mask) where each mask uses bits 0..n_free-1.
    """
    if n_free < 0:
        raise ValueError("n_free must be non negative")
    if n_free == 0:
        # Empty variable set. CNF is satisfiable iff it has no non-tautological empty clause.
        for pos, neg in clauses:
            if (pos | neg) == 0:
                return SATResult(satisfiable=False, n_vars=0, model_state=None)
        return SATResult(satisfiable=True, n_vars=0, model_state=0)

    size = 1 << n_free
    universe = _truth_table_universe_bits(n_free)
    bits = universe

    for pos, neg in clauses:
        clause_bits = 0
        p = pos
        while p:
            lsb = p & -p
            var = lsb.bit_length() - 1
            clause_bits |= _literal_assignment_mask(n_free, var, 1)
            p ^= lsb
        n = neg
        while n:
            lsb = n & -n
            var = lsb.bit_length() - 1
            clause_bits |= _literal_assignment_mask(n_free, var, 0)
            n ^= lsb

        bits &= clause_bits
        if bits == 0:
            return SATResult(satisfiable=False, n_vars=n_free, model_state=None)

    lsb = bits & -bits
    state = lsb.bit_length() - 1
    return SATResult(satisfiable=True, n_vars=n_free, model_state=state)


def _clauses_satisfied(clauses: Sequence[Tuple[int, int]], state: int) -> bool:
    """Return True if and only if assignment state satisfies all clauses."""
    for pos, neg in clauses:
        if (pos & state) or (neg & (~state)):
            continue
        return False
    return True


def _find_unsatisfied_clause(clauses: Sequence[Tuple[int, int]], state: int) -> Optional[Tuple[int, int]]:
    for pos, neg in clauses:
        if (pos & state) or (neg & (~state)):
            continue
        return pos, neg
    return None


def sat_solve_cnf_lns(
    cnf: CNF,
    *,
    n_vars: Optional[int] = None,
    config: Optional[LNSConfig] = None,
    truth_table_max_vars: int = 20,
) -> SATHeuristicResult:
    """Heuristic SAT solve using large neighborhood search.

    This function only returns a satisfying model or "unknown".
    It never claims unsatisfiability.

    The solver works by.

    - picking a full assignment
    - selecting a subset of variables to leave free
    - simplifying the CNF under the fixed variables
    - solving the reduced CNF by truth table enumeration over the free variables
    """
    if n_vars is None:
        n_vars = _infer_n_vars_from_cnf(cnf)
    if n_vars < 0:
        raise ValueError("n_vars must be non negative")

    if config is None:
        config = LNSConfig()
    if config.free_vars <= 0:
        raise ValueError("free_vars must be positive")
    if config.free_vars > truth_table_max_vars:
        raise ValueError("free_vars must be <= truth_table_max_vars")
    if config.max_iters <= 0:
        raise ValueError("max_iters must be positive")
    if config.restarts <= 0:
        raise ValueError("restarts must be positive")

    clauses = _cnf_to_clause_masks(cnf, n_vars)
    full_var_mask = (1 << n_vars) - 1

    rng = random.Random(config.seed)
    tries = 0

    def choose_free_vars(state: int) -> List[int]:
        if config.focus_on_unsatisfied_clause:
            unsat = _find_unsatisfied_clause(clauses, state)
        else:
            unsat = None

        chosen: List[int] = []
        chosen_set: Dict[int, None] = {}
        if unsat is not None:
            pos, neg = unsat
            vars_mask = pos | neg
            while vars_mask and len(chosen) < config.free_vars:
                lsb = vars_mask & -vars_mask
                var = lsb.bit_length() - 1
                chosen.append(var)
                chosen_set[var] = None
                vars_mask ^= lsb

        # Fill remaining slots uniformly.
        while len(chosen) < config.free_vars and len(chosen) < n_vars:
            var = rng.randrange(n_vars)
            if var in chosen_set:
                continue
            chosen.append(var)
            chosen_set[var] = None
        chosen.sort()
        return chosen

    for _restart in range(config.restarts):
        state = rng.getrandbits(n_vars) & full_var_mask

        for _it in range(config.max_iters):
            tries += 1

            if _clauses_satisfied(clauses, state):
                return SATHeuristicResult(status="sat", n_vars=n_vars, model_state=state, tries=tries)

            free_vars = choose_free_vars(state)
            free_mask = 0
            for v in free_vars:
                free_mask |= 1 << v

            fixed_mask = (~free_mask) & full_var_mask
            fixed_state = state & fixed_mask

            # Simplify clauses and remap free variables to 0..k-1.
            mapping: Dict[int, int] = {v: i for i, v in enumerate(free_vars)}
            simplified: List[Tuple[int, int]] = []

            conflict = False
            for pos, neg in clauses:
                # Clause satisfied by fixed assignment.
                # A negative literal ¬x_i is satisfied by the fixed part when i is fixed and x_i is false.
                if (pos & fixed_state) or (neg & fixed_mask & (~fixed_state)):
                    continue

                # Clause not satisfied by fixed vars, keep only free vars.
                pos_free = pos & free_mask
                neg_free = neg & free_mask
                if (pos_free | neg_free) == 0:
                    conflict = True
                    break

                pos2 = 0
                neg2 = 0
                p = pos_free
                while p:
                    lsb = p & -p
                    var = lsb.bit_length() - 1
                    pos2 |= 1 << mapping[var]
                    p ^= lsb
                n = neg_free
                while n:
                    lsb = n & -n
                    var = lsb.bit_length() - 1
                    neg2 |= 1 << mapping[var]
                    n ^= lsb
                simplified.append((pos2, neg2))

            if conflict:
                # Repair by random flip in an unsatisfied clause if one exists.
                unsat = _find_unsatisfied_clause(clauses, state)
                if unsat is None:
                    state = rng.getrandbits(n_vars) & full_var_mask
                    continue
                pos, neg = unsat
                vars_mask = (pos | neg)
                if vars_mask == 0:
                    state = rng.getrandbits(n_vars) & full_var_mask
                    continue
                lsb = vars_mask & -vars_mask
                var = lsb.bit_length() - 1
                state ^= 1 << var
                continue

            k = len(free_vars)
            res = _truth_table_solve_clause_masks(simplified, k)
            if res.satisfiable and res.model_state is not None:
                free_state = int(res.model_state)
                new_state = fixed_state
                for i, var in enumerate(free_vars):
                    if (free_state >> i) & 1:
                        new_state |= 1 << var

                # Verify.
                if _clauses_satisfied(clauses, new_state):
                    return SATHeuristicResult(status="sat", n_vars=n_vars, model_state=new_state, tries=tries)

            # No model found in this neighborhood. Make a small change.
            unsat = _find_unsatisfied_clause(clauses, state)
            if unsat is None:
                state = rng.getrandbits(n_vars) & full_var_mask
                continue
            pos, neg = unsat
            vars_mask = (pos | neg)
            if vars_mask == 0:
                state = rng.getrandbits(n_vars) & full_var_mask
                continue
            # Flip a random variable in the clause.
            bit_positions: List[int] = []
            tmp = vars_mask
            while tmp:
                lsb = tmp & -tmp
                bit_positions.append(lsb.bit_length() - 1)
                tmp ^= lsb
            var = bit_positions[rng.randrange(len(bit_positions))]
            state ^= 1 << var

    return SATHeuristicResult(status="unknown", n_vars=n_vars, model_state=None, tries=tries)


def _dpll_solve(clauses: Sequence[Tuple[int, int]], n_vars: int) -> SATResult:
    """Internal DPLL SAT solver using bitmasks.

    This is a pragmatic implementation for moderate sized CNFs.
    It uses

    - unit propagation
    - pure literal elimination
    - a simple occurrence based branching heuristic
    """
    full_var_mask = (1 << n_vars) - 1

    # If CNF has an explicit empty clause, it is unsatisfiable.
    for pos, neg in clauses:
        if (pos | neg) == 0:
            return SATResult(satisfiable=False, n_vars=n_vars, model_state=None)

    def propagate(true_mask: int, false_mask: int) -> Optional[Tuple[int, int]]:
        """Apply unit propagation and pure literal elimination.

        Returns updated (true_mask, false_mask) or None if a conflict occurs.
        """
        while True:
            changed = False
            assigned = true_mask | false_mask
            unassigned = (~assigned) & full_var_mask

            # Unit propagation.
            for pos, neg in clauses:
                # Skip satisfied clauses.
                if (pos & true_mask) or (neg & false_mask):
                    continue

                un = (pos | neg) & unassigned
                if un == 0:
                    # Clause is unsatisfied.
                    return None

                if un.bit_count() == 1:
                    bit = un
                    # Determine which polarity is present.
                    if bit & pos:
                        # Need var True.
                        if bit & false_mask:
                            return None
                        if not (bit & true_mask):
                            true_mask |= bit
                            changed = True
                    else:
                        # Need var False.
                        if bit & true_mask:
                            return None
                        if not (bit & false_mask):
                            false_mask |= bit
                            changed = True

            # Pure literal elimination.
            assigned = true_mask | false_mask
            unassigned = (~assigned) & full_var_mask
            pos_seen = 0
            neg_seen = 0
            for pos, neg in clauses:
                if (pos & true_mask) or (neg & false_mask):
                    continue
                pos_seen |= pos & unassigned
                neg_seen |= neg & unassigned

            pure_pos = pos_seen & ~neg_seen
            pure_neg = neg_seen & ~pos_seen
            if pure_pos:
                true_mask |= pure_pos
                changed = True
            if pure_neg:
                false_mask |= pure_neg
                changed = True

            if not changed:
                return true_mask, false_mask

    def all_satisfied(true_mask: int, false_mask: int) -> bool:
        for pos, neg in clauses:
            if (pos & true_mask) or (neg & false_mask):
                continue
            return False
        return True

    def choose_branch_var(true_mask: int, false_mask: int) -> int:
        assigned = true_mask | false_mask
        unassigned = (~assigned) & full_var_mask
        if unassigned == 0:
            return 0

        # Occurrence count heuristic.
        counts = {}
        for pos, neg in clauses:
            if (pos & true_mask) or (neg & false_mask):
                continue
            un = (pos | neg) & unassigned
            while un:
                lsb = un & -un
                counts[lsb] = counts.get(lsb, 0) + 1
                un ^= lsb
        if not counts:
            # Fall back to first unassigned.
            return unassigned & -unassigned
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def dpll(true_mask: int, false_mask: int) -> Optional[Tuple[int, int]]:
        propagated = propagate(true_mask, false_mask)
        if propagated is None:
            return None
        true_mask, false_mask = propagated

        if all_satisfied(true_mask, false_mask):
            return true_mask, false_mask

        bit = choose_branch_var(true_mask, false_mask)
        if bit == 0:
            return None

        # Try True then False.
        res = dpll(true_mask | bit, false_mask)
        if res is not None:
            return res
        return dpll(true_mask, false_mask | bit)

    out = dpll(0, 0)
    if out is None:
        return SATResult(satisfiable=False, n_vars=n_vars, model_state=None)

    true_mask, false_mask = out
    # Choose False for unassigned variables.
    state = true_mask
    return SATResult(satisfiable=True, n_vars=n_vars, model_state=state)


def sat_solve_cnf(
    cnf: CNF,
    *,
    n_vars: Optional[int] = None,
    truth_table_max_vars: int = 20,
    lns_first: bool = False,
    lns_config: Optional[LNSConfig] = None,
) -> SATResult:
    """Solve CNF satisfiability and return a satisfying assignment if one exists.

    Parameters
    ----------
    cnf
        CNF formula.
    n_vars
        Number of variables. If None it is inferred from the CNF.
    truth_table_max_vars
        If n_vars is at most this value, the solver uses a fast truth table
        bitset backend.
        Otherwise it uses an internal DPLL solver.
    lns_first
        If True and n_vars is larger than truth_table_max_vars, attempt a fast
        large neighborhood search first.
        If a model is found, it is returned immediately.
        If no model is found, the solver falls back to exact DPLL.
    lns_config
        Configuration for the large neighborhood search.
    """
    if n_vars is None:
        n_vars = _infer_n_vars_from_cnf(cnf)
    if n_vars < 0:
        raise ValueError("n_vars must be non negative")

    if n_vars <= truth_table_max_vars:
        return _truth_table_solve(cnf, n_vars)

    if lns_first:
        h = sat_solve_cnf_lns(cnf, n_vars=n_vars, config=lns_config, truth_table_max_vars=truth_table_max_vars)
        if h.status == "sat" and h.model_state is not None:
            return SATResult(satisfiable=True, n_vars=n_vars, model_state=h.model_state)

    clauses = _cnf_to_clause_masks(cnf, n_vars)
    return _dpll_solve(clauses, n_vars)


def sat_is_satisfiable(
    cnf: CNF,
    *,
    n_vars: Optional[int] = None,
    truth_table_max_vars: int = 20,
) -> bool:
    """Return True if and only if the CNF is satisfiable."""
    return sat_solve_cnf(cnf, n_vars=n_vars, truth_table_max_vars=truth_table_max_vars).satisfiable
