<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# SAT solving

Layer 2 provides SAT solving utilities for CNF formulas.
The focus is.
Correctness.
Portability.
A fast path for small numbers of variables.

The solver is designed to work well with Stack Theory representations.
For small n_vars, extensional truth table reasoning is extremely fast.
For large n_vars, any truth table approach is impossible.

## API

Use sat_solve_cnf.

```python
from stacktheory.layer1.logic import Literal, Clause, CNF
from stacktheory.layer2 import sat_solve_cnf

cnf = CNF.from_iterable([
    Clause.from_iterable([Literal(var=0, value=1)]),
    Clause.from_iterable([Literal(var=0, value=0), Literal(var=1, value=1)]),
])

res = sat_solve_cnf(cnf, n_vars=2)
print(res.satisfiable, res.model_state)
```

## Backends

The solver chooses a backend based on n_vars.

Truth table backend

If n_vars is at most truth_table_max_vars, the solver compiles the CNF to a Program on BooleanCubeEnvironment.
It then reads off a satisfying assignment by locating any true bit.

This is fast because it reduces the problem to a small number of big integer bitwise operations.
It is complete and exact.

DPLL backend

If n_vars is larger than truth_table_max_vars, the solver uses an internal DPLL style algorithm.
It supports.

- unit propagation
- pure literal elimination
- a simple occurrence based branching heuristic

Plain English definitions

Unit propagation means
If a clause has only one way left to be satisfied, you force that choice.

Pure literal elimination means
If a variable appears with only one polarity, you can set it to satisfy all those clauses and never regret it.

Branching heuristic means
When you have to guess, you pick a variable to split on.
Occurrence based means you often pick a variable that shows up in many clauses.

This is exact but it is not meant to beat highly optimised external solvers.

## Large neighborhood search

For large n_vars, sat_solve_cnf can optionally try a heuristic first.
Set lns_first True.

Large neighborhood search works by.

- sampling a full assignment
- choosing a small subset of variables to leave free
- simplifying the CNF under the fixed variables
- solving the reduced CNF by the fast truth table backend

This can find a model quickly for some structured formulas.
If it fails, it returns unknown and the solver falls back to exact DPLL.

## When the native solver is unusually fast

The truth table backend is unusually fast when.

- n_vars is moderate, typically up to around 20
- clauses are not too many
- you need a satisfying assignment, not a count

This is the regime where Stack Theory bit representations can beat generic symbolic pipelines.
