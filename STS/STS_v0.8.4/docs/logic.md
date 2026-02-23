<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Logic objects

Layer 1 includes syntactic logic objects for Boolean cubes.
They are convenience wrappers.
The semantics are always extensional.

The core idea is.
You build a CNF or DNF object.
You compile it to a Program on a BooleanCubeEnvironment.
That Program is the exact truth set of the formula.

## BooleanCubeEnvironment

BooleanCubeEnvironment(n) represents Phi_n equals {0,1}^n.
The canonical enumeration maps each assignment to an integer state.
Variable i is the i th least significant bit of that integer.

## Literals

A Literal is x_var equals value, where value is 0 or 1.
It compiles to the set of assignments that satisfy it.

In code.

```python
from stacktheory.layer1 import BooleanCubeEnvironment
from stacktheory.layer1.logic import Literal

env = BooleanCubeEnvironment(n=3)
lit = Literal(var=1, value=1)
p = lit.to_program(env)
```

## Clauses and CNF

A Clause is a disjunction of literals.
A CNF is a conjunction of clauses.

Empty objects

- An empty clause is false everywhere.
- An empty CNF is true everywhere.

Compilation

- Clause.to_program returns the union of its literals.
- CNF.to_program returns the intersection of its clauses.

## Terms and DNF

A Term is a conjunction of literals.
A DNF is a disjunction of terms.

Empty objects

- An empty term is true everywhere.
- An empty DNF is false everywhere.

Compilation

- Term.to_program returns the intersection of its literals.
- DNF.to_program returns the union of its terms.

## Thesis style ternary tensor encoding

The suite supports the ternary encoding used in your thesis prototype code.

Clause encoding

- Shape is (n_vars, 2)
- t[var, 0] is True if the clause contains the negative literal x_var equals 0
- t[var, 1] is True if the clause contains the positive literal x_var equals 1

CNF encoding

- Shape is (n_clauses, n_vars, 2)
- It is the stacked clause encoding

DNF encoding

- Shape is (n_terms, n_vars, 2)
- It is the stacked term encoding

Evaluation

Layer 1 implements vectorised evaluators that operate on batches of complete assignments encoded as one hot tensors.

Plain English. A one hot encoding stores each boolean variable as two slots.
Exactly one slot is True.
Slot 0 means the variable is 0.
Slot 1 means the variable is 1.

Vectorised means the code evaluates many assignments at once using tensor operations instead of a Python loop.

- cnf_tensor_satisfied
- dnf_tensor_satisfied

## Pretty printing and parsing

Layer 2 provides a small human readable parser.
This is intended for experiments and debugging, not as a full logic language.

- cnf_to_str
- dnf_to_str
- parse_cnf
- parse_dnf

## DIMACS

Layer 2 supports DIMACS CNF import and export.
This is a standard interchange format for SAT solvers.

- cnf_to_dimacs
- cnf_from_dimacs

## SymPy bridge

Layer 2 includes an optional SymPy bridge.
The core library does not depend on SymPy.
If SymPy is installed, you can convert between Stack Theory CNF and DNF objects and SymPy boolean expressions.

- cnf_to_sympy
- dnf_to_sympy
- sympy_to_cnf
- sympy_to_dnf
