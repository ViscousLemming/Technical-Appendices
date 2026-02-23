<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Stack Theory Suite documentation

This documentation is the user guide for the Stack Theory Suite.
It is written for people who want a practical toolkit that implements Stack Theory objects exactly for finite environments.

The guiding principle is simple.
Everything is extensional.
If two objects pick out the same states in the same environment, they are treated as the same thing.

## What you get

- Exact set semantics for finite environments using a canonical packed bitset representation
- Stack Theory objects such as environments, programs, vocabularies, statements, induced languages, completions, extensions, and weakness
- Boolean cube logic helpers including literals, clauses, CNF, DNF, and the thesis style ternary tensor encoding
- A native SAT solver interface with two backends
- Learning utilities that select correct policies using weakness, simplicity, or any custom heuristic
- A built in quality system with golden fixtures, property tests, and benchmarks

## What you do not get

- Symbolic algebra as a primary interface
- A claim that the internal DPLL solver beats state of the art external SAT solvers for large formulas
- A promise that exponential objects become polynomial

The library will happily tell you when you are asking it to do an exponential thing.

## The five layers

Layer 1 is the semantic core.
It is the smallest set of objects needed to model the Stack Theory definitions over a finite environment.

Layer 2 is representation and interoperability.
It adds tensor encodings, DIMACS, a SymPy bridge, packed word bitsets, and SAT solving.

Layer 3 is learning.
It adds tasks, correctness, heuristics, and a menu of learners.

Layer 4 is scaling support.
It adds embodied languages, index based environments, and batched packed operations.

Layer 5 is the quality system.
It adds golden fixtures, property tests, and benchmarks that lock down correctness.

## Where to start

- If you want the definitions and notation in one place, read glossary.md.
- If you want to understand the objects and their exact meaning, read core-concepts.md.
- If you want to work with CNF and SAT, read logic.md and sat.md.
- If you want learning, read learning.md, search-algorithms.md, and heuristics.md.
- If you want performance advice, read representations.md.
