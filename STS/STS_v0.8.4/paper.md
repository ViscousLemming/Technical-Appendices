---
title: 'Stack Theory Suite'
author: "Michael Timothy Bennett"

tags:
  - Python
  - formal semantics
  - satisfiability
  - learning theory
  - cognitive science
authors:
  - name: Michael Timothy Bennett
    affiliation: 1
affiliations:
  - name: The Australian National University, Canberra, Australia
    index: 1
date: 9 February 2026
bibliography: paper.bib
---


# Abstract

Stack Theory is a unifying formal mathematical framework for studying adaptive systems, whether they be computational, biological, human organisation or any other sort of system. Here I present Stack Theory Suite, which turns Stack Theory definitions into runnable code so that results can be tested and falsified. It serves as a correctness first reference library for implementations of the formal Stack Theory objects, using efficient bitset implementations in place of the set and tensor implementations used in earlier experiments.

URL: https://github.com/ViscousLemming/Technical-Appendices/STS

# Introduction


Stack Theory is a mathematical formalism of systems in general. It describes a system as a stack of abstraction layers. For example software is interpreted by hardware, which is interpreted by the physical world according to the laws of physics. Likewise, a living organism can be framed as a behaviour of organs, which are in turn a behaviour of cells. An army company is made of platoons, then of squads and finally of soldiers.  
Each layer describes the same underlying world using a different vocabulary.
Stack Theory formalises these vocabularies in stacks.
Vocabularies are sets of potential facts called programs, and each program is a set of physical microstates in which some condition holds.
Within this framework goal directed, task oriented behaviour is expressed using sets of facts as constraints over microstates.
The reason a reference library of code is needed is because the Stack Theory definitions are short and implementations are easy to get subtly wrong.
Incorrect implementations tend to confound weakness and simplicity, producing a misleading collapse between the two [@bennett2024b].
So Stack Theory Suite is a correctness first Python library that makes the core Stack Theory objects executable for finite environments and finite vocabularies [@bennett2025thesis; bennett2022c; bennett2024a; @bennett2024c; bennett2024b; @bennett2025d; bennett2025e; @bennett2023b; bennett2023c; @bennett2026a; @bennett2026b; @bennett2026c; @perrierbennett2026a; @sole2026cognitionspacesnaturalartificial].
The library represents all semantic objects extensionally as truth sets.
This means every policy and every statement denotes a concrete set of states, allowing formal definitions testable in code.

The formal mathematical definitions of the objects which Stack Theory Suite implements are available in the Technical Appendices GitHub [@bennett_appendix], alongside the Stack Theory Suite itself. The following library description assumes a passing familiarity. The library includes

- An exact extensional implementation of induced languages, extensions, and weakness for finite environments
- Learning utilities that select correct policies using weakness and other proxies under the exact correctness equation
- A layered design that keeps the semantic core small and testable while still supporting SAT and tensor interop
- A quality system that treats the formal definitions as executable specifications

# Statement of need

Stack Theory uses induced languages.
An induced language is built from a vocabulary, which is a finite set of programs.
The induced language is the set of all consistent statements over that vocabulary.
Weakness is defined as the cardinality of the extension of a statement in the induced language.
Description length is defined as the size of the statement, which is a simple proxy for simplicity.

These objects are combinatorial.
Naively enumerating completions can be exponential in the vocabulary size.
At the same time, approximating them can silently change the theory.
Researchers need a single reference implementation that

- Keeps the extensional meaning of every object explicit
- Makes induced language membership explicit and checkable
- Computes weakness as extension cardinality in the induced language
- Exposes a clear interface for tasks, policies, and learning objectives
- Provides tests that fail when a definition is implemented incorrectly

Stack Theory Suite is designed to be that implementation target.
It is intended for authors who want to validate Stack Theory theorems on small cases, build reproducible experiments, and integrate Stack Theory semantics into larger workflows without changing the definitions.

# State of the field

Stack Theory connects learning, abstraction, and semantics.
It borrows tools from logic and satisfiability.
The core induced language condition is a satisfiability condition on a conjunction.
For Boolean cube environments this can be checked with standard SAT ideas such as DPLL style search [@davis1962dpll].

There are excellent general purpose solvers and symbolic systems.
Stack Theory Suite is not a competitor to them.
Its role is narrower.
It provides Stack Theory specific data structures and the exact objective functions needed to run Stack Theory learning rules.

For example, general purpose SAT and SMT tools such as Z3, and Python toolkits such as PySAT, solve satisfiability problems at scales far beyond exhaustive enumeration [@demoura2008z3; @ignatiev2018pysat].
Symbolic systems such as SymPy can manipulate logic expressions and simplify formulas [@meurer2017sympy].
Stack Theory Suite focuses on the Stack Theory meanings and objective functions.
It also provides adapters to interchange CNF with standard formats such as DIMACS.

# Software design

## Layered structure

The library is organised into Layers 1 to 5.

Layer 1 implements the extensional semantics.
It includes environments, programs, vocabularies, statements, truth sets, completions, extensions, weakness, and description length.

Layer 2 adds interoperability helpers for Boolean cube logic and CNF formats.

Layer 3 adds learning utilities, including standard lexicographic tie breaking rules that combine weakness and description length into a total order.

Layer 4 adds induced language utilities that are designed for SAT style representations when the environment is too large to enumerate.

Layer 5 adds golden tests and small benchmark tasks that are useful for regression testing and tutorial examples.

The library is pure Python.
It uses NumPy and PyTorch for tensor conversion utilities and batch evaluation helpers [@harris2020numpy; @paszke2019pytorch].

## Core computations

Given a finite environment \(\Phi\) and vocabulary \(\mathfrak{v} \subseteq 2^{\Phi}\), the suite implements

- Truth sets \(T(\ell) = \bigcap_{p \in \ell} p\)
- Induced language membership \(\ell \in L_\mathfrak{v}\) if and only if \(T(\ell)\) is nonempty
- Completions and extensions in \(L_\mathfrak{v}\)
- Weakness \(w(\ell) = |\mathrm{Ext}(\ell)|\) computed as a count of completions in \(L_\mathfrak{v}\)
- Description length \(|\ell|\) computed as the number of selected vocabulary elements

The suite includes both direct bitset evaluation for small Boolean cubes and an internal SAT backend for larger cubes.
The internal solver uses a DPLL style backtracking core for completeness and a bit parallel truth table backend for very small subproblems [@davis1962dpll].

# Quality assurance

The suite ships with unit tests and property tests.
These check that the extensional meaning of programs and statements matches the formal definitions.
They also check that weakness is computed as an extension count in the induced language, not as a shortcut such as \(2^{|\mathfrak{v}|-|\ell|}\), which anecdotally appears to happen frequently when code is generated using contemporary language models.
If a theorem only works on paper, the test suite will say so.

# Research impact statement

Stack Theory Suite provides a single citable implementation target for Stack Theory.
It lowers the cost of checking small cases, reproducing results, and comparing alternative learning rules under a shared semantics.
This makes disagreements more informative.
If two papers report different behaviour, it becomes easier to localise whether the difference is theory, vocabulary design, or implementation.

# Financial support

No specific funding was received for this work.

# AI usage disclosure

Generative models were used to help track down and fix bugs, repeatedly refactor code, redo comments and licensing boilerplate, fix inconsistent formatting and help maintain documentation as new functionality was added.
LLMs were not used to indiscriminately generate code or ideas.
This project began in 2022 with the experiments in [@bennett2022c], and has evolved slowly over several years.
Any change that involved LLMs has been painstakingly checked and rechecked.
The most significant change implemented with the assistance of LLMs was moving from the original boolean tensor implementation used in earlier published experiments to the new and more scalable bitset implementation.

# References
