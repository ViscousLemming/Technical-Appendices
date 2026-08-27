<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Stack Theory Suite

This repository contains a reference implementation of Layers 1 to 5 of the Stack Theory Suite.

Layer 1 implements core Stack Theory objects for finite environments and finite vocabularies.

- Environment, programs, vocabularies
- Statements and truth sets
- Completions, extensions, equivalence, weakness
- Abstractor function
- Description length
- Boolean cube logic helpers for literals, clauses, CNF, DNF
- Packed bitset and boolean tensor conversions for extensional sets

Layer 2 adds interoperability and convenience interfaces.

- Ternary boolean tensor encoding and validation
- DIMACS interchange for CNF
- Human readable CNF and DNF rendering and parsing
- Optional SymPy bridge
- Packed uint64 bitset tensors for fast bitwise operations
- Word packed bitset tensors that support CPU and GPU devices
- SAT solving for CNF with an internal DPLL backend and a fast truth table backend for small n

Layer 3 implements the task and learning objects from the appendices.

- Tasks and the child relation
- Policy correctness
- Brute force learning under weakness and description length proxies
- Minimal description length search that stops early when a short correct policy exists
- Custom heuristic keys for lexicographic combinations of weakness and simplicity
- Additional learners that can reduce work compared to brute force
  - best first search
  - branch and bound
  - beam search
  - genetic search
  - random search baseline
  - hill climbing
  - simulated annealing
  - reinforcement learning scaffolding for policy construction

See docs/search-algorithms.md for plain English definitions of these learners and their parameters.

Layer 4 adds experiment and scaling utilities.

- A Language wrapper for the induced language L_v
  - Use layer3 candidates to restrict policy search without redefining L_v
- Index based environments for large finite Phi without storing explicit states
- SAT backed induced language tooling for large Boolean cubes
- Packed vocabulary backend for GPU accelerated extensional operations
- Packed vocabulary backend for large finite environments with batched bitwise operations

Layer 5 is the quality system.

- Golden fixtures for small hand checkable vocabularies
- Property style tests for algebraic laws and representation invariance
- Lightweight benchmarks for hot paths and end to end learning loops

This is a correctness first implementation.

## What is novel

- A single reference implementation of Stack Theory Layer 1 semantics for finite environments
- Exact induced language membership, extension, and weakness computations that do not collapse into simplicity
- A learning layer that selects correct policies using weakness, description length, or any custom proxy
- A tight integration between extensional set semantics, SAT tooling, and reproducible learning experiments
- A built in quality system that treats the definitions as executable specifications

## Start here

- If you need definitions and notation, read docs/glossary.md
- If you want the semantic core, read docs/core-concepts.md
- If you want to run learning experiments, read docs/learning.md

## Reader paths

### If you are here for the math and theory

- Read docs/glossary.md
- Then read docs/core-concepts.md
- Then read docs/provenance.md
- Then read docs/learning.md and docs/heuristics.md
- Then read paper.md if you want the write up

### If you are here to run code and extend the library

- Read docs/installation.md
- Then read docs/api.md
- Run the Quick start in this README
- Then read docs/search-algorithms.md
- Then read docs/benchmarks-and-quality.md

### If you are reviewing and want to verify claims quickly

- Run pytest -q
- Run the tiny benchmarks in Layer 5
- Open docs/provenance.md and spot check the definition to test mapping
- Check that the core equations in paper.md match docs/glossary.md and the code



## Release notes

0.8.4

- Added Reader paths so math readers, engineering readers, and reviewers can start quickly
- Updated the paper financial support statement. No specific funding was received

0.8.3

- Clarity and correctness audit. Fixed a documentation math error about weakness monotonicity
- Added missing glossary entries for abstractor, statement equivalence, child tasks, and search algorithms
- Added a new search algorithms doc for cross-disciplinary readers
- Removed cached build artefacts from the release archive and added a standard .gitignore
- Fixed a Python docstring escape warning and tightened a misleading Statement docstring


0.8.2

- Hard clarity and provenance pass on the documentation
- Added docs/glossary.md and docs/provenance.md
- Expanded learning and heuristics documentation with plain English definitions and equations
- Expanded benchmark documentation including parameter explanations and a plotting snippet

0.8.1

- Enforced the task well formedness condition O_alpha subset Ext(I_alpha)
- Tightened error messages and validation around tasks and policies

0.8.0

- Switched license to Apache 2.0 and added a NOTICE file
- Added author and copyright headers to every source and documentation file
- Expanded plain english comments that map code back to the Stack Theory math

0.7.2

- Truth table SAT backend now caches the all ones assignment universe bitset per n_vars
- Language.size no longer copies the induced language list when the language is induced
- Added a docs folder with a complete user guide and API level documentation

0.7.1

- BooleanCubeEnvironment literal masks are now constructed using a repeated pattern method that avoids an O(2^n) loop for small variable indices
- WordBitset canonicalisation avoids an unconditional clone when the input words are already canonical

## Documentation

See docs for the full user guide and API documentation.
Start with docs/index.md.

## Installation

Install from source.

```bash
pip install .
```

If you want to run the quality suite.

```bash
pip install .[quality]
pytest -q
```

For more detail see docs/installation.md.

## Quick start

```python
from stacktheory.layer1 import BooleanCubeEnvironment, Vocabulary, Program

env = BooleanCubeEnvironment(n=3)
p0 = env.literal_program(var=0, value=0)
p1 = env.literal_program(var=1, value=1)

v = Vocabulary(env, programs=[p0, p1], names=["x0=0", "x1=1"])
s = v.statement(["x0=0"])

truth = s.truth_set()
print(truth.cardinality())
print(v.weakness(s))
```

## Learning and heuristics

Layer 3 represents a task as inputs and acceptable outputs, both subsets of the
induced language.

Learning selects a correct policy from the induced language.
The default proxy is the weakness proxy.

You can also combine weakness and simplicity in a lexicographic priority order
by supplying a key function.

```python
from stacktheory.layer3 import Task, learn, learn_min_description_length, make_lexicographic_key

# Build a small vocabulary.
env = BooleanCubeEnvironment(n=2)
p0 = env.literal_program(var=0, value=1)
p1 = env.literal_program(var=1, value=1)
v = Vocabulary(env, programs=[p0, p1], names=["x0=1", "x1=1"])

# Task with one input and one acceptable output.
i = v.statement(["x0=1"])
o = v.statement(["x0=1", "x1=1"])
task = Task.from_statements([i], [o])

# Weakness first, then simplicity.
key = make_lexicographic_key([("weakness", "max"), ("description_length", "min")])
pi = learn(task, heuristic=key)

# Exact minimal description length learning.
pi_mdl = learn_min_description_length(task)
```

## SAT solving

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

## Induced language wrapper

Layer 4 provides a small `Language` wrapper around a Layer 1 `Vocabulary`.
It does not redefine the induced language.
Extension and weakness are still computed as completions in the induced language.

```python
from stacktheory.layer1 import BooleanCubeEnvironment, Vocabulary
from stacktheory.layer4 import Language

env = BooleanCubeEnvironment(n=1)
progs = [env.literal_program(var=0, value=1), env.literal_program(var=0, value=0)]
vocab = Vocabulary(env, programs=progs, names=["x0=1", "x0=0"])

L = Language(vocab)

# Mask 3 means {x0=1, x0=0} which is unsatisfiable so it is not in L_v.
print(L.is_in_language_mask(3))
print(L.completion_masks_for_subset_mask(3))
```

If you want to restrict learning to a finite candidate set Q, keep the language
fixed and pass Q into the learner as `candidates`.
That matches the appendix definition where Q is a subset of L_v.

```python
from stacktheory.layer3 import Task, learn

env = BooleanCubeEnvironment(n=2)
p0 = env.literal_program(var=0, value=1)
p1 = env.literal_program(var=1, value=1)
vocab = Vocabulary(env, programs=[p0, p1], names=["x0=1", "x1=1"])

task = Task(vocab=vocab, inputs=frozenset({1}), outputs=frozenset({3}))

# Only consider policies in Q = {2, 3}.
pi = learn(task, candidates=[2, 3])
print(pi)
```


## Benchmarks

A quick smoke benchmark run uses the tiny budget.

```bash
python -m stacktheory.layer5.benchmarks --budget tiny --repeat 1
```

## Tests

```bash
pytest -q
```

## License

This project is licensed under the Apache License, Version 2.0.
See LICENSE and NOTICE.

## Community and support

To report bugs or request features, open an issue in the repository issue tracker.
If you want to contribute, read CONTRIBUTING.md.
