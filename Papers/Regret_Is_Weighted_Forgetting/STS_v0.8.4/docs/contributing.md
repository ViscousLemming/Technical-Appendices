<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Contributing

This is a correctness first codebase.
If you want to add features, the expectation is.

- you add tests
- you explain semantics in plain english
- you do not change Layer 1 meanings without updating the specification and the proof obligations

## Development workflow

Run tests before committing.

```bash
pytest -q
```

Run a tiny benchmark sweep when you change hot code.

```bash
python -m stacktheory.layer5.benchmarks --budget tiny --repeat 1
```

## Adding new learners

A new learner should.

- accept a Task
- only return a policy mask that is in the induced language
- evaluate correctness using is_correct_policy or PolicyEvaluator
- document what proxy or heuristic it tries to optimise

## Adding new representations

A new representation must have a clear equivalence mapping to the Layer 1 int bitset.
Tests should include round trip conversions and representation invariance.
