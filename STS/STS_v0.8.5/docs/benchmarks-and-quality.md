<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Quality system and benchmarks

Layer 5 exists to keep the suite trustworthy.
It treats the mathematics as executable specifications.

It provides three kinds of checks.

Golden fixtures  
Small hand checkable cases with expected outputs.

Property tests  
Randomised tests of algebraic laws and representation invariants.

Benchmarks  
Repeatable timing checks for hot paths and end to end workflows.

## Validation and verification

Plain English. Validation asks whether you built the right thing.
Verification asks whether you built the thing right.

This repository does both.

Validation  
The public docs define the mathematical objects in plain language and in symbols.
Those definitions match the appendix definitions for finite environments.

Verification  
The test suite checks that the implementation obeys those definitions on toy cases.
If a definition is implemented incorrectly, tests fail.

## Running tests

Run pytest.

```bash
pytest -q
```

Doctests are enabled.
That means some examples in docstrings are executed as tests.

## Golden fixtures

Golden fixtures are small vocabularies and expected results that can be inspected by a human.
They are intended to catch off by one errors in enumeration and representation.

Example.
A fixture can say that the completions of a statement are exactly a certain set of masks.
You can check that by hand on a three state environment.

## Property tests

Property tests check laws that should always hold.
Examples include.

- set operation laws for `Program` and `WordBitset`
- equivalence of representations such as int bitsets and packed word tensors
- SAT solver agreement with truth table semantics on small formulas

Property tests do not prove correctness.
They are a strong regression net.
They make it hard for subtle bugs to survive unnoticed.

## Benchmarks

Benchmarks are development tools.
They answer one question.
Did a recent change make the hot paths slower.

Run the benchmark module.

```bash
python -m stacktheory.layer5.benchmarks --budget tiny --repeat 1
```

### Benchmark parameters

`--budget`  
Selects problem sizes.
Budgets are `tiny`, `small`, and `medium`.
Tiny is designed to finish in a few seconds.

`--repeat`  
Number of repetitions per benchmark.
The harness reports the minimum time across repeats.
This reduces noise.

`--seed`  
Seed for randomised benchmark inputs.

`--device`  
Torch device used for tensor based benchmarks.
Default is cpu.

### Benchmark output

The benchmark runner prints one tab separated line per benchmark.

The columns are.

- name  
- seconds  
- iterations  
- seconds_per_iter

`seconds_per_iter` is computed as `seconds / iterations`.

### Plotting benchmark results

The suite does not ship plots.
It prints machine readable output so you can plot however you like.

Example that turns the output into a bar chart.

```python
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
import subprocess
import sys

out = subprocess.check_output(
    [sys.executable, "-m", "stacktheory.layer5.benchmarks", "--budget", "tiny", "--repeat", "1"],
    text=True,
)

df = pd.read_csv(StringIO(out), sep="\t", header=None, names=["name", "seconds", "iterations", "seconds_per_iter"])
df = df.sort_values("seconds_per_iter", ascending=False)

plt.figure()
plt.bar(df["name"], df["seconds_per_iter"])
plt.xticks(rotation=90)
plt.ylabel("seconds per iteration")
plt.tight_layout()
plt.show()
```

If you want stable plots across machines, record both the hardware and the Python and PyTorch versions.
