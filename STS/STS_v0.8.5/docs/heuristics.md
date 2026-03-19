<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Heuristics

This page explains how the library ranks policies once they are known to be correct.

If you want the definition of policy and correctness, read learning.md first.
If you want the definitions of weakness and description length, read glossary.md.

## What a heuristic is

Plain English. A heuristic is a scoring rule for candidate policies.
Higher scores mean better policies.

In code, a heuristic is a function that maps `PolicyStats` to a tuple of integers.
Python compares tuples lexicographically.
That means it compares the first entry.
If those tie, it compares the second.
If those tie, it compares the third.

So a key like `(10, -2)` is better than `(9, 100)` because 10 is greater than 9.

## What is inside PolicyStats

`PolicyStats` contains the quantities you usually care about.

- `mask` which is the statement mask
- `weakness` which is \(w(\pi)\)
- `description_length` which is \(|\pi|\)

These are computed exactly for finite vocabularies.

## Built in keys

The library provides two common built in keys.

`key_weakness_then_simplicity`  
Prefer larger weakness.
Break ties by shorter description length.
Break ties by smaller mask so results are deterministic.

`key_simplicity_then_weakness`  
Prefer shorter description length.
Break ties by larger weakness.
Break ties by smaller mask.

These match the usual Stack Theory experimental comparisons.
Weakness versus simplicity only makes sense if ties are broken deterministically.

## Building lexicographic keys

`make_lexicographic_key` builds a key from a priority list.

Example that prefers weakness first and simplicity second.

```python
from stacktheory.layer3 import make_lexicographic_key

key = make_lexicographic_key([
    ("weakness", "max"),
    ("description_length", "min"),
])
```

Example that reverses the priorities.

```python
key = make_lexicographic_key([
    ("description_length", "min"),
    ("weakness", "max"),
])
```

## Custom heuristics

To add a custom heuristic, write a function that takes `PolicyStats` and returns a tuple of integers.

Example.
Prefer maximal weakness.
If weakness ties, prefer policies whose mask is numerically small.

```python
from stacktheory.layer3.heuristics import PolicyStats

def my_key(stats: PolicyStats):
    return (stats.weakness, -stats.mask)
```

## Monotone key assumption

Some search algorithms accept `assume_monotone_key`.

Plain English. A key is monotone when adding more vocabulary elements cannot make the key larger.
In other words, as you make a statement more specific, the score cannot improve.

This matters for early stopping and pruning.
If the key is monotone, a learner can safely ignore a whole subtree once the current partial policy is already worse than the best correct policy found so far.

### Why description length is monotone

Description length is |π| which is the number of selected vocabulary elements.
If you add one more element then |π| increases by 1.
So any key that prefers shorter policies using `-|π|` cannot increase when you add elements.

### Why weakness is monotone

Weakness is w(π) which equals |Ext(π)|.

If π is a subset of π' then every completion of π' is also a completion of π.
That means Ext(π') is a subset of Ext(π).
So its size cannot be larger.
So w(π') is less than or equal to w(π).

This is a theorem that follows directly from the definition of extension.
It does not require extra assumptions about satisfiability.
The only requirement is that both masks represent statements in the induced language.

### When you can set assume_monotone_key

It is safe to set `assume_monotone_key=True` for keys built from

- maximising weakness
- minimising description length
- deterministic mask tie breaking

This includes the built in keys and any key built with `make_lexicographic_key` using only `weakness` and `description_length`.

If you use a custom key that can increase when you add an element, set `assume_monotone_key=False`.
That disables the pruning logic and makes the learner more conservative.
