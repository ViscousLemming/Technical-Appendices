<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Learning

Layer 3 implements the task and learning machinery from the appendices.
It is designed for small to medium vocabularies where exact enumeration is possible.
It also provides search based learners that can cut work without changing the definitions.

This file defines each object before it is used.
It then explains what the code computes.

If you want the definitions in one place, read glossary.md first.

## Objects in the learning layer

Learning sits on top of the Layer 1 objects.

You start with an environment \(\Phi\).
You choose a vocabulary \(\mathfrak{v}\) which is a finite set of programs \(p \subseteq \Phi\).
That vocabulary induces a language \(L_{\mathfrak{v}}\) which is the set of satisfiable statements.

Layer 3 then adds tasks, policies, correctness, and learning rules.

## Task

Mathematics. A \(\mathfrak{v}\) task is a pair \(\alpha = \langle I_\alpha, O_\alpha \rangle\) where

\[
I_\alpha \subseteq L_{\mathfrak{v}}
\qquad\text{and}\qquad
O_\alpha \subseteq \Ext{I_\alpha}.
\]

Plain English. A task tells you which inputs are allowed and which outputs count as correct.
An output must extend at least one admissible input.
That requirement is what stops tasks from smuggling in answers that do not correspond to any input.

In code, `Task` stores

- `vocab` which is the `Vocabulary` object \(\mathfrak{v}\)
- `inputs` which is a set of statement masks
- `outputs` which is a set of statement masks

Both inputs and outputs are validated to be in the induced language.
Outputs are validated to extend at least one input.

Create a task from statement objects.

```python
from stacktheory.layer1 import BooleanCubeEnvironment, Vocabulary
from stacktheory.layer3 import Task

env = BooleanCubeEnvironment(n=2)
progs = [env.literal_program(var=0, value=1), env.literal_program(var=1, value=1)]
vocab = Vocabulary(env, programs=progs, names=["x0=1", "x1=1"])

i = vocab.statement(["x0=1"])
o = vocab.statement(["x0=1", "x1=1"])

task = Task.from_statements([i], [o])
```

### Totality

Some results and experiments assume tasks are total.

Mathematics. A task is total when for every input \(i \in I_\alpha\) there exists an output \(o \in O_\alpha\) with \(i \subseteq o\).

Plain English. Totality means every input has at least one correct answer.

In code, `Task.is_total()` checks totality.
Layer 3 does not require totality.
Layer 4 `EmbodiedTask` enforces totality as a convenience restriction.

## Policy

Mathematics. A policy for a task is any statement \(\pi \in L_{\mathfrak{v}}\).

Plain English. A policy is a constraint.
It says what must be included in every completion.
When you apply a policy to an input, you keep only the completions that are compatible with the policy.

In code, a policy is represented as a statement mask.
Learners return a `Statement`.

## Correctness

Correctness is the central equation.
This is what all learners must satisfy.

Mathematics. A policy \(\pi\) is correct for \(\alpha\) when

\[
\Ext{I_\alpha} \cap \Ext{\pi} = O_\alpha.
\]

Plain English. First take every completion of every admissible input.
Then filter those completions to keep only the ones that contain the policy.
If the remaining completions are exactly the acceptable outputs, the policy is correct.

The suite implements this exact condition.
It does not relax it.
It does not approximate it.
Search based learners only change how fast you find a candidate.
They never change the definition of correct.

In code, use `is_correct_policy(task, policy)`.

### How correctness is computed

The library uses a direct extensional computation.

1. Precompute \(\Ext{I_\alpha}\) once by enumerating completions of each input.
2. For each policy candidate \(\pi\), filter that precomputed set by subset inclusion.
3. Compare the filtered set to the stored output set \(O_\alpha\).

That is why `is_correct_policy` accepts an optional `ext_inputs` cache.

## Proxies

A proxy is a preference rule used to choose among correct policies.

Plain English. Many policies can be correct.
A proxy is the rule that tells you which correct policy you want.

Layer 3 includes two built in proxies.

Weakness proxy  
Prefer larger weakness \(w(\pi)\).

Description length proxy  
Prefer smaller \(|\pi|\).

Both proxies use deterministic tie breaking so results are reproducible.

In code, choose a proxy with `Proxy.WEAKNESS` or `Proxy.DESCRIPTION_LENGTH`.

## Heuristics

A heuristic is the concrete scoring rule used in code.

Plain English. A heuristic turns a policy into a sortable key.
The library treats larger keys as better.

In code, a heuristic is a `KeyFn`.
It maps `PolicyStats` to a tuple of integers.

The default weakness proxy uses a heuristic that prefers.

1. larger weakness
2. shorter description length
3. smaller mask for determinism

Read heuristics.md for details and examples.

## Learners

A learner returns one policy statement.

All learners share the same structure.

1. Decide which candidates to consider.
2. Check correctness using the Stack Theory equation.
3. Choose a preferred correct policy using a proxy or heuristic.

### Exact learner

`learn` is the reference learner.
It enumerates the candidate set and is correct by inspection.
It is slow for large vocabularies because enumeration is exponential in \(|\mathfrak{v}|\).

`learn_min_description_length` is also exact.
It searches by increasing description length and stops early when it finds a correct policy of a given size.

### Search based learners

Search based learners are still exact in the sense that the correctness check is exact.
They are allowed to be incomplete if you cut them off early.
For example if you set a small expansion budget.

Examples include best first search, branch and bound, beam search, random search baselines, and local search utilities.

If you want a plain English description of each search algorithm and its parameters, read search-algorithms.md.
These live in `stacktheory.layer3.search`, `stacktheory.layer3.local_search`, and `stacktheory.layer3.genetic`.

## Parameters you will see in the APIs

Here are the parameters that show up repeatedly and what they mean.

`candidates`  
A subset \(Q \subseteq L_{\mathfrak{v}}\).
If you pass candidates, you are restricting the search space without redefining the language.
This matches the appendix definition where learning is relative to a candidate set.

`proxy`  
Selects a built in preference rule such as weakness or description length.

`heuristic`  
A custom key function used to rank policies.
It is used only among correct policies in the exact learners.

`max_expansions` or `max_steps`  
A computation budget.
If you set a budget, some learners become incomplete by design.
They can return a suboptimal correct policy or fail to find any policy.

`seed` or `rng`  
Controls randomness in random search, genetic search, and local search.
Fixing the seed makes experiments reproducible.

## Common failure modes

The most common conceptual failure mode is redefining the induced language.

If you treat the language as \(2^{\mathfrak{v}}\) instead of \(L_{\mathfrak{v}}\), weakness collapses into a simple function of description length.
That changes the theory.
It is also the easiest way to accidentally produce a false weakness equals simplicity result.

This suite keeps the induced language definition explicit.
If a statement is not satisfiable, it is not in the language.
