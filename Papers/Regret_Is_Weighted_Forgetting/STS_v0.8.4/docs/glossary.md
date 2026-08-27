<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Glossary and notation

This page defines every mathematical and technical term that the documentation uses.
Each term has three parts.
A mathematical definition.
A plain English explanation.
The name of the main code object that implements it.

If you only read one file before using the library, read this one.

## Environment

Mathematics. An environment is a nonempty set \(\Phi\) of mutually exclusive states.

Plain English. \(\Phi\) is the list of all possible situations your toy world allows.
Only one state is true at a time.

In code. Use `FiniteEnvironment`, `BooleanCubeEnvironment`, or `IndexEnvironment`.

## State

Mathematics. A state is an element \(\varphi \in \Phi\).

Plain English. A state is one concrete world.
It is one full configuration.

In code. In `FiniteEnvironment` a state can be any hashable Python object.
In `BooleanCubeEnvironment` a state is an integer that encodes a full Boolean assignment.
In `IndexEnvironment` a state is an integer index from 0 to \(|\Phi|-1\).

## Program

Mathematics. A program is any subset \(p \subseteq \Phi\).
You can also view it as a predicate that is true in exactly those states.

Plain English. A program is a set of states where a condition holds.
It is not code that runs.
It is a meaning.

In code. Use `Program`.
A `Program` stores its meaning as a packed bitset.

## Vocabulary

Mathematics. A vocabulary \(\mathfrak{v}\) is a finite set of programs.
Formally \(\mathfrak{v} \subseteq 2^{\Phi}\).

Plain English. A vocabulary is the set of questions you are allowed to ask about the world.
Each question is a program.

In code. Use `Vocabulary`.
A `Vocabulary` stores its programs as an ordered list so it can represent statements as bit masks.

## Statement

Mathematics. A statement \(\ell\) is a finite subset of the vocabulary.
It is satisfiable in the environment.
Formally \(\ell \in L_{\mathfrak{v}}^{\mathrm{fin}}\).

Plain English. A statement is a bundle of vocabulary elements that can all be true together.
You can think of it as a conjunction of vocabulary programs.

In code. Use `Statement`.
A `Statement` is represented as an integer mask over the vocabulary list.

## Mask

Mathematics. A mask is an integer encoding of a finite set.
Bit \(i\) is 1 when element \(i\) is included.

Plain English. A mask is a compact way to store a set.
It is fast because union and intersection become bit operations.

In code. A statement mask lives in `Statement.mask`.
A state set mask lives in `Program.bitset`.

## Truth set

Mathematics. The truth set of a statement \(\ell\) is
\[
T(\ell) = \bigcap_{p \in \ell} p.
\]

Plain English. The truth set is the region of the environment where the whole statement holds.
It is what the statement means extensionally.

In code. `Statement.truth_set()` returns a `Program`.

## Induced language

Mathematics. The induced language of a vocabulary \(\mathfrak{v}\) is
\[
L_{\mathfrak{v}} = \{\, \ell \subseteq \mathfrak{v} \mid T(\ell) \neq \emptyset \,\}.
\]

Plain English. The induced language is the set of statements that are actually consistent in the world.
It is smaller than \(2^{\mathfrak{v}}\) because most bundles of predicates cannot all be true at once.

In code. `Vocabulary.is_in_language_mask(mask)` implements membership.
`Vocabulary.induced_language_masks()` enumerates it for small vocabularies.

## Completion

Mathematics. A completion of \(x\) is any statement \(y \in L_{\mathfrak{v}}\) with \(x \subseteq y\).

Plain English. A completion adds extra vocabulary elements while keeping the old ones.
It is a way to make a statement more specific.

In code. `Vocabulary.extension_masks_of_mask(x_mask)` enumerates the completion masks.

## Extension

Mathematics. The extension of a statement \(x\) is the set of its completions
\[
\Ext{x} = \{\, y \in L_{\mathfrak{v}} \mid x \subseteq y \,\}.
\]
For a set of statements \(X\), define \(\Ext{X} = \bigcup_{x \in X} \Ext{x}\).

Plain English. The extension of \(x\) is everything you can say while still saying \(x\).
The extension of \(X\) is everything compatible with at least one statement in \(X\).

In code. `Vocabulary.extension_masks_of_mask(mask)` gives \(\Ext{\ell}\).
`stacktheory.layer3.learning.extension_of_inputs(task)` gives \(\Ext{I_\alpha}\).

## Weakness

Mathematics. The weakness of \(\ell\) is the size of its extension
\[
w(\ell) = |\Ext{\ell}|.
\]

Plain English. A statement is weak when it leaves you many ways to continue.
Weakness counts how many completions remain possible.

In code. `Vocabulary.weakness(statement)` and `Vocabulary.weakness_of_mask(mask)` compute it exactly.

## Description length

Mathematics. The description length proxy is \(|\ell|\).
This is the number of vocabulary elements in the statement.

Plain English. This is a simple measure of how short the statement is.
Shorter statements use fewer ingredients.

In code. `len(statement)` and `statement.mask.bit_count()` compute it.

## Task

Mathematics. A \(\mathfrak{v}\) task is a pair \(\alpha = \langle I_\alpha, O_\alpha \rangle\) where
\[
I_\alpha \subseteq L_{\mathfrak{v}}
\qquad\text{and}\qquad
O_\alpha \subseteq \Ext{I_\alpha}.
\]

Plain English. A task says which inputs are allowed and which outputs count as correct.
Every correct output must extend at least one allowed input.

In code. Use `stacktheory.layer3.Task` for the exact definition.
Use `stacktheory.layer4.EmbodiedTask` when you want extra convenience checks.

## Total task

Mathematics. A task is total when every input \(i \in I_\alpha\) has at least one output \(o \in O_\alpha\) with \(i \subseteq o\).

Plain English. Totality means the task always has at least one correct answer for every allowed input.

In code. `Task.is_total()` tests this.
`EmbodiedTask` enforces it as an additional suite assumption.

## Policy

Mathematics. A policy for a task is any statement \(\pi \in L_{\mathfrak{v}}\).

Plain English. A policy is a rule that says what you must include in every completion.
It is a constraint on inference.

In code. A policy is represented by a statement mask.
Learners return a `Statement`.

## Correct policy

Mathematics. A policy \(\pi\) is correct for \(\alpha\) when
\[
\Ext{I_\alpha} \cap \Ext{\pi} = O_\alpha.
\]

Plain English. A policy is correct when the completions it allows on valid inputs are exactly the acceptable outputs.
It must allow every acceptable output and forbid every unacceptable one.

In code. Use `stacktheory.layer3.learning.is_correct_policy(task, policy)`.

## Proxy

Mathematics. A proxy is a preference relation used to choose among correct policies.
The appendices define learning relative to a proxy.

Plain English. Many policies can be correct.
A proxy tells you which correct policy you prefer.

In code. `Proxy.WEAKNESS` and `Proxy.DESCRIPTION_LENGTH` are built in proxies.

## Heuristic

Mathematics. A heuristic is a computable ranking key that approximates or implements a proxy.
In the exact learners it is used only after correctness holds.

Plain English. A heuristic is a scoring rule for candidate policies.
The library treats higher scores as better.

In code. A heuristic is a `KeyFn` that maps `PolicyStats` to a tuple of integers.

## Candidate set

Mathematics. A candidate set \(Q\) is any subset of the induced language.
Learning can be defined as selecting a proxy maximal correct policy from \(Q\).

Plain English. \(Q\) is the set of policies you are willing to consider.
Restricting \(Q\) changes the computational problem without redefining the language.

In code. Pass `candidates=...` into learners in `stacktheory.layer3`.

## Extensional

Mathematics. Two objects are treated as the same when they pick out the same subset of \(\Phi\).

Plain English. Only the truth set matters.
How you built it does not matter.

In code. `Program` equality is equality of bitsets.
Many other objects cache results based on that extensional identity.

## Packed bitset

Mathematics. A packed bitset is an integer encoding of a subset of a finite set.

Plain English. It is a fast representation for finite sets.
Union, intersection, and complement become bit operations.

In code. `Program.bitset` is a packed bitset.
Many helper functions live in `stacktheory.layer1.bitset`.

## SAT

Mathematics. SAT is the satisfiability problem.
Given a Boolean formula, decide whether there exists an assignment that makes it true.

Plain English. SAT asks whether a set of logical constraints can all be satisfied at once.

In code. Layer 2 provides `sat_solve_cnf` for CNF formulas.

## CNF

Mathematics. A formula in conjunctive normal form is a conjunction of clauses.
A clause is a disjunction of literals.

Plain English. CNF is the standard format used by most SAT solvers.
It is a list of lists of signed variables.

In code. Use `CNF`, `Clause`, and `Literal` in `stacktheory.layer1.logic`.

## DPLL

Mathematics. DPLL is a complete backtracking algorithm for SAT with pruning rules such as unit propagation.

Plain English. DPLL tries assignments recursively.
It prunes the search using forced choices.

In code. The internal SAT backend uses a DPLL style core for exact solving.
## Statement equivalence

Mathematics. Two statements \(\ell_1\) and \(\ell_2\) are equivalent when they have the same extension.
That is
\[
\Ext{\ell_1} = \Ext{\ell_2}.
\]

Plain English. Equivalence means the two statements have the same set of possible ways to keep talking.
They permit exactly the same completions inside the induced language.

In code. `Vocabulary.equivalent(s1, s2)` implements this.

## Abstractor

Mathematics. The abstractor of a statement \(\ell\) is the set of distinct truth sets of its completions.
One extensional form is
\[
f(\mathfrak{v},\ell) = \{\, T(o) \mid o \in \Ext{\ell} \,\}.
\]
This is a set of programs, so duplicates collapse.

Plain English. The abstractor forgets which completion you chose and keeps only the distinct meanings you can reach by completing \(\ell\).
If many completions carve out the same region of the environment, the abstractor keeps that region once.

In code. Use `Vocabulary.abstractor_programs(statement)` for the program list, or `Vocabulary.abstractor(statement)` for a new `Vocabulary` instance built from those programs.

## Child task

Mathematics. A task \(\alpha\) is a child of \(\omega\) when
\[
I_\alpha \subset I_\omega
\qquad\text{and}\qquad
O_\alpha \subseteq O_\omega.
\]

Plain English. A child task is a restriction.
It allows fewer inputs and it does not introduce any new acceptable outputs.
So if you can solve the parent, you might still fail on the child because you are now evaluated on a smaller set of inputs.

In code. Use `is_child_task(alpha, omega)`.

## Downward closure

Mathematics. A set of statements \(S\) is downward closed when
if \(x \in S\) and \(y \subseteq x\) then \(y \in S\).

Plain English. Downward closure means that if a bundle of constraints is consistent, then any weaker bundle of constraints is also consistent.
Induced languages \(L_\mathfrak{v}\) are downward closed because removing conjuncts cannot make a conjunction unsatisfiable.

In code. The suite relies on downward closure when it enumerates induced languages and extensions by recursively adding elements.


## Bitset

Plain English. A bitset is an integer that encodes a set.
Bit k is 1 if and only if element k is in the set.

In this library, a Program stores a bitset over environment state indices.
A Statement stores a bitset mask over vocabulary indices.

In code. See `Program.bitset` and `Statement.mask`.

## Mask

Plain English. A mask is a bitset used as a compact selector.
It tells you which vocabulary elements are included in a statement.

Mathematics. If the vocabulary is \(\{p_0,\dots,p_{m-1}\}\), a mask is an integer \(m\) where bit j is 1 exactly when \(p_j\) is included.

In code. `Vocabulary.statement` builds a mask from names or indices.
