<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Architecture and layering

The Stack Theory Suite is organised as five layers.
Each layer depends only on the layers below it.
This is deliberate.
It keeps the semantic core small and testable.

The conceptual order is also important.
If you are new to the library, read the layers in this order.

States  
A state is one concrete possibility in your toy world.
The environment \(\Phi\) is the set of all states.

Programs  
A program is just a subset of states.
It is a predicate.
It picks out where some condition holds.

Vocabulary choice  
A vocabulary is the finite set of programs you decide to talk about.
This choice is part of the experiment.
Different vocabularies induce different languages and different weakness structure.

Induced language  
The induced language \(L_\mathfrak{v}\) is the set of all satisfiable statements over the vocabulary.
This is where the suite refuses to cheat.
If a statement has an empty truth set, it is not in the language.

Tasks and proxies  
A task tells you which inputs are allowed and which outputs count as correct.
A proxy is the preference rule used to choose among correct policies.

Learning algorithms and tools  
A learner searches for a correct policy and then applies the proxy or heuristic.
Tools like SAT interop and tensor encodings live here because they change representation, not meaning.

That order mirrors the appendices.
It also mirrors how you should build experiments.

## Layer 1

Layer 1 is the semantic core.
If you want to argue about definitions, this is where you look.
Everything else must preserve the Layer 1 semantics.

Layer 1 makes a strong representation choice.
The canonical internal representation of an extensional set is a Python int bitset.

Plain English. An int bitset is an integer where bit \(k\) is 1 when state \(k\) is included.
That makes equality, hashing, and set operations simple and exact.

Layer 1 includes.

- environments \(\Phi\)
- programs \(p \subseteq \Phi\)
- vocabularies \(\mathfrak{v}\)
- statements \(\ell\)
- truth sets \(T(\ell)\)
- induced languages \(L_{\mathfrak{v}}\)
- completions and extensions \(\Ext{\ell}\)
- weakness \(w(\ell)\)

## Layer 2

Layer 2 adds representation and interoperability.
It does not change what any object means.
It only changes how you store it and how you talk to external tools.

Examples include tensor encodings, DIMACS interchange, and SAT solving helpers.

## Layer 3

Layer 3 adds tasks and learning.

It introduces tasks \(\alpha = \langle I_\alpha, O_\alpha \rangle\), policies \(\pi\), and the correctness equation.
All learners use the Layer 1 definitions of induced language, extension, weakness, and correctness.

The only difference between learners is how they search.

## Layer 4

Layer 4 adds scaling utilities.

The key idea is to avoid materialising exponential objects until you have to.
It provides wrappers and alternate backends that preserve the same semantics.

Examples include index based environments and SAT backed language tooling for larger Boolean cubes.

## Layer 5

Layer 5 is the quality system.
It adds fixtures, property tests, and benchmarks that lock down correctness.

If you change a definition by accident, Layer 5 is where you find out.
