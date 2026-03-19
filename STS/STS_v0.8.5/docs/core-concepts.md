<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Core concepts

This section explains the core Stack Theory objects implemented in Layer 1.
The aim is that you can read this file without reading the code.

Everything in the suite is finite and extensional.
Finite means the environment has a finite number of states.
Extensional means we care only about which states are included, not how the object was constructed.

## Environment

An environment is written Phi in the appendices.
It is a nonempty set of mutually exclusive states.

In the code, a finite environment also includes an enumeration.
The enumeration gives every state an index from 0 up to |Phi| minus 1.
That index is what lets us represent sets of states as bitsets.

Layer 1 provides these environment implementations.

- FiniteEnvironment for arbitrary hashable states
- BooleanCubeEnvironment for Phi_n equals {0,1}^n
- IndexEnvironment for Phi equals {0,1,...,N-1} without storing explicit states

## Program

A program is any subset of Phi.
In code, Program is the extensional set of states where the program is true.

You can combine programs with set operations.

- Conjunction is set intersection
- Disjunction is set union
- Negation is set complement relative to Phi

Programs are represented canonically using a packed bitset.
A packed bitset is an integer whose binary representation encodes membership.
Bit k is 1 if and only if the state with index k is in the set.

## Vocabulary

A vocabulary is a finite set of programs.
It is written v in the appendices.

In code, Vocabulary holds.

- env which is the environment Phi
- programs which is the list of vocabulary programs
- names which is an optional list of unique names for those programs

Vocabulary is treated as immutable after construction.
That lets the implementation cache derived objects such as truth sets and weakness.

## Statement

A statement is a finite subset of the vocabulary.
It is written l in the appendices.

In code, Statement is represented as an integer mask over the vocabulary list.
Bit i is 1 if and only if program i is included in the statement.

Description length proxy

The suite uses the natural proxy |l| equals the number of selected vocabulary elements.
In code this is just popcount(mask).

## Truth set

The truth set T(l) is the set of states where all programs in the statement are true.
Since the statement is a conjunction of vocabulary programs, the truth set is their intersection.

- If l is empty, T(l) is all of Phi
- If l contains programs p1, p2, ..., pk then T(l) is p1 ∩ p2 ∩ ... ∩ pk

In code, Statement.truth_set() returns a Program object.

## Induced language

The induced language L_v is the set of all statement masks whose truth set is nonempty.

In code, Vocabulary.is_in_language_mask(mask) implements this check.
It returns True if and only if truth_set_of_mask(mask) is not empty.

Layer 1 can enumerate L_v exactly, but only for small vocabularies.
This is exponential in |v|.

## Completion and extension

A completion of a statement l is any statement y in L_v such that l is a subset of y.
The extension E_l is the set of all such completions.

In code.

- Vocabulary.extension_masks_of_mask(base_mask) returns a list of completion masks y
- Vocabulary.iter_extension_masks_of_mask yields the same completions lazily

Both only accept base masks that are already in L_v.

## Weakness

Weakness is written w(l) in the appendices.
It is defined as the size of the extension.

w(l) equals |E_l|.

In code, Vocabulary.weakness(statement) and Vocabulary.weakness_of_mask(mask) implement this.

Weakness is exact for finite vocabularies.
Computing it is exponential in the worst case.
That is not a bug.
It is the definition.

## Equivalence

Two statements are equivalent when they have the same extension set.

In code, Vocabulary.equivalent(s1, s2) implements E_{s1} equals E_{s2}.

## Abstractor

The abstractor f(v,l) is defined as the set of distinct truth sets of completions of l.

In code.

- Vocabulary.abstractor_programs returns the distinct truth set programs
- Vocabulary.abstractor returns a Vocabulary object that contains those programs

This is a direct extensional translation of the definition.
