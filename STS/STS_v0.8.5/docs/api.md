<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# API overview

This page is a map of the public API.
It tells you what each object is for in plain English.

For the full details and edge cases, read the docstrings in the code.

## Layer 1

Import path  
`from stacktheory.layer1 import ...`

FiniteEnvironment  
A finite environment where you explicitly list the states.

BooleanCubeEnvironment  
The special environment \(\{0,1\}^n\) with a fixed bit based enumeration.

IndexEnvironment  
An environment with states \(\{0,1,\dots,N-1\}\) when you only care about indices.

Program  
A subset of environment states.
This is the extensional meaning of a predicate.

Vocabulary  
A finite set of programs plus names.
It induces a language of satisfiable statements.

Statement  
A subset of the vocabulary represented by an integer mask.
Use `Statement.is_valid()` to check membership in the induced language.

Logic helpers  
Literal, Clause, Term, CNF, DNF.
These are small syntax wrappers for Boolean cubes.
They compile to Programs.

Useful functions

- pack_bool_tensor  
  Pack a boolean membership tensor into a Python int bitset.
- unpack_bool_tensor  
  Unpack a Python int bitset into a boolean membership tensor.
- bitset_popcount  
  Count set bits in a Python int.

## Layer 2

Import path  
`from stacktheory.layer2 import ...`

Adapters  
Convert between suite objects and tensor or text formats.

- vocabulary_to_tensor and vocabulary_from_tensor  
  Convert a Vocabulary to a tensor representation and back.
- statement_to_mask_tensor and statement_from_mask_tensor  
  Convert a Statement mask to a tensor and back.

Logic encodings  
Convert clauses and formulas to the thesis style ternary tensor encoding.

Formats  
Convert CNF to and from text formats.

- cnf_to_str and dnf_to_str  
  Human readable string format.
- parse_cnf and parse_dnf  
  Parse the human readable format.
- cnf_to_dimacs and cnf_from_dimacs  
  DIMACS CNF interchange format for SAT tools.

Representations

WordBitset  
A word packed bitset stored as an int64 tensor.
It is logically equivalent to the Python int bitset.

SAT helpers

- sat_solve_cnf  
  Solve a CNF and return a satisfying assignment if one exists.
- sat_is_satisfiable  
  Return True if a CNF is satisfiable.

## Layer 3

Import path  
`from stacktheory.layer3 import ...`

Task  
A task \(\alpha = \langle I_\alpha, O_\alpha \rangle\) over an induced language.
Inputs and outputs are sets of statement masks.

is_correct_policy  
Check the Stack Theory correctness equation.

correct_policies  
Enumerate all correct policies in a candidate set.

learn and learn_min_description_length  
Exact learners that return a preferred correct policy.

Heuristics

PolicyStats  
The basic statistics the learners use for ranking.

make_lexicographic_key  
Build a deterministic key from weakness, description length, and mask tie breaking.

Search learners  
Best first, branch and bound, beam search, random search, genetic search, and local search.
See search-algorithms.md for the plain English descriptions.

## Layer 4

Import path  
`from stacktheory.layer4 import ...`

Language  
A small wrapper around a Vocabulary that provides language level operations.

PackedVocabulary  
A batched representation designed for speed.

SatLanguage  
A language backend for Boolean cubes that uses SAT to avoid full enumeration.

## Layer 5

Import path  
`from stacktheory.layer5 import ...`

Golden fixtures  
Hand checkable examples used in tests and tutorials.

Benchmark harness  
A small repeatable benchmark runner.
It prints machine readable output so you can plot it.
