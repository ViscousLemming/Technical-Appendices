<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Provenance and appendix alignment

This page explains where each Stack Theory definition lives in the code.
It also lists the small extra assumptions that some higher layers add for convenience.

The reference mathematical source for this mapping is your appendix file.
The labels below match the LaTeX labels in that file.

If you are reading this repository without the appendix open, start with glossary.md.
This page is about mapping, not rederiving the definitions.

## Layer 1 semantic core

Environment, programs, vocabularies  
Appendix label `environment`  
Implemented in `stacktheory.layer1.environment`, `stacktheory.layer1.program`, and `stacktheory.layer1.vocabulary`.

Abstraction layer, statements, truth sets  
Appendix label `abstractionlayer`  
Implemented in `stacktheory.layer1.vocabulary`.

Completions, extensions, equivalence  
Appendix label `def:extensions`  
Implemented in `stacktheory.layer1.vocabulary`.

Weakness  
Appendix label `def:weakness`  
Implemented in `stacktheory.layer1.vocabulary`.

Abstractor function  
Appendix label `def:abstractor`  
Implemented in `stacktheory.layer1.vocabulary`.

## Tasks and policies

Tasks  
Appendix label `def:vtask`  
Implemented in `stacktheory.layer3.tasks.Task`.

The suite enforces the appendix condition  
\(O_\alpha \subseteq \Ext{I_\alpha}\)  
This means every output extends at least one admissible input.

Totality  
Appendix text after `def:vtask`  
Implemented as an optional check in `Task.is_total()`.

Policies and correctness  
Appendix label `inference`  
Implemented in `stacktheory.layer3.learning.is_correct_policy`.

The exact condition implemented is  
\(\Ext{I_\alpha} \cap \Ext{\pi} = O_\alpha\)

## Proxies and learning

Preference proxies  
Appendix section on learning and proxies  
Implemented in `stacktheory.layer3.learning.Proxy` and `stacktheory.layer3.heuristics`.

Weakness proxy  
Implemented as maximising \(w(\pi)\) with deterministic tie breaking.

Description length proxy  
Implemented as minimising \(|\pi|\) with deterministic tie breaking.

## Higher layer assumptions

Layer 4 provides `EmbodiedTask`.
It keeps the same task semantics.
It adds two convenience restrictions.

It requires inputs and outputs to be non empty.
It enforces totality.

These restrictions are not part of the bare task definition.
They are suite level assumptions that make it easier to build experiments where at least one correct policy exists.

If you need partial tasks, use `stacktheory.layer3.Task` instead.

## How to validate alignment

Run the test suite.

```bash
pytest -q
```

Layer 1 tests are the most important ones for mathematical alignment.
They check induced language membership, truth sets, completions, extensions, and weakness on hand checkable toy cases.

Layer 3 tests check the correct policy equation and check that all search based learners return policies that satisfy it.
