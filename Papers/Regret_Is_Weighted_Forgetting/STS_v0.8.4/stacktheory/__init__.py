# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

r"""Stack Theory Suite.

This package is a reference implementation of the Stack Theory primitives used
across the Stack Theory papers and thesis.

Everything in Layer 1 is finite and exact.
If an object exists in Layer 1, then this library can compute its truth sets,
extensions, weakness, and task correctness with no approximation.

Stack Theory mapping

Environment
The environment is the finite set \Phi of states.

Program
A program is a set p \subseteq \Phi.
In code a Program stores p extensionally as a packed bitset.

Vocabulary
A vocabulary is a finite set v \subseteq P, where P = 2^\Phi.

Statement
A statement is a set \ell \subseteq v.
It is in the induced language L_v when its truth set is not empty.

Truth set
The truth set is T(\ell) = \bigcap_{p \in \ell} p.

Completion and extension
A completion of \ell is any \ell' \in L_v with \ell \subseteq \ell'.
The extension is Ext(\ell) = {\ell' \in L_v \mid \ell \subseteq \ell'}.

Weakness
Weakness is w(\ell) = |Ext(\ell)|.

Canonical internal representation

- Extensional sets of states are stored as packed bitsets.
- Bit i is 1 when state i is in the set.
- This representation is exactly equivalent to the boolean tensor semantics.
  It is the same set, just stored in a tighter form.

Layers

Layer 1
Core finite Stack Theory objects and exact extensional semantics.

Layer 2
Tensor encodings, adapters, and SAT tooling.

Layer 3
Tasks, correctness, heuristics, and learning algorithms.

Layer 4
Scaling utilities, packed vocab backends, and SAT backed induced languages.

Layer 5
Quality system, golden fixtures, and benchmarks.
"""

from __future__ import annotations

__version__ = "0.8.4"

__all__ = [
    "__version__",
    "layer1",
    "layer2",
    "layer3",
    "layer4",
    "layer5",
]
