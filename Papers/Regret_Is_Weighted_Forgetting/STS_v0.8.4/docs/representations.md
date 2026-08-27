<!--
Stack Theory Suite
Author: Michael Timothy Bennett
Copyright 2026 Michael Timothy Bennett
SPDX-License-Identifier: Apache-2.0
-->

# Representations and performance

Stack Theory objects are semantic.
They do not care how you store them.
The suite therefore supports multiple representations of the same extensional set.

This file explains which representations exist, how they relate, and when they are fast.

## Canonical internal representation

Layer 1 uses a packed bitset as the canonical internal representation for extensional sets of environment states.

A packed bitset is a Python integer.
Bit k is 1 if and only if the state with index k is in the set.

Why this is canonical

- It is exact
- It is immutable
- Bitwise operations on Python integers are implemented in C and are very fast
- It is portable and has no device complications

## Boolean tensors

You can view a bitset as a boolean tensor of length |Phi|.
This representation is convenient for debugging and for integration with PyTorch.

Conversions

- stacktheory.layer1.bitset.pack_bool_tensor packs a boolean tensor into a Python int bitset
- stacktheory.layer1.bitset.unpack_bool_tensor unpacks a Python int bitset to a boolean tensor

These conversions are exact.
They preserve the index convention.

## Word packed bitsets

For batched operations and GPU workloads, a single Python integer is not a good shape.
Layer 2 therefore provides WordBitset.

WordBitset stores the same bitset as a 1D tensor of int64 words.
Each word holds 64 bits.
Bit 0 is the least significant bit of word 0.

This is logically equivalent to the Python int bitset.
It is just stored differently.

Canonical invariants

A WordBitset is canonical when.

- words is contiguous
- words has the correct length for n_bits
- tail bits above n_bits are zero

The WordBitset constructor enforces these invariants.

## Equivalence of representations

All representations are equivalent when they agree on membership for every environment index.

Formally, for an environment of size N.

- Let b be the Python int bitset.
- Let w be the WordBitset with n_bits equals N.
- Let t be the boolean tensor of length N.

They are equivalent if and only if for every i in 0 to N minus 1.

- The i th bit of b is 1
- The i th bit of w is 1
- t[i] is True

The suite tests these equivalences in Layer 5.

## Performance guidance

Packed Python int bitsets are usually fastest when.

- you operate on one set at a time
- you run on CPU
- the environment size is moderate

WordBitset is usually worth it when.

- you want to compute with many sets at once
- you want to keep data on GPU
- you want predictable memory layout and avoid Python big int overhead in tight loops

Boolean tensors are usually not the right internal representation.
They are larger in memory and most operations are bandwidth bound.

Plain English. Bandwidth bound means the bottleneck is moving data in memory, not doing arithmetic.
So using a larger representation can slow you down even if the math is simple.
They are still useful for integration and for writing vectorised kernels.
