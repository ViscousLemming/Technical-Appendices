# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Tests for the WordBitset backend.

WordBitset is a packed uint64 tensor view of the same extensional sets used in
Layer 1.
These tests check that word operations match integer bitset operations.
"""

import random

import torch

from stacktheory.layer1.bitset import pack_bool_tensor
from stacktheory.layer2 import WordBitset


def test_wordbitset_round_trip_matches_layer1_bitset():
    rng = random.Random(0)
    n = 137
    x = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)

    bitset = pack_bool_tensor(x)
    wb = WordBitset.from_bool_tensor(x)

    assert wb.to_bitset_int() == bitset
    assert torch.equal(wb.to_bool_tensor(device="cpu"), x)
    assert wb.cardinality() == int(x.to(dtype=torch.int64).sum().item())


def test_wordbitset_operations_match_integer_bitset_semantics():
    rng = random.Random(1)
    n = 131
    x = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)
    y = torch.tensor([rng.random() < 0.5 for _ in range(n)], dtype=torch.bool)

    bx = pack_bool_tensor(x)
    by = pack_bool_tensor(y)
    full = (1 << n) - 1

    wx = WordBitset.from_bool_tensor(x)
    wy = WordBitset.from_bool_tensor(y)

    assert (wx & wy).to_bitset_int() == (bx & by)
    assert (wx | wy).to_bitset_int() == (bx | by)
    assert (wx ^ wy).to_bitset_int() == (bx ^ by)
    assert (~wx).to_bitset_int() == ((~bx) & full)
    assert (wx - wy).to_bitset_int() == (bx & (~by))

    assert wx.issubset(wy) == ((bx & (~by)) == 0)