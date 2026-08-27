# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Tests for the PackedVocabulary backend.

PackedVocabulary provides batched bitset operations on CPU and GPU.
These tests check that it matches the Layer 1 Vocabulary results for truth sets
and weakness.
"""

import random

import torch

from stacktheory.layer1 import IndexEnvironment, Program, Vocabulary
from stacktheory.layer2 import WordBitset
from stacktheory.layer4 import PackedVocabulary


def test_packed_vocabulary_truth_sets_match_layer1():
    rng = random.Random(0)
    env = IndexEnvironment(n_states=100)

    programs = []
    names = []
    for i in range(8):
        idxs = [j for j in range(env.size) if rng.random() < 0.4]
        programs.append(Program.from_state_indices(env, idxs))
        names.append(f"p{i}")

    vocab = Vocabulary(env, programs=programs, names=names)
    pv = PackedVocabulary.from_vocabulary(vocab)

    for mask in range(1 << vocab.size):
        t = vocab.truth_set_of_mask(mask).bitset
        words = pv.truth_words_of_mask(mask)
        wb = WordBitset(words, n_bits=env.size)
        assert wb.to_bitset_int() == t


def test_packed_vocabulary_batch_application_matches_scalar_application():
    env = IndexEnvironment(n_states=65)
    p0 = Program.from_state_indices(env, [0, 1, 2, 3, 64])
    p1 = Program.from_state_indices(env, [1, 2, 64])
    p2 = Program.from_state_indices(env, [2, 3])
    vocab = Vocabulary(env, programs=[p0, p1, p2], names=["p0", "p1", "p2"])
    pv = PackedVocabulary.from_vocabulary(vocab)

    base_mask = 1
    base_truth = vocab.truth_set_of_mask(base_mask).bitset
    base_words = pv.truth_words_of_mask(base_mask)

    idx = torch.tensor([0, 1, 2], dtype=torch.int64)
    batch_words = pv.apply_programs_batch(base_words, idx)
    for j, prog_idx in enumerate([0, 1, 2]):
        expected = base_truth & vocab.programs[prog_idx].bitset
        got = WordBitset(batch_words[j], n_bits=env.size).to_bitset_int()
        assert got == expected


def test_packed_vocabulary_can_move_to_cuda_if_available():
    if not torch.cuda.is_available():
        return
    env = IndexEnvironment(n_states=128)
    p0 = Program.from_state_indices(env, [0, 1, 2, 3])
    p1 = Program.from_state_indices(env, [2, 3, 4, 5])
    vocab = Vocabulary(env, programs=[p0, p1], names=["p0", "p1"])
    pv = PackedVocabulary.from_vocabulary(vocab).to("cuda")

    words = pv.truth_words_of_mask(3)
    assert words.is_cuda