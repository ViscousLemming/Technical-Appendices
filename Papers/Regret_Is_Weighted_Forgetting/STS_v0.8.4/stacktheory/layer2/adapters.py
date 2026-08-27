# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Adapters between Stack Theory objects and convenient tensor representations.

This module is intentionally shallow.
Layer 1 defines the semantics.
Layer 2 provides useful representations.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch

from stacktheory.layer1.environment import FiniteEnv
from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Statement, Vocabulary
from stacktheory.layer1.bitset import pack_bool_tensor, unpack_bool_tensor


def statement_to_mask_tensor(statement: Statement, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a Statement mask to a boolean tensor of shape (|v|,).

    Tensor index i is True if and only if program i is in the statement.
    """
    m = statement.vocab.size
    return unpack_bool_tensor(statement.mask, m, device=device)


def statement_from_mask_tensor(vocab: Vocabulary, mask_tensor: torch.Tensor, *, validate: bool = True) -> Statement:
    """Convert a boolean mask tensor of shape (|v|,) into a Statement.

    If validate is True then the resulting statement must be in the induced language.
    """
    if not isinstance(mask_tensor, torch.Tensor):
        raise TypeError("mask_tensor must be a torch.Tensor")

    t = mask_tensor.to(dtype=torch.bool, device="cpu").flatten()
    if t.numel() != vocab.size:
        raise ValueError("mask_tensor must have exactly vocab.size elements")

    mask = pack_bool_tensor(t)

    s = Statement(vocab, mask)
    if validate and not s.is_valid():
        raise ValueError("mask_tensor does not encode a valid statement in the induced language")
    return s


def vocabulary_to_tensor(vocab: Vocabulary, *, device: str | torch.device = "cpu") -> torch.Tensor:
    """Convert a Vocabulary into a boolean tensor of shape (|v|, |Phi|).

    Row i is the membership vector of program i.
    """
    m = vocab.size
    n = vocab.env.size
    out = torch.zeros((m, n), dtype=torch.bool, device=device)
    for i, p in enumerate(vocab.programs):
        out[i, :] = p.to_bool_tensor(device=device)
    return out


def vocabulary_from_tensor(
    env: FiniteEnv,
    tensor: torch.Tensor,
    *,
    names: Optional[Sequence[str]] = None,
    validate_uniqueness: bool = True,
) -> Vocabulary:
    """Create a Vocabulary from a boolean tensor of shape (m, |Phi|).

    Each row is interpreted as a program.

    If validate_uniqueness is True then programs must be distinct as sets of states.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("tensor must be a torch.Tensor")

    x = tensor.to(dtype=torch.bool, device="cpu")
    if x.ndim != 2:
        raise ValueError("tensor must be 2D with shape (m, env.size)")
    if x.shape[1] != env.size:
        raise ValueError("tensor second dimension must equal env.size")

    programs: list[Program] = []
    for i in range(x.shape[0]):
        programs.append(Program.from_bool_tensor(env, x[i, :]))

    if validate_uniqueness:
        bitsets = [p.bitset for p in programs]
        if len(set(bitsets)) != len(bitsets):
            raise ValueError("tensor rows are not unique programs")

    return Vocabulary(env, programs=programs, names=names)
