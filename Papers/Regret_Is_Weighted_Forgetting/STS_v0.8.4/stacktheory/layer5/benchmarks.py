# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Lightweight benchmarks.

This module is a small benchmarking harness.
It is designed for repeatable development benchmarks.

It does not try to be a full performance lab.
It tries to answer one question.

Did a recent change make the hot paths slower.

The benchmarking API is pure Python.
It uses time.perf_counter and repetition.
This keeps it dependency free.

If you want deeper profiling, use cProfile or py-spy.

Doctest

>>> from stacktheory.layer5 import run_benchmarks, BenchmarkConfig
>>> res = run_benchmarks(BenchmarkConfig(budget="tiny", repeat=1))
>>> len(res) > 0
True
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, List, Optional

import random

import torch

from stacktheory.layer1.environment import IndexEnvironment
from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Vocabulary
from stacktheory.layer1.logic import Clause, CNF, Literal

from stacktheory.layer2.wordbitset import WordBitset
from stacktheory.layer3.tasks import Task
from stacktheory.layer3.search import learn_best_first
from stacktheory.layer3.heuristics import make_lexicographic_key


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark configuration.

    budget
        tiny, small, medium.
        tiny is meant to run in a few seconds.
    repeat
        Number of repetitions per benchmark.
        The minimum time across repeats is reported.
    seed
        Seed for randomised inputs.
    device
        Torch device for tensor based benchmarks.
        Default is cpu.
    """

    budget: str = "small"
    repeat: int = 3
    seed: int = 0
    device: str = "cpu"


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    seconds: float
    iterations: int

    @property
    def seconds_per_iter(self) -> float:
        return self.seconds / max(1, int(self.iterations))


def _timeit(fn: Callable[[], None], *, repeat: int) -> float:
    best = float("inf")
    for _ in range(max(1, int(repeat))):
        t0 = perf_counter()
        fn()
        t1 = perf_counter()
        best = min(best, t1 - t0)
    return float(best)


def _budget_params(budget: str) -> Dict[str, int]:
    b = budget.lower().strip()
    if b == "tiny":
        return {
            "n_bits": 2048,
            "n_pairs": 256,
            "vocab_size": 14,
            "env_size": 256,
            "n_masks": 256,
            "batch": 256,
            "n_vars": 24,
            "n_clauses": 64,
            "search_steps": 128,
        }
    if b == "small":
        return {
            "n_bits": 8192,
            "n_pairs": 1024,
            "vocab_size": 16,
            "env_size": 512,
            "n_masks": 512,
            "batch": 1024,
            "n_vars": 32,
            "n_clauses": 128,
            "search_steps": 256,
        }
    if b == "medium":
        return {
            "n_bits": 32768,
            "n_pairs": 4096,
            "vocab_size": 18,
            "env_size": 1024,
            "n_masks": 1024,
            "batch": 4096,
            "n_vars": 48,
            "n_clauses": 256,
            "search_steps": 512,
        }
    raise ValueError("Unknown budget")


def run_benchmarks(config: Optional[BenchmarkConfig] = None) -> List[BenchmarkResult]:
    """Run the built in benchmark scenarios.

    Returns a list of BenchmarkResult.

    This function is safe to call in unit tests using the tiny budget.
    """

    if config is None:
        config = BenchmarkConfig()
    params = _budget_params(config.budget)

    rng = random.Random(config.seed)

    results: List[BenchmarkResult] = []

    # ---------------------------------------------------------------------
    # Packed int bitset operations
    # ---------------------------------------------------------------------

    n_bits = params["n_bits"]
    n_pairs = params["n_pairs"]

    ints_a = [rng.getrandbits(n_bits) for _ in range(n_pairs)]
    ints_b = [rng.getrandbits(n_bits) for _ in range(n_pairs)]

    def bench_int_and() -> None:
        acc = 0
        for a, b in zip(ints_a, ints_b):
            acc ^= (a & b)
        if acc == 123456789:
            raise RuntimeError("unreachable")

    sec = _timeit(bench_int_and, repeat=config.repeat)
    results.append(BenchmarkResult(name="int_bitset_and", seconds=sec, iterations=n_pairs))

    def bench_int_or() -> None:
        acc = 0
        for a, b in zip(ints_a, ints_b):
            acc ^= (a | b)
        if acc == 123456789:
            raise RuntimeError("unreachable")

    sec = _timeit(bench_int_or, repeat=config.repeat)
    results.append(BenchmarkResult(name="int_bitset_or", seconds=sec, iterations=n_pairs))

    # ---------------------------------------------------------------------
    # WordBitset operations
    # ---------------------------------------------------------------------

    wb_a = WordBitset.from_bitset_int(ints_a[0], n_bits=n_bits, device=config.device)
    wb_b = WordBitset.from_bitset_int(ints_b[0], n_bits=n_bits, device=config.device)

    def bench_word_and() -> None:
        x = wb_a
        y = wb_b
        acc = x
        for _ in range(n_pairs):
            acc = acc ^ (x & y)
        if acc.n_bits != n_bits:
            raise RuntimeError("unreachable")

    sec = _timeit(bench_word_and, repeat=config.repeat)
    results.append(BenchmarkResult(name="wordbitset_and", seconds=sec, iterations=n_pairs))

    # ---------------------------------------------------------------------
    # Vocabulary truth set and weakness
    # ---------------------------------------------------------------------

    env = IndexEnvironment(n_states=params["env_size"])
    programs = []
    names = []
    for i in range(params["vocab_size"]):
        # Random subset with about half the states.
        mask = rng.getrandbits(env.size)
        programs.append(Program(env, mask))
        names.append(f"p{i}")
    vocab = Vocabulary(env, programs=programs, names=names)

    masks = [rng.randrange(1 << vocab.size) for _ in range(params["n_masks"])]

    def bench_truth_sets() -> None:
        acc = 0
        for m in masks:
            acc ^= vocab.truth_set_of_mask(m).bitset
        if acc == 123456789:
            raise RuntimeError("unreachable")

    sec = _timeit(bench_truth_sets, repeat=config.repeat)
    results.append(BenchmarkResult(name="truth_set_of_mask", seconds=sec, iterations=len(masks)))

    # Weakness is only well behaved for small vocab sizes.
    # We pick a few masks that are in the language.
    valid_masks = [m for m in masks if vocab.is_in_language_mask(m)]
    if valid_masks:
        weak_masks = valid_masks[: min(len(valid_masks), 64)]

        def bench_weakness() -> None:
            acc = 0
            for m in weak_masks:
                acc ^= int(vocab.weakness_of_mask(m))
            if acc == 123456789:
                raise RuntimeError("unreachable")

        sec = _timeit(bench_weakness, repeat=config.repeat)
        results.append(BenchmarkResult(name="weakness_of_mask", seconds=sec, iterations=len(weak_masks)))

    # ---------------------------------------------------------------------
    # CNF tensor evaluation
    # ---------------------------------------------------------------------

    n_vars = params["n_vars"]
    n_clauses = params["n_clauses"]
    batch = params["batch"]

    assignments = torch.zeros((batch, n_vars, 2), dtype=torch.bool, device=config.device)
    states = torch.randint(0, 2, (batch, n_vars), device=config.device)
    assignments.scatter_(2, states.unsqueeze(2), True)

    cnf_clauses = []
    for _ in range(n_clauses):
        lits = []
        # Small clause width.
        for _k in range(3):
            var = rng.randrange(n_vars)
            value = rng.randrange(2)
            lits.append(Literal(var=var, value=value))
        cnf_clauses.append(Clause.from_iterable(lits))
    cnf = CNF.from_iterable(cnf_clauses)
    cnf_tensor = cnf.to_tensor(n_vars, device=config.device)

    from stacktheory.layer1.logic import cnf_tensor_satisfied

    def bench_cnf_eval() -> None:
        out = cnf_tensor_satisfied(assignments, cnf_tensor, validate_assignments=False)
        if out.numel() != batch:
            raise RuntimeError("unreachable")

    sec = _timeit(bench_cnf_eval, repeat=config.repeat)
    results.append(BenchmarkResult(name="cnf_tensor_satisfied", seconds=sec, iterations=batch))

    # ---------------------------------------------------------------------
    # Best first search tiny task
    # ---------------------------------------------------------------------

    # Build a tiny task in the same vocabulary.
    #
    # The goal here is not to solve a hard instance.
    # The goal is to exercise the search loop without flakiness.
    # We therefore construct a task with a guaranteed correct policy.
    if vocab.size <= 18:
        prog_bits = vocab.program_bitsets()
        full_bits = vocab.env.all_states_bitset()

        # Greedily build a maximal statement so its extension is a singleton.
        mask = 0
        bits = full_bits
        for j in range(vocab.size):
            new_bits = bits & prog_bits[j]
            if new_bits == 0:
                continue
            mask |= 1 << j
            bits = new_bits

        # Task with one input and one output.
        # Any policy that is a subset of the maximal input is correct.
        # In particular the empty statement is correct and is found quickly.
        task = Task(vocab=vocab, inputs=[mask], outputs=frozenset([mask]))

        key = make_lexicographic_key([("weakness", "max"), ("description_length", "min")])

        def bench_search() -> None:
            _ = learn_best_first(task, heuristic=key, max_expansions=params["search_steps"])

        sec = _timeit(bench_search, repeat=max(1, config.repeat))
        results.append(BenchmarkResult(name="best_first_search", seconds=sec, iterations=params["search_steps"]))
    return results


def _format_results(results: List[BenchmarkResult]) -> str:
    lines = []
    for r in results:
        lines.append(f"{r.name}\t{r.seconds:.6f}\t{r.iterations}\t{r.seconds_per_iter:.3e}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run Stack Theory Suite benchmarks")
    parser.add_argument("--budget", default="small", choices=["tiny", "small", "medium"])
    parser.add_argument("--repeat", default=3, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cpu")

    args = parser.parse_args(argv)

    cfg = BenchmarkConfig(budget=args.budget, repeat=args.repeat, seed=args.seed, device=args.device)
    res = run_benchmarks(cfg)
    print(_format_results(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
