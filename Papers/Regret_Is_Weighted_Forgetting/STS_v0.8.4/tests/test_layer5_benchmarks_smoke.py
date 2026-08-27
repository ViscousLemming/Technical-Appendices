# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0


"""Smoke tests for the benchmark harness.

Benchmarks are not unit tests, but we at least ensure the harness runs on a tiny
budget without error.
"""

from stacktheory.layer5 import BenchmarkConfig, run_benchmarks


def test_benchmark_runner_smoke():
    res = run_benchmarks(BenchmarkConfig(budget="tiny", repeat=1, seed=0, device="cpu"))
    assert len(res) > 0
    for r in res:
        assert r.name
        assert r.seconds >= 0.0
        assert r.iterations >= 1
        assert r.seconds_per_iter >= 0.0
