# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""Layer 5 quality system.

Layer 5 is not about new mathematics.
It is about preventing silent breakage.

This layer contains.

- Golden fixtures for small finite vocabularies.
- Lightweight benchmark runners.
- Helpers for performance regression checks.

The tests that enforce the quality system live under the repository tests/ folder.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .golden import golden_toy_vocabulary

if TYPE_CHECKING:
    from .benchmarks import BenchmarkConfig, BenchmarkResult, run_benchmarks


def __getattr__(name: str) -> Any:
    """Lazy accessors to avoid importing benchmark modules at package import time."""

    if name in {"run_benchmarks", "BenchmarkConfig", "BenchmarkResult"}:
        from .benchmarks import BenchmarkConfig, BenchmarkResult, run_benchmarks

        mapping = {
            "run_benchmarks": run_benchmarks,
            "BenchmarkConfig": BenchmarkConfig,
            "BenchmarkResult": BenchmarkResult,
        }
        return mapping[name]
    raise AttributeError(name)


__all__ = [
    "golden_toy_vocabulary",
    "run_benchmarks",
    "BenchmarkConfig",
    "BenchmarkResult",
]
