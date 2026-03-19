# Stack Theory Suite
# Author: Michael Timothy Bennett
# Copyright 2026 Michael Timothy Bennett
# SPDX-License-Identifier: Apache-2.0

"""SAT backed induced language tooling.

This module supports a common large n_vars regime where the underlying
environment is the Boolean cube {0,1}^n but n is too large to enumerate.

Layer 1 encodes programs extensionally as membership bitsets over Phi.
That is ideal when |Phi| is small.
For Boolean cubes, |Phi| = 2^n.
For n even moderately large, extensional encodings are impossible.

Instead, this module treats vocabulary elements as clauses and uses a SAT solver
to answer induced language membership queries.

This is faithful to the Stack Theory semantics.
A statement mask corresponds to the conjunction of the selected clauses.
The truth set T(mask) is the set of satisfying assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from stacktheory.layer1.logic import Clause, CNF
from stacktheory.layer2.sat import LNSConfig, SATResult, sat_solve_cnf


@dataclass(frozen=True)
class WeaknessEstimate:
    """Monte Carlo estimate of weakness w(mask) in the induced language.

    estimate
        Estimated number of completions.
    se
        Estimated standard error of the estimator.
    n_samples
        Number of random completions sampled.
    """

    estimate: float
    se: float
    n_samples: int
    completions_total: int


class ClauseVocabulary:
    """A vocabulary whose programs are CNF clauses.

    The induced language consists of all satisfiable subsets of clauses.

    This class does not enumerate the Boolean cube.
    All satisfiability checks are delegated to the internal SAT solver.
    """

    def __init__(self, clauses: Sequence[Clause], *, n_vars: int, names: Optional[Sequence[str]] = None):
        if n_vars < 0:
            raise ValueError("n_vars must be non negative")
        if len(clauses) == 0:
            raise ValueError("clauses must be non empty")
        self.clauses: Tuple[Clause, ...] = tuple(clauses)
        self.n_vars = int(n_vars)
        if names is None:
            names = [f"c{i}" for i in range(len(clauses))]
        if len(names) != len(clauses):
            raise ValueError("names length must match clauses")
        self.names = tuple(str(x) for x in names)
        if len(set(self.names)) != len(self.names):
            raise ValueError("names must be unique")

        self._full_mask = (1 << len(self.clauses)) - 1
        # Cache satisfiable results for statement masks.
        self._sat_cache: Dict[int, bool] = {}

    @property
    def size(self) -> int:
        return len(self.clauses)

    def cnf_of_mask(self, mask: int) -> CNF:
        if mask < 0 or mask > self._full_mask:
            raise ValueError("mask contains bits above vocabulary size")
        selected: List[Clause] = []
        for i in range(self.size):
            if (mask >> i) & 1:
                selected.append(self.clauses[i])
        return CNF.from_iterable(selected)

    def is_in_language_mask(
        self,
        mask: int,
        *,
        truth_table_max_vars: int = 20,
        lns_first: bool = True,
        lns_config: Optional[LNSConfig] = None,
    ) -> bool:
        """Return True if and only if mask is satisfiable.

        This is the induced language membership test.
        """
        if mask < 0 or mask > self._full_mask:
            return False
        if mask in self._sat_cache:
            return self._sat_cache[mask]
        cnf = self.cnf_of_mask(mask)
        res = sat_solve_cnf(
            cnf,
            n_vars=self.n_vars,
            truth_table_max_vars=truth_table_max_vars,
            lns_first=lns_first,
            lns_config=lns_config,
        )
        self._sat_cache[mask] = res.satisfiable
        return res.satisfiable

    def find_model(
        self,
        mask: int,
        *,
        truth_table_max_vars: int = 20,
        lns_first: bool = True,
        lns_config: Optional[LNSConfig] = None,
    ) -> SATResult:
        """Return a SATResult for the CNF defined by mask."""
        cnf = self.cnf_of_mask(mask)
        return sat_solve_cnf(
            cnf,
            n_vars=self.n_vars,
            truth_table_max_vars=truth_table_max_vars,
            lns_first=lns_first,
            lns_config=lns_config,
        )

    def estimate_weakness(
        self,
        base_mask: int,
        *,
        n_samples: int = 2000,
        seed: int = 0,
        truth_table_max_vars: int = 20,
        lns_first: bool = True,
        lns_config: Optional[LNSConfig] = None,
    ) -> WeaknessEstimate:
        """Estimate weakness by sampling random completions.

        Weakness is |E_l|.
        For the induced language over clauses, a completion is a superset mask y
        such that y is satisfiable.

        This estimator samples y uniformly from supersets of base_mask.
        """
        if base_mask < 0 or base_mask > self._full_mask:
            raise ValueError("base_mask contains bits above vocabulary size")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")

        if not self.is_in_language_mask(
            base_mask,
            truth_table_max_vars=truth_table_max_vars,
            lns_first=lns_first,
            lns_config=lns_config,
        ):
            return WeaknessEstimate(estimate=0.0, se=0.0, n_samples=n_samples, completions_total=0)

        m = self.size
        remaining = [i for i in range(m) if ((base_mask >> i) & 1) == 0]
        n_remaining = len(remaining)
        completions_total = 1 << n_remaining

        rng = random.Random(seed)
        hits = 0

        for _ in range(n_samples):
            r = rng.getrandbits(n_remaining)
            y = int(base_mask)
            for j, idx in enumerate(remaining):
                if (r >> j) & 1:
                    y |= 1 << idx
            if self.is_in_language_mask(
                y,
                truth_table_max_vars=truth_table_max_vars,
                lns_first=lns_first,
                lns_config=lns_config,
            ):
                hits += 1

        p = hits / float(n_samples)
        est = p * float(completions_total)
        se = math.sqrt(max(p * (1.0 - p) / float(n_samples), 0.0)) * float(completions_total)
        return WeaknessEstimate(estimate=est, se=se, n_samples=n_samples, completions_total=completions_total)
