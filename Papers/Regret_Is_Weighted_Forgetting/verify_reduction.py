from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass(frozen=True)
class Point:
    cell: str
    label: int
    mu: float
    rho: float


def cell_sums(points: Sequence[Point]) -> Dict[str, Tuple[float, float]]:
    out: Dict[str, List[float]] = {}
    for p in points:
        if p.cell not in out:
            out[p.cell] = [0.0, 0.0]
        out[p.cell][p.label] += p.mu * p.rho
    return {k: (v[1], v[0]) for k, v in out.items()}  # (A_C, B_C)


def k_rho(points: Sequence[Point]) -> float:
    return sum(min(A, B) for A, B in cell_sums(points).values())


def regret(points: Sequence[Point], q_by_cell: Dict[str, float]) -> float:
    total = 0.0
    for p in points:
        q = q_by_cell[p.cell]
        if p.label == 1:
            total += p.mu * p.rho * (1.0 - q)
        else:
            total += p.mu * p.rho * q
    return total


def optimal_q_and_cost(points: Sequence[Point]) -> Tuple[Dict[str, float], float]:
    q_star: Dict[str, float] = {}
    total = 0.0
    for cell, (A, B) in cell_sums(points).items():
        if A > B:
            q_star[cell] = 1.0
            total += B
        elif A < B:
            q_star[cell] = 0.0
            total += A
        else:
            q_star[cell] = 0.0
            total += A
    return q_star, total


def decomposition_rhs(points: Sequence[Point], q_by_cell: Dict[str, float]) -> float:
    sums = cell_sums(points)
    q_star, base = optimal_q_and_cost(points)
    excess = 0.0
    for cell, (A, B) in sums.items():
        excess += abs(A - B) * abs(q_by_cell[cell] - q_star[cell])
    return base + excess


def toy_example() -> None:
    # C1 has A=0.30, B=0.20 and C2 has A=0, B=0.25.
    points = [
        Point("C1", 1, 0.30, 1.0),
        Point("C1", 0, 0.20, 1.0),
        Point("C2", 0, 0.25, 1.0),
    ]
    q = {"C1": 0.8, "C2": 0.0}
    kr = k_rho(points)
    r = regret(points, q)
    rhs = decomposition_rhs(points, q)
    print("Toy example")
    print(f"K_rho(M) = {kr:.2f}")
    print(f"regret     = {r:.2f}")
    print(f"decomp rhs = {rhs:.2f}")
    assert math.isclose(kr, 0.20, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(r, 0.22, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(rhs, 0.22, rel_tol=0, abs_tol=1e-9)


def random_points(rng: random.Random, n_cells: int = 4, pts_per_cell: int = 4) -> List[Point]:
    raw: List[Tuple[str, int, float, float]] = []
    for c in range(n_cells):
        cell = f"C{c}"
        for _ in range(pts_per_cell):
            label = rng.randint(0, 1)
            mu = rng.random() + 0.05
            rho = rng.random()
            raw.append((cell, label, mu, rho))
    z = sum(mu for _, _, mu, _ in raw)
    return [Point(cell, label, mu / z, rho) for cell, label, mu, rho in raw]


def random_checks(num_trials: int = 200, seed: int = 0) -> None:
    rng = random.Random(seed)
    for _ in range(num_trials):
        pts = random_points(rng)
        q_star, base = optimal_q_and_cost(pts)
        assert math.isclose(base, k_rho(pts), rel_tol=0, abs_tol=1e-10)
        assert math.isclose(regret(pts, q_star), base, rel_tol=0, abs_tol=1e-10)
        q = {cell: rng.random() for cell in q_star}
        lhs = regret(pts, q)
        rhs = decomposition_rhs(pts, q)
        assert math.isclose(lhs, rhs, rel_tol=0, abs_tol=1e-10)
    print(f"Randomized checks passed for {num_trials} trials.")


if __name__ == "__main__":
    toy_example()
    random_checks()
