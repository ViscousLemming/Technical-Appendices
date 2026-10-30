"""E3: POBAX RockSample (Smith and Simmons 2004; Tao et al. 2025), vectorised in NumPy.

Dynamics follow pobax/envs/jax/rocksample.py with the rocksample_5_5 configuration
(size 5, 5 rocks, half-efficiency distance 20, rewards +10 good rock, -10 bad
rock, +10 exit). Observation: one-hot row, one-hot column, and one entry per rock
that is +1 or -1 after a check of that rock (0 otherwise). Actions: 0 north,
1 east, 2 south, 3 west, 4 sample, 5..4+k check rock i. Moving into the last
column ends the episode with the exit reward. Sampling a good rock turns it bad.
A check returns the true quality with probability (1 + 2^(-d/20)) / 2 at distance d.
"""
from __future__ import annotations

import numpy as np

SIZE, ROCKS = 5, 5
D0 = 20.0
GOOD, BAD, EXIT = 10.0, -10.0, 10.0
DIRS = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
OBS_DIM = 2 * SIZE + ROCKS
ACTIONS = 5 + ROCKS
STEP_LIMIT = 100


def rock_map(seed: int, size: int = SIZE, k: int = ROCKS) -> np.ndarray:
    """k distinct rock cells (row, col) outside the exit column."""
    rng = np.random.default_rng(int(seed))
    cells = np.array([(r, c) for c in range(size - 1) for r in range(size)])
    return cells[rng.choice(len(cells), size=k, replace=False)]


def accuracy(dist: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.power(2.0, -np.asarray(dist) / D0))


class RockSample:
    def __init__(self, rocks: np.ndarray, start: np.ndarray, good: np.ndarray, rng: np.random.Generator):
        self.rocks = np.asarray(rocks)
        self.n = len(start)
        self.pos = np.asarray(start, dtype=np.int64).copy()
        self.good = np.asarray(good, dtype=np.int64).copy()         # (n, k), changes when a good rock is sampled
        self.good0 = self.good.copy()
        self.rng = rng
        self.t = np.zeros(self.n, dtype=np.int64)
        self.done = np.zeros(self.n, dtype=bool)
        self.ret = np.zeros(self.n)
        self.last_check = np.zeros((self.n, ROCKS), dtype=np.float32)

    def rock_here(self) -> np.ndarray:
        """Index of the rock under the agent, -1 if none."""
        eq = np.all(self.pos[:, None, :] == self.rocks[None], axis=-1)   # (n, k)
        return np.where(eq.any(1), eq.argmax(1), -1)

    def obs(self) -> np.ndarray:
        o = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        o[np.arange(self.n), self.pos[:, 0]] = 1.0
        o[np.arange(self.n), SIZE + self.pos[:, 1]] = 1.0
        o[:, 2 * SIZE:] = self.last_check
        return o

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.int64)
        live = ~self.done
        r = np.zeros(self.n)
        # sample
        here = self.rock_here()
        smp = live & (a == 4) & (here >= 0)
        if smp.any():
            idx = np.nonzero(smp)[0]
            g = self.good[idx, here[idx]]
            r[idx] += np.where(g == 1, GOOD, BAD)
            self.good[idx, here[idx]] = 0
        # move
        mv = live & (a < 4)
        newp = np.clip(self.pos + DIRS[np.minimum(a, 3)], 0, SIZE - 1)
        self.pos = np.where(mv[:, None], newp, self.pos)
        exit_ = live & (self.pos[:, 1] == SIZE - 1)
        r += np.where(exit_, EXIT, 0.0)
        # check
        self.last_check[:] = 0.0
        chk = live & (a >= 5)
        if chk.any():
            idx = np.nonzero(chk)[0]
            ri = a[idx] - 5
            d = np.linalg.norm(self.pos[idx] - self.rocks[ri], axis=1)
            truth = self.good[idx, ri]
            right = self.rng.random(len(idx)) < accuracy(d)
            seen = np.where(right, truth, 1 - truth)
            self.last_check[idx, ri] = np.where(seen == 1, 1.0, -1.0)
        self.t += live
        timeout = live & ~exit_ & (self.t >= STEP_LIMIT)
        self.done |= exit_ | timeout
        self.ret += r
        return r.astype(np.float32), exit_ | timeout


def bayes_post(rocks: np.ndarray, pos_hist: np.ndarray, act_hist: np.ndarray, chk_hist: np.ndarray) -> np.ndarray:
    """Exact posterior that each rock is good now, for one history.

    pos_hist[t] is the position after step t's action, act_hist[t] the action, chk_hist[t] the check reading (+1/-1/0)
    for the checked rock. Checks are made from the position after the action (a check does not move).
    """
    k = len(rocks)
    logit = np.zeros(k)
    sampled = np.zeros(k, dtype=bool)
    for t in range(len(act_hist)):
        a = act_hist[t]
        if a >= 5:
            i = a - 5
            if sampled[i]:
                continue
            eta = accuracy(np.linalg.norm(pos_hist[t] - rocks[i]))
            llr = np.log(eta / (1 - eta))
            logit[i] += llr if chk_hist[t] > 0 else -llr
        elif a == 4:
            here = np.all(pos_hist[t] == rocks, axis=1)
            sampled |= here
    p = 1.0 / (1.0 + np.exp(-logit))
    p[sampled] = 0.0
    return p
