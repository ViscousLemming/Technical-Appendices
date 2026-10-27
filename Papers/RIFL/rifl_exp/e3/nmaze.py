"""E3: noisy T-Maze with a visible regime, the fallback named in the frozen plan.

The POBAX T-Maze with three changes.
  1. Each step at the start cell shows a fresh noisy reading of the goal, correct with
     probability q[regime]. North or south at the start cell does nothing, so the agent
     can stay and read again. Reading more is the sensing action.
  2. A regime bit is visible at every step. It sets cue reliability and the reward scale:
     regime 0 reads at 0.95, pays 4 for the right turn and -0.1 for the wrong one (POBAX),
     regime 1 reads at 0.70, pays 8 for the right turn and -4 for the wrong one.
  3. A start flag marks the start cell, since a north reading would otherwise look like hallway.
Observation: [reading (start only), hallway, junction, start, regime].
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 5
ACTIONS = 4
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
GOOD = (4.0, 8.0)
Q = (0.95, 0.70)
PENALTY = (0.1, 4.0)


def tmax(length: np.ndarray) -> np.ndarray:
    return 4 * (np.asarray(length) + 2) + 16


class NTMaze:
    def __init__(self, lengths, goal, regime, rng: np.random.Generator):
        self.L = np.asarray(lengths, dtype=np.int64)
        self.goal = np.asarray(goal, dtype=np.int64)
        self.regime = np.asarray(regime, dtype=np.int64)
        self.q = np.asarray(Q)[self.regime]
        self.pen = np.asarray(PENALTY)[self.regime]
        self.good = np.asarray(GOOD)[self.regime]
        self.rng = rng
        self.n = len(self.L)
        self.idx = np.zeros(self.n, dtype=np.int64)
        self.t = np.zeros(self.n, dtype=np.int64)
        self.done = np.zeros(self.n, dtype=bool)
        self.success = np.zeros(self.n, dtype=bool)
        self.ret = np.zeros(self.n)
        self.tmax = tmax(self.L)
        self.reads = np.zeros((self.n, 2), dtype=np.int64)     # count of north and south readings seen

    def obs(self) -> np.ndarray:
        o = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        start = self.idx == 0
        junc = self.idx == self.L + 1
        o[:, 1] = (~junc & ~start).astype(np.float32)
        o[junc, 2] = 1.0
        o[start, 3] = 1.0
        o[:, 4] = self.regime.astype(np.float32)
        if start.any():
            right = self.rng.random(self.n) < self.q
            reading = np.where(right, self.goal, 1 - self.goal)
            live = start & ~self.done
            o[start, 0] = reading[start].astype(np.float32)
            self.reads[live, reading[live]] += 1
        return o

    def at_junction(self) -> np.ndarray:
        return self.idx == self.L + 1

    def step(self, a):
        a = np.asarray(a, dtype=np.int64)
        live = ~self.done
        junc = self.at_junction()
        ud = a < 2
        term = live & junc & ud
        correct = ((a == NORTH) & (self.goal == 0)) | ((a == SOUTH) & (self.goal == 1))
        r = np.where(term, np.where(correct, self.good, -self.pen), 0.0).astype(np.float32)
        self.success |= term & correct
        move = live & ~ud
        east = a == EAST
        self.idx = np.where(move & east, np.minimum(self.idx + 1, self.L + 1), self.idx)
        self.idx = np.where(move & ~east, np.maximum(self.idx - 1, 0), self.idx)
        self.t = self.t + live
        timeout = live & ~term & (self.t >= self.tmax)
        newly = term | timeout
        self.done |= newly
        self.ret += r
        return r, newly


def exact_post(readings: np.ndarray, regime: np.ndarray) -> np.ndarray:
    """Posterior that the goal is south given counts of north and south readings, uniform prior."""
    q = np.asarray(Q)[np.asarray(regime)]
    llr = np.log(q / (1 - q))
    z = (readings[:, 1] - readings[:, 0]) * llr
    return 1.0 / (1.0 + np.exp(-z))
