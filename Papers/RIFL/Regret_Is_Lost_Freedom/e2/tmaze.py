"""E2: the POBAX T-Maze (Bakker 2001; Tao et al. 2025), vectorised in NumPy.

Dynamics follow pobax/envs/jax/tmaze.py. The agent starts at cell 0, whose
observation shows the goal bit when the goal is south. Cells 1..L are hallway
and cell L+1 is the junction. North or south at the junction ends the episode
with reward 4 when it matches the goal and -0.1 otherwise. North and south
elsewhere do nothing. East and west move along the hallway. Observations are
four bits: goal bit (start cell only), hallway flag, junction flag, unused.
"""
from __future__ import annotations

import numpy as np

OBS_DIM = 4
ACTIONS = 4
NORTH, SOUTH, EAST, WEST = 0, 1, 2, 3
GOOD, BAD = 4.0, -0.1


def tmax(length: np.ndarray) -> np.ndarray:
    """Step limit per episode: four times the shortest path."""
    return 4 * (np.asarray(length) + 2)


class TMaze:
    """A batch of independent T-Maze episodes with per-episode hallway lengths."""

    def __init__(self, lengths: np.ndarray, goal: np.ndarray):
        self.L = np.asarray(lengths, dtype=np.int64)
        self.goal = np.asarray(goal, dtype=np.int64)       # 0: north is correct, 1: south is correct
        self.n = len(self.L)
        self.idx = np.zeros(self.n, dtype=np.int64)
        self.t = np.zeros(self.n, dtype=np.int64)
        self.done = np.zeros(self.n, dtype=bool)
        self.reached = np.full(self.n, -1, dtype=np.int64)  # first step at the junction
        self.success = np.zeros(self.n, dtype=bool)
        self.tmax = tmax(self.L)

    def obs(self) -> np.ndarray:
        o = np.zeros((self.n, OBS_DIM), dtype=np.float32)
        start = self.idx == 0
        junc = self.idx == self.L + 1
        o[:, 1] = (~junc).astype(np.float32)
        o[start, 0] = self.goal[start].astype(np.float32)
        o[junc, 2] = 1.0
        return o

    def at_junction(self) -> np.ndarray:
        return self.idx == self.L + 1

    def step(self, a: np.ndarray):
        a = np.asarray(a, dtype=np.int64)
        live = ~self.done
        junc = self.at_junction()
        ud = a < 2
        term = live & junc & ud
        correct = ((a == NORTH) & (self.goal == 0)) | ((a == SOUTH) & (self.goal == 1))
        r = np.where(term, np.where(correct, GOOD, BAD), 0.0).astype(np.float32)
        self.success |= term & correct
        move = live & ~ud
        east = a == EAST
        self.idx = np.where(move & east, np.minimum(self.idx + 1, self.L + 1), self.idx)
        self.idx = np.where(move & ~east, np.maximum(self.idx - 1, 0), self.idx)
        self.t = self.t + live
        timeout = live & ~term & (self.t >= self.tmax)
        newly = term | timeout
        self.done = self.done | newly
        return r, newly
