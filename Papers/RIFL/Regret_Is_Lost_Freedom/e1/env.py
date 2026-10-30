"""E1: delayed-cue POMDP, version 2.

Both regimes place three cues at steps 1 to 3, so every episode needs memory
across the delay. Regimes differ in cue reliability and frequency. Answers at the
three test gates earn reward and never change later observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import numpy as np

CLASSES = 4
CUES = 3
DISTRACTORS = 5
GATES = 3
# observation layout
REG = 0          # 2 regime flags (0 common, 1 rare)
CUE = 2          # 4 cue symbols
DIS = 6          # 5 distractors
GAT = 11         # 3 gate flags
START = 14
PAD = 15
OBS_DIM = 16


def all_tests() -> np.ndarray:
    """Every non-constant binary test on four classes, one per complement pair (7 rows)."""
    out = []
    seen = set()
    for bits in itertools.product((0, 1), repeat=CLASSES):
        b = tuple(bits)
        if sum(b) in (0, CLASSES):
            continue
        comp = tuple(1 - v for v in b)
        if comp in seen:
            continue
        seen.add(b)
        out.append(b)
    return np.asarray(out, dtype=np.int64)


@dataclass(frozen=True)
class World:
    seed: int
    q_rare: float
    q_common: float
    tests: np.ndarray            # (3, 4) the balanced tests, distinguishing all classes
    frac_source: float = 0.05
    frac_target: float = 0.30

    @property
    def everything(self) -> np.ndarray:
        return all_tests()


def make_world(seed: int, q_common: float) -> World:
    rng = np.random.default_rng(int(seed))
    base = np.asarray([[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 0]], dtype=np.int64)
    tests = base[rng.permutation(GATES)][:, rng.permutation(CLASSES)]
    tests = np.bitwise_xor(tests, rng.integers(0, 2, size=(GATES, 1), dtype=np.int64))
    assert np.all(tests.sum(1) == 2)
    assert len({tuple(tests[:, g]) for g in range(CLASSES)}) == CLASSES
    q_rare = float(rng.uniform(0.96, 0.995))
    jitter = float(rng.uniform(-0.02, 0.02))
    return World(int(seed), q_rare, float(q_common + jitter), tests)


@dataclass
class Bank:
    obs: np.ndarray       # (n, T, OBS_DIM) float32
    valid: np.ndarray     # (n, T) bool
    gate_t: np.ndarray    # (n, 3) gate times
    regime: np.ndarray    # (n,)
    cues: np.ndarray      # (n, 3)
    delay: np.ndarray     # (n,)
    goal: np.ndarray      # (n,) hidden class, never read before the lock on target banks
    ask: np.ndarray       # (3, 4) test asked at each gate
    seed: int

    @property
    def n(self) -> int:
        return len(self.goal)

    def correct(self) -> np.ndarray:
        return np.stack([self.ask[j, self.goal] for j in range(GATES)], axis=1)


def sym_post(cues: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Posterior over the four classes from three symmetric-noise cues."""
    cues = np.asarray(cues, dtype=np.int64)
    q = np.clip(np.asarray(q, dtype=np.float64), 0.2501, 0.9999)
    n = len(cues)
    logp = np.zeros((n, CLASSES))
    right = np.log(q)
    wrong = np.log((1.0 - q) / (CLASSES - 1))
    for j in range(cues.shape[1]):
        logp += wrong[:, None]
        logp[np.arange(n), cues[:, j]] += right - wrong
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def answer_probs(post: np.ndarray, tests: np.ndarray) -> np.ndarray:
    """P(answer 1 is correct) for each history and test row."""
    return post @ np.asarray(tests, dtype=np.float64).T


def make_bank(world: World, n: int, delays, seed: int, frac: float, ask: np.ndarray) -> Bank:
    rng = np.random.default_rng(int(seed))
    delays = np.asarray(list(delays), dtype=np.int64)
    goal = rng.integers(0, CLASSES, size=n)
    regime = (rng.random(n) < frac).astype(np.int64)
    q = np.where(regime == 1, world.q_rare, world.q_common)
    ok = rng.random((n, CUES)) < q[:, None]
    wrong = rng.integers(0, CLASSES - 1, size=(n, CUES))
    wrong = wrong + (wrong >= goal[:, None])
    cues = np.where(ok, goal[:, None], wrong)
    d = rng.choice(delays, size=n)
    gate0 = 1 + CUES + d
    gate_t = np.stack([gate0, gate0 + 3, gate0 + 6], axis=1)
    length = gate0 + 7
    T = int(length.max())
    t = np.arange(T)[None, :]
    valid = t < length[:, None]
    obs = np.zeros((n, T, OBS_DIM), dtype=np.float32)
    rows = np.arange(n)[:, None]
    # regime flag on every valid step
    obs[rows, t, REG + regime[:, None]] = valid.astype(np.float32)
    # start token
    obs[:, 0, START] = 1.0
    # cues at steps 1..3
    for j in range(CUES):
        obs[np.arange(n), 1 + j, CUE + cues[:, j]] = 1.0
    # gates
    is_gate = np.zeros((n, T), dtype=bool)
    for j in range(GATES):
        obs[np.arange(n), gate_t[:, j], GAT + j] = 1.0
        is_gate[np.arange(n), gate_t[:, j]] = True
    # distractors on every other valid step after the start
    is_cue = (t >= 1) & (t <= CUES)
    fill = valid & (t >= 1) & ~is_cue & ~is_gate
    did = rng.integers(0, DISTRACTORS, size=(n, T))
    rr, tt = np.nonzero(fill)
    obs[rr, tt, DIS + did[rr, tt]] = 1.0
    # padding
    obs[~valid, PAD] = 1.0
    obs[~valid, REG:REG + 2] = 0.0
    return Bank(obs, valid, gate_t, regime, cues, d, goal, np.asarray(ask, dtype=np.int64), int(seed))


def true_post(world: World, bank: Bank) -> np.ndarray:
    q = np.where(bank.regime == 1, world.q_rare, world.q_common)
    return sym_post(bank.cues, q)


def last_cue_targets(bank: Bank) -> np.ndarray:
    """Self-supervised target at each step: the most recent cue symbol seen, 4 before any cue."""
    n, T = bank.valid.shape
    out = np.full((n, T), 4, dtype=np.int64)
    for j in range(CUES):
        out[:, 1 + j:] = bank.cues[:, j][:, None]
    return out
