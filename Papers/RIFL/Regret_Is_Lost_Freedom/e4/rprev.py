"""Batched NumPy port of POPGym RepeatPrevious(num_decks=1, k), its check against popgym, and the lag rules.

An episode is the suit order of a shuffled 52-card deck. The reset shows o_1 and step t
(t = 1..51) is taken having seen o_1..o_t. Under lag k the correct action at step t is
o_{t-k+1}, defined for t >= k, and the reward there is +-1/(52-k). Arrays below index
steps from 0, so step t is column t-1 and the lag-k answer at column i is suits[:, i-k+1].
"""
from __future__ import annotations


import numpy as np

CARDS = 52
STEPS = CARDS - 1          # actions per episode
SUITS = 4
OBS_DIM = 4
ACTIONS = 4
_DECK_SUITS = np.tile(np.arange(SUITS), CARDS // SUITS)     # popgym Deck.suits_idx for one deck


def deal(rng: np.random.Generator, n: int) -> np.ndarray:
    """Suit orders (n, 52) of n shuffled decks, the port's episode generator for training."""
    return rng.permuted(np.tile(_DECK_SUITS, (n, 1)), axis=1).astype(np.int64)


def obs(suits: np.ndarray) -> np.ndarray:
    """One-hot observations (n, 51, 4): the card seen at each of the 51 steps."""
    return np.eye(SUITS, dtype=np.float32)[suits[:, :STEPS]]


def labels(suits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Correct actions (n, 51) under lag k and the mask of steps t >= k where the question is posed."""
    n = len(suits)
    y = np.zeros((n, STEPS), dtype=np.int64)
    m = np.zeros(STEPS, dtype=bool)
    m[k - 1:] = True
    y[:, k - 1:] = suits[:, :STEPS - k + 1]
    return y, np.broadcast_to(m, (n, STEPS))


def rewards(suits: np.ndarray, act: np.ndarray, k: int) -> np.ndarray:
    """Per-step rewards (n, 51) of actions under lag k."""
    y, m = labels(suits, k)
    r = np.where(act == y, 1.0, -1.0) / (CARDS - k)
    return np.where(m, r, 0.0)


def expected_rewards(suits: np.ndarray, probs: np.ndarray, k: int) -> np.ndarray:
    """Per-step expected rewards (n, 51) of a stochastic policy with action probabilities probs (n, 51, 4)."""
    y, m = labels(suits, k)
    pc = np.take_along_axis(probs, y[..., None], -1)[..., 0]
    return np.where(m, (2.0 * pc - 1.0) / (CARDS - k), 0.0)


# ---------------------------------------------------------------------------
# the official environment
# ---------------------------------------------------------------------------

def official(k: int = 4):
    from popgym.envs.repeat_previous import RepeatPrevious
    return RepeatPrevious(num_decks=1, k=k)


def dealt_order(env) -> np.ndarray:
    """Suit order the environment will deal after reset: Deck.deal takes cards from the end of deck.idx."""
    return env.deck.suits_idx[env.deck.idx[::-1]].astype(np.int64)


def official_episodes(seeds, k: int = 4, act_fn=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the official environment on seeded episodes. Returns the dealt suit orders (n, 52), the observations
    it returned at steps 1..51 (n, 51) and its rewards (n, 51) for the action sequence act_fn(i, t)."""
    env = official(k)
    n = len(seeds)
    order = np.zeros((n, CARDS), dtype=np.int64)
    seen = np.zeros((n, STEPS), dtype=np.int64)
    rew = np.zeros((n, STEPS))
    for i, s in enumerate(seeds):
        o, _ = env.reset(seed=int(s))
        order[i] = dealt_order(env)
        for t in range(STEPS):
            seen[i, t] = int(o)
            a = 0 if act_fn is None else int(act_fn(i, t))
            o, r, term, trunc, _ = env.step(a)
            rew[i, t] = r
            assert term == (t == STEPS - 1)
        assert int(o) == order[i, -1]
    return order, seen, rew


def port_check(seed: int, n: int = 200, lags=range(1, 17)) -> bool:
    """Official RepeatPrevious(1, 4) against the port on n seeded episodes: observations, rewards under a fixed
    action sequence, and the labels of every lag in lags (each checked by the official reward of lag k)."""
    seeds = [int(seed) + i for i in range(n)]
    act = np.random.default_rng(int(seed)).integers(0, ACTIONS, size=(n, STEPS))
    order, seen, rew = official_episodes(seeds, 4, lambda i, t: act[i, t])
    assert np.array_equal(seen, order[:, :STEPS]), "observations differ"
    assert np.array_equal(obs(order).argmax(-1), seen), "port observations differ"
    assert np.allclose(rewards(order, act, 4), rew, atol=1e-15, rtol=0), "rewards differ"
    # labels for every lag: the official environment with lag k pays +1/(52-k) exactly when the port's label is played
    env = {k: official(k) for k in lags}
    for i in range(min(n, 200)):
        for k in lags:
            e = env[k]
            e.reset(seed=seeds[i])
            y, m = labels(order[i:i + 1], k)
            for t in range(STEPS):
                _, r, _, _, _ = e.step(int(y[0, t]))
                want = 1.0 / (CARDS - k) if m[0, t] else 0.0
                assert abs(r - want) < 1e-15, f"label differs at lag {k} step {t + 1}"
    return True


def popgym_version() -> str:
    from importlib.metadata import version
    return version("popgym")


def official_orders(seeds, k: int = 4) -> np.ndarray:
    """Suit orders (n, 52) the official environment deals after reset with each seed, without stepping."""
    env = official(k)
    out = np.zeros((len(seeds), CARDS), dtype=np.int64)
    for i, s in enumerate(seeds):
        env.reset(seed=int(s))
        out[i] = dealt_order(env)
    return out
