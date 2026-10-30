"""Episodic PPO on Repeat Previous with the E2 settings and the E2 agents.

Actions never change what the agent sees, so an episode's observations are fixed by its
suit order and one causal forward pass over the 51 steps gives the policy, value and
memory at every step (the same computation as stepping the recurrent or windowed memory).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from e2.agent import RNNNet, Spec, TrXL
from e2.ppo import gae
from e4 import rprev as R


@dataclass(frozen=True)
class Cfg:
    updates: int = 300
    episodes: int = 64
    epochs: int = 3
    minibatch: int = 32
    lr: float = 1e-3
    ent: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf: float = 0.5
    grad: float = 0.5
    k_src: int = 4                                   # the lag training poses
    new_lags: tuple = (2, 3, 5, 6, 7, 8)             # the lags deployment poses beside k_src
    margin: float = 0.10                             # admission: native source return within this of the best

    def __post_init__(self):
        object.__setattr__(self, "new_lags", tuple(int(k) for k in self.new_lags))


def build(spec: Spec) -> nn.Module:
    """gru and lstm of width size, trxl with two blocks and window size, trxl1 with one block and window size."""
    if spec.arch in ("gru", "lstm"):
        return RNNNet(spec, R.OBS_DIM, R.ACTIONS)
    if spec.arch == "trxl":
        return TrXL(spec, R.OBS_DIM, R.ACTIONS, layers=2)
    if spec.arch == "trxl1":
        return TrXL(spec, R.OBS_DIM, R.ACTIONS, layers=1)
    raise ValueError(spec.arch)


def load(spec: Spec, state: dict) -> nn.Module:
    net = build(spec)
    net.load_state_dict(state)
    net.eval()
    return net


def n_params(net: nn.Module) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def probe(net, suits: np.ndarray, batch: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Memories (n, 51, width) read by the policy head and action probabilities (n, 51, 4) on fixed episodes."""
    feats, probs = [], []
    with torch.no_grad():
        for s in range(0, len(suits), batch):
            logits, _, feat = net(torch.from_numpy(R.obs(suits[s:s + batch])))
            feats.append(feat.numpy())
            probs.append(torch.softmax(logits, -1).numpy())
    return np.concatenate(feats), np.concatenate(probs)


def train(spec: Spec, cfg: Cfg, seed: int, snaps):
    """PPO on source episodes (lag cfg.k_src). Returns state dicts at the snapshot updates and a training log."""
    net = build(spec)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    rng = np.random.default_rng(int(seed))
    snaps = set(int(s) for s in snaps)
    out, rets = {}, []
    n, T = cfg.episodes, R.STEPS
    valid = np.ones((n, T), dtype=bool)
    for u in range(1, cfg.updates + 1):
        net.eval()
        suits = R.deal(rng, n)
        obs = torch.from_numpy(R.obs(suits))
        gen = torch.Generator().manual_seed(int(rng.integers(2**31 - 1)))
        with torch.no_grad():
            logits, v, _ = net(obs)
            p = torch.softmax(logits, -1)
            a = torch.multinomial(p.reshape(-1, R.ACTIONS), 1, generator=gen).reshape(n, T)
            oldlp = torch.log(p.gather(-1, a[..., None])[..., 0] + 1e-12)
        act = a.numpy()
        rew = R.rewards(suits, act, cfg.k_src).astype(np.float32)
        rets.append(float(rew.sum(1).mean()))
        adv, ret = gae(rew, v.numpy().astype(np.float32), valid, cfg.gamma, cfg.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        act_t = torch.from_numpy(act)
        advt = torch.from_numpy(adv.astype(np.float32))
        rett = torch.from_numpy(ret.astype(np.float32))
        net.train()
        for _ in range(cfg.epochs):
            order = rng.permutation(n)
            for s in range(0, n, cfg.minibatch):
                ids = torch.from_numpy(order[s:s + cfg.minibatch])
                logits, vv, _ = net(obs[ids])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act_t[ids])
                ratio = torch.exp(lp - oldlp[ids])
                ad = advt[ids]
                pol = -torch.minimum(ratio * ad, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ad).mean()
                vloss = ((vv - rett[ids]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pol + cfg.vf * vloss - cfg.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad)
                opt.step()
        if u in snaps:
            out[u] = copy.deepcopy(net.state_dict())
    net.eval()
    return out, {"train_return": float(np.mean(rets[-10:])),
                 "curve": [float(np.mean(rets[i:i + 10])) for i in range(0, len(rets), 10)]}
