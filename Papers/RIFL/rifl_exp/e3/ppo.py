"""Episodic recurrent PPO for the noisy T-Maze (E3), plus on-policy rollouts that record memories and readings."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from e2.agent import Spec, build
from e3.nmaze import ACTIONS, OBS_DIM, NTMaze


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


def rollout(net, lengths, goal, regime, seed: int, greedy: bool = False, keep_obs: bool = True):
    """Run the policy on a batch of episodes. Returns arrays over (episode, step)."""
    env = NTMaze(lengths, goal, regime, np.random.default_rng(int(seed) + 17))
    n = env.n
    T = int(env.tmax.max())
    gen = torch.Generator().manual_seed(int(seed))
    state = net.init_state(n)
    obs = np.zeros((n, T, OBS_DIM), dtype=np.float32)
    act = np.zeros((n, T), dtype=np.int64)
    logp = np.zeros((n, T), dtype=np.float32)
    val = np.zeros((n, T), dtype=np.float32)
    rew = np.zeros((n, T), dtype=np.float32)
    valid = np.zeros((n, T), dtype=bool)
    probs = np.zeros((n, T, ACTIONS), dtype=np.float32)
    feat_j = None
    arrive = np.full(n, -1, dtype=np.int64)
    reads_j = np.zeros((n, 2), dtype=np.int64)
    last = 0
    for t in range(T):
        live = ~env.done
        if not live.any():
            break
        last = t
        o = env.obs()
        with torch.no_grad():
            logits, v, feat, state = net.step(torch.from_numpy(o), state)
            p = torch.softmax(logits, -1)
            a = p.argmax(-1) if greedy else torch.multinomial(p, 1, generator=gen)[:, 0]
        if feat_j is None:
            feat_j = np.zeros((n, feat.shape[1]), dtype=np.float32)
        newj = live & env.at_junction() & (arrive < 0)
        if newj.any():
            arrive[newj] = t
            feat_j[newj] = feat.numpy()[newj]
            reads_j[newj] = env.reads[newj]
        obs[:, t] = o
        a_np = a.numpy()
        act[:, t] = a_np
        logp[:, t] = torch.log(p.gather(1, a[:, None])[:, 0] + 1e-12).numpy()
        val[:, t] = v.numpy()
        probs[:, t] = p.numpy()
        r, _ = env.step(a_np)
        valid[:, t] = live
        rew[:, t] = r
    Tu = last + 1
    out = {"obs": obs[:, :Tu] if keep_obs else None, "act": act[:, :Tu], "logp": logp[:, :Tu], "val": val[:, :Tu],
           "rew": rew[:, :Tu], "valid": valid[:, :Tu], "probs": probs[:, :Tu], "arrive": arrive,
           "feat_j": feat_j, "success": env.success.copy(), "goal": env.goal.copy(), "lengths": env.L.copy(),
           "regime": env.regime.copy(), "reads_j": reads_j, "reads": env.reads.copy(), "ret": env.ret.copy()}
    return out


def gae(rew, val, valid, gamma, lam):
    n, T = rew.shape
    adv = np.zeros((n, T), dtype=np.float32)
    x = np.zeros(n, dtype=np.float32)
    for t in range(T - 1, -1, -1):
        nonterm = valid[:, t + 1] if t + 1 < T else np.zeros(n, dtype=bool)
        nextv = val[:, t + 1] if t + 1 < T else np.zeros(n, dtype=np.float32)
        delta = rew[:, t] + gamma * nextv * nonterm - val[:, t]
        x = (delta + gamma * lam * nonterm * x) * valid[:, t]
        adv[:, t] = x
    return adv, adv + val


def train(spec: Spec, cfg: Cfg, lengths_fn, seed: int, snaps, regime_share: float = 0.1):
    """PPO on source episodes. lengths_fn(rng, n) draws hallway lengths."""
    net = build(spec, OBS_DIM, ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    rng = np.random.default_rng(int(seed))
    snaps = set(int(s) for s in snaps)
    out, succ = {}, []
    for u in range(1, cfg.updates + 1):
        net.eval()
        L = lengths_fn(rng, cfg.episodes)
        g = rng.integers(0, 2, size=cfg.episodes)
        rg = (rng.random(cfg.episodes) < regime_share).astype(np.int64)
        ro = rollout(net, L, g, rg, int(rng.integers(2**31 - 1)))
        succ.append(float(ro["ret"].mean()))
        adv, ret = gae(ro["rew"], ro["val"], ro["valid"], cfg.gamma, cfg.lam)
        m = ro["valid"]
        a = adv[m]
        adv[m] = (a - a.mean()) / (a.std() + 1e-8)
        obs = torch.from_numpy(ro["obs"])
        act = torch.from_numpy(ro["act"])
        oldlp = torch.from_numpy(ro["logp"])
        advt = torch.from_numpy(adv)
        rett = torch.from_numpy(ret.astype(np.float32))
        mt = torch.from_numpy(m)
        net.train()
        for _ in range(cfg.epochs):
            order = rng.permutation(cfg.episodes)
            for s in range(0, cfg.episodes, cfg.minibatch):
                ids = torch.from_numpy(order[s:s + cfg.minibatch])
                logits, v, _ = net(obs[ids])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act[ids])
                mm = mt[ids]
                ratio = torch.exp(lp - oldlp[ids])[mm]
                ad = advt[ids][mm]
                pol = -torch.minimum(ratio * ad, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ad).mean()
                vloss = ((v - rett[ids]) ** 2)[mm].mean()
                ent = dist.entropy()[mm].mean()
                loss = pol + cfg.vf * vloss - cfg.ent * ent
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad)
                opt.step()
        if u in snaps:
            out[u] = copy.deepcopy(net.state_dict())
    net.eval()
    return out, {"train_return": float(np.mean(succ[-10:])), "curve": [float(np.mean(succ[i:i + 10])) for i in range(0, len(succ), 10)]}


def load(spec: Spec, state: dict):
    net = build(spec, OBS_DIM, ACTIONS)
    net.load_state_dict(state)
    net.eval()
    return net


def feed(net, obs: np.ndarray, valid: np.ndarray):
    """Teacher-forced pass over fixed observation histories. Returns features and action probabilities at each step."""
    with torch.no_grad():
        logits, _, feat = net(torch.from_numpy(obs))
        return feat.numpy(), torch.softmax(logits, -1).numpy()
