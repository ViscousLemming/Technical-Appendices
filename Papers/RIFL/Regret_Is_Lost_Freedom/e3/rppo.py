"""Episodic recurrent PPO for RockSample and rollouts that record decision points, as used in the RockSample pilot of E3."""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from e2.agent import Spec, build
from e3.rocks import ACTIONS, OBS_DIM, ROCKS, SIZE, STEP_LIMIT, RockSample


@dataclass(frozen=True)
class Cfg:
    updates: int = 400
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
    rscale: float = 0.1


def starts(rng, n, cols):
    return np.stack([rng.integers(0, SIZE, size=n), rng.choice(np.asarray(cols), size=n)], axis=1)


def rollout(net, rocks, start, good, seed: int, greedy: bool = False, readout=None, record: bool = True):
    """Run a batch of episodes.

    readout, if given, is a function feat -> (n,) 0/1 that decides sampling at the first visit to each rock cell.
    With a readout the policy's sample action is masked everywhere else.
    """
    rng = np.random.default_rng(int(seed))
    gen = torch.Generator().manual_seed(int(seed))
    env = RockSample(rocks, start, good, rng)
    n, T = env.n, STEP_LIMIT
    state = net.init_state(n)
    obs = np.zeros((n, T, OBS_DIM), dtype=np.float32)
    act = np.zeros((n, T), dtype=np.int64)
    logp = np.zeros((n, T), dtype=np.float32)
    val = np.zeros((n, T), dtype=np.float32)
    rew = np.zeros((n, T), dtype=np.float32)
    valid = np.zeros((n, T), dtype=bool)
    pos = np.zeros((n, T, 2), dtype=np.int64)       # position at the start of step t (what obs[t] shows)
    visited = np.zeros((n, ROCKS), dtype=bool)
    dec = []                                          # decision points: (episode, step, rock, feat, probs)
    feats = [] if record else None
    last = 0
    for t in range(T):
        live = ~env.done
        if not live.any():
            break
        last = t
        o = env.obs()
        pos[:, t] = env.pos
        with torch.no_grad():
            logits, v, feat, state = net.step(torch.from_numpy(o), state)
            p = torch.softmax(logits, -1)
        here = env.rock_here()
        first = live & (here >= 0)
        first[first] = ~visited[np.nonzero(first)[0], here[first]]
        if readout is not None:
            q = p.clone()
            q[:, 4] = 0.0
            q = q / q.sum(-1, keepdim=True)
            a = q.argmax(-1) if greedy else torch.multinomial(q, 1, generator=gen)[:, 0]
            if first.any():
                idx = np.nonzero(first)[0]
                take = readout(feat.numpy()[idx]).astype(bool)
                a[torch.from_numpy(idx[take])] = 4
        else:
            a = p.argmax(-1) if greedy else torch.multinomial(p, 1, generator=gen)[:, 0]
        if first.any():
            idx = np.nonzero(first)[0]
            fn = feat.numpy()
            pn = p.numpy()
            for i in idx:
                dec.append((int(i), t, int(here[i]), fn[i].copy(), pn[i].copy()))
            visited[idx, here[idx]] = True
        a_np = a.numpy()
        obs[:, t] = o
        act[:, t] = a_np
        logp[:, t] = torch.log(p.gather(1, a[:, None])[:, 0] + 1e-12).numpy()
        val[:, t] = v.numpy()
        r, _ = env.step(a_np)
        valid[:, t] = live
        rew[:, t] = r
        if record:
            feats.append(feat.numpy())
    Tu = last + 1
    return {"obs": obs[:, :Tu], "act": act[:, :Tu], "logp": logp[:, :Tu], "val": val[:, :Tu], "rew": rew[:, :Tu],
            "valid": valid[:, :Tu], "pos": pos[:, :Tu], "dec": dec, "ret": env.ret.copy(), "good0": env.good0,
            "feats": np.stack(feats, 1) if record else None}


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


def train(spec: Spec, cfg: Cfg, rocks, cols, seed: int, snaps, deadline=None):
    net = build(spec, OBS_DIM, ACTIONS)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)
    rng = np.random.default_rng(int(seed))
    snaps = set(int(s) for s in snaps)
    out, rets = {}, []
    for u in range(1, cfg.updates + 1):
        net.eval()
        st = starts(rng, cfg.episodes, cols)
        good = rng.integers(0, 2, size=(cfg.episodes, ROCKS))
        ro = rollout(net, rocks, st, good, int(rng.integers(2**31 - 1)), record=False)
        rets.append(float(ro["ret"].mean()))
        adv, ret = gae(ro["rew"] * cfg.rscale, ro["val"], ro["valid"], cfg.gamma, cfg.lam)
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
        if deadline is not None and u % 20 == 0:
            import time
            if time.time() > deadline:
                break
    net.eval()
    out["final"] = copy.deepcopy(net.state_dict())
    out["updates_done"] = u
    return out, {"train_return": float(np.mean(rets[-10:])), "curve": [float(np.mean(rets[i:i + 20])) for i in range(0, len(rets), 20)]}


def load(spec: Spec, state: dict):
    net = build(spec, OBS_DIM, ACTIONS)
    net.load_state_dict(state)
    net.eval()
    return net
