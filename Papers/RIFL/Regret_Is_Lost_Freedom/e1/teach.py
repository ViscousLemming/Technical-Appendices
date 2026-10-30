"""Teachers for E1. Each maps a history to a posterior over the four hidden classes.

All teachers train on labelled source episodes only. The set teacher pools
per-step embeddings by summation (Zaheer et al. 2017) and is primary. The
sequence teacher is a GRU over the episode. The analytic teacher applies the
symmetric-noise posterior with cue reliabilities estimated from source.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from e1.env import Bank, CLASSES, CUE, OBS_DIM, sym_post


def fit_analytic(bank: Bank) -> tuple[float, float]:
    match = bank.cues == bank.goal[:, None]
    out = []
    for r in (1, 0):
        m = bank.regime == r
        out.append(float(np.clip(match[m].mean(), 0.2501, 0.999)) if m.any() else 0.9)
    return out[0], out[1]


def analytic_post(q: tuple[float, float], bank: Bank) -> np.ndarray:
    qq = np.where(bank.regime == 1, q[0], q[1])
    return sym_post(bank.cues, qq)


class SetNet(nn.Module):
    def __init__(self, h: int = 64):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(OBS_DIM, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h + 2, h), nn.ReLU(), nn.Linear(h, CLASSES))

    def forward(self, obs, valid):
        e = self.phi(obs) * valid.unsqueeze(-1)
        pooled = e.sum(1)
        reg = obs[:, 0, 0:2]
        return self.rho(torch.cat([pooled, reg], -1))


class SeqNet(nn.Module):
    def __init__(self, h: int = 64):
        super().__init__()
        self.rnn = nn.GRU(OBS_DIM, h, batch_first=True)
        self.out = nn.Linear(h, CLASSES)

    def forward(self, obs, valid):
        hid, _ = self.rnn(obs)
        last = (valid.sum(1) - 1).long()
        return self.out(hid[torch.arange(len(obs)), last])


def fit_net(kind: str, bank: Bank, seed: int, epochs: int = 40, lr: float = 3e-3):
    torch.manual_seed(int(seed))
    net = SetNet() if kind == "set" else SeqNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    obs = torch.from_numpy(bank.obs)
    valid = torch.from_numpy(bank.valid).float()
    y = torch.from_numpy(bank.goal)
    rng = np.random.default_rng(int(seed))
    n = bank.n
    for _ in range(epochs):
        order = rng.permutation(n)
        for s in range(0, n, 128):
            ids = torch.from_numpy(order[s:s + 128])
            loss = nn.functional.cross_entropy(net(obs[ids], valid[ids]), y[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return net


def net_post(net, bank: Bank) -> np.ndarray:
    with torch.no_grad():
        logits = net(torch.from_numpy(bank.obs), torch.from_numpy(bank.valid).float())
        return torch.softmax(logits, -1).numpy().astype(np.float64)


def noisy(post: np.ndarray, sigma: float, seed: int) -> np.ndarray:
    """Teacher error for the P5 test: Gaussian noise on log-posteriors, renormalised."""
    if sigma <= 0:
        return post
    rng = np.random.default_rng(int(seed))
    lp = np.log(np.clip(post, 1e-9, 1.0)) + sigma * rng.standard_normal(post.shape)
    lp -= lp.max(1, keepdims=True)
    p = np.exp(lp)
    return p / p.sum(1, keepdims=True)
