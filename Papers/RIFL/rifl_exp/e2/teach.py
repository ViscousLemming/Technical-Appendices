"""Set-pooling teacher for E2: per-step embeddings, max pooled over the history, trained on labelled source episodes."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SetMax(nn.Module):
    def __init__(self, obs_dim: int, out: int = 1, h: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, out))

    def forward(self, obs, valid):
        e = self.phi(obs) * valid.unsqueeze(-1)       # ReLU features are non-negative, so padding at zero is neutral
        return self.rho(e.max(1).values)


def pad(histories: list[np.ndarray], dim: int) -> tuple[np.ndarray, np.ndarray]:
    T = max(len(h) for h in histories)
    obs = np.zeros((len(histories), T, dim), dtype=np.float32)
    valid = np.zeros((len(histories), T), dtype=bool)
    for i, h in enumerate(histories):
        obs[i, :len(h)] = h
        valid[i, :len(h)] = True
    return obs, valid


def fit(obs: np.ndarray, valid: np.ndarray, y: np.ndarray, seed: int, epochs: int = 30, lr: float = 3e-3) -> SetMax:
    """Binary teacher: y is the label (goal south)."""
    torch.manual_seed(int(seed))
    net = SetMax(obs.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    o, v, t = torch.from_numpy(obs), torch.from_numpy(valid).float(), torch.from_numpy(y.astype(np.float32))
    rng = np.random.default_rng(int(seed))
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), 256):
            ids = torch.from_numpy(order[s:s + 256])
            loss = nn.functional.binary_cross_entropy_with_logits(net(o[ids], v[ids])[:, 0], t[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    net.eval()
    return net


def post(net: SetMax, obs: np.ndarray, valid: np.ndarray, chunk: int = 2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(obs), chunk):
            out.append(torch.sigmoid(net(torch.from_numpy(obs[s:s + chunk]), torch.from_numpy(valid[s:s + chunk]).float())[:, 0]).numpy())
    return np.concatenate(out).astype(np.float64)
