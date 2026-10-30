"""Memory agents for E2 and E3: GRU, LSTM and a windowed gated transformer (GTrXL-style).

Every network exposes the same interface.
  forward(obs)            full sequences (B, T, D) -> logits, value, feat
  init_state(n)           empty memory for n episodes
  step(obs_t, state)      one step -> logits, value, feat, new state
The feature feat is what the policy head reads. It is the memory M(h) that the
readout languages act on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class Spec:
    arch: str      # gru, lstm or trxl
    size: int      # hidden width for gru and lstm, attention window for trxl
    seed: int

    @property
    def name(self) -> str:
        return f"{self.arch}{self.size}_s{self.seed}"


def _heads(width: int, actions: int):
    actor = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, actions))
    critic = nn.Sequential(nn.Linear(width, 64), nn.Tanh(), nn.Linear(64, 1))
    return actor, critic


def _init(module: nn.Module, actor_last: nn.Linear):
    """Orthogonal weights and zero biases for linear and recurrent layers. Layer norms keep their defaults."""
    for mname, m in module.named_modules():
        if isinstance(m, nn.Linear):
            gain = math.sqrt(2) if mname.startswith(("actor", "critic")) else 1.0
            nn.init.orthogonal_(m.weight, gain=gain)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GRU, nn.LSTM)):
            for pname, p in m.named_parameters():
                if "weight" in pname:
                    nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)
    nn.init.orthogonal_(actor_last.weight, gain=0.01)


class RNNNet(nn.Module):
    def __init__(self, spec: Spec, obs_dim: int, actions: int):
        super().__init__()
        torch.manual_seed(int(spec.seed))
        self.kind = spec.arch
        h = int(spec.size)
        self.rnn = (nn.GRU if spec.arch == "gru" else nn.LSTM)(obs_dim, h, batch_first=True)
        self.actor, self.critic = _heads(h, actions)
        _init(self, self.actor[-1])
        self.width = h

    def forward(self, obs):
        feat, _ = self.rnn(obs)
        return self.actor(feat), self.critic(feat).squeeze(-1), feat

    def init_state(self, n: int):
        z = torch.zeros(1, n, self.width)
        return z if self.kind == "gru" else (z, z.clone())

    def step(self, obs_t, state):
        feat, state = self.rnn(obs_t.unsqueeze(1), state)
        feat = feat[:, 0]
        return self.actor(feat), self.critic(feat).squeeze(-1), feat, state


class Gate(nn.Module):
    """GRU-type gating of a residual stream (Parisotto et al. 2020)."""

    def __init__(self, d: int, bias: float = 2.0):
        super().__init__()
        self.wr = nn.Linear(d, d, bias=False)
        self.ur = nn.Linear(d, d, bias=False)
        self.wz = nn.Linear(d, d, bias=False)
        self.uz = nn.Linear(d, d, bias=False)
        self.wg = nn.Linear(d, d, bias=False)
        self.ug = nn.Linear(d, d, bias=False)
        self.gate_bias = nn.Parameter(torch.full((d,), bias))

    def forward(self, x, y):
        r = torch.sigmoid(self.wr(y) + self.ur(x))
        z = torch.sigmoid(self.wz(y) + self.uz(x) - self.gate_bias)
        h = torch.tanh(self.wg(y) + self.ug(r * x))
        return (1 - z) * x + z * h


class Block(nn.Module):
    def __init__(self, d: int, heads: int, window: int):
        super().__init__()
        self.d, self.h, self.w = d, heads, window
        self.dh = d // heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.rel = nn.Parameter(torch.zeros(heads, window))    # bias by offset 0..window-1
        self.g1 = Gate(d)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
        self.g2 = Gate(d)

    def _split(self, z):
        b, t, _ = z.shape
        q, k, v = self.qkv(z).chunk(3, -1)
        f = lambda a: a.view(b, t, self.h, self.dh).transpose(1, 2)   # (b, h, t, dh)
        return f(q), f(k), f(v)

    def forward(self, x):
        b, t, _ = x.shape
        q, k, v = self._split(self.ln1(x))
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dh)             # (b, h, t, t)
        off = torch.arange(t)[:, None] - torch.arange(t)[None, :]       # query minus key
        band = (off >= 0) & (off < self.w)
        bias = torch.zeros(self.h, t, t)
        bias[:, band] = self.rel[:, off[band]]
        att = att + bias.unsqueeze(0)
        att = att.masked_fill(~band, float("-inf"))
        y = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, self.d)
        x = self.g1(x, torch.relu(self.out(y)))
        x = self.g2(x, torch.relu(self.mlp(self.ln2(x))))
        return x

    def step(self, x_t, cache):
        """x_t (b, d). cache: keys and values of the previous window-1 steps and their validity."""
        kc, vc, ok = cache
        b = x_t.shape[0]
        q, k, v = self._split(self.ln1(x_t).unsqueeze(1))                  # (b, h, 1, dh)
        keys = torch.cat([kc, k], 2)                                       # (b, h, w, dh)
        vals = torch.cat([vc, v], 2)
        valid = torch.cat([ok, torch.ones(b, 1, dtype=torch.bool)], 1)    # (b, w)
        att = (q @ keys.transpose(-1, -2))[:, :, 0] / math.sqrt(self.dh)   # (b, h, w)
        att = att + self.rel.flip(-1).unsqueeze(0)                         # oldest key has offset w-1
        att = att.masked_fill(~valid.unsqueeze(1), float("-inf"))
        y = (att.softmax(-1).unsqueeze(2) @ vals)[:, :, 0].reshape(b, self.d)
        x = self.g1(x_t, torch.relu(self.out(y)))
        x = self.g2(x, torch.relu(self.mlp(self.ln2(x))))
        return x, (keys[:, :, 1:], vals[:, :, 1:], valid[:, 1:])


class TrXL(nn.Module):
    def __init__(self, spec: Spec, obs_dim: int, actions: int, d: int = 32, heads: int = 2, layers: int = 2):
        super().__init__()
        torch.manual_seed(int(spec.seed))
        self.w, self.d, self.hn = int(spec.size), d, heads
        self.embed = nn.Linear(obs_dim, d)
        self.blocks = nn.ModuleList([Block(d, heads, self.w) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.actor, self.critic = _heads(d, actions)
        _init(self, self.actor[-1])
        self.width = d

    def forward(self, obs):
        x = self.embed(obs)
        for blk in self.blocks:
            x = blk(x)
        feat = self.ln(x)
        return self.actor(feat), self.critic(feat).squeeze(-1), feat

    def init_state(self, n: int):
        dh = self.d // self.hn
        z = torch.zeros(n, self.hn, self.w - 1, dh)
        ok = torch.zeros(n, self.w - 1, dtype=torch.bool)
        return [(z.clone(), z.clone(), ok.clone()) for _ in self.blocks]

    def step(self, obs_t, state):
        x = self.embed(obs_t)
        new = []
        for blk, c in zip(self.blocks, state):
            x, c2 = blk.step(x, c)
            new.append(c2)
        feat = self.ln(x)
        return self.actor(feat), self.critic(feat).squeeze(-1), feat, new


def build(spec: Spec, obs_dim: int, actions: int) -> nn.Module:
    if spec.arch in ("gru", "lstm"):
        return RNNNet(spec, obs_dim, actions)
    if spec.arch == "trxl":
        return TrXL(spec, obs_dim, actions)
    raise ValueError(spec.arch)


def select_state(state, keep: torch.Tensor):
    """Rows of a batched memory state (used when episodes finish)."""
    if isinstance(state, torch.Tensor):
        return state[:, keep]
    if isinstance(state, tuple) and len(state) == 2 and isinstance(state[0], torch.Tensor) and state[0].dim() == 3:
        return (state[0][:, keep], state[1][:, keep])
    return [(k[keep], v[keep], ok[keep]) for (k, v, ok) in state]
