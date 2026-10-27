"""Recurrent actor-critic for E1 and episodic PPO trained on source rewards only."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from e1.env import Bank, World, make_bank, last_cue_targets, OBS_DIM, GATES


@dataclass(frozen=True)
class Spec:
    arch: str        # gru or lstm
    width: int
    delays: str      # short (3-7), mid (3-14) or long (3-28)
    mix: str         # nat (world source share) or rare (0.5)
    aux: int         # 1 adds a self-supervised last-cue head
    seed: int

    @property
    def name(self) -> str:
        return f"{self.arch}{self.width}_{self.delays}_{self.mix}_a{self.aux}_s{self.seed}"

    def delay_range(self):
        return {"short": range(3, 8), "mid": range(3, 15), "long": range(3, 29)}[self.delays]


class Net(nn.Module):
    def __init__(self, spec: Spec, actions: int = 2):
        super().__init__()
        torch.manual_seed(int(spec.seed))
        h = int(spec.width)
        rnn = {"gru": nn.GRU, "lstm": nn.LSTM}[spec.arch]
        self.rnn = rnn(OBS_DIM, h, batch_first=True)
        self.actor = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, actions))
        self.critic = nn.Sequential(nn.Linear(h, h), nn.Tanh(), nn.Linear(h, 1))
        self.aux = nn.Linear(h, 5) if spec.aux else None
        for name, p in self.named_parameters():
            if p.ndim >= 2:
                nn.init.orthogonal_(p, gain=1.0)
            else:
                nn.init.zeros_(p)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def forward(self, obs: torch.Tensor):
        hidden, _ = self.rnn(obs)
        logits = self.actor(hidden)
        value = self.critic(hidden).squeeze(-1)
        aux = self.aux(hidden) if self.aux is not None else None
        return logits, value, hidden, aux


@dataclass(frozen=True)
class Cfg:
    updates: int = 150
    episodes: int = 192
    epochs: int = 2
    minibatch: int = 64
    lr: float = 2.5e-3
    ent: float = 0.01
    aux_w: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    vf: float = 0.5
    grad: float = 1.0


def _gate_mask(bank: Bank) -> np.ndarray:
    n, T = bank.valid.shape
    m = np.zeros((n, T), dtype=bool)
    for j in range(GATES):
        m[np.arange(n), bank.gate_t[:, j]] = True
    return m


def _gae(rew, val, valid, gamma, lam):
    b, T = rew.shape
    adv = torch.zeros_like(rew)
    x = torch.zeros(b)
    for t in range(T - 1, -1, -1):
        if t + 1 < T:
            nonterm = valid[:, t + 1].float()
            nextv = val[:, t + 1]
        else:
            nonterm = torch.zeros(b)
            nextv = torch.zeros(b)
        delta = rew[:, t] + gamma * nextv * nonterm - val[:, t]
        x = (delta + gamma * lam * nonterm * x) * valid[:, t].float()
        adv[:, t] = x
    return adv, adv + val


def train(world: World, spec: Spec, ask: np.ndarray, frac_nat: float, cfg: Cfg, snaps, seed: int):
    """Train one agent on source episodes. Returns {update: state_dict} and a small log."""
    net = Net(spec)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    rng = np.random.default_rng(int(seed))
    frac = frac_nat if spec.mix == "nat" else 0.5
    snaps = set(int(s) for s in snaps)
    out, rets = {}, []
    for u in range(1, cfg.updates + 1):
        bank = make_bank(world, cfg.episodes, spec.delay_range(), int(rng.integers(2**31 - 1)), frac, ask)
        obs = torch.from_numpy(bank.obs)
        valid = torch.from_numpy(bank.valid)
        gm = torch.from_numpy(_gate_mask(bank))
        correct = torch.from_numpy(bank.correct())
        auxy = torch.from_numpy(last_cue_targets(bank)) if spec.aux else None
        with torch.no_grad():
            logits, val, _, _ = net(obs)
            dist = torch.distributions.Categorical(logits=logits)
            act = dist.sample()
            old_lp = dist.log_prob(act)
            rew = torch.zeros_like(val)
            for j in range(GATES):
                tj = torch.from_numpy(bank.gate_t[:, j])
                rows = torch.arange(bank.n)
                rew[rows, tj] = (act[rows, tj] == correct[:, j]).float()
            adv, ret = _gae(rew, val, valid, cfg.gamma, cfg.lam)
            a = adv[gm]
            adv[gm] = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
            rets.append(float(rew.sum(1).mean() / GATES))
        order = np.arange(bank.n)
        for _ in range(cfg.epochs):
            rng.shuffle(order)
            for s in range(0, bank.n, cfg.minibatch):
                ids = torch.from_numpy(order[s:s + cfg.minibatch].astype(np.int64))
                logits, v, _, auxl = net(obs[ids])
                dist = torch.distributions.Categorical(logits=logits)
                lp = dist.log_prob(act[ids])
                m = gm[ids]
                ratio = torch.exp(lp - old_lp[ids])[m]
                ad = adv[ids][m]
                pol = -torch.minimum(ratio * ad, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * ad).mean()
                vm = valid[ids].float()
                vloss = ((v - ret[ids]).square() * vm).sum() / vm.sum()
                ent = dist.entropy()[m].mean()
                loss = pol + cfg.vf * vloss - cfg.ent * ent
                if auxl is not None:
                    ce = nn.functional.cross_entropy(auxl.reshape(-1, 5), auxy[ids].reshape(-1), reduction="none")
                    loss = loss + cfg.aux_w * (ce * vm.reshape(-1)).sum() / vm.sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.grad)
                opt.step()
        if u in snaps:
            out[u] = copy.deepcopy(net.state_dict())
    return out, {"train_success": float(np.mean(rets[-10:]))}


def load(spec: Spec, state: dict) -> Net:
    net = Net(spec)
    net.load_state_dict(state)
    net.eval()
    return net


def run(net: Net, bank: Bank):
    """Pre-test memory (hidden state before the first gate) and answer probabilities at each gate."""
    with torch.no_grad():
        logits, _, hidden, _ = net(torch.from_numpy(bank.obs))
        probs = torch.softmax(logits, -1).numpy().astype(np.float64)
        hidden = hidden.numpy().astype(np.float64)
    rows = np.arange(bank.n)
    mem = hidden[rows, bank.gate_t[:, 0] - 1]
    gp = np.stack([probs[rows, bank.gate_t[:, j]] for j in range(GATES)], axis=1)
    return mem, gp
