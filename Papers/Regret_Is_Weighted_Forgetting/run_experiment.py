"""
Experiment.
Uses the Stack Theory Suite (STS) code library (included with but not a contribution of this paper) for two-sided verification on the Boolean domain.
Uses PyTorch for GRU/MLP training on a 1024-state structured POMDP.
Vectorised posterior computation for scalability.

Outputs all results to results_v28.json for paper integration.

Requirements: torch (cpu is fine), numpy

Experiments:
  1) Two-sided bridge verification (Boolean, 5 bits, 32 states) — unchanged.
  2) Discretised encoder at varying granularity (1024 states, 20 obs, L=15).
  3) Architecture comparison: GRU vs MLP vs last-obs vs random (1024 states).
  4) Learned representation during training (1024 states).
"""

from __future__ import annotations
import sys, os, json, time, math, random
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
import numpy as np

# ---- PyTorch (lazy — only needed for experiments 2-4) ----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
    torch.manual_seed(42)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ---- STS (only for experiment 1) ----
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "sts", "STS_v0.8.4"))
from stacktheory.layer1.environment import IndexEnvironment
from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Vocabulary, Statement


# =========================================================================
# POMDP infrastructure
# =========================================================================

def make_structured_pomdp(rng, n_states=1024, n_obs=20):
    """Block-structured POMDP. First half → class 1, second half → class 0.
    States within a class share similar observation profiles."""
    half = n_states // 2
    O = np.zeros((n_states, n_obs), dtype=np.float64)
    # Create smooth within-class variation
    for s in range(n_states):
        if s < half:
            # Class 1: higher weight on low-index observations
            base = np.linspace(3.0, 0.3, n_obs)
        else:
            # Class 0: higher weight on high-index observations
            base = np.linspace(0.3, 3.0, n_obs)
        # Per-state noise for within-class variation
        noise = rng.dirichlet(np.ones(n_obs)) * 0.5
        O[s] = base + noise
        O[s] /= O[s].sum()
    # Transition: mostly within block
    T = np.zeros((n_states, n_states), dtype=np.float64)
    for s in range(n_states):
        block_start = 0 if s < half else half
        block_end = half if s < half else n_states
        # Strong intra-block transitions
        intra = rng.exponential(1.0, size=half)
        T[s, block_start:block_end] = intra
        # Weak cross-block transitions
        cross_start = half if s < half else 0
        cross_end = n_states if s < half else half
        cross = rng.exponential(0.05, size=half)
        T[s, cross_start:cross_end] = cross
        T[s] /= T[s].sum()
    return O, T


def state_label(s, n_states):
    return 1 if s < n_states // 2 else 0


def roll_out_batch(O, T, rng, n_rollouts, seq_len):
    """Vectorised rollout: returns (final_states, obs_sequences)."""
    n_s, n_o = O.shape
    states = rng.integers(0, n_s, size=n_rollouts)
    obs_all = np.zeros((n_rollouts, seq_len), dtype=np.int64)
    for t in range(seq_len):
        # Vectorised emission: sample obs from O[states[i]] for each i
        cum_obs = np.cumsum(O[states], axis=1)  # (N, n_obs)
        u_obs = rng.random(n_rollouts)
        obs_all[:, t] = np.clip((cum_obs < u_obs[:, None]).sum(axis=1), 0, n_o - 1)
        # Vectorised transition (not on last step)
        if t < seq_len - 1:
            cum_trans = np.cumsum(T[states], axis=1)  # (N, n_states)
            u_trans = rng.random(n_rollouts)
            states = np.clip((cum_trans < u_trans[:, None]).sum(axis=1), 0, n_s - 1)
    return states, obs_all


def posterior_prob_batch(obs_batch, O, T):
    """Vectorised posterior computation.
    obs_batch: (N, seq_len) integer array.
    Returns: (N,) array of P(class=1 | obs).
    """
    N, seq_len = obs_batch.shape
    n_s = O.shape[0]
    half = n_s // 2
    # Belief: (N, n_states)
    belief = np.ones((N, n_s), dtype=np.float64) / n_s
    # First observation update
    belief *= O[:, obs_batch[:, 0]].T  # (N, n_s)
    for t in range(1, seq_len):
        # Transition: belief = belief @ T
        belief = belief @ T
        # Observation update
        belief *= O[:, obs_batch[:, t]].T
    # Normalise
    totals = belief.sum(axis=1, keepdims=True)
    totals = np.where(totals == 0, 1.0, totals)
    belief /= totals
    # P(class=1) = sum of belief over first half of states
    return belief[:, :half].sum(axis=1)


# =========================================================================
# Bridge computation
# =========================================================================

@dataclass(frozen=True)
class Pt:
    cell: str; label: int; mu: float; rho: float


def cell_sums(pts):
    out = {}
    for p in pts:
        if p.cell not in out:
            out[p.cell] = [0.0, 0.0]
        out[p.cell][p.label] += p.mu * p.rho
    return {k: (v[1], v[0]) for k, v in out.items()}


def k_rho(pts):
    return sum(min(A, B) for A, B in cell_sums(pts).values())


def optimal_q(pts):
    return {c: (1.0 if A > B else (0.0 if A < B else 0.5))
            for c, (A, B) in cell_sums(pts).items()}


def regret_fn(pts, q):
    return sum(p.mu * p.rho * ((1 - q[p.cell]) if p.label == 1 else q[p.cell])
               for p in pts)


def exec_cost(pts, q):
    sums = cell_sums(pts)
    qs = optimal_q(pts)
    return sum(abs(A - B) * abs(q[c] - qs[c]) for c, (A, B) in sums.items())


def raw_impurity(pts):
    cells = defaultdict(lambda: [0, 0])
    for p in pts:
        cells[p.cell][p.label] += 1
    total = len(pts)
    return sum(min(v[0], v[1]) for v in cells.values()) / total if total > 0 else 0.0


def accuracy_from_pts(pts):
    cells = defaultdict(lambda: [0, 0])
    for p in pts:
        cells[p.cell][p.label] += 1
    total = len(pts)
    return sum(max(v[0], v[1]) for v in cells.values()) / total if total > 0 else 0.0


def build_pts_from_arrays(cells_list, posteriors, n_states):
    """Build Pt list from cell strings, posterior array."""
    n = len(cells_list)
    mu = 1.0 / n
    pts = []
    for i in range(n):
        p = posteriors[i]
        m = abs(p - 0.5)
        y = 1 if p >= 0.5 else 0
        rho = (2 * m) / (0.5 + m) if (0.5 + m) > 0 else 0.0
        pts.append(Pt(cell=cells_list[i], label=y, mu=mu, rho=rho))
    return pts


# =========================================================================
# PyTorch models (only available when torch is installed)
# =========================================================================

if HAS_TORCH:
    class TorchMLP(nn.Module):
        def __init__(self, d_in, d_hid, d_out=2):
            super().__init__()
            self.fc1 = nn.Linear(d_in, d_hid)
            self.fc2 = nn.Linear(d_hid, d_out)

        def forward(self, x):
            h = torch.tanh(self.fc1(x))
            return self.fc2(h), h

        def get_cells(self, x, n_bins=5):
            with torch.no_grad():
                _, h = self.forward(x)
                half_bins = n_bins / 2.0
                h_disc = torch.round(h * half_bins) / half_bins
                return [str(tuple(row.tolist())) for row in h_disc]


    class TorchGRUClassifier(nn.Module):
        def __init__(self, d_in, d_hid, d_out=2):
            super().__init__()
            self.gru = nn.GRU(d_in, d_hid, batch_first=True)
            self.fc = nn.Linear(d_hid, d_out)
            self.d_hid = d_hid

        def forward(self, x):
            _, h_n = self.gru(x)
            h = h_n.squeeze(0)
            return self.fc(h), h

        def get_cells(self, x, n_bins=5):
            with torch.no_grad():
                _, h = self.forward(x)
                half_bins = n_bins / 2.0
                h_disc = torch.round(h * half_bins) / half_bins
                return [str(tuple(row.tolist())) for row in h_disc]


    def train_model(model, X_train, Y_train, n_epochs, lr, batch_size):
        """Train a torch model with cross-entropy loss."""
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        N = len(X_train)
        for epoch in range(n_epochs):
            perm = torch.randperm(N)[:batch_size]
            X_batch = X_train[perm]
            Y_batch = Y_train[perm]
            logits, _ = model(X_batch)
            loss = criterion(logits, Y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    def encode_flat(obs_batch, n_obs, n_states):
        """One-hot encode flattened observation sequences. Returns torch tensor."""
        N, seq_len = obs_batch.shape
        X = np.zeros((N, seq_len * n_obs), dtype=np.float32)
        rows = np.repeat(np.arange(N), seq_len)
        cols = np.tile(np.arange(seq_len), N) * n_obs + obs_batch.ravel()
        X[rows, cols] = 1.0
        return torch.from_numpy(X)


    def encode_seq(obs_batch, n_obs):
        """One-hot encode observation sequences as (N, T, n_obs). Returns torch tensor."""
        N, seq_len = obs_batch.shape
        X = np.zeros((N, seq_len, n_obs), dtype=np.float32)
        idx_i = np.repeat(np.arange(N), seq_len)
        idx_t = np.tile(np.arange(seq_len), N)
        X[idx_i, idx_t, obs_batch.ravel()] = 1.0
        return torch.from_numpy(X)


# =========================================================================
# Experiment 1: Two-sided bridge verification (32 states, unchanged)
# =========================================================================

def make_boolean_env(n_bits=5):
    n_states = 1 << n_bits
    env = IndexEnvironment(n_states=n_states)
    programs, names = [], []
    for i in range(n_bits):
        mask_1 = 0
        for s in range(n_states):
            if (s >> i) & 1:
                mask_1 |= (1 << s)
        programs.append(Program(env, mask_1))
        names.append(f"b{i}_1")
        mask_0 = ((1 << n_states) - 1) ^ mask_1
        programs.append(Program(env, mask_0))
        names.append(f"b{i}_0")
    vocab = Vocabulary(env, programs, names)
    return env, vocab, n_bits


def random_structured_label(rng, n_bits):
    n_states = 1 << n_bits
    n_relevant = rng.randint(2, min(3, n_bits))
    relevant = sorted(rng.sample(range(n_bits), n_relevant))
    n_patterns = 1 << n_relevant
    pattern_labels = [rng.randint(0, 1) for _ in range(n_patterns)]
    labels = []
    for s in range(n_states):
        pattern = 0
        for j, b in enumerate(relevant):
            if (s >> b) & 1:
                pattern |= (1 << j)
        labels.append(pattern_labels[pattern])
    return labels, relevant


def side_a_cells_and_k_rho(n_bits, observed_bits, labels):
    n_states = 1 << n_bits; mu = 1.0 / n_states
    cells_a = defaultdict(list)
    for s in range(n_states):
        key = tuple((s >> b) & 1 for b in observed_bits) if observed_bits else ("all",)
        cells_a[key].append(s)
    k_rho_a = 0.0; cell_details_a = {}
    for key, members in cells_a.items():
        n1 = sum(labels[s] for s in members)
        p = n1 / len(members); m = abs(p - 0.5)
        rho = (2 * m) / (0.5 + m) if (0.5 + m) > 0 else 0.0
        A_C = sum(mu * rho for s in members if labels[s] == 1)
        B_C = sum(mu * rho for s in members if labels[s] == 0)
        k_rho_a += min(A_C, B_C)
        cell_details_a[key] = {"members": frozenset(members), "A": A_C, "B": B_C}
    return cells_a, k_rho_a, cell_details_a


def side_b_cells_and_k_rho(env, vocab, n_bits, observed_bits, labels):
    n_states = 1 << n_bits; mu = 1.0 / n_states
    if not observed_bits:
        members = frozenset(range(n_states))
        n1 = sum(labels[s] for s in members)
        p = n1 / len(members); m = abs(p - 0.5)
        rho = (2 * m) / (0.5 + m) if (0.5 + m) > 0 else 0.0
        A = sum(mu * rho for s in members if labels[s] == 1)
        B = sum(mu * rho for s in members if labels[s] == 0)
        return {("all",): list(members)}, min(A, B), \
               {("all",): {"members": members, "A": A, "B": B}}
    n_patterns = 1 << len(observed_bits)
    cells_b = {}; cell_details_b = {}; k_rho_b = 0.0
    for pat in range(n_patterns):
        obs_mask = 0
        for j, bit_idx in enumerate(observed_bits):
            bit_val = (pat >> j) & 1
            prog_idx = bit_idx * 2 + (0 if bit_val == 1 else 1)
            obs_mask |= (1 << prog_idx)
        if not vocab.is_in_language_mask(obs_mask):
            continue
        obs_stmt = Statement(vocab, obs_mask)
        truth_bitset = obs_stmt.truth_set().bitset
        members = frozenset(s for s in range(n_states) if (truth_bitset >> s) & 1)
        if not members:
            continue
        n1 = sum(labels[s] for s in members)
        p = n1 / len(members); m_val = abs(p - 0.5)
        rho = (2 * m_val) / (0.5 + m_val) if (0.5 + m_val) > 0 else 0.0
        A_C = sum(mu * rho for s in members if labels[s] == 1)
        B_C = sum(mu * rho for s in members if labels[s] == 0)
        key = tuple((pat >> j) & 1 for j in range(len(observed_bits)))
        cells_b[key] = list(members)
        cell_details_b[key] = {"members": members, "A": A_C, "B": B_C}
        k_rho_b += min(A_C, B_C)
    return cells_b, k_rho_b, cell_details_b


def run_experiment_1(results):
    print("=" * 70)
    print("EXPERIMENT 1: Two-sided bridge verification (5 bits, 32 states)")
    print("=" * 70)
    t0 = time.perf_counter()
    n_bits = 5; n_trials = 100; rng = random.Random(42)
    env, vocab, _ = make_boolean_env(n_bits)
    print(f"Environment: {1 << n_bits} states, vocab size {vocab.size}, "
          f"|L_v| = {len(vocab.induced_language_masks())}")

    all_k_rho = defaultdict(list)
    all_baselines = defaultdict(lambda: defaultdict(list))
    bridge_matches = 0; cell_matches = 0; total_checks = 0

    for trial in range(n_trials):
        labels, _ = random_structured_label(rng, n_bits)
        for n_obs in range(6):
            if n_obs == 0: obs_bits = []
            elif n_obs >= n_bits: obs_bits = list(range(n_bits))
            else: obs_bits = sorted(rng.sample(range(n_bits), n_obs))
            _, k_rho_a, details_a = side_a_cells_and_k_rho(n_bits, obs_bits, labels)
            _, k_rho_b, details_b = side_b_cells_and_k_rho(env, vocab, n_bits, obs_bits, labels)
            cells_agree = True
            if not obs_bits:
                cells_agree = details_a[("all",)]["members"] == details_b[("all",)]["members"]
            else:
                for key_a, det_a in details_a.items():
                    if key_a not in details_b or det_a["members"] != details_b[key_a]["members"]:
                        cells_agree = False; break
                if cells_agree: cells_agree = len(details_a) == len(details_b)
            cell_matches += int(cells_agree)
            bridge_matches += int(abs(k_rho_a - k_rho_b) < 1e-10)
            total_checks += 1
            all_k_rho[n_obs].append(k_rho_a)
            n_states = 1 << n_bits
            cells_bl = defaultdict(list)
            for s in range(n_states):
                key = tuple((s >> b) & 1 for b in obs_bits) if obs_bits else ("all",)
                cells_bl[key].append(s)
            correct = sum(max(sum(labels[s] for s in m), len(m)-sum(labels[s] for s in m)) for m in cells_bl.values())
            ri = sum(min(sum(labels[s] for s in m), len(m)-sum(labels[s] for s in m)) for m in cells_bl.values())
            all_baselines[n_obs]["accuracy"].append(correct / n_states)
            all_baselines[n_obs]["raw_impurity"].append(ri / n_states)
            all_baselines[n_obs]["n_cells"].append(len(cells_bl))

    elapsed = time.perf_counter() - t0
    print(f"Cell agreement: {cell_matches}/{total_checks} ({'PASSED' if cell_matches == total_checks else 'FAILED'})")
    print(f"K_rho agreement: {bridge_matches}/{total_checks} ({'PASSED' if bridge_matches == total_checks else 'FAILED'})")

    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
    exp1 = {"cell_agreement": f"{cell_matches}/{total_checks}",
            "krho_agreement": f"{bridge_matches}/{total_checks}",
            "time_seconds": round(elapsed, 1), "rows": []}
    for n_obs in range(6):
        kr = np.array(all_k_rho[n_obs])
        ri = np.array(all_baselines[n_obs]["raw_impurity"])
        acc = np.array(all_baselines[n_obs]["accuracy"])
        nc = np.array(all_baselines[n_obs]["n_cells"])
        row = {"bits": n_obs, "k_rho_mean": round(float(kr.mean()), 4),
               "k_rho_se": round(se(kr), 4),
               "raw_impurity_mean": round(float(ri.mean()), 4),
               "raw_impurity_se": round(se(ri), 4),
               "accuracy_mean": round(float(acc.mean()), 3),
               "accuracy_se": round(se(acc), 3),
               "n_cells": round(float(nc.mean()), 0)}
        exp1["rows"].append(row)
        print(f"  {n_obs} bits: K_rho={kr.mean():.3f}±{se(kr):.3f}, "
              f"impurity={ri.mean():.3f}, acc={acc.mean():.2f}, cells={nc.mean():.0f}")

    # Discrimination
    n_discrim = 0; n_compared = 200
    for trial in range(n_compared):
        labels, _ = random_structured_label(rng, n_bits)
        obs_2 = sorted(rng.sample(range(n_bits), 2))
        obs_3 = sorted(rng.sample(range(n_bits), 3))
        _, k2, _ = side_a_cells_and_k_rho(n_bits, obs_2, labels)
        _, k3, _ = side_a_cells_and_k_rho(n_bits, obs_3, labels)
        cells_bl2 = defaultdict(list); cells_bl3 = defaultdict(list)
        n_s = 1 << n_bits
        for s in range(n_s):
            cells_bl2[tuple((s >> b) & 1 for b in obs_2)].append(s)
            cells_bl3[tuple((s >> b) & 1 for b in obs_3)].append(s)
        acc2 = sum(max(sum(labels[s] for s in m), len(m)-sum(labels[s] for s in m)) for m in cells_bl2.values()) / n_s
        acc3 = sum(max(sum(labels[s] for s in m), len(m)-sum(labels[s] for s in m)) for m in cells_bl3.values()) / n_s
        k_max = max(k2, k3)
        if abs(acc2 - acc3) < 0.05 and k_max > 0.01 and abs(k2-k3)/k_max > 0.3:
            n_discrim += 1
    exp1["discrimination"] = f"{n_discrim}/{n_compared} ({100*n_discrim/n_compared:.1f}%)"
    print(f"Discrimination: {exp1['discrimination']}")
    results["experiment_1"] = exp1


# =========================================================================
# Experiment 2: Discretised encoder at varying granularity
# =========================================================================

def run_experiment_2(results, n_states=1024, n_obs=20, seq_len=15,
                     n_trials=20, n_eval=3000, n_train=8000, d_hid=16):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Discretised MLP encoder ({n_states} states, {n_obs} obs, L={seq_len})")
    print("=" * 70)
    t0 = time.perf_counter()

    granularities = [2, 3, 4, 5, 6, 8, 10, 15]
    res = {g: {"k_rho": [], "raw_impurity": [], "accuracy": [], "n_cells": []}
           for g in granularities}
    rng = np.random.default_rng(42)

    for trial in range(n_trials):
        O, T = make_structured_pomdp(rng, n_states, n_obs)
        train_states, train_obs = roll_out_batch(O, T, rng, n_train, seq_len)
        final_states, eval_obs = roll_out_batch(O, T, rng, n_eval, seq_len)
        Y_train_np = np.array([state_label(s, n_states) for s in train_states])

        X_train = encode_flat(train_obs, n_obs, n_states)
        Y_train = torch.from_numpy(Y_train_np).long()

        # Train MLP to convergence
        model = TorchMLP(seq_len * n_obs, d_hid)
        train_model(model, X_train, Y_train, n_epochs=400, lr=1e-3, batch_size=256)

        # Posteriors (vectorised)
        posteriors = posterior_prob_batch(eval_obs, O, T)
        X_eval = encode_flat(eval_obs, n_obs, n_states)

        for g in granularities:
            cells = model.get_cells(X_eval, n_bins=g)
            pts = build_pts_from_arrays(cells, posteriors, n_states)
            res[g]["k_rho"].append(k_rho(pts))
            res[g]["raw_impurity"].append(raw_impurity(pts))
            res[g]["accuracy"].append(accuracy_from_pts(pts))
            res[g]["n_cells"].append(len(set(cells)))

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials} done ({time.perf_counter()-t0:.0f}s)")

    elapsed = time.perf_counter() - t0
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    exp2 = {"n_states": n_states, "n_obs": n_obs, "seq_len": seq_len,
            "n_trials": n_trials, "time_seconds": round(elapsed, 1), "rows": []}

    print(f"\n{'Bins':>5} {'K_rho':>14} {'Raw impur':>14} {'Accuracy':>14} {'Cells':>10}")
    print("-" * 65)
    for g in granularities:
        kr = np.array(res[g]["k_rho"]); ri = np.array(res[g]["raw_impurity"])
        acc = np.array(res[g]["accuracy"]); nc = np.array(res[g]["n_cells"])
        row = {"bins": g,
               "k_rho_mean": round(float(kr.mean()), 5), "k_rho_se": round(se(kr), 5),
               "raw_impurity_mean": round(float(ri.mean()), 4), "raw_impurity_se": round(se(ri), 4),
               "accuracy_mean": round(float(acc.mean()), 4), "accuracy_se": round(se(acc), 4),
               "n_cells_mean": round(float(nc.mean()), 1), "n_cells_se": round(se(nc), 1)}
        exp2["rows"].append(row)
        print(f"{g:>5} {kr.mean():>7.5f}±{se(kr):.5f} {ri.mean():>7.4f}±{se(ri):.4f} "
              f"{acc.mean():>7.4f}±{se(acc):.4f} {nc.mean():>7.1f}±{se(nc):.1f}")

    results["experiment_2"] = exp2


# =========================================================================
# Experiment 3: Architecture comparison
# =========================================================================

def run_experiment_3(results, n_states=1024, n_obs=20, seq_len=15,
                     n_trials=20, n_eval=3000, n_train=8000, d_hid=16):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 3: Architecture comparison ({n_states} states, {n_obs} obs, L={seq_len})")
    print("=" * 70)
    t0 = time.perf_counter()

    arch_results = {name: {"k_rho": [], "accuracy": [], "raw_impurity": [], "n_cells": []}
                    for name in ["GRU", "MLP", "Last-obs", "Random"]}
    rng = np.random.default_rng(123)

    for trial in range(n_trials):
        O, T = make_structured_pomdp(rng, n_states, n_obs)
        train_states, train_obs = roll_out_batch(O, T, rng, n_train, seq_len)
        eval_states, eval_obs = roll_out_batch(O, T, rng, n_eval, seq_len)
        Y_train_np = np.array([state_label(s, n_states) for s in train_states])
        posteriors = posterior_prob_batch(eval_obs, O, T)

        X_flat_train = encode_flat(train_obs, n_obs, n_states)
        X_seq_train = encode_seq(train_obs, n_obs)
        X_flat_eval = encode_flat(eval_obs, n_obs, n_states)
        X_seq_eval = encode_seq(eval_obs, n_obs)
        Y_train = torch.from_numpy(Y_train_np).long()

        # --- GRU ---
        gru = TorchGRUClassifier(n_obs, d_hid)
        train_model(gru, X_seq_train, Y_train, n_epochs=300, lr=1e-3, batch_size=256)
        gru_cells = gru.get_cells(X_seq_eval, n_bins=5)

        # --- MLP ---
        mlp = TorchMLP(seq_len * n_obs, d_hid)
        train_model(mlp, X_flat_train, Y_train, n_epochs=400, lr=1e-3, batch_size=256)
        mlp_cells = mlp.get_cells(X_flat_eval, n_bins=5)

        # --- Last observation ---
        last_cells = [f"last_{eval_obs[i, -1]}" for i in range(n_eval)]

        # --- Random projection ---
        rng_rp = np.random.default_rng(trial + 9999)
        W_rand = rng_rp.standard_normal((seq_len * n_obs, d_hid)).astype(np.float32)
        H_rand = np.tanh(X_flat_eval.numpy() @ W_rand)
        H_disc = np.round(H_rand * 2.5) / 2.5
        rand_cells = [str(tuple(row)) for row in H_disc]

        for name, cells in [("GRU", gru_cells), ("MLP", mlp_cells),
                            ("Last-obs", last_cells), ("Random", rand_cells)]:
            pts = build_pts_from_arrays(cells, posteriors, n_states)
            arch_results[name]["k_rho"].append(k_rho(pts))
            arch_results[name]["accuracy"].append(accuracy_from_pts(pts))
            arch_results[name]["raw_impurity"].append(raw_impurity(pts))
            arch_results[name]["n_cells"].append(len(set(cells)))

        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials} done ({time.perf_counter()-t0:.0f}s)")

    elapsed = time.perf_counter() - t0
    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    exp3 = {"n_states": n_states, "time_seconds": round(elapsed, 1), "rows": []}
    n_rank_disagree = 0
    for trial in range(n_trials):
        kr_trial = {n: arch_results[n]["k_rho"][trial] for n in arch_results}
        acc_trial = {n: arch_results[n]["accuracy"][trial] for n in arch_results}
        if sorted(kr_trial, key=lambda n: kr_trial[n])[0] != sorted(acc_trial, key=lambda n: -acc_trial[n])[0]:
            n_rank_disagree += 1

    print(f"\n{'Arch':>10} {'K_rho':>14} {'Accuracy':>14} {'Raw impur':>14} {'Cells':>10}")
    print("-" * 70)
    for name in ["GRU", "MLP", "Last-obs", "Random"]:
        d = arch_results[name]
        kr = np.array(d["k_rho"]); acc = np.array(d["accuracy"])
        ri = np.array(d["raw_impurity"]); nc = np.array(d["n_cells"])
        row = {"arch": name,
               "k_rho_mean": round(float(kr.mean()), 5), "k_rho_se": round(se(kr), 5),
               "accuracy_mean": round(float(acc.mean()), 4), "accuracy_se": round(se(acc), 4),
               "raw_impurity_mean": round(float(ri.mean()), 4), "raw_impurity_se": round(se(ri), 4),
               "n_cells_mean": round(float(nc.mean()), 1)}
        exp3["rows"].append(row)
        print(f"{name:>10} {kr.mean():>7.5f}±{se(kr):.5f} {acc.mean():>7.4f}±{se(acc):.4f} "
              f"{ri.mean():>7.4f}±{se(ri):.4f} {nc.mean():>7.1f}")

    exp3["rank_disagree"] = f"{n_rank_disagree}/{n_trials} ({100*n_rank_disagree/n_trials:.0f}%)"
    print(f"\nRank disagreement (best-K_rho != best-acc): {exp3['rank_disagree']}")
    results["experiment_3"] = exp3


# =========================================================================
# Experiment 4: Learned representation during training
# =========================================================================

def run_experiment_4(results, n_states=1024, n_obs=20, seq_len=15,
                     n_trials=20, n_eval=3000, n_train=8000, d_hid=16,
                     n_epochs=500, eval_every=25):
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 4: Training dynamics ({n_states} states, {n_obs} obs, L={seq_len})")
    print("=" * 70)
    t0 = time.perf_counter()

    rng = np.random.default_rng(77)
    all_hist = []

    for trial in range(n_trials):
        O, T = make_structured_pomdp(rng, n_states, n_obs)
        train_states, train_obs = roll_out_batch(O, T, rng, n_train, seq_len)
        eval_states, eval_obs = roll_out_batch(O, T, rng, n_eval, seq_len)
        Y_train_np = np.array([state_label(s, n_states) for s in train_states])
        Y_eval_np = np.array([state_label(s, n_states) for s in eval_states])
        posteriors = posterior_prob_batch(eval_obs, O, T)

        X_train = encode_flat(train_obs, n_obs, n_states)
        X_eval = encode_flat(eval_obs, n_obs, n_states)
        Y_train = torch.from_numpy(Y_train_np).long()

        model = TorchMLP(seq_len * n_obs, d_hid)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        N = len(X_train)
        history = []

        for epoch in range(n_epochs + 1):
            if epoch % eval_every == 0:
                cells = model.get_cells(X_eval, n_bins=5)
                with torch.no_grad():
                    logits, _ = model(X_eval)
                    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
                pts = build_pts_from_arrays(cells, posteriors, n_states)
                kr = k_rho(pts)
                # Per-cell average q
                cell_q_acc = defaultdict(lambda: [0.0, 0])
                for i in range(n_eval):
                    cell_q_acc[cells[i]][0] += probs[i]
                    cell_q_acc[cells[i]][1] += 1
                q_model = {c: s / cnt for c, (s, cnt) in cell_q_acc.items()}
                mr = regret_fn(pts, q_model)
                ec = exec_cost(pts, q_model)
                preds = (probs >= 0.5).astype(int)
                acc = float((preds == Y_eval_np).mean())
                total_r = kr + ec
                ratio = kr / total_r if total_r > 1e-12 else 0.0
                history.append({"epoch": epoch, "k_rho": kr, "exec_cost": ec,
                                "total_regret": mr, "accuracy": acc,
                                "ratio": ratio, "n_cells": len(set(cells))})

            if epoch < n_epochs:
                perm = torch.randperm(N)[:256]
                logits, _ = model(X_train[perm])
                loss = criterion(logits, Y_train[perm])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        all_hist.append(history)
        if (trial + 1) % 5 == 0:
            print(f"  Trial {trial+1}/{n_trials} done ({time.perf_counter()-t0:.0f}s)")

    # Identity check
    ok = all(abs(h["total_regret"] - (h["k_rho"] + h["exec_cost"])) < 1e-6
             for hist in all_hist for h in hist)
    elapsed = time.perf_counter() - t0
    print(f"\nDecomposition identity check: {'PASSED' if ok else 'FAILED'}")

    se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
    epochs = [h["epoch"] for h in all_hist[0]]

    exp4 = {"n_states": n_states, "identity_check": "PASSED" if ok else "FAILED",
            "time_seconds": round(elapsed, 1), "rows": []}

    print(f"\n{'Epoch':>6} {'K_rho':>10} {'Exec cost':>10} {'Ratio':>8} {'Accuracy':>10} {'Cells':>8}")
    for ei, ep in enumerate(epochs):
        krs = [h[ei]["k_rho"] for h in all_hist]
        ecs = [h[ei]["exec_cost"] for h in all_hist]
        rats = [h[ei]["ratio"] for h in all_hist]
        accs = [h[ei]["accuracy"] for h in all_hist]
        cells = [h[ei]["n_cells"] for h in all_hist]
        row = {"epoch": ep,
               "k_rho_mean": round(float(np.mean(krs)), 5), "k_rho_se": round(se(krs), 5),
               "exec_cost_mean": round(float(np.mean(ecs)), 5), "exec_cost_se": round(se(ecs), 5),
               "ratio_mean": round(float(np.mean(rats)), 3),
               "accuracy_mean": round(float(np.mean(accs)), 4), "accuracy_se": round(se(accs), 4),
               "n_cells_mean": round(float(np.mean(cells)), 1)}
        exp4["rows"].append(row)
        print(f"{ep:>6} {np.mean(krs):>10.5f} {np.mean(ecs):>10.5f} "
              f"{np.mean(rats):>8.3f} {np.mean(accs):>10.4f} {np.mean(cells):>8.1f}")

    results["experiment_4"] = exp4


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    if not HAS_TORCH:
        print("ERROR: PyTorch is required. Install with: pip install torch")
        sys.exit(1)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Using {torch.get_num_threads()} threads")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
    print()

    results = {"version": "v28", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "torch_version": torch.__version__,
               "n_threads": torch.get_num_threads()}

    overall_t0 = time.perf_counter()

    run_experiment_1(results)
    run_experiment_2(results)
    run_experiment_3(results)
    run_experiment_4(results)

    total_time = time.perf_counter() - overall_t0
    results["total_time_seconds"] = round(total_time, 1)
    results["total_time_minutes"] = round(total_time / 60, 1)

    out_path = os.path.join(os.path.dirname(__file__), "results_v28.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"ALL EXPERIMENTS COMPLETE in {total_time/60:.1f} minutes.")
    print(f"Results written to: {out_path}")
    print(f"{'=' * 70}")
