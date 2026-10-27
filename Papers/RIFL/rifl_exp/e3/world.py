"""One E3 world: memory agents on the noisy T-Maze with a visible regime.

Source: hallways 5-10, risky regime in 10 percent of episodes. Target: hallways
10-20, risky regime in half of the episodes. Agents differ in how many readings
they take at the start (acquisition), in what their memory keeps (aliasing) and
in how they act on it (execution).
"""
from __future__ import annotations

import io
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common import fsel
from e2.agent import Spec
from e3.nmaze import GOOD, NORTH, OBS_DIM, PENALTY, SOUTH, exact_post
from e3.ppo import Cfg, feed, load, rollout, train

ARCHS = [("gru", 8), ("gru", 32), ("gru", 128), ("lstm", 8), ("lstm", 32), ("lstm", 128), ("trxl", 8), ("trxl", 32)]
MIXES = (0.1, 0.5)                 # risky-regime share while training: natural or raised
SNAPS_FRAC = (0.5, 1.0)
SRC_LEN, SRC_SHARE = (5, 10), 0.1
TGT_LEN, TGT_SHARE = (10, 20), 0.5
N_SRC, N_PANEL, N_FRESH, N_COMMON = 256, 384, 600, 240
STAKE = np.asarray(GOOD) + np.asarray(PENALTY)     # return lost by the wrong turn, per regime


def specs_for(seed: int) -> list[tuple[Spec, float]]:
    out, i = [], 0
    for arch, size in ARCHS:
        for mix in MIXES:
            out.append((Spec(arch, size, int(seed + 7919 * i + 13)), mix))
            i += 1
    return out


def lengths(rng, n, rng_len):
    return rng.integers(rng_len[0], rng_len[1] + 1, size=n)


def cache_for(out: Path, wid: int) -> Path:
    return Path(os.environ.get("RIWF_CACHE", str(Path(out) / "agents"))) / f"w{wid:03d}"


def n_agents() -> int:
    return len(ARCHS) * len(MIXES)


def train_one(seed: int, cfg: Cfg, cache: Path, i: int) -> None:
    sp, mix = specs_for(seed)[i]
    f = cache / f"{sp.name}.pt"
    if f.exists():
        return
    cache.mkdir(parents=True, exist_ok=True)
    snaps = [int(cfg.updates * fr) for fr in SNAPS_FRAC]
    st, log = train(sp, cfg, lambda r, n: lengths(r, n, SRC_LEN), sp.seed + 1, snaps, regime_share=mix)
    buf = io.BytesIO()
    torch.save({"state": st, "log": log}, buf)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(buf.getvalue())
    tmp.rename(f)


def train_agents(seed: int, cfg: Cfg, cache: Path, deadline=None) -> bool:
    for i, (sp, mix) in enumerate(specs_for(seed)):
        if (cache / f"{sp.name}.pt").exists():
            continue
        if deadline is not None and time.time() > deadline:
            return False
        train_one(seed, cfg, cache, i)
    return True


class SetSum(nn.Module):
    """Teacher: per-step embeddings summed over the history (Zaheer et al. 2017)."""

    def __init__(self, obs_dim: int, h: int = 32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(obs_dim, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
        self.rho = nn.Sequential(nn.Linear(h + 1, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, obs, valid):
        e = (self.phi(obs) * valid.unsqueeze(-1)).sum(1)
        return self.rho(torch.cat([e, obs[:, :1, 4]], -1))


def pad(hist, dim):
    T = max(len(h) for h in hist)
    o = np.zeros((len(hist), T, dim), dtype=np.float32)
    v = np.zeros((len(hist), T), dtype=bool)
    for i, h in enumerate(hist):
        o[i, :len(h)] = h
        v[i, :len(h)] = True
    return o, v


def fit_teacher(obs, valid, y, seed, epochs=40):
    torch.manual_seed(int(seed))
    net = SetSum(obs.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
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


def teacher_post(net, obs, valid):
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(obs), torch.from_numpy(valid).float())[:, 0]).numpy().astype(np.float64)


def histories(ro):
    out = []
    last = ro["valid"].sum(1) - 1
    for i in range(len(ro["arrive"])):
        e = ro["arrive"][i] if ro["arrive"][i] >= 0 else last[i]
        out.append(ro["obs"][i, :e + 1])
    return out


def return_terms(p, regime):
    """Best answer and the return lost by the other answer, per episode."""
    y = (p >= 0.5).astype(np.int64)
    w = STAKE[regime] * np.abs(2 * p - 1)
    return y, w


def source_stage(cohort: str, kind: str, q_common, wid: int, seed: int, cfg: Cfg, out: Path,
                 deadline=None, score_by=None):
    t0 = time.time()
    cache = cache_for(out, wid)
    if not train_agents(seed, cfg, cache, deadline):
        return None
    if score_by is not None and time.time() > score_by:
        return None
    snaps = [int(cfg.updates * f) for f in SNAPS_FRAC]
    sd = np.random.default_rng(seed + 1).integers(2**31 - 1, size=16)
    cands = []
    for sp, mix in specs_for(seed):
        rec = torch.load(cache / f"{sp.name}.pt", weights_only=False)
        for u in snaps:
            cands.append({"spec": sp, "mix": mix, "update": u, "state": rec["state"][u], "train": rec["log"]})

    # labelled source episodes: validation return and teacher data
    rng = np.random.default_rng(int(sd[0]))
    L_s = lengths(rng, N_SRC, SRC_LEN)
    g_s = rng.integers(0, 2, size=N_SRC)
    r_s = (rng.random(N_SRC) < SRC_SHARE).astype(np.int64)
    src, th, ty = [], [], []
    for ci, c in enumerate(cands):
        ro = rollout(load(c["spec"], c["state"]), L_s, g_s, r_s, int(sd[1]) + ci)
        hist = histories(ro)
        th.extend(hist[:96])
        ty.extend(ro["goal"][:96])
        src.append({"ro": ro, "hist": hist})
    teacher = fit_teacher(*pad(th, OBS_DIM), np.asarray(ty), int(sd[2]))

    # unlabelled target panels, one per candidate, rolled out by the candidate itself
    rng = np.random.default_rng(int(sd[3]))
    L_p = lengths(rng, N_PANEL, TGT_LEN)
    g_p = rng.integers(0, 2, size=N_PANEL)
    r_p = (rng.random(N_PANEL) < TGT_SHARE).astype(np.int64)
    panel_seed = int(sd[4])
    rows, readouts, pool_hist = [], [], []
    for ci, c in enumerate(cands):
        net = load(c["spec"], c["state"])
        ro = rollout(net, L_p, g_p, r_p, panel_seed + ci)
        hist = histories(ro)
        p_hat = teacher_post(teacher, *pad(hist, OBS_DIM))
        reached = ro["arrive"] >= 0
        reach = float(reached.mean())
        rg = ro["regime"]
        # teacher-based policy evaluation of the native actor, in return units
        n = N_PANEL
        last = ro["valid"].sum(1) - 1
        a_last = ro["act"][np.arange(n), last]
        ended = reached & ((a_last == NORTH) | (a_last == SOUTH))
        pr = ro["probs"][np.arange(n), last]
        pn, ps = pr[:, NORTH], pr[:, SOUTH]
        good, pen = np.asarray(GOOD)[rg], np.asarray(PENALTY)[rg]
        vN = (1 - p_hat) * good - p_hat * pen
        vS = p_hat * good - (1 - p_hat) * pen
        tpe = float(np.where(ended, (pn * vN + ps * vS) / np.maximum(pn + ps, 1e-12), 0.0).mean())
        # freedom terms at the junction memory
        m = ro["feat_j"][reached]
        pr_ = p_hat[reached]
        y, w_ret = return_terms(pr_, rg[reached])
        w_suc = np.abs(2 * pr_ - 1)
        w_one = np.ones_like(pr_)
        enough = reached.sum() >= 20 and len(np.unique(y)) > 1
        Y = y[:, None]
        d_ret = fsel.deficit_linear_cv(m, Y, w_ret[:, None], seed + ci) if enough else float(STAKE.max())
        d_suc = fsel.deficit_linear_cv(m, Y, w_suc[:, None], seed + ci) if enough else 1.0
        d_one = fsel.deficit_linear_cv(m, Y, w_one[:, None], seed + ci) if enough else 1.0
        d_knn = fsel.deficit_knn_cv(m, Y, w_ret[:, None], seed + ci) if enough else float(STAKE.max())
        acq = float((STAKE[rg[reached]] * np.minimum(pr_, 1 - pr_)).sum() / n)
        best = float(good[reached].sum() / n)
        # predicted return of the refitted readout: P3 floor on own rollouts with the P4 split
        fp4 = best - acq - reach * d_ret
        # label-free scores at the first junction arrival
        pj = ro["probs"][np.arange(n), np.maximum(ro["arrive"], 0)][reached]
        s_ro = src[ci]["ro"]
        s_reached = s_ro["arrive"] >= 0
        s_pj = s_ro["probs"][np.arange(N_SRC), np.maximum(s_ro["arrive"], 0)][s_reached]
        s_ok = s_ro["success"][s_reached]
        t_atc = fsel.atc_threshold(s_pj.max(-1), s_ok) if s_reached.any() else 1.0
        s_hat = teacher_post(teacher, *pad(src[ci]["hist"], OBS_DIM))[s_reached]
        ys, ws = return_terms(s_hat, s_ro["regime"][s_reached])
        s_enough = s_reached.sum() >= 20 and len(np.unique(ys)) > 1
        row = {
            "cand": ci, "name": c["spec"].name, "arch": c["spec"].arch, "size": c["spec"].size, "mix": c["mix"],
            "update": c["update"], "train_return": c["train"]["train_return"],
            "nat_val": float(s_ro["ret"].mean()), "src_reads1": float(s_ro["reads_j"][s_reached & (s_ro["regime"] == 1)].sum(1).mean()) if (s_reached & (s_ro["regime"] == 1)).any() else 0.0,
            "reach": reach, "reads1": float(ro["reads_j"][reached & (rg == 1)].sum(1).mean()) if (reached & (rg == 1)).any() else 0.0,
            "tpe": tpe, "acq": acq, "best": best,
            "d_ret": float(d_ret), "d_suc": float(d_suc), "d_one": float(d_one), "d_knn": float(d_knn),
            "fp4": float(fp4), "fk4": float(best - acq - reach * d_knn),
            "mp": float(pj.max(-1).mean()) if len(pj) else 0.0,
            "ent": fsel.entropy(pj) if len(pj) else 10.0,
            "atc": fsel.atc_estimate(pj.max(-1), t_atc) if len(pj) else 0.0,
            "dev": (fsel.dev_risk(s_ro["feat_j"][s_reached], m, -s_ro["ret"][s_reached], seed + ci)
                    if (len(m) > 5 and s_reached.sum() > 5) else 0.0),
            "slin": (fsel.deficit_linear_cv(s_ro["feat_j"][s_reached], ys[:, None], ws[:, None], seed + ci)
                     if s_enough else float(STAKE.max())),
        }
        rows.append(row)
        readouts.append(fsel.fit_linear_readout(m, Y, w_ret[:, None]) if enough else None)
        keep = np.nonzero(reached)[0][:20]
        pool_hist.extend(hist[k] for k in keep)
        c["_panel"] = {"p_hat": p_hat, "reached": reached, "reads_j": ro["reads_j"], "regime": rg, "feat_j": ro["feat_j"]}

    rng = np.random.default_rng(int(sd[5]))
    pick = rng.choice(len(pool_hist), size=min(N_COMMON, len(pool_hist)), replace=False)
    co, cv = pad([pool_hist[k] for k in pick], OBS_DIM)
    lastc = cv.sum(1) - 1
    dec = []
    for c in cands:
        _, pr = feed(load(c["spec"], c["state"]), co, cv)
        pl = pr[np.arange(len(co)), lastc]
        dec.append((pl[:, SOUTH] > pl[:, NORTH]).astype(int))
    dec = np.stack(dec)
    maj = (dec.mean(0) >= 0.5).astype(int)
    for row, d in zip(rows, dec):
        row["dis"] = float((d != maj).mean())

    lock = {
        "cohort": cohort, "kind": "nmaze", "wid": wid, "seed": seed, "cfg": asdict(cfg),
        "archs": ARCHS, "mixes": MIXES, "panel_seed": panel_seed, "fresh_seed": int(sd[6]), "scores": rows,
        "readouts": [None if r is None else {k: np.asarray(v).tolist() for k, v in r.items()} for r in readouts],
        "target_truth_used": False, "seconds": time.time() - t0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lock{wid:03d}.json").write_text(json.dumps(lock))
    blob = {"cands": [{"spec": asdict(c["spec"]), "mix": c["mix"], "update": c["update"], "state": c["state"]} for c in cands],
            "teacher": teacher.state_dict(), "panel": [c["_panel"] for c in cands], "panel_goal": g_p}
    buf = io.BytesIO()
    torch.save(blob, buf)
    (out / f"state{wid:03d}.pt").write_bytes(buf.getvalue())
    shutil.rmtree(cache, ignore_errors=True)
    return lock


def target_stage(out: Path, wid: int):
    lock = json.loads((out / f"lock{wid:03d}.json").read_text())
    blob = torch.load(out / f"state{wid:03d}.pt", weights_only=False)
    rng = np.random.default_rng(lock["fresh_seed"])
    L = lengths(rng, N_FRESH, TGT_LEN)
    g = rng.integers(0, 2, size=N_FRESH)
    rg = (rng.random(N_FRESH) < TGT_SHARE).astype(np.int64)
    good, pen = np.asarray(GOOD)[rg], np.asarray(PENALTY)[rg]
    res, hits_n, hits_l = [], [], []
    for ci, (c, rd) in enumerate(zip(blob["cands"], lock["readouts"])):
        net = load(Spec(**c["spec"]), c["state"])
        ro = rollout(net, L, g, rg, lock["fresh_seed"] + 1 + ci, keep_obs=False)
        reached = ro["arrive"] >= 0
        if rd is not None:
            model = {k: np.asarray(v) for k, v in rd.items()}
            dec = fsel.apply_linear_readout(model, ro["feat_j"])[:, 0]
        else:
            dec = np.zeros(N_FRESH, dtype=int)
        ok = dec == g
        ret_l = np.where(reached, np.where(ok, good, -pen), 0.0)
        ret_n = ro["ret"]
        # exact posteriors at arrival from the readings the episode showed
        p_ex = exact_post(ro["reads_j"], rg)
        acq_t = float(np.where(reached, STAKE[rg] * np.minimum(p_ex, 1 - p_ex), 0.0).mean())
        best_t = float(np.where(reached, good, 0.0).mean())
        bayes_t = float(np.where(reached, good - STAKE[rg] * np.minimum(p_ex, 1 - p_ex), 0.0).mean())
        # teacher error on the panel, in return units
        pan = blob["panel"][ci]
        r = pan["reached"]
        p_pan_ex = exact_post(pan["reads_j"], pan["regime"])
        eps = float((STAKE[pan["regime"][r]] * 2 * np.abs(pan["p_hat"][r] - p_pan_ex[r])).sum() / len(r)) if r.any() else float("nan")
        # true aliasing on the panel: linear-language deficit with exact labels and weights
        if r.sum() >= 20:
            ye, we = return_terms(p_pan_ex[r], pan["regime"][r])
            al_t = float(fsel.deficit_linear_cv(pan["feat_j"][r], ye[:, None], we[:, None], 5 + ci) * r.mean()) if len(np.unique(ye)) > 1 else 0.0
        else:
            al_t = float("nan")
        row = {"cand": ci, "n_ret": float(ret_n.mean()), "l_ret": float(ret_l.mean()),
               "n_succ": float(ro["success"].mean()), "l_succ": float((reached & ok).mean()),
               "reach_f": float(reached.mean()), "acq_true": acq_t, "best_true": best_t, "bayes_true": bayes_t,
               "alias_true_panel": al_t, "eps": eps,
               "n_ret_r1": float(ret_n[rg == 1].mean()), "l_ret_r1": float(ret_l[rg == 1].mean()),
               "n_ret_r0": float(ret_n[rg == 0].mean()), "l_ret_r0": float(ret_l[rg == 0].mean())}
        res.append(row)
        hits_n.append(ret_n)
        hits_l.append(ret_l)
    out_res = {"wid": wid, "cands": res}
    (out / f"res{wid:03d}.json").write_text(json.dumps(out_res))
    np.savez_compressed(out / f"hits{wid:03d}.npz", n=np.stack(hits_n).astype(np.float32), l=np.stack(hits_l).astype(np.float32))
    return out_res
