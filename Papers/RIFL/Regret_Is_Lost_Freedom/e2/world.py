"""One E2 world: a pool of memory agents trained on T-Maze hallways 5-10 and deployed on hallways 20, 40 and 80.

Source stage: train the pool, fit the teacher on labelled source episodes, roll
every candidate out on unlabelled target hallways (its own panel), compute every
selector score and every refitted readout, and write the lock. Target stage:
fresh target episodes, native and refitted success, teacher error.
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

from common import fsel
from e2 import teach as TT
from e2.agent import Spec
from e2.ppo import Cfg, feed, load, rollout, train
from e2.tmaze import NORTH, OBS_DIM, SOUTH

ARCHS = [("gru", 8), ("gru", 32), ("gru", 128), ("lstm", 8), ("lstm", 32), ("lstm", 128), ("trxl", 8), ("trxl", 32)]
SEEDS_PER_ARCH = 2
SNAPS_FRAC = (0.5, 1.0)
SRC_LEN = (5, 10)
TGT_LEN = (20, 40, 80)
N_SRC = 256          # source episodes per candidate for validation and teacher data
N_PANEL = 128        # unlabelled target episodes per candidate and hallway length
N_FRESH = 200        # fresh target episodes per candidate and hallway length
N_COMMON = 240       # common histories for the disagreement score


def specs_for(seed: int) -> list[Spec]:
    out, i = [], 0
    for arch, size in ARCHS:
        for _ in range(SEEDS_PER_ARCH):
            out.append(Spec(arch, size, int(seed + 7919 * i + 13)))
            i += 1
    return out


def src_lengths(rng, n):
    return rng.integers(SRC_LEN[0], SRC_LEN[1] + 1, size=n)


def train_agents(seed: int, cfg: Cfg, cache: Path, deadline=None) -> bool:
    snaps = [int(cfg.updates * f) for f in SNAPS_FRAC]
    cache.mkdir(parents=True, exist_ok=True)
    for sp in specs_for(seed):
        f = cache / f"{sp.name}.pt"
        if f.exists():
            continue
        if deadline is not None and time.time() > deadline:
            return False
        st, log = train(sp, cfg, src_lengths, sp.seed + 1, snaps)
        buf = io.BytesIO()
        torch.save({"state": st, "log": log}, buf)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.rename(f)
    return True


def histories(ro) -> tuple[list[np.ndarray], np.ndarray]:
    """Observation history up to the first junction arrival, or the whole episode if the junction was never reached."""
    out, ends = [], []
    n = len(ro["arrive"])
    last = ro["valid"].sum(1) - 1
    for i in range(n):
        e = ro["arrive"][i] if ro["arrive"][i] >= 0 else last[i]
        out.append(ro["obs"][i, :e + 1])
        ends.append(e)
    return out, np.asarray(ends)


def exact_post(hist: list[np.ndarray]) -> np.ndarray:
    """Bayes posterior that the goal is south: the start cell shows bit 0 exactly when it is."""
    return np.asarray([float(h[:, 0].max() > 0.5) for h in hist])


def terminal_terms(ro):
    """For episodes that ended by north or south at the junction: the policy's north and south probabilities there."""
    n = len(ro["arrive"])
    last = ro["valid"].sum(1) - 1
    a = ro["act"][np.arange(n), last]
    ended = (ro["arrive"] >= 0) & ((a == NORTH) | (a == SOUTH))
    pr = ro["probs"][np.arange(n), last]
    return ended, pr[:, NORTH], pr[:, SOUTH]


def source_stage(cohort: str, kind: str, q_common, wid: int, seed: int, cfg: Cfg, out: Path,
                 deadline=None, score_by=None):
    t0 = time.time()
    cache = Path(os.environ.get("RIWF_CACHE", str(out / "agents"))) / f"w{wid:03d}"
    if not train_agents(seed, cfg, cache, deadline):
        return None
    if score_by is not None and time.time() > score_by:
        return None
    specs = specs_for(seed)
    snaps = [int(cfg.updates * f) for f in SNAPS_FRAC]
    sd = np.random.default_rng(seed + 1).integers(2**31 - 1, size=16)
    cands = []
    for sp in specs:
        rec = torch.load(cache / f"{sp.name}.pt", weights_only=False)
        for u in snaps:
            cands.append({"spec": sp, "update": u, "state": rec["state"][u], "train": rec["log"]})

    # source rollouts: validation success, memories and teacher data (labelled source episodes)
    rng = np.random.default_rng(int(sd[0]))
    L_src = src_lengths(rng, N_SRC)
    g_src = rng.integers(0, 2, size=N_SRC)
    src = []
    th, ty = [], []
    for ci, c in enumerate(cands):
        net = load(c["spec"], c["state"])
        ro = rollout(net, L_src, g_src, int(sd[1]) + ci)
        hist, ends = histories(ro)
        th.extend(hist[:96])
        ty.extend(ro["goal"][:96])
        src.append({"ro": ro, "hist": hist})
    tobs, tval = TT.pad(th, OBS_DIM)
    teacher = TT.fit(tobs, tval, np.asarray(ty), int(sd[2]))

    # target panels: each candidate's own unlabelled rollouts on the target hallways
    rng = np.random.default_rng(int(sd[3]))
    L_pan = np.repeat(np.asarray(TGT_LEN), N_PANEL)
    g_pan = rng.integers(0, 2, size=len(L_pan))     # nature's goals, hidden from every selector
    panel_seed = int(sd[4])
    rows, readouts, pool_hist = [], [], []
    for ci, c in enumerate(cands):
        net = load(c["spec"], c["state"])
        ro = rollout(net, L_pan, g_pan, panel_seed + ci)
        hist, ends = histories(ro)
        o, v = TT.pad(hist, OBS_DIM)
        p_hat = TT.post(teacher, o, v)
        reached = ro["arrive"] >= 0
        reach = float(reached.mean())
        # teacher-based policy evaluation of the native actor
        ended, pn, ps = terminal_terms(ro)
        est = np.where(ended, (pn * (1 - p_hat) + ps * p_hat) / np.maximum(pn + ps, 1e-12), 0.0)
        # freedom deficits of the readout language on the junction memory, and the P3 floor on own rollouts
        y, w = fsel.answer_terms(p_hat[reached], "delta")
        y, w = y[:, None], w[:, None]
        m = ro["feat_j"][reached]
        enough = reached.sum() >= 20 and len(np.unique(y)) > 1
        lin = fsel.deficit_linear_cv(m, y, w, seed + ci) if enough else 1.0
        knn = fsel.deficit_knn_cv(m, y, w, seed + ci) if enough else 1.0
        # label-free scores at the first junction arrival
        pj = ro["probs"][np.arange(len(L_pan)), np.maximum(ro["arrive"], 0)][reached]
        s_ro = src[ci]["ro"]
        s_reached = s_ro["arrive"] >= 0
        s_pj = s_ro["probs"][np.arange(N_SRC), np.maximum(s_ro["arrive"], 0)][s_reached]
        s_ok = s_ro["success"][s_reached]
        t_atc = fsel.atc_threshold(s_pj.max(-1), s_ok) if s_reached.any() else 1.0
        s_hat = TT.post(teacher, *TT.pad(src[ci]["hist"], OBS_DIM))[s_reached]
        ys, ws = fsel.answer_terms(s_hat, "delta")
        s_enough = s_reached.sum() >= 20 and len(np.unique(ys)) > 1
        row = {
            "cand": ci, "name": c["spec"].name, "arch": c["spec"].arch, "size": c["spec"].size, "update": c["update"],
            "train_success": c["train"]["train_success"],
            "nat_val": float(s_ro["success"].mean()),
            "reach": reach,
            "tpe": float(est.mean()),
            "lin": float(lin), "knn": float(knn),
            "fp3": float(1 - reach + reach * lin), "kp3": float(1 - reach + reach * knn),
            "mp": float(pj.max(-1).mean()) if len(pj) else 0.0,
            "ent": fsel.entropy(pj) if len(pj) else 10.0,
            "atc": fsel.atc_estimate(pj.max(-1), t_atc) if len(pj) else 0.0,
            "dev": (fsel.dev_risk(s_ro["feat_j"][s_reached], m, 1.0 - s_ok, seed + ci)
                    if (len(m) > 5 and s_reached.sum() > 5) else 1.0),
            "slin": (fsel.deficit_linear_cv(s_ro["feat_j"][s_reached], ys[:, None], ws[:, None], seed + ci)
                     if s_enough else 1.0),
        }
        rows.append(row)
        readouts.append(fsel.fit_linear_readout(m, y, w) if enough else None)
        keep = np.nonzero(reached)[0][:20]
        pool_hist.extend(hist[k] for k in keep)
        c["_panel"] = {"p_hat": p_hat, "hist": hist, "reached": reached}

    # disagreement with the pool on common histories (the same inputs for every candidate)
    rng = np.random.default_rng(int(sd[5]))
    pick = rng.choice(len(pool_hist), size=min(N_COMMON, len(pool_hist)), replace=False)
    co, cv = TT.pad([pool_hist[k] for k in pick], OBS_DIM)
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
        "cohort": cohort, "kind": "tmaze", "wid": wid, "seed": seed, "cfg": asdict(cfg),
        "archs": ARCHS, "seeds_per_arch": SEEDS_PER_ARCH, "tgt_len": TGT_LEN, "panel_seed": panel_seed,
        "fresh_seed": int(sd[6]), "scores": rows,
        "readouts": [None if r is None else {k: np.asarray(v).tolist() for k, v in r.items()} for r in readouts],
        "target_truth_used": False, "seconds": time.time() - t0,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lock{wid:03d}.json").write_text(json.dumps(lock))
    blob = {
        "cands": [{"spec": asdict(c["spec"]), "update": c["update"], "state": c["state"]} for c in cands],
        "teacher": teacher.state_dict(),
        "panel": [{"p_hat": c["_panel"]["p_hat"], "reached": c["_panel"]["reached"],
                   "exact": exact_post(c["_panel"]["hist"])} for c in cands],
        "panel_goal": g_pan,
    }
    buf = io.BytesIO()
    torch.save(blob, buf)
    (out / f"state{wid:03d}.pt").write_bytes(buf.getvalue())
    shutil.rmtree(cache, ignore_errors=True)
    return lock


def target_stage(out: Path, wid: int):
    lock = json.loads((out / f"lock{wid:03d}.json").read_text())
    blob = torch.load(out / f"state{wid:03d}.pt", weights_only=False)
    rng = np.random.default_rng(lock["fresh_seed"])
    L = np.repeat(np.asarray(TGT_LEN), N_FRESH)
    g = rng.integers(0, 2, size=len(L))
    res, hits_n, hits_l = [], [], []
    for ci, (c, rd) in enumerate(zip(blob["cands"], lock["readouts"])):
        net = load(Spec(**c["spec"]), c["state"])
        ro = rollout(net, L, g, lock["fresh_seed"] + 1 + ci, keep_obs=False)
        reached = ro["arrive"] >= 0
        if rd is not None:
            model = {k: np.asarray(v) for k, v in rd.items()}
            dec = fsel.apply_linear_readout(model, ro["feat_j"])[:, 0]
        else:
            dec = np.zeros(len(L), dtype=int)
        hit_l = reached & (dec == g)
        hit_n = ro["success"]
        pan = blob["panel"][ci]
        r = pan["reached"]
        eps = float((2 * np.abs(pan["p_hat"][r] - pan["exact"][r])).mean()) if r.any() else float("nan")
        row = {"cand": ci, "n_succ": float(hit_n.mean()), "l_succ": float(hit_l.mean()), "reach_f": float(reached.mean()),
               "eps": eps}
        for Lk in TGT_LEN:
            mk = L == Lk
            row[f"n_succ_{Lk}"] = float(hit_n[mk].mean())
            row[f"l_succ_{Lk}"] = float(hit_l[mk].mean())
        res.append(row)
        hits_n.append(hit_n)
        hits_l.append(hit_l)
    out_res = {"wid": wid, "cands": res}
    (out / f"res{wid:03d}.json").write_text(json.dumps(out_res))
    np.savez_compressed(out / f"hits{wid:03d}.npz", n=np.stack(hits_n).astype(np.float32), l=np.stack(hits_l).astype(np.float32))
    return out_res
