"""One E1 world: source stage (train, fit, score, choose, lock) and target stage (open, evaluate).

Chronology. The source stage sees labelled source episodes and input-only target
panels. It writes every selector score and every refitted policy into a lock.
The fresh target bank is generated only in the target stage, after the driver
has written the global opening record.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import shutil
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from common import fsel
from e1 import env as E
from e1.agent import Cfg, Spec, load, run, train
from e1 import teach as TT

ALLOC = [(5, 1), (4, 2), (3, 3), (2, 4), (1, 5)]
STATES = 6
SNAPS_FRAC = (0.5, 1.0)


# ---------------------------------------------------------------------------
# quantiser: standardise on source-fit memories, k-means per regime
# ---------------------------------------------------------------------------

def _kmeans(x, k, seed, iters=40):
    rng = np.random.default_rng(int(seed))
    n = len(x)
    cent = np.empty((k, x.shape[1]))
    cent[0] = x[int(rng.integers(n))]
    d2 = ((x - cent[0]) ** 2).sum(1)
    for j in range(1, k):
        tot = d2.sum()
        idx = int(rng.integers(n)) if tot <= 1e-15 else int(rng.choice(n, p=d2 / tot))
        cent[j] = x[idx]
        d2 = np.minimum(d2, ((x - cent[j]) ** 2).sum(1))
    for _ in range(iters):
        lab = ((x[:, None, :] - cent[None]) ** 2).sum(2).argmin(1)
        new = cent.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                new[j] = x[m].mean(0)
        if np.allclose(new, cent, atol=1e-8):
            break
        cent = new
    return cent


class Quant:
    def __init__(self, mem_fit, regime_fit, khi, klo, seed):
        self.mu = mem_fit.mean(0)
        self.sd = mem_fit.std(0)
        self.sd[self.sd < 1e-8] = 1.0
        z = (mem_fit - self.mu) / self.sd
        self.khi, self.klo = khi, klo
        self.hi = _kmeans(z[regime_fit == 1], khi, seed + 11)
        self.lo = _kmeans(z[regime_fit == 0], klo, seed + 29)

    def codes(self, mem, regime):
        z = (mem - self.mu) / self.sd
        out = np.empty(len(z), dtype=np.int64)
        m = regime == 1
        if m.any():
            out[m] = ((z[m, None, :] - self.hi[None]) ** 2).sum(2).argmin(1)
        if (~m).any():
            out[~m] = self.khi + ((z[~m, None, :] - self.lo[None]) ** 2).sum(2).argmin(1)
        return out


# ---------------------------------------------------------------------------
# world specification
# ---------------------------------------------------------------------------

MAIN_DELAYS = ("short", "mid", "long")


def specs_for(kind: str, seed: int) -> list[Spec]:
    out, i = [], 0
    if kind == "main":
        for arch in ("gru", "lstm"):
            for width in (16, 64):
                for delays in MAIN_DELAYS:
                    for mix in ("nat", "rare"):
                        out.append(Spec(arch, width, delays, mix, 0, int(seed + 7919 * i + 13)))
                        i += 1
    elif kind == "ds":
        for arch in ("gru", "lstm"):
            for width in (16, 64):
                for aux in (0, 1):
                    for mix in ("nat", "rare"):
                        out.append(Spec(arch, width, "short", mix, aux, int(seed + 7919 * i + 13)))
                        i += 1
    else:
        raise ValueError(kind)
    return out


def asks(world: E.World, kind: str):
    if kind == "main":
        return world.tests, world.tests, range(3, 8), [40, 60, 80]
    # demand shift: training and selection know one test, deployment asks three
    known = np.repeat(world.tests[:1], 3, axis=0)
    return known, world.tests, range(3, 8), range(3, 8)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def terms_for(post, tests, kind):
    p = E.answer_probs(post, tests)
    return fsel.answer_terms(p, kind)


def expected_success(ans, p):
    """Expected share of correct answers given true answer probabilities."""
    return float(np.where(ans == 1, p, 1.0 - p).mean())


def expected_nreg(ans, p):
    v = np.where(ans == 1, p, 1.0 - p)
    return float((1.0 - v / np.maximum(np.maximum(p, 1 - p), 1e-12)).mean())


def native_success(gp, p):
    return float((gp[:, :, 1] * p + gp[:, :, 0] * (1 - p)).mean())


# ---------------------------------------------------------------------------
# source stage
# ---------------------------------------------------------------------------

def train_agents(kind: str, q_common: float, seed: int, cfg: Cfg, cache: Path, deadline: float | None = None) -> bool:
    """Train every candidate agent of a world that is not in the cache. False if the deadline cut it short."""
    world = E.make_world(seed, q_common)
    ask_src = asks(world, kind)[0]
    snaps = [int(cfg.updates * f) for f in SNAPS_FRAC]
    cache.mkdir(parents=True, exist_ok=True)
    for sp in specs_for(kind, seed):
        f = cache / f"{sp.name}.pt"
        if f.exists():
            continue
        if deadline is not None and time.time() > deadline:
            return False
        st, log = train(world, sp, ask_src, world.frac_source, cfg, snaps, seed=sp.seed + 1)
        buf = io.BytesIO()
        torch.save({"state": st, "log": log}, buf)
        tmp = f.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.rename(f)
    return True


def source_stage(cohort: str, kind: str, q_common: float, wid: int, seed: int, cfg: Cfg, out: Path,
                 deadline: float | None = None, score_by: float | None = None):
    t0 = time.time()
    # train candidates (cached per agent so a world can be built across several short sessions)
    cache = Path(os.environ.get("RIWF_CACHE", str(out / "agents"))) / f"w{wid:03d}"
    if not train_agents(kind, q_common, seed, cfg, cache, deadline):
        return None
    if score_by is not None and time.time() > score_by:
        return None
    world = E.make_world(seed, q_common)
    ask_src, ask_tgt, d_src, d_tgt = asks(world, kind)
    specs = specs_for(kind, seed)
    snaps = [int(cfg.updates * f) for f in SNAPS_FRAC]
    fs, ft = world.frac_source, world.frac_target

    # banks (fresh target bank is not generated here)
    sd = np.random.default_rng(seed + 1).integers(2**31 - 1, size=6)
    fit = E.make_bank(world, 3000, d_src, int(sd[0]), fs, ask_src)
    cal = E.make_bank(world, 3000, d_src, int(sd[1]), fs, ask_src)
    val = E.make_bank(world, 2400, d_src, int(sd[2]), fs, ask_src)
    panel = E.make_bank(world, 1600, d_tgt, int(sd[3]), ft, ask_tgt)
    fresh_seed = int(sd[4])
    panel_goal_hidden = panel.goal.copy()
    panel.goal[:] = -1  # input-only: the hidden class never enters the source stage

    # teachers from labelled source episodes
    q_hat = TT.fit_analytic(cal)
    set_net = TT.fit_net("set", cal, seed + 3)
    seq_net = TT.fit_net("seq", cal, seed + 5)
    teach_panel = {
        "set": TT.net_post(set_net, panel),
        "seq": TT.net_post(seq_net, panel),
        "an": TT.analytic_post(q_hat, panel),
    }
    for s in (0.5, 1.0, 2.0):
        teach_panel[f"n{s}"] = TT.noisy(teach_panel["an"], s, seed + int(100 * s))
    teach_val = TT.net_post(set_net, val)

    tests_known = ask_src[:1] if kind == "ds" else ask_src
    tests_all = E.all_tests()

    encoders = []
    for sp in specs:
        rec = torch.load(cache / f"{sp.name}.pt", weights_only=False)
        for u in snaps:
            encoders.append({"spec": sp, "update": u, "state": rec["state"][u], "train": rec["log"]})

    rows_q, rows_n = [], []
    enc_panel_probs = []
    lock_enc = []
    val_correct = val.correct()
    for ei, e in enumerate(encoders):
        net = load(e["spec"], e["state"])
        m_fit, _ = run(net, fit)
        m_val, gp_val = run(net, val)
        m_pan, gp_pan = run(net, panel)
        enc_panel_probs.append(gp_pan)
        # native source success on the known tests with realised source answers
        hit_val = gp_val[np.arange(val.n)[:, None], np.arange(3)[None, :], val_correct]
        nat_val = float(hit_val.mean())
        # teacher-free scores on the panel (gate answers), known tests only in DS
        gsel = [0] if kind == "ds" else [0, 1, 2]
        conf_pan = gp_pan[:, gsel].max(-1)
        conf_val = gp_val[:, gsel].max(-1)
        corr_val = (gp_val[:, gsel].argmax(-1) == val_correct[:, gsel])
        t_atc = fsel.atc_threshold(conf_val, corr_val)
        loss_val = 1.0 - hit_val[:, gsel].mean(1)
        # continuous memory: linear language, kNN floor, TPE
        y_d, w_d = terms_for(teach_panel["set"], tests_known, "delta")
        tpe_p = E.answer_probs(teach_panel["set"], tests_known)
        pk = gp_pan[:, gsel if kind == "ds" else [0, 1, 2]]
        tpe = float((np.maximum(tpe_p, 1 - tpe_p) - (pk[:, :, 1] * tpe_p + pk[:, :, 0] * (1 - tpe_p))).mean())
        y_sv, w_sv = terms_for(teach_val, tests_known, "delta")
        rn = {
            "enc": ei, "name": e["spec"].name, "update": e["update"], **asdict(e["spec"]),
            "train_success": e["train"]["train_success"], "nat_val": nat_val,
            "mp": fsel.maxprob(gp_pan[:, gsel]), "ent": fsel.entropy(gp_pan[:, gsel]),
            "atc": fsel.atc_estimate(conf_pan, t_atc),
            "dev": fsel.dev_risk(m_val, m_pan, loss_val, seed + ei),
            "tpe": tpe,
            "lin": fsel.deficit_linear_cv(m_pan, y_d, w_d, seed + ei),
            "knn": fsel.deficit_knn_cv(m_pan, y_d, w_d, seed + ei),
            "slin": fsel.deficit_linear_cv(m_val[:1600], y_sv[:1600], w_sv[:1600], seed + ei),
        }
        rn["_mem_pan"] = m_pan
        rows_n.append(rn)
        # quantised memories
        for ai, (khi, klo) in enumerate(ALLOC):
            qz = Quant(m_fit, fit.regime, khi, klo, seed + 97 * ei + ai)
            c_fit = qz.codes(m_fit, fit.regime)
            c_val = qz.codes(m_val, val.regime)
            c_pan = qz.codes(m_pan, panel.regime)
            # source policy on realised source answers, validated on held-out source
            fit_corr = fit.correct()
            pol_src = fsel.fit_lookup(c_fit, fit_corr, np.ones_like(fit_corr, dtype=float), STATES)
            val_succ = float((fsel.apply_lookup(pol_src, c_val) == val_correct).mean())
            rq = {"enc": ei, "alloc": ai, "khi": khi, "klo": klo, "cand": len(rows_q), "val": val_succ,
                  "cells_pan": int(len(np.unique(c_pan)))}
            for tname, post in teach_panel.items():
                for wk in ("delta", "rho", "unit"):
                    if tname != "set" and wk != "delta":
                        continue
                    y, w = terms_for(post, tests_known, wk)
                    rq[f"k_{wk}_{tname}"] = fsel.deficit_lookup(c_pan, y, w, STATES)
            yi, wi = terms_for(teach_panel["set"], tests_all, "delta")
            rq["f_ign"] = fsel.deficit_lookup(c_pan, yi, wi, STATES)
            yu, wu = terms_for(teach_panel["set"], tests_all, "unit")
            rq["f_unw"] = fsel.deficit_lookup(c_pan, yu, wu, STATES)
            if kind == "ds":
                ya, wa = terms_for(teach_panel["set"], ask_tgt, "delta")
                rq["k_deploy"] = fsel.deficit_lookup(c_pan, ya, wa, STATES)
            rq["ll"] = fsel.cell_logloss(c_pan, teach_panel["set"], STATES)
            ys, ws = terms_for(teach_val, tests_known, "delta")
            rq["skd"] = fsel.deficit_lookup(c_val, ys, ws, STATES)
            # refitted policies for deployment, fitted on the panel with teacher answers
            yd, wd = terms_for(teach_panel["set"], ask_tgt, "delta")
            yr, wr = terms_for(teach_panel["set"], ask_tgt, "rho")
            rq["_pol_delta"] = fsel.fit_lookup(c_pan, yd, wd, STATES)
            rq["_pol_rho"] = fsel.fit_lookup(c_pan, yr, wr, STATES)
            rq["_quant"] = qz
            rows_q.append(rq)
        lock_enc.append({"enc": ei, "name": e["spec"].name, "update": e["update"]})

    # pool disagreement: share of panel answers that differ from the pool majority
    maj = (np.stack([g.argmax(-1) for g in enc_panel_probs]).mean(0) >= 0.5).astype(int)
    gsel = [0] if kind == "ds" else [0, 1, 2]
    for rn, g in zip(rows_n, enc_panel_probs):
        rn["dis"] = float((g.argmax(-1)[:, gsel] != maj[:, gsel]).mean())

    # linear refit readouts per encoder, fitted on the panel with teacher answers
    yd_t, wd_t = terms_for(teach_panel["set"], ask_tgt, "delta")
    for rn in rows_n:
        rn["_lin_panel"] = fsel.fit_linear_readout(rn.pop("_mem_pan"), yd_t, wd_t)

    lock = {
        "cohort": cohort, "kind": kind, "wid": wid, "seed": seed, "q_common": q_common,
        "cfg": asdict(cfg),
        "world": {"q_rare": world.q_rare, "q_common": world.q_common, "tests": world.tests.tolist()},
        "q_hat": q_hat, "fresh_seed": fresh_seed,
        "encoders": lock_enc,
        "scores_q": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_q],
        "scores_n": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows_n],
        "policies": [{"cand": r["cand"], "delta": r["_pol_delta"].tolist(), "rho": r["_pol_rho"].tolist()} for r in rows_q],
        "target_truth_used": False, "seconds": time.time() - t0,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"lock{wid:03d}.json", "w") as f:
        json.dump(lock, f)
    # heavy state for the target stage: models, quantisers, panel memories
    blob = {
        "encoders": [{"spec": asdict(e["spec"]), "update": e["update"], "state": e["state"]} for e in encoders],
        "quant": [r["_quant"] for r in rows_q],
        "lin": [r["_lin_panel"] for r in rows_n],
        "panel_goal": panel_goal_hidden,
        "teach_panel": teach_panel,
        "panel_seed": int(sd[3]),
    }
    buf = io.BytesIO()
    torch.save(blob, buf)
    (out / f"state{wid:03d}.pt").write_bytes(buf.getvalue())
    shutil.rmtree(cache, ignore_errors=True)
    return lock


# ---------------------------------------------------------------------------
# target stage
# ---------------------------------------------------------------------------

def target_stage(out: Path, wid: int):
    lock = json.loads((out / f"lock{wid:03d}.json").read_text())
    blob = torch.load(out / f"state{wid:03d}.pt", weights_only=False)
    kind = lock["kind"]
    world = E.make_world(lock["seed"], lock["q_common"])
    ask_src, ask_tgt, d_src, d_tgt = asks(world, kind)
    fresh = E.make_bank(world, 2400, d_tgt, lock["fresh_seed"], world.frac_target, ask_tgt)
    p_true = E.answer_probs(E.true_post(world, fresh), ask_tgt)
    correct = fresh.correct()
    newj = [1, 2] if kind == "ds" else [0, 1, 2]

    bayes = float(np.maximum(p_true, 1 - p_true)[:, newj].mean())
    bayes_known = float(np.maximum(p_true, 1 - p_true)[:, :1].mean())

    # teacher error and true deficits on the panel, computed only after the lock
    panel = E.make_bank(world, 1600, d_tgt, blob["panel_seed"], world.frac_target, ask_tgt)
    tests_known = ask_src[:1] if kind == "ds" else ask_src
    post_pan = E.true_post(world, panel)
    p_known = E.answer_probs(post_pan, tests_known)
    p_all = E.answer_probs(post_pan, E.all_tests())
    p_dep = E.answer_probs(post_pan, ask_tgt)[:, newj]
    eps = {}
    for tname, tp in blob["teach_panel"].items():
        eps[f"eps_{tname}"] = float((2 * np.abs(E.answer_probs(tp, tests_known) - p_known)).mean())
        eps[f"epsall_{tname}"] = float((2 * np.abs(E.answer_probs(tp, E.all_tests()) - p_all)).mean())
    yk, wk = fsel.answer_terms(p_known, "delta")
    ya, wa = fsel.answer_terms(p_all, "delta")
    yp, wp = fsel.answer_terms(p_dep, "delta")

    enc_out = []
    mems = []
    pan_mems = []
    for ei, e in enumerate(blob["encoders"]):
        net = load(Spec(**e["spec"]), e["state"])
        m, gp = run(net, fresh)
        pan_mems.append(run(net, panel)[0])
        mems.append(m)
        ans_lin = fsel.apply_linear_readout(blob["lin"][ei], m)
        enc_out.append({
            "enc": ei,
            "n_succ": native_success(gp, p_true) if kind == "main" else float("nan"),
            "l_succ": expected_success(ans_lin[:, newj], p_true[:, newj]),
            "l_succ_known": expected_success(ans_lin[:, :1], p_true[:, :1]),
            "_hits_n": (gp[np.arange(fresh.n)[:, None], np.arange(3)[None, :], correct]).mean(1) if kind == "main" else None,
            "_hits_l": (ans_lin[:, newj] == correct[:, newj]).mean(1),
        })
    q_out = []
    for r, qz, pol in zip(lock["scores_q"], blob["quant"], lock["policies"]):
        c = qz.codes(mems[r["enc"]], fresh.regime)
        cp = qz.codes(pan_mems[r["enc"]], panel.regime)
        pd_ = np.asarray(pol["delta"])
        pr_ = np.asarray(pol["rho"])
        ad = fsel.apply_lookup(pd_, c)
        ar = fsel.apply_lookup(pr_, c)
        y_t, w_t = fsel.answer_terms(p_true, "delta")
        y_r, w_r = fsel.answer_terms(p_true, "rho")
        q_out.append({
            "cand": r["cand"],
            "q_succ": expected_success(ad[:, newj], p_true[:, newj]),
            "q_succ_known": expected_success(ad[:, :1], p_true[:, :1]),
            "q_nreg": expected_nreg(ar[:, newj], p_true[:, newj]),
            "q_nreg_d": expected_nreg(ad[:, newj], p_true[:, newj]),
            "q_succ_r": expected_success(ar[:, newj], p_true[:, newj]),
            "true_kd": fsel.deficit_lookup(c, y_t[:, newj], w_t[:, newj], STATES),
            "true_kr": fsel.deficit_lookup(c, y_r[:, newj], w_r[:, newj], STATES),
            "pk_known": fsel.deficit_lookup(cp, yk, wk, STATES),
            "pk_all": fsel.deficit_lookup(cp, ya, wa, STATES),
            "pk_deploy": fsel.deficit_lookup(cp, yp, wp, STATES),
            "_hits": (ad[:, newj] == correct[:, newj]).mean(1),
        })
    res = {
        "wid": wid,
        "bayes": bayes, "bayes_known": bayes_known, **eps,
        "enc": [{k: v for k, v in o.items() if not k.startswith("_")} for o in enc_out],
        "q": [{k: v for k, v in o.items() if not k.startswith("_")} for o in q_out],
    }
    with open(out / f"res{wid:03d}.json", "w") as f:
        json.dump(res, f)
    np.savez_compressed(
        out / f"hits{wid:03d}.npz",
        q=np.stack([o["_hits"] for o in q_out]).astype(np.float32),
        l=np.stack([o["_hits_l"] for o in enc_out]).astype(np.float32),
        n=(np.stack([o["_hits_n"] for o in enc_out]).astype(np.float32) if kind == "main" else np.zeros(1)),
    )
    return res
