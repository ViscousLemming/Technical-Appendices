"""One E4 world: a pool of memory agents trained on Repeat Previous at lag 4 and deployed on lags nobody trained for.

Source stage: check the port against popgym, train the pool, roll every candidate over
held-out source episodes and a shared unlabelled target panel from the official
environment, compute every selector and every refitted readout for lags 1 to 16, and
write the lock. Target stage: fresh episodes from the official environment, refit
success at lags 1 to 16 and native return at lag 4.
"""
from __future__ import annotations

import base64
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
from e2.agent import Spec
from e4 import rprev as R
from e4.ppo4 import Cfg, load, n_params, probe, train

ARCHS = [("gru", 8), ("gru", 32), ("gru", 128), ("lstm", 8), ("lstm", 32), ("lstm", 128),
         ("trxl", 8), ("trxl", 32), ("trxl1", 4)]
WIN4 = "trxl1"                 # one block, window 4: keeps lags 1 to 4 only
SEEDS_PER_ARCH = 2
SNAPS_FRAC = (0.5, 1.0)
LAGS = range(1, R.STEPS + 1)   # every lag an episode can pose
REFIT = range(1, 17)           # lags with a stored readout and a target endpoint
N_VAL = 512                    # held-out source episodes
N_PANEL = 512                  # shared unlabelled target panel
N_FRESH = 1000                 # fresh target episodes, opened after the lock
N_CHECK = 200                  # port check episodes
FOLDS = 5
L2 = 1e-3
NEAR_TIE = 0.005


def specs_for(seed: int) -> list[Spec]:
    out, i = [], 0
    for arch, size in ARCHS:
        for _ in range(SEEDS_PER_ARCH):
            out.append(Spec(arch, size, int(seed + 7919 * i + 13)))
            i += 1
    return out


def cache_for(out: Path, wid: int) -> Path:
    return Path(os.environ.get("RIWF_CACHE", str(Path(out) / "agents"))) / f"w{wid:03d}"


def snaps_for(cfg: Cfg) -> list[int]:
    return [int(cfg.updates * f) for f in SNAPS_FRAC]


def train_one(seed: int, cfg: Cfg, cache: Path, i: int) -> None:
    sp = specs_for(seed)[i]
    f = cache / f"{sp.name}.pt"
    if f.exists():
        return
    cache.mkdir(parents=True, exist_ok=True)
    t = time.time()
    st, log = train(sp, cfg, sp.seed + 1, snaps_for(cfg))
    log["seconds"] = time.time() - t
    buf = io.BytesIO()
    torch.save({"state": st, "log": log}, buf)
    tmp = f.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(buf.getvalue())
    tmp.rename(f)


def train_agents(seed: int, cfg: Cfg, cache: Path, deadline=None) -> bool:
    for i, sp in enumerate(specs_for(seed)):
        if (cache / f"{sp.name}.pt").exists():
            continue
        if deadline is not None and time.time() > deadline:
            return False
        train_one(seed, cfg, cache, i)
    return True


def enc(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a, dtype=np.float32)
    return {"shape": list(a.shape), "f32": base64.b64encode(a.tobytes()).decode()}


def dec(d: dict) -> np.ndarray:
    return np.frombuffer(base64.b64decode(d["f32"]), dtype=np.float32).reshape(d["shape"]).astype(np.float64)


def ep_folds(n: int, seed: int) -> np.ndarray:
    """Fold of each episode, five folds of near-equal size."""
    f = np.zeros(n, dtype=np.int64)
    for j, part in enumerate(fsel.folds(n, FOLDS, seed)):
        f[part] = j
    return f


def lag_data(feat: np.ndarray, suits: np.ndarray, k: int):
    """Memories and lag-k answers at the steps t >= k where lag k is posed, flattened over episodes."""
    y, _ = R.labels(suits, k)
    return feat[:, k - 1:].reshape(-1, feat.shape[-1]), y[:, k - 1:].ravel()


def deficit(feat: np.ndarray, suits: np.ndarray, k: int, fold_ep: np.ndarray) -> float:
    """d_k: cross-fitted error of the multinomial linear readout from the memory at step t to o_{t-k+1}."""
    x, y = lag_data(feat, suits, k)
    fold = np.repeat(fold_ep, R.STEPS - k + 1)
    return fsel.deficit_multi_cv(x, y, fold, R.SUITS, L2)


def source_stage(cohort: str, kind: str, q_common, wid: int, seed: int, cfg: Cfg, out: Path,
                 deadline=None, score_by=None):
    t0 = time.time()
    cache = cache_for(out, wid)
    sd = np.random.default_rng(seed + 1).integers(2**31 - 1, size=16)
    port_ok = R.port_check(int(sd[3]), N_CHECK)
    if not train_agents(seed, cfg, cache, deadline):
        return None
    if score_by is not None and time.time() > score_by:
        return None
    t_train = time.time() - t0
    specs = specs_for(seed)
    snaps = snaps_for(cfg)
    cands = []
    for sp in specs:
        rec = torch.load(cache / f"{sp.name}.pt", weights_only=False)
        for u in snaps:
            cands.append({"spec": sp, "update": u, "state": rec["state"][u], "train": rec["log"]})

    # held-out source episodes and the shared target panel, both dealt by the official environment
    val_seeds = [int(sd[0]) + i for i in range(N_VAL)]
    pan_seeds = [int(sd[1]) + i for i in range(N_PANEL)]
    s_val = R.official_orders(val_seeds)
    s_pan = R.official_orders(pan_seeds)
    fold_pan = ep_folds(N_PANEL, int(sd[4]))
    fold_val = ep_folds(N_VAL, int(sd[4]) + 1)
    k = cfg.k_src
    y_val4, m_val4 = R.labels(s_val, k)
    m4 = np.arange(R.STEPS) >= k - 1              # steps where the known lag is posed

    rows, readouts, acts_pan = [], [], []
    t_def = 0.0
    for ci, c in enumerate(cands):
        net = load(c["spec"], c["state"])
        f_val, p_val = probe(net, s_val)
        f_pan, p_pan = probe(net, s_pan)
        # native return on held-out source episodes (expected return of the stochastic policy at lag 4)
        val = float(R.expected_rewards(s_val, p_val, k).sum(1).mean())
        # freedom deficits for every lag on the panel
        t1 = time.time()
        d = [deficit(f_pan, s_pan, kk, fold_pan) for kk in LAGS]
        skd = deficit(f_val, s_val, k, fold_val)
        t_def += time.time() - t1
        w = np.asarray([R.CARDS - kk for kk in LAGS], dtype=np.float64)
        dd = np.asarray(d)
        # label-free scores on the known lag
        pp = p_pan[:, m4].reshape(-1, R.ACTIONS)
        pv = p_val[:, m4].reshape(-1, R.ACTIONS)
        corr_v = pv.argmax(-1) == y_val4[:, m4].ravel()
        t_atc = fsel.atc_threshold(pv.max(-1), corr_v)
        loss_v = 1.0 - np.take_along_axis(pv, y_val4[:, m4].reshape(-1, 1), -1)[:, 0]
        fv = f_val[:, m4].reshape(-1, f_val.shape[-1])
        fp = f_pan[:, m4].reshape(-1, f_pan.shape[-1])
        row = {
            "cand": ci, "name": c["spec"].name, "arch": c["spec"].arch, "size": c["spec"].size,
            "seed": c["spec"].seed, "update": c["update"], "win4": c["spec"].arch == WIN4,
            "train_return": c["train"]["train_return"],
            "val": val,
            "d": d,
            "k_known": float(dd[k - 1]),
            "f_ign": float((w * dd).sum() / w.sum()),
            "f_ign16": float(dd[:16].mean()),
            "f_ign8": float(dd[:8].mean()),
            "k_deploy": float(np.mean([dd[kk - 1] for kk in cfg.new_lags])),
            "skd": float(skd),
            "mp": fsel.maxprob(pp),
            "ent": fsel.entropy(pp),
            "atc": fsel.atc_estimate(pp.max(-1), t_atc),
            "dev": fsel.dev_risk(fv, fp, loss_v, seed + ci),
            "params": n_params(net),
        }
        rows.append(row)
        acts_pan.append(p_pan[:, m4].argmax(-1))
        # refitted readouts for lags 1 to 16 on the full panel
        rd = {}
        for kk in REFIT:
            x, y = lag_data(f_pan, s_pan, kk)
            mdl = fsel.fit_multi_readout(x, y, R.SUITS, L2)
            rd[str(kk)] = {"mu": enc(mdl["mu"]), "sd": enc(mdl["sd"]), "beta": enc(mdl["beta"])}
        readouts.append(rd)

    # disagreement with the pool's majority answer on the panel (the same inputs for every candidate)
    A = np.stack(acts_pan)                                     # (cands, episodes, steps)
    counts = np.stack([(A == a).sum(0) for a in range(R.ACTIONS)])
    maj = counts.argmax(0)
    for row, a in zip(rows, A):
        row["dis"] = float((a != maj).mean())

    kk_sorted = np.sort([r["k_known"] for r in rows])
    lock = {
        "cohort": cohort, "kind": "rprev", "wid": wid, "seed": seed, "cfg": asdict(cfg),
        "popgym": R.popgym_version(), "port_check": bool(port_ok), "port_check_seed": int(sd[3]),
        "archs": ARCHS, "seeds_per_arch": SEEDS_PER_ARCH, "snaps": snaps, "win4_arch": WIN4,
        "win4_cands": [r["cand"] for r in rows if r["win4"]],
        "k_src": k, "new_lags": list(cfg.new_lags),
        "deploy_lags_used_by_selector": False,
        "reference_selectors": ["k_deploy"],
        "val_seeds": [val_seeds[0], len(val_seeds)], "panel_seeds": [pan_seeds[0], len(pan_seeds)],
        "fresh_seed": int(sd[2]), "fold_seed": int(sd[4]),
        "near_tie_k_known": bool(kk_sorted[1] - kk_sorted[0] <= NEAR_TIE),
        "scores": rows, "readouts": readouts,
        "target_truth_used": False,
        "seconds": time.time() - t0, "seconds_train": t_train, "seconds_deficits": t_def,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"lock{wid:03d}.json").write_text(json.dumps(lock))
    blob = {"cands": [{"spec": asdict(c["spec"]), "update": c["update"], "state": c["state"],
                       "curve": c["train"]["curve"]} for c in cands]}
    buf = io.BytesIO()
    torch.save(blob, buf)
    (out / f"state{wid:03d}.pt").write_bytes(buf.getvalue())
    shutil.rmtree(cache, ignore_errors=True)
    return lock


def target_stage(out: Path, wid: int):
    lock = json.loads((out / f"lock{wid:03d}.json").read_text())
    blob = torch.load(out / f"state{wid:03d}.pt", weights_only=False)
    assert lock["port_check"] and lock["popgym"] == R.popgym_version()
    k = lock["k_src"]
    new = lock["new_lags"]
    # fresh episodes: the official environment deals and steps them; the port only computes labels
    seeds = [lock["fresh_seed"] + i for i in range(N_FRESH)]
    suits, seen, _ = R.official_episodes(seeds, k)
    assert np.array_equal(seen, suits[:, :R.STEPS])
    res, hits = [], []
    for ci, (c, rd) in enumerate(zip(blob["cands"], lock["readouts"])):
        net = load(Spec(**c["spec"]), c["state"])
        feat, probs = probe(net, suits)
        row = {"cand": ci}
        per_ep = {}
        for kk in REFIT:
            m = rd[str(kk)]
            mdl = {"mu": dec(m["mu"]), "sd": dec(m["sd"]), "beta": dec(m["beta"])}
            x, y = lag_data(feat, suits, kk)
            hit = (fsel.apply_multi_readout(mdl, x) == y).reshape(N_FRESH, -1)
            row[f"s{kk}"] = float(hit.mean())
            per_ep[kk] = hit.mean(1)
        row["l_new"] = float(np.mean([row[f"s{kk}"] for kk in new]))
        row["l_beyond"] = float(np.mean([row[f"s{kk}"] for kk in (5, 6, 7, 8)]))
        row["l_win"] = float(np.mean([row[f"s{kk}"] for kk in (2, 3)]))
        row["l_known"] = row[f"s{k}"]
        row["n_known"] = float(R.expected_rewards(suits, probs, k).sum(1).mean())
        y4, _ = R.labels(suits, k)
        row["n_acc"] = float((probs.argmax(-1) == y4)[:, k - 1:].mean())
        res.append(row)
        hits.append(np.mean([per_ep[kk] for kk in new], 0))
    out_res = {"wid": wid, "cands": res}
    (out / f"res{wid:03d}.json").write_text(json.dumps(out_res))
    np.savez_compressed(out / f"hits{wid:03d}.npz", l=np.stack(hits).astype(np.float32))
    return out_res
