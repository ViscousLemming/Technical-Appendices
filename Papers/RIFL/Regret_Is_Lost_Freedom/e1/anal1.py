"""E1 analysis: delayed-cue worlds, main strata and the demand-shift cohort."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import anal as A

D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e1")
REPS = int(sys.argv[2]) if len(sys.argv) > 2 else 200
locks, res = A.load_dir(D)
op = D / "open.json"
if op.exists():
    opened = json.loads(op.read_text())
    assert set(locks) <= set(opened["worlds"]), "a world was not opened"

QSEL = {"k_delta_set": False, "k_rho_set": False, "k_unit_set": False, "f_ign": False, "f_unw": False, "ll": False,
        "val": True, "skd": False, "k_delta_seq": False, "k_delta_an": False, "k_delta_n0.5": False,
        "k_delta_n1.0": False, "k_delta_n2.0": False, "k_deploy": False}
NSEL = {"lin": False, "knn": False, "tpe": False, "mp": True, "ent": False, "atc": True, "dev": False, "dis": False,
        "nat_val": True, "slin": False}
QEP = ["q_succ", "q_succ_known", "q_nreg", "q_succ_r", "q_nreg_d", "pk_known", "pk_all", "pk_deploy", "true_kd", "khi"]
NEP = ["n_succ", "l_succ", "l_succ_known", "reg_l", "lin"]

qw, nw, meta = {}, {}, {}
for w in locks:
    lk, rs = locks[w], res[w]
    n = pd.DataFrame(lk["scores_n"]).merge(pd.DataFrame(rs["enc"]), on="enc")
    n["reg_l"] = rs["bayes"] - n["l_succ"]
    keep = n.enc[n.nat_val >= n.nat_val.max() - 0.10]
    q = pd.DataFrame(lk["scores_q"]).merge(pd.DataFrame(rs["q"]), on="cand")
    q = q[q.enc.isin(keep)].copy()
    q["khi"] = q["khi"].astype(float)
    nw[w] = n[n.enc.isin(keep)].copy()
    qw[w] = q
    meta[w] = {"kind": lk["kind"], "cohort": lk["cohort"], "q_common": lk["q_common"], "bayes": rs["bayes"],
               "bayes_known": rs["bayes_known"], **{k: v for k, v in rs.items() if k.startswith("eps")},
               "admitted_enc": int(len(keep)), "enc": int(len(n))}

main = [w for w in locks if meta[w]["kind"] == "main"]
ds = [w for w in locks if meta[w]["kind"] == "ds"]
out = {"worlds_main": len(main), "worlds_ds": len(ds),
       "admitted_enc_mean": float(np.mean([meta[w]["admitted_enc"] for w in locks]))}


def seltabs(ws):
    qs = A.select({w: qw[w] for w in ws}, QSEL, QEP, lower=("q_nreg", "q_nreg_d", "pk_known", "pk_all", "pk_deploy", "true_kd"))
    ns = A.select({w: nw[w] for w in ws}, NSEL, [e for e in NEP])
    return qs, ns


def table(qs, ns, ws, eps_q, eps_n, sels_q, sels_n):
    rows = []
    for i, s in enumerate(sels_q):
        if s not in set(qs.sel):
            continue
        r = {"level": "quantised", "sel": s}
        for e, lower in eps_q:
            g = A.gap_closure(qs, s, e, 200 + i, lower_better=lower)
            r[e] = g["raw"]; r[f"{e}_gc"] = g["gc"]; r[f"{e}_ci"] = g["ci"]
        r["tail_q_succ"] = A.tail_rate({w: qw[w] for w in ws}, s, QSEL[s], "q_succ")
        rows.append(r)
    for i, s in enumerate(sels_n):
        r = {"level": "encoder", "sel": s}
        for e, lower in eps_n:
            g = A.gap_closure(ns, s, e, 300 + i, lower_better=lower)
            r[e] = g["raw"]; r[f"{e}_gc"] = g["gc"]; r[f"{e}_ci"] = g["ci"]
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main cohort
if main:
    qs, ns = seltabs(main)
    out["main_table"] = table(qs, ns, main, [("q_succ", False), ("q_nreg", True)], [("n_succ", False), ("l_succ", False)],
                              [s for s in QSEL if s != "k_deploy"], list(NSEL)).to_dict("records")
    for e in ("q_succ", "q_nreg"):
        out[f"main_pool_{e}"] = float(qs[qs.sel == "mean"][e].mean())
        out[f"main_oracle_{e}"] = float(qs[qs.sel == f"oracle_{e}"][e].mean())
    for e in ("n_succ", "l_succ"):
        out[f"main_pool_{e}"] = float(ns[ns.sel == "mean"][e].mean())
        out[f"main_oracle_{e}"] = float(ns[ns.sel == f"oracle_{e}"][e].mean())
    # H2
    h2 = {}
    for rival in ["val", "skd", "ll", "k_unit_set"]:
        h2[f"q:k_delta_set>{rival}"] = A.paired(A.diff(qs, "k_delta_set", rival, "q_succ"), 21)
    for rival in ["nat_val", "mp", "ent", "atc", "dis", "dev", "slin"]:
        h2[f"l:lin>{rival}"] = A.paired(A.diff(ns, "lin", rival, "l_succ"), 22)
    out["H2"] = h2
    # H3
    sl, r2 = [], []
    for w in main:
        df = nw[w]
        if df["lin"].std() > 1e-9:
            b = np.polyfit(df["lin"], df["reg_l"], 1)
            sl.append(b[0]); r2.append(np.corrcoef(df["lin"], df["reg_l"])[0, 1] ** 2)
    out["H3"] = {"slope_mean": float(np.mean(sl)), "slope_ci": A.boot_mean(sl, 31), "r2_median": float(np.median(r2)),
                 "lin_vs_knn": A.paired(A.diff(ns, "lin", "knn", "l_succ"), 32),
                 "mae": float(np.mean(np.concatenate([(nw[w]["reg_l"] - nw[w]["lin"]).abs().to_numpy() for w in main])))}
    # H4
    asis = A.diff(ns, "tpe", "lin", "n_succ")
    refit = A.diff(ns, "lin", "tpe", "l_succ")
    out["H4"] = {"tpe_minus_lin_asis": A.paired(asis, 41), "lin_minus_tpe_refit": A.paired(refit, 42),
                 "interaction": {"mean": float((asis + refit).mean()), "ci": A.boot_mean(asis + refit, 43)}}
    # H5 and H6 by stratum
    strata = sorted({meta[w]["q_common"] for w in main}, reverse=True)
    h5, h6 = {}, {}
    gains, levels = [], []
    for k, qc in enumerate(strata):
        ws = [w for w in main if meta[w]["q_common"] == qc]
        qss, _ = seltabs(ws)
        g = A.diff(qss, "k_delta_set", "k_unit_set", "q_succ")
        h5[str(qc)] = {**A.paired(g, 51 + k), "gc_delta": A.gap_closure(qss, "k_delta_set", "q_succ", 5)["gc"],
                       "gc_unit": A.gap_closure(qss, "k_unit_set", "q_succ", 5)["gc"],
                       "gc_val": A.gap_closure(qss, "val", "q_succ", 5)["gc"],
                       "pool": float(qss[qss.sel == "mean"]["q_succ"].mean()),
                       "oracle": float(qss[qss.sel == "oracle_q_succ"]["q_succ"].mean())}
        gains.extend(g.tolist()); levels.extend([k] * len(g))
        nreg = A.diff(qss, "k_delta_set", "k_rho_set", "q_nreg")        # positive favours K_rho on normalised regret
        succ = A.diff(qss, "k_delta_set", "k_rho_set", "q_succ")        # positive favours K_delta on success
        h6[str(qc)] = {"nreg_delta_minus_rho": A.paired(nreg, 61 + k), "succ_delta_minus_rho": A.paired(succ, 71 + k)}
    gains, levels = np.asarray(gains), np.asarray(levels)
    obs = np.corrcoef(levels, gains)[0, 1] if gains.std() > 0 else 0.0
    rng = np.random.default_rng(55)
    perm = np.array([np.corrcoef(rng.permutation(levels), gains)[0, 1] for _ in range(20000)]) if gains.std() > 0 else np.zeros(1)
    out["H5"] = {"by_stratum": h5, "trend_r": float(obs), "trend_p_one_sided": float((perm >= obs - 1e-12).mean())}
    out["H6"] = h6
    # H7 teacher error bound on the panel
    h7 = {}
    for t in ["set", "seq", "an", "n0.5", "n1.0", "n2.0"]:
        loss, eps = [], []
        for w in main:
            df = qw[w]
            s = df.loc[A.tied(df, f"k_delta_{t}", False)]
            loss.append(float(s["pk_known"].mean() - df["pk_known"].min()))
            eps.append(meta[w][f"eps_{t}"])
        loss, eps = np.asarray(loss), np.asarray(eps)
        h7[t] = {"eps_mean": float(eps.mean()), "loss_mean": float(loss.mean()), "within_2eps": float((loss <= 2 * eps + 1e-12).mean()),
                 "loss": loss.tolist(), "eps": eps.tolist()}
    out["H7"] = h7
    # H9 allocation of states to the rare regime
    kd = qs[qs.sel == "k_delta_set"].set_index("wid")["khi"]
    ku = qs[qs.sel == "k_unit_set"].set_index("wid")["khi"]
    ko = qs[qs.sel == "oracle_q_succ"].set_index("wid")["khi"]
    kv = qs[qs.sel == "val"].set_index("wid")["khi"]
    out["H9"] = {"khi_delta": float(kd.mean()), "khi_unit": float(ku.mean()), "khi_oracle": float(ko.mean()), "khi_val": float(kv.mean()),
                 "delta_minus_unit": A.paired((kd - ku).to_numpy(), 91),
                 "absdev_unit_minus_delta": A.paired(((ku - ko).abs() - (kd - ko).abs()).to_numpy(), 92)}
    # H8 online validation
    ns_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    cur = {"q_succ": [], "l_succ": [], "n_succ": []}
    for w in main:
        h = np.load(D / f"hits{w:03d}.npz")
        dq = qw[w]
        o = A.online(h["q"][dq.cand.to_numpy()], dq["q_succ"].to_numpy(), ns_list, REPS, 900 + w)
        cur["q_succ"].append([(o[k] - dq["q_succ"].mean(), dq["q_succ"].max() - dq["q_succ"].mean()) for k in ns_list])
        dn = nw[w]
        for key, e in (("l", "l_succ"), ("n", "n_succ")):
            o = A.online(h[key][dn.enc.to_numpy()], dn[e].to_numpy(), ns_list, REPS, 950 + w)
            cur[e].append([(o[k] - dn[e].mean(), dn[e].max() - dn[e].mean()) for k in ns_list])
    out["H8"] = {e: {str(k): {"gc": float(np.asarray(v)[:, i, 0].sum() / np.asarray(v)[:, i, 1].sum()),
                              "ci": A.boot_ratio(np.asarray(v)[:, i, 0], np.asarray(v)[:, i, 1], 80 + i)}
                     for i, k in enumerate(ns_list)} for e, v in cur.items()}

# ------------------------------------------------------------------ demand shift
if ds:
    qs, ns = seltabs(ds)
    rows = []
    for i, s in enumerate(["f_ign", "f_unw", "k_delta_set", "k_rho_set", "k_unit_set", "k_deploy", "ll", "val", "skd"]):
        r = {"sel": s}
        for e in ("q_succ", "q_succ_known"):
            g = A.gap_closure(qs, s, e, 400 + i)
            r[e] = g["raw"]; r[f"{e}_gc"] = g["gc"]; r[f"{e}_ci"] = g["ci"]
        rows.append(r)
    out["ds_table"] = rows
    for e in ("q_succ", "q_succ_known"):
        out[f"ds_pool_{e}"] = float(qs[qs.sel == "mean"][e].mean())
        out[f"ds_oracle_{e}"] = float(qs[qs.sel == f"oracle_{e}"][e].mean())
    new = A.diff(qs, "f_ign", "k_delta_set", "q_succ")
    old = A.diff(qs, "k_delta_set", "f_ign", "q_succ_known")
    out["H1"] = {"new_f_ign_minus_k": A.paired(new, 11), "known_k_minus_f_ign": A.paired(old, 12),
                 "crossover": {"mean": float((new + old).mean()), "ci": A.boot_mean(new + old, 13)},
                 "f_unw_minus_k_new": A.paired(A.diff(qs, "f_unw", "k_delta_set", "q_succ"), 14),
                 "f_ign_minus_f_unw_new": A.paired(A.diff(qs, "f_ign", "f_unw", "q_succ"), 15),
                 "f_ign_minus_val_new": A.paired(A.diff(qs, "f_ign", "val", "q_succ"), 16)}
    aux = lambda w: (lambda r: bool(locks[w]["scores_n"][int(r["enc"])]["aux"]))
    out["H1"]["aux_share_f_ign"] = float(np.mean([A.share(qw[w], "f_ign", False, aux(w)) for w in ds]))
    out["H1"]["aux_share_k"] = float(np.mean([A.share(qw[w], "k_delta_set", False, aux(w)) for w in ds]))

# which training delays each selector deploys (descriptive)
if main:
    shares = {}
    for s, hi, level in [("k_delta_set", False, "q"), ("val", True, "q"), ("lin", False, "n"), ("nat_val", True, "n"), ("tpe", False, "n")]:
        shares[s] = {}
        for d in ("short", "mid", "long"):
            vals = []
            for w in main:
                df = qw[w] if level == "q" else nw[w]
                vals.append(A.share(df, s, hi, lambda r, w=w: locks[w]["scores_n"][int(r["enc"])]["delays"] == d))
            shares[s][d] = float(np.mean(vals))
    for level, e in (("q", "q_succ"), ("n", "l_succ")):
        shares[f"oracle_{e}"] = {}
        for d in ("short", "mid", "long"):
            vals = []
            for w in main:
                df = qw[w] if level == "q" else nw[w]
                vals.append(A.share(df, e, True, lambda r, w=w: locks[w]["scores_n"][int(r["enc"])]["delays"] == d))
            shares[f"oracle_{e}"][d] = float(np.mean(vals))
    out["delay_shares"] = shares

# Holm over the confirmatory families H1 and H2
pv = {}
for k, v in out.get("H2", {}).items():
    pv[f"H2 {k}"] = v["p"]
if "H1" in out:
    pv["H1 new"] = out["H1"]["new_f_ign_minus_k"]["p"]
if pv:
    ph = A.holm(pv)
    out["holm"] = ph

(D / f"e1_summary{A.SUFFIX}.json").write_text(json.dumps(out, indent=1, default=float))
if "main_table" in out:
    t = pd.DataFrame(out["main_table"])
    pd.set_option("display.width", 220)
    cols = [c for c in ["level", "sel", "q_succ", "q_succ_gc", "q_nreg", "q_nreg_gc", "n_succ", "n_succ_gc", "l_succ", "l_succ_gc", "tail_q_succ"] if c in t]
    print(t[cols].round(3).to_string())
if "ds_table" in out:
    print(pd.DataFrame(out["ds_table"])[["sel", "q_succ", "q_succ_gc", "q_succ_known", "q_succ_known_gc"]].round(3).to_string())
for k in ["H1", "H3", "H4", "H5", "H6", "H9", "holm"]:
    if k in out:
        print(k, json.dumps(out[k], default=float)[:1500])
if "H7" in out:
    print("H7", {t: (round(v["eps_mean"], 3), round(v["loss_mean"], 4), v["within_2eps"]) for t, v in out["H7"].items()})
if "H8" in out:
    for e in out["H8"]:
        print("H8", e, {k: round(v["gc"], 3) for k, v in out["H8"][e].items()})
