"""E4 analysis: Repeat Previous, lags unknown. Writes e4_summary.json and e4_table.csv in the run folder.

  python3 e4/anal4.py runs/e4
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import anal as A

D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e4")
TOL = 1e-9                     # ties: every candidate within 1e-9 of the best score deploys with equal chance
NEAR = 0.005
LAGS = range(1, 17)

locks, res = A.load_dir(D)
op = D / "open.json"
if op.exists():
    opened = json.loads(op.read_text())
    assert set(locks) <= set(opened["worlds"]), "a world was not opened"

# selectors: True means higher is better
SEL = {"k_known": False, "f_ign": False, "f_ign16": False, "f_ign8": False, "k_deploy": False, "val": True,
       "skd": False, "mp": True, "ent": False, "atc": True, "dis": False, "dev": False, "params": True}
RIVALS = ["val", "mp", "ent", "atc", "dis", "dev", "params"]
EP = ["l_new", "l_beyond", "l_win", "l_known", "n_known"] + [f"s{k}" for k in LAGS]

allw, adm = {}, {}
for w in sorted(locks):
    lk = locks[w]
    sc = pd.DataFrame([{k: v for k, v in r.items() if k != "d"} for r in lk["scores"]])
    df = sc.merge(pd.DataFrame(res[w]["cands"]), on="cand")
    allw[w] = df
    margin = lk["cfg"]["margin"]
    adm[w] = df[df.val >= df.val.max() - margin].copy()
nowin = {w: df[~df.win4].copy() for w, df in adm.items()}


def one_pool(worlds, tag, seed0):
    sel = A.select(worlds, SEL, EP, tol=TOL)
    rows = []
    for i, s in enumerate(SEL):
        r = {"sel": s}
        for e in ("l_new", "l_known", "n_known", "l_beyond", "l_win"):
            g = A.gap_closure(sel, s, e, seed0 + 10 * i + len(r))
            r[e] = g["raw"]
            if e in ("l_new", "l_known", "n_known"):
                r[f"{e}_gc"] = g["gc"]
                r[f"{e}_ci"] = g["ci"]
        r["l_new_tail"] = A.tail_rate(worlds, s, SEL[s], "l_new", tol=TOL)
        rows.append(r)
    new = A.diff(sel, "f_ign", "k_known", "l_new")
    old = A.diff(sel, "k_known", "f_ign", "l_known")
    h1 = {"new_f_ign_minus_k": A.paired(new, seed0 + 1), "known_k_minus_f_ign": A.paired(old, seed0 + 2),
          "crossover": {"mean": float((new + old).mean()), "ci": A.boot_mean(new + old, seed0 + 3)}}
    h2 = {r: A.paired(A.diff(sel, "f_ign", r, "l_new"), seed0 + 20 + j) for j, r in enumerate(RIVALS)}
    return sel, rows, h1, h2


def curves(sel):
    out = {}
    for s in list(SEL) + ["mean", "oracle_l_new"]:
        sub = sel[sel.sel == s]
        out[s] = [float(sub[f"s{k}"].mean()) for k in LAGS]
    return out


def same_deploy(df, a, b):
    """Probability that selectors a and b deploy the same candidate, each tied candidate equally likely."""
    ta = set(A.tied(df, a, SEL[a], TOL))
    tb = set(A.tied(df, b, SEL[b], TOL))
    return len(ta & tb) / (len(ta) * len(tb))


sel, rows, h1, h2 = one_pool(adm, "full", 1000)
sel_n, rows_n, h1_n, h2_n = one_pool(nowin, "nowin4", 2000)

# Holm over the confirmatory family: H1 on l_new and H2 against seven rivals, on both pools (sixteen tests)
fam = {"H1": h1["new_f_ign_minus_k"], **{f"H2 {r}": h2[r] for r in RIVALS},
       "nowin4 H1": h1_n["new_f_ign_minus_k"], **{f"nowin4 H2 {r}": h2_n[r] for r in RIVALS}}
ph = A.holm({k: v["p"] for k, v in fam.items()})
for k, v in fam.items():
    v["p_holm"] = ph[k]

out = {"worlds": len(adm), "tie_tol": TOL,
       "admitted_mean": float(np.mean([len(v) for v in adm.values()])),
       "admitted_nowin4_mean": float(np.mean([len(v) for v in nowin.values()])),
       "pool_size": float(np.mean([len(v) for v in allw.values()]))}
for e in ("l_new", "l_known", "n_known"):
    out[f"pool_{e}"] = float(sel[sel.sel == "mean"][e].mean())
    out[f"oracle_{e}"] = float(sel[sel.sel == f"oracle_{e}"][e].mean())
out["table"] = rows
out["H1"] = h1
out["H2"] = h2
out["holm"] = ph

# H3 calibration: within worlds, l_new against 1 - k_deploy over the admitted candidates
sl, r2 = [], []
for w, df in adm.items():
    x = 1 - df.k_deploy.to_numpy()
    y = df.l_new.to_numpy()
    if np.std(x) > 1e-9 and np.std(y) > 1e-9:
        sl.append(np.polyfit(x, y, 1)[0])
        r2.append(np.corrcoef(x, y)[0, 1] ** 2)
out["calib"] = {"slope_mean": float(np.mean(sl)), "slope_ci": A.boot_mean(sl, 31), "r2_median": float(np.median(r2)),
                "mae": float(np.mean(np.concatenate([np.abs(1 - df.k_deploy - df.l_new).to_numpy() for df in adm.values()]))),
                "worlds": len(sl)}

# H4: does l_beyond separate the selectors more than l_win? Per world, range across the selectors
conf = ["k_known", "f_ign"] + RIVALS
rng_b, rng_w = [], []
for w in adm:
    sw = sel[(sel.wid == w) & sel.sel.isin(conf)]
    rng_b.append(sw.l_beyond.max() - sw.l_beyond.min())
    rng_w.append(sw.l_win.max() - sw.l_win.min())
out["H4"] = {"range_beyond_minus_win": A.paired(np.asarray(rng_b) - np.asarray(rng_w), 41),
             "gc_beyond": {s: A.gap_closure(sel, s, "l_beyond", 42)["gc"] for s in conf},
             "gc_win": {s: A.gap_closure(sel, s, "l_win", 43)["gc"] for s in conf},
             "range_beyond_mean": float(np.mean(rng_b)), "range_win_mean": float(np.mean(rng_w))}
out["H5"] = {"f_ign8_minus_f_ign": A.paired(A.diff(sel, "f_ign8", "f_ign", "l_new"), 51),
             "f_ign16_minus_f_ign": A.paired(A.diff(sel, "f_ign16", "f_ign", "l_new"), 52)}
out["curves"] = curves(sel)
out["near_tie_share"] = float(np.mean([
    (lambda v: v[1] - v[0] <= NEAR if len(v) > 1 else True)(np.sort(df.k_known.to_numpy())) for df in adm.values()]))
out["near_tie_share_pool"] = float(np.mean([locks[w]["near_tie_k_known"] for w in locks]))
out["window4_share"] = float(np.mean([A.share(df, "k_known", False, lambda r: bool(r["win4"]), TOL) for df in adm.values()]))
out["window4_share_f_ign"] = float(np.mean([A.share(df, "f_ign", False, lambda r: bool(r["win4"]), TOL) for df in adm.values()]))
out["same_deploy_f_ign_k_known"] = float(np.mean([same_deploy(df, "f_ign", "k_known") for df in adm.values()]))
out["same_deploy_nowin4"] = float(np.mean([same_deploy(df, "f_ign", "k_known") for df in nowin.values()]))
out["arch_share"] = {s: {a: float(np.mean([A.share(df, s, SEL[s], lambda r, a=a: r["arch"] == a, TOL) for df in adm.values()]))
                         for a in sorted(set(allw[next(iter(allw))].arch))} for s in ("k_known", "f_ign", "val", "params")}
# gates (as defined for the pilot), reported for every run
out["gates"] = {"G1_share_val_ge_0.9": float(np.mean(np.concatenate([(df.val >= 0.9).to_numpy() for df in allw.values()]))),
                "G1_min_val": float(min(df.val.min() for df in allw.values())),
                "G2_spread_l_new": float(np.mean([df.l_new.max() - df.l_new.min() for df in adm.values()])),
                "all_above_0.95": bool(all((df.val > 0.95).all() for df in allw.values()))}
out["nowin4"] = {"table": rows_n, "H1": h1_n, "H2": h2_n, "holm": {k[7:]: v for k, v in ph.items() if k.startswith("nowin4 ")},
                 "curves": curves(sel_n),
                 "pool_l_new": float(sel_n[sel_n.sel == "mean"]["l_new"].mean()),
                 "oracle_l_new": float(sel_n[sel_n.sel == "oracle_l_new"]["l_new"].mean())}
out["timing"] = {"source_s_mean": float(np.mean([locks[w]["seconds"] for w in locks])),
                 "deficit_s_mean": float(np.mean([locks[w]["seconds_deficits"] for w in locks]))}

(D / "e4_summary.json").write_text(json.dumps(out, indent=1, default=float))
tab = pd.DataFrame(rows)
tab_n = pd.DataFrame(rows_n)
tab_n["pool"] = "nowin4"
tab["pool"] = "full"
both = pd.concat([tab, tab_n])
for c in [c for c in both.columns if c.endswith("_ci")]:
    both[c + "_lo"] = both[c].map(lambda v: v[0])
    both[c + "_hi"] = both[c].map(lambda v: v[1])
    both = both.drop(columns=c)
both.to_csv(D / "e4_table.csv", index=False)

pd.set_option("display.width", 220)
cols = ["sel", "l_new", "l_new_gc", "l_new_tail", "l_beyond", "l_win", "l_known", "l_known_gc", "n_known"]
print("full admitted pool, worlds", len(adm), "admitted mean", round(out["admitted_mean"], 2))
print(tab[cols].round(3).to_string())
print("without window-4 candidates")
print(tab_n[cols].round(3).to_string())
for name, v in fam.items():
    print(f"{name:18s} mean {v['mean']:+.4f} ci [{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}] p {v['p']:.2e} "
          f"holm {v['p_holm']:.2e} W/T/L {v['wins']}/{v['ties']}/{v['losses']}")
print("known lag, k - f_ign:", json.dumps(h1["known_k_minus_f_ign"]), "crossover", h1["crossover"])
print("nowin4 known lag:", json.dumps(h1_n["known_k_minus_f_ign"]), "crossover", h1_n["crossover"])
for k in ("calib", "H4", "H5", "near_tie_share", "near_tie_share_pool", "window4_share", "window4_share_f_ign",
          "same_deploy_f_ign_k_known", "same_deploy_nowin4", "arch_share", "gates", "timing"):
    print(k, json.dumps(out[k], default=float))
