"""E2 analysis: T-Maze selection under hallway shift."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import anal as A

D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e2")
locks, res = A.load_dir(D)
opened = json.loads((D / "open.json").read_text())
assert set(locks) <= set(opened["worlds"]), "a world was not opened"
worlds, allw = {}, {}
for w in locks:
    df = pd.DataFrame(locks[w]["scores"]).merge(pd.DataFrame(res[w]["cands"]), on="cand")
    allw[w] = df
    adm = df[df.nat_val >= df.nat_val.max() - 0.10].copy()
    worlds[w] = adm
SEL = {"fp3": False, "kp3": False, "lin": False, "knn": False, "tpe": True, "mp": True, "ent": False, "atc": True,
       "dis": False, "dev": False, "nat_val": True, "slin": False}
EP = ["n_succ", "l_succ", "n_succ_20", "n_succ_40", "n_succ_80", "l_succ_20", "l_succ_40", "l_succ_80"]
sel = A.select(worlds, SEL, EP)
out = {"worlds": len(worlds), "admitted_mean": float(np.mean([len(v) for v in worlds.values()])),
       "pool": float(np.mean([len(v) for v in allw.values()]))}
rows = []
for i, s in enumerate(list(SEL)):
    r = {"sel": s}
    for e in ("n_succ", "l_succ"):
        g = A.gap_closure(sel, s, e, 100 + i)
        r[f"{e}"] = g["raw"]; r[f"{e}_gc"] = g["gc"]; r[f"{e}_ci"] = g["ci"]
        r[f"{e}_tail"] = A.tail_rate(worlds, s, SEL[s], e)
    rows.append(r)
tab = pd.DataFrame(rows)
for e in ("n_succ", "l_succ"):
    out[f"oracle_{e}"] = float(sel[sel.sel == f"oracle_{e}"][e].mean())
    out[f"pool_{e}"] = float(sel[sel.sel == "mean"][e].mean())
# H2: fp3 against source validation and each teacher-free selector on refit success
h2 = {}
for rival in ["nat_val", "mp", "ent", "atc", "dis", "dev"]:
    h2[rival] = A.paired(A.diff(sel, "fp3", rival, "l_succ"), 7)
ph = A.holm({k: v["p"] for k, v in h2.items()})
for k in h2: h2[k]["p_holm"] = ph[k]
out["H2"] = h2
# H3: calibration of predicted refit success 1 - fp3 against realised refit success, and fp3 vs kp3
sl, r2 = [], []
for w, df in worlds.items():
    x = 1 - df.fp3.to_numpy(); y = df.l_succ.to_numpy()
    if np.std(x) > 1e-9:
        b = np.polyfit(x, y, 1); sl.append(b[0]); r2.append(np.corrcoef(x, y)[0, 1] ** 2)
out["H3"] = {"slope_mean": float(np.mean(sl)), "slope_ci": A.boot_mean(sl, 9), "r2_median": float(np.median(r2)),
             "fp3_vs_kp3": A.paired(A.diff(sel, "fp3", "kp3", "l_succ"), 10),
             "pooled_mae": float(np.mean(np.abs(np.concatenate([(1 - df.fp3 - df.l_succ).to_numpy() for df in worlds.values()]))))}
# H4: selector x mode
asis = A.diff(sel, "tpe", "fp3", "n_succ"); refit = A.diff(sel, "fp3", "tpe", "l_succ")
out["H4"] = {"tpe_minus_fp3_asis": A.paired(asis, 11), "fp3_minus_tpe_refit": A.paired(refit, 12),
             "interaction": {"mean": float((asis + refit).mean()), "ci": A.boot_mean(asis + refit, 13)}}
# H8: online validation with n target episodes per candidate
ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
curves = {"n_succ": [], "l_succ": []}
for w, df in worlds.items():
    h = np.load(D / f"hits{w:03d}.npz")
    idx = df.cand.to_numpy()
    for key, e in (("n", "n_succ"), ("l", "l_succ")):
        hits = h[key][idx]
        o = A.online(hits, df[e].to_numpy(), ns, 200, 1000 + w)
        m, orc = df[e].mean(), df[e].max()
        curves[e].append([(o[n] - m, orc - m) for n in ns])
h8 = {}
for e in curves:
    arr = np.asarray(curves[e])      # worlds x ns x 2
    h8[e] = {str(n): {"gc": float(arr[:, k, 0].sum() / arr[:, k, 1].sum()),
                      "ci": A.boot_ratio(arr[:, k, 0], arr[:, k, 1], 20 + k)} for k, n in enumerate(ns)}
out["H8"] = h8
# teacher error
out["eps_mean"] = float(np.nanmean(np.concatenate([df.eps.to_numpy() for df in allw.values()])))
# per-length gap closure for the proposed selectors
out["by_length"] = {s: {e: A.gap_closure(sel, s, e, 50)["gc"] for e in EP if e not in ("n_succ", "l_succ")} for s in ["fp3", "tpe", "nat_val", "mp"]}
pd.set_option("display.width", 200)
print(tab.round(3).to_string())
print(json.dumps({k: v for k, v in out.items() if k not in ("H8",)}, indent=1, default=float)[:6000])
print("H8 l_succ", {n: round(v["gc"], 3) for n, v in h8["l_succ"].items()})
print("H8 n_succ", {n: round(v["gc"], 3) for n, v in h8["n_succ"].items()})
tab.to_csv(D / f"e2_table{A.SUFFIX}.csv", index=False)
# the floor's first term: the P3 floor against the linear-language deficit on reached episodes alone
out["first_term"] = A.paired(A.diff(sel, "fp3", "lin", "l_succ"), 8)
# how many admitted candidates tie at source validation's best score
out["val_tied_mean"] = float(np.mean([len(A.tied(df, "nat_val", True)) for df in worlds.values()]))
d = A.diff(sel, "fp3", "tpe", "l_succ")
out["refit_max_absdiff"] = float(np.abs(d).max())
(D / f"e2_summary{A.SUFFIX}.json").write_text(json.dumps({"table": rows, **out}, indent=1, default=float))
