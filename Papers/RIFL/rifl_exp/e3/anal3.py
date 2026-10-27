"""E3 analysis: noisy T-Maze, selection when agents differ in what they sense."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import anal as A

D = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/e3")
locks, res = A.load_dir(D)
opened = json.loads((D / "open.json").read_text())
assert set(locks) <= set(opened["worlds"]), "a world was not opened"
worlds, allw = {}, {}
for w in locks:
    df = pd.DataFrame(locks[w]["scores"]).merge(pd.DataFrame(res[w]["cands"]), on="cand")
    allw[w] = df
    worlds[w] = df[df.nat_val >= df.nat_val.max() - 1.0].copy()
SEL = {"fp4": True, "fk4": True, "d_ret": False, "d_suc": False, "d_one": False, "d_knn": False, "acq": False,
       "tpe": True, "mp": True, "ent": False, "atc": True, "dis": False, "dev": False, "nat_val": True, "slin": False}
EP = ["n_ret", "l_ret", "n_succ", "l_succ", "l_ret_r1", "l_ret_r0", "n_ret_r1", "n_ret_r0", "reads1", "reach_f"]
sel = A.select(worlds, SEL, EP)
out = {"worlds": len(worlds), "admitted_mean": float(np.mean([len(v) for v in worlds.values()]))}
rows = []
for i, s in enumerate(SEL):
    r = {"sel": s}
    for e in ("n_ret", "l_ret"):
        g = A.gap_closure(sel, s, e, 100 + i)
        r[e] = g["raw"]; r[f"{e}_gc"] = g["gc"]; r[f"{e}_ci"] = g["ci"]
        r[f"{e}_tail"] = A.tail_rate(worlds, s, SEL[s], e)
    r["reads1"] = float(sel[sel.sel == s]["reads1"].mean())
    rows.append(r)
tab = pd.DataFrame(rows)
for e in ("n_ret", "l_ret"):
    out[f"oracle_{e}"] = float(sel[sel.sel == f"oracle_{e}"][e].mean())
    out[f"pool_{e}"] = float(sel[sel.sel == "mean"][e].mean())
out["oracle_reads1"] = float(sel[sel.sel == "oracle_l_ret"]["reads1"].mean())
out["pool_reads1"] = float(np.mean([df.reads1.mean() for df in worlds.values()]))
h2 = {r: A.paired(A.diff(sel, "fp4", r, "l_ret"), 7) for r in ["nat_val", "mp", "ent", "atc", "dis", "dev"]}
ph = A.holm({k: v["p"] for k, v in h2.items()})
for k in h2: h2[k]["p_holm"] = ph[k]
out["H2"] = h2
asis = A.diff(sel, "tpe", "fp4", "n_ret"); refit = A.diff(sel, "fp4", "tpe", "l_ret")
out["H4"] = {"tpe_minus_fp4_asis": A.paired(asis, 11), "fp4_minus_tpe_refit": A.paired(refit, 12),
             "interaction": {"mean": float((asis + refit).mean()), "ci": A.boot_mean(asis + refit, 13)}}
out["H10"] = {"fp4_minus_d_ret": A.paired(A.diff(sel, "fp4", "d_ret", "l_ret"), 14),
              "fp4_minus_acq": A.paired(A.diff(sel, "fp4", "acq", "l_ret"), 15)}
out["H11"] = {"d_ret_minus_d_suc": A.paired(A.diff(sel, "d_ret", "d_suc", "l_ret"), 16),
              "d_ret_minus_d_one": A.paired(A.diff(sel, "d_ret", "d_one", "l_ret"), 17)}
# calibration of the P4 prediction
sl, r2, mae = [], [], []
for w, df in worlds.items():
    x, y = df.fp4.to_numpy(), df.l_ret.to_numpy()
    if np.std(x) > 1e-9:
        b = np.polyfit(x, y, 1); sl.append(b[0]); r2.append(np.corrcoef(x, y)[0, 1] ** 2)
    mae.append(np.abs(x - y).mean())
out["calib"] = {"slope_mean": float(np.mean(sl)), "slope_ci": A.boot_mean(sl, 18), "r2_median": float(np.median(r2)), "mae": float(np.mean(mae))}
# H12: the split. Realised refit shortfall from the informed return = acquisition + aliasing + execution (true terms)
split = []
for w, df in allw.items():
    short = df.best_true - df.l_ret
    acq = df.acq_true
    ali = df.alias_true_panel
    exe = short - acq - ali
    split.append({"wid": w, "short": float(short.mean()), "acq": float(acq.mean()), "alias": float(ali.mean()), "exec": float(exe.mean()),
                  "exec_neg_share": float((exe < -0.05).mean()), "est_acq": float(df.acq.mean()), "est_alias": float((df.reach * df.d_ret).mean())})
sp = pd.DataFrame(split)
out["H12"] = {k: float(sp[k].mean()) for k in ["short", "acq", "alias", "exec", "exec_neg_share", "est_acq", "est_alias"]}
# teacher error and the P5 bound for fp4 selection on the panel scale
out["eps_mean"] = float(np.nanmean(np.concatenate([df.eps.to_numpy() for df in allw.values()])))
# online validation
ns = [1, 2, 4, 8, 16, 32, 64, 128, 256]
cur = {"n_ret": [], "l_ret": []}
for w, df in worlds.items():
    h = np.load(D / f"hits{w:03d}.npz")
    idx = df.cand.to_numpy()
    for key, e in (("n", "n_ret"), ("l", "l_ret")):
        o = A.online(h[key][idx], df[e].to_numpy(), ns, 200, 2000 + w)
        m, orc = df[e].mean(), df[e].max()
        cur[e].append([(o[n] - m, orc - m) for n in ns])
out["H8"] = {e: {str(n): {"gc": float(np.asarray(v)[:, k, 0].sum() / np.asarray(v)[:, k, 1].sum()),
                          "ci": A.boot_ratio(np.asarray(v)[:, k, 0], np.asarray(v)[:, k, 1], 30 + k)} for k, n in enumerate(ns)}
             for e, v in cur.items()}
pd.set_option("display.width", 220)
print(tab[["sel", "n_ret", "n_ret_gc", "l_ret", "l_ret_gc", "l_ret_tail", "reads1"]].round(3).to_string())
print(json.dumps({k: out[k] for k in out if k not in ("H8", "H2")}, default=float)[:3000])
print("H2", {k: (round(v["mean"], 3), v["wins"], v["losses"], round(v["p_holm"], 6)) for k, v in h2.items()})
for e in out["H8"]:
    print("H8", e, {n: round(v["gc"], 3) for n, v in out["H8"][e].items()})
tab.to_csv(D / f"e3_table{A.SUFFIX}.csv", index=False)
(D / f"e3_summary{A.SUFFIX}.json").write_text(json.dumps({"table": rows, **out}, indent=1, default=float))
