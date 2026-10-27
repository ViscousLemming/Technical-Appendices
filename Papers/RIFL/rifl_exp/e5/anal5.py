"""E5 analysis. Every selector is scored by its ranking of the admitted pool, Kendall's tau_b between the
oriented score and the target endpoint in each world (e5/plan.md). Writes e5_summary.json and e5_table.csv in
the run folder, and runs the E4 pick-based analysis on the same worlds as a secondary result (e5_pick.json,
e5_pick.csv).

  python3 e5/anal5.py runs/e5              E5, confirmatory
  python3 e5/anal5.py runs/e4 --explore    the same ranking analysis on E4, exploratory, runs/e4/e4_rank.json
"""
import json
import shutil
import subprocess
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import anal as A  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
D = Path(ARGS[0] if ARGS else "runs/e5")
EXPLORE = "--explore" in sys.argv
TOL = 1e-9                      # score ties, as the tie rule
ETOL = 1e-12                    # endpoint ties
# selectors: True means higher is better, as in e4/anal4.py
SEL = {"k_known": False, "f_ign": False, "f_ign16": False, "f_ign8": False, "k_deploy": False, "val": True,
       "skd": False, "mp": True, "ent": False, "atc": True, "dis": False, "dev": False, "params": True}
RIVALS = ["val", "mp", "ent", "atc", "dis", "dev", "params"]
FAMILY = ["k_known"] + RIVALS   # H1r, then H2r
OTHER = ["skd", "f_ign8", "f_ign16", "k_deploy"]
L16 = [2, 3] + list(range(5, 17))
POOLS = ("admitted", "nowin4")
ENDS = ("l_new", "l16")


def _sgn(v, tol):
    return np.where(np.abs(v) <= tol, 0.0, np.sign(v))


def kendall_b(x, y) -> float:
    """Kendall's tau_b. A constant score or endpoint gives 0."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return 0.0
    iu = np.triu_indices(len(x), 1)
    dx = _sgn(x[:, None] - x[None, :], TOL)[iu]
    dy = _sgn(y[:, None] - y[None, :], ETOL)[iu]
    den = np.sqrt(float((dx != 0).sum()) * float((dy != 0).sum()))
    return float((dx * dy).sum() / den) if den > 0 else 0.0


def spearman(x, y) -> float:
    rx = pd.Series(np.asarray(x, dtype=np.float64)).rank(method="average").to_numpy()
    ry = pd.Series(np.asarray(y, dtype=np.float64)).rank(method="average").to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def regret_at(score, y, k: int = 3) -> float:
    """Target oracle minus the best endpoint among the selector's top k, each ordering of the candidates tied
    at the cut equally likely."""
    s = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(s) <= k:
        return 0.0
    cut = np.sort(s)[::-1][k - 1]
    above = np.nonzero(s > cut + TOL)[0]
    at = np.nonzero(np.abs(s - cut) <= TOL)[0]
    m = k - len(above)
    base = y[above].max() if len(above) else -np.inf
    ya = np.sort(y[at])[::-1]
    na = len(ya)
    best = sum(comb(na - r - 1, m - 1) / comb(na, m) * max(base, v) for r, v in enumerate(ya))
    return float(y.max() - best)


locks, res = A.load_dir(D)
op = D / "open.json"
if op.exists():
    opened = json.loads(op.read_text())
    assert set(locks) <= set(opened["worlds"]), "a world was not opened"

pools = {p: {} for p in POOLS}
for w in sorted(locks):
    lk = locks[w]
    sc = pd.DataFrame([{k: v for k, v in r.items() if k != "d"} for r in lk["scores"]])
    df = sc.merge(pd.DataFrame(res[w]["cands"]), on="cand")
    df["l16"] = df[[f"s{k}" for k in L16]].mean(1)
    adm = df[df.val >= df.val.max() - lk["cfg"]["margin"]].copy()
    pools["admitted"][w] = adm
    pools["nowin4"][w] = adm[~adm.win4].copy()

rows = []
for p in POOLS:
    for w, df in pools[p].items():
        for e in ENDS:
            y = df[e].to_numpy(dtype=np.float64)
            for s, hi in SEL.items():
                o = df[s].to_numpy(dtype=np.float64) * (1.0 if hi else -1.0)
                rows.append({"pool": p, "wid": int(w), "endpoint": e, "sel": s, "n": int(len(df)),
                             "tau": kendall_b(o, y), "rho": spearman(o, y), "regret3": regret_at(o, y, 3)})
T = pd.DataFrame(rows)


def vec(p, e, s, col="tau"):
    sub = T[(T.pool == p) & (T.endpoint == e) & (T.sel == s)].sort_values("wid")
    return sub[col].to_numpy(dtype=np.float64)


seed = 5000
tests, pv = {}, {}
for p in POOLS:
    for r in FAMILY:
        name = f"{p}: f_ign - {r}"
        tests[name] = A.paired(vec(p, "l_new", "f_ign") - vec(p, "l_new", r), seed)
        pv[name] = tests[name]["p"]
        seed += 1
adj = A.holm(pv)
for k in tests:
    tests[k]["p_holm"] = adj[k]

means = {}
for p in POOLS:
    means[p] = {}
    for e in ENDS:
        means[p][e] = {}
        for s in SEL:
            v = vec(p, e, s)
            means[p][e][s] = {"mean": float(v.mean()), "ci": A.boot_mean(v, seed)}
            seed += 1

secondary = {"tau_l16": {}, "spearman_l_new": {}, "regret3_l_new": {}}
exploratory = {"tau_l_new_other": {}}
for p in POOLS:
    secondary["tau_l16"][p] = {r: A.paired(vec(p, "l16", "f_ign") - vec(p, "l16", r), seed + i)
                               for i, r in enumerate(FAMILY)}
    seed += len(FAMILY)
    secondary["spearman_l_new"][p] = {
        "mean": {s: float(vec(p, "l_new", s, "rho").mean()) for s in SEL},
        "f_ign_minus": {r: A.paired(vec(p, "l_new", "f_ign", "rho") - vec(p, "l_new", r, "rho"), seed + i)
                        for i, r in enumerate(FAMILY)}}
    seed += len(FAMILY)
    secondary["regret3_l_new"][p] = {
        "mean": {s: float(vec(p, "l_new", s, "regret3").mean()) for s in SEL},
        "minus_f_ign": {r: A.paired(vec(p, "l_new", r, "regret3") - vec(p, "l_new", "f_ign", "regret3"), seed + i)
                        for i, r in enumerate(FAMILY)}}
    seed += len(FAMILY)
    exploratory["tau_l_new_other"][p] = {r: A.paired(vec(p, "l_new", "f_ign") - vec(p, "l_new", r), seed + i)
                                         for i, r in enumerate(OTHER)}
    seed += len(OTHER)

summary = {
    "experiment": "E4, ranking analysis, exploratory (chosen after E4's results)" if EXPLORE else "E5, confirmatory",
    "status": "exploratory" if EXPLORE else "confirmatory",
    "worlds": len(locks),
    "admitted_mean": float(np.mean([len(v) for v in pools["admitted"].values()])),
    "admitted_nowin4_mean": float(np.mean([len(v) for v in pools["nowin4"].values()])),
    "statistic": "Kendall tau_b between the oriented score (score if higher is better, else minus score) and the "
                 "endpoint over the pool, per world; score ties within 1e-9; a constant score gives 0",
    "endpoints": {"l_new": "mean refit success over lags 2, 3, 5, 6, 7, 8", "l16": "lags 2, 3 and 5 to 16"},
    "tau": means,
    "family": {"tests": tests, "holm_over": len(tests),
               "note": "one-sided sign-flip tests on per-world tau(f_ign) - tau(r), one million flips"},
    "secondary": secondary,
    "exploratory": exploratory,
    "per_world": rows,
}

if not EXPLORE:
    subprocess.run([sys.executable, str(ROOT / "e4" / "anal4.py"), str(D)], check=True, cwd=ROOT)
    for a, b in (("e4_summary.json", "e5_pick.json"), ("e4_table.csv", "e5_pick.csv")):
        if (D / a).exists():
            shutil.move(str(D / a), str(D / b))
    pick = json.loads((D / "e5_pick.json").read_text())
    summary["pick"] = {k: pick[k] for k in ("table", "H1", "H2", "holm", "calib", "nowin4", "window4_share",
                                            "near_tie_share") if k in pick}
    T.to_csv(D / "e5_table.csv", index=False)

out = D / ("e4_rank.json" if EXPLORE else "e5_summary.json")
out.write_text(json.dumps(summary, indent=1))

print(summary["experiment"], "| worlds", len(locks), "| admitted mean", round(summary["admitted_mean"], 2))
for p in POOLS:
    print(f"mean tau on l_new, {p}:", {s: round(means[p]["l_new"][s]["mean"], 3) for s in ["f_ign"] + FAMILY})
for k, t in tests.items():
    print(f"{k:28s} mean {t['mean']:+.3f} ci [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}] p {t['p']:.2e} "
          f"holm {t['p_holm']:.2e} W/T/L {t['wins']}/{t['ties']}/{t['losses']}")
print("wrote", out)
