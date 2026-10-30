"""Analysis shared by E1, E2 and E3: per-world selection, gap closure, paired tests, online validation."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

B = 50000
# How a selector that gives several candidates its best score is scored. "mean" (the default) deploys each tied
# candidate with equal probability and reports the expected endpoint, the mean over the tied set. "first" takes the
# lowest candidate index, the rule of the analysis before 28 September 2026, kept only to report what it gave.
TIES = os.environ.get("RIWF_TIES", "mean")
SUFFIX = "" if TIES == "mean" else f"_{TIES}"


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def boot_mean(x, seed: int, n: int = B) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def boot_ratio(num, den, seed: int, n: int = B) -> list[float]:
    """Percentile interval for sum(num)/sum(den) resampling worlds."""
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(n, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))]


def signflip(x, draws: int = 1_000_000, seed: int = 17) -> float:
    """One-sided p-value for mean(x) > 0: exact up to 24 worlds, Monte Carlo above."""
    x = np.asarray(x, dtype=np.float64)
    target = x.sum() - 1e-12
    if len(x) <= 24:
        half = len(x) // 2

        def sums(v):
            if len(v) == 0:
                return np.zeros(1)
            s = np.array(list(itertools.product((-1.0, 1.0), repeat=len(v))))
            return s @ v

        left = sums(x[:half])
        right = np.sort(sums(x[half:]))
        total = 0
        for a in left:
            total += len(right) - np.searchsorted(right, target - a, side="left")
        return float(total / (len(left) * len(right)))
    rng = np.random.default_rng(seed)
    hit = 0
    done = 0
    while done < draws:
        k = min(100_000, draws - done)
        s = rng.integers(0, 2, size=(k, len(x)), dtype=np.int8) * 2 - 1
        hit += int(((s * x).sum(1) >= target).sum())
        done += k
    return float((hit + 1) / (draws + 1))


def holm(pvals: dict) -> dict:
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    out, run = {}, 0.0
    for i, k in enumerate(keys):
        run = max(run, min(1.0, (m - i) * pvals[k]))
        out[k] = run
    return out


def paired(x, seed: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "ci": boot_mean(x, seed), "p": signflip(x),
            "wins": int((x > 1e-12).sum()), "ties": int((np.abs(x) <= 1e-12).sum()), "losses": int((x < -1e-12).sum()),
            "n": int(len(x))}


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def tied(df: pd.DataFrame, col: str, higher: bool, tol: float = 1e-12) -> pd.Index:
    """Row labels of the candidates that share the selector's best score (within tol, 1e-12 unless given).
    Under TIES == "first" only the lowest candidate index is returned."""
    v = df[col].to_numpy(dtype=np.float64)
    v = np.where(np.isfinite(v), v, -np.inf if higher else np.inf)
    best = v.max() if higher else v.min()
    rows = np.nonzero(np.isclose(v, best, rtol=0, atol=tol))[0]
    if TIES == "first":
        rows = rows[:1]
    return df.index[rows]


def pick(df: pd.DataFrame, col: str, higher: bool) -> int:
    """Row label of the lowest-index candidate among those tied at the best score."""
    return int(tied(df, col, higher)[0])


def share(df: pd.DataFrame, col: str, higher: bool, pred, tol: float = 1e-12) -> float:
    """Probability that the deployed candidate satisfies pred, each tied candidate being equally likely."""
    sub = df.loc[tied(df, col, higher, tol)]
    return float(np.mean([bool(pred(r)) for _, r in sub.iterrows()]))


def select(worlds: dict[int, pd.DataFrame], selectors: dict[str, bool], endpoints: list[str], lower=(),
           tol: float = 1e-12) -> pd.DataFrame:
    """One row per world and selector with the expected endpoints of the deployed candidate, each candidate tied at
    the selector's best score being equally likely, plus mean and oracle rows."""
    rows = []
    for wid, df in worlds.items():
        for s, hi in selectors.items():
            if s not in df.columns:
                continue
            sub = df.loc[tied(df, s, hi, tol)]
            rows.append({"wid": wid, "sel": s, **{e: float(sub[e].mean()) for e in endpoints},
                         "cand": int(sub["cand"].iloc[0]) if "cand" in sub else -1, "n_tied": int(len(sub))})
        rows.append({"wid": wid, "sel": "mean", **{e: float(df[e].mean()) for e in endpoints}})
        for e in endpoints:
            if df[e].isna().all():
                continue
            sub = df.loc[tied(df, e, e not in lower, tol)]
            rows.append({"wid": wid, "sel": f"oracle_{e}", **{e2: float(sub[e2].mean()) for e2 in endpoints}})
    return pd.DataFrame(rows)


def gap_closure(sel: pd.DataFrame, s: str, e: str, seed: int, lower_better: bool = False) -> dict:
    a = sel[sel.sel == s].set_index("wid")[e]
    m = sel[sel.sel == "mean"].set_index("wid")[e]
    o = sel[sel.sel == f"oracle_{e}"].set_index("wid")[e]
    w = a.index
    num = (a - m.loc[w]).to_numpy()
    den = (o.loc[w] - m.loc[w]).to_numpy()
    if lower_better:
        num, den = -num, -den
    return {"gc": float(num.sum() / max(den.sum(), 1e-12)), "ci": boot_ratio(num, den, seed),
            "raw": float(a.mean()), "oracle": float(o.mean()), "pool": float(m.mean())}


def diff(sel: pd.DataFrame, a: str, b: str, e: str) -> np.ndarray:
    x = sel[sel.sel == a].set_index("wid")[e]
    y = sel[sel.sel == b].set_index("wid")[e]
    w = x.index.intersection(y.index)
    return (x.loc[w] - y.loc[w]).to_numpy()


def tail_rate(worlds: dict[int, pd.DataFrame], s: str, hi: bool, e: str, tol: float = 1e-12) -> float:
    """Expected share of worlds where selector s deploys a candidate from the bottom quartile of endpoint e."""
    hits = []
    for wid, df in worlds.items():
        if s not in df.columns or df[e].isna().all():
            continue
        q = np.quantile(df[e], 0.25)
        hits.append(share(df, s, hi, lambda r: r[e] <= q + 1e-12, tol))
    return float(np.mean(hits))


# ---------------------------------------------------------------------------
# online validation with n target episodes per candidate (H8)
# ---------------------------------------------------------------------------

def online(hits: np.ndarray, endpoint: np.ndarray, ns, reps: int, seed: int) -> dict[int, float]:
    """hits (cands, episodes) realised per-episode outcomes; returns the mean endpoint of the candidate chosen by n episodes."""
    rng = np.random.default_rng(seed)
    c, e = hits.shape
    out = {}
    for n in ns:
        val = 0.0
        for _ in range(reps):
            idx = rng.choice(e, size=min(n, e), replace=False)
            score = hits[:, idx].mean(1) + 1e-9 * rng.random(c)
            val += endpoint[int(np.argmax(score))]
        out[int(n)] = val / reps
    return out


def load_dir(d: Path):
    locks, res = {}, {}
    for f in sorted(Path(d).glob("lock*.json")):
        wid = int(f.stem[4:])
        r = Path(d) / f"res{wid:03d}.json"
        if r.exists():
            locks[wid] = json.loads(f.read_text())
            res[wid] = json.loads(r.read_text())
    return locks, res
