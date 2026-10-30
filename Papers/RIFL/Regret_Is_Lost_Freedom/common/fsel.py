"""Freedom deficits, readout probes and teacher-free selectors shared by E1, E2 and E3.

Every proposed selector in the study is a freedom deficit: the least weighted
loss that any readout in a named language can reach on the lifted support.
For a lookup table over a discrete memory this equals the weighted Bayes risk
of the memory's partition (Theorem 3.10 with Corollary 4.10). For a linear
readout over a continuous memory it is the linear-language deficit (P1).
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Binary test terms from teacher answer probabilities
# ---------------------------------------------------------------------------

def answer_terms(p1: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Best answer y and the weight of answering wrongly.

    p1 is the probability that the answer 1 is correct. The weight is the loss of
    the wrong answer under the named regret notion:
      delta : expected success lost, |2p - 1|
      rho   : normalised regret, 1 - min(p,1-p)/max(p,1-p)
      unit  : one per point (raw impurity)
    """
    p1 = np.clip(np.asarray(p1, dtype=np.float64), 0.0, 1.0)
    y = (p1 >= 0.5).astype(np.int64)
    hi = np.maximum(p1, 1.0 - p1)
    lo = np.minimum(p1, 1.0 - p1)
    if kind == "delta":
        w = hi - lo
    elif kind == "rho":
        w = 1.0 - lo / np.maximum(hi, 1e-12)
    elif kind == "unit":
        w = np.ones_like(p1)
    else:
        raise ValueError(kind)
    return y, w


# ---------------------------------------------------------------------------
# Lookup-table language: the lift
# ---------------------------------------------------------------------------

def deficit_lookup(codes: np.ndarray, y: np.ndarray, w: np.ndarray, states: int) -> float:
    """Weighted freedom deficit of a lookup-table readout, per point-test.

    codes: (n,) memory state per history (one code reused for every test)
    y, w : (n, J) best answers and wrong-answer weights
    """
    codes = np.asarray(codes, dtype=np.int64)
    y = np.asarray(y, dtype=np.int64)
    w = np.asarray(w, dtype=np.float64)
    n, J = y.shape
    total = 0.0
    for j in range(J):
        a1 = np.bincount(codes, weights=w[:, j] * (y[:, j] == 1), minlength=states)
        a0 = np.bincount(codes, weights=w[:, j] * (y[:, j] == 0), minlength=states)
        total += float(np.minimum(a1, a0).sum())
    return total / float(n * J)


def fit_lookup(codes: np.ndarray, y: np.ndarray, w: np.ndarray, states: int) -> np.ndarray:
    """Cellwise optimal lookup table (J, states). Empty cells take the test majority."""
    codes = np.asarray(codes, dtype=np.int64)
    n, J = y.shape
    pol = np.zeros((J, states), dtype=np.int64)
    for j in range(J):
        a1 = np.bincount(codes, weights=w[:, j] * (y[:, j] == 1), minlength=states)
        a0 = np.bincount(codes, weights=w[:, j] * (y[:, j] == 0), minlength=states)
        default = int(a1.sum() >= a0.sum())
        pol[j] = np.where(a1 + a0 > 0, (a1 >= a0).astype(np.int64), default)
    return pol


def apply_lookup(pol: np.ndarray, codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.int64)
    return np.stack([pol[j, codes] for j in range(pol.shape[0])], axis=1)


def cell_logloss(codes: np.ndarray, post: np.ndarray, states: int) -> float:
    """Cross-entropy of the teacher's class posterior against the cell's pooled posterior."""
    codes = np.asarray(codes, dtype=np.int64)
    post = np.asarray(post, dtype=np.float64)
    k = post.shape[1]
    pooled = np.zeros((states, k))
    for c in range(k):
        pooled[:, c] = np.bincount(codes, weights=post[:, c], minlength=states)
    cnt = np.bincount(codes, minlength=states).astype(np.float64)
    pooled = pooled / np.maximum(cnt[:, None], 1.0)
    q = np.clip(pooled[codes], 1e-9, 1.0)
    return float(-(post * np.log(q)).sum(1).mean())


# ---------------------------------------------------------------------------
# Continuous memories: linear language and a nearest-neighbour stand-in for Bayes risk
# ---------------------------------------------------------------------------

def _standardise(train: np.ndarray, *others: np.ndarray):
    mu = train.mean(0)
    sd = train.std(0)
    sd[sd < 1e-8] = 1.0
    return [(x - mu) / sd for x in (train,) + others]


def fit_linear(x: np.ndarray, y: np.ndarray, w: np.ndarray, l2: float = 1e-3, iters: int = 200) -> np.ndarray:
    """Weighted logistic readout by Newton steps. Returns coefficients with bias last."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    w = np.asarray(w, dtype=np.float64)
    if w.sum() <= 0:
        return np.zeros(d + 1)
    ws = w / w.mean()
    beta = np.zeros(d + 1)
    reg = l2 * np.eye(d + 1)
    reg[-1, -1] = 0.0
    for _ in range(iters):
        z = np.clip(xb @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        g = xb.T @ (ws * (p - y)) / n + reg @ beta
        h = (xb * (ws * p * (1 - p))[:, None]).T @ xb / n + reg + 1e-9 * np.eye(d + 1)
        step = np.linalg.solve(h, g)
        beta -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    return beta


def predict_linear(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.asarray(x, dtype=np.float64), np.ones((len(x), 1))], axis=1)
    return (xb @ beta >= 0).astype(np.int64)


def folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [np.sort(p) for p in np.array_split(idx, k)]


def deficit_linear_cv(x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int, k: int = 5) -> float:
    """Cross-fitted weighted error of the linear readout language, per point-test."""
    x = np.asarray(x, dtype=np.float64)
    n, J = y.shape
    err = 0.0
    for part in folds(n, k, seed):
        mask = np.ones(n, dtype=bool)
        mask[part] = False
        xt, xv = _standardise(x[mask], x[part])
        for j in range(J):
            beta = fit_linear(xt, y[mask, j], w[mask, j])
            pred = predict_linear(beta, xv)
            err += float((w[part, j] * (pred != y[part, j])).sum())
    return err / float(n * J)


def deficit_knn_cv(x: np.ndarray, y: np.ndarray, w: np.ndarray, seed: int, k: int = 5, nn: int = 15) -> float:
    """Cross-fitted weighted error of a nearest-neighbour readout, the finer-language floor."""
    x = np.asarray(x, dtype=np.float64)
    n, J = y.shape
    err = 0.0
    for part in folds(n, k, seed):
        mask = np.ones(n, dtype=bool)
        mask[part] = False
        xt, xv = _standardise(x[mask], x[part])
        d2 = (xv ** 2).sum(1)[:, None] + (xt ** 2).sum(1)[None, :] - 2.0 * xv @ xt.T
        nbr = np.argpartition(d2, kth=min(nn, len(xt) - 1), axis=1)[:, :nn]
        yt = y[mask]
        wt = w[mask]
        for j in range(J):
            v1 = (wt[nbr, j] * (yt[nbr, j] == 1)).sum(1)
            v0 = (wt[nbr, j] * (yt[nbr, j] == 0)).sum(1)
            pred = (v1 >= v0).astype(np.int64)
            err += float((w[part, j] * (pred != y[part, j])).sum())
    return err / float(n * J)


def linear_refit(x_panel: np.ndarray, y: np.ndarray, w: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Fit the linear readout on the panel and return answers on the evaluation histories."""
    xt, xe = _standardise(np.asarray(x_panel, dtype=np.float64), np.asarray(x_eval, dtype=np.float64))
    J = y.shape[1]
    out = np.zeros((len(x_eval), J), dtype=np.int64)
    for j in range(J):
        beta = fit_linear(xt, y[:, j], w[:, j])
        out[:, j] = predict_linear(beta, xe)
    return out


# ---------------------------------------------------------------------------
# Teacher-free selectors computed from an actor's answer probabilities
# ---------------------------------------------------------------------------

def maxprob(probs: np.ndarray) -> float:
    return float(np.asarray(probs).max(-1).mean())


def entropy(probs: np.ndarray) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-(p * np.log(p)).sum(-1).mean())


def atc_threshold(conf_src: np.ndarray, correct_src: np.ndarray) -> float:
    """Threshold t so the share of source confidences above t equals source accuracy (Garg et al. 2022)."""
    acc = float(np.mean(correct_src))
    conf = np.sort(np.asarray(conf_src, dtype=np.float64).ravel())
    q = np.clip(1.0 - acc, 0.0, 1.0)
    return float(np.quantile(conf, q))


def atc_estimate(conf_tgt: np.ndarray, t: float) -> float:
    return float(np.mean(np.asarray(conf_tgt).ravel() >= t))


def dev_risk(feat_src: np.ndarray, feat_tgt: np.ndarray, loss_src: np.ndarray, seed: int) -> float:
    """Deep embedded validation (You et al. 2019): importance-weighted source risk with a control variate."""
    xs, xt = _standardise(np.asarray(feat_src, float), np.asarray(feat_tgt, float))
    x = np.concatenate([xs, xt])
    lab = np.concatenate([np.zeros(len(xs)), np.ones(len(xt))])
    beta = fit_linear(x, lab, np.ones(len(x)), l2=1e-2, iters=100)
    z = np.clip(np.concatenate([xs, np.ones((len(xs), 1))], 1) @ beta, -20, 20)
    ratio = np.exp(z) * (len(xs) / len(xt))
    ratio = np.clip(ratio, 0.0, 50.0)
    wl = ratio * np.asarray(loss_src, float)
    var = np.var(ratio)
    eta = -np.cov(wl, ratio)[0, 1] / var if var > 1e-12 else 0.0
    return float(wl.mean() + eta * (ratio.mean() - 1.0))


def fit_linear_readout(x_panel: np.ndarray, y: np.ndarray, w: np.ndarray) -> dict:
    """Linear readout fitted on panel memories: standardisation and one coefficient row per test."""
    x = np.asarray(x_panel, dtype=np.float64)
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    z = (x - mu) / sd
    betas = np.stack([fit_linear(z, y[:, j], w[:, j]) for j in range(y.shape[1])])
    return {"mu": mu, "sd": sd, "betas": betas}


def apply_linear_readout(model: dict, x: np.ndarray) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - model["mu"]) / model["sd"]
    zb = np.concatenate([z, np.ones((len(z), 1))], axis=1)
    return (zb @ model["betas"].T >= 0).astype(np.int64)


# ---------------------------------------------------------------------------
# Multi-class answers (E4): multinomial logistic readout, the loss of Theorem A.3
# ---------------------------------------------------------------------------

def fit_multi(x: np.ndarray, y: np.ndarray, classes: int, l2: float = 1e-3, iters: int = 500) -> np.ndarray:
    """Unweighted multinomial logistic readout by L-BFGS. Minimises mean cross-entropy plus l2/2 times the squared
    weights (bias unpenalised), the multi-class form of fit_linear. Returns (classes, d + 1) with bias last."""
    from scipy.optimize import minimize

    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    xb = np.concatenate([x, np.ones((n, 1))], axis=1)
    onehot = np.zeros((n, classes))
    onehot[np.arange(n), np.asarray(y, dtype=np.int64)] = 1.0
    pen = np.ones((classes, d + 1))
    pen[:, -1] = 0.0

    def f(b):
        B = b.reshape(classes, d + 1)
        z = xb @ B.T
        z -= z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        p = np.exp(z - lse[:, None])
        loss = float((lse - (z * onehot).sum(1)).mean()) + 0.5 * l2 * float((pen * B * B).sum())
        g = (p - onehot).T @ xb / n + l2 * pen * B
        return loss, g.ravel()

    r = minimize(f, np.zeros(classes * (d + 1)), jac=True, method="L-BFGS-B",
                 options={"maxiter": iters, "gtol": 1e-6, "ftol": 1e-10})
    return r.x.reshape(classes, d + 1)


def predict_multi(beta: np.ndarray, x: np.ndarray) -> np.ndarray:
    xb = np.concatenate([np.asarray(x, dtype=np.float64), np.ones((len(x), 1))], axis=1)
    return (xb @ beta.T).argmax(1).astype(np.int64)


def deficit_multi_cv(x: np.ndarray, y: np.ndarray, fold: np.ndarray, classes: int, l2: float = 1e-3) -> float:
    """Cross-fitted error of the multinomial linear readout language with unit weights. fold (n,) gives each point's
    fold, so grouping by episode is the caller's choice. Standardised on each training part."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    err = 0
    for f in np.unique(fold):
        test = fold == f
        xt, xv = _standardise(x[~test], x[test])
        beta = fit_multi(xt, y[~test], classes, l2)
        err += int((predict_multi(beta, xv) != y[test]).sum())
    return err / float(len(y))


def fit_multi_readout(x_panel: np.ndarray, y: np.ndarray, classes: int, l2: float = 1e-3) -> dict:
    """Multinomial linear readout fitted on panel memories: standardisation and one coefficient row per class."""
    x = np.asarray(x_panel, dtype=np.float64)
    mu = x.mean(0)
    sd = x.std(0)
    sd[sd < 1e-8] = 1.0
    return {"mu": mu, "sd": sd, "beta": fit_multi((x - mu) / sd, y, classes, l2)}


def apply_multi_readout(model: dict, x: np.ndarray) -> np.ndarray:
    return predict_multi(model["beta"], (np.asarray(x, dtype=np.float64) - model["mu"]) / model["sd"])
