"""Write nums.tex, every number quoted in the paper, and t_reg.tex, the regret table, from the analysis summaries and records.

Usage: python3 paper/mknums.py  (from the riwf2 folder)
In LaTeX, \\R{key} prints the value stored under key.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "nums.tex"
vals = {}


def sgn(t):
    """Print a leading minus as a true minus sign in text or math."""
    return "\\ensuremath{-}" + t[1:] if t.startswith("-") else t


def pct(x):
    return sgn(f"{100 * x:.0f}") + "\\%"


def pci(ci):
    return f"[{pct(ci[0])}, {pct(ci[1])}]"


def f(x, d=3):
    t = f"{x:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    return sgn(t)


def ci(c, d=3):
    return f"[{f(c[0], d)}, {f(c[1], d)}]"


def p(x):
    if x <= 0:
        return "0"
    if x >= 0.001:
        return f"{x:.3f}"
    e = math.floor(math.log10(x))
    m = round(x / 10 ** e, 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"\\ensuremath{{{m:.1f}\\times10^{{{e}}}}}"


def paired(prefix, d, digits=3):
    vals[f"{prefix}.mean"] = f(d["mean"], digits)
    vals[f"{prefix}.ci"] = ci(d["ci"], digits)
    vals[f"{prefix}.p"] = p(d["p"])
    if "p_holm" in d:
        vals[f"{prefix}.ph"] = p(d["p_holm"])
    vals[f"{prefix}.wins"] = str(d["wins"])
    vals[f"{prefix}.ties"] = str(d["ties"])
    vals[f"{prefix}.losses"] = str(d["losses"])
    vals[f"{prefix}.n"] = str(d["n"])


def table(prefix, rows, eps, digits):
    for r in rows:
        s = r["sel"]
        for e in eps:
            if e not in r or r[e] is None or (isinstance(r[e], float) and math.isnan(r[e])):
                continue
            vals[f"{prefix}.{s}.{e}"] = f(r[e], digits)
            vals[f"{prefix}.{s}.{e}.gc"] = pct(r[f"{e}_gc"])
            vals[f"{prefix}.{s}.{e}.ci"] = pci(r[f"{e}_ci"])
            if f"{e}_tail" in r:
                vals[f"{prefix}.{s}.{e}.tail"] = pct(r[f"{e}_tail"])
            if f"tail_{e}" in r and r[f"tail_{e}"] is not None and not (isinstance(r[f"tail_{e}"], float) and math.isnan(r[f"tail_{e}"])):
                vals[f"{prefix}.{s}.{e}.tail"] = pct(r[f"tail_{e}"])
        if "reads1" in r:
            vals[f"{prefix}.{s}.reads1"] = f(r["reads1"], 1)


def h8(prefix, d):
    for e, curve in d.items():
        for n, v in curve.items():
            vals[f"{prefix}.h8.{e}.{n}"] = pct(v["gc"])


# ------------------------------------------------------------------ E2
s2p = ROOT / "runs" / "e2" / "e2_summary.json"
if s2p.exists():
    s2 = json.loads(s2p.read_text())
    vals["e2.worlds"] = str(s2["worlds"])
    vals["e2.admitted"] = f(s2["admitted_mean"], 1)
    for e in ("n_succ", "l_succ"):
        vals[f"e2.oracle.{e}"] = f(s2[f"oracle_{e}"])
        vals[f"e2.pool.{e}"] = f(s2[f"pool_{e}"])
    table("e2", s2["table"], ["n_succ", "l_succ"], 3)
    for k, v in s2["H2"].items():
        paired(f"e2.h2.{k}", v)
    paired("e2.h4.asis", s2["H4"]["tpe_minus_fp3_asis"])
    paired("e2.h4.refit", s2["H4"]["fp3_minus_tpe_refit"])
    vals["e2.h4.int"] = f(s2["H4"]["interaction"]["mean"])
    vals["e2.h4.int.ci"] = ci(s2["H4"]["interaction"]["ci"])
    vals["e2.h3.slope"] = f(s2["H3"]["slope_mean"], 2)
    vals["e2.h3.slope.ci"] = ci(s2["H3"]["slope_ci"], 2)
    vals["e2.h3.r2"] = f(s2["H3"]["r2_median"], 3)
    vals["e2.h3.mae"] = f(s2["H3"]["pooled_mae"], 3)
    paired("e2.h3.kp3", s2["H3"]["fp3_vs_kp3"])
    vals["e2.eps"] = p(s2["eps_mean"])
    if "first_term" in s2:
        paired("e2.first", s2["first_term"])
        vals["e2.valtie"] = f(s2["val_tied_mean"], 1)
        vals["e2.refitdiff"] = f(s2["refit_max_absdiff"], 4)
    h8("e2", s2["H8"])
    for s, d in s2["by_length"].items():
        for e, v in d.items():
            vals[f"e2.len.{s}.{e}"] = pct(v)

    # share of admitted candidates at the refit ceiling
    sh = []
    for lk in sorted((ROOT / "runs" / "e2").glob("lock*.json")):
        L = json.loads(lk.read_text()); R = json.loads((lk.parent / ("res" + lk.stem[4:] + ".json")).read_text())
        best = max(c["nat_val"] for c in L["scores"])
        adm = [r for c, r in zip(L["scores"], R["cands"]) if c["nat_val"] >= best - 0.10]
        sh.append(sum(r["l_succ"] >= 0.9995 for r in adm) / len(adm))
    vals["e2.ceiling"] = pct(sum(sh) / len(sh))

# ------------------------------------------------------------------ E3
s3p = ROOT / "runs" / "e3dev" / "e3_summary.json"
if s3p.exists():
    s3 = json.loads(s3p.read_text())
    vals["e3.worlds"] = str(s3["worlds"])
    vals["e3.admitted"] = f(s3["admitted_mean"], 1)
    for e in ("n_ret", "l_ret"):
        vals[f"e3.oracle.{e}"] = f(s3[f"oracle_{e}"], 2)
        vals[f"e3.pool.{e}"] = f(s3[f"pool_{e}"], 2)
    vals["e3.oracle.reads1"] = f(s3["oracle_reads1"], 1)
    vals["e3.pool.reads1"] = f(s3["pool_reads1"], 1)
    table("e3", s3["table"], ["n_ret", "l_ret"], 2)
    for k, v in s3["H2"].items():
        paired(f"e3.h2.{k}", v, 2)
    paired("e3.h4.asis", s3["H4"]["tpe_minus_fp4_asis"], 2)
    paired("e3.h4.refit", s3["H4"]["fp4_minus_tpe_refit"], 2)
    paired("e3.h10.dret", s3["H10"]["fp4_minus_d_ret"], 2)
    paired("e3.h10.acq", s3["H10"]["fp4_minus_acq"], 2)
    paired("e3.h11.suc", s3["H11"]["d_ret_minus_d_suc"], 2)
    paired("e3.h11.one", s3["H11"]["d_ret_minus_d_one"], 2)
    for k in ("slope_mean", "r2_median", "mae"):
        vals[f"e3.calib.{k}"] = f(s3["calib"][k], 2)
    vals["e3.calib.slope_ci"] = ci(s3["calib"]["slope_ci"], 2)
    for k, v in s3["H12"].items():
        vals[f"e3.split.{k}"] = f(v, 2)
    vals["e3.eps"] = f(s3["eps_mean"], 2)
    h8("e3", s3["H8"])

# ------------------------------------------------------------------ E1
s1p = ROOT / "runs" / "e1" / "e1_summary.json"
if s1p.exists():
    s1 = json.loads(s1p.read_text())
    vals["e1.worlds.main"] = str(s1["worlds_main"])
    vals["e1.worlds.ds"] = str(s1["worlds_ds"])
    vals["e1.admitted"] = f(s1["admitted_enc_mean"], 1)
    table("e1", s1["main_table"], ["q_succ", "q_nreg", "n_succ", "l_succ"], 3)
    for e in ("q_succ", "q_nreg", "n_succ", "l_succ"):
        vals[f"e1.oracle.{e}"] = f(s1[f"main_oracle_{e}"])
        vals[f"e1.pool.{e}"] = f(s1[f"main_pool_{e}"])
    table("e1ds", s1["ds_table"], ["q_succ", "q_succ_known"], 3)
    for e in ("q_succ", "q_succ_known"):
        vals[f"e1ds.oracle.{e}"] = f(s1[f"ds_oracle_{e}"])
        vals[f"e1ds.pool.{e}"] = f(s1[f"ds_pool_{e}"])
    for k, v in s1["H1"].items():
        if isinstance(v, dict) and "mean" in v and "wins" in v:
            paired(f"e1.h1.{k}", v)
    vals["e1.h1.cross"] = f(s1["H1"]["crossover"]["mean"])
    vals["e1.h1.cross.ci"] = ci(s1["H1"]["crossover"]["ci"])
    for k, v in s1["H2"].items():
        paired("e1.h2." + k.replace(":", "_").replace(">", "_"), v)
    for k, v in s1.get("holm", {}).items():
        vals["e1.holm." + k.replace(" ", "_").replace(":", "_").replace(">", "_")] = p(v)
    vals["e1.h3.slope"] = f(s1["H3"]["slope_mean"], 2)
    vals["e1.h3.slope.ci"] = ci(s1["H3"]["slope_ci"], 2)
    vals["e1.h3.r2"] = f(s1["H3"]["r2_median"], 3)
    vals["e1.h3.mae"] = f(s1["H3"]["mae"], 3)
    paired("e1.h3.knn", s1["H3"]["lin_vs_knn"])
    paired("e1.h4.asis", s1["H4"]["tpe_minus_lin_asis"])
    paired("e1.h4.refit", s1["H4"]["lin_minus_tpe_refit"])
    vals["e1.h4.int"] = f(s1["H4"]["interaction"]["mean"])
    vals["e1.h4.int.ci"] = ci(s1["H4"]["interaction"]["ci"])
    for qc, d in s1["H5"]["by_stratum"].items():
        key = qc.replace("0.", "")
        paired(f"e1.h5.{key}", d)
        for k in ("gc_delta", "gc_unit", "gc_val"):
            vals[f"e1.h5.{key}.{k}"] = pct(d[k])
        vals[f"e1.h5.{key}.pool"] = f(d["pool"])
        vals[f"e1.h5.{key}.oracle"] = f(d["oracle"])
    vals["e1.h5.r"] = f(s1["H5"]["trend_r"], 3)
    vals["e1.h5.p"] = p(s1["H5"]["trend_p_one_sided"])
    for qc, d in s1["H6"].items():
        key = qc.replace("0.", "")
        paired(f"e1.h6.{key}.nreg", d["nreg_delta_minus_rho"], 4)
        paired(f"e1.h6.{key}.succ", d["succ_delta_minus_rho"], 4)
    for t, d in s1["H7"].items():
        key = t.replace(".", "")
        vals[f"e1.h7.{key}.eps"] = f(d["eps_mean"], 3)
        vals[f"e1.h7.{key}.loss"] = f(d["loss_mean"], 4)
        vals[f"e1.h7.{key}.within"] = pct(d["within_2eps"])
    for k in ("khi_delta", "khi_unit", "khi_oracle", "khi_val"):
        vals[f"e1.h9.{k}"] = f(s1["H9"][k], 2)
    paired("e1.h9.du", s1["H9"]["delta_minus_unit"], 2)
    paired("e1.h9.dev", s1["H9"]["absdev_unit_minus_delta"], 2)
    h8("e1", s1["H8"])
    vals["e1.h1.aux_f"] = pct(s1["H1"]["aux_share_f_ign"])
    vals["e1.h1.aux_k"] = pct(s1["H1"]["aux_share_k"])
    for sel, d in s1.get("delay_shares", {}).items():
        for k, v in d.items():
            vals[f"e1.del.{sel}.{k}"] = pct(v)

# ------------------------------------------------------------------ derived counts
if s3p.exists():
    d = s3["H10"]["fp4_minus_acq"]
    vals["e3.h10.acq.nontie"] = str(d["wins"] + d["losses"])
    from scipy.stats import spearmanr
    rhos = []
    for lk in sorted((ROOT / "runs" / "e3dev").glob("lock*.json")):
        S = json.loads(lk.read_text())["scores"]
        best = max(c["nat_val"] for c in S)
        adm = [c for c in S if c["nat_val"] >= best - 1.0]
        rhos.append(spearmanr([c["reads1"] for c in adm], [c["nat_val"] for c in adm]).correlation)
    rhos = [r for r in rhos if not math.isnan(r)]
    vals["e3.readsrc.rho"] = f(sorted(rhos)[len(rhos) // 2] if len(rhos) % 2 else 0.5 * (sorted(rhos)[len(rhos) // 2 - 1] + sorted(rhos)[len(rhos) // 2]), 2)
    vals["e3.readsrc.pos"] = str(sum(r > 0 for r in rhos))
if s1p.exists():
    # demand shift: the auxiliary head alone, and selection within candidates that have it
    import sys
    import pandas as pd
    sys.path.insert(0, str(ROOT))
    from common import anal as A
    Arows = []
    for lk in sorted((ROOT / "runs" / "e1").glob("lock*.json")):
        L = json.loads(lk.read_text())
        if L["kind"] != "ds":
            continue
        R = json.loads((lk.parent / ("res" + lk.stem[4:] + ".json")).read_text())
        enc = {e["enc"]: e["name"] for e in L["encoders"]}
        sn = {r["enc"]: r for r in L["scores_n"]}
        best = max(r["nat_val"] for r in sn.values())
        adm = {e for e, r in sn.items() if r["nat_val"] >= best - 0.10}
        q = pd.DataFrame(L["scores_q"]).merge(pd.DataFrame(R["q"]), on="cand")
        q = q[q.enc.isin(adm)].reset_index(drop=True)
        aux = q[[("_a1_" in enc[e]) for e in q.enc]]
        Arows.append((q.q_succ.mean(), q.q_succ.max(), aux.q_succ.mean(),
                      aux.loc[A.tied(aux, "f_ign", False)].q_succ.mean(), aux.loc[A.tied(aux, "k_delta_set", False)].q_succ.mean()))
    gc = lambda j: sum(a[j] - a[0] for a in Arows) / sum(a[1] - a[0] for a in Arows)
    vals["e1ds.aux.rand"] = pct(gc(2))
    vals["e1ds.aux.f_ign"] = pct(gc(3))
    vals["e1ds.aux.k"] = pct(gc(4))
    d = [a[3] - a[4] for a in Arows]
    vals["e1ds.aux.wins"] = str(sum(x > 1e-9 for x in d))
    vals["e1ds.aux.ties"] = str(sum(abs(x) <= 1e-9 for x in d))
    vals["e1ds.aux.losses"] = str(sum(x < -1e-9 for x in d))
    vals["e1ds.aux.n"] = str(len(Arows))

if s1p.exists():
    ties = [min(d["nreg_delta_minus_rho"]["ties"], d["succ_delta_minus_rho"]["ties"]) for d in s1["H6"].values()]
    ns = {d["succ_delta_minus_rho"]["n"] for d in s1["H6"].values()}
    vals["e1.h6.minties"] = str(min(ties))
    vals["e1.h6.n"] = str(ns.pop()) if len(ns) == 1 else "??"

# ------------------------------------------------------------------ E4 and E5, Repeat Previous
RIVALS = ["val", "mp", "ent", "atc", "dis", "dev", "params"]


def rp_load(run):
    """Admitted candidates of every world of an E4-style run, merged with their target results."""
    import pandas as pd
    out = {}
    for lk in sorted((ROOT / "runs" / run).glob("lock*.json")):
        L = json.loads(lk.read_text())
        R = json.loads((lk.parent / ("res" + lk.stem[4:] + ".json")).read_text())
        sc = pd.DataFrame([{k: v for k, v in r.items() if k != "d"} for r in L["scores"]])
        df = sc.merge(pd.DataFrame(R["cands"]), on="cand")
        out[L["wid"]] = df[df.val >= df.val.max() - L["cfg"]["margin"]].copy()
    return out


def rank_block(prefix, s):
    """Mean Kendall tau of every selector per pool and endpoint, and the range over the seven rivals."""
    for pool, tag in (("admitted", ""), ("nowin4", ".nw")):
        for e, et in (("l_new", ""), ("l16", ".l16")):
            m = s["tau"][pool][e]
            for sel, d in m.items():
                vals[f"{prefix}{tag}{et}.tau.{sel}"] = f(d["mean"], 2)
                vals[f"{prefix}{tag}{et}.tau.{sel}.ci"] = ci(d["ci"], 2)
            vals[f"{prefix}{tag}{et}.tau.rmin"] = f(min(m[r]["mean"] for r in RIVALS), 2)
            vals[f"{prefix}{tag}{et}.tau.rmax"] = f(max(m[r]["mean"] for r in RIVALS), 2)


s4p = ROOT / "runs" / "e4" / "e4_summary.json"
if s4p.exists():
    s4 = json.loads(s4p.read_text())
    vals["e4.worlds"] = str(s4["worlds"])
    vals["e4.admitted"] = f(s4["admitted_mean"], 1)
    vals["e4.admitted.nw"] = f(s4["admitted_nowin4_mean"], 1)
    for e in ("l_new", "l_known"):
        vals[f"e4.pool.{e}"] = f(s4[f"pool_{e}"])
        vals[f"e4.oracle.{e}"] = f(s4[f"oracle_{e}"])
    table("e4", s4["table"], ["l_new", "l_known"], 3)
    paired("e4.h1", s4["H1"]["new_f_ign_minus_k"])
    paired("e4.h1k", s4["H1"]["known_k_minus_f_ign"])
    vals["e4.cross"] = f(s4["H1"]["crossover"]["mean"])
    vals["e4.cross.ci"] = ci(s4["H1"]["crossover"]["ci"])
    for r, d in s4["H2"].items():
        paired(f"e4.h2.{r}", d)
    paired("e4.nw.h1", s4["nowin4"]["H1"]["new_f_ign_minus_k"])
    for r, d in s4["nowin4"]["H2"].items():
        paired(f"e4.nw.h2.{r}", d)
    fail = [r for r in RIVALS if s4["H2"][r]["p_holm"] > 0.05]
    vals["e4.h2.fail.ph"] = p(min(s4["H2"][r]["p_holm"] for r in fail))
    vals["e4.h2.pass.ph"] = p(max(s4["H2"][r]["p_holm"] for r in RIVALS if r not in fail))
    vals["e4.win4.k"] = pct(s4["window4_share"])
    vals["e4.win4.f"] = pct(s4["window4_share_f_ign"])
    vals["e4.near"] = pct(s4["near_tie_share"])
    for sel, cur in s4["curves"].items():
        for k, v in enumerate(cur, 1):
            vals[f"e4.curve.{sel}.{k}"] = f(v, 2)
    vals["e4.curve.f_ign.min5"] = f(min(s4["curves"]["f_ign"][:5]), 2)
    paired("e4.h5.f8", s4["H5"]["f_ign8_minus_f_ign"])
    pil = json.loads((ROOT / "e4" / "pilot.json").read_text())
    vals["e4.pilot.g1"] = f(pil["gates"]["G1"]["value"], 2)
    vals["e4.pilot.g2"] = f(pil["gates"]["G2"]["value"], 2)

    # why one pick per world could not separate freedom from the source-based rivals
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from common import anal as A
    W4 = rp_load("e4")
    rows = []
    for w, df in W4.items():
        def dep(s, hi):
            return df.loc[A.tied(df, s, hi, 1e-9)]
        tf, tv, tm, td = dep("f_ign", False), dep("val", True), dep("mp", True), dep("dev", False)
        big = lambda t: bool(((t["size"] == 128) & t.arch.isin(["gru", "lstm"])).all())
        rows.append({"w": w, "f": tf.l_new.mean(), "v": tv.l_new.mean(), "orc": df.l_new.max(),
                     "same": set(tv.cand) == set(tm.cand) == set(td.cand), "bigv": big(tv), "bigf": big(tf),
                     "trv": bool(tv.arch.str.startswith("trxl").any())})
    d = np.array([r["f"] - r["v"] for r in rows])
    i = int(np.argmax(np.abs(d)))
    vals["e4.diag.same"] = str(sum(r["same"] for r in rows))
    vals["e4.diag.bigv"] = str(sum(r["bigv"] for r in rows))
    vals["e4.diag.bigf"] = str(sum(r["bigf"] for r in rows))
    vals["e4.diag.wv"] = f(rows[i]["v"])
    vals["e4.diag.wf"] = f(rows[i]["f"])
    vals["e4.diag.maxd"] = f(float(np.max(np.abs(np.delete(d, i)))))
    vals["e4.diag.meanwo"] = f(float(np.mean(np.delete(d, i))))
    vals["e4.diag.near"] = str(int(np.sum(np.abs(d) < 0.005)))
    vals["e4.diag.orc"] = f(float(np.mean([r["orc"] - r["f"] for r in rows])))
    vals["e4.s4min"] = f(min(float(df.s4.min()) for df in W4.values()), 2)
    s16 = max(float(df.s16.max()) for df in W4.values())
    vals["e4.s16max"] = f(s16, 2)
    # refit success of the one-block window-4 transformer, averaged over its admitted candidates
    allc = __import__("pandas").concat(list(W4.values()))
    wc = allc[allc.win4]
    for k in (1, 2, 3, 4, 5):
        vals[f"e4.win4.s{k}"] = f(float(wc[f"s{k}"].mean()), 2)
    wts = np.array([52 - k for k in range(1, 52)], dtype=float)
    vals["e4.w9"] = pct(wts[8:].sum() / wts.sum())

s4r = ROOT / "runs" / "e4" / "e4_rank.json"
if s4r.exists():
    rank_block("e4r", json.loads(s4r.read_text()))

s5p = ROOT / "runs" / "e5" / "e5_summary.json"
if s5p.exists():
    s5 = json.loads(s5p.read_text())
    vals["e5.worlds"] = str(s5["worlds"])
    vals["e5.admitted"] = f(s5["admitted_mean"], 1)
    vals["e5.admitted.nw"] = f(s5["admitted_nowin4_mean"], 1)
    rank_block("e5", s5)
    T = s5["family"]["tests"]
    for k, d in T.items():
        pool, r = k.split(": f_ign - ")
        paired(f"e5.t.{'nw' if pool == 'nowin4' else 'adm'}.{r}", d, 2)
    vals["e5.ph"] = p(max(d["p_holm"] for d in T.values()))
    vals["e5.nfam"] = str(len(T))
    adm = [d for k, d in T.items() if k.startswith("admitted")]
    vals["e5.adm.minwins"] = str(min(d["wins"] for d in adm))
    sec = s5["secondary"]["admitted"] if "admitted" in s5["secondary"] else None
    S = s5["secondary"]
    comp = ["k_known"] + RIVALS
    vals["e5.sec.rho.f"] = f(S["spearman_l_new"]["admitted"]["mean"]["f_ign"], 2)
    vals["e5.sec.rho.max"] = f(max(S["spearman_l_new"]["admitted"]["mean"][r] for r in comp), 2)
    vals["e5.sec.l16.f"] = f(s5["tau"]["admitted"]["l16"]["f_ign"]["mean"], 2)
    vals["e5.sec.l16.max"] = f(max(s5["tau"]["admitted"]["l16"][r]["mean"] for r in comp), 2)
    vals["e5.sec.r3.f"] = f(S["regret3_l_new"]["admitted"]["mean"]["f_ign"], 4)
    vals["e5.sec.r3.min"] = f(min(S["regret3_l_new"]["admitted"]["mean"][r] for r in comp), 3)
    vals["e5.sec.r3.max"] = f(max(S["regret3_l_new"]["admitted"]["mean"][r] for r in comp), 3)
    ps = ([d["p"] for d in S["tau_l16"]["admitted"].values()] + [d["p"] for d in S["spearman_l_new"]["admitted"]["f_ign_minus"].values()]
          + [d["p"] for d in S["regret3_l_new"]["admitted"]["minus_f_ign"].values()])
    vals["e5.sec.pmax"] = p(max(ps))
    r3w = [d["wins"] for d in S["regret3_l_new"]["admitted"]["minus_f_ign"].values()]
    vals["e5.sec.r3.minwins"] = str(min(r3w))
    for k, d in s5["exploratory"]["tau_l_new_other"]["admitted"].items():
        paired(f"e5.x.{k}", d, 2)
    vals["e5.x.k_deploy.abs"] = f(abs(s5["exploratory"]["tau_l_new_other"]["admitted"]["k_deploy"]["mean"]), 2)
    pk = s5["pick"]
    table("e5p", pk["table"], ["l_new", "l_known"], 3)
    paired("e5p.h1", pk["H1"]["new_f_ign_minus_k"])
    for r, d in pk["H2"].items():
        paired(f"e5p.h2.{r}", d)
    W5 = rp_load("e5")
    vals["e5.s16max"] = f(max(float(df.s16.max()) for df in W5.values()), 2)

# ------------------------------------------------------------------ regret on new tasks, Table 2
# E1 demand shift: normalised regret of the deployed lookup readout on the two new tests, per world, admitted pool,
# equal chance among tied candidates. E4 and E5: a perfect agent scores one and a wrong prediction loses what a correct
# one earns, so normalised regret on the new lags is twice the share of wrong predictions, 2(1 - success).
REG = {}


def reg_rows(worlds, sels, endpoint, lower_better):
    """Per-world regret of the agent each selector deploys, of random choice and of the best agent in hindsight."""
    import sys
    sys.path.insert(0, str(ROOT))
    from common import anal as A
    out = {s: [] for s in list(sels) + ["pool", "oracle"]}
    for df in worlds.values():
        r = df[endpoint].to_numpy(dtype=float)
        for s, hi in sels.items():
            out[s].append(float(df.loc[A.tied(df, s, hi, 1e-9)][endpoint].mean()))
        out["pool"].append(float(r.mean()))
        out["oracle"].append(float(r.min() if lower_better else r.max()))
    return {k: np.asarray(v) for k, v in out.items()}


if s1p.exists():
    import numpy as np
    import pandas as pd
    W = {}
    for lk in sorted((ROOT / "runs" / "e1").glob("lock*.json")):
        L = json.loads(lk.read_text())
        if L["kind"] != "ds":
            continue
        R = json.loads((lk.parent / ("res" + lk.stem[4:] + ".json")).read_text())
        n = pd.DataFrame(L["scores_n"]).merge(pd.DataFrame(R["enc"]), on="enc")
        keep = n.enc[n.nat_val >= n.nat_val.max() - 0.10]
        q = pd.DataFrame(L["scores_q"]).merge(pd.DataFrame(R["q"]), on="cand")
        W[L["wid"]] = q[q.enc.isin(keep)].reset_index(drop=True)
    REG["e1ds"] = reg_rows(W, {"f_ign": False, "k_delta_set": False, "val": True}, "q_nreg_d", True)
    REG["e1ds"]["k"] = REG["e1ds"].pop("k_delta_set")
if s4p.exists() and s5p.exists():
    import numpy as np
    for run in ("e4", "e5"):
        Wr = rp_load(run)
        for df in Wr.values():
            df["reg_new"] = 2.0 * (1.0 - df["l_new"])
        REG[run] = reg_rows(Wr, {"f_ign": False, "k_known": False, "val": True}, "reg_new", True)
        REG[run]["k"] = REG[run].pop("k_known")
for exp, d in REG.items():
    import sys
    sys.path.insert(0, str(ROOT))
    from common import anal as A
    for s, v in d.items():
        vals[f"{exp}.reg.{s}"] = f(float(v.mean()), 3)
    vals[f"{exp}.reg.n"] = str(len(d["f_ign"]))
    for base in ("k", "val", "pool"):
        r = float(d["f_ign"].sum() / d[base].sum())
        c = A.boot_ratio(d["f_ign"], d[base], 700 + len(exp))
        vals[f"{exp}.regcut.{base}"] = pct(1 - r)
        vals[f"{exp}.regcut.{base}.ci"] = pci([1 - c[1], 1 - c[0]])
        vals[f"{exp}.regcut.{base}.wins"] = str(int(np.sum(d["f_ign"] < d[base] - 1e-12)))
    vals[f"{exp}.regcut"] = vals[f"{exp}.regcut.k"]
if REG:
    def rc(exp, base):
        return f"{vals[f'{exp}.regcut.{base}']} {vals[f'{exp}.regcut.{base}.ci']}"
    T = [r"\begin{table}[t]", r"\centering",
         r"\caption{Normalised regret on the new tasks of the agent that each selector deployed, averaged over replications. The experiments are E1 with new tasks (two new tasks, "
         + vals["e1ds.reg.n"] + r" replications) and Repeat Previous (six new lags, " + vals["e4.reg.n"] + r" replications in each of E4 and E5). "
         + r"Regret on the trained task means regret on the known task with success weights in E1, and regret on lag 4 in E4 and E5. "
         + r"On Repeat Previous, a perfect agent scores one and a wrong prediction loses what a correct one earns. Normalised regret computed from this return is twice the share of wrong predictions. "
         + r"The last two rows contain the reduction in normalised regret when freedom over all tasks replaces the named selector. Each reduction is a ratio of sums over replications, with a 95\% bootstrap interval over replications.}",
         r"\label{tab:reg}", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.3cm}rrr@{}}", r"\toprule",
         r"Selector & E1 new tasks & E4 new lags & E5 new lags \\", r"\midrule"]
    for key, name in (("f_ign", "Freedom over all tasks"), ("k", "Regret on the trained task"), ("val", "Source validation"),
                      ("pool", "Random choice"), ("oracle", "Best agent in hindsight")):
        T.append(f"{name} & " + " & ".join(vals[f"{e}.reg.{key}"] for e in ("e1ds", "e4", "e5")) + r" \\")
    T += [r"\midrule",
          r"Reduction against regret on the trained task & " + " & ".join(rc(e, "k") for e in ("e1ds", "e4", "e5")) + r" \\",
          r"Reduction against random choice & " + " & ".join(rc(e, "pool") for e in ("e1ds", "e4", "e5")) + r" \\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "paper" / "t_reg.tex").write_text("\n".join(T) + "\n")

lines = ["% generated by paper/mknums.py", "\\makeatletter", "\\newcommand{\\R}[1]{\\@ifundefined{num@#1}{\\textbf{??\\detokenize{#1}}}{\\csname num@#1\\endcsname}}"]
for k in sorted(vals):
    lines.append(f"\\expandafter\\def\\csname num@{k}\\endcsname{{{vals[k]}}}")
lines.append("\\makeatother")
OUT.write_text("\n".join(lines) + "\n")
print(f"wrote {len(vals)} values to {OUT}")
