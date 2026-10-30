"""Black-and-white figures and LaTeX tables from the analysis summaries.

Usage: python3 paper/mkfigs.py  (from the riwf2 folder)
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "paper"
S1 = ROOT / "runs" / "e1" / "e1_summary.json"
S2 = ROOT / "runs" / "e2" / "e2_summary.json"
S3 = ROOT / "runs" / "e3dev" / "e3_summary.json"
s1 = json.loads(S1.read_text()) if S1.exists() else None
s2 = json.loads(S2.read_text())
s3 = json.loads(S3.read_text())
S4 = ROOT / "runs" / "e4" / "e4_summary.json"
S4R = ROOT / "runs" / "e4" / "e4_rank.json"
S5 = ROOT / "runs" / "e5" / "e5_summary.json"
s4 = json.loads(S4.read_text()) if S4.exists() else None
s4r = json.loads(S4R.read_text()) if S4R.exists() else None
s5 = json.loads(S5.read_text()) if S5.exists() else None

plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "hatch.linewidth": 0.5, "pdf.fonttype": 42})

NAMES = {
    "fp3": "Freedom", "kp3": "Freedom, nearest-neighbour rule", "fp4": "Freedom", "fk4": "Freedom, nearest-neighbour rule",
    "lin": "Memory term only", "knn": "Freedom, nearest-neighbour rule", "d_ret": "Memory term only", "d_suc": "Memory term only, success weights",
    "d_one": "Memory term only, unit weights", "acq": "Never-gained term only", "tpe": "Teacher evaluation",
    "mp": "Maximum action probability", "ent": "Action entropy", "atc": "Thresholded confidence", "dis": "Disagreement",
    "dev": "Deep embedded validation", "nat_val": "Source validation", "slin": "Freedom on source histories",
    "k_delta_set": "Freedom, success weights", "k_rho_set": "Freedom, normalised-regret weights", "k_unit_set": "Freedom, unit weights",
    "f_ign": "Freedom over all tasks", "f_unw": "Freedom over all tasks, unit weights", "ll": "Teacher log loss", "val": "Source validation",
    "skd": "Freedom on source histories", "k_deploy": "Freedom on the deployed tasks",
}


def bars(ax, rows, keys, cols, title, names=None, labels=True, xlabel=True):
    """Horizontal grouped bars: one solid (first endpoint) and one hatched (second). Keys may be None for an empty row."""
    y = np.arange(len(keys))[::-1]
    h = 0.38
    styles = [dict(color="black", edgecolor="black"), dict(color="white", edgecolor="black", hatch="////")]
    for j, (c, lab) in enumerate(cols):
        idx = [i for i, k in enumerate(keys) if k is not None and k in rows]
        vals = np.array([100 * rows[keys[i]][f"{c}_gc"] for i in idx])
        lo = np.array([100 * rows[keys[i]][f"{c}_ci"][0] for i in idx])
        hi = np.array([100 * rows[keys[i]][f"{c}_ci"][1] for i in idx])
        yy = y[idx] + (h / 2 if j == 0 else -h / 2)
        ax.barh(yy, vals, height=h * 0.92, linewidth=0.5, label=lab, **styles[j])
        ax.errorbar(vals, yy, xerr=[vals - lo, hi - vals], fmt="none", ecolor="black", elinewidth=0.5, capsize=1.0)
    ax.axvline(0, color="black", linewidth=0.6)
    ax.axvline(100, color="black", linewidth=0.5, linestyle=":")
    ax.set_yticks(y)
    if labels:
        ax.set_yticklabels([(names or {}).get(k, NAMES.get(k, "")) if k else "" for k in keys])
    if xlabel:
        ax.set_xlabel("Share of the gap closed (%)")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(-130, 118)
    ax.set_xticks([-100, 0, 100])
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


# Figure 1: one row per kind of selector, one panel per environment, shared row labels
ROWS = ["Freedom (proposed)", "Freedom, nearest-neighbour rule", "Memory term only", "Never-gained term only",
        "Teacher evaluation", "Source validation", "Freedom on source histories", "Deep embedded validation",
        "Disagreement", "Thresholded confidence", "Action entropy", "Maximum action probability"]
t2 = {r["sel"]: r for r in s2["table"]}
t3 = {r["sel"]: r for r in s3["table"]}
panels = []
if s1:
    t1 = {r["sel"]: r for r in s1["main_table"] if r["level"] == "encoder"}
    panels.append((t1, ["lin", "knn", None, None, "tpe", "nat_val", "slin", "dev", "dis", "atc", "ent", "mp"],
                   [("l_succ", "refitted decision rule"), ("n_succ", "unchanged")], "E1, longer delays"))
panels.append((t2, ["fp3", "kp3", "lin", None, "tpe", "nat_val", "slin", "dev", "dis", "atc", "ent", "mp"],
               [("l_succ", "refitted decision rule"), ("n_succ", "unchanged")], "E2, longer hallways"))
panels.append((t3, ["fp4", "fk4", "d_ret", "acq", "tpe", "nat_val", "slin", "dev", "dis", "atc", "ent", "mp"],
               [("l_ret", "refitted decision rule"), ("n_ret", "unchanged")], "E3, more to sense"))
fig, axes = plt.subplots(1, len(panels), figsize=(6.0, 3.3), constrained_layout=True, sharey=True)
for j, (ax, (rows, keys, cols, title)) in enumerate(zip(np.atleast_1d(axes), panels)):
    bars(ax, rows, keys, cols, title, {k: r for k, r in zip(keys, ROWS) if k}, labels=(j == 0), xlabel=False)
np.atleast_1d(axes)[0].set_yticklabels(ROWS)
hnd, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(hnd, lab, loc="outside upper center", ncol=2, frameon=False, fontsize=7.5)
fig.supxlabel("Share of the gap closed (%)", fontsize=8)
fig.savefig(P / "fig_gc.pdf")
plt.close(fig)

# online validation curves
curves = []
if s1:
    curves.append(("E1, refitted table", s1["H8"]["q_succ"], 100 * [r for r in s1["main_table"] if r["sel"] == "k_delta_set"][0]["q_succ_gc"],
                   100 * [r for r in s1["main_table"] if r["sel"] == "val"][0]["q_succ_gc"]))
curves.append(("E2, refitted decision rule", s2["H8"]["l_succ"], 100 * t2["fp3"]["l_succ_gc"], 100 * t2["nat_val"]["l_succ_gc"]))
curves.append(("E3, refitted decision rule", s3["H8"]["l_ret"], 100 * t3["fp4"]["l_ret_gc"], 100 * t3["nat_val"]["l_ret_gc"]))
fig, axes = plt.subplots(1, len(curves), figsize=(6.0, 2.35), constrained_layout=True, sharey=True)
for ax, (title, cur, zr, sv) in zip(np.atleast_1d(axes), curves):
    ns = sorted(int(k) for k in cur)
    gc = [100 * cur[str(n)]["gc"] for n in ns]
    lo = [100 * cur[str(n)]["ci"][0] for n in ns]
    hi = [100 * cur[str(n)]["ci"][1] for n in ns]
    ax.fill_between(ns, lo, hi, color="0.85", linewidth=0)
    ax.plot(ns, gc, color="black", linestyle="-", marker="o", markersize=3, linewidth=1.0, label="Online validation")
    ax.axhline(zr, color="black", linestyle="--", linewidth=1.0, label="Freedom, no rewards")
    ax.axhline(sv, color="black", linestyle=":", linewidth=1.2, label="Source validation")
    ax.set_xscale("log", base=2)
    ticks = [n for n in ns if n in (1, 4, 16, 64, 256)]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(n) for n in ticks])
    ax.minorticks_off()
    ax.set_title(title, fontsize=8.5)
    ax.set_ylim(-5, 105)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("Share of the gap closed (%)")
fig.supxlabel("Rewarded target episodes per candidate", fontsize=8)
hnd, lab = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(hnd, lab, loc="outside upper center", ncol=3, frameon=False, fontsize=7.5)
fig.savefig(P / "fig_online.pdf")
plt.close(fig)

# E3: readings taken by the agents each selector deploys. The main figure shows the P4 floor and its two ablations,
# and the appendix figure every selector.
def reads_fig(keys, path, size):
    fig, ax = plt.subplots(figsize=size, constrained_layout=True)
    x = [t3[k]["reads1"] for k in keys]
    yv = [t3[k]["l_ret"] for k in keys]
    ax.scatter(x, yv, s=18, facecolors="white", edgecolors="black", linewidths=0.8, zorder=3)
    for k, a, b in zip(keys, x, yv):
        ax.annotate(NAMES[k], (a, b), textcoords="offset points", xytext=(4, -3), fontsize=6.5)
    ax.axhline(s3["oracle_l_ret"], color="black", linestyle="--", linewidth=0.8)
    ax.axhline(s3["pool_l_ret"], color="black", linestyle=":", linewidth=0.9)
    ax.text(12.9, s3["oracle_l_ret"] + 0.04, "best agent in hindsight", fontsize=6.5, ha="right")
    ax.text(12.9, s3["pool_l_ret"] + 0.04, "random choice", fontsize=6.5, ha="right")
    ax.set_xlabel("Samples of the cue in the risky regime before walking")
    ax.set_ylabel("Target return after refitting")
    ax.set_xlim(0, 13)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(P / path)
    plt.close(fig)


reads_fig(["fp4", "acq", "d_ret"], "fig_reads.pdf", (4.2, 2.4))
reads_fig(["fp4", "acq", "tpe", "nat_val", "dev", "dis", "d_one", "d_ret", "mp"], "fig_readf.pdf", (4.2, 2.7))
# E4: refit success at each lag of the agents freedom and regret on lag 4 deployed
if s4:
    fig, ax = plt.subplots(figsize=(3.6, 2.4), constrained_layout=True)
    lags = np.arange(1, 17)
    for lo, hi in ((1.5, 3.5), (4.5, 8.5)):
        ax.axvspan(lo, hi, color="0.9", linewidth=0)
    for sel, ls, mk, lab in (("f_ign", "-", "o", "Freedom over all tasks"), ("k_known", "--", "s", "Regret on lag 4")):
        ax.plot(lags, s4["curves"][sel], color="black", linestyle=ls, marker=mk, markersize=2.8, linewidth=1.0,
                markerfacecolor="black" if sel == "f_ign" else "white", label=lab)
    ax.axhline(0.25, color="black", linestyle=":", linewidth=0.7)
    ax.text(11, 0.212, "guessing", fontsize=6.5, ha="center")
    ax.text(6.5, 0.30, "deployed lags", fontsize=6.5, ha="center")
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(0.2, 1.05)
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 12, 16])
    ax.set_xticklabels(["1", "2", "3", "4\ntrained", "5", "6", "7", "8", "12", "16"])
    ax.set_xlabel("Lag $k$, the $k$th most recent card")
    ax.set_ylabel("Share of correct predictions")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5, handlelength=1.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.savefig(P / "fig_lag.pdf")
    plt.close(fig)

# E4 and E5: how well each selector ranked the whole admitted pool
if s4 and s5 and s4r:
    fig, bx = plt.subplots(figsize=(3.8, 2.6), constrained_layout=True)
    order = ["f_ign", "k_known", "val", "dev", "dis", "atc", "ent", "mp", "params"]
    names = {"f_ign": "Freedom over all tasks", "k_known": "Regret on lag 4", "val": "Source validation",
             "dev": "Deep embedded validation", "dis": "Disagreement", "atc": "Thresholded confidence",
             "ent": "Action entropy", "mp": "Maximum action probability", "params": "Network size"}
    y = np.arange(len(order))[::-1]
    for j, (src, off, fc, lab) in enumerate(((s4r, 0.16, "white", "E4, exploratory"), (s5, -0.16, "black", "E5, confirmatory"))):
        m = [src["tau"]["admitted"]["l_new"][k]["mean"] for k in order]
        lo = [src["tau"]["admitted"]["l_new"][k]["ci"][0] for k in order]
        hi = [src["tau"]["admitted"]["l_new"][k]["ci"][1] for k in order]
        bx.errorbar(m, y + off, xerr=[np.array(m) - lo, np.array(hi) - m], fmt="o", color="black", mfc=fc, ms=3.2,
                    elinewidth=0.6, capsize=1.2, label=lab)
    bx.axvline(0, color="black", linewidth=0.5, linestyle=":")
    bx.set_yticks(y)
    bx.set_yticklabels([names[k] for k in order])
    bx.set_xlim(-0.1, 1.0)
    bx.set_xlabel("Rank correlation with success\non the new lags")
    bx.legend(loc="lower right", frameon=False, fontsize=6.5)
    for sp in ("top", "right"):
        bx.spines[sp].set_visible(False)
    fig.savefig(P / "fig_tau.pdf")
    plt.close(fig)

# Summary figure: gap closed after refitting, one panel per experiment, a few selectors each
if s1 and s4 and s5:
    t1n = {r["sel"]: r for r in s1["main_table"] if r["level"] == "encoder"}
    t1d = {r["sel"]: r for r in s1["ds_table"]}
    t4 = {r["sel"]: r for r in s4["table"]}
    t5 = {r["sel"]: r for r in s5["pick"]["table"]}
    top = ["Freedom", "Teacher evaluation", "Source validation", "Maximum action probability"]
    bot = ["Freedom over all tasks", "Regret on the trained task", "Source validation"]
    grid = [[(t1n, ["lin", "tpe", "nat_val", "mp"], "l_succ", "E1, longer delays"),
             (t2, ["fp3", "tpe", "nat_val", "mp"], "l_succ", "E2, longer hallways"),
             (t3, ["fp4", "tpe", "nat_val", "mp"], "l_ret", "E3, more to sense")],
            [(t1d, ["f_ign", "k_delta_set", "val"], "q_succ", "E1, new tasks"),
             (t4, ["f_ign", "k_known", "val"], "l_new", "E4, new lags"),
             (t5, ["f_ign", "k_known", "val"], "l_new", "E5, new lags")]]
    fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.4), constrained_layout=True, gridspec_kw={"height_ratios": [4, 3]})
    for i, row in enumerate(grid):
        labels = top if i == 0 else bot
        for j, (rows, keys, e, title) in enumerate(row):
            ax = axes[i, j]
            y = np.arange(len(keys))[::-1]
            vals = np.array([100 * rows[k][f"{e}_gc"] for k in keys])
            lo = np.array([100 * rows[k][f"{e}_ci"][0] for k in keys])
            hi = np.array([100 * rows[k][f"{e}_ci"][1] for k in keys])
            ax.barh(y, vals, height=0.62, color="black", edgecolor="black", linewidth=0.5)
            ax.errorbar(vals, y, xerr=[vals - lo, hi - vals], fmt="none", ecolor="0.45", elinewidth=0.6, capsize=1.2)
            ax.axvline(0, color="black", linewidth=0.6)
            ax.axvline(100, color="black", linewidth=0.5, linestyle=":")
            ax.set_yticks(y)
            ax.set_yticklabels(labels if j == 0 else [""] * len(keys))
            ax.set_title(title, fontsize=7.5)
            ax.set_xlim(-108, 112)
            ax.set_xticks([-100, 0, 100])
            ax.tick_params(axis="x", labelsize=7)
            if i == 1:
                ax.set_xlabel("Share of the gap closed (%)", fontsize=7.5)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
    fig.savefig(P / "fig_sum.pdf")
    plt.close(fig)
print("figures written")


# ------------------------------------------------------------------ tables
def sg(x, d):
    t = f"{x:.{d}f}"
    if float(t) == 0:
        t = t.lstrip("-")
    return "$-$" + t[1:] if t.startswith("-") else t


def cell(r, e, d):
    if r is None or e not in r:
        return "--"
    return f"{sg(r[e], d)} & {sg(100 * r[e + '_gc'], 0)}"


def e23_table():
    keys = [("fp3", "fp4", "Freedom"), ("kp3", "fk4", "Freedom, nearest-neighbour rule"), ("lin", "d_ret", "Memory term only"),
            (None, "acq", "Never-gained term only"), (None, "d_one", "Memory term only, unit weights"),
            ("tpe", "tpe", "Teacher evaluation"), ("nat_val", "nat_val", "Source validation"),
            ("slin", "slin", "Freedom on source histories"), ("dev", "dev", "Deep embedded validation"), ("dis", "dis", "Disagreement"),
            ("atc", "atc", "Thresholded confidence"), ("ent", "ent", "Action entropy"), ("mp", "mp", "Maximum action probability")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{E2 and E3, over "
         + f"{s2['worlds']} T-Maze replications and {s3['worlds']} noisy T-Maze replications. "
         + r"Mean target performance of the deployed agent, and the share of the gap closed by each selector. The gap runs from random choice to the best agent in hindsight. Performance is success in E2 and return in E3. Freedom is the freedom selector of each experiment (\Cref{sec:estimate}).}",
         r"\label{tab:e23}", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{4}{c}{E2 T-Maze} & \multicolumn{4}{c}{E3 noisy T-Maze} \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         r"Selector & \multicolumn{2}{c}{Refitted} & \multicolumn{2}{c}{Unchanged} & \multicolumn{2}{c}{Refitted} & \multicolumn{2}{c}{Unchanged} \\",
         r" & success & \% & success & \% & return & \% & return & \% \\", r"\midrule"]
    for k2, k3, name in keys:
        r2 = t2.get(k2) if k2 else None
        r3 = t3.get(k3) if k3 else None
        L.append(f"{name} & {cell(r2, 'l_succ', 3)} & {cell(r2, 'n_succ', 3)} & {cell(r3, 'l_ret', 2)} & {cell(r3, 'n_ret', 2)} \\\\".replace("-- &", "-- & --") if False else
                 f"{name} & {cell(r2, 'l_succ', 3) if r2 else '-- & --'} & {cell(r2, 'n_succ', 3) if r2 else '-- & --'} & {cell(r3, 'l_ret', 2) if r3 else '-- & --'} & {cell(r3, 'n_ret', 2) if r3 else '-- & --'} \\\\")
    L += [r"\midrule",
          f"Random choice & {s2['pool_l_succ']:.3f} & 0 & {s2['pool_n_succ']:.3f} & 0 & {s3['pool_l_ret']:.2f} & 0 & {s3['pool_n_ret']:.2f} & 0 \\\\",
          f"Best agent in hindsight & {s2['oracle_l_succ']:.3f} & 100 & {s2['oracle_n_succ']:.3f} & 100 & {s3['oracle_l_ret']:.2f} & 100 & {s3['oracle_n_ret']:.2f} & 100 \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


def e1_tables():
    if not s1:
        return ""
    q = {r["sel"]: r for r in s1["main_table"] if r["level"] == "quantised"}
    n = {r["sel"]: r for r in s1["main_table"] if r["level"] == "encoder"}
    qkeys = [("k_delta_set", "Freedom, success weights"), ("k_rho_set", "Freedom, normalised-regret weights"), ("k_unit_set", "Freedom, unit weights"),
             ("f_ign", "Freedom over all tasks"), ("k_delta_seq", "Freedom, GRU teacher"), ("k_delta_an", "Freedom, analytic posterior"),
             ("ll", "Teacher log loss"), ("skd", "Freedom on source histories"), ("val", "Source validation")]
    nkeys = [("lin", "Freedom, linear decision rule"), ("knn", "Freedom, nearest-neighbour rule"), ("tpe", "Teacher evaluation"), ("nat_val", "Source validation"),
             ("slin", "Freedom on source histories"), ("dev", "Deep embedded validation"), ("dis", "Disagreement"), ("atc", "Thresholded confidence"),
             ("ent", "Action entropy"), ("mp", "Maximum action probability")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{E1 with longer delays, " + f"{s1['worlds_main']}" + r" replications. Target success of the deployed agent, and the share of the gap closed by each selector. The gap runs from random choice to the best agent in hindsight. For memories rounded to six states, the predictions come from a table with one entry per state, refitted on the target sample. For the agents' own continuous memories, the predictions come from a refitted linear layer or from the agent's own policy.}",
         r"\label{tab:e1}", r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Six-state memories & \multicolumn{2}{c}{Refitted table} & & & & \\", r"\midrule"]
    for k, name in qkeys:
        r = q.get(k)
        if r:
            L.append(f"{name} & {r['q_succ']:.3f} & {sg(100 * r['q_succ_gc'], 0)} & & & & \\\\")
    L.append(f"Random choice & {s1['main_pool_q_succ']:.3f} & 0 & & & & \\\\")
    L.append(f"Best agent in hindsight & {s1['main_oracle_q_succ']:.3f} & 100 & & & & \\\\")
    L += [r"\midrule", r"Continuous memories & \multicolumn{2}{c}{Refitted linear} & \multicolumn{2}{c}{Unchanged} & & \\", r"\midrule"]
    for k, name in nkeys:
        r = n.get(k)
        if r:
            L.append(f"{name} & {r['l_succ']:.3f} & {sg(100 * r['l_succ_gc'], 0)} & {r['n_succ']:.3f} & {sg(100 * r['n_succ_gc'], 0)} & & \\\\")
    L.append(f"Random choice & {s1['main_pool_l_succ']:.3f} & 0 & {s1['main_pool_n_succ']:.3f} & 0 & & \\\\")
    L.append(f"Best agent in hindsight & {s1['main_oracle_l_succ']:.3f} & 100 & {s1['main_oracle_n_succ']:.3f} & 100 & & \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    d = {r["sel"]: r for r in s1["ds_table"]}
    dkeys = [("f_ign", "Freedom over all tasks"), ("f_unw", "Freedom over all tasks, unit weights"), ("k_delta_set", "Regret on the known task, success weights"),
             ("k_rho_set", "Regret on the known task, normalised-regret weights"), ("k_unit_set", "Regret on the known task, unit weights"), ("ll", "Teacher log loss"),
             ("skd", "Regret on the known task, source histories"), ("val", "Source validation"), ("k_deploy", "Freedom on the deployed tasks (reference)")]
    L += [r"\begin{table}[t]", r"\centering",
          r"\caption{E1 with new tasks, " + f"{s1['worlds_ds']}" + r" replications. Memories are chosen with one task known. Deployment includes that task and two new tasks, and a table of predictions is refitted once all three are named. The reference row is a selector computed with the deployed tasks.}",
          r"\label{tab:ds}", r"\small", r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
          r"Selector & \multicolumn{2}{c}{New tasks} & \multicolumn{2}{c}{Known task} \\", r"\midrule"]
    for k, name in dkeys:
        r = d.get(k)
        if r:
            L.append(f"{name} & {r['q_succ']:.3f} & {sg(100 * r['q_succ_gc'], 0)} & {r['q_succ_known']:.3f} & {sg(100 * r['q_succ_known_gc'], 0)} \\\\")
    L.append(f"Random choice & {s1['ds_pool_q_succ']:.3f} & 0 & {s1['ds_pool_q_succ_known']:.3f} & 0 \\\\")
    L.append(f"Best agent in hindsight & {s1['ds_oracle_q_succ']:.3f} & 100 & {s1['ds_oracle_q_succ_known']:.3f} & 100 \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L)


(P / "t_main.tex").write_text(e1_tables() + "\n\n" + e23_table() + "\n")
print("tables written")


def prow(env, label, d, digits=3, holm=None):
    ci = d["ci"]
    pv = d.get("p_holm", d["p"]) if holm is None else holm
    def pf(x):
        if x < 1e-3:
            import math
            e = math.floor(math.log10(x)); m = round(x / 10 ** e, 1)
            if m >= 10:
                m, e = m / 10, e + 1
            return f"${m:.1f}\\times10^{{{e}}}$"
        return f"{x:.3f}"
    return (f"{env} & {label} & {sg(d['mean'], digits)} & [{sg(ci[0], digits)}, {sg(ci[1], digits)}] & "
            f"{d['wins']}/{d['ties']}/{d['losses']} & {pf(pv)} \\\\")


def paired_table():
    R = [r"\begin{table}[t]", r"\centering",
         r"\caption{Paired comparisons within replications. Each row contains the mean difference within replications in the deployed agent's target outcome, its 95\% bootstrap interval, the numbers of replications won, tied and lost, and the one-sided sign-flip $p$ value. An asterisk marks a $p$ value that is Holm-adjusted within the environment's group of confirmatory tests. E1 new is the part of E1 with new tasks, and freedom is the freedom selector of each experiment.}",
         r"\label{tab:paired}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}l>{\raggedright\arraybackslash}p{6.4cm}rlrr@{}}", r"\toprule",
         r"Env. & Comparison & Mean & 95\% interval & W/T/L & $p$ \\", r"\midrule"]
    if s1:
        h1 = s1["H1"]
        R.append(prow("E1 new", "Freedom $-$ regret on the known task, new tasks", h1["new_f_ign_minus_k"], holm=s1["holm"].get("H1 new")) .replace("new tasks &", "new tasks$^*$ &"))
        R.append(prow("E1 new", "Regret on the known task $-$ freedom, known task", h1["known_k_minus_f_ign"]))
        for k, lab in [("q:k_delta_set>val", "Freedom, table $-$ source validation"), ("q:k_delta_set>skd", "Freedom, table $-$ freedom on source histories"),
                       ("q:k_delta_set>ll", "Freedom, table $-$ teacher log loss"), ("q:k_delta_set>k_unit_set", "Freedom, table, success $-$ unit weights"),
                       ("l:lin>nat_val", "Freedom, linear $-$ source validation"), ("l:lin>mp", "Freedom, linear $-$ maximum action probability"),
                       ("l:lin>ent", "Freedom, linear $-$ action entropy"), ("l:lin>atc", "Freedom, linear $-$ thresholded confidence"),
                       ("l:lin>dis", "Freedom, linear $-$ disagreement"), ("l:lin>dev", "Freedom, linear $-$ deep embedded validation"),
                       ("l:lin>slin", "Freedom, linear $-$ freedom on source histories")]:
            R.append(prow("E1", lab + "$^*$", s1["H2"][k], holm=s1["holm"][f"H2 {k}"]))
        R.append(prow("E1", "Teacher evaluation $-$ freedom, linear, unchanged", s1["H4"]["tpe_minus_lin_asis"]))
        R.append(prow("E1", "Freedom, linear $-$ teacher evaluation, refitted", s1["H4"]["lin_minus_tpe_refit"]))
        R.append(r"\midrule")
    for k, lab in [("nat_val", "source validation"), ("mp", "maximum action probability"), ("ent", "action entropy"), ("atc", "thresholded confidence"),
                   ("dis", "disagreement"), ("dev", "deep embedded validation")]:
        R.append(prow("E2", f"Freedom $-$ {lab}$^*$", s2["H2"][k]))
    R.append(prow("E2", "Teacher evaluation $-$ freedom, unchanged", s2["H4"]["tpe_minus_fp3_asis"]))
    R.append(prow("E2", "Freedom $-$ teacher evaluation, refitted", s2["H4"]["fp3_minus_tpe_refit"]))
    R.append(r"\midrule")
    R.append(prow("E3", "Freedom $-$ memory term only", s3["H10"]["fp4_minus_d_ret"], 2))
    R.append(prow("E3", "Freedom $-$ never-gained term only", s3["H10"]["fp4_minus_acq"], 2))
    for k, lab in [("nat_val", "source validation"), ("mp", "maximum action probability"), ("ent", "action entropy"), ("atc", "thresholded confidence"),
                   ("dis", "disagreement"), ("dev", "deep embedded validation")]:
        R.append(prow("E3", f"Freedom $-$ {lab}$^*$", s3["H2"][k], 2))
    R.append(prow("E3", "Memory term only, return $-$ success weights", s3["H11"]["d_ret_minus_d_suc"], 2))
    R.append(prow("E3", "Memory term only, return $-$ unit weights", s3["H11"]["d_ret_minus_d_one"], 2))
    R.append(prow("E3", "Teacher evaluation $-$ freedom, unchanged", s3["H4"]["tpe_minus_fp4_asis"], 2))
    R.append(prow("E3", "Freedom $-$ teacher evaluation, refitted", s3["H4"]["fp4_minus_tpe_refit"], 2))
    R += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(R)


(P / "t_paired.tex").write_text(paired_table() + "\n")


def ties_table():
    """Gap closures that move by at least 2 points between the lowest-index rule and equal chance among tied candidates."""
    first = {}
    for key, path in (("e1", S1), ("e2", S2), ("e3", S3)):
        fp = path.with_name(path.stem + "_first.json")
        first[key] = json.loads(fp.read_text()) if fp.exists() else None
    if not all(first.values()):
        return ""
    lab = dict(NAMES)
    lab.update({"lin": "Freedom, linear decision rule", "k_delta_seq": "Freedom, GRU teacher", "k_delta_an": "Freedom, analytic posterior"})
    labnew = dict(lab)
    labnew.update({"k_delta_set": "Regret on the known task, success weights", "k_rho_set": "Regret on the known task, normalised-regret weights",
                   "k_unit_set": "Regret on the known task, unit weights", "skd": "Regret on the known task, source histories"})
    # E2 and E3 rows as Table 4 names them, and only the selectors Table 4 shows
    t4 = {"e2": {}, "e3": {}}
    for k2, k3, name in [("fp3", "fp4", "Freedom"), ("kp3", "fk4", "Freedom, nearest-neighbour rule"), ("lin", "d_ret", "Memory term only"),
                         (None, "acq", "Never-gained term only"), (None, "d_one", "Memory term only, unit weights"), ("tpe", "tpe", "Teacher evaluation"),
                         ("nat_val", "nat_val", "Source validation"), ("slin", "slin", "Freedom on source histories"), ("dev", "dev", "Deep embedded validation"),
                         ("dis", "dis", "Disagreement"), ("atc", "atc", "Thresholded confidence"), ("ent", "ent", "Action entropy"),
                         ("mp", "mp", "Maximum action probability")]:
        if k2:
            t4["e2"][k2] = name
        t4["e3"][k3] = name
    blocks = []
    if s1:
        blocks.append(("E1", "refitted table", [r for r in s1["main_table"] if r["level"] == "quantised"],
                       [r for r in first["e1"]["main_table"] if r["level"] == "quantised"], "q_succ"))
        blocks.append(("E1", "refitted linear", [r for r in s1["main_table"] if r["level"] == "encoder"],
                       [r for r in first["e1"]["main_table"] if r["level"] == "encoder"], "l_succ"))
        blocks.append(("E1", "unchanged", [r for r in s1["main_table"] if r["level"] == "encoder"],
                       [r for r in first["e1"]["main_table"] if r["level"] == "encoder"], "n_succ"))
        blocks.append(("E1 new", "new tasks", s1["ds_table"], first["e1"]["ds_table"], "q_succ"))
        blocks.append(("E1 new", "known task", s1["ds_table"], first["e1"]["ds_table"], "q_succ_known"))
    blocks += [("E2", "refitted", s2["table"], first["e2"]["table"], "l_succ"), ("E2", "unchanged", s2["table"], first["e2"]["table"], "n_succ"),
               ("E3", "refitted", s3["table"], first["e3"]["table"], "l_ret"), ("E3", "unchanged", s3["table"], first["e3"]["table"], "n_ret")]
    R = [r"\begin{table}[t]", r"\centering",
         r"\caption{Shares of the gap closed that move by at least two points between the two tie rules. Equal chance is the rule of the text and tables, and under it each tied candidate is deployed with equal probability. Under lowest index, the tied candidate listed first is deployed.}",
         r"\label{tab:ties}", r"\footnotesize", r"\begin{tabular}{@{}ll>{\raggedright\arraybackslash}p{6.4cm}rr@{}}", r"\toprule",
         r"Env. & Deployment & Selector & Equal chance & Lowest index \\", r"\midrule"]
    n = 0
    for env, dep, new, old, e in blocks:
        o = {r["sel"]: r for r in old}
        for r in new:
            if r["sel"] not in o or f"{e}_gc" not in r or r["sel"] == "k_deploy":
                continue
            names = t4.get(env.lower(), labnew if env == "E1 new" else lab)
            if env in ("E2", "E3") and r["sel"] not in names:
                continue
            a, b = 100 * r[f"{e}_gc"], 100 * o[r["sel"]][f"{e}_gc"]
            if abs(round(a) - round(b)) >= 2:
                R.append(f"{env} & {dep} & {names.get(r['sel'], r['sel'])} & {sg(a, 0)}\\% & {sg(b, 0)}\\% \\\\")
                n += 1
    R += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    print("tie rows", n)
    return "\n".join(R)


(P / "t_ties.tex").write_text(ties_table() + "\n")


def rp_tables():
    """Repeat Previous: E4 and E5 picks and rankings, and the two confirmatory families."""
    if not (s4 and s5 and s4r):
        return ""
    t4 = {r["sel"]: r for r in s4["table"]}
    t5 = {r["sel"]: r for r in s5["pick"]["table"]}
    tau4 = s4r["tau"]["admitted"]["l_new"]
    tau5 = s5["tau"]["admitted"]["l_new"]
    rows = [("f_ign", "Freedom over all tasks"), ("k_known", "Regret on lag 4"), ("val", "Source validation"),
            ("dev", "Deep embedded validation"), ("dis", "Disagreement"), ("atc", "Thresholded confidence"),
            ("ent", "Action entropy"), ("mp", "Maximum action probability"), ("params", "Network size"), None,
            ("skd", "Regret on lag 4, source histories"), ("f_ign16", "Freedom, lags 1 to 16"), ("f_ign8", "Freedom, lags 1 to 8"),
            ("k_deploy", "Freedom on the deployed lags (reference)")]
    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Repeat Previous, " + f"{s4['worlds']}" + r" replications in E4 and " + f"{s5['worlds']}" + r" in E5. For each selector, the table contains the success after refitting on the six new lags of the agent it deployed. It also contains the share of the gap closed and the mean rank correlation $\tau$ between the selector's score and the candidates' success on the new lags. The gap runs from random choice to the best agent in hindsight. The ranking is exploratory in E4 and confirmatory in E5. Selectors below the first horizontal line are outside the groups of confirmatory tests. Freedom on the deployed lags is a reference that uses the deployed lags.}",
         r"\label{tab:rp}", r"\small", r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r" & \multicolumn{3}{c}{E4} & \multicolumn{3}{c}{E5} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"Selector & success & \% & $\tau$ & success & \% & $\tau$ \\", r"\midrule"]
    for r in rows:
        if r is None:
            L.append(r"\midrule")
            continue
        k, name = r
        a, b = t4[k], t5[k]
        L.append(f"{name} & {a['l_new']:.3f} & {sg(100 * a['l_new_gc'], 0)} & {sg(tau4[k]['mean'], 2)} & "
                 f"{b['l_new']:.3f} & {sg(100 * b['l_new_gc'], 0)} & {sg(tau5[k]['mean'], 2)} \\\\")
    p5 = json.loads((ROOT / "runs" / "e5" / "e5_pick.json").read_text())
    L += [r"\midrule",
          f"Random choice & {s4['pool_l_new']:.3f} & 0 & & {p5['pool_l_new']:.3f} & 0 & \\\\",
          f"Best agent in hindsight & {s4['oracle_l_new']:.3f} & 100 & & {p5['oracle_l_new']:.3f} & 100 & \\\\",
          r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    first = "\n".join(L)
    L = []

    def pv(x):
        return prow("", "", {"mean": 0, "ci": [0, 0], "wins": 0, "ties": 0, "losses": 0, "p": x}, holm=x).split("&")[-1].replace("\\\\", "").strip()

    comp = [("k_known", "regret on lag 4"), ("val", "source validation"), ("dev", "deep embedded validation"),
            ("dis", "disagreement"), ("atc", "thresholded confidence"), ("ent", "action entropy"),
            ("mp", "maximum action probability"), ("params", "network size")]
    L += [r"\begin{table}[t]", r"\centering",
          r"\caption{Confirmatory tests on Repeat Previous, within replications. In E4, each test compares the success after refitting on the new lags of two agents, the one deployed by freedom over all tasks and the one deployed by the named selector. In E5, each test compares the rank correlations $\tau$ of the two selectors' scores with the candidates' success on the new lags. Each entry contains the mean difference within replications, the numbers of replications won, tied and lost, and the one-sided sign-flip $p$ value. Each difference is freedom minus the named selector, and each $p$ value is Holm-adjusted within the experiment's group of sixteen tests.}",
          r"\label{tab:rppair}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
          r" & & \multicolumn{3}{c}{E4, deployed agent} & \multicolumn{3}{c}{E5, ranking} \\", r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
          r"Pool & Freedom $-$ & Mean & W/T/L & $p$ & Mean & W/T/L & $p$ \\", r"\midrule"]
    T5 = s5["family"]["tests"]
    for pool, lab, h1, h2 in (("admitted", "Whole pool", s4["H1"], s4["H2"]), ("nowin4", "Without four-card", s4["nowin4"]["H1"], s4["nowin4"]["H2"])):
        for j, (k, name) in enumerate(comp):
            a = h1["new_f_ign_minus_k"] if k == "k_known" else h2[k]
            b = T5[f"{pool}: f_ign - {k}"]
            L.append(f"{lab if j == 0 else ''} & {name} & {sg(a['mean'], 3)} & {a['wins']}/{a['ties']}/{a['losses']} & {pv(a['p_holm'])} & "
                     f"{sg(b['mean'], 2)} & {b['wins']}/{b['ties']}/{b['losses']} & {pv(b['p_holm'])} \\\\")
        if pool == "admitted":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return first, "\n".join(L)


if s4 and s5 and s4r:
    _a, _b = rp_tables()
    (P / "t_rp.tex").write_text(_a + "\n")
    (P / "t_rpp.tex").write_text(_b + "\n")
print("paired table written")
