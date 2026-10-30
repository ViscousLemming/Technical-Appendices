"""E1 pilots, rerun with the settings of the original pilot worlds.

  python3 -m e1.pilot --out runs/e1pilot --wid 0 --kind main --q 0.6 --seed 1001 --first
  python3 -m e1.pilot --out runs/e1pilot --wid 1 --kind ds --q 0.6 --seed 2002
  python3 -m e1.pilot --out runs/e1pilot --wid 10 --kind main --q 0.95 --seed 3010
  python3 -m e1.pilot --out runs/e1pilot --wid 11 --kind main --q 0.45 --seed 3011
  python3 -m e1.pilot --out runs/e1pilot --summary

Both pilots trained the main pool with training delays 3-7 and 3-28 only, so 32 encoders per main
world. The 3-14 delays joined the pool after the pilots. --first restores the settings of the first
pilot, a rare-regime share of 0.10 at source and 0.25 at target and target delays of 12, 20 and 28.
The second pilot used the settings of e1/plan2.json. Every world runs its selection stage and then
its evaluation stage with the training settings of e1/plan2.json. --summary writes pilots.json with
the spread of target success after refitting in every pilot world.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from e1 import env as E  # noqa: E402
from e1 import world as W  # noqa: E402
from e1.agent import Cfg  # noqa: E402

FIRST = {"frac_source": 0.10, "frac_target": 0.25, "target_delays": [12, 20, 28]}


def settings(kind: str, first: bool) -> dict:
    w = E.World(0, 0.0, 0.0, np.zeros((3, 4), dtype=np.int64))
    s = {"frac_source": w.frac_source, "frac_target": w.frac_target,
         "target_delays": list(W.asks(E.make_world(1, 0.6), kind)[3]),
         "pool_delays": ["3-7", "3-28"] if kind == "main" else ["3-7"]}
    if first:
        s.update(FIRST)
    return s


def use_first():
    make, asks = E.make_world, W.asks

    def make_first(seed, q_common):
        return dataclasses.replace(make(seed, q_common), frac_source=FIRST["frac_source"],
                                   frac_target=FIRST["frac_target"])

    def asks_first(world, kind):
        a = asks(world, kind)
        return (a[0], a[1], a[2], list(FIRST["target_delays"])) if kind == "main" else a

    E.make_world = make_first
    W.asks = asks_first


def spread(lock: dict, res: dict) -> dict:
    """Spread of target success after refitting a lookup table, over all candidates and over the admitted
    ones, and what source validation deployed against the best candidate in hindsight."""
    n = pd.DataFrame(lock["scores_n"]).merge(pd.DataFrame(res["enc"]), on="enc")
    keep = n.enc[n.nat_val >= n.nat_val.max() - 0.10]
    q = pd.DataFrame(lock["scores_q"]).merge(pd.DataFrame(res["q"]), on="cand")
    qa, na = q[q.enc.isin(keep)], n[n.enc.isin(keep)]
    v, nv = qa.loc[qa.val.idxmax()], na.loc[na.nat_val.idxmax()]
    out = {"candidates": int(len(q)), "admitted": int(len(qa)),
           "spread_all": float(q.q_succ.max() - q.q_succ.min()),
           "spread_admitted": float(qa.q_succ.max() - qa.q_succ.min()),
           "source_validation_q_succ": float(v.q_succ), "best_q_succ": float(qa.q_succ.max()),
           "source_validation_l_succ": float(nv.l_succ), "best_l_succ": float(na.l_succ.max())}
    if lock["kind"] == "main":
        out.update({"source_validation_n_succ": float(nv.n_succ), "best_n_succ": float(na.n_succ.max())})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wid", type=int)
    ap.add_argument("--kind", choices=["main", "ds"])
    ap.add_argument("--q", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--first", action="store_true")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    book = out / "pilots.json"
    rec = json.loads(book.read_text()) if book.exists() else {"worlds": {}}
    if a.summary:
        for wid, w in sorted(rec["worlds"].items(), key=lambda kv: int(kv[0])):
            f = out / f"res{int(wid):03d}.json"
            if f.exists():
                lock = json.loads((out / f"lock{int(wid):03d}.json").read_text())
                w["result"] = spread(lock, json.loads(f.read_text()))
        book.write_text(json.dumps(rec, indent=1))
        print(json.dumps(rec, indent=1))
        return
    torch.set_num_threads(1)
    W.MAIN_DELAYS = ("short", "long")
    rec["worlds"][str(a.wid)] = {"pilot": "first" if a.first else "second", "kind": a.kind, "q_common": a.q,
                                 "seed": a.seed, **settings(a.kind, a.first)}
    book.write_text(json.dumps(rec, indent=1))
    if a.first:
        use_first()
    t = time.time()
    if not (out / f"lock{a.wid:03d}.json").exists():
        W.source_stage("pilot", a.kind, a.q, a.wid, a.seed, Cfg(updates=150), out)
    print("selection stage s", round(time.time() - t, 1), flush=True)
    t = time.time()
    if not (out / f"res{a.wid:03d}.json").exists():
        W.target_stage(out, a.wid)
    print("evaluation stage s", round(time.time() - t, 1), flush=True)


if __name__ == "__main__":
    main()
