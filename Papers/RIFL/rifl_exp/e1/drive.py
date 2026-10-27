"""E1 driver. Three phases with resume.

  python -m e1.drive source --plan e1/plan2.json --out DIR --procs 4 [--part 0/3]
  python -m e1.drive open   --plan e1/plan2.json --out DIR
  python -m e1.drive target --plan e1/plan2.json --out DIR --procs 2

The source phase writes one lock per world. The open phase runs once every lock
exists and records the opening time and the worlds it covers. Only then may the
target phase generate the fresh target episodes and read their hidden classes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path


def worlds(plan: dict) -> list[dict]:
    out, wid = [], 0
    for c in plan["cohorts"]:
        for k in range(c["worlds"]):
            out.append({"wid": wid, "cohort": c["name"], "kind": c["kind"], "q_common": c["q_common"],
                        "seed": int(plan["seed0"] + plan["seed_step"] * wid)})
            wid += 1
    return out


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)


def _source(w: dict, out: str, cfg: dict, deadline=None, score_by=None) -> tuple[int, float, str]:
    try:
        from e1.agent import Cfg
        from e1.world import source_stage
        t = time.time()
        lock = source_stage(w["cohort"], w["kind"], w["q_common"], w["wid"], w["seed"], Cfg(**cfg), Path(out),
                            deadline, score_by)
        return w["wid"], time.time() - t, "" if lock is not None else "partial"
    except Exception:
        return w["wid"], 0.0, traceback.format_exc()


def _target(wid: int, out: str) -> tuple[int, float, str]:
    try:
        from e1.world import target_stage
        t = time.time()
        target_stage(Path(out), wid)
        return wid, time.time() - t, ""
    except Exception:
        return wid, 0.0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["source", "open", "target", "status"])
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--part", default="0/1", help="i/k runs worlds with wid %% k == i")
    ap.add_argument("--only", default="", help="comma list of wids")
    ap.add_argument("--budget", type=float, default=0.0, help="seconds after which no new agent starts (0 = none)")
    args = ap.parse_args()

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ws = worlds(plan)
    i, k = (int(v) for v in args.part.split("/"))
    ws = [w for w in ws if w["wid"] % k == i]
    if args.only:
        keep = {int(v) for v in args.only.split(",")}
        ws = [w for w in ws if w["wid"] in keep]
    log = open(out / f"log_{args.phase}_{i}of{k}.txt", "a")

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    if args.phase == "status":
        locks = sorted(out.glob("lock*.json"))
        res = sorted(out.glob("res*.json"))
        say(f"worlds {len(worlds(plan))} locks {len(locks)} results {len(res)}")
        return

    if args.phase == "source":
        todo = [w for w in ws if not (out / f"lock{w['wid']:03d}.json").exists()
                or not (out / f"state{w['wid']:03d}.pt").exists()]
        say(f"source: {len(todo)} of {len(ws)} worlds to run")
        ctx = get_context("spawn")
        with ProcessPoolExecutor(args.procs, mp_context=ctx, initializer=_init) as ex:
            t0 = time.time()
            dl = t0 + args.budget if args.budget > 0 else None
            sb = t0 + args.budget + 20 if args.budget > 0 else None
            futs = [ex.submit(_source, w, str(out), plan["cfg"], dl, sb) for w in todo]
            for f in as_completed(futs):
                wid, sec, err = f.result()
                if err == "partial":
                    continue
                say(f"source wid {wid:03d} " + (f"done {sec:.0f}s" if not err else "FAILED\n" + err))
        return

    if args.phase == "open":
        all_w = worlds(plan)
        missing = [w["wid"] for w in all_w if not (out / f"lock{w['wid']:03d}.json").exists()]
        if missing:
            say(f"open refused: {len(missing)} locks missing, first {missing[:10]}")
            sys.exit(1)
        if (out / "open.json").exists():
            say("open.json exists, leaving it unchanged")
            return
        rec = {"opened": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "worlds": []}
        for w in all_w:
            lk = json.loads((out / f"lock{w['wid']:03d}.json").read_text())
            assert not lk["target_truth_used"]
            rec["worlds"].append(w["wid"])
        (out / "open.json").write_text(json.dumps(rec, indent=1))
        say(f"opened {len(rec['worlds'])} worlds")
        return

    if args.phase == "target":
        opened = json.loads((out / "open.json").read_text())
        todo = []
        for w in ws:
            assert w["wid"] in opened["worlds"], f"world {w['wid']} was not opened"
            if not (out / f"res{w['wid']:03d}.json").exists():
                todo.append(w["wid"])
        say(f"target: {len(todo)} of {len(ws)} worlds to run")
        ctx = get_context("spawn")
        with ProcessPoolExecutor(args.procs, mp_context=ctx, initializer=_init) as ex:
            futs = [ex.submit(_target, wid, str(out)) for wid in todo]
            for f in as_completed(futs):
                wid, sec, err = f.result()
                say(f"target wid {wid:03d} " + (f"done {sec:.0f}s" if not err else "FAILED\n" + err))


if __name__ == "__main__":
    main()
