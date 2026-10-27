"""E2 pilot, rerun. Each call trains one agent with e2/t_train.py and saves what it prints.

  python3 e2/pilot.py OUT [--oldinit] ARCH SIZE SEED UPDATES [LR ENT]
  python3 e2/pilot.py OUT --summary

--oldinit restores the initialisation that the first two rounds of the pilot used, which set
every layer-norm gain to zero and so froze the transformers. The third round ran after the fix.
The commands of the three rounds, in order:

  round 1, before the fix: gru 32 1 150, lstm 128 2 150, trxl 8 3 150, trxl 32 4 150
  round 2, before the fix: trxl 32 4 400 0.0003 0.01, trxl 32 4 400 0.001 0.003,
                           trxl 8 3 400 0.0003 0.003, gru 8 5 400 0.001 0.01
  round 3, after the fix:  trxl 32 4 300 0.001 0.01, trxl 8 3 300 0.001 0.01,
                           lstm 32 6 300 0.001 0.01, gru 128 7 300 0.001 0.01
"""
import contextlib
import io
import json
import math
import runpy
import sys
import time
from pathlib import Path

import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def old_init(module: nn.Module, actor_last: nn.Linear):
    for name, p in module.named_parameters():
        if "gate_bias" in name or "rel" in name:
            continue
        if p.ndim >= 2:
            nn.init.orthogonal_(p, gain=math.sqrt(2) if "actor" in name or "critic" in name else 1.0)
        else:
            nn.init.zeros_(p)
    nn.init.orthogonal_(actor_last.weight, gain=0.01)


def summary(out: Path):
    """The two pilot conditions over every agent snapshot except the transformers trained before the fix:
    the share with success at least 0.9 at hallway 7, inside the source range 5 to 10, and the spread of
    success at hallway 40."""
    import ast
    rows = []
    for f in sorted(out.glob("*_*.json")):
        d = json.loads(f.read_text())
        if d["args"][0] == "trxl" and d["init"] == "before the fix":
            continue
        t = d["printed"]
        res = ast.literal_eval(t[t.index("{"):])
        for u in sorted({int(k.split("_")[0][1:]) for k in res}):
            rows.append({"run": f.stem, "update": u, "success_L7": res[f"u{u}_L7"][1], "success_L40": res[f"u{u}_L40"][1]})
    s7 = [r["success_L7"] for r in rows]
    s40 = [r["success_L40"] for r in rows]
    rec = {"snapshots": rows, "share_source_success_0.9": sum(x >= 0.9 for x in s7) / len(s7),
           "spread_success_L40": max(s40) - min(s40)}
    (out / "summary.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "snapshots"}, indent=1))


def main():
    args = sys.argv[1:]
    out = Path(args.pop(0))
    if args == ["--summary"]:
        summary(out)
        return
    old = "--oldinit" in args
    args = [x for x in args if x != "--oldinit"]
    if old:
        import e2.agent
        e2.agent._init = old_init
    out.mkdir(parents=True, exist_ok=True)
    name = "_".join(["old" if old else "new"] + args)
    buf = io.StringIO()
    t = time.time()
    sys.argv = [str(ROOT / "e2" / "t_train.py")] + args
    with contextlib.redirect_stdout(buf):
        runpy.run_path(str(ROOT / "e2" / "t_train.py"), run_name="__main__")
    text = buf.getvalue()
    print(text, end="")
    (out / f"{name}.txt").write_text(text)
    (out / f"{name}.json").write_text(json.dumps({"args": args, "init": "before the fix" if old else "after the fix",
                                                  "seconds": round(time.time() - t, 1), "printed": text.strip()}, indent=1))


if __name__ == "__main__":
    main()
