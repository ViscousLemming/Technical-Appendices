"""Lookup cost of every native memory on its panel, computed after opening. Usage: python3 e1/kcheck.py (from the package folder)."""
import json, sys, time
import numpy as np, torch
sys.path.insert(0, ".")
from pathlib import Path
from e1 import env as E
from e1 import world as W
from e1.agent import Spec
out = Path("runs/e1")
res = {}
t0 = time.time()
for wid in range(192):
    lock = json.loads((out / f"lock{wid:03d}.json").read_text())
    blob = torch.load(out / f"state{wid:03d}.pt", weights_only=False)
    kind = lock["kind"]
    world = E.make_world(lock["seed"], lock["q_common"])
    ask_src, ask_tgt, d_src, d_tgt = W.asks(world, kind)
    panel = E.make_bank(world, 1600, d_tgt, blob["panel_seed"], world.frac_target, ask_tgt)
    post_pan = E.true_post(world, panel)
    p_all = E.answer_probs(post_pan, ask_tgt)
    y = (p_all >= 0.5).astype(int)
    w = np.abs(2 * p_all - 1)
    r = {"enc": 0, "dup": 0, "k_delta": []}
    for ei, e in enumerate(blob["encoders"]):
        net = W.load(Spec(**e["spec"]), e["state"])
        m = np.asarray(W.run(net, panel)[0], dtype=np.float32)
        u, inv = np.unique(m, axis=0, return_inverse=True)
        r["enc"] += 1
        k = 0.0
        if len(u) < len(m):
            r["dup"] += 1
            for c in range(len(u)):
                idx = np.where(inv == c)[0]
                if len(idx) > 1:
                    for t in range(y.shape[1]):
                        a = w[idx, t][y[idx, t] == 1].sum(); b = w[idx, t][y[idx, t] == 0].sum()
                        k += min(a, b)
        r["k_delta"].append(float(k / len(m)))
    res[wid] = r
    if wid % 16 == 15:
        json.dump(res, open("runs/e1/kcheck.json", "w"))
        print(wid, f"{time.time()-t0:.0f}s", flush=True)
json.dump(res, open("runs/e1/kcheck.json", "w"))
print("done")
