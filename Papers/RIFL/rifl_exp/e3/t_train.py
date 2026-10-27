import sys, time, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '.')
from e2.agent import Spec
from e3.ppo import Cfg, train, load, rollout
arch, size, seed, upd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
ent = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
share = float(sys.argv[7]) if len(sys.argv) > 7 else 0.1
sp = Spec(arch, size, seed)
t = time.time()
st, log = train(sp, Cfg(updates=upd, lr=lr, ent=ent), lambda r, n: r.integers(5, 11, size=n), seed + 1, [upd // 2, upd], regime_share=share)
tt = time.time() - t
res = {}
for u in (upd // 2, upd):
    net = load(sp, st[u])
    for share, Lr in ((0.1, (5, 11)), (0.5, (10, 21))):
        rng = np.random.default_rng(3)
        n = 600
        L = rng.integers(*Lr, size=n); g = rng.integers(0, 2, n); rg = (rng.random(n) < share).astype(int)
        ro = rollout(net, L, g, rg, seed=11, keep_obs=False)
        r1 = rg == 1
        res[f"u{u}_s{share}"] = (round(float(ro['ret'].mean()), 2), round(float(ro['success'][r1].mean()), 2), round(float(ro['success'][~r1].mean()), 2),
                                 round(float(ro['reads_j'][r1].sum(1).mean()), 1), round(float(ro['reads_j'][~r1].sum(1).mean()), 1), round(float((ro['arrive'] >= 0).mean()), 2))
print(sp.name, share, lr, ent, 'train s', round(tt, 1), 'curve', [round(c, 2) for c in log['curve'][::3]], 'ret, succ r1, succ r0, reads r1, reads r0, reach', res, flush=True)
