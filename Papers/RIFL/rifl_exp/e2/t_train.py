import sys, time, numpy as np, torch, json
torch.set_num_threads(1)
sys.path.insert(0, '.')
from e2.agent import Spec
from e2.ppo import Cfg, train, load, rollout
arch, size, seed, upd = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
ent = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
sp = Spec(arch, size, seed)
t = time.time()
st, log = train(sp, Cfg(updates=upd, lr=lr, ent=ent), lambda r, n: r.integers(5, 11, size=n), seed + 1, [upd // 2, upd])
tt = time.time() - t
res = {}
for u in (upd // 2, upd):
    net = load(sp, st[u])
    for L in (7, 20, 40, 80):
        ro = rollout(net, np.full(200, L), np.random.default_rng(L).integers(0, 2, 200), seed=L, keep_obs=False)
        res[f"u{u}_L{L}"] = (round(float((ro['arrive'] >= 0).mean()), 2), round(float(ro['success'].mean()), 2))
print(sp.name, lr, ent, 'train s', round(tt, 1), 'curve', log['curve'][::3], res, flush=True)
