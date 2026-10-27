"""RockSample pilot of E3: train one agent on RockSample(5,5) and report what it learned.

  python3 e3/t_rocks.py ARCH SIZE SEED BUDGET [LR ENT UPDATES]

Training stops after BUDGET seconds, checked every 20 updates, or after UPDATES updates (5000 by default).
The printed curve is the mean training return over each block of 20 updates. The evaluation runs 400
episodes from the source start columns 0 and 1 and 400 from the target start columns 2 and 3, and
reports the mean return, checks per episode, samples per episode and visits to rock cells.
"""
import sys, time, numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, '.')
from e2.agent import Spec
from e3.rppo import Cfg, train, load, rollout, starts
from e3.rocks import rock_map, ROCKS
arch, size, seed, budget = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4])
lr = float(sys.argv[5]) if len(sys.argv) > 5 else 1e-3
ent = float(sys.argv[6]) if len(sys.argv) > 6 else 0.01
updates = int(sys.argv[7]) if len(sys.argv) > 7 else 5000
rocks = rock_map(1)
sp = Spec(arch, size, seed)
t = time.time()
st, log = train(sp, Cfg(updates=updates, lr=lr, ent=ent), rocks, [0, 1], seed + 1, [100000], deadline=t + budget)
net = load(sp, st['final'])
res = {}
for nm, cols in (('src', [0, 1]), ('tgt', [2, 3])):
    rng = np.random.default_rng(5)
    ro = rollout(net, rocks, starts(rng, 400, cols), rng.integers(0, 2, size=(400, ROCKS)), 7, record=False)
    a = ro['act'][ro['valid']]
    res[nm] = (round(float(ro['ret'].mean()), 2), round(float((a >= 5).sum() / 400), 2), round(float((a == 4).sum() / 400), 2), len(ro['dec']))
print(sp.name, lr, ent, 'upd', st['updates_done'], 'train s', round(time.time() - t, 1), 'curve', [round(c, 1) for c in log['curve']], 'ret/checks/samples/visits', res, flush=True)
