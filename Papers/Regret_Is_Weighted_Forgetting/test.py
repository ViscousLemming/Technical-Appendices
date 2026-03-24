"""
Unit tests.
Run with: python test.py

Tests that need torch/gymnasium are marked and skipped gracefully if unavailable.
"""
import sys, os, math, traceback
import numpy as np
import random as pyrandom
from collections import defaultdict

# ---- Conditional imports ----
try:
    import torch; import torch.nn as nn; HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
try:
    import gymnasium as gym; HAS_GYM = True
except ImportError:
    HAS_GYM = False

passed = 0; failed = 0; skipped = 0; errors = []

def check(name, condition, detail=""):
    global passed, failed, errors
    if condition:
        passed += 1; print(f"  PASS  {name}")
    else:
        failed += 1; msg = f"  FAIL  {name}" + (f" -- {detail}" if detail else "")
        print(msg); errors.append(msg)

def skip(name, reason="needs torch+gymnasium"):
    global skipped; skipped += 1; print(f"  SKIP  {name} ({reason})")

def needs_torch(fn):
    def wrapper():
        if HAS_TORCH and HAS_GYM: fn()
        else: skip(fn.__name__)
    wrapper.__name__ = fn.__name__; return wrapper

# =====================================================================
print("=" * 60); print("SECTION 1: Bridge math (verify_reduction.py)"); print("=" * 60)
# =====================================================================

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_reduction import Point, cell_sums, k_rho as kr_verify, regret, \
    optimal_q_and_cost, decomposition_rhs

# Toy example from paper
pts = [Point("C1", 1, 0.30, 1.0), Point("C1", 0, 0.20, 1.0), Point("C2", 0, 0.25, 1.0)]
check("Toy K_rho = 0.20", abs(kr_verify(pts) - 0.20) < 1e-10)
q = {"C1": 0.8, "C2": 0.0}
check("Toy regret = 0.22", abs(regret(pts, q) - 0.22) < 1e-10)
check("Toy decomposition identity", abs(regret(pts, q) - decomposition_rhs(pts, q)) < 1e-10)

# Pure cell
pts_pure = [Point("C1", 1, 0.5, 0.8), Point("C1", 1, 0.5, 0.6)]
check("Pure cell K_rho = 0", kr_verify(pts_pure) == 0.0)

# Random decomposition identity (200 trials)
rng_d = pyrandom.Random(42)
decomp_fail = None
for trial in range(200):
    raw = []
    for c in range(rng_d.randint(2, 6)):
        for _ in range(rng_d.randint(1, 5)):
            raw.append((f"C{c}", rng_d.randint(0, 1), rng_d.random() + 0.01, rng_d.random()))
    z = sum(m for _, _, m, _ in raw)
    pts_r = [Point(c, l, m/z, r) for c, l, m, r in raw]
    q_star, _ = optimal_q_and_cost(pts_r)
    q_rand = {c: rng_d.random() for c in q_star}
    if abs(regret(pts_r, q_rand) - decomposition_rhs(pts_r, q_rand)) > 1e-9:
        decomp_fail = trial; break
check("Random decomposition identity (200 trials)", decomp_fail is None,
      f"failed on trial {decomp_fail}" if decomp_fail else "")

# =====================================================================
print("\n" + "=" * 60); print("SECTION 2: POMDP infrastructure"); print("=" * 60)
# =====================================================================

from run_experiment_v28 import (make_structured_pomdp, roll_out_batch,
    posterior_prob_batch, state_label, Pt, k_rho, raw_impurity, accuracy_from_pts,
    build_pts_from_arrays)

# POMDP construction
rng_p = np.random.default_rng(42)
O, T = make_structured_pomdp(rng_p, 64, 10)
check("O rows sum to 1", np.allclose(O.sum(axis=1), 1.0))
check("T rows sum to 1", np.allclose(T.sum(axis=1), 1.0))
check("O non-negative", (O >= 0).all())
check("T non-negative", (T >= 0).all())
check("O shape", O.shape == (64, 10))
check("T shape", T.shape == (64, 64))

# Block structure
check("Class 1 (s<32) favours low obs", O[:32].mean(0)[0] > O[:32].mean(0)[-1])
check("Class 0 (s>=32) favours high obs", O[32:].mean(0)[-1] > O[32:].mean(0)[0])

# Rollout shapes and ranges
rng_r = np.random.default_rng(99)
states, obs = roll_out_batch(O, T, rng_r, 500, 10)
check("Rollout states shape", states.shape == (500,))
check("Rollout obs shape", obs.shape == (500, 10))
check("Rollout states in range", states.min() >= 0 and states.max() < 64)
check("Rollout obs in range", obs.min() >= 0 and obs.max() < 10)

# Posterior: correctness vs naive
def posterior_naive(obs_seq, O, T):
    n_s = O.shape[0]; half = n_s // 2
    b = np.ones(n_s) / n_s; b *= O[:, obs_seq[0]]
    for t in range(1, len(obs_seq)):
        b = T.T @ b; b *= O[:, obs_seq[t]]
    s = b.sum()
    if s == 0: return 0.5
    b /= s; return b[:half].sum()

rng_pv = np.random.default_rng(123)
O_pv, T_pv = make_structured_pomdp(rng_pv, 64, 10)
obs_pv = rng_pv.integers(0, 10, size=(100, 8))
naive = np.array([posterior_naive(obs_pv[i], O_pv, T_pv) for i in range(100)])
batch = posterior_prob_batch(obs_pv, O_pv, T_pv)
check("Posterior batch == naive (max diff < 1e-12)",
      np.max(np.abs(naive - batch)) < 1e-12, f"max diff = {np.max(np.abs(naive - batch))}")
check("Posterior in [0,1]", (batch >= 0).all() and (batch <= 1).all())

# Label function
check("state_label(0, 64) == 1", state_label(0, 64) == 1)
check("state_label(31, 64) == 1", state_label(31, 64) == 1)
check("state_label(32, 64) == 0", state_label(32, 64) == 0)
check("state_label(63, 64) == 0", state_label(63, 64) == 0)

# K_rho, raw_impurity, accuracy on known data
pts_known = [
    Pt("A", 1, 0.25, 0.8), Pt("A", 0, 0.25, 0.6),
    Pt("B", 1, 0.25, 0.9), Pt("B", 1, 0.25, 0.7)]
# Cell A: A=0.25*0.8=0.2, B=0.25*0.6=0.15 -> min=0.15; Cell B: A=0.25*0.9+0.25*0.7=0.4, B=0 -> min=0
check("k_rho known", abs(k_rho(pts_known) - 0.15) < 1e-10)
check("raw_impurity known", abs(raw_impurity(pts_known) - 0.25) < 1e-10)
check("accuracy known", abs(accuracy_from_pts(pts_known) - 0.75) < 1e-10)

# build_pts_from_arrays
cells_list = ["A", "A", "B", "B"]
posteriors_arr = np.array([0.7, 0.3, 0.8, 0.9])
pts_built = build_pts_from_arrays(cells_list, posteriors_arr, 100)
check("build_pts length", len(pts_built) == 4)
check("build_pts labels", [p.label for p in pts_built] == [1, 0, 1, 1])
check("build_pts mu", all(abs(p.mu - 0.25) < 1e-10 for p in pts_built))

# =====================================================================
print("\n" + "=" * 60); print("SECTION 3: Encoding functions (needs torch)"); print("=" * 60)
# =====================================================================

if HAS_TORCH:
    from run_experiment_v28 import encode_flat, encode_seq
    obs_test = np.array([[0, 3, 1], [2, 0, 4]], dtype=np.int64)

    xf = encode_flat(obs_test, 5, 32)
    check("encode_flat shape", xf.shape == (2, 15))
    check("encode_flat row sums = 3", torch.allclose(xf.sum(1), torch.tensor([3., 3.])))
    check("encode_flat [0, 0]=1", xf[0, 0].item() == 1.0)  # obs 0 at t=0
    check("encode_flat [0, 8]=1", xf[0, 8].item() == 1.0)  # obs 3 at t=1: col=1*5+3=8
    check("encode_flat [0, 11]=1", xf[0, 11].item() == 1.0) # obs 1 at t=2: col=2*5+1=11

    xs = encode_seq(obs_test, 5)
    check("encode_seq shape", xs.shape == (2, 3, 5))
    check("encode_seq [0,0,0]=1", xs[0, 0, 0].item() == 1.0)
    check("encode_seq [0,1,3]=1", xs[0, 1, 3].item() == 1.0)
    check("encode_seq [1,2,4]=1", xs[1, 2, 4].item() == 1.0)
    check("encode_seq step sums = 1", torch.allclose(xs.sum(2), torch.ones(2, 3)))
else:
    for name in ["encode_flat", "encode_seq"]:
        skip(name, "needs torch")

# =====================================================================
print("\n" + "=" * 60); print("SECTION 4: STS two-sided verification"); print("=" * 60)
# =====================================================================

from run_experiment_v28 import (make_boolean_env, random_structured_label,
    side_a_cells_and_k_rho, side_b_cells_and_k_rho)

env_sts, vocab_sts, _ = make_boolean_env(5)
rng_sts = pyrandom.Random(0)
sts_ok = True; sts_detail = ""
for trial in range(20):
    labels, _ = random_structured_label(rng_sts, 5)
    n_obs = rng_sts.randint(0, 5)
    obs_bits = sorted(rng_sts.sample(range(5), n_obs)) if n_obs > 0 else []
    _, ka, da = side_a_cells_and_k_rho(5, obs_bits, labels)
    _, kb, db = side_b_cells_and_k_rho(env_sts, vocab_sts, 5, obs_bits, labels)
    if abs(ka - kb) > 1e-10:
        sts_ok = False; sts_detail = f"K_rho mismatch at trial {trial}: {ka} vs {kb}"; break
    for key in da:
        if key not in db or da[key]["members"] != db[key]["members"]:
            sts_ok = False; sts_detail = f"Cell mismatch at trial {trial}"; break
    if not sts_ok: break
check("STS two-sided K_rho + cells (20 trials)", sts_ok, sts_detail)

# =====================================================================
print("\n" + "=" * 60); print("SECTION 5: RL environments (needs torch+gym)"); print("=" * 60)
# =====================================================================

if HAS_TORCH and HAS_GYM:
    from run_experiment_v28_rl import CorridorMemory, TMaze, NoisyCartPole, RepeatCopy

    # CorridorMemory
    env = CorridorMemory(corridor_length=10, n_cue_steps=2, cue_accuracy=0.8, obs_dim=4)
    obs, info = env.reset(seed=0)
    check("Corridor: obs shape", obs.shape == (4,))
    check("Corridor: label in {0,1}", info["hidden_label"] in [0, 1])
    check("Corridor: max_steps", env.max_steps == 10)
    steps = 0
    for _ in range(20):
        _, r, done, _, info = env.step(info["hidden_label"]); steps += 1
        if done: break
    check("Corridor: terminates at L", steps == 10)

    # TMaze
    env2 = TMaze(corridor_length=10); obs2, info2 = env2.reset(seed=0)
    check("TMaze: cue visible at reset", obs2[1] != 0.0)
    obs2b, _, _, _, _ = env2.step(0)
    check("TMaze: cue gone after step 1", obs2b[1] == 0.0)
    for _ in range(20):
        _, r, done, _, _ = env2.step(info2["hidden_label"])
        if done: break
    check("TMaze: +4 for correct junction", r == 4.0)

    # NoisyCartPole
    env3 = NoisyCartPole(max_steps=50); obs3, _ = env3.reset(seed=0)
    check("CartPole: velocity masked", obs3[1] == 0.0 and obs3[3] == 0.0)
    env3.close()

    # RepeatCopy
    env4 = RepeatCopy(seq_length=3, alphabet_size=4); _, _ = env4.reset(seed=42)
    check("RepeatCopy: max_steps", env4.max_steps == 6)
    check("RepeatCopy: N_HIDDEN_CLASSES", env4.N_HIDDEN_CLASSES == 4)
    # Observe 3 symbols
    for t in range(3):
        o, r, d, _, _ = env4.step(0)
        check(f"RepeatCopy: no reward phase 1 step {t}", r == 0.0)
    # Recall 3 symbols correctly
    for t in range(3):
        o, r, d, _, _ = env4.step(env4.sequence[t])
        check(f"RepeatCopy: +1 correct recall step {t}", r == 1.0)
    check("RepeatCopy: done after 6 steps", d)
else:
    for n in ["Corridor", "TMaze", "CartPole", "RepeatCopy"]:
        skip(n)

# =====================================================================
print("\n" + "=" * 60); print("SECTION 6: Agent + PPO + K_rho (needs torch+gym)"); print("=" * 60)
# =====================================================================

if HAS_TORCH and HAS_GYM:
    from run_experiment_v28_rl import (GRUAgent, collect_episodes,
        compute_advantages, ppo_update, compute_krho)

    # Agent shapes
    ag = GRUAgent(5, 32, 2, 2)
    x = torch.randn(4, 10, 5)
    lo, va, go, hn = ag(x)
    check("Agent logits shape", lo.shape == (4, 10, 2))
    check("Agent values shape", va.shape == (4, 10))
    check("Agent gru_out shape", go.shape == (4, 10, 32))

    # Single-step batch
    lo1, va1, hn1 = ag.act_batch(torch.randn(3, 1, 5), None)
    check("Agent.act_batch logits shape", lo1.shape == (3, 2))

    # Probe
    pr = ag.probe(go[:, -1, :])
    check("Probe shape", pr.shape == (4, 2))

    # Discretise hidden states
    cells = ag.discretise_hidden(go[:, -1, :], n_bins=5)
    check("discretise_hidden returns list", isinstance(cells, list) and len(cells) == 4)

    # Collect episodes
    ag_c = GRUAgent(5, 16, 2, 2)
    ro = collect_episodes(ag_c, CorridorMemory, {"corridor_length": 10,
         "n_cue_steps": 2, "cue_accuracy": 0.8, "obs_dim": 5}, 8)
    check("Collect obs shape", ro["obs"].shape == (8, 10, 5))
    check("Collect mask sums", ro["mask"].sum() == 80.0)
    check("Collect lens shape", ro["lens"].shape == (8,))

    # GAE
    adv, ret = compute_advantages(ro["rew"], ro["val"], ro["mask"])
    check("GAE returns = adv + values", np.allclose(ret, adv + ro["val"]))

    # PPO all modes including mdl
    opt = torch.optim.Adam(ag_c.parameters(), lr=1e-3)
    for mode in ["none", "unweighted", "krho_weighted", "mdl"]:
        try:
            loss = ppo_update(ag_c, ro, opt, aux_mode=mode, device='cpu')
            check(f"PPO aux={mode} runs, loss finite", math.isfinite(loss))
        except Exception as e:
            check(f"PPO aux={mode} runs", False, str(e))

    # K_rho diagnostic
    diag = compute_krho(ag_c, CorridorMemory, {"corridor_length": 10,
                        "n_cue_steps": 2, "cue_accuracy": 0.8, "obs_dim": 5},
                        n_episodes=30)
    check("K_rho >= 0", diag["k_rho"] >= 0)
    check("Accuracy in [0,1]", 0 <= diag["accuracy"] <= 1)
    check("N_cells >= 1", diag["n_cells"] >= 1)

    # Mini training run - all conditions
    from run_experiment_v28_rl import train_agent, ENV_CONFIGS
    ENV_CONFIGS["CorridorMemory"]["total_steps"] = 3000
    ENV_CONFIGS["CorridorMemory"]["ep_per_update"] = 10
    for cond in ["baseline", "aux_unweighted", "aux_krho", "aux_mdl"]:
        try:
            hist = train_agent("CorridorMemory", cond, seed=42,
                               hidden_dim=16, eval_every=1500)
            check(f"train_agent {cond}", len(hist) >= 1 and "k_rho" in hist[-1])
        except Exception as e:
            check(f"train_agent {cond}", False, f"{e}")
    ENV_CONFIGS["CorridorMemory"]["total_steps"] = 200_000
    ENV_CONFIGS["CorridorMemory"]["ep_per_update"] = 100
else:
    for n in ["Agent", "Collect", "GAE", "PPO", "K_rho", "train_agent"]:
        skip(n)

# =====================================================================
print("\n" + "=" * 60)
total = passed + failed
print(f"RESULTS: {passed} passed, {failed} failed, {skipped} skipped (of {total + skipped} total)")
if errors:
    print("\nFAILURES:"); [print(e) for e in errors]
print("=" * 60)
sys.exit(0 if failed == 0 else 1)
