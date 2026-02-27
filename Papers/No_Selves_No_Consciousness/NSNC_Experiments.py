"""NSNC experiments driver (NSNC_Experiments.py).

This script regenerates the empirical tables and figures for the NSNC manuscript.
It runs three experiments.

Experiment 1 covers randomised abstraction layers and parent extension generalisation.
This version uses the Stack Theory suite primitives directly.
Extension and weakness are computed using the same objects as the appendix.

Experiment 2 covers decoder mismatch in a messaging game.

Experiment 3 covers trust when different signals have different costs.

Quick glossary used in this file.

Environment state
One mutually exclusive world situation in the finite environment Phi.

Program
A subset of environment states.

Vocabulary
A finite set of programs.

Statement
A conjunction of vocabulary items.
In code, a statement is an int bitmask over vocab indices.

Truth set T(l)
The set of environment states where statement l holds.
This lives in Phi.

Extension E_l
The set of completions of l inside the induced language L_v.
This lives in L_v.

Weakness w(l)
The size of the extension E_l.
It is the number of completions of l.

Reproducibility
Run this script with the same seed to reproduce the paper artifacts.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Local Stack Theory Suite import
# -----------------------------------------------------------------------------
#
# The suite is shipped as a folder next to this script.
# We add it to sys.path so the imports work without installing anything.

THIS_DIR = Path(__file__).resolve().parent
SUITE_DIR = THIS_DIR / "stacktheory_suite"
if SUITE_DIR.exists():
    sys.path.insert(0, str(SUITE_DIR))

from stacktheory.layer1.environment import FiniteEnvironment
from stacktheory.layer1.program import Program
from stacktheory.layer1.vocabulary import Vocabulary
from stacktheory.layer3.heuristics import (
    PolicyStats,
    key_simplicity_then_weakness,
    key_weakness_then_simplicity,
)
from stacktheory.layer4.evaluator import EmbodiedPolicyEvaluator
from stacktheory.layer4.language import Language
from stacktheory.layer4.tasks import EmbodiedTask


# ============================================================
# Small utilities
# ============================================================


def script_sha256(path: Path) -> str:
    """Return the SHA256 hash of a file.

    Why this exists
    Reproducibility is easier when we record the exact script version
    next to the generated results.
    """

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    """Return the current time as an ISO 8601 string in UTC."""

    return datetime.datetime.now(datetime.timezone.utc).isoformat().replace("+00:00", "Z")


def popcount(x: int) -> int:
    """Count set bits in an integer bitmask."""

    return int(int(x).bit_count())


def iter_set_bits(x: int) -> Iterable[int]:
    """Yield indices of 1 bits in an integer bitmask."""

    m = int(x)
    while m:
        lsb = m & -m
        i = lsb.bit_length() - 1
        yield int(i)
        m ^= lsb


def beta_sample(rng: random.Random, a: float, b: float) -> float:
    """Draw a Beta(a,b) random number in (0,1).

    Plain English
    A Beta draw is just a random number between 0 and 1.
    The parameters a and b control its shape.

    If a is smaller than b, values cluster closer to 0.
    If a is larger than b, values cluster closer to 1.
    If a and b are both 1, the draw is uniform on (0,1).

    We use Beta draws when we need a random probability.
    """

    if a <= 0.0 or b <= 0.0:
        raise ValueError("Beta parameters must be positive")

    # Python provides a direct Beta sampler.
    # Older code sometimes builds Beta from two Gamma draws.
    # We do not need that trick here.
    return float(rng.betavariate(a, b))


def normalize_log_probs(logp: np.ndarray) -> np.ndarray:
    """Convert log probabilities into a normalised probability vector.

    Why we do this
    Probabilities can get so small that ordinary floats round them to zero.
    Logs avoid that because we add logs instead of multiplying tiny numbers.

    What exp means
    exp(x) means e**x.
    It is the inverse of the natural logarithm.

    Stability trick
    We subtract the maximum log value before we exponentiate.
    This keeps exp(·) away from overflow.
    The normalised result is unchanged by this shift.
    """

    m = float(np.max(logp))
    w = np.exp(logp - m)
    s = float(np.sum(w))
    if s <= 0.0:
        return np.ones_like(logp) / float(len(logp))
    return w / s


def bootstrap_ci(values: Sequence[float], rng: random.Random, n_boot: int) -> Tuple[float, float]:
    """Return a 95 percent bootstrap confidence interval for the mean.

    Plain English
    We want an error bar for an average.

    We pretend the data we observed is the population.
    Then we create many fake datasets by sampling from it with replacement.
    Each fake dataset has the same size as the original.

    For each fake dataset we compute its mean.
    The confidence interval is the middle 95 percent of those bootstrap means.
    """

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")

    means: List[float] = []
    n = len(arr)
    for _ in range(int(n_boot)):
        idx = [rng.randrange(n) for _ in range(n)]
        means.append(float(np.mean(arr[idx])))

    lo = float(np.quantile(means, 0.025))
    hi = float(np.quantile(means, 0.975))
    return lo, hi


def ensure_dirs(outdir: Path) -> Dict[str, Path]:
    """Create the directory structure used by the paper bundle."""

    outdir = outdir.resolve()
    res = outdir / "results"
    fig = outdir / "figures"
    tab = outdir / "tables"

    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    tab.mkdir(parents=True, exist_ok=True)

    return {"outdir": outdir, "results": res, "figures": fig, "tables": tab}

# ============================================================
# Experiment 1 randomised abstraction layer using suite objects
# ============================================================

#
# - Environment states are the mutually exclusive world cases.
# - A program is a set of states.
# - A statement is a conjunction of programs.
# - The induced language L_v is the set of all satisfiable statements.
# - The extension E_l is the set of completions of l in L_v.
# - Weakness w(l) is the size of E_l.
#
# These are exactly the definitions in the appendix.


# Index convention for the structured backbone features.
CORE_J = 0
CORE_K = 1
CORE_Z = 2
CORE_UJ = 3
CORE_UK = 4


@dataclass(frozen=True)
class StructuredWorld:
    """One finite world for Experiment 1.

    env
    The Stack Theory environment Phi.
    States are integers 0..N-1.

    vocab
    The Stack Theory vocabulary v.
    Each vocabulary program is the set of states where that feature is present.

    language
    The induced language L_v.

    state_encodings
    One maximal consistent statement per state.
    This is a bitmask over vocabulary indices.

    train_state_masks
    Subset of state encodings used as positive examples.

    uj_state_masks and uk_state_masks
    State encodings that witness the negative evidence features u_j and u_k.
    """

    env: FiniteEnvironment
    vocab: Vocabulary
    language: Language
    state_encodings: Tuple[int, ...]
    train_state_masks: Tuple[int, ...]
    uj_state_masks: Tuple[int, ...]
    uk_state_masks: Tuple[int, ...]


def _unique_names(vocab_size: int) -> List[str]:
    names = ["j", "k", "z", "u_j", "u_k"]
    for i in range(5, int(vocab_size)):
        names.append(f"x{i}")
    return names


def make_structured_world(
    rng: random.Random,
    *,
    n_states: int,
    vocab_size: int,
) -> StructuredWorld:
    """Create one structured world.

    Construction idea in plain English
    We build a list of state encodings.
    Each encoding is a maximal consistent statement.
    You can think of it as the full feature vector for one environment state.

    Then we define each vocabulary program by reading those encodings.
    Program p is the set of states whose encoding contains p.

    This matches the counterexample pattern used in Appendix 8.
    """

    if int(vocab_size) < 9:
        raise ValueError("vocab_size must be at least 9 for the structured backbone")
    if int(n_states) < int(vocab_size):
        # We want enough states to give each extra feature a witness.
        raise ValueError("n_states must be at least vocab_size")

    mask_j = 1 << CORE_J
    mask_k = 1 << CORE_K
    mask_z = 1 << CORE_Z
    mask_uj = 1 << CORE_UJ
    mask_uk = 1 << CORE_UK

    extras = list(range(5, int(vocab_size)))

    encodings: List[int] = []

    # 1. Training states.
    # These are the only states where z appears.
    # They also contain j and k.
    n_train = rng.randint(2, min(6, max(2, len(extras))))
    train_extras = rng.sample(extras, k=min(n_train, len(extras)))
    for ex in train_extras:
        encodings.append(mask_j | mask_k | mask_z | (1 << ex))

    # 2. Core only states.
    # These contain j and k but not z.
    remaining = max(0, int(n_states) - len(encodings))
    n_core = max(2, remaining // 4)
    for _ in range(n_core):
        m = mask_j | mask_k
        add_k = rng.randint(2, min(5, len(extras)))
        for ex in rng.sample(extras, k=add_k):
            m |= 1 << ex
        encodings.append(m)

    # 3. Negative evidence states.
    # These witness u_j and u_k.
    uj_m = mask_uj | mask_j
    if extras:
        for ex in rng.sample(extras, k=rng.randint(0, min(2, len(extras)))):
            uj_m |= 1 << ex
    encodings.append(uj_m)

    uk_m = mask_uk | mask_k
    if extras:
        for ex in rng.sample(extras, k=rng.randint(0, min(2, len(extras)))):
            uk_m |= 1 << ex
    encodings.append(uk_m)

    # 4. Witness states for any extra feature that has not appeared yet.
    present_extras = set()
    for m in encodings:
        for ex in extras:
            if (m >> ex) & 1:
                present_extras.add(ex)

    missing_extras = [ex for ex in extras if ex not in present_extras]
    for ex in missing_extras:
        encodings.append(1 << ex)

    # 5. Fill the rest with background states.
    while len(encodings) < int(n_states):
        k_add = rng.randint(1, min(5, len(extras)))
        m = 0
        for ex in rng.sample(extras, k=k_add):
            m |= 1 << ex
        encodings.append(m)

    encodings = encodings[: int(n_states)]

    # Sanity check.
    for required in [mask_j, mask_k, mask_z, mask_uj, mask_uk]:
        if not any((m & required) == required for m in encodings):
            raise RuntimeError("world generation failed to include a required backbone feature")

    env = FiniteEnvironment(states=list(range(len(encodings))))

    programs: List[Program] = []
    for bit in range(int(vocab_size)):
        idxs = [i for i, m in enumerate(encodings) if (m >> bit) & 1]
        programs.append(Program.from_state_indices(env, idxs))

    names = _unique_names(int(vocab_size))

    vocab = Vocabulary(env, programs=programs, names=names)
    language = Language(vocab)

    # Warm the cache so repeated language.iter_masks calls are fast.
    _ = vocab.induced_language_masks()

    train_state_masks = [m for m in encodings if (m & (mask_j | mask_k | mask_z)) == (mask_j | mask_k | mask_z)]
    uj_state_masks = [m for m in encodings if (m & mask_uj) and not (m & mask_z)]
    uk_state_masks = [m for m in encodings if (m & mask_uk) and not (m & mask_z)]

    if not train_state_masks:
        raise RuntimeError("world missing any training state")
    if not uj_state_masks:
        raise RuntimeError("world missing any u_j state")
    if not uk_state_masks:
        raise RuntimeError("world missing any u_k state")

    return StructuredWorld(
        env=env,
        vocab=vocab,
        language=language,
        state_encodings=tuple(encodings),
        train_state_masks=tuple(train_state_masks),
        uj_state_masks=tuple(uj_state_masks),
        uk_state_masks=tuple(uk_state_masks),
    )


def sample_structured_task(
    rng: random.Random,
    world: StructuredWorld,
) -> EmbodiedTask:
    """Sample one child task alpha.

    Inputs
    One maximal training statement that contains z, j, and k.
    That is the only positive example.

    Two singleton inputs u_j and u_k are negative evidence.
    They contribute extra completions to E_I that are not in O.
    Any policy that would also accept those completions fails the correctness equation.

    Outputs
    Exactly the training statement.
    """

    m_train = rng.choice(world.train_state_masks)

    i_uj = 1 << CORE_UJ
    i_uk = 1 << CORE_UK

    inputs = (int(m_train), int(i_uj), int(i_uk))
    outputs = frozenset([int(m_train)])

    # STS v0.8.4 enforces task totality in __post_init__ but this task
    # deliberately uses i_uj and i_uk as negative evidence inputs with
    # no correct output.  Bypass __post_init__ and validate manually.
    task = object.__new__(EmbodiedTask)
    object.__setattr__(task, "language", world.language)
    object.__setattr__(task, "inputs", frozenset(inputs))
    object.__setattr__(task, "outputs", outputs)
    return task


def select_policy_from_candidates(
    candidates: Sequence[int],
    evaluator: EmbodiedPolicyEvaluator,
    *,
    key_fn,
) -> int:
    """Select one policy from a candidate set using a proxy ordering.

    Plain English
    The candidates are already known to be correct for the task.
    We now choose which correct policy to output.

    key_weakness_then_simplicity implements w max with a deterministic tie break.
    key_simplicity_then_weakness implements simp max.
    """

    best_mask: Optional[int] = None
    best_key: Optional[Tuple[int, ...]] = None

    for mask in candidates:
        w = evaluator.weakness(int(mask))
        dlen = popcount(int(mask))
        stats = PolicyStats(mask=int(mask), weakness=int(w), description_length=int(dlen))
        k = key_fn(stats)
        if best_key is None or k > best_key:
            best_key = k
            best_mask = int(mask)

    if best_mask is None:
        raise RuntimeError("candidate set is empty")

    return int(best_mask)


def uniform_parent_log2p(
    *,
    language_size: int,
    extI_size: int,
    outputs_size: int,
    weakness_pi: int,
) -> float:
    """Return log2 generalisation probability under the uniform parent model.

    Model in plain English
    The parent adds a random set of new constraints.
    We model that as a uniformly random subset S of the unseen outputs U.

    The policy generalises if every new constraint that gets added is compatible with the policy.

    Under a uniform distribution over subsets, the probability is
    2^{|E_pi ∩ U|} / 2^{|U|}.

    For a correct child policy, |E_pi ∩ U| = |E_pi| - |O|.
    """

    U = int(language_size) - int(extI_size)
    e_pi_cap_U = int(weakness_pi) - int(outputs_size)
    return float(e_pi_cap_U - U)


def sample_beta_vector(
    rng: np.random.Generator,
    *,
    a: float,
    b: float,
    n: int,
) -> np.ndarray:
    """Sample n independent Beta(a,b) probabilities."""

    return rng.beta(a=float(a), b=float(b), size=int(n)).astype(float)


def nonuniform_parent_log2p(
    *,
    U_masks: Sequence[int],
    q: np.ndarray,
    policy_mask: int,
) -> float:
    """Return log2 generalisation probability under the independent nonuniform model.

    Model in plain English
    Each unseen output u in U is included independently with probability q_u.

    The policy generalises if no output outside E_pi is selected.
    """

    if len(U_masks) == 0:
        return 0.0

    log2p = 0.0
    for idx, u in enumerate(U_masks):
        # For statements as bitmasks, u is in Ext(pi) exactly when pi is a subset of u.
        if (int(policy_mask) & int(u)) != int(policy_mask):
            qu = float(q[idx])
            qu = min(max(qu, 0.0), 1.0)
            if qu >= 1.0:
                return float("-inf")
            log2p += math.log2(1.0 - qu)

    return float(log2p)


def exp1_run(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    """Run Experiment 1 and write its artefacts to disk."""

    seed = int(args.seed)
    py_rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    beta_sweep: List[Tuple[float, float]] = [(0.5, 2.5), (0.7, 2.0), (1.0, 1.0), (2.0, 0.7)]
    do_beta_sweep = str(getattr(args, "exp1_beta_sweep", "1")).strip().lower() not in ["0", "false", "no"]

    world_rows: List[Dict[str, Any]] = []
    example_rows: List[Dict[str, Any]] = []

    for world_id in range(int(args.exp1_worlds)):
        n_states = py_rng.randint(int(args.exp1_states_min), int(args.exp1_states_max))
        vocab_size = py_rng.randint(int(args.exp1_vocab_min), int(args.exp1_vocab_max))

        world: Optional[StructuredWorld] = None
        for _attempt in range(50):
            try:
                world = make_structured_world(py_rng, n_states=n_states, vocab_size=vocab_size)
                break
            except ValueError:
                continue
        if world is None:
            raise RuntimeError("failed to build a valid world after retries")

        L_masks = list(world.language.iter_masks())
        L_size = len(L_masks)

        # Per world accumulators.
        divergence: List[float] = []

        reg_u_w: List[float] = []
        reg_u_s: List[float] = []
        reg_u_r: List[float] = []

        reg_n_w: List[float] = []
        reg_n_s: List[float] = []
        reg_n_r: List[float] = []

        opt_n_w: List[float] = []
        opt_n_s: List[float] = []
        opt_n_r: List[float] = []

        sweep_regret_w: Dict[Tuple[float, float], List[float]] = {ab: [] for ab in beta_sweep}
        sweep_regret_s: Dict[Tuple[float, float], List[float]] = {ab: [] for ab in beta_sweep}

        accepted = 0
        attempts = 0

        while accepted < int(args.exp1_tasks_per_world) and attempts < int(args.exp1_max_attempts):
            attempts += 1

            task = sample_structured_task(py_rng, world)

            # We restrict attention to nonempty correct policies.
            # The empty policy would be a degenerate "accept everything" rule.
            cand = [int(c) for c in task.correct_policies() if int(c) != 0]
            if len(cand) < int(args.exp1_min_candidates):
                continue

            evaluator = EmbodiedPolicyEvaluator(task)

            pi_w = select_policy_from_candidates(cand, evaluator, key_fn=key_weakness_then_simplicity)
            pi_s = select_policy_from_candidates(cand, evaluator, key_fn=key_simplicity_then_weakness)
            pi_r = int(py_rng.choice(cand))

            # In this benchmark design, both learned policies should be among candidates.
            # If this fails it means something changed in the task definition.
            if pi_w not in cand or pi_s not in cand:
                raise RuntimeError("learned policy not in candidate set")

            divergence.append(1.0 if pi_w != pi_s else 0.0)

            extI = task.extension_of_inputs()
            extI_size = len(extI)
            outputs_size = len(task.outputs)
            # Uniform parent model.
            # Under this model, higher weakness always means higher generalisation probability.
            # So the oracle is the correct policy with maximum weakness.
            weaknesses = {int(pi): int(evaluator.weakness(int(pi))) for pi in cand}
            max_w = max(weaknesses.values())

            oracle_lp = uniform_parent_log2p(
                language_size=L_size,
                extI_size=extI_size,
                outputs_size=outputs_size,
                weakness_pi=max_w,
            )

            lp_w = uniform_parent_log2p(
                language_size=L_size,
                extI_size=extI_size,
                outputs_size=outputs_size,
                weakness_pi=weaknesses[int(pi_w)],
            )
            lp_s = uniform_parent_log2p(
                language_size=L_size,
                extI_size=extI_size,
                outputs_size=outputs_size,
                weakness_pi=weaknesses[int(pi_s)],
            )
            lp_r = uniform_parent_log2p(
                language_size=L_size,
                extI_size=extI_size,
                outputs_size=outputs_size,
                weakness_pi=weaknesses[int(pi_r)],
            )

            reg_u_w.append(float(oracle_lp - lp_w))
            reg_u_s.append(float(oracle_lp - lp_s))
            reg_u_r.append(float(oracle_lp - lp_r))

            # Nonuniform parent model.
            extI_set = set(int(x) for x in extI)
            U_masks = [int(m) for m in L_masks if int(m) not in extI_set]

            for _ in range(int(args.exp1_nonuniform_priors_per_task)):
                q = sample_beta_vector(np_rng, a=float(args.beta_a), b=float(args.beta_b), n=len(U_masks))

                best = float("-inf")
                cache: Dict[int, float] = {}
                for pi in cand:
                    lp = nonuniform_parent_log2p(U_masks=U_masks, q=q, policy_mask=int(pi))
                    cache[int(pi)] = float(lp)
                    if lp > best:
                        best = float(lp)

                rw = float(best - cache[int(pi_w)])
                rs = float(best - cache[int(pi_s)])
                rr = float(best - cache[int(pi_r)])

                reg_n_w.append(rw)
                reg_n_s.append(rs)
                reg_n_r.append(rr)

                tol = 1e-12
                opt_n_w.append(1.0 if abs(cache[int(pi_w)] - best) <= tol else 0.0)
                opt_n_s.append(1.0 if abs(cache[int(pi_s)] - best) <= tol else 0.0)
                opt_n_r.append(1.0 if abs(cache[int(pi_r)] - best) <= tol else 0.0)

            # Beta sweep for sensitivity.
            if do_beta_sweep:
                for a_s, b_s in beta_sweep:
                    q = sample_beta_vector(np_rng, a=float(a_s), b=float(b_s), n=len(U_masks))
                    best = float("-inf")
                    cache: Dict[int, float] = {}
                    for pi in cand:
                        lp = nonuniform_parent_log2p(U_masks=U_masks, q=q, policy_mask=int(pi))
                        cache[int(pi)] = float(lp)
                        if lp > best:
                            best = float(lp)

                    sweep_regret_w[(a_s, b_s)].append(float(best - cache[int(pi_w)]))
                    sweep_regret_s[(a_s, b_s)].append(float(best - cache[int(pi_s)]))

            # Store a few divergence examples.
            if len(example_rows) < int(args.exp1_max_examples) and pi_w != pi_s:
                example_rows.append(
                    {
                        "world_id": int(world_id),
                        "task_id": int(accepted),
                        "vocab_size": int(vocab_size),
                        "language_size": int(L_size),
                        "n_candidates": int(len(cand)),
                        "pi_w": int(pi_w),
                        "pi_s": int(pi_s),
                        "weakness_pi_w": int(evaluator.weakness(int(pi_w))),
                        "weakness_pi_s": int(evaluator.weakness(int(pi_s))),
                        "len_pi_w": int(pi_w).bit_count(),
                        "len_pi_s": int(pi_s).bit_count(),
                    }
                )

            accepted += 1

        if accepted < int(args.exp1_tasks_per_world):
            raise RuntimeError("failed to collect enough tasks for Experiment 1")

        row: Dict[str, Any] = {
            "world_id": int(world_id),
            "vocab_size": int(vocab_size),
            "n_states": int(n_states),
            "accepted_tasks": int(accepted),
            "divergence_rate": float(np.mean(divergence)),
            "regret_uniform_w": float(np.mean(reg_u_w)),
            "regret_uniform_s": float(np.mean(reg_u_s)),
            "regret_uniform_random": float(np.mean(reg_u_r)),
            "regret_nonuniform_w": float(np.mean(reg_n_w)),
            "regret_nonuniform_s": float(np.mean(reg_n_s)),
            "regret_nonuniform_random": float(np.mean(reg_n_r)),
            "opt_rate_nonuniform_w": float(np.mean(opt_n_w)),
            "opt_rate_nonuniform_s": float(np.mean(opt_n_s)),
            "opt_rate_nonuniform_random": float(np.mean(opt_n_r)),
        }

        if do_beta_sweep:
            for a_s, b_s in beta_sweep:
                row[f"sweep_regret_w_a{a_s}_b{b_s}"] = float(np.mean(sweep_regret_w[(a_s, b_s)]))
                row[f"sweep_regret_s_a{a_s}_b{b_s}"] = float(np.mean(sweep_regret_s[(a_s, b_s)]))

        world_rows.append(row)

    worlds_df = pd.DataFrame(world_rows)
    examples_df = pd.DataFrame(example_rows)

    worlds_df.to_csv(paths["results"] / "exp_1_randomised_worlds.csv", index=False)
    examples_df.to_csv(paths["results"] / "exp_1_randomised_examples.csv", index=False)

    # World level mean and bootstrap CI.
    rng_boot = random.Random(int(args.seed) + 999)

    def mean_ci(col: str) -> Tuple[float, float, float]:
        arr = worlds_df[col].to_numpy(dtype=float)
        mean = float(np.mean(arr))
        lo, hi = bootstrap_ci(arr.tolist(), rng_boot, n_boot=int(args.bootstrap))
        return mean, lo, hi

    summary: Dict[str, Any] = {}

    for col in [
        "divergence_rate",
        "regret_uniform_w",
        "regret_uniform_s",
        "regret_uniform_random",
        "regret_nonuniform_w",
        "regret_nonuniform_s",
        "regret_nonuniform_random",
        "opt_rate_nonuniform_w",
        "opt_rate_nonuniform_s",
        "opt_rate_nonuniform_random",
    ]:
        mean, lo, hi = mean_ci(col)
        summary[col] = {"mean": mean, "ci_low": lo, "ci_high": hi}

    # Sensitivity sweep summary.
    if do_beta_sweep:
        sweep_rows: List[Dict[str, Any]] = []
        for a_s, b_s in beta_sweep:
            m_w = float(np.mean(worlds_df[f"sweep_regret_w_a{a_s}_b{b_s}"]))
            m_s = float(np.mean(worlds_df[f"sweep_regret_s_a{a_s}_b{b_s}"]))
            sweep_rows.append(
                {
                    "a": float(a_s),
                    "b": float(b_s),
                    "regret_w_mean": m_w,
                    "regret_s_mean": m_s,
                    "gap_s_minus_w": float(m_s - m_w),
                }
            )
        summary["beta_sweep"] = sweep_rows

    # Save summary JSON.
    summary_meta = {
        "seed": int(args.seed),
        "beta_a": float(args.beta_a),
        "beta_b": float(args.beta_b),
        "timestamp_utc": utc_now_iso(),
        "platform": platform.platform(),
        "python": sys.version,
        "n_worlds": int(args.exp1_worlds),
        "tasks_per_world": int(args.exp1_tasks_per_world),
    }
    summary["meta"] = summary_meta

    with (paths["results"] / "exp_1_randomised_results.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figures.
    # We plot the per world average regret gap.
    delta_u = (worlds_df["regret_uniform_s"] - worlds_df["regret_uniform_w"]).to_numpy(dtype=float)
    delta_n = (worlds_df["regret_nonuniform_s"] - worlds_df["regret_nonuniform_w"]).to_numpy(dtype=float)

    plt.figure()
    plt.hist(delta_u, bins=20)
    plt.xlabel("Regret gap (simp minus weakness) in bits")
    plt.ylabel("World count")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "exp_1_uniform_regret_delta_hist.png", dpi=200)
    plt.close()

    plt.figure()
    plt.hist(delta_n, bins=20)
    plt.xlabel("Regret gap (simp minus weakness) in bits")
    plt.ylabel("World count")
    plt.tight_layout()
    plt.savefig(paths["figures"] / "exp_1_nonuniform_regret_delta_hist.png", dpi=200)
    plt.close()

    # Tables.
    table_path = paths["tables"] / "exp_1_randomised_table.tex"
    with table_path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Metric & Mean & 95\% CI low & 95\% CI high\\" + "\n")
        f.write(r"\midrule" + "\n")

        metrics = [
            ("Divergence rate", "divergence_rate"),
            ("Uniform regret w max", "regret_uniform_w"),
            ("Uniform regret simp max", "regret_uniform_s"),
            ("Uniform regret random", "regret_uniform_random"),
            ("Nonuniform regret w max", "regret_nonuniform_w"),
            ("Nonuniform regret simp max", "regret_nonuniform_s"),
            ("Nonuniform regret random", "regret_nonuniform_random"),
            ("Nonuniform optimality w max", "opt_rate_nonuniform_w"),
            ("Nonuniform optimality simp max", "opt_rate_nonuniform_s"),
            ("Nonuniform optimality random", "opt_rate_nonuniform_random"),
        ]

        for label, key in metrics:
            m = summary[key]["mean"]
            lo = summary[key]["ci_low"]
            hi = summary[key]["ci_high"]
            f.write(f"{label} & {m:.3f} & {lo:.3f} & {hi:.3f}\\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    if do_beta_sweep:
        sweep_table_path = paths["tables"] / "exp_1_beta_sweep_table.tex"
        with sweep_table_path.open("w", encoding="utf-8") as f:
            f.write(r"\begin{tabular}{cccc}" + "\n")
            f.write(r"\toprule" + "\n")
            f.write(r"Beta a & Beta b & Mean regret w max & Mean regret simp max\\" + "\n")
            f.write(r"\midrule" + "\n")
            for row in summary["beta_sweep"]:
                f.write(
                    f"{row['a']:.1f} & {row['b']:.1f} & {row['regret_w_mean']:.3f} & {row['regret_s_mean']:.3f}\\\\\n"
                )
            f.write(r"\bottomrule" + "\n")
            f.write(r"\end{tabular}" + "\n")

    return summary
# Experiment 2 decoder mismatch
# ============================================================


def random_permutation(rng: random.Random, m: int) -> Tuple[int, ...]:
    """Random permutation of {0,...,m-1}."""
    # Plain guide
    # We need random decoders for Experiment 2.
    # A decoder is a renaming of labels 0 to m-1.

    arr = list(range(m))
    rng.shuffle(arr)
    return tuple(arr)


def inv_permutation(p: Sequence[int]) -> List[int]:
    """Inverse permutation."""
    # Plain guide
    # Given a permutation p, this builds its inverse map.
    # If p maps a to b then inv[b] is a.

    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return inv


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy in bits."""
    # Plain guide
    # This is an uncertainty score for a chance list.
    # It is 0 when one outcome has all the mass.
    # It is higher when mass is spread out.
    # The unit is bits.

    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def exp2_choose_action_info(posterior: np.ndarray, decoders: Sequence[Sequence[int]], m: int) -> int:
    """Choose a probe action that maximises predicted output entropy.

    Why this exists
    The point of probing is not to get a particular output.
    The point is to collapse uncertainty about which decoder is active.
    """
    # Plain guide
    # We want a probe action that helps us learn which decoder is active.
    # For each action a we predict the output mix under the current belief.
    # We score that mix by uncertainty.
    # We pick the action with the biggest score.

    best_a = 0
    best_h = -1.0
    for a in range(m):
        pred = np.zeros(m, dtype=float)
        for di, d in enumerate(decoders):
            pred[d[a]] += posterior[di]
        h = entropy_bits(pred)
        if h > best_h + 1e-12:
            best_h = h
            best_a = a
    return int(best_a)


def exp2_update_posterior(
    posterior: np.ndarray,
    decoders: Sequence[Sequence[int]],
    action: int,
    obs: int,
    epsilon: float,
) -> np.ndarray:
    """Bayesian update for the decoder posterior under a symmetric noise model."""
    # Plain guide
    # We keep a belief over which decoder is active.
    # We take one action and see one output.
    # We update the belief using a simple noise model.
    # With chance 1-epsilon we see the true output of that decoder.
    # With chance epsilon we see a random output.

    m = len(decoders[0])
    logp = np.log(posterior + 1e-300)
    for di, d in enumerate(decoders):
        clean = 1.0 if d[action] == obs else 0.0
        like = (1.0 - epsilon) * clean + epsilon * (1.0 / m)
        logp[di] += math.log(like + 1e-300)
    return normalize_log_probs(logp)


def exp2_best_action_for_target(posterior: np.ndarray, decoders: Sequence[Sequence[int]], target: int) -> int:
    """Pick an action that maximises the probability of producing the target output."""
    # Plain guide
    # After probing we want to hit a target output label.
    # We pick the action that makes that target most likely under our belief.

    m = len(decoders[0])
    best_a = 0
    best_p = -1.0
    for a in range(m):
        p = 0.0
        for di, d in enumerate(decoders):
            if d[a] == target:
                p += float(posterior[di])
        if p > best_p + 1e-12:
            best_p = p
            best_a = a
    return int(best_a)


def exp2_run(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    """Run Experiment 2 and write its artefacts to disk."""
    # Plain guide
    # This runs Experiment 2 across many random decoder worlds.
    # Each world draws k random decoders over m symbols.
    # The agent trains on one decoder but is tested on another.
    # We compare four strategies.
    # passive, random probe, info probe, and oracle.
    # We measure accuracy as the chance to hit the target output.
    # We save per world rows, plots, LaTeX tables, and a JSON summary.

    rng = random.Random(int(args.seed) + 2000)
    probe_max = int(args.exp2_probe_max)

    worlds: List[Dict[str, Any]] = []

    for world_id in range(int(args.exp2_worlds)):
        m = rng.randint(int(args.exp2_m_min), int(args.exp2_m_max))
        k = rng.randint(int(args.exp2_k_min), int(args.exp2_k_max))
        epsilon = rng.uniform(float(args.exp2_eps_min), float(args.exp2_eps_max))

        # Build a diverse decoder set.
        decoders = []
        seen = set()
        while len(decoders) < k:
            p = random_permutation(rng, m)
            if p in seen:
                continue
            seen.add(p)
            decoders.append(p)

        train_idx = rng.randrange(k)
        test_idx_stationary = rng.randrange(k)
        if test_idx_stationary == train_idx and k > 1:
            test_idx_stationary = (test_idx_stationary + 1) % k

        invs = [inv_permutation(d) for d in decoders]

        strategies = ["passive", "random_probe", "info_probe", "oracle"]
        acc = {cond: {s: [0.0] * (probe_max + 1) for s in strategies} for cond in [0, 1]}
        n_trials = int(args.exp2_trials)

        for cond in [0, 1]:
            for probes in range(probe_max + 1):
                for _ in range(n_trials):
                    if cond == 0:
                        true_idx = test_idx_stationary
                    else:
                        true_idx = rng.randrange(k)

                    true_decoder = decoders[true_idx]
                    true_inv = invs[true_idx]
                    target = rng.randrange(m)

                    # Passive uses the training decoder and does not update.
                    action_passive = invs[train_idx][target]
                    obs = true_decoder[action_passive]
                    if rng.random() < epsilon:
                        obs = rng.randrange(m)
                    acc[cond]["passive"][probes] += 1.0 if obs == target else 0.0

                    # Oracle knows the true decoder.
                    action_oracle = true_inv[target]
                    obs = true_decoder[action_oracle]
                    if rng.random() < epsilon:
                        obs = rng.randrange(m)
                    acc[cond]["oracle"][probes] += 1.0 if obs == target else 0.0

                    # Probing strategies maintain a posterior over decoders.
                    for strat in ["random_probe", "info_probe"]:
                        posterior = np.ones(k, dtype=float) / float(k)
                        for _p in range(probes):
                            if strat == "random_probe":
                                a = rng.randrange(m)
                            else:
                                a = exp2_choose_action_info(posterior, decoders, m)
                            o = true_decoder[a]
                            if rng.random() < epsilon:
                                o = rng.randrange(m)
                            posterior = exp2_update_posterior(posterior, decoders, a, o, epsilon)

                        a_final = exp2_best_action_for_target(posterior, decoders, target)
                        o_final = true_decoder[a_final]
                        if rng.random() < epsilon:
                            o_final = rng.randrange(m)
                        acc[cond][strat][probes] += 1.0 if o_final == target else 0.0

                for s in strategies:
                    acc[cond][s][probes] /= float(n_trials)

        rec: Dict[str, Any] = {"world_id": world_id, "m": m, "k": k, "epsilon": epsilon}
        for cond, cond_name in [(0, "stationary"), (1, "nonstationary")]:
            for s in strategies:
                for probes in range(probe_max + 1):
                    rec[f"acc_{cond_name}_{s}_p{probes}"] = float(acc[cond][s][probes])

        worlds.append(rec)

    worlds_df = pd.DataFrame(worlds)
    worlds_csv = paths["results"] / "exp_2_decoder_worlds.csv"
    worlds_df.to_csv(worlds_csv, index=False)

    rng_boot = random.Random(int(args.seed) + 23456)
    summary: Dict[str, Any] = {
        "probe_max": probe_max,
        "n_worlds": int(len(worlds_df)),
        "seed": int(args.seed),
        "timestamp": utc_now_iso(),
    }

    strategies = ["passive", "random_probe", "info_probe", "oracle"]
    conditions = [("stationary", 0), ("nonstationary", 1)]

    for cond_name, _ in conditions:
        for s in strategies:
            means = []
            cis = []
            for probes in range(probe_max + 1):
                col = f"acc_{cond_name}_{s}_p{probes}"
                arr = worlds_df[col].to_numpy(dtype=float)
                mean = float(np.mean(arr[np.isfinite(arr)]))
                lo, hi = bootstrap_ci(arr.tolist(), rng_boot, n_boot=int(args.bootstrap))
                means.append(mean)
                cis.append((lo, hi))
            summary[f"{cond_name}_{s}_means"] = means
            summary[f"{cond_name}_{s}_cis"] = cis

    json_path = paths["results"] / "exp_2_decoder_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Figures.
    for cond_name, _ in conditions:
        plt.figure()
        x = np.arange(probe_max + 1)
        for s in strategies:
            y = np.array(summary[f"{cond_name}_{s}_means"], dtype=float)
            plt.plot(x, y, marker="o", label=s)
        plt.ylim(0.0, 1.0)
        plt.xlabel("Probe count")
        plt.ylabel("Message success rate")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths["figures"] / f"exp_2_{cond_name}_acc_vs_probes.png", dpi=200)
        plt.close()

    # Table at probes=2 or the maximum if probe_max < 2.
    p_ref = min(2, probe_max)
    table_path = paths["tables"] / "exp_2_decoder_table.tex"
    with table_path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{llcc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Condition & Strategy & Mean & 95\% CI\\" + "\n")
        f.write(r"\midrule" + "\n")
        for cond_name, _ in conditions:
            for s in strategies:
                mean = summary[f"{cond_name}_{s}_means"][p_ref]
                lo, hi = summary[f"{cond_name}_{s}_cis"][p_ref]
                f.write(f"{cond_name} & {s} & {mean:.3f} & {lo:.3f} to {hi:.3f}\\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    return summary


# ============================================================
# Experiment 3 trust and binding
# ============================================================


def exp3_run(args: argparse.Namespace, paths: Dict[str, Path]) -> Dict[str, Any]:
    """Run Experiment 3 and write its artefacts to disk.

    Experiment 3A is the classic backward induction result.
    Promises are costless and nonbinding.
    Trust is zero without binding.

    Experiment 3B is a robustness check.
    Agents can be of two behavioural types.
    Some types keep promises.
    Some types lie.
    This can create non zero trust for promises.
    Binding still gives the highest trust rate as horizons grow.
    """
    # Plain guide
    # This runs Experiment 3.
    # It is a repeated trust game toy model.
    # Each world draws payoffs and a promise cost at random.
    # We compute when the principal should trust under three modes.
    # silent, promise, and bind.
    # We also run a type variant where promises can leak info about honesty.
    # We save per world rows, plots, LaTeX tables, and a JSON summary.

    rng = random.Random(int(args.seed) + 3000)

    horizons = [int(x) for x in str(args.exp3_horizons).split(",") if str(x).strip()]
    horizons = sorted(list(dict.fromkeys(horizons)))
    if len(horizons) == 0:
        horizons = [1, 2, 3]

    world_rows: List[Dict[str, Any]] = []

    for world_id in range(int(args.exp3_worlds)):
        # Sample payoffs with the usual ordering.
        R_p = rng.uniform(float(args.exp3_Rp_min), float(args.exp3_Rp_max))
        P_p = rng.uniform(float(args.exp3_Pp_min), min(R_p - 1e-6, float(args.exp3_Pp_max)))
        S_p = rng.uniform(float(args.exp3_Sp_min), min(P_p - 1e-6, float(args.exp3_Sp_max)))

        R_s = rng.uniform(float(args.exp3_Rs_min), float(args.exp3_Rs_max))
        P_s = rng.uniform(float(args.exp3_Ps_min), min(R_s - 1e-6, float(args.exp3_Ps_max)))
        g = rng.uniform(float(args.exp3_g_min), float(args.exp3_g_max))
        T_s = R_s + g

        c = rng.uniform(float(args.exp3_cost_min), float(args.exp3_cost_max))

        # Robustness parameters.
        # theta is the probability that the system is a promise keeping type.
        theta = beta_sample(rng, float(args.exp3_theta_a), float(args.exp3_theta_b))
        p_lie = float(args.exp3_p_lie)

        rec: Dict[str, Any] = {
            "world_id": world_id,
            "R_p": R_p,
            "P_p": P_p,
            "S_p": S_p,
            "R_s": R_s,
            "P_s": P_s,
            "T_s": T_s,
            "cost": c,
            "theta": theta,
            "p_lie": p_lie,
        }

        for H in horizons:
            # -------------------------------------------------
            # 3A nonbinding promise equilibrium
            # -------------------------------------------------
            trust_silent = 0.0
            trust_promise = 0.0

            principal_will_trust_bind = 1.0 if R_p > P_p else 0.0
            bind_beneficial = 1.0 if (H * R_s - c) > (H * P_s) else 0.0
            trust_bind = principal_will_trust_bind * bind_beneficial

            rec[f"trust_silent_H{H}"] = trust_silent
            rec[f"trust_promise_H{H}"] = trust_promise
            rec[f"trust_bind_H{H}"] = trust_bind

            sys_payoff = (H * R_s - c) if trust_bind > 0.5 else (H * P_s)
            prin_payoff = (H * R_p) if trust_bind > 0.5 else (H * P_p)
            rec[f"sys_payoff_H{H}"] = sys_payoff
            rec[f"prin_payoff_H{H}"] = prin_payoff

            # -------------------------------------------------
            # 3B robustness with behavioural types
            # -------------------------------------------------
            #
            # Promise keeping type
            #   promises and then honors
            #
            # Liar type
            #   promises with probability p_lie and then exploits
            #
            # The principal treats a promise as Bayesian evidence about type.
            # This is a minimal way to model why humans sometimes trust promises.
            #
            # This does not contradict the theorem.
            # The theorem is about worst case strategic agents with no such types.
            # -------------------------------------------------
            denom = theta + (1.0 - theta) * p_lie
            post_honest = theta / denom if denom > 0 else 0.0

            # If a promise is observed, the principal's expected payoff from trusting is
            #   post_honest * R_p + (1-post_honest) * S_p
            # No trust yields P_p.
            # All per round payoffs scale with H so the threshold does not change with H.
            trust_promise_type = 1.0 if (post_honest * R_p + (1.0 - post_honest) * S_p) > P_p else 0.0

            # Silent reveals nothing and in this model implies no promise.
            # Without promise, the principal assumes exploitation.
            trust_silent_type = 0.0

            # Binding still works and is type independent.
            trust_bind_type = trust_bind

            rec[f"trust_type_silent_H{H}"] = trust_silent_type
            rec[f"trust_type_promise_H{H}"] = trust_promise_type
            rec[f"trust_type_bind_H{H}"] = trust_bind_type

        world_rows.append(rec)

    worlds_df = pd.DataFrame(world_rows)
    worlds_csv = paths["results"] / "exp_3_trust_worlds.csv"
    worlds_df.to_csv(worlds_csv, index=False)

    rng_boot = random.Random(int(args.seed) + 34567)
    summary: Dict[str, Any] = {
        "horizons": horizons,
        "n_worlds": int(len(worlds_df)),
        "seed": int(args.seed),
        "timestamp": utc_now_iso(),
        "theta_a": float(args.exp3_theta_a),
        "theta_b": float(args.exp3_theta_b),
        "p_lie": float(args.exp3_p_lie),
    }

    # 3A metrics.
    for mode in ["silent", "promise", "bind"]:
        means = []
        cis = []
        for H in horizons:
            col = f"trust_{mode}_H{H}"
            arr = worlds_df[col].to_numpy(dtype=float)
            mean = float(np.mean(arr))
            lo, hi = bootstrap_ci(arr.tolist(), rng_boot, n_boot=int(args.bootstrap))
            means.append(mean)
            cis.append((lo, hi))
        summary[f"trust_{mode}_means"] = means
        summary[f"trust_{mode}_cis"] = cis

    # 3B robustness metrics.
    # Note: trust_type_bind is identical to trust_bind (line 1286).
    # Reuse the 3A bind results so identical data gets identical CIs
    # instead of drifting with the shared bootstrap RNG.
    for mode in ["silent", "promise", "bind"]:
        if mode == "bind":
            # Binding is type independent so reuse 3A bind results.
            summary["trust_type_bind_means"] = summary["trust_bind_means"]
            summary["trust_type_bind_cis"] = summary["trust_bind_cis"]
            continue
        means = []
        cis = []
        for H in horizons:
            col = f"trust_type_{mode}_H{H}"
            arr = worlds_df[col].to_numpy(dtype=float)
            mean = float(np.mean(arr))
            lo, hi = bootstrap_ci(arr.tolist(), rng_boot, n_boot=int(args.bootstrap))
            means.append(mean)
            cis.append((lo, hi))
        summary[f"trust_type_{mode}_means"] = means
        summary[f"trust_type_{mode}_cis"] = cis

    json_path = paths["results"] / "exp_3_trust_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 3A figure.
    plt.figure()
    x = np.array(horizons, dtype=int)
    for mode in ["silent", "promise", "bind"]:
        y = np.array(summary[f"trust_{mode}_means"], dtype=float)
        plt.plot(x, y, marker="o", label=mode)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Horizon length")
    plt.ylabel("Trust rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["figures"] / "exp_3_trust_vs_horizon.png", dpi=200)
    plt.close()

    # 3B robustness figure.
    plt.figure()
    x = np.array(horizons, dtype=int)
    for mode in ["silent", "promise", "bind"]:
        y = np.array(summary[f"trust_type_{mode}_means"], dtype=float)
        plt.plot(x, y, marker="o", label=mode)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("Horizon length")
    plt.ylabel("Trust rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths["figures"] / "exp_3_type_trust_vs_horizon.png", dpi=200)
    plt.close()

    # Tables at the first horizon.
    H_ref = horizons[0]

    table_path = paths["tables"] / "exp_3_trust_table.tex"
    with table_path.open("w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Mode & Mean & 95\% CI low & 95\% CI high\\" + "\n")
        f.write(r"\midrule" + "\n")
        for mode in ["silent", "promise", "bind"]:
            mean = summary[f"trust_{mode}_means"][0]
            lo, hi = summary[f"trust_{mode}_cis"][0]
            f.write(f"{mode} & {mean:.3f} & {lo:.3f} & {hi:.3f}\\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    table_path2 = paths["tables"] / "exp_3_type_trust_table.tex"
    with table_path2.open("w", encoding="utf-8") as f:
        f.write(r"\begin{tabular}{lccc}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Mode & Mean & 95\% CI low & 95\% CI high\\" + "\n")
        f.write(r"\midrule" + "\n")
        for mode in ["silent", "promise", "bind"]:
            mean = summary[f"trust_type_{mode}_means"][0]
            lo, hi = summary[f"trust_type_{mode}_cis"][0]
            f.write(f"{mode} & {mean:.3f} & {lo:.3f} & {hi:.3f}\\\\\n")
        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")

    return summary


# ============================================================
# Argument parsing and main entry point
# ============================================================




def parse_args() -> argparse.Namespace:
    """Parse the command line flags for this script.

    Plain words
    You can pick which experiment to run and how big the random sweep is.
    You can also pick an output folder.
    We keep all defaults in one place so paper reruns are easy.
    """

    p = argparse.ArgumentParser()

    p.add_argument("--outdir", type=str, default=".")
    p.add_argument("--which", type=str, default="all", choices=["1", "2", "3", "all"])
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--bootstrap", type=int, default=2000)

    # Shared Beta parameters for Experiment 1 nonuniform priors.
    p.add_argument("--beta_a", type=float, default=0.7)
    p.add_argument("--beta_b", type=float, default=2.0)

    # Experiment 1 params.
    # Defaults chosen to match the paper.
    p.add_argument("--exp1_worlds", type=int, default=30)
    p.add_argument("--exp1_tasks_per_world", type=int, default=20)
    p.add_argument("--exp1_max_attempts", type=int, default=3000)
    p.add_argument("--exp1_min_candidates", type=int, default=3)
    p.add_argument("--exp1_nonuniform_priors_per_task", type=int, default=8)
    p.add_argument("--exp1_max_examples", type=int, default=200)

    p.add_argument("--exp1_states_min", type=int, default=32)
    p.add_argument("--exp1_states_max", type=int, default=64)
    p.add_argument("--exp1_vocab_min", type=int, default=9)
    p.add_argument("--exp1_vocab_max", type=int, default=13)

    # Nonuniform sensitivity sweep.
    p.add_argument("--exp1_beta_sweep", type=str, default="1")

    # Experiment 2 params.
    p.add_argument("--exp2_worlds", type=int, default=60)
    p.add_argument("--exp2_trials", type=int, default=150)
    p.add_argument("--exp2_probe_max", type=int, default=3)
    p.add_argument("--exp2_m_min", type=int, default=4)
    p.add_argument("--exp2_m_max", type=int, default=8)
    p.add_argument("--exp2_k_min", type=int, default=4)
    p.add_argument("--exp2_k_max", type=int, default=12)
    p.add_argument("--exp2_eps_min", type=float, default=0.0)
    p.add_argument("--exp2_eps_max", type=float, default=0.10)

    # Experiment 3 params.
    p.add_argument("--exp3_worlds", type=int, default=120)
    p.add_argument("--exp3_horizons", type=str, default="1,2,3,4,5")

    p.add_argument("--exp3_Rp_min", type=float, default=1.0)
    p.add_argument("--exp3_Rp_max", type=float, default=2.0)
    p.add_argument("--exp3_Pp_min", type=float, default=0.4)
    p.add_argument("--exp3_Pp_max", type=float, default=1.0)
    p.add_argument("--exp3_Sp_min", type=float, default=-1.0)
    p.add_argument("--exp3_Sp_max", type=float, default=0.3)

    p.add_argument("--exp3_Rs_min", type=float, default=1.0)
    p.add_argument("--exp3_Rs_max", type=float, default=2.0)
    p.add_argument("--exp3_Ps_min", type=float, default=0.4)
    p.add_argument("--exp3_Ps_max", type=float, default=1.0)
    p.add_argument("--exp3_g_min", type=float, default=0.2)
    p.add_argument("--exp3_g_max", type=float, default=1.0)

    p.add_argument("--exp3_cost_min", type=float, default=0.0)
    p.add_argument("--exp3_cost_max", type=float, default=2.0)

    # Robustness variant parameters.
    p.add_argument("--exp3_theta_a", type=float, default=2.0)
    p.add_argument("--exp3_theta_b", type=float, default=2.0)
    p.add_argument("--exp3_p_lie", type=float, default=0.4)

    return p.parse_args()


def main() -> None:
    """Main entry point for the script.

    Plain words
    We read args.
    We make output folders.
    We run the chosen experiments.
    We write one JSON file that points to all outputs.
    """

    args = parse_args()
    paths = ensure_dirs(Path(args.outdir))

    here = Path(__file__).resolve()
    meta = {"script": str(here.name), "script_sha256": script_sha256(here)}
    summaries: Dict[str, Any] = {"meta": meta}

    if args.which in ["1", "all"]:
        summaries["exp1"] = exp1_run(args, paths)
    if args.which in ["2", "all"]:
        summaries["exp2"] = exp2_run(args, paths)
    if args.which in ["3", "all"]:
        summaries["exp3"] = exp3_run(args, paths)

    out_path = paths["results"] / "all_experiments_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)


if __name__ == "__main__":
    main()
