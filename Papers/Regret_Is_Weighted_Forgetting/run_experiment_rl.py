"""
RL experiment.

Tests three representation-learning principles on partially observable RL:
  - WEAKNESS:  keep all representational distinctions equally open
  - K_rho:     keep diagnostic-important distinctions (from bridge theorem)
  - MDL:       compress the representation to minimum description length

Environments (all partially observable — the agent cannot see the full state):
  1. CorridorMemory   - see a noisy hint, then guess for 20 steps
  2. TMaze            - see a direction cue once, walk 50 steps, choose at end
  3. NoisyCartPole    - balance a pole but velocity is hidden
  4. RepeatCopy       - memorise 4 symbols, then recall them in order

Conditions (all use the same PPO policy optimiser + GRU memory):
  1. baseline         - no auxiliary loss (just reward)
  2. aux_unweighted   - predict the hidden state with equal weight (WEAKNESS)
  3. aux_krho         - predict the hidden state with margin weight (BRIDGE)
  4. aux_mdl          - predict hidden state + penalise representation size (MDL)

The weakness theorems predict: unweighted >= krho, and unweighted > mdl.

Crash-safe: saves after every (environment, condition) pair.
Resumable: skips completed pairs on restart.

Output: results_v28_rl.json
Requirements: torch, numpy, gymnasium
"""

from __future__ import annotations
import os, json, time
from collections import defaultdict
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import torch
import torch.nn as nn
from torch.distributions import Categorical


# ---------------------------------------------------------------------------
# Device selection.
# Apple M2 has a GPU ("MPS") that can speed up the PPO gradient updates.
# Episode collection stays on CPU (too small to benefit from GPU overhead).
# ---------------------------------------------------------------------------
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    PPO_DEVICE = torch.device('mps')
elif torch.cuda.is_available():
    PPO_DEVICE = torch.device('cuda')
else:
    PPO_DEVICE = torch.device('cpu')

# Use all CPU cores minus one (leave one for the OS).
torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))

# Where to save results (same folder as this script).
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_v28_rl.json")


# =========================================================================
# Environments
#
# Each environment is "partially observable": the agent sees observations
# but not the full hidden state. There is always a hidden label (an integer)
# that the agent's representation should learn to distinguish.
# =========================================================================

class CorridorMemory(gym.Env):
    """
    A corridor of fixed length. At the start of each episode, a hidden
    coin flip sets the context to 0 or 1. In the first few steps the agent
    receives noisy hints about the context (cue_accuracy = 75% correct).
    After the cue steps, observations are pure noise.

    At every step the agent picks action 0 or 1.
    Small reward (+/-0.05) each step for matching/mismatching the context.
    Large reward (+/-1.0) on the final step.

    The challenge: remember a noisy hint across many steps of noise.
    """
    metadata = {"render_modes": []}
    ENV_NAME = "CorridorMemory"
    N_HIDDEN_CLASSES = 2

    def __init__(self, corridor_length=20, n_cue_steps=3,
                 cue_accuracy=0.75, obs_dim=5):
        super().__init__()
        self.corridor_length = corridor_length
        self.max_steps = corridor_length
        self.n_cue_steps = n_cue_steps
        self.cue_accuracy = cue_accuracy
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(-3., 3., (obs_dim,), np.float32)
        self.action_space = spaces.Discrete(2)
        self.context = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.context = int(self.np_random.integers(0, 2))
        self.step_count = 0
        return self._obs(), {"hidden_label": self.context}

    def step(self, action):
        is_final = (self.step_count >= self.corridor_length - 1)
        correct = (action == self.context)
        # Big reward at the end, small reward along the way.
        reward = (1.0 if correct else -1.0) if is_final \
            else (0.05 if correct else -0.05)
        self.step_count += 1
        obs = np.zeros(self.obs_dim, np.float32) if is_final else self._obs()
        return obs, float(reward), is_final, False, \
            {"hidden_label": self.context}

    def _obs(self):
        # Gaussian noise in all dimensions.
        obs = self.np_random.standard_normal(self.obs_dim).astype(np.float32)
        obs *= 0.3
        # First dimension: normalised timestep (so the agent knows where it is).
        obs[0] = self.step_count / self.corridor_length
        # During cue steps: a noisy signal about the context.
        if self.step_count < self.n_cue_steps:
            signal = 1.0 if self.context == 1 else -1.0
            # Flip the signal with probability (1 - cue_accuracy).
            if self.np_random.random() > self.cue_accuracy:
                signal *= -1.0
            obs[1] = signal + self.np_random.standard_normal() * 0.2
            obs[2] = -signal + self.np_random.standard_normal() * 0.2
        return obs

    def get_hidden_label(self):
        return self.context


class TMaze(gym.Env):
    """
    T-Maze (Bakker 2002). The agent starts at position 0 and sees a
    directional cue (up or down). It then walks right for corridor_length
    steps, seeing no further cues. At the junction it must go in the
    direction the cue indicated.

    Reward: +4 for correct turn, -3 for wrong turn, -0.1 per step.

    The challenge: remember one bit of information across 50 steps.
    """
    metadata = {"render_modes": []}
    ENV_NAME = "TMaze"
    N_HIDDEN_CLASSES = 2

    def __init__(self, corridor_length=50):
        super().__init__()
        self.corridor_length = corridor_length
        self.max_steps = corridor_length
        self.obs_dim = 3
        self.observation_space = spaces.Box(-1., 2., (3,), np.float32)
        self.action_space = spaces.Discrete(2)
        self.cue = 0
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.cue = int(self.np_random.integers(0, 2))
        self.step_count = 0
        return self._obs(), {"hidden_label": self.cue}

    def step(self, action):
        self.step_count += 1
        at_junction = (self.step_count >= self.corridor_length)
        if at_junction:
            reward = 4.0 if (action == self.cue) else -3.0
        else:
            reward = -0.1
        return self._obs(), float(reward), at_junction, False, \
            {"hidden_label": self.cue}

    def _obs(self):
        obs = np.zeros(3, np.float32)
        obs[0] = self.step_count / self.corridor_length  # position
        if self.step_count == 0:
            obs[1] = 1.0 if self.cue == 0 else -1.0     # cue (only at start)
        obs[2] = 1.0                                      # corridor indicator
        return obs

    def get_hidden_label(self):
        return self.cue


class NoisyCartPole(gym.Env):
    """
    Standard CartPole-v1, but with the two velocity components hidden
    (zeroed out in observations). The agent sees position and angle
    but not how fast they're changing.

    Hidden label: 0 if the pole is falling left, 1 if falling right.

    The challenge: infer velocity direction from position changes over time.
    """
    metadata = {"render_modes": []}
    ENV_NAME = "NoisyCartPole"
    N_HIDDEN_CLASSES = 2

    def __init__(self, max_steps=200):
        super().__init__()
        self.max_steps = max_steps
        self.obs_dim = 4
        self.observation_space = spaces.Box(-4.8, 4.8, (4,), np.float32)
        self.action_space = spaces.Discrete(2)
        self._inner = gym.make("CartPole-v1")
        self.step_count = 0
        self._full_state = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        full_obs, info = self._inner.reset(seed=seed)
        self._full_state = full_obs.copy()
        self.step_count = 0
        return self._mask(full_obs), {"hidden_label": self._label()}

    def step(self, action):
        full_obs, reward, terminated, truncated, info = self._inner.step(action)
        self._full_state = full_obs.copy()
        self.step_count += 1
        done = terminated or truncated or (self.step_count >= self.max_steps)
        return self._mask(full_obs), float(reward), done, False, \
            {"hidden_label": self._label()}

    def _mask(self, obs):
        """Zero out the velocity components (indices 1 and 3)."""
        masked = obs.copy()
        masked[1] = 0.0  # cart velocity
        masked[3] = 0.0  # pole angular velocity
        return masked

    def _label(self):
        """Binary label: which way is the pole falling?"""
        if self._full_state is None:
            return 0
        return 1 if self._full_state[3] >= 0 else 0

    def get_hidden_label(self):
        return self._label()

    def close(self):
        self._inner.close()


class RepeatCopy(gym.Env):
    """
    Two-phase task. Phase 1 (observe): the agent sees K one-hot symbols.
    Phase 2 (recall): the agent must output them back in order.

    Reward: +1 per correct recall, -1 per incorrect recall, 0 during observe.

    Hidden label: the target symbol for the current recall step
    (or 0 during the observation phase).

    The challenge: memorise a sequence and reproduce it.
    """
    metadata = {"render_modes": []}
    ENV_NAME = "RepeatCopy"

    def __init__(self, seq_length=4, alphabet_size=4):
        super().__init__()
        self.seq_length = seq_length
        self.alphabet_size = alphabet_size
        self.N_HIDDEN_CLASSES = alphabet_size  # instance attr (4 classes)
        self.max_steps = 2 * seq_length        # observe + recall
        self.obs_dim = alphabet_size + 2       # one-hot + phase + position
        self.observation_space = spaces.Box(0., 1., (self.obs_dim,), np.float32)
        self.action_space = spaces.Discrete(alphabet_size)
        self.sequence = []
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sequence = [
            int(self.np_random.integers(0, self.alphabet_size))
            for _ in range(self.seq_length)
        ]
        self.step_count = 0
        return self._obs(), {"hidden_label": self._label()}

    def step(self, action):
        self.step_count += 1
        in_recall = (self.step_count > self.seq_length)
        if in_recall:
            target = self.sequence[self.step_count - self.seq_length - 1]
            reward = 1.0 if action == target else -1.0
        else:
            reward = 0.0
        done = (self.step_count >= self.max_steps)
        obs = np.zeros(self.obs_dim, np.float32) if done else self._obs()
        return obs, float(reward), done, False, {"hidden_label": self._label()}

    def _obs(self):
        obs = np.zeros(self.obs_dim, np.float32)
        if self.step_count < self.seq_length:
            # Observation phase: show the symbol as a one-hot vector.
            obs[self.sequence[self.step_count]] = 1.0
            obs[-2] = 0.0  # phase indicator: observe
        else:
            obs[-2] = 1.0  # phase indicator: recall
        obs[-1] = self.step_count / self.max_steps  # normalised position
        return obs

    def _label(self):
        if self.step_count >= self.seq_length:
            recall_idx = min(self.step_count - self.seq_length,
                             self.seq_length - 1)
            return self.sequence[recall_idx]
        return 0

    def get_hidden_label(self):
        return self._label()


# =========================================================================
# Configuration
# =========================================================================

ENV_CONFIGS = {
    "CorridorMemory": {
        "cls": CorridorMemory,
        "kwargs": {"corridor_length": 20, "n_cue_steps": 3,
                   "cue_accuracy": 0.75, "obs_dim": 5},
        "total_steps": 200_000,   # total environment steps to train for
        "ep_per_update": 100,     # episodes to collect before each PPO update
    },
    "TMaze": {
        "cls": TMaze,
        "kwargs": {"corridor_length": 50},
        "total_steps": 300_000,
        "ep_per_update": 80,
    },
    "NoisyCartPole": {
        "cls": NoisyCartPole,
        "kwargs": {"max_steps": 200},
        "total_steps": 300_000,
        "ep_per_update": 50,
    },
    "RepeatCopy": {
        "cls": RepeatCopy,
        "kwargs": {"seq_length": 4, "alphabet_size": 4},
        "total_steps": 200_000,
        "ep_per_update": 200,
    },
}

# Each condition specifies which auxiliary loss to use.
# "none"           = no auxiliary loss (baseline)
# "unweighted"     = predict hidden state, all states weighted equally (weakness)
# "krho_weighted"  = predict hidden state, confident states weighted more (bridge)
# "mdl"            = predict hidden state + KL compression penalty (MDL)
COND_MAP = {
    "baseline":       {"aux": "none"},
    "aux_unweighted": {"aux": "unweighted"},
    "aux_krho":       {"aux": "krho_weighted"},
    "aux_mdl":        {"aux": "mdl"},
}


# =========================================================================
# Neural network agent
#
# The agent has four parts:
#   1. GRU: reads observations one at a time and builds a hidden state
#           (its "memory" of what it has seen so far)
#   2. Policy head: maps hidden state -> action probabilities
#   3. Value head:  maps hidden state -> predicted future reward
#   4. Probe:       maps hidden state -> predicted hidden label
#           (the auxiliary "state prediction" head used by the aux losses)
# =========================================================================

class GRUAgent(nn.Module):
    def __init__(self, obs_dim, hidden_dim, n_actions, n_classes):
        super().__init__()
        self.hidden_dim = hidden_dim

        # GRU: the memory network. Reads observations sequentially.
        self.gru = nn.GRU(obs_dim, hidden_dim, batch_first=True)

        # Policy head: "what action should I take?"
        self.pi = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, n_actions))

        # Value head: "how much future reward do I expect?"
        self.vf = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1))

        # Probe: "what is the hidden state of the environment?"
        # Only used during training (not at test time).
        self.probe = nn.Linear(hidden_dim, n_classes)

    def forward(self, obs_sequence, hidden=None):
        """
        Process a batch of observation sequences through the full network.

        Args:
            obs_sequence: (batch, timesteps, obs_dim) tensor
            hidden: optional GRU hidden state from previous call

        Returns:
            action_logits: (batch, timesteps, n_actions)
            values:        (batch, timesteps)
            gru_output:    (batch, timesteps, hidden_dim) — raw GRU states
            hidden_out:    updated GRU hidden state
        """
        gru_out, hidden_out = self.gru(obs_sequence, hidden)
        action_logits = self.pi(gru_out)
        values = self.vf(gru_out).squeeze(-1)
        return action_logits, values, gru_out, hidden_out

    def act_batch(self, obs_batch, hidden):
        """
        Single-timestep forward pass for batched episode collection.

        Args:
            obs_batch: (n_episodes, 1, obs_dim)
            hidden:    (1, n_episodes, hidden_dim) or None

        Returns:
            action_logits: (n_episodes, n_actions)
            values:        (n_episodes,)
            hidden_out:    updated hidden state
        """
        gru_out, hidden_out = self.gru(obs_batch, hidden)
        action_logits = self.pi(gru_out[:, 0, :])
        values = self.vf(gru_out[:, 0, :]).squeeze(-1)
        return action_logits, values, hidden_out

    def discretise_hidden(self, hidden_states, n_bins=5):
        """
        Snap hidden states to a grid to define representation "cells".
        Two states in the same cell are indistinguishable to the agent.
        Used by the K_rho diagnostic.
        """
        with torch.no_grad():
            scale = n_bins / 2.0
            snapped = torch.round(hidden_states * scale) / scale
            return [str(tuple(row.tolist())) for row in snapped]


# =========================================================================
# Episode collection (batched)
#
# Creates n_ep independent copies of the environment, runs them all
# in parallel, and uses a single batched GRU call per timestep.
# This is ~20x faster than running episodes one at a time.
#
# All inference is done on CPU. The GRU calls here are tiny
# (100 episodes × 1 timestep × 64 hidden dims) so GPU overhead
# would actually slow things down.
# =========================================================================

def collect_episodes(agent, env_cls, env_kwargs, n_episodes):
    """
    Run n_episodes episodes and record everything needed for PPO training.

    Returns a dict with numpy arrays:
        obs:   (n_episodes, max_steps, obs_dim)  — observations
        act:   (n_episodes, max_steps)            — actions taken
        rew:   (n_episodes, max_steps)            — rewards received
        lp:    (n_episodes, max_steps)            — log-prob of action taken
        val:   (n_episodes, max_steps)            — value estimate at each step
        mask:  (n_episodes, max_steps)            — 1.0 for valid steps, 0.0 after
        lab:   (n_episodes, max_steps)            — true hidden label at each step
        lens:  (n_episodes,)                      — episode length
    """
    # Create one environment per episode.
    envs = [env_cls(**env_kwargs) for _ in range(n_episodes)]
    max_len = envs[0].max_steps
    obs_dim = envs[0].observation_space.shape[0]

    # Pre-allocate storage for all episodes.
    obs   = np.zeros((n_episodes, max_len, obs_dim), np.float32)
    act   = np.zeros((n_episodes, max_len), np.int64)
    rew   = np.zeros((n_episodes, max_len), np.float32)
    lp    = np.zeros((n_episodes, max_len), np.float32)
    val   = np.zeros((n_episodes, max_len), np.float32)
    mask  = np.zeros((n_episodes, max_len), np.float32)
    lab   = np.zeros((n_episodes, max_len), np.int64)
    lens  = np.zeros(n_episodes, np.int64)

    # Reset all environments and collect their first observations.
    current_obs = []
    current_info = []
    for env in envs:
        o, info = env.reset()
        current_obs.append(o)
        current_info.append(info)

    # Track which episodes are still running.
    alive = np.ones(n_episodes, dtype=bool)

    # GRU hidden state (shared across all episodes, one slot per episode).
    gru_hidden = None

    for t in range(max_len):
        # ---- Record this timestep's data for alive episodes ----
        for i in range(n_episodes):
            if alive[i]:
                obs[i, t] = current_obs[i]
                lab[i, t] = current_info[i].get("hidden_label", 0)
                mask[i, t] = 1.0

        # ---- One batched GRU forward pass for ALL episodes ----
        # (Dead episodes process zeros; this wastes a little compute
        #  but keeps the batch contiguous, which is faster overall.)
        obs_tensor = torch.FloatTensor(obs[:, t:t+1, :])  # (N, 1, obs_dim)
        with torch.no_grad():
            logits, values, gru_hidden = agent.act_batch(obs_tensor, gru_hidden)

        # ---- Sample actions from the policy ----
        dist = Categorical(logits=logits)
        actions = dist.sample()

        act[:, t] = actions.numpy()
        lp[:, t]  = dist.log_prob(actions).numpy()
        val[:, t] = values.numpy()

        # ---- Step all alive environments ----
        for i in range(n_episodes):
            if not alive[i]:
                continue
            o, r, done, _, info = envs[i].step(act[i, t])
            rew[i, t] = r
            if done:
                alive[i] = False
                lens[i] = t + 1
            else:
                current_obs[i] = o
                current_info[i] = info

        # If every episode is done, stop early.
        if not alive.any():
            break

    # Episodes that ran to the time limit without terminating.
    for i in range(n_episodes):
        if lens[i] == 0:
            lens[i] = max_len

    # Clean up environments.
    for env in envs:
        if hasattr(env, 'close'):
            try:
                env.close()
            except Exception:
                pass

    return {"obs": obs, "act": act, "rew": rew, "lp": lp, "val": val,
            "mask": mask, "lab": lab, "lens": lens}


# =========================================================================
# Generalised Advantage Estimation (GAE)
#
# GAE computes "how much better was this action than expected?" for every
# timestep. This is the signal PPO uses to update the policy.
#
# The computation walks backwards through time, accumulating a
# discounted sum of "temporal difference" errors.
# =========================================================================

def compute_advantages(rewards, values, mask, gamma=0.99, gae_lambda=0.95):
    """
    Compute GAE advantages for all episodes simultaneously.

    Args:
        rewards: (n_episodes, max_steps)
        values:  (n_episodes, max_steps) — value estimates from the network
        mask:    (n_episodes, max_steps) — 1.0 for valid steps

    Returns:
        advantages: (n_episodes, max_steps) — how much better each action was
        returns:    (n_episodes, max_steps) — advantages + values (PPO target)
    """
    n_episodes, max_steps = rewards.shape
    advantages = np.zeros_like(rewards)
    running_gae = np.zeros(n_episodes)

    for t in reversed(range(max_steps)):
        # Is there a valid next step?
        if t < max_steps - 1:
            next_mask = mask[:, t + 1]
            next_value = values[:, t + 1] * next_mask
        else:
            next_mask = np.zeros(n_episodes)
            next_value = np.zeros(n_episodes)

        # TD error: actual reward + discounted next value - predicted value
        td_error = rewards[:, t] + gamma * next_value - values[:, t]

        # Accumulate with exponential decay
        running_gae = td_error + gamma * gae_lambda * running_gae * next_mask

        # Only store for valid timesteps
        advantages[:, t] = running_gae * mask[:, t]

    returns = advantages + values
    return advantages, returns


# =========================================================================
# PPO update
#
# PPO (Proximal Policy Optimisation) updates the policy using the collected
# episodes. It does multiple passes over the data, clipping large updates
# to keep training stable.
#
# The auxiliary losses are applied here:
#   - "unweighted": train the probe to predict the hidden label,
#                   treating all states equally. This is WEAKNESS.
#   - "krho_weighted": same prediction task, but states where the probe
#                      is already confident get higher weight. This is
#                      the bridge theorem's margin weighting.
#   - "mdl": same as unweighted, plus a penalty on the size of the
#            GRU hidden state (KL divergence to a zero-mean Gaussian).
#            This compresses the representation. This is MDL.
# =========================================================================

def ppo_update(agent, rollout, optimiser, aux_mode="none",
               aux_coef=0.5, kl_coef=0.1, n_epochs=4, n_minibatches=4,
               clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01,
               max_grad_norm=0.5, device=PPO_DEVICE):
    """
    Run PPO gradient updates on collected episodes.

    Args:
        agent:      the GRU agent (stays on CPU; we move data to device)
        rollout:    dict from collect_episodes()
        optimiser:  torch optimiser for agent's parameters
        aux_mode:   "none", "unweighted", "krho_weighted", or "mdl"
        aux_coef:   weight of the auxiliary loss relative to PPO loss
        kl_coef:    weight of the KL penalty (only used in "mdl" mode)
        device:     where to run the gradient computation (CPU or GPU)
    """
    # Move data to the compute device (CPU or GPU).
    obs_t  = torch.FloatTensor(rollout["obs"]).to(device)
    act_t  = torch.LongTensor(rollout["act"]).to(device)
    old_lp = torch.FloatTensor(rollout["lp"]).to(device)
    mask_t = torch.FloatTensor(rollout["mask"]).to(device)
    lab_t  = torch.LongTensor(rollout["lab"]).to(device)

    # Compute advantages (on CPU, then move result to device).
    advantages, returns = compute_advantages(
        rollout["rew"], rollout["val"], rollout["mask"])
    adv_t = torch.FloatTensor(advantages).to(device)
    ret_t = torch.FloatTensor(returns).to(device)

    # Normalise advantages (only over valid timesteps).
    valid = mask_t.bool()
    adv_valid = adv_t[valid]
    adv_t[valid] = (adv_valid - adv_valid.mean()) / (adv_valid.std() + 1e-8)

    # Move agent to compute device for gradient computation.
    agent.to(device)

    n_episodes = obs_t.shape[0]
    batch_size = max(1, n_episodes // n_minibatches)
    total_loss = 0.0
    n_updates = 0

    for _epoch in range(n_epochs):
        # Shuffle episodes and process in mini-batches.
        perm = torch.randperm(n_episodes, device=device)

        for start in range(0, n_episodes, batch_size):
            idx = perm[start : start + batch_size]
            B = len(idx)

            # ---- Forward pass through the full observation sequences ----
            logits, values, gru_out, _ = agent(obs_t[idx])

            # ---- PPO policy loss ----
            dist = Categorical(logits=logits)
            new_lp = dist.log_prob(act_t[idx])
            entropy = dist.entropy()
            mb_mask = mask_t[idx]
            mask_sum = mb_mask.sum().clamp(min=1)

            # Probability ratio between new and old policy.
            ratio = torch.exp(new_lp - old_lp[idx]) * mb_mask
            # Clipped surrogate objective (the core of PPO).
            surr1 = ratio * adv_t[idx]
            surr2 = torch.clamp(ratio, 1 - clip_ratio,
                                1 + clip_ratio) * adv_t[idx]
            policy_loss = -torch.min(surr1, surr2).sum() / mask_sum

            # ---- Value loss (mean squared error) ----
            value_loss = 0.5 * ((values - ret_t[idx])**2 * mb_mask
                                ).sum() / mask_sum

            # ---- Entropy bonus (encourages exploration) ----
            entropy_loss = -(entropy * mb_mask).sum() / mask_sum

            # ---- Total PPO loss ----
            loss = (policy_loss
                    + value_coef * value_loss
                    + entropy_coef * entropy_loss)

            # ---- Auxiliary loss (if any) ----
            if aux_mode != "none":
                # Find the last valid timestep for each episode in this batch.
                ep_lengths = mb_mask.sum(dim=1).long()
                last_step = (ep_lengths - 1).clamp(min=0)

                # Extract the GRU hidden state at the last valid timestep.
                batch_idx = torch.arange(B, device=device)
                h_final = gru_out[batch_idx, last_step]

                # The true hidden label at the last valid timestep.
                true_label = lab_t[idx][batch_idx, last_step]

                # Probe prediction: what does the network think the hidden
                # label is, given its representation?
                probe_logits = agent.probe(h_final)

                # Cross-entropy: how wrong is the probe's prediction?
                ce_per_sample = nn.functional.cross_entropy(
                    probe_logits, true_label, reduction='none')

                if aux_mode == "unweighted":
                    # WEAKNESS: all states contribute equally.
                    aux_loss = ce_per_sample.mean()

                elif aux_mode == "krho_weighted":
                    # BRIDGE: states where the probe is confident (high margin)
                    # get higher weight. States near the decision boundary
                    # (coin-flip predictions) get low weight.
                    with torch.no_grad():
                        probs = torch.softmax(probe_logits, dim=1)
                        p_correct = probs[batch_idx, true_label]
                        margin = torch.abs(p_correct - 0.5)
                        # rho: 0 when p=0.5 (coin flip), 1 when p=1.0 (certain)
                        rho = 2 * margin / (0.5 + margin)
                    aux_loss = (rho * ce_per_sample).mean()

                elif aux_mode == "mdl":
                    # MDL: same uniform prediction as "unweighted", PLUS
                    # a penalty on the representation's complexity.
                    #
                    # The penalty is KL(N(h, I) || N(0, I)) = 0.5 * ||h||^2.
                    # This pushes the hidden state toward zero (maximum entropy),
                    # forcing the encoder to compress: only information that
                    # actually helps predict the hidden label survives.
                    prediction_loss = ce_per_sample.mean()
                    compression_penalty = 0.5 * (h_final ** 2).mean()
                    aux_loss = prediction_loss + kl_coef * compression_penalty

                else:
                    aux_loss = torch.tensor(0.0, device=device)

                loss = loss + aux_coef * aux_loss

            # ---- Gradient step ----
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimiser.step()

            total_loss += loss.item()
            n_updates += 1

    # Move agent back to CPU (collect_episodes expects CPU).
    agent.to('cpu')

    return total_loss / max(n_updates, 1)


# =========================================================================
# K_rho diagnostic
#
# K_rho measures how much "representational aliasing" the agent's encoder
# has: how often does it put two different-label states into the same cell?
#
# It's computed by:
#   1. Running episodes and recording the GRU hidden state at each last step
#   2. Discretising those hidden states into cells (snapping to a grid)
#   3. For each cell, measuring how mixed the labels are, weighted by
#      the probe's confidence (margin)
#   4. Summing the minimum of each label's weight per cell
#
# K_rho = 0 means the representation perfectly separates all labels.
# K_rho > 0 means some cells contain states with different labels.
# =========================================================================

def compute_krho(agent, env_cls, env_kwargs, n_episodes=100, n_bins=5):
    """
    Compute K_rho and related diagnostics.

    Returns:
        k_rho:        the margin-weighted aliasing cost
        accuracy:     fraction of correct last-step actions
        mean_reward:  average episode reward
        n_cells:      number of distinct representation cells
    """
    # Collect episodes with the current agent (always on CPU).
    rollout = collect_episodes(agent, env_cls, env_kwargs, n_episodes)

    # Run the full observation sequences through the GRU to get hidden states.
    obs_tensor = torch.FloatTensor(rollout["obs"])
    with torch.no_grad():
        _, _, gru_out, _ = agent(obs_tensor)

    # Find each episode's last valid timestep.
    ep_lengths = rollout["mask"].sum(axis=1).astype(int)
    last_indices = np.clip(ep_lengths - 1, 0, None)

    # Extract hidden state and label at the last valid timestep.
    batch_range = torch.arange(n_episodes)
    h_final = gru_out[batch_range, torch.from_numpy(last_indices)]
    final_labels = np.array([rollout["lab"][i, last_indices[i]]
                             for i in range(n_episodes)])

    # Discretise hidden states into cells.
    cells = agent.discretise_hidden(h_final, n_bins)

    # Get probe confidence for each sample.
    with torch.no_grad():
        probe_probs = torch.softmax(agent.probe(h_final), dim=1)

    # Compute K_rho: for each cell, accumulate margin-weighted mass per label,
    # then sum the minimum across labels.
    uniform_weight = 1.0 / n_episodes
    cell_label_mass = defaultdict(lambda: defaultdict(float))
    for i in range(n_episodes):
        label = final_labels[i]
        p_correct = probe_probs[i, label].item()
        margin = abs(p_correct - 0.5)
        rho = (2 * margin) / (0.5 + margin) if (0.5 + margin) > 0 else 0.0
        cell_label_mass[cells[i]][label] += uniform_weight * rho

    # K_rho: sum of min label mass per cell (cells with only one label contribute 0).
    k_rho = sum(
        min(label_masses.values()) if len(label_masses) > 1 else 0.0
        for label_masses in cell_label_mass.values()
    )

    # Mean episode reward.
    ep_rewards = [rollout["rew"][i, :ep_lengths[i]].sum()
                  for i in range(n_episodes)]
    mean_reward = float(np.mean(ep_rewards))

    # Decision accuracy: did the agent's last action match the hidden label?
    correct = sum(
        1 for i in range(n_episodes)
        if rollout["act"][i, max(0, last_indices[i])] == final_labels[i]
    )
    accuracy = correct / n_episodes

    return {
        "k_rho": float(k_rho),
        "accuracy": float(accuracy),
        "mean_reward": mean_reward,
        "n_cells": len(set(cells)),
    }


# =========================================================================
# Training loop for one (environment, condition) pair
# =========================================================================

def train_agent(env_name, condition, seed, hidden_dim=64, lr=3e-4,
                aux_coef=0.5, eval_every=10_000):
    """
    Train a GRU agent with PPO on one environment under one condition.

    Args:
        env_name:   which environment (key in ENV_CONFIGS)
        condition:  which condition (key in COND_MAP)
        seed:       random seed for reproducibility
        eval_every: how often (in env steps) to run the K_rho diagnostic

    Returns:
        List of evaluation snapshots, each a dict with k_rho, accuracy,
        mean_reward, n_cells, and step count.
    """
    config = ENV_CONFIGS[env_name]
    cond = COND_MAP[condition]

    # Set random seeds for reproducibility.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Figure out environment dimensions by briefly creating one.
    probe_env = config["cls"](**config["kwargs"])
    probe_env.reset(seed=seed)
    n_actions = probe_env.action_space.n
    obs_dim = probe_env.observation_space.shape[0]
    n_classes = getattr(probe_env, 'N_HIDDEN_CLASSES', 2)
    if hasattr(probe_env, 'close'):
        probe_env.close()

    # Create agent and optimiser.
    agent = GRUAgent(obs_dim, hidden_dim, n_actions, n_classes)
    optimiser = torch.optim.Adam(agent.parameters(), lr=lr)

    steps_done = 0
    eval_history = []

    while steps_done < config["total_steps"]:
        # ---- Collect episodes ----
        rollout = collect_episodes(
            agent, config["cls"], config["kwargs"], config["ep_per_update"])
        steps_done += int(rollout["mask"].sum())

        # ---- PPO update ----
        ppo_update(agent, rollout, optimiser, aux_mode=cond["aux"],
                   aux_coef=aux_coef)

        # ---- Periodic evaluation ----
        if not eval_history or steps_done - eval_history[-1]["step"] >= eval_every:
            diagnostics = compute_krho(
                agent, config["cls"], config["kwargs"], n_episodes=100)
            diagnostics["step"] = steps_done
            eval_history.append(diagnostics)

    return eval_history


# =========================================================================
# Main experiment runner
# =========================================================================

def run_full_experiment():
    print("=" * 70)
    print("RL EXPERIMENT: weakness vs compression vs biased weighting")
    print("=" * 70)
    print(f"PPO device: {PPO_DEVICE}")
    print(f"CPU threads: {torch.get_num_threads()}")

    env_names = ["CorridorMemory", "TMaze", "NoisyCartPole", "RepeatCopy"]
    conditions = ["baseline", "aux_unweighted", "aux_krho", "aux_mdl"]
    n_seeds = 10

    print(f"Environments: {env_names}")
    print(f"Conditions:   {conditions}")
    print(f"Seeds:        {n_seeds}")
    print()

    # ---- Load partial results if resuming from a crash ----
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            results = json.load(f)
        done = len(results.get("env_results", {}))
        print(f"Resuming from checkpoint ({done} pairs already done)")
    else:
        results = {
            "version": "v28_rl",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": str(PPO_DEVICE),
            "n_seeds": n_seeds,
            "envs": env_names,
            "conds": conditions,
            "env_results": {},
        }

    overall_start = time.perf_counter()

    for env_name in env_names:
        for condition in conditions:
            key = f"{env_name}__{condition}"

            # Skip if this pair was already completed in a previous run.
            if key in results.get("env_results", {}):
                print(f"  [{key}] already done, skipping.")
                continue

            print(f"\n--- {env_name} / {condition} ---")
            pair_start = time.perf_counter()
            all_curves = []

            for seed_idx in range(n_seeds):
                seed = seed_idx + 1000
                curve = train_agent(env_name, condition, seed)
                all_curves.append(curve)

                # Progress update every 5 seeds.
                if (seed_idx + 1) % 5 == 0:
                    last = curve[-1]
                    print(f"  Seed {seed_idx+1}/{n_seeds}: "
                          f"rew={last['mean_reward']:.2f} "
                          f"K={last['k_rho']:.4f}")

            pair_elapsed = time.perf_counter() - pair_start

            # ---- Compute summary statistics across seeds ----
            # Standard error of the mean.
            def se(values):
                if len(values) > 1:
                    return float(np.std(values, ddof=1) / np.sqrt(len(values)))
                return 0.0

            # Curves may have different lengths (variable-length episodes
            # cause eval checkpoints to land at different step counts).
            # Align to the shortest curve and interpolate.
            min_curve_len = min(len(c) for c in all_curves)
            ref_steps = [all_curves[0][i]["step"] for i in range(min_curve_len)]

            summary = []
            for step_idx in range(min_curve_len):
                target_step = ref_steps[step_idx]
                row = {"step": target_step}
                for metric in ["mean_reward", "k_rho", "accuracy"]:
                    # For each seed, find the eval point closest to target_step.
                    values = []
                    for curve in all_curves:
                        closest = min(range(len(curve)),
                                      key=lambda j: abs(curve[j]["step"]
                                                        - target_step))
                        values.append(curve[closest][metric])
                    row[f"{metric}_mean"] = round(float(np.mean(values)), 4)
                    row[f"{metric}_se"] = round(se(values), 4)
                summary.append(row)

            final = summary[-1]
            print(f"  Final: rew={final['mean_reward_mean']:.3f}"
                  f"+/-{final['mean_reward_se']:.3f}"
                  f" K={final['k_rho_mean']:.4f}"
                  f"+/-{final['k_rho_se']:.4f}"
                  f" ({pair_elapsed:.0f}s)")

            # ---- Save this pair's results ----
            results.setdefault("env_results", {})[key] = {
                "env": env_name,
                "cond": condition,
                "time_s": round(pair_elapsed, 1),
                "summary": summary,
                "raw": [
                    [{k: round(v, 5) if isinstance(v, float) else v
                      for k, v in point.items()}
                     for point in curve]
                    for curve in all_curves
                ],
            }

            # Save after every pair (crash safety).
            with open(OUT_PATH, "w") as f:
                json.dump(results, f, indent=2)
            print(f"  Saved checkpoint.")

    # ---- Final summary ----
    total_elapsed = time.perf_counter() - overall_start
    results["total_s"] = round(total_elapsed, 1)
    results["total_min"] = round(total_elapsed / 60, 1)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for env_name in env_names:
        for condition in conditions:
            key = f"{env_name}__{condition}"
            if key in results.get("env_results", {}):
                f = results["env_results"][key]["summary"][-1]
                print(f"  {env_name:>16} {condition:>16}"
                      f" rew={f['mean_reward_mean']:>7.3f}"
                      f"+/-{f['mean_reward_se']:.3f}"
                      f" K={f['k_rho_mean']:>7.4f}"
                      f"+/-{f['k_rho_se']:.4f}")
        print()

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Total: {total_elapsed/60:.1f}min  Written: {OUT_PATH}")


if __name__ == "__main__":
    run_full_experiment()
