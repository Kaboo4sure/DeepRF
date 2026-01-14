import os
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.env.maintenance_NASABearing_env import NASABearingMaintenanceEnv

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)

def discount_cumsum(x, gamma):
    """Compute discounted cumulative sums."""
    y = np.zeros_like(x, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(x))):
        running = x[t] + gamma * running
        y[t] = running
    return y

# -----------------------------
# Policy / Value Networks
# -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)
        self.cvalue_head = nn.Linear(128, 1)  # constraint value

    def forward(self, obs):
        h = self.shared(obs)
        logits = self.policy_head(h)
        v = self.value_head(h).squeeze(-1)
        vc = self.cvalue_head(h).squeeze(-1)
        return logits, v, vc

    def step(self, obs):
        logits, v, vc = self.forward(obs)
        dist = Categorical(logits=logits)
        a = dist.sample()
        logp = dist.log_prob(a)
        return a, logp, v, vc

# -----------------------------
# PPO-Lagrangian Trainer
# -----------------------------
def train(
    total_iters=200,
    steps_per_iter=4000,
    gamma=0.99,
    lam=0.95,
    clip_ratio=0.2,
    pi_lr=3e-4,
    vf_lr=1e-3,
    train_epochs=10,
    minibatch_size=512,
    target_kl=0.02,
    seed=42,
    # constraint settings
    cost_limit=0.05,
    lambda_lr=0.05,
    # env settings
    model_dir="models/ensemble_rul_sim",
    n_models=5,
    rul_min=15.0,
    max_steps=300,                  # ✅ ADDED
    device=None,
    log_dir="runs/ims_bearing/ppo_lagrangian",
    debug_first_iter=False,         # ✅ optional
):
    set_seed(seed)
    os.makedirs(log_dir, exist_ok=True)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Bearing env, max_steps passed in
    env = NASABearingMaintenanceEnv(
        runs_pkl="src/data/data/raw/ims_bearing/runs.pkl",
        model_dir="src/data/models/ensemble_rul_bearing",
        rul_min=rul_min,
        max_steps=max_steps,
        c_failure=200.0,
        c_operate=0.2,
        c_inspect=1.0,
        c_minor=8.0,
        c_replace=25.0,
    )

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    ac = ActorCritic(obs_dim, act_dim).to(device)
    pi_opt = torch.optim.Adam(ac.parameters(), lr=pi_lr)
    vf_opt = torch.optim.Adam(ac.parameters(), lr=vf_lr)

    # Lagrange multiplier
    lam_mult = torch.tensor(0.0, device=device)

    # Rollout buffers
    buf_obs  = np.zeros((steps_per_iter, obs_dim), dtype=np.float32)
    buf_act  = np.zeros((steps_per_iter,), dtype=np.int64)
    buf_logp = np.zeros((steps_per_iter,), dtype=np.float32)
    buf_rew  = np.zeros((steps_per_iter,), dtype=np.float32)
    buf_cost = np.zeros((steps_per_iter,), dtype=np.float32)
    buf_val  = np.zeros((steps_per_iter,), dtype=np.float32)
    buf_cval = np.zeros((steps_per_iter,), dtype=np.float32)
    buf_done = np.zeros((steps_per_iter,), dtype=np.float32)

    def compute_gae(rews, vals, dones, gamma, lam):
        adv = np.zeros_like(rews, dtype=np.float32)
        lastgaelam = 0.0
        for t in reversed(range(len(rews))):
            nonterminal = 1.0 - dones[t]
            nextval = vals[t + 1] if t + 1 < len(vals) else 0.0
            delta = rews[t] + gamma * nextval * nonterminal - vals[t]
            lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
            adv[t] = lastgaelam
        return adv

    history = []

    obs, info = env.reset(seed=seed)
    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
    ep_rets, ep_costs, ep_lens = [], [], []
    start_time = time.time()

    for it in range(1, total_iters + 1):
        # -------------------------
        # Collect rollouts
        # -------------------------
        for t in range(steps_per_iter):
            buf_obs[t] = obs

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, logp, v, vc = ac.step(obs_t)
            a = int(a.item())

            next_obs, reward, terminated, truncated, info = env.step(a)
            done = terminated or truncated

            if debug_first_iter and it == 1 and t < 30:
                print(
                    "DBG", t,
                    "a", a,
                    "term", terminated,
                    "trunc", truncated,
                    "env_t", info.get("t"),
                    "true_rul", info.get("true_rul"),
                    "T", getattr(env, "current_T", None),
                    "ep_step", getattr(env, "episode_step", None),
                )

            cost = float(info.get("constraint_cost", 0.0))

            buf_act[t]  = a
            buf_logp[t] = float(logp.item())
            buf_rew[t]  = float(reward)
            buf_cost[t] = float(cost)
            buf_val[t]  = float(v.item())
            buf_cval[t] = float(vc.item())
            buf_done[t] = float(done)

            ep_ret += reward
            ep_cost += cost
            ep_len += 1

            obs = next_obs

            if done:
                ep_rets.append(ep_ret)
                ep_costs.append(ep_cost)
                ep_lens.append(ep_len)
                obs, info = env.reset()
                ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

        # Reward advantages
        adv_r = compute_gae(buf_rew, buf_val, buf_done, gamma, lam)
        ret_r = adv_r + buf_val

        # Cost advantages
        adv_c = compute_gae(buf_cost, buf_cval, buf_done, gamma, lam)
        ret_c = adv_c + buf_cval

        # Normalize advantages
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + 1e-8)
        adv_c = adv_c - adv_c.mean()

        # Convert to tensors
        obs_t      = torch.tensor(buf_obs,  dtype=torch.float32, device=device)
        act_t      = torch.tensor(buf_act,  dtype=torch.int64,   device=device)
        logp_old_t = torch.tensor(buf_logp, dtype=torch.float32, device=device)
        adv_r_t    = torch.tensor(adv_r,    dtype=torch.float32, device=device)
        adv_c_t    = torch.tensor(adv_c,    dtype=torch.float32, device=device)
        ret_r_t    = torch.tensor(ret_r,    dtype=torch.float32, device=device)
        ret_c_t    = torch.tensor(ret_c,    dtype=torch.float32, device=device)

        # -------------------------
        # Update Lagrange multiplier
        # -------------------------
        N = 30
        lambda_max = 20.0
        cost_limit_step = float(cost_limit)

        if len(ep_costs) > 0 and len(ep_lens) > 0:
            avg_ep_cost = float(np.mean(ep_costs[-N:]))
            avg_ep_len  = float(np.mean(ep_lens[-N:]))
        else:
            avg_ep_cost = float(np.sum(buf_cost))
            avg_ep_len  = float(steps_per_iter)

        avg_step_cost = avg_ep_cost / max(1.0, avg_ep_len)

        eps_deadband = 0.002
        diff = avg_step_cost - cost_limit_step
        if abs(diff) < eps_deadband:
            diff = 0.0

        max_delta = 0.03
        lam_update = float(np.clip(lambda_lr * diff, -max_delta, max_delta))

        lam_val = float(lam_mult.item())
        lam_val = lam_val + lam_update
        lam_val = max(0.0, min(lambda_max, lam_val))
        lam_mult = torch.tensor(lam_val, device=device)

        # -------------------------
        # PPO Updates
        # -------------------------
        n = steps_per_iter
        idxs = np.arange(n)

        approx_kl = 0.0
        for epoch in range(train_epochs):
            np.random.shuffle(idxs)
            for start in range(0, n, minibatch_size):
                mb = idxs[start:start + minibatch_size]

                logits, v, vc = ac(obs_t[mb])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(act_t[mb])
                ratio = torch.exp(logp - logp_old_t[mb])

                adv_comb = adv_r_t[mb] - lam_mult.detach() * adv_c_t[mb]

                unclipped = ratio * adv_comb
                clipped   = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv_comb
                pi_loss   = -(torch.min(unclipped, clipped)).mean()

                v_loss  = ((v  - ret_r_t[mb]) ** 2).mean()
                vc_loss = ((vc - ret_c_t[mb]) ** 2).mean()

                ent = dist.entropy().mean()
                ent_bonus = 0.01 * ent

                loss = pi_loss + 0.5 * v_loss + 0.5 * vc_loss - ent_bonus

                pi_opt.zero_grad()
                vf_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                pi_opt.step()
                vf_opt.step()

                with torch.no_grad():
                    approx_kl = (logp_old_t[mb] - logp).mean().abs().item()

            if approx_kl > 1.5 * target_kl:
                break

        # -------------------------
        # Save + log
        # -------------------------
        avg_ret  = float(np.mean(ep_rets[-10:]))  if len(ep_rets)  >= 1 else float(np.mean(buf_rew))
        avg_cost = float(np.mean(ep_costs[-10:])) if len(ep_costs) >= 1 else float(np.mean(buf_cost))
        avg_len  = float(np.mean(ep_lens[-10:]))  if len(ep_lens)  >= 1 else 0.0
        avg_p_unsafe_per_step = avg_cost / max(1.0, avg_len)

        log = {
            "iter": it,
            "avg_return_last10eps": avg_ret,
            "avg_cost_last10eps": avg_cost,
            "avg_len_last10eps": avg_len,
            "avg_p_unsafe_per_step_last10eps": float(avg_p_unsafe_per_step),
            "lambda": float(lam_mult.item()),
            "approx_kl": float(approx_kl),
            "avg_step_cost_lastN": float(avg_step_cost),
            "cost_limit_step": float(cost_limit_step),
            "lambda_update_delta": float(lam_update),
            "lambda_value": float(lam_mult.item()),
            "time_elapsed_sec": float(time.time() - start_time),
        }
        history.append(log)

        if it % 10 == 0 or it == 1:
            print(json.dumps(log, indent=2), flush=True)

        if it % 50 == 0:
            with open(os.path.join(log_dir, "progress.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(log) + "\n")

    torch.save(
        {"ac_state_dict": ac.state_dict(), "lambda": float(lam_mult.item())},
        os.path.join(log_dir, "final.pt"),
    )
    with open(os.path.join(log_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("Training complete. Saved to:", log_dir)


if __name__ == "__main__":
    train(
        total_iters=200,
        steps_per_iter=4000,
        seed=42,
        cost_limit=0.01,
        lambda_lr=0.3,
        rul_min=100.0,
        max_steps=100000,  # ✅ ADDED
        model_dir="models/ensemble_rul_sim",
        n_models=5,
        debug_first_iter=False,
    )
