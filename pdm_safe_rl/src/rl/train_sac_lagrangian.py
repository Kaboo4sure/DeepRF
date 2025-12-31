# src/rl/train_sac_lagrangian.py
import os
import json
import time
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gymnasium as gym

# calling from where the environment is registered
from src.env.maintenance_env import MaintenanceEnv


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


# ----------------------------
# Replay Buffer (DISCRETE ACTIONS)
# ----------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, size=1_000_000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)

        # DISCRETE actions stored as int64
        self.act_buf = np.zeros((size,), dtype=np.int64)

        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.cost_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, cost, obs2, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = int(act)
        self.rew_buf[self.ptr] = float(rew)
        self.cost_buf[self.ptr] = float(cost)
        self.obs2_buf[self.ptr] = obs2
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            cost=self.cost_buf[idxs],
            done=self.done_buf[idxs],
        )


# ----------------------------
# Networks
# ----------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class DiscretePolicy(nn.Module):
    """
    Categorical policy pi(a|s) for discrete SAC.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim],
                              activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs):
        return self.logits_net(obs)  # logits

    def dist(self, obs):
        logits = self.forward(obs)
        return torch.distributions.Categorical(logits=logits)

    def sample(self, obs):
        dist = self.dist(obs)
        a = dist.sample()  # (B,)
        logp = dist.log_prob(a).unsqueeze(-1)  # (B,1)
        return a, logp


class DiscreteQ(nn.Module):
    """
    Q(s, ·) outputs Q-values for all discrete actions.
    """
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation=nn.ReLU)

    def forward(self, obs):
        return self.q(obs)  # (B, act_dim)


# ----------------------------
# Config
# ----------------------------
@dataclass
class SACConfig:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # env
    max_steps: int = 300

    # training
    total_steps: int = 300_000
    start_steps: int = 2_000          # random actions before policy
    update_after: int = 2_000
    update_every: int = 50            # how often to update (in env steps)
    updates_per_step: int = 1          # gradient steps per env step when updating
    batch_size: int = 256
    replay_size: int = 500_000

    # SAC
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float = None  # if None -> 0.8 * log(act_dim)

    # constraint / lagrangian
    cost_limit_step: float = 0.01
    lambda_lr: float = 0.01
    lambda_max: float = 50.0

    # dual update averaging window (steps)
    dual_avg_window: int = 2000

    # logging
    log_every_steps: int = 10_000
    run_dir: str = "runs/sac_lagrangian"


# ----------------------------
# Main
# ----------------------------
def main(cfg: SACConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)

    env = MaintenanceEnv(max_steps=cfg.max_steps)
    assert isinstance(env.action_space, gym.spaces.Discrete), \
        "Discrete SAC-Lagrangian expects env.action_space = gym.spaces.Discrete."

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    device = torch.device(cfg.device)

    # networks
    actor = DiscretePolicy(obs_dim, act_dim).to(device)

    # reward critics (twin)
    q1r = DiscreteQ(obs_dim, act_dim).to(device)
    q2r = DiscreteQ(obs_dim, act_dim).to(device)
    q1r_targ = DiscreteQ(obs_dim, act_dim).to(device)
    q2r_targ = DiscreteQ(obs_dim, act_dim).to(device)
    q1r_targ.load_state_dict(q1r.state_dict())
    q2r_targ.load_state_dict(q2r.state_dict())

    # cost critics (twin)
    q1c = DiscreteQ(obs_dim, act_dim).to(device)
    q2c = DiscreteQ(obs_dim, act_dim).to(device)
    q1c_targ = DiscreteQ(obs_dim, act_dim).to(device)
    q2c_targ = DiscreteQ(obs_dim, act_dim).to(device)
    q1c_targ.load_state_dict(q1c.state_dict())
    q2c_targ.load_state_dict(q2c.state_dict())

    # optimizers
    pi_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q_opt = optim.Adam(
        list(q1r.parameters()) + list(q2r.parameters()) +
        list(q1c.parameters()) + list(q2c.parameters()),
        lr=cfg.critic_lr
    )

    # entropy temperature (learned)
    log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.alpha_lr)

    # lagrange multiplier (dual variable)
    lam_val = 0.0

    # target entropy (discrete)
    if cfg.target_entropy is None:
        target_entropy = 0.8 * float(np.log(act_dim))
    else:
        target_entropy = float(cfg.target_entropy)

    # replay
    replay = ReplayBuffer(obs_dim, size=cfg.replay_size)

    # logging containers
    history = []
    ep_returns, ep_costs, ep_lens = [], [], []
    step_costs = []  # per-step constraint cost history for dual update

    t0 = time.time()
    obs, info = env.reset(seed=cfg.seed)
    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

    def soft_update(net, targ, tau):
        with torch.no_grad():
            for p, p_t in zip(net.parameters(), targ.parameters()):
                p_t.data.mul_(1 - tau)
                p_t.data.add_(tau * p.data)

    def compute_stats(lastN=10):
        if len(ep_returns) < 1:
            return dict(
                avg_return_last10eps=0.0,
                avg_cost_last10eps=0.0,
                avg_len_last10eps=0.0,
                avg_p_unsafe_per_step_last10eps=0.0,
            )
        N = min(lastN, len(ep_returns))
        # avg per-step unsafe over recent steps (windowed)
        recent_steps = step_costs[-max(1, 1000):] if len(step_costs) else [0.0]
        return dict(
            avg_return_last10eps=float(np.mean(ep_returns[-N:])),
            avg_cost_last10eps=float(np.mean(ep_costs[-N:])),
            avg_len_last10eps=float(np.mean(ep_lens[-N:])),
            avg_p_unsafe_per_step_last10eps=float(np.mean(recent_steps)),
        )

    # ----------------------------
    # Training loop
    # ----------------------------
    for step in range(1, cfg.total_steps + 1):

        # action selection
        if step < cfg.start_steps:
            a_env = env.action_space.sample()
        else:
            with torch.no_grad():
                o_t = to_tensor(obs, device).unsqueeze(0)
                a_t, _ = actor.sample(o_t)
                a_env = int(a_t.item())

        # step env
        obs2, rew, terminated, truncated, info = env.step(a_env)
        done = bool(terminated or truncated)

        cost = float(info.get("constraint_cost", 0.0))
        step_costs.append(cost)

        # store transition
        replay.store(obs, a_env, rew, cost, obs2, float(done))

        obs = obs2
        ep_ret += float(rew)
        ep_cost += float(cost)
        ep_len += 1

        # end episode
        if done:
            ep_returns.append(ep_ret)
            ep_costs.append(ep_cost)
            ep_lens.append(ep_len)
            obs, info = env.reset()
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

        # updates
        if step >= cfg.update_after and (step % cfg.update_every == 0):
            num_updates = cfg.update_every * cfg.updates_per_step
            for _ in range(num_updates):
                batch = replay.sample_batch(cfg.batch_size)
                o = to_tensor(batch["obs"], device)
                o2 = to_tensor(batch["obs2"], device)
                a_idx = torch.as_tensor(batch["act"], device=device, dtype=torch.long)  # (B,)
                r = to_tensor(batch["rew"], device).unsqueeze(-1)   # (B,1)
                c = to_tensor(batch["cost"], device).unsqueeze(-1)  # (B,1)
                d = to_tensor(batch["done"], device).unsqueeze(-1)  # (B,1)

                alpha = log_alpha.exp()

                # ---------- Targets ----------
                with torch.no_grad():
                    dist2 = actor.dist(o2)
                    pi2 = dist2.probs                         # (B,A)
                    logpi2 = torch.log(pi2 + 1e-8)            # (B,A)

                    # reward soft value V_r(s')
                    q1r_next = q1r_targ(o2)
                    q2r_next = q2r_targ(o2)
                    q_r_next = torch.min(q1r_next, q2r_next)  # (B,A)
                    v_r = (pi2 * (q_r_next - alpha.detach() * logpi2)).sum(dim=-1, keepdim=True)
                    backup_r = r + cfg.gamma * (1 - d) * v_r

                    # cost value V_c(s') (no entropy)
                    q1c_next = q1c_targ(o2)
                    q2c_next = q2c_targ(o2)
                    q_c_next = torch.min(q1c_next, q2c_next)  # (B,A)
                    v_c = (pi2 * q_c_next).sum(dim=-1, keepdim=True)
                    backup_c = c + cfg.gamma * (1 - d) * v_c

                # ---------- Critic losses ----------
                q1r_sa = q1r(o).gather(1, a_idx.unsqueeze(1))
                q2r_sa = q2r(o).gather(1, a_idx.unsqueeze(1))
                q1c_sa = q1c(o).gather(1, a_idx.unsqueeze(1))
                q2c_sa = q2c(o).gather(1, a_idx.unsqueeze(1))

                q_loss = ((q1r_sa - backup_r) ** 2).mean() + ((q2r_sa - backup_r) ** 2).mean() + \
                         ((q1c_sa - backup_c) ** 2).mean() + ((q2c_sa - backup_c) ** 2).mean()

                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                # ---------- Actor loss ----------
                dist = actor.dist(o)
                pi = dist.probs
                logpi = torch.log(pi + 1e-8)

                q_r_min = torch.min(q1r(o), q2r(o))  # (B,A)
                q_c_min = torch.min(q1c(o), q2c(o))  # (B,A)

                lam = torch.tensor(lam_val, dtype=torch.float32, device=device)

                # minimize: E_a[ alpha log pi - Qr + lam Qc ]
                pi_loss = (pi * (alpha.detach() * logpi - q_r_min + lam * q_c_min)).sum(dim=-1).mean()

                pi_opt.zero_grad()
                pi_loss.backward()
                pi_opt.step()

                # ---------- Alpha (temperature) loss ----------
                # entropy = -sum pi log pi
                entropy = -(pi * logpi).sum(dim=-1).mean()
                alpha_loss = -(log_alpha * (entropy.detach() - target_entropy))

                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # ---------- Soft update targets ----------
                soft_update(q1r, q1r_targ, cfg.tau)
                soft_update(q2r, q2r_targ, cfg.tau)
                soft_update(q1c, q1c_targ, cfg.tau)
                soft_update(q2c, q2c_targ, cfg.tau)

            # ---------- Dual update ----------
            if len(step_costs) > 0:
                avg_step_cost = float(np.mean(step_costs[-cfg.dual_avg_window:]))
            else:
                avg_step_cost = 0.0

            lam_update = cfg.lambda_lr * (avg_step_cost - cfg.cost_limit_step)
            lam_val = float(np.clip(lam_val + lam_update, 0.0, cfg.lambda_max))

        # periodic logging to history.json
        # periodic logging to history.json
        if (step % cfg.log_every_steps == 0) or (step == cfg.total_steps):
            stats = compute_stats(lastN=10)

            entry = {
                "iter": int(step),
                "avg_return_last10eps": float(stats["avg_return_last10eps"]),
                "avg_cost_last10eps": float(stats["avg_cost_last10eps"]),
                "avg_len_last10eps": float(stats["avg_len_last10eps"]),
                "avg_p_unsafe_per_step_last10eps": float(stats["avg_p_unsafe_per_step_last10eps"]),
                "lambda": float(lam_val),
                "cost_limit_step": float(cfg.cost_limit_step),
                "alpha": float(log_alpha.exp().item()),
                "time_elapsed_sec": float(time.time() - t0),
            }

            history.append(entry)

            with open(os.path.join(cfg.run_dir, "history.json"), "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

            print(json.dumps(entry, indent=2))

    print(f"Training complete. Saved to: {cfg.run_dir}")


if __name__ == "__main__":
    cfg = SACConfig(
        total_steps=300_000,
        lambda_lr=0.01,
        cost_limit_step=0.01,
        run_dir="runs/sac_lagrangian",
        log_every_steps=10_000,
    )
    main(cfg)
