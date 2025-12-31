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

# If your env is registered elsewhere, you can replace this with gym.make("YourEnvId")
from src.envs.maintenance_env import MaintenanceEnv


# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size=1_000_000):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size,), dtype=np.float32)
        self.cost_buf = np.zeros((size,), dtype=np.float32)
        self.done_buf = np.zeros((size,), dtype=np.float32)

        self.max_size = size
        self.ptr = 0
        self.size = 0

    def store(self, obs, act, rew, cost, obs2, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.cost_buf[self.ptr] = cost
        self.obs2_buf[self.ptr] = obs2
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            cost=self.cost_buf[idxs],
            done=self.done_buf[idxs],
        )
        return batch


# ----------------------------
# Networks
# ----------------------------
def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation=nn.ReLU, output_activation=nn.ReLU)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, obs):
        mu, std = self.forward(obs)
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()  # reparameterization trick
        a = torch.tanh(x_t)
        # log prob correction for tanh squashing
        logp = dist.log_prob(x_t).sum(-1, keepdim=True)
        logp -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, logp, torch.tanh(mu)


class QCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation=nn.ReLU)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q(x)


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
    start_steps: int = 2_000  # random actions before policy
    update_after: int = 2_000
    update_every: int = 50
    batch_size: int = 256

    # SAC
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    init_alpha: float = 0.2
    target_entropy: float = None  # if None -> -act_dim

    # constraint / lagrangian
    cost_limit_step: float = 0.01
    lambda_lr: float = 0.01
    lambda_max: float = 50.0

    # logging
    log_every_iters: int = 10_000  # by steps
    run_dir: str = "runs/sac_lagrangian"


# ----------------------------
# Main
# ----------------------------
def main(cfg: SACConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.run_dir, exist_ok=True)

    # ===== IMPORTANT NOTE =====
    # SAC is for continuous actions.
    # If your env action_space is Discrete(4), you should either:
    #  (A) switch to continuous maintenance intensity action, OR
    #  (B) use PPO for discrete and keep SAC for continuous version.
    #
    # For now, we assume a continuous action env OR a wrapper that maps tanh(-1,1) to {0,1,2,3}.
    #
    env = MaintenanceEnv(max_steps=cfg.max_steps)
    obs_dim = env.observation_space.shape[0]

    # --- SIMPLE DISCRETE WRAPPER OPTION ---
    # If env.action_space is Discrete(4), we map a ∈ [-1,1] to {0,1,2,3}.
    is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
    if is_discrete:
        act_dim = 1
        n_discrete = env.action_space.n
        def map_action(a_cont):
            # a_cont shape (1,) in [-1,1]
            # map to bins [0..n-1]
            x = float(a_cont[0])
            idx = int(np.clip(((x + 1) / 2) * n_discrete, 0, n_discrete - 1e-6))
            return idx
    else:
        act_dim = env.action_space.shape[0]
        act_low = env.action_space.low
        act_high = env.action_space.high
        def map_action(a_cont):
            # scale from tanh [-1,1] to env bounds
            a = (a_cont + 1) / 2
            return (act_low + a * (act_high - act_low)).astype(np.float32)

    device = torch.device(cfg.device)

    # networks
    actor = GaussianPolicy(obs_dim, act_dim).to(device)

    # reward critics (twin)
    q1r = QCritic(obs_dim, act_dim).to(device)
    q2r = QCritic(obs_dim, act_dim).to(device)
    q1r_targ = QCritic(obs_dim, act_dim).to(device)
    q2r_targ = QCritic(obs_dim, act_dim).to(device)
    q1r_targ.load_state_dict(q1r.state_dict())
    q2r_targ.load_state_dict(q2r.state_dict())

    # cost critics (twin)
    q1c = QCritic(obs_dim, act_dim).to(device)
    q2c = QCritic(obs_dim, act_dim).to(device)
    q1c_targ = QCritic(obs_dim, act_dim).to(device)
    q2c_targ = QCritic(obs_dim, act_dim).to(device)
    q1c_targ.load_state_dict(q1c.state_dict())
    q2c_targ.load_state_dict(q2c.state_dict())

    # optimizers
    pi_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q_opt = optim.Adam(list(q1r.parameters()) + list(q2r.parameters()) +
                       list(q1c.parameters()) + list(q2c.parameters()), lr=cfg.critic_lr)

    # entropy temperature (learned)
    log_alpha = torch.tensor(math.log(cfg.init_alpha), requires_grad=True, device=device)
    alpha_opt = optim.Adam([log_alpha], lr=cfg.alpha_lr)

    # lagrange multiplier (learned but updated manually)
    lam_val = 0.0

    # target entropy
    if cfg.target_entropy is None:
        target_entropy = -float(act_dim)
    else:
        target_entropy = float(cfg.target_entropy)

    # replay
    replay = ReplayBuffer(obs_dim, act_dim, size=500_000)

    # logging containers
    history = []
    ep_returns, ep_costs, ep_lens = [], [], []
    step_p_unsafe = []

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
                avg_p_unsafe_per_step_last10eps=0.0
            )
        N = min(lastN, len(ep_returns))
        return dict(
            avg_return_last10eps=float(np.mean(ep_returns[-N:])),
            avg_cost_last10eps=float(np.mean(ep_costs[-N:])),
            avg_len_last10eps=float(np.mean(ep_lens[-N:])),
            avg_p_unsafe_per_step_last10eps=float(np.mean(step_p_unsafe[-max(1, 1000):])) if len(step_p_unsafe) else 0.0,
        )

    # ----------------------------
    # Training loop
    # ----------------------------
    for step in range(1, cfg.total_steps + 1):

        # action selection
        if step < cfg.start_steps:
            if is_discrete:
                a_env = env.action_space.sample()
                a_cont = np.array([np.random.uniform(-1, 1)], dtype=np.float32)
            else:
                a_env = env.action_space.sample()
                # also store a_cont in buffer: invert scaling approx (ok for warm start)
                a_cont = np.clip(np.random.uniform(-1, 1, size=(act_dim,)), -1, 1).astype(np.float32)
        else:
            with torch.no_grad():
                o_t = to_tensor(obs, device).unsqueeze(0)
                a_cont_t, _, _ = actor.sample(o_t)
                a_cont = a_cont_t.cpu().numpy()[0]
                a_env = map_action(a_cont)

        # step env
        obs2, rew, terminated, truncated, info = env.step(a_env)
        done = terminated or truncated

        cost = float(info.get("constraint_cost", 0.0))
        step_p_unsafe.append(cost)

        # store transition
        replay.store(obs, a_cont, rew, cost, obs2, float(done))

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
        if step >= cfg.update_after and step % cfg.update_every == 0:
            for _ in range(cfg.update_every):
                batch = replay.sample_batch(cfg.batch_size)
                o = to_tensor(batch["obs"], device)
                o2 = to_tensor(batch["obs2"], device)
                a = to_tensor(batch["act"], device)
                r = to_tensor(batch["rew"], device).unsqueeze(-1)
                c = to_tensor(batch["cost"], device).unsqueeze(-1)
                d = to_tensor(batch["done"], device).unsqueeze(-1)

                alpha = log_alpha.exp().detach()

                # target actions
                with torch.no_grad():
                    a2, logp_a2, _ = actor.sample(o2)

                    # reward targets
                    q1r_pi_t = q1r_targ(o2, a2)
                    q2r_pi_t = q2r_targ(o2, a2)
                    q_r_targ = torch.min(q1r_pi_t, q2r_pi_t) - alpha * logp_a2
                    backup_r = r + cfg.gamma * (1 - d) * q_r_targ

                    # cost targets (no entropy term)
                    q1c_pi_t = q1c_targ(o2, a2)
                    q2c_pi_t = q2c_targ(o2, a2)
                    q_c_targ = torch.min(q1c_pi_t, q2c_pi_t)
                    backup_c = c + cfg.gamma * (1 - d) * q_c_targ

                # critic losses
                q1r_loss = ((q1r(o, a) - backup_r) ** 2).mean()
                q2r_loss = ((q2r(o, a) - backup_r) ** 2).mean()
                q1c_loss = ((q1c(o, a) - backup_c) ** 2).mean()
                q2c_loss = ((q2c(o, a) - backup_c) ** 2).mean()
                q_loss = q1r_loss + q2r_loss + q1c_loss + q2c_loss

                q_opt.zero_grad()
                q_loss.backward()
                q_opt.step()

                # actor loss (reward - λ * cost + entropy)
                a_pi, logp_pi, _ = actor.sample(o)
                q_r_pi = torch.min(q1r(o, a_pi), q2r(o, a_pi))
                q_c_pi = torch.min(q1c(o, a_pi), q2c(o, a_pi))

                lam = torch.tensor(lam_val, dtype=torch.float32, device=device)
                pi_loss = (alpha * logp_pi - q_r_pi + lam * q_c_pi).mean()

                pi_opt.zero_grad()
                pi_loss.backward()
                pi_opt.step()

                # alpha loss (temperature)
                alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
                alpha_opt.zero_grad()
                alpha_loss.backward()
                alpha_opt.step()

                # soft update targets
                soft_update(q1r, q1r_targ, cfg.tau)
                soft_update(q2r, q2r_targ, cfg.tau)
                soft_update(q1c, q1c_targ, cfg.tau)
                soft_update(q2c, q2c_targ, cfg.tau)

            # dual update using recent per-step costs (stable)
            # use last ~10k costs as moving sample; you can narrow/widen
            if len(step_p_unsafe) > 0:
                avg_step_cost = float(np.mean(step_p_unsafe[-2000:]))
            else:
                avg_step_cost = 0.0

            lam_update = cfg.lambda_lr * (avg_step_cost - cfg.cost_limit_step)
            lam_val = float(np.clip(lam_val + lam_update, 0.0, cfg.lambda_max))

        # periodic logging to history.json
        if step % cfg.log_every_iters == 0 or step == cfg.total_steps:
            stats = compute_stats(lastN=10)
            entry = {
                "iter": int(step),
                "avg_return_last10eps": stats["avg_return_last10eps"],
                "avg_cost_last10eps": stats["avg_cost_last10eps"],
                "avg_len_last10eps": stats["avg_len_last10eps"],
                "avg_p_unsafe_per_step_last10eps": stats["avg_p_unsafe_per_step_last10eps"],
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
        # Start conservative; tune after first stable run
        total_steps=300_000,
        lambda_lr=0.01,
        cost_limit_step=0.01,
        run_dir="runs/sac_lagrangian",
    )
    main(cfg)
