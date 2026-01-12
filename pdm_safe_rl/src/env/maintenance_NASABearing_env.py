import os
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pdm_safe_rl.src.data.predict_rul_ensemble_bearing import EnsembleRULBearing

class NASABearingMaintenanceEnv(gym.Env):
    """
    IMS Bearing run-to-failure environment aligned with your CMAPSS CMDP setup.

    Episode:
      - pick one bearing run (X: Txd, rul: T)
      - step through time t = 0..T-1

    Observation:
      [features_d..., mu_rul, sigma_rul]  (float32)

    Actions (Discrete):
      0: do_nothing
      1: inspect
      2: minor_repair  (ends episode, moderate cost)
      3: replace       (ends episode, higher cost)

    Reward:
      negative of step cost:
        operating cost + action cost + failure penalty at end if you reached failure without maintenance

    Constraint cost:
      p_unsafe = fraction of ensemble models predicting RUL < rul_min
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        runs_pkl="pdm_safe_rl/src/data/data/raw/ims_bearing/runs.pkl",
        model_dir="pdm_safe_rl/src/data/models/ensemble_rul_bearing",
        n_models=5,
        rul_min=15.0,
        # costs
        c_inspect=1.0,
        c_minor=8.0,
        c_replace=25.0,
        c_failure=200.0,
        c_operate=0.2,
        seed=42,
    ):
        super().__init__()

        self.rng = np.random.default_rng(seed)

        self.runs_pkl = runs_pkl
        if not os.path.exists(self.runs_pkl):
            raise FileNotFoundError(f"runs.pkl not found: {self.runs_pkl}")

        with open(self.runs_pkl, "rb") as f:
            self.runs = pickle.load(f)

        if len(self.runs) == 0:
            raise ValueError("runs.pkl contains no runs.")

        # Ensemble predictor
        self.ens = EnsembleRULBearing(model_dir=model_dir, n_models=n_models)

        self.rul_min = float(rul_min)

        self.c_inspect = float(c_inspect)
        self.c_minor = float(c_minor)
        self.c_replace = float(c_replace)
        self.c_failure = float(c_failure)
        self.c_operate = float(c_operate)

        # Actions: 0..3
        self.action_space = spaces.Discrete(4)

        # Determine feature dimension from first run
        d = int(self.runs[0]["X"].shape[1])
        self.feature_dim = d

        # Observation: features + mu + sigma
        self.obs_dim = self.feature_dim + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

        # Internal episode state
        self.run_idx = None
        self.X = None
        self.rul = None
        self.t = 0
        self.T = 0

    def _predict_mu_sigma_and_risk(self, feat: np.ndarray):
        # feat shape (d,)
        Xinp = feat.reshape(1, -1).astype(np.float32)

        mu, sigma = self.ens.predict_mu_sigma(Xinp)
        mu_val = float(mu[0])
        sigma_val = float(sigma[0])

        P = self.ens.predict_all(Xinp)  # (M, 1)
        p_unsafe = float((P[:, 0] < self.rul_min).mean())
        return mu_val, sigma_val, p_unsafe

    def _get_obs(self):
        feat = self.X[self.t].astype(np.float32)
        mu, sigma, p_unsafe = self._predict_mu_sigma_and_risk(feat)
        obs = np.concatenate([feat, np.array([mu, sigma], dtype=np.float32)], axis=0).astype(np.float32)
        return obs, mu, sigma, p_unsafe

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.run_idx = int(self.rng.integers(0, len(self.runs)))
        run = self.runs[self.run_idx]

        self.X = run["X"].astype(np.float32)
        self.rul = run["rul"].astype(np.float32)
        self.T = int(self.X.shape[0])
        self.t = 0

        # sanity: enforce consistent dimension
        if self.X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Run feature dim mismatch. expected {self.feature_dim}, got {self.X.shape[1]} "
                f"(run_idx={self.run_idx}, exp_dir={run.get('exp_dir','?')})"
            )

        obs, mu, sigma, p_unsafe = self._get_obs()

        info = {
            "run_idx": self.run_idx,
            "t": self.t,
            "true_rul": float(self.rul[self.t]),
            "mu_rul": mu,
            "sigma_rul": sigma,
            "p_unsafe": p_unsafe,
            "constraint_cost": p_unsafe,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        done_by_maint = False
        action_cost = 0.0

        if action == 0:
            action_cost = 0.0
        elif action == 1:
            action_cost = self.c_inspect
        elif action == 2:
            action_cost = self.c_minor
            done_by_maint = True
        elif action == 3:
            action_cost = self.c_replace
            done_by_maint = True

        # operating cost always applies each step you keep running
        step_cost = self.c_operate + action_cost

        # If maintenance action, terminate episode immediately (you intervened)
        if done_by_maint:
            obs, mu, sigma, p_unsafe = self._get_obs()
            reward = -float(step_cost)
            terminated = True
            truncated = False
            info = {
                "run_idx": self.run_idx,
                "t": self.t,
                "true_rul": float(self.rul[self.t]),
                "mu_rul": mu,
                "sigma_rul": sigma,
                "p_unsafe": p_unsafe,
                "constraint_cost": p_unsafe,
                "done_reason": "maintenance",
                "action": action,
            }
            return obs, reward, terminated, truncated, info

        # Otherwise proceed to next time step
        self.t += 1

        # If we reached end, that's "failure" (run-to-failure completed)
        failed = self.t >= (self.T - 1)
        terminated = bool(failed)
        truncated = False  # no max_steps needed; T defines horizon

        if failed:
            # apply failure penalty
            step_cost += self.c_failure
            # clamp t to last index for obs/info
            self.t = self.T - 1

        obs, mu, sigma, p_unsafe = self._get_obs()
        reward = -float(step_cost)

        info = {
            "run_idx": self.run_idx,
            "t": self.t,
            "true_rul": float(self.rul[self.t]),
            "mu_rul": mu,
            "sigma_rul": sigma,
            "p_unsafe": p_unsafe,
            "constraint_cost": p_unsafe,
            "failed": int(failed),
            "done_reason": "failure" if failed else "continue",
            "action": action,
        }
        return obs, reward, terminated, truncated, info
