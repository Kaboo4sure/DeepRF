import os
import pickle
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.data.predict_rul_ensemble_bearing import EnsembleRULBearing


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
      2: minor_repair
      3: replace

    Reward:
      negative of step cost:
        operating cost + action cost + failure penalty at end if run ends

    Constraint cost:
      p_unsafe = fraction of ensemble models predicting RUL < rul_min
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        runs_pkl="src/data/data/raw/ims_bearing/runs.pkl",
        model_dir="src/data/models/ensemble_rul_bearing",
        n_models=5,
        rul_min=15.0,
        # costs
        c_inspect=1.0,
        c_minor=8.0,
        c_replace=25.0,
        c_failure=200.0,
        max_steps=300,
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

        # Store max_steps (you were using it but never saved it)
        self.max_steps = int(max_steps)

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
        d0 = int(self.runs[0]["X"].shape[1])
        self.feature_dim = d0

        # Observation: features + mu + sigma
        self.obs_dim = self.feature_dim + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )

        # Internal episode state
        self.run_idx = None
        self.t = 0
        self.episode_step = 0

        # Current run buffers (these are what your step() should use)
        self.current_run = None
        self.current_X = None
        self.current_rul = None
        self.current_T = None

    def _predict_mu_sigma_and_risk(self, feat: np.ndarray):
        # Ensemble was trained with an appended load column -> expects in_dim
        Xinp = feat.reshape(1, -1).astype(np.float32)

        # If env provides (in_dim - 1), append dummy load=0.0
        if Xinp.shape[1] == (self.ens.in_dim - 1):
            Xinp = np.concatenate([Xinp, np.zeros((1, 1), dtype=np.float32)], axis=1)
        elif Xinp.shape[1] != self.ens.in_dim:
            raise ValueError(
                f"Feature dim mismatch: env={Xinp.shape[1]}, ensemble expects {self.ens.in_dim}"
            )

        mu, sigma = self.ens.predict_mu_sigma(Xinp)
        mu_val = float(mu[0])
        sigma_val = float(sigma[0])

        P = self.ens.predict_all(Xinp)  # (M,1)
        p_unsafe = float((P[:, 0] < self.rul_min).mean())

        return mu_val, sigma_val, p_unsafe

    def _get_obs(self, t_idx: int = None):
        if t_idx is None:
            t_idx = self.t

        feat = self.current_X[t_idx].astype(np.float32)
        mu, sigma, p_unsafe = self._predict_mu_sigma_and_risk(feat)

        obs = np.concatenate(
            [feat, np.array([mu, sigma], dtype=np.float32)], axis=0
        ).astype(np.float32)

        return obs, mu, sigma, p_unsafe

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_step = 0
        self.t = 0

        # pick a run
        self.run_idx = int(self.rng.integers(0, len(self.runs)))
        self.current_run = self.runs[self.run_idx]

        self.current_X = self.current_run["X"].astype(np.float32)      # (T,d)
        self.current_rul = self.current_run["rul"].astype(np.float32)  # (T,)
        self.current_T = int(self.current_X.shape[0])

        # sanity: enforce consistent dimension
        if self.current_X.shape[1] != self.feature_dim:
            raise ValueError(
                f"Run feature dim mismatch. expected {self.feature_dim}, got {self.current_X.shape[1]} "
                f"(run_idx={self.run_idx}, exp_dir={self.current_run.get('exp_dir','?')})"
            )

        obs, mu, sigma, p_unsafe = self._get_obs(self.t)

        info = {
            "run_idx": int(self.run_idx),
            "t": int(self.t),
            "true_rul": float(self.current_rul[self.t]),
            "mu_rul": float(mu),
            "sigma_rul": float(sigma),
            "p_unsafe": float(p_unsafe),
            "constraint_cost": float(p_unsafe),
        }
        return obs, info

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        # --- action cost/effects ---
        action_cost = 0.0
        if action == 0:
            action_cost = 0.0
        elif action == 1:
            action_cost = self.c_inspect
        elif action == 2:
            action_cost = self.c_minor
            # Optional: repair effect (if you want)
            # self.t = max(0, self.t - 5)
        elif action == 3:
            action_cost = self.c_replace
            # Optional: replace effect (if you want)
            # self.t = 0

        # --- advance time ---
        self.t += 1
        self.episode_step += 1

        T = int(self.current_T)

        # end-of-run termination (ONLY)
        terminated = (self.t >= (T - 1))

        # training horizon truncation (ONLY)
        truncated = (self.episode_step >= self.max_steps)

        # clamp index to avoid crash at end
        t_idx = min(self.t, T - 1)

        obs, mu, sigma, p_unsafe = self._get_obs(t_idx)

        # --- reward / cost ---
        cost = float(p_unsafe)
        reward = -(self.c_operate + action_cost + (self.c_failure if terminated else 0.0))

        info = {
            "run_idx": int(self.run_idx),
            "t": int(t_idx),
            "true_rul": float(self.current_rul[t_idx]),
            "mu_rul": float(mu),
            "sigma_rul": float(sigma),
            "p_unsafe": float(p_unsafe),
            "constraint_cost": float(cost),
        }

        return obs, float(reward), bool(terminated), bool(truncated), info
