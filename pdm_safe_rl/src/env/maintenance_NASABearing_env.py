import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import List, Dict, Optional

from src.data.predict_rul_ensemble import EnsembleRUL


class NASABearingMaintenanceEnv(gym.Env):
    """
    NASA IMS Bearing run-to-failure maintenance environment.
    Mirrors the structure of src/env/maintenance_env.py:

    Observation (vector):
      [feat_1..feat_d, load, mu_rul, sigma_rul]
      - load is kept for compatibility (0 for bearings unless you define otherwise)
      - mu_rul, sigma_rul come from EnsembleRUL

    Action space:
      0: do_nothing
      1: inspect
      2: minor_repair (partial restoration)
      3: replace (reset)

    Reward:
      negative of step_cost:
        c_operate + action_cost + c_failure(if failure)

    Constraint cost (CMDP):
      p_unsafe = fraction of ensemble members predicting RUL < rul_min
      returned in info["constraint_cost"]
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        runs: List[Dict],
        model_dir: str = "src/data/models/ensemble_rul_bearing",
        n_models: int = 5,
        max_steps: Optional[int] = None,
        rul_min: float = 15.0,
        # costs
        c_inspect: float = 1.0,
        c_minor: float = 8.0,
        c_replace: float = 25.0,
        c_failure: float = 200.0,
        c_operate: float = 0.2,
        # maintenance effects
        minor_repair_jump_frac: float = 0.15,  # jump back fraction of life
        seed: int = 42,
    ):
        super().__init__()
        assert len(runs) > 0, "runs must be a non-empty list of trajectories."
        assert "X" in runs[0], "each run must contain 'X' (T,d) features."
        self.runs = runs

        self.max_steps = max_steps
        self.rul_min = float(rul_min)

        self.c_inspect = float(c_inspect)
        self.c_minor = float(c_minor)
        self.c_replace = float(c_replace)
        self.c_failure = float(c_failure)
        self.c_operate = float(c_operate)

        self.minor_repair_jump_frac = float(minor_repair_jump_frac)

        self.rng = np.random.default_rng(seed)

        # actions identical to C-MAPSS env
        self.action_space = spaces.Discrete(4)

        # features dimension + [load, mu, sigma]
        d = int(self.runs[0]["X"].shape[1])
        self.feat_dim = d
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(d + 3,), dtype=np.float32
        )

        # load ensemble predictor
        self.ens = EnsembleRUL(model_dir=model_dir, n_models=n_models)

        # internal state
        self.run_idx = 0
        self.t = 0
        self.steps = 0

    def _select_run(self) -> int:
        return int(self.rng.integers(0, len(self.runs)))

    def _get_load(self, run: Dict, t: int) -> int:
        # keep placeholder (bearings usually don't have discrete load regimes)
        if "load" in run:
            if np.isscalar(run["load"]):
                return int(run["load"])
            return int(run["load"][t])
        return 0

    def _get_rul_true(self, run: Dict, t: int) -> float:
        if "rul" in run:
            return float(run["rul"][t])
        # fallback: steps remaining
        return float(len(run["X"]) - 1 - t)

    def _predict_mu_sigma_and_risk(self, feat: np.ndarray, load: int):
        # Ensemble expects 2D array input
        X = np.concatenate([feat.astype(np.float32), np.array([float(load)], dtype=np.float32)])
        X = X.reshape(1, -1)

        mu, sigma = self.ens.predict_mu_sigma(X)
        mu_val = float(mu[0])
        sigma_val = float(sigma[0])

        P = self.ens.predict_all(X)  # (M,1)
        p_unsafe = float((P[:, 0] < self.rul_min).mean())
        return mu_val, sigma_val, p_unsafe

    def _get_obs(self):
        run = self.runs[self.run_idx]
        feat = run["X"][self.t].astype(np.float32)
        load = self._get_load(run, self.t)

        mu_rul, sigma_rul, p_unsafe = self._predict_mu_sigma_and_risk(feat, load)

        obs = np.concatenate(
            [feat, np.array([float(load), mu_rul, sigma_rul], dtype=np.float32)],
            axis=0,
        )
        return obs, p_unsafe, mu_rul, sigma_rul

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.run_idx = self._select_run()
        self.t = 0
        self.steps = 0

        obs, p_unsafe, mu_rul, sigma_rul = self._get_obs()

        info = {
            "p_unsafe": p_unsafe,
            "mu_rul": mu_rul,
            "sigma_rul": sigma_rul,
            "run_idx": int(self.run_idx),
            "t": int(self.t),
            "rul_true": self._get_rul_true(self.runs[self.run_idx], self.t),
            "constraint_cost": p_unsafe,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        run = self.runs[self.run_idx]
        T = len(run["X"])

        action_cost = 0.0

        # --- action effects in time/trajectory space ---
        if action == 0:  # do nothing
            pass

        elif action == 1:  # inspect
            action_cost = self.c_inspect
            # You can later model inspection as lowering sigma

        elif action == 2:  # minor repair (partial restoration)
            action_cost = self.c_minor
            jump = int(self.minor_repair_jump_frac * T)
            self.t = max(0, self.t - jump)

        elif action == 3:  # replace
            action_cost = self.c_replace
            self.t = 0

        # --- time progression ---
        self.t += 1
        self.steps += 1

        failed = False
        terminated = False
        truncated = False

        if self.t >= T:
            terminated = True
            failed = True
            self.t = T - 1  # clamp so we can create obs

        if self.max_steps is not None and self.steps >= self.max_steps:
            truncated = True

        obs, p_unsafe, mu_rul, sigma_rul = self._get_obs()

        step_cost = self.c_operate + action_cost + (self.c_failure if failed else 0.0)
        reward = -float(step_cost)

        info = {
            "p_unsafe": p_unsafe,
            "mu_rul": mu_rul,
            "sigma_rul": sigma_rul,
            "run_idx": int(self.run_idx),
            "t": int(self.t),
            "rul_true": self._get_rul_true(run, self.t),
            "failed": int(failed),
            "constraint_cost": p_unsafe,
        }

        return obs, reward, terminated, truncated, info
