import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Import the ensemble predictor (relative import-safe)
# If you prefer, you can copy EnsembleRUL into this file instead.
from src.data.predict_rul_ensemble import EnsembleRUL, FEATURES as ENS_FEATURES


class MaintenanceEnv(gym.Env):
    """
    Turbofan-like Predictive Maintenance Environment with:
      - latent degradation x in [0, 1+], failure when x >= x_fail
      - noisy sensors derived from x and load regime
      - discrete maintenance actions
      - uncertainty from an ensemble RUL predictor
      - risk as P_unsafe estimated from ensemble disagreement

    Observation:
      [s1, s2, s3, s4, load, mu_rul, sigma_rul]

    Action space:
      0: do_nothing
      1: inspect (small cost, no reset, may reduce uncertainty later if you model it)
      2: minor_repair (partially restores degradation)
      3: replace (resets degradation near zero)

    Reward:
      negative of step cost:
        - operating cost
        - maintenance action cost
        - failure penalty if failure occurs

    Constraint cost:
      p_unsafe = fraction of ensemble members predicting RUL < RUL_min
      (smooth risk signal for CMDP / Lagrangian methods)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        model_dir="models/ensemble_rul_sim",
        n_models=5,
        max_steps=300,
        x_fail=1.0,
        rul_min=15.0,
        # degradation dynamics
        drift_base=0.003,
        drift_load=0.004,
        process_noise=0.01,
        sensor_noise=0.03,
        # costs
        c_inspect=1.0,
        c_minor=8.0,
        c_replace=25.0,
        c_failure=200.0,
        c_operate=0.2,
        seed=42,
    ):
        super().__init__()

        self.max_steps = int(max_steps)
        self.x_fail = float(x_fail)
        self.rul_min = float(rul_min)

        self.drift_base = float(drift_base)
        self.drift_load = float(drift_load)
        self.process_noise = float(process_noise)
        self.sensor_noise = float(sensor_noise)

        self.c_inspect = float(c_inspect)
        self.c_minor = float(c_minor)
        self.c_replace = float(c_replace)
        self.c_failure = float(c_failure)
        self.c_operate = float(c_operate)

        self.rng = np.random.default_rng(seed)

        # Discrete actions: 0..3
        self.action_space = spaces.Discrete(4)

        # Observation: s1..s4, load, mu, sigma
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        # Load ensemble predictor
        # IMPORTANT: run scripts from project root so model_dir resolves correctly
        self.ens = EnsembleRUL(model_dir=model_dir, n_models=n_models)

        # internal state
        self.t = 0
        self.load = 0
        self.x = 0.0
        self.last_obs = None

    # ------------------------
    # Simulator: sensors
    # ------------------------
    def _sensors_from_state(self, x: float, load: int, t: int) -> np.ndarray:
        # engineered sensor proxies, noisy
        n = self.sensor_noise

        s1 = 1.0 - 0.8 * x + 0.10 * load + n * self.rng.normal()
        s2 = 0.5 + 1.2 * x + 0.15 * load + n * self.rng.normal()
        s3 = np.sin(2 * np.pi * (0.03 * t)) + 0.6 * x + 0.05 * load + n * self.rng.normal()
        s4 = 0.2 + 0.3 * load + 0.9 * (x ** 2) + n * self.rng.normal()

        return np.array([s1, s2, s3, s4], dtype=np.float32)

    def _predict_mu_sigma_and_risk(self, sensors: np.ndarray, load: int):
        # Build input row for ensemble: [s1,s2,s3,s4,load]
        X = np.array([[sensors[0], sensors[1], sensors[2], sensors[3], float(load)]], dtype=np.float32)

        # mean/std uncertainty
        mu, sigma = self.ens.predict_mu_sigma(X)  # each is shape (1,)
        mu_val = float(mu[0])
        sigma_val = float(sigma[0])

        # risk: fraction of ensemble predictions below RUL_min
        P = self.ens.predict_all(X)  # (M, 1)
        p_unsafe = float((P[:, 0] < self.rul_min).mean())

        return mu_val, sigma_val, p_unsafe

    def _get_obs(self):
        sensors = self._sensors_from_state(self.x, self.load, self.t)
        mu_rul, sigma_rul, p_unsafe = self._predict_mu_sigma_and_risk(sensors, self.load)

        obs = np.array(
            [sensors[0], sensors[1], sensors[2], sensors[3], float(self.load), mu_rul, sigma_rul],
            dtype=np.float32,
        )
        return obs, p_unsafe, mu_rul, sigma_rul

    # ------------------------
    # Gym API
    # ------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.t = 0
        self.load = int(self.rng.integers(0, 3))  # 0..2
        self.x = 0.0  # start healthy

        obs, p_unsafe, mu_rul, sigma_rul = self._get_obs()
        self.last_obs = obs

        info = {
            "p_unsafe": p_unsafe,
            "mu_rul": mu_rul,
            "sigma_rul": sigma_rul,
            "x_true": float(self.x),
            "load": int(self.load),
            "constraint_cost": p_unsafe,
        }
        return obs, info

    def step(self, action):
        action = int(action)
        assert self.action_space.contains(action)

        # --- action effects ---
        action_cost = 0.0
        if action == 0:  # do nothing
            action_cost = 0.0
        elif action == 1:  # inspect
            action_cost = self.c_inspect
            # (Optional later) you can model inspection as reducing uncertainty
        elif action == 2:  # minor repair: partial restoration
            action_cost = self.c_minor
            self.x = max(0.0, self.x - 0.25)  # tune this
        elif action == 3:  # replace: reset
            action_cost = self.c_replace
            self.x = 0.0

        # --- degradation transition ---
        self.t += 1
        drift = self.drift_base + self.drift_load * (self.load / 2.0)
        self.x = max(0.0, self.x + drift + self.process_noise * self.rng.normal())

        failed = self.x >= self.x_fail
        terminated = bool(failed)
        truncated = bool(self.t >= self.max_steps)

        obs, p_unsafe, mu_rul, sigma_rul = self._get_obs()
        self.last_obs = obs

        # --- cost / reward ---
        # Operating cost (small each step), failure penalty if failure occurs
        step_cost = self.c_operate + action_cost + (self.c_failure if failed else 0.0)

        # reward is negative cost
        reward = -float(step_cost)

        info = {
            "p_unsafe": p_unsafe,
            "mu_rul": mu_rul,
            "sigma_rul": sigma_rul,
            "x_true": float(self.x),
            "failed": int(failed),
            "load": int(self.load),
            # CMDP constraint cost:
            "constraint_cost": p_unsafe,
        }

        return obs, reward, terminated, truncated, info
