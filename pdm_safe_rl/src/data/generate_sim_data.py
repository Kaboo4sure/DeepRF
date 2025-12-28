import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def simulate_episode(ep_id: int,
                     max_steps: int = 300,
                     x_fail: float = 1.0,
                     drift_base: float = 0.003,
                     drift_load: float = 0.004,
                     process_noise: float = 0.01,
                     sensor_noise: float = 0.03,
                     seed: int | None = None):
    """
    Turbofan-like degradation simulator:
      x_t = latent degradation in [0, 1+]
      load in {0,1,2} affects drift
      sensors are noisy functions of x_t and load
    """
    rng = np.random.default_rng(seed)

    # Operating regimes (e.g., low/med/high)
    load = rng.integers(0, 3)
    drift = drift_base + drift_load * (load / 2.0)

    x = 0.0
    rows = []

    for t in range(max_steps):
        # latent degradation evolves (monotone-ish)
        x = x + drift + process_noise * rng.normal()
        x = max(x, 0.0)

        failed = x >= x_fail

        # turbofan-like sensor proxies (just engineered features)
        # (can expand to 10-20 sensors later)
        s1 = 1.0 - 0.8 * x + 0.10 * load + sensor_noise * rng.normal()
        s2 = 0.5 + 1.2 * x + 0.15 * load + sensor_noise * rng.normal()
        s3 = np.sin(2*np.pi*(0.03*t)) + 0.6 * x + 0.05 * load + sensor_noise * rng.normal()
        s4 = 0.2 + 0.3 * load + 0.9 * (x**2) + sensor_noise * rng.normal()

        # Ground-truth RUL: how many steps until failure under same drift (approx)
        if failed:
            rul_true = 0
        else:
            # expected remaining steps = (x_fail - x) / drift (clip)
            rul_true = int(max(0.0, (x_fail - x) / max(drift, 1e-8)))

        rows.append({
            "episode_id": ep_id,
            "t": t,
            "load": int(load),
            "x_true": float(x),
            "s1": float(s1),
            "s2": float(s2),
            "s3": float(s3),
            "s4": float(s4),
            "RUL_true": int(rul_true),
            "failed": int(failed),
        })

        if failed:
            break

    return pd.DataFrame(rows)

def main(out_path: str,
         n_episodes: int = 2000,
         max_steps: int = 300,
         seed: int = 42):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    dfs = []
    for ep in tqdm(range(n_episodes), desc="Generating episodes"):
        ep_seed = int(rng.integers(0, 1_000_000_000))
        dfs.append(simulate_episode(ep, max_steps=max_steps, seed=ep_seed))

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print(df.head())

if __name__ == "__main__":
    main(out_path="data/processed/sim/sim_turbofan_like.csv",
         n_episodes=2000,
         max_steps=300,
         seed=42)
