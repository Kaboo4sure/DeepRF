# This is used to train the uncertainty model
# First on simulated date 
# Then validate on C_MAPSS
# The training assume Gaussian Process (GP)

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

DATA_PATH = "data/processed/sim/sim_turbofan_like.csv"
OUT_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)

def main(sample_n: int = 12000, seed: int = 42):
    df = pd.read_csv(DATA_PATH)

    FEATURES = ["s1", "s2", "s3", "s4", "load"]
    ycol = "RUL_true"

    # --- sample to avoid NxN kernel explosion ---
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed).reset_index(drop=True)

    X = df[FEATURES].values
    y = df[ycol].values.astype(float)

    # Kernel: constant * RBF + white noise (helps stability)
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(len(FEATURES)),
                                       length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-3,
        normalize_y=True,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=1,   # keep small
        random_state=seed
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("gp", gp)
    ])

    print(f"Training GP on simulated data sample: n={len(df):,} rows")
    model.fit(X, y)

    joblib.dump(model, f"{OUT_DIR}/gp_rul_sim.pkl")
    print("Saved GP model to models/gp_rul_sim.pkl")
    print("Learned kernel:", model.named_steps["gp"].kernel_)

if __name__ == "__main__":
    main(sample_n=12000, seed=42)
