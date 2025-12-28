
#For this use case (hundreds of thousands of rows), a Gaussian Process in sklearn will always hit memory limits, 
#but a PyTorch Deep Ensemble scales well and still gives you uncertainty (via variance across models). 
#This is a standard, publishable approach for epistemic uncertainty.


import os
import json
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = "data/processed/sim/sim_turbofan_like.csv"
OUT_DIR = "models/ensemble_rul_sim"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = ["s1", "s2", "s3", "s4", "load"]
TARGET = "RUL_true"

# -------------------------
# Dataset
# -------------------------
class TabDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# -------------------------
# Model
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)

def train_one(model, train_loader, val_loader, lr=1e-3, epochs=15, device="cpu"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_loader.dataset)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(val_loader.dataset)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        print(f"epoch {ep:02d} | train MSE {tr_loss:.4f} | val MSE {va_loss:.4f}")

    model.load_state_dict(best_state)
    return best_val

def main(
    n_models: int = 5,
    sample_n: int = 200_000,   # you can use full data; this caps for speed
    batch_size: int = 2048,
    epochs: int = 15,
    lr: float = 1e-3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    df = pd.read_csv(DATA_PATH)

    # Optional downsample for speed (can increase later)
    if len(df) > sample_n:
        df = df.sample(sample_n, random_state=seed).reset_index(drop=True)

    X = df[FEATURES].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    # Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)

    joblib.dump(scaler, os.path.join(OUT_DIR, "scaler.pkl"))

    train_ds = TabDataset(X_train, y_train)
    val_ds = TabDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = {
        "features": FEATURES,
        "target": TARGET,
        "n_models": n_models,
        "sample_n": len(df),
        "batch_size": batch_size,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "device": device,
    }

    print("Training ensemble...")
    print(json.dumps(meta, indent=2))

    vals = []
    for i in range(n_models):
        print(f"\n=== Model {i+1}/{n_models} ===")
        # Different init seed per model
        torch.manual_seed(int(rng.integers(0, 1_000_000_000)))
        model = MLP(in_dim=len(FEATURES))

        best_val = train_one(model, train_loader, val_loader, lr=lr, epochs=epochs, device=device)
        vals.append(best_val)

        ckpt_path = os.path.join(OUT_DIR, f"model_{i}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved: {ckpt_path} | best val MSE {best_val:.4f}")

    meta["val_mse_each"] = vals
    meta["val_mse_mean"] = float(np.mean(vals))
    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")
    print(f"Saved scaler + {n_models} models to: {OUT_DIR}")

if __name__ == "__main__":
    main()
