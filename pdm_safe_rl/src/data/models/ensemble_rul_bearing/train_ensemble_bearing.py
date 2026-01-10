import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# IMPORTANT:
# You must provide a function to load the bearing runs into (X,y).
# If you already have a loader, import it here.
# from src.data.build_ims_bearing_runs import load_runs

DEVICE = "cpu"

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

def flatten_runs(runs):
    X_list, y_list = [], []
    for run in runs:
        Xr = run["X"]  # (T,d)
        yr = run["rul"]  # (T,)
        X_list.append(Xr)
        y_list.append(yr.reshape(-1, 1))
    X = np.vstack(X_list).astype(np.float32)
    y = np.vstack(y_list).astype(np.float32).reshape(-1)
    return X, y

def train_one_model(Xtr, ytr, Xva, yva, input_dim, seed=0, epochs=20, lr=1e-3, batch=256):
    torch.manual_seed(seed)
    model = MLPRegressor(input_dim=input_dim, hidden=64).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr.reshape(-1, 1), dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva.reshape(-1, 1), dtype=torch.float32, device=DEVICE)

    n = Xtr_t.shape[0]
    idx = np.arange(n)

    best = float("inf")
    best_state = None

    for ep in range(epochs):
        np.random.shuffle(idx)
        model.train()
        for i in range(0, n, batch):
            j = idx[i:i+batch]
            pred = model(Xtr_t[j])
            loss = loss_fn(pred, ytr_t[j])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), yva_t).item()
        if val_loss < best:
            best = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def main(
    runs_path_npz: str,
    out_dir: str,
    n_models: int = 5,
):
    """
    runs_path_npz: path to an .npz that contains runs flattened arrays OR a pickled runs list.
                  simplest approach: save runs as pickle and load here.
    out_dir: folder to save model_i.pt, scaler.pkl, meta.json
    """
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load runs ----
    # Recommended: create runs.pkl once and use it.
    # runs = pickle.load(open("data/ims/runs.pkl", "rb"))
    runs = pickle.load(open(runs_path_npz, "rb"))

    X, y = flatten_runs(runs)

    # Add "load" placeholder column at the end to match env input [feat..., load]
    load_col = np.zeros((X.shape[0], 1), dtype=np.float32)
    X = np.hstack([X, load_col]).astype(np.float32)

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)

    Xtr, Xva, ytr, yva = train_test_split(Xs, y, test_size=0.2, random_state=42)

    input_dim = Xtr.shape[1]

    # Save scaler
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Train ensemble
    for m in range(n_models):
        model = train_one_model(Xtr, ytr, Xva, yva, input_dim=input_dim, seed=100 + m)
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_{m}.pt"))

    # Save meta.json (keep simple but consistent)
    meta = {
        "input_dim": int(input_dim),
        "n_models": int(n_models),
        "note": "Bearing ensemble RUL predictor; features + load(0).",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved ensemble to: {out_dir}")

if __name__ == "__main__":
    # Example:
    # python src/data/models/ensemble_rul_bearing/train_ensemble_bearing.py
    #   -- but simplest is to edit the variables below.
    RUNS_PKL = "data/ims/runs.pkl"  # you create this once
    OUT_DIR = "src/data/models/ensemble_rul_bearing"
    main(runs_path_npz=RUNS_PKL, out_dir=OUT_DIR, n_models=5)
