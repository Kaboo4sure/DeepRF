import os
import pickle
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
print("RUNNING FILE:", os.path.abspath(__file__))

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_runs(runs_pkl: str):
    with open(runs_pkl, "rb") as f:
        runs = pickle.load(f)
    return runs


def make_xy_from_runs(runs, add_load: bool = True, load_value: float = 0.0):
    """
    runs: list of dicts { "X": (T,d), "rul": (T,) }
    Returns:
      X: (N, d [+1 load])
      y: (N,)
    """
    X_list, y_list = [], []
    for run in runs:
        X = run["X"].astype(np.float32)         # (T,d)
        y = run["rul"].astype(np.float32)       # (T,)
        if add_load:
            load_col = np.full((X.shape[0], 1), load_value, dtype=np.float32)
            X = np.concatenate([X, load_col], axis=1)  # (T, d+1)
        X_list.append(X)
        y_list.append(y)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


def train_one_model(Xtr, ytr, Xva, yva, in_dim, device, seed=0, epochs=25, batch_size=512, lr=1e-3):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = MLP(in_dim=in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device).unsqueeze(-1)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=device).unsqueeze(-1)

    n = Xtr.shape[0]
    idx = np.arange(n)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        np.random.shuffle(idx)
        model.train()

        for start in range(0, n, batch_size):
            b = idx[start:start + batch_size]
            xb = Xtr_t[b]
            yb = ytr_t[b]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vpred = model(Xva_t)
            vloss = loss_fn(vpred, yva_t).item()

        if vloss < best_val:
            best_val = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if ep % 5 == 0 or ep == 1:
            print(f"  epoch {ep:02d} | val MSE={vloss:.4f}")

    model.load_state_dict(best_state)
    return model, best_val


def main():
    # ---- paths ----
    runs_pkl = "src/data/data/raw/ims_bearing/runs.pkl"
    out_dir = "src/data/models/ensemble_rul_bearing"
    os.makedirs(out_dir, exist_ok=True)

    # ---- load data ----
    runs = load_runs(runs_pkl)
    X, y = make_xy_from_runs(runs, add_load=True, load_value=0.0)

    in_dim = X.shape[1]
    print(f"Loaded runs: {len(runs)} | X={X.shape} | y={y.shape} | in_dim={in_dim}")

    # ---- scaler ----
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    print("Saved scaler.pkl")

    # ---- split ----
    Xtr, Xva, ytr, yva = train_test_split(Xs, y, test_size=0.2, random_state=42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # ---- train ensemble ----
    best_vals = []
    for i in range(5):
        print(f"\nTraining model_{i}...")
        model, best_val = train_one_model(
            Xtr, ytr, Xva, yva,
            in_dim=in_dim,
            device=device,
            seed=100 + i,
            epochs=25,
            batch_size=512,
            lr=1e-3,
        )
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_{i}.pt"))
        best_vals.append(best_val)
        print(f"Saved model_{i}.pt | best val MSE={best_val:.4f}")

    print("\nDone. Val MSEs:", best_vals)
    print("Model dir:", out_dir)
    print("IMPORTANT: Use EnsembleRUL(model_dir=..., in_dim=%d)" % in_dim)


if __name__ == "__main__":
    main()
