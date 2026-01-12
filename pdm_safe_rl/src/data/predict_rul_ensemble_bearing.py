import os
import joblib
import numpy as np
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)

class EnsembleRULBearing:
    """
    Ensemble predictor for IMS bearing features.
    Run from pdm_safe_rl root.
    """

    def __init__(self, model_dir="src/data/models/ensemble_rul_bearing", n_models=5):
        self.model_dir = model_dir
        self.n_models = n_models

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"scaler.pkl not found: {scaler_path}")

        self.scaler = joblib.load(scaler_path)

        # ✅ Infer input dimension from scaler
        if hasattr(self.scaler, "n_features_in_"):
            self.in_dim = int(self.scaler.n_features_in_)
        else:
            # older sklearn fallback
            mean_ = getattr(self.scaler, "mean_", None)
            if mean_ is None:
                raise ValueError("Could not infer in_dim from scaler.pkl")
            self.in_dim = int(len(mean_))

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []

        for i in range(n_models):
            ckpt = os.path.join(model_dir, f"model_{i}.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

            m = MLP(in_dim=self.in_dim, hidden=64)
            state = torch.load(ckpt, map_location="cpu")
            m.load_state_dict(state)
            m.eval()
            m.to(self.device)
            self.models.append(m)

    @torch.no_grad()
    def predict_all(self, X_np: np.ndarray) -> np.ndarray:
        if X_np.ndim != 2 or X_np.shape[1] != self.in_dim:
            raise ValueError(f"Expected X shape (N, {self.in_dim}), got {X_np.shape}")

        Xs = self.scaler.transform(X_np).astype(np.float32)
        xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)

        preds = []
        for m in self.models:
            p = m(xt).squeeze(-1)
            preds.append(p)

        P = torch.stack(preds, dim=0)
        return P.cpu().numpy()

    def predict_mu_sigma(self, X_np: np.ndarray):
        P = self.predict_all(X_np)
        return P.mean(axis=0), P.std(axis=0)
