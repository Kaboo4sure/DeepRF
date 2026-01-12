import os
import json
import joblib
import numpy as np
import torch
import torch.nn as nn

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

class EnsembleRULBearing:
    """
    Loads an ensemble trained on IMS bearing features.
    Expects X_np shape: (N, in_dim) where in_dim matches the training feature dimension.
    """

    def __init__(self, model_dir: str, n_models: int = 5):
        self.model_dir = model_dir
        self.n_models = n_models

        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.json not found: {meta_path}")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # We store input dimension in meta.json during training
        self.in_dim = int(meta.get("in_dim", -1))
        if self.in_dim <= 0:
            raise ValueError(f"Invalid in_dim in meta.json: {self.in_dim}")

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"scaler.pkl not found: {scaler_path}")
        self.scaler = joblib.load(scaler_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []

        for i in range(n_models):
            ckpt = os.path.join(model_dir, f"model_{i}.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

            m = MLP(in_dim=self.in_dim)
            state = torch.load(ckpt, map_location="cpu")
            m.load_state_dict(state)
            m.eval()
            m.to(self.device)
            self.models.append(m)

    @torch.no_grad()
    def predict_all(self, X_np: np.ndarray) -> np.ndarray:
        """
        Returns per-model predictions: shape (M, N)
        """
        if X_np.ndim != 2 or X_np.shape[1] != self.in_dim:
            raise ValueError(f"Expected X shape (N, {self.in_dim}), got {X_np.shape}")

        Xs = self.scaler.transform(X_np).astype(np.float32)
        xt = torch.tensor(Xs, dtype=torch.float32, device=self.device)

        preds = []
        for m in self.models:
            p = m(xt).squeeze(-1)  # (N,)
            preds.append(p)

        P = torch.stack(preds, dim=0)  # (M, N)
        return P.cpu().numpy()

    def predict_mu_sigma(self, X_np: np.ndarray):
        """
        Returns:
          mu: (N,)
          sigma: (N,) ensemble std dev (epistemic proxy)
        """
        P = self.predict_all(X_np)
        mu = P.mean(axis=0)
        sigma = P.std(axis=0)
        return mu, sigma
