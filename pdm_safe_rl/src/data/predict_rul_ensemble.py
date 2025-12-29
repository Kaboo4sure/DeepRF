import os
import joblib
import numpy as np
import torch
import torch.nn as nn

FEATURES = ["s1", "s2", "s3", "s4", "load"]

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

class EnsembleRUL:
    def __init__(self, model_dir="models/ensemble_rul_sim", n_models=5):
        self.model_dir = model_dir
        self.n_models = n_models

        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        self.scaler = joblib.load(scaler_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.models = []

        for i in range(n_models):
            ckpt = os.path.join(model_dir, f"model_{i}.pt")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

            m = MLP(in_dim=len(FEATURES))
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
        X_np shape: (N, 5) in FEATURE order
        Returns:
          mu: (N,)
          sigma: (N,)  (ensemble std dev)
        """
        P = self.predict_all(X_np)   # (M, N)
        mu = P.mean(axis=0)
        sigma = P.std(axis=0)        # epistemic uncertainty proxy
        return mu, sigma

if __name__ == "__main__":
    ens = EnsembleRUL(model_dir="models/ensemble_rul_sim", n_models=5)

    # smoke test using random inputs
    X = np.random.randn(3, 5).astype(np.float32)
    mu, sig = ens.predict_mu_sigma(X)
    print("mu:", mu)
    print("sigma:", sig)
