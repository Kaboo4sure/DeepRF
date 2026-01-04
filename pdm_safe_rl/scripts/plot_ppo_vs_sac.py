import os
import json
import numpy as np
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/..
PPO_HIST = os.path.join(ROOT, "runs", "ppo_lagrangian", "history.json")
SAC_HIST = os.path.join(ROOT, "runs", "sac_lagrangian_discrete", "history.json")
OUT_DIR  = os.path.join(ROOT, "runs", "comparisons")
os.makedirs(OUT_DIR, exist_ok=True)

def load_hist(path):
    with open(path, "r", encoding="utf-8") as f:
        h = json.load(f)

    # sort by iter to avoid plotting out of order
    h = sorted(h, key=lambda e: float(e.get("iter", 0)))

    x = np.array([e["iter"] for e in h], dtype=float)
    ret = np.array([e.get("avg_return_last10eps", np.nan) for e in h], dtype=float)
    pus = np.array([e.get("avg_p_unsafe_per_step_last10eps", np.nan) for e in h], dtype=float)
    lam = np.array([e.get("lambda", np.nan) for e in h], dtype=float)
    kl  = np.array([e.get("approx_kl", np.nan) for e in h], dtype=float)
    alpha = np.array([e.get("alpha", np.nan) for e in h], dtype=float)
    ent = np.array([e.get("entropy", np.nan) for e in h], dtype=float)
    limit = float(h[0].get("cost_limit_step", 0.01)) if len(h) else 0.01

    return dict(x=x, ret=ret, pus=pus, lam=lam, kl=kl, alpha=alpha, ent=ent, limit=limit)

def rolling_mean(x, w=10):
    """Same-length rolling mean; ignores NaNs."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    y = np.copy(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        window = x[lo:i+1]
        window = window[~np.isnan(window)]
        y[i] = np.mean(window) if len(window) else np.nan
    return y

def normalize_progress(x):
    """Map x to [0,1] to make PPO iterations and SAC steps comparable on one axis."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax == xmin:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)

ppo = load_hist(PPO_HIST)
sac = load_hist(SAC_HIST)

limit = ppo["limit"]  # assume same limit across runs

# ---- normalized progress for fair overlays ----
ppo_p = normalize_progress(ppo["x"])
sac_p = normalize_progress(sac["x"])

# ---------- Overlay: p_unsafe (normalized progress) ----------
plt.figure(figsize=(9, 5))
plt.plot(ppo_p, rolling_mean(ppo["pus"], 10), linewidth=2, label="PPO-Lagrangian (roll-10)")
plt.plot(sac_p, rolling_mean(sac["pus"], 10), linewidth=2, label="SAC-Lagrangian (roll-10)")
plt.axhline(limit, linestyle="--", label=f"constraint limit = {limit}")
plt.xlabel("Normalized training progress (0 → 1)")
plt.ylabel("Avg p_unsafe per step")
plt.title("Safety Tracking (Normalized): PPO vs SAC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "overlay_safety_ppo_vs_sac.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "overlay_safety_ppo_vs_sac.pdf"))
plt.close()

# ---------- Overlay: lambda (normalized progress) ----------
plt.figure(figsize=(9, 5))
plt.plot(ppo_p, rolling_mean(ppo["lam"], 10), linewidth=2, label="PPO-Lagrangian (roll-10)")
plt.plot(sac_p, rolling_mean(sac["lam"], 10), linewidth=2, label="SAC-Lagrangian (roll-10)")
plt.xlabel("Normalized training progress (0 → 1)")
plt.ylabel("λ (Lagrange multiplier)")
plt.title("Dual Variable Evolution (Normalized): PPO vs SAC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "overlay_lambda_ppo_vs_sac.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "overlay_lambda_ppo_vs_sac.pdf"))
plt.close()

# ---------- Overlay: return (normalized progress) ----------
plt.figure(figsize=(9, 5))
plt.plot(ppo_p, rolling_mean(ppo["ret"], 10), linewidth=2, label="PPO-Lagrangian (roll-10)")
plt.plot(sac_p, rolling_mean(sac["ret"], 10), linewidth=2, label="SAC-Lagrangian (roll-10)")
plt.xlabel("Normalized training progress (0 → 1)")
plt.ylabel("Avg return (last 10 eps)")
plt.title("Performance (Normalized): PPO vs SAC")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "overlay_return_ppo_vs_sac.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "overlay_return_ppo_vs_sac.pdf"))
plt.close()

# ---------- SAC-specific: alpha & entropy ----------
plt.figure(figsize=(9, 5))
plt.plot(sac["x"], sac["alpha"], label="alpha")
plt.xlabel("Environment steps")
plt.ylabel("α")
plt.title("SAC Temperature (α)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sac_alpha.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "sac_alpha.pdf"))
plt.close()

plt.figure(figsize=(9, 5))
plt.plot(sac["x"], sac["ent"], label="entropy")
plt.xlabel("Environment steps")
plt.ylabel("Entropy")
plt.title("SAC Policy Entropy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "sac_entropy.png"), dpi=200)
plt.savefig(os.path.join(OUT_DIR, "sac_entropy.pdf"))
plt.close()

print(f"Done. Saved comparison figures to: {OUT_DIR}")
print("Note: PPO and SAC overlays use normalized training progress on the x-axis (0→1).")
