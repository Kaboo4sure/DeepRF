import json
import os
import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))

history_path = os.path.join(HERE, "history.json")
out_dir = os.path.join(HERE, "figures")
os.makedirs(out_dir, exist_ok=True)

with open(history_path, "r", encoding="utf-8") as f:
    history = json.load(f)

iters   = np.array([h["iter"] for h in history], dtype=float)
returns = np.array([h["avg_return_last10eps"] for h in history], dtype=float)
costs   = np.array([h["avg_cost_last10eps"] for h in history], dtype=float)
lens    = np.array([h["avg_len_last10eps"] for h in history], dtype=float)
p_unsafe= np.array([h.get("avg_p_unsafe_per_step_last10eps", np.nan) for h in history], dtype=float)
lambdas = np.array([h.get("lambda", np.nan) for h in history], dtype=float)
alpha   = np.array([h.get("alpha", np.nan) for h in history], dtype=float)
entropy = np.array([h.get("entropy", np.nan) for h in history], dtype=float)

limit = float(history[0].get("cost_limit_step", 0.01))

def rolling_mean(x, w=10):
    if len(x) < w:
        return x
    y = np.copy(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        y[i] = np.mean(x[lo:i+1])
    return y

# ---------- 01 Safety ----------
plt.figure()
plt.plot(iters, p_unsafe, alpha=0.35, label="p_unsafe (raw)")
plt.plot(iters, rolling_mean(p_unsafe, 10), linewidth=2, label="p_unsafe (roll-10)")
plt.axhline(limit, linestyle="--", label=f"target limit = {limit}")
plt.xlabel("Environment Steps")
plt.ylabel("Avg unsafe probability / step")
plt.title("SAC-Lagrangian Safety Tracking")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "01_safety_p_unsafe_sac.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "01_safety_p_unsafe_sac.pdf"))
plt.close()

# ---------- 02 Lambda ----------
plt.figure()
plt.plot(iters, lambdas, alpha=0.35, label="lambda (raw)")
plt.plot(iters, rolling_mean(lambdas, 10), linewidth=2, label="lambda (roll-10)")
plt.xlabel("Environment Steps")
plt.ylabel("Lagrange multiplier (λ)")
plt.title("SAC-Lagrangian Dual Variable (λ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "02_lambda_sac.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "02_lambda_sac.pdf"))
plt.close()

# ---------- 03 Return ----------
plt.figure()
plt.plot(iters, returns, alpha=0.35, label="avg return (raw)")
plt.plot(iters, rolling_mean(returns, 10), linewidth=2, label="avg return (roll-10)")
plt.xlabel("Environment Steps")
plt.ylabel("Average return (last 10 eps)")
plt.title("SAC-Lagrangian Performance Under Safety Constraint")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "03_return_sac.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "03_return_sac.pdf"))
plt.close()

# ---------- 04 Alpha / Entropy (SAC-specific) ----------
plt.figure()
plt.plot(iters, alpha, label="alpha")
plt.xlabel("Environment Steps")
plt.ylabel("Entropy temperature (α)")
plt.title("SAC Temperature (α)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "04_alpha_sac.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "04_alpha_sac.pdf"))
plt.close()

plt.figure()
plt.plot(iters, entropy, label="policy entropy")
plt.xlabel("Environment Steps")
plt.ylabel("Entropy")
plt.title("SAC Policy Entropy (Discrete)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "05_entropy_sac.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "05_entropy_sac.pdf"))
plt.close()

print(f"Done. Saved figures to: {out_dir}")
