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
lambdas = np.array([h["lambda"] for h in history], dtype=float)
kl      = np.array([h.get("approx_kl", np.nan) for h in history], dtype=float)
limit   = float(history[0].get("cost_limit_step", 0.01))

def rolling_mean(x, w=10):
    if len(x) < w:
        return x
    y = np.copy(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        y[i] = np.mean(x[lo:i+1])
    return y

# Precompute rolling means for reuse
p_unsafe_rm = rolling_mean(p_unsafe, 10)
lambda_rm   = rolling_mean(lambdas, 10)
return_rm   = rolling_mean(returns, 10)


# ---------- Plot 1: Safety (p_unsafe per step) ----------
plt.figure()
plt.plot(iters, p_unsafe, label="p_unsafe (last10eps)")
plt.plot(iters, rolling_mean(p_unsafe, 10), label="rolling mean (10)")
plt.axhline(limit, linestyle="--", label=f"target limit = {limit}")
plt.xlabel("Iteration")
plt.ylabel("Avg unsafe probability per step")
plt.title("Safety Constraint Tracking (PPO-Lagrangian)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "01_safety_p_unsafe.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "01_safety_p_unsafe.pdf"))
plt.close()

# ---------- Plot 2: Lambda ----------
plt.figure()
plt.plot(iters, lambdas, label="lambda")
plt.plot(iters, rolling_mean(lambdas, 10), label="rolling mean (10)")
plt.xlabel("Iteration")
plt.ylabel("Lagrange multiplier (λ)")
plt.title("Dual Variable Evolution (λ)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "02_lambda.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "02_lambda.pdf"))
plt.close()

# ---------- Plot 3: Return ----------
plt.figure()
plt.plot(iters, returns, label="avg return (last10eps)")
plt.plot(iters, rolling_mean(returns, 10), label="rolling mean (10)")
plt.xlabel("Iteration")
plt.ylabel("Average return")
plt.title("Performance Under Safety Constraint")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "03_return.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "03_return.pdf"))
plt.close()

# ---------- Plot 4: Approx KL ----------
plt.figure()
plt.plot(iters, kl, label="approx_kl")
plt.xlabel("Iteration")
plt.ylabel("Approx KL")
plt.title("PPO Update Size (Approx KL)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "04_approx_kl.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "04_approx_kl.pdf"))
plt.close()

# ------------- Composite Overlay -------------
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))

# 1) Safety
axs[0].plot(iters, p_unsafe, alpha=0.35, label="raw")
axs[0].plot(iters, p_unsafe_rm, linewidth=2, label="rolling mean (10)")
axs[0].axhline(limit, linestyle="--", label=f"limit={limit}")
axs[0].set_ylabel("p_unsafe")
axs[0].grid(True)
axs[0].legend()

# 2) Lambda
axs[1].plot(iters, lambdas, alpha=0.35, label="raw")
axs[1].plot(iters, lambda_rm, linewidth=2, label="rolling mean (10)")
axs[1].set_ylabel("lambda")
axs[1].grid(True)
axs[1].legend()

# 3) Return
axs[2].plot(iters, returns, alpha=0.35, label="raw")
axs[2].plot(iters, return_rm, linewidth=2, label="rolling mean (10)")
axs[2].set_ylabel("avg return")
axs[2].grid(True)
axs[2].legend()

# 4) Approx KL
axs[3].plot(iters, kl, label="approx_kl")
axs[3].set_ylabel("approx KL")
axs[3].set_xlabel("Iteration")
axs[3].grid(True)
axs[3].legend()

fig.suptitle("PPO-Lagrangian Training Dynamics (Safety, Dual Variable, Return, KL)", y=0.98)
plt.tight_layout()

plt.savefig(os.path.join(out_dir, "00_composite_overlay.png"), dpi=200)
plt.savefig(os.path.join(out_dir, "00_composite_overlay.pdf"))
plt.close()

print(f"Done. Saved figures to: {out_dir}")
