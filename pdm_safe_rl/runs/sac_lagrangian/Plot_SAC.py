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

# ---- sort by iter (critical) ----
history = sorted(history, key=lambda h: int(h.get("iter", 0)))

iters   = np.array([h["iter"] for h in history], dtype=float)
returns = np.array([h.get("avg_return_last10eps", np.nan) for h in history], dtype=float)
costs   = np.array([h.get("avg_cost_last10eps", np.nan) for h in history], dtype=float)
lens    = np.array([h.get("avg_len_last10eps", np.nan) for h in history], dtype=float)
p_unsafe= np.array([h.get("avg_p_unsafe_per_step_last10eps", np.nan) for h in history], dtype=float)
lambdas = np.array([h.get("lambda", np.nan) for h in history], dtype=float)
alpha   = np.array([h.get("alpha", np.nan) for h in history], dtype=float)
entropy = np.array([h.get("entropy", np.nan) for h in history], dtype=float)

limit = float(history[0].get("cost_limit_step", 0.01))

# ---- enforce plotting range ----
x_min = float(np.nanmin(iters))
x_max = float(np.nanmax(iters))

def rolling_mean(x, w=10):
    """Same-length rolling mean; ignores NaNs."""
    x = np.asarray(x, dtype=float)
    y = np.copy(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        window = x[lo:i+1]
        window = window[~np.isnan(window)]
        y[i] = np.mean(window) if len(window) else np.nan
    return y

def style_xaxis(ax):
    ax.set_xlim(x_min, x_max)
    # ticks every 50k (adjust if you want)
    step = 50_000
    ticks = np.arange(int(np.ceil(x_min/step)*step), int(x_max) + 1, step)
    if len(ticks) > 0:
        ax.set_xticks(ticks)
    ax.ticklabel_format(style='plain', axis='x')  # avoid 1e5 offset notation

# ---------- 01 Safety ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, p_unsafe, alpha=0.35, label="p_unsafe (raw)")
ax.plot(iters, rolling_mean(p_unsafe, 10), linewidth=2, label="p_unsafe (roll-10)")
ax.axhline(limit, linestyle="--", label=f"target limit = {limit}")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Avg unsafe probability / step")
ax.set_title("SAC-Lagrangian Safety Tracking")
ax.grid(True)
ax.legend()
style_xaxis(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "01_safety_p_unsafe_sac.png"), dpi=200)
fig.savefig(os.path.join(out_dir, "01_safety_p_unsafe_sac.pdf"))
plt.close(fig)

# ---------- 02 Lambda ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, lambdas, alpha=0.35, label="lambda (raw)")
ax.plot(iters, rolling_mean(lambdas, 10), linewidth=2, label="lambda (roll-10)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Lagrange multiplier (λ)")
ax.set_title("SAC-Lagrangian Dual Variable (λ)")
ax.grid(True)
ax.legend()
style_xaxis(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "02_lambda_sac.png"), dpi=200)
fig.savefig(os.path.join(out_dir, "02_lambda_sac.pdf"))
plt.close(fig)

# ---------- 03 Return ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, returns, alpha=0.35, label="avg return (raw)")
ax.plot(iters, rolling_mean(returns, 10), linewidth=2, label="avg return (roll-10)")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Average return (last 10 eps)")
ax.set_title("SAC-Lagrangian Performance Under Safety Constraint")
ax.grid(True)
ax.legend()
style_xaxis(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "03_return_sac.png"), dpi=200)
fig.savefig(os.path.join(out_dir, "03_return_sac.pdf"))
plt.close(fig)

# ---------- 04 Alpha ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, alpha, label="alpha")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Entropy temperature (α)")
ax.set_title("SAC Temperature (α)")
ax.grid(True)
ax.legend()
style_xaxis(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "04_alpha_sac.png"), dpi=200)
fig.savefig(os.path.join(out_dir, "04_alpha_sac.pdf"))
plt.close(fig)

# ---------- 05 Entropy ----------
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(iters, entropy, label="policy entropy")
ax.set_xlabel("Environment Steps")
ax.set_ylabel("Entropy")
ax.set_title("SAC Policy Entropy (Discrete)")
ax.grid(True)
ax.legend()
style_xaxis(ax)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "05_entropy_sac.png"), dpi=200)
fig.savefig(os.path.join(out_dir, "05_entropy_sac.pdf"))
plt.close(fig)

print(f"Done. Saved figures to: {out_dir}")
print(f"X-axis range enforced: {x_min:.0f} → {x_max:.0f}")
