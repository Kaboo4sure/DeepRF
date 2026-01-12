from src.env.maintenance_NASABearing_env import NASABearingMaintenanceEnv

env = NASABearingMaintenanceEnv()
obs, info = env.reset()
print("obs shape:", obs.shape)
print("info:", info)

for i in range(5):
    obs, r, term, trunc, info = env.step(0)  # do nothing
    print(i, r, term, info["t"], info["true_rul"], info["mu_rul"], info["sigma_rul"], info["p_unsafe"])
    if term or trunc:
        break
