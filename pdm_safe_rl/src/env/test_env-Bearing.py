import pickle
from src.env.maintenance_NASABearing_env import NASABearingMaintenanceEnv

runs = pickle.load(open("data/ims/runs.pkl", "rb"))

env = NASABearingMaintenanceEnv(
    runs=runs,
    model_dir="src/data/models/ensemble_rul_bearing",
    n_models=5,
    max_steps=300,
)

obs, info = env.reset()
print("obs:", obs.shape, "info keys:", info.keys())

for _ in range(5):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    print(a, r, term, trunc, info["constraint_cost"], info["mu_rul"], info["sigma_rul"])
