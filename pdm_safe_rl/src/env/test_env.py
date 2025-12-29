# Add an environment smoke test
from src.env.maintenance_env import MaintenanceEnv

def main():
    env = MaintenanceEnv(
        model_dir="models/ensemble_rul_sim",
        n_models=5,
        max_steps=50,
        rul_min=15.0
    )

    obs, info = env.reset()
    print("RESET obs:", obs)
    print("RESET info:", info)

    total_reward = 0.0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        print(f"step {i:02d} action={action} reward={reward:.2f} p_unsafe={info['p_unsafe']:.2f} mu={info['mu_rul']:.2f} sigma={info['sigma_rul']:.2f}")
        if terminated or truncated:
            print("done:", terminated, truncated)
            break

    print("total_reward:", total_reward)

if __name__ == "__main__":
    main()
