import matplotlib.pyplot as plt


"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on creating a cartpole base environment.")
parser.add_argument("--num_envs", type=int, default=16, help="Number of environments to spawn.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch
from tqdm import tqdm

from isaaclab.envs import ManagerBasedRLEnv

from cartpole_config import CartpoleEnvCfg
from q_table import QTable, QAgent


def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    n_episodes = 10

    # setup Q-table
    n_obs = 21
    n_actions = 51

    q_table = QTable(env, n_obs, n_actions)
    q_table.load("scripts/rl_playground/q_learning/out/q_table_best.pt")
    q_agent = QAgent(env, q_table, epsilon=0.0)

    # simulate physics
    last_obs, _ = env.reset()

    # initial action, observation
    last_action = torch.randn_like(env.action_manager.action)
    obs, rewards, terminated, truncated, info = env.step(last_action)
    last_obs = obs["policy"]

    cumulative_reward, cumulative_eval_reward = 0, 0
    rewards_list = []
    terminated = False

    # run the simulation
    pbar = tqdm(total=n_episodes)
    for ep in range(n_episodes + 1):
        with torch.inference_mode():
            while simulation_app.is_running() and not terminated:
                action = q_agent.get_action(last_obs)

                # step the environment
                obs, rewards, terminated, truncated, info = env.step(action)

                cumulative_reward += rewards.item()

                last_obs = obs["policy"]

            rewards_list.append(cumulative_reward)
            cumulative_reward, cumulative_eval_reward = 0, 0

            obs, _ = env.reset()
            terminated = False

            pbar.update(1)
            pbar.set_description(
                ", ".join(
                    [
                        f"Episode: {ep}",
                        f"Rewards: {rewards_list[-1]:.4f}",
                    ]
                )
            )

    # close the environment
    env.close()

    # fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    #
    # axs[0].scatter(range(len(rewards_list)), rewards_list, s=4, alpha=0.5, label="Rewards")
    # axs[0].plot(range(0, len(rewards_list), 50), eval_rewards, "x--", color="magenta", alpha=0.5, label="Eval Rewards")
    # axs[0].set(title="Rewards")
    # axs[1].plot(alphas)
    # axs[1].set(title="$\\alpha$")
    # axs[2].plot(epsilons)
    # axs[2].set(title="$\\epsilon$")
    #
    # plt.tight_layout()
    #
    # fig.savefig("scripts/rl_playground/q_learning/out/q_learning_rewards.png", dpi=300)

    # print(f"max cart vel: {max_cart_vel}, max pole vel: {max_pole_vel}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
