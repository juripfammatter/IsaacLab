# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""
import os

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


class CustomScheduler:
    def __init__(self, initial_lr: float, gamma: float = 0.99):
        self.value = torch.nn.Parameter(torch.tensor([0.0]))
        self.optimizer = torch.optim.SGD([self.value], lr=initial_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self.optimizer.step()

    def step(self) -> float:
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]


def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # alpha scheduler
    alpha_scheduler = CustomScheduler(0.3, gamma=0.9995)
    alpha = alpha_scheduler.step()

    # exploration scheduler (GLIE policy)
    epsilon_scheduler = CustomScheduler(0.5, gamma=0.9995)
    epsilon = epsilon_scheduler.step()

    n_episodes = 500

    # setup Q-table
    n_obs = 21
    n_actions = 51

    q_table = QTable(env, n_obs, n_actions)
    q_agent = QAgent(env, q_table, epsilon=0.3)

    # simulate physics
    last_obs, _ = env.reset()

    # initial action, observation
    last_action = torch.randn_like(env.action_manager.action)
    obs, rewards, terminated, truncated, info = env.step(last_action)
    last_obs = obs["policy"]

    cumulative_reward, cumulative_eval_reward = 0, 0
    rewards_list = []
    eval_rewards = []
    alphas = []
    epsilons = []
    terminated = False

    # find max velocity
    max_cart_vel = 0.0
    max_pole_vel = 0.0

    # run the simulation
    pbar = tqdm(total=n_episodes)
    for ep in range(n_episodes + 1):
        with torch.inference_mode():
            while simulation_app.is_running() and not terminated:
                action = q_agent.get_action(last_obs, epsilon)

                # step the environment
                obs, rewards, terminated, truncated, info = env.step(action)
                q_table.update(last_obs, obs["policy"], action, rewards, alpha)

                cumulative_reward += rewards.item()

                last_obs = obs["policy"]

                if obs["policy"][0][1] > max_cart_vel:
                    max_cart_vel = obs["policy"][0][1]
                if obs["policy"][0][3] > max_pole_vel:
                    max_pole_vel = obs["policy"][0][3]

            # evaluation
            if ep % 50 == 0:
                n_eval = 5
                for _ in range(n_eval):
                    last_obs, _ = env.reset()
                    last_obs = last_obs["policy"]
                    terminated = False

                    while simulation_app.is_running() and not terminated:
                        action = q_agent.get_action(last_obs, epsilon=0.0)

                        # step the environment
                        obs, rewards, terminated, truncated, info = env.step(action)
                        # q_table.update(last_obs, obs["policy"], action, rewards, alpha)

                        cumulative_eval_reward += rewards.item()

                        last_obs = obs["policy"]
                eval_rewards.append(cumulative_eval_reward / n_eval)

            rewards_list.append(cumulative_reward)
            cumulative_reward, cumulative_eval_reward = 0, 0

            obs, _ = env.reset()
            terminated = False

            alpha = alpha_scheduler.step()
            alphas.append(alpha)
            epsilon = epsilon_scheduler.step()
            epsilons.append(epsilon)

            pbar.update(1)
            pbar.set_description(
                ", ".join(
                    [
                        f"Episode: {ep}",
                        f"Rewards: {rewards_list[-1]:.4f}",
                        f"Eval Rewards: {eval_rewards[-1]:.4f}",
                        f"Alpha: {alpha:.4f}",
                        f"Epsilon: {epsilon:.4f}",
                    ]
                )
            )

    # close the environment
    env.close()
    q_table.save("scripts/rl_playground/q_learning/out/q_table_5k.pt")

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))

    axs[0].scatter(range(len(rewards_list)), rewards_list, s=4, alpha=0.5, label="Rewards")
    axs[0].plot(range(0, len(rewards_list), 50), eval_rewards, "x--", color="magenta", alpha=0.5, label="Eval Rewards")
    axs[0].set(title="Rewards")
    axs[1].plot(alphas)
    axs[1].set(title="$\\alpha$")
    axs[2].plot(epsilons)
    axs[2].set(title="$\\epsilon$")

    plt.tight_layout()

    fig.savefig("scripts/rl_playground/q_learning/out/q_learning_rewards.png", dpi=300)

    print(f"max cart vel: {max_cart_vel}, max pole vel: {max_pole_vel}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
