"""Launch Isaac Sim Simulator first."""

import argparse
from datetime import datetime
import os

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
import torch.optim as optim
from tqdm import tqdm
from cartpole_config import CartpoleEnvCfg
from dqn_agent import DQNAgent, DQN
from utils import ReplayMemory


from isaaclab.envs import ManagerBasedRLEnv

import matplotlib.pyplot as plt
import atexit
import signal
import sys

hyperparameters = {
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "target_update": 10,
    "epsilon": 0.1,
    "n_episodes": 10,
    "batch_size": 128,
    "memory_capacity": 10_000,
}

timestamp = "2025-04-09_21-53-35"


def main():
    torch.manual_seed(402)

    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    n_observations = env.observation_manager.group_obs_dim["policy"][0]
    n_actions = env.action_manager.total_action_dim
    n_action_steps = 7

    agent = DQNAgent(
        n_observations,
        n_action_steps,
        gamma=hyperparameters["gamma"],
        lr=hyperparameters["learning_rate"],
        device=env.device,
    )

    agent.load(os.path.join(os.getcwd(), "scripts", "rl_playground", "DQN", "out", timestamp, "target_net.pt"))

    # logging
    rewards_list = []
    average_rewards = []
    cumulative_reward = 0.0

    pbar = tqdm(total=hyperparameters["n_episodes"])

    # run the simulation

    # TODO: make this prettier
    def cleanup(*args):
        env.close()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        axs[0].scatter(range(len(rewards_list)), rewards_list, s=4, alpha=0.5, label="Rewards")
        axs[0].plot(range(len(rewards_list)), average_rewards, "x--", color="magenta", alpha=0.5, label="Eval Rewards")
        axs[0].set(title="Rewards")

        plt.tight_layout()
        out_path = os.path.join(os.getcwd(), "scripts", "rl_playground", "DQN", "out", timestamp)
        plt.savefig(os.path.join(out_path, "eval_rewards.png"))
        sys.exit(0)

    atexit.register(cleanup)
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    for ep in range(hyperparameters["n_episodes"] + 1):
        terminated = False
        state, _ = env.reset()
        state = state["policy"]
        while simulation_app.is_running() and not terminated:
            action = agent.get_action(state, 0.0)

            # step the environment
            continuous_action = agent.index_to_action(action)
            obs, rewards, terminated, truncated, info = env.step(continuous_action)

            state = obs["policy"]

            # logging
            cumulative_reward += rewards.item()

        rewards_list.append(cumulative_reward)
        cumulative_reward = 0.0

        n_avg = min(ep + 1, 100)
        average_rewards.append(sum(rewards_list[-n_avg:]) / n_avg)
        pbar.update(1)
        pbar.set_description(
            ", ".join(
                [
                    f"Episode: {ep}",
                    f"Rewards: {rewards_list[-1]:.4f}",
                    f"Avg. rewards ({n_avg}): {average_rewards[-1]:.4f}",
                ]
            )
        )

    cleanup(None)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
