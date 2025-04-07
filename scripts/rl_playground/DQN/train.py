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
import torch.optim as optim
from tqdm import tqdm
from cartpole_config import CartpoleEnvCfg
from dqn_agent import DQNAgent, DQN
from utils import ReplayMemory


from isaaclab.envs import ManagerBasedRLEnv

import matplotlib.pyplot as plt


class CustomScheduler:
    def __init__(self, initial_lr: float, gamma: float = 0.99):
        self.value = torch.nn.Parameter(torch.tensor([0.0]))
        self.optimizer = torch.optim.SGD([self.value], lr=initial_lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        self.optimizer.step()

    def step(self) -> float:
        self.scheduler.step()
        return self.scheduler.get_last_lr()[0]


hyperparameters = {
    "gamma": 0.99,
    "learning_rate": 1e-4,
    "target_update": 10,
    "epsilon": 0.1,
    "n_episodes": 100,
    "batch_size": 6,
    "memory_capacity": 10_000,
}


def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)
    n_observations = env.observation_manager.group_obs_dim["policy"][0]
    n_actions = env.action_manager.total_action_dim
    n_action_steps = 7

    # print(f"{n_observations = }")
    # print(f"{n_actions = }")

    # TODO
    # exploration scheduler (GLIE policy)
    epsilon_scheduler = CustomScheduler(hyperparameters["gamma"], gamma=0.9995)
    epsilon = epsilon_scheduler.step()

    # setup Agent
    action_value_net = DQN(n_observations, n_action_steps).to(env.device)
    target_value_net = DQN(n_observations, n_action_steps).to(env.device)
    target_value_net.load_state_dict(action_value_net.state_dict())

    # Optimizer
    # optimizer = optim.AdamW(action_value_net.parameters(), lr=hyperparameters["learning_rate"], amsgrad=True)
    # TODO: add scheduler

    agent = DQNAgent(
        action_value_net, target_value_net, gamma=hyperparameters["gamma"], lr=hyperparameters["learning_rate"]
    )

    # Replay memory
    memory = ReplayMemory(hyperparameters["memory_capacity"])

    # logging
    rewards_list = []
    cumulative_reward = 0.0
    pbar = tqdm(total=hyperparameters["n_episodes"])

    # run the simulation

    for ep in range(hyperparameters["n_episodes"] + 1):
        terminated = False
        with torch.inference_mode():
            state, _ = env.reset()
            state = state["policy"]
            while simulation_app.is_running() and not terminated:
                # action = torch.randn_like(env.action_manager.action)
                action = agent.get_action(state, epsilon)
                print(f"action: {agent.get_action(state, epsilon)}")
                print(f"continuous action: {agent.index_to_action(action)}")

                # step the environment
                continuous_action = agent.index_to_action(action)
                obs, rewards, terminated, truncated, info = env.step(continuous_action)

                if terminated:
                    next_state = None
                else:
                    next_state = obs["policy"]

                # print(f"{state = }, {action = }, {next_state = }, {rewards = }")
                memory.push(state, action, next_state, rewards)
                state = next_state

                if len(memory) >= hyperparameters["batch_size"]:
                    agent.update(memory.sample(hyperparameters["batch_size"]))

                # logging
                cumulative_reward += rewards.item()

                last_obs = obs["policy"]

            pbar.update(1)
            pbar.set_description(
                ", ".join(
                    [
                        f"Episode: {ep}",
                        # f"Rewards: {rewards_list[-1]:.4f}",
                        # f"Eval Rewards: {eval_rewards[-1]:.4f}",
                        # f"Alpha: {alpha:.4f}",
                        f"Epsilon: {epsilon:.4f}",
                    ]
                )
            )

    print(f"memory size: {len(memory)}")
    print(f"sample from memory: {memory.sample(1)}")

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
