# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to create a simple environment with a cartpole. It combines the concepts of
scene, action, observation and event managers to create an environment.
"""

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

import math
import torch

import isaaclab.envs.mdp as mdp
from isaaclab.envs import ManagerBasedEnv, ManagerBasedEnvCfg, ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleSceneCfg


@configclass
class ActionsCfg:
    """Action specifications for the environment."""

    joint_efforts = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=1.0)


@configclass
class ObservationsCfg:
    """Observation specifications for the environment."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # on startup
    add_pole_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["pole"]),
            "mass_distribution_params": (0.1, 0.5),
            "operation": "add",
        },
    )

    # on reset
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.1, 0.1),
        },
    )

    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.125 * math.pi, 0.125 * math.pi),
            "velocity_range": (-0.01 * math.pi, 0.01 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # (1) Constant running reward
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    # (2) Failure penalty
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    # (3) Primary task: keep pole upright
    pole_pos = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
    # (4) Shaping tasks: lower cart velocity
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    # (5) Shaping tasks: lower pole angular velocity
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Time out
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # (2) Cart out of bounds
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the cartpole environment."""

    # Scene settings
    scene = CartpoleSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    events = EventCfg()
    rewards = RewardsCfg()
    terminations = TerminationsCfg()

    def __post_init__(self):
        """Post initialization."""
        # viewer settings
        self.viewer.eye = [4.5, 0.0, 6.0]
        self.viewer.lookat = [0.0, 0.0, 2.0]
        # step settings
        self.decimation = 4  # env step every 4 sim steps: 200Hz / 4 = 50Hz
        # simulation settings
        self.sim.dt = 0.005  # sim step every 5ms: 200Hz

        # RL
        self.episode_length_s = 5.0  # 5 seconds per episode


class QTable:
    """Q-table for Q-learning algorithm."""

    action_range = {"joint_efforts": (-1.0, 1.0)}
    observation_range = {
        "slider_to_cart": {"joint_pos_rel": (-1.0, 1.0), "joint_vel_rel": (-0.5, 0.5)},
        "cart_to_pole": {"joint_pos_rel": (-math.pi, math.pi), "joint_vel_rel": (-math.pi, math.pi)},
    }
    observation_keys = ["slider_to_cart", "cart_to_pole"]
    observation_type_keys = ["joint_pos_rel", "joint_vel_rel"]

    def __init__(
        self, env: ManagerBasedRLEnv, n_obs: int = 101, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.9
    ):

        self.gamma = gamma
        self.alpha = alpha

        self.n_obs = n_obs
        self.n_actions = n_actions
        self.action_space = env.action_manager.action_term_dim[0]
        self.observation_space = env.observation_manager.group_obs_dim.get("policy")[0]
        self.q_table = torch.zeros((self.n_obs,) * self.observation_space + (self.n_actions,) * self.action_space)

        print("Initialized Q-table with shape:", self.q_table.shape)
        print(
            f"Q-table range:\n action: {self._index_to_action(0)} - {self._index_to_action(self.n_actions - 1)}\n "
            f"observation: {self._index_to_observation(0, 'slider_to_cart', 'joint_pos_rel')} - {self._index_to_observation(self.n_obs - 1, 'slider_to_cart', 'joint_pos_rel')}"
        )

        self._test_index_conversion()

    def _test_index_conversion(self):
        # sanity check
        assert self._action_to_index(self._index_to_action(0)) == 0
        assert self._action_to_index(self._index_to_action(self.n_actions - 1)) == self.n_actions - 1
        assert (
            self._observation_to_index(
                self._index_to_observation(0, "slider_to_cart", "joint_pos_rel"), "slider_to_cart", "joint_pos_rel"
            )
            == 0
        )
        assert (
            self._observation_to_index(
                self._index_to_observation(self.n_obs - 1, "slider_to_cart", "joint_pos_rel"),
                "slider_to_cart",
                "joint_pos_rel",
            )
            == self.n_obs - 1
        )
        assert (
            torch.norm(
                self.all_observations_to_index(self.all_index_to_observations(torch.tensor([0.0, 0.0, 0.0, 0.0])))
                - torch.tensor([0.0, 0.0, 0.0, 0.0])
            )
            < 1e-6
        )

    def _action_to_index(self, action: float) -> int:
        action_range = self.action_range["joint_efforts"][1] - self.action_range["joint_efforts"][0]
        action_index = int((action - self.action_range["joint_efforts"][0]) / action_range * (self.n_actions - 1))
        return action_index % self.n_actions

    def _index_to_action(self, index: int) -> float:
        action_range = self.action_range["joint_efforts"][1] - self.action_range["joint_efforts"][0]
        action = index * action_range / (self.n_actions - 1) + self.action_range["joint_efforts"][0]
        return action

    def _observation_to_index(self, observation: float, joint: str, obs_type: str) -> int:
        obs_range = self.observation_range[joint][obs_type][1] - self.observation_range[joint][obs_type][0]
        obs_index = int((observation - self.observation_range[joint][obs_type][0]) / obs_range * (self.n_obs - 1))
        return obs_index % self.n_obs

    def all_observations_to_index(self, observations: torch.Tensor) -> torch.Tensor:
        t = [
            self._observation_to_index(obs, self.observation_keys[i // 2], self.observation_type_keys[i % 2])
            for i, obs in enumerate(observations[0])
        ]
        return torch.tensor(t)

    def all_index_to_observations(self, indices: torch.Tensor) -> torch.Tensor:
        t = [
            self._index_to_observation(ind, self.observation_keys[i // 2], self.observation_type_keys[i % 2])
            for i, ind in enumerate(indices)
        ]
        return torch.tensor([t])

    def _index_to_observation(self, index: int, joint: str, obs_type: str) -> float:
        obs_range = self.observation_range[joint][obs_type][1] - self.observation_range[joint][obs_type][0]
        observation = index * obs_range / (self.n_obs - 1) + self.observation_range[joint][obs_type][0]
        return observation

    def update(self, obs: torch.Tensor, next_obs: torch.Tensor, act: torch.Tensor, rewards: torch.Tensor):

        obs_indices = tuple((self.all_observations_to_index(obs)).tolist())
        next_obs_indices = tuple((self.all_observations_to_index(next_obs)).tolist())
        action_indices = self._action_to_index(act)
        current_indices = obs_indices + (action_indices,)

        self.q_table[current_indices] += self.alpha * (
            rewards.item()
            + self.gamma * torch.max(self.q_table[next_obs_indices]).item()
            - self.q_table[current_indices]
        )


class QAgent:

    def __init__(self, env: ManagerBasedRLEnv, q_table: QTable, epsilon: float = 0.1):
        self.q_table = q_table
        self.epsilon = epsilon

    def get_action(self, obs: torch.Tensor) -> torch.Tensor:
        obs_indices = tuple((self.q_table.all_observations_to_index(obs)).tolist())

        # epsilon-greedy policy

        if torch.rand(1) < self.epsilon:
            action_idx = torch.randint(0, self.q_table.n_actions, (1,))
        else:
            action_idx = torch.argmax(self.q_table.q_table[obs_indices])

        action = torch.tensor([[self.q_table._index_to_action(action_idx.item())]])
        return action


def main():
    """Main function."""
    # parse the arguments
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    # setup base environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # setup Q-table
    n_obs = 101
    n_actions = 11

    q_table = QTable(env, n_obs, n_actions)
    q_agent = QAgent(env, q_table, epsilon=0.3)

    # simulate physics
    count = 0
    last_obs, _ = env.reset()

    # initial action, observation
    last_action = torch.randn_like(env.action_manager.action)
    obs, rewards, terminated, truncated, info = env.step(last_action)
    last_obs = obs["policy"]

    cummulative_reward = 0
    # run the simulation
    while simulation_app.is_running():
        with torch.inference_mode():
            # reset
            if count % 300 == 0:
                count = 0
                obs, _ = env.reset()
                print("-" * 80)
                print("[INFO]: Resetting environment...")
            # sample random actions
            # action = torch.randn_like(env.action_manager.action)
            action = q_agent.get_action(last_obs)

            # step the environment
            obs, rewards, terminated, truncated, info = env.step(action)
            q_table.update(last_obs, obs["policy"], last_action, rewards)

            cummulative_reward += rewards.item()
            if count % 300 == 0:
                print(f"Random joint velocity: {action.item()}")
                print(f"Observation: {obs['policy']}")
                # print(f"indices: {q_table.all_observations_to_index(obs['policy'])}")
                print(f"Action: {action.item()}")
                print(f"Cummulative Rewards: {cummulative_reward}")

                cummulative_reward = 0

            # update counter
            count += 1
            last_obs = obs["policy"]
            last_action = action

    # close the environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
