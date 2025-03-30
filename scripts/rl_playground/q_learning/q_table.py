import math
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnv


class QTable:
    """Q-table for Q-learning algorithm."""

    action_range = {"joint_efforts": (-12.0, 12.0)}
    observation_range = {
        "slider_to_cart": {"joint_pos_rel": (-3.0, 3.0), "joint_vel_rel": (-1.0, 1.0)},
        "cart_to_pole": {"joint_pos_rel": (-math.pi / 4, math.pi / 4), "joint_vel_rel": (-3.5, 3.5)},
    }
    observation_keys = ["slider_to_cart", "cart_to_pole"]
    observation_type_keys = ["joint_pos_rel", "joint_vel_rel"]

    def __init__(
        self, env: ManagerBasedRLEnv, n_obs: int = 101, n_actions: int = 3, alpha: float = 0.1, gamma: float = 0.95
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
            f"Q-table range:\naction: {self._index_to_action(0)} - {self._index_to_action(self.n_actions - 1)}"
            f"\nobservation: {self._index_to_observation(0, 'slider_to_cart', 'joint_pos_rel')} - {self._index_to_observation(self.n_obs - 1, 'slider_to_cart', 'joint_pos_rel')}",
            f"\nobservation: {self._index_to_observation(0, 'slider_to_cart', 'joint_vel_rel')} - {self._index_to_observation(self.n_obs - 1, 'slider_to_cart', 'joint_vel_rel')}",
            f"\nobservation: {self._index_to_observation(0, 'cart_to_pole', 'joint_pos_rel')} - {self._index_to_observation(self.n_obs - 1, 'cart_to_pole', 'joint_pos_rel')}",
            f"\nobservation: {self._index_to_observation(0, 'cart_to_pole', 'joint_vel_rel')} - {self._index_to_observation(self.n_obs - 1, 'cart_to_pole', 'joint_vel_rel')}",
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
        assert self._observation_to_index(math.pi / 3, "cart_to_pole", "joint_pos_rel") == self.n_obs - 1

    def _action_to_index(self, action: float) -> int:
        action_range = self.action_range["joint_efforts"][1] - self.action_range["joint_efforts"][0]
        action_index = int((action - self.action_range["joint_efforts"][0]) / action_range * (self.n_actions - 1))
        return np.clip(action_index, 0, self.n_actions - 1)

    def _index_to_action(self, index: int) -> float:
        action_range = self.action_range["joint_efforts"][1] - self.action_range["joint_efforts"][0]
        action = index * action_range / (self.n_actions - 1) + self.action_range["joint_efforts"][0]
        return action

    def _observation_to_index(self, observation: float, joint: str, obs_type: str) -> int:
        obs_range = self.observation_range[joint][obs_type][1] - self.observation_range[joint][obs_type][0]
        obs_index = int((observation - self.observation_range[joint][obs_type][0]) / obs_range * (self.n_obs - 1))
        return np.clip(obs_index, 0, self.n_obs - 1)

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

    def update(
        self,
        obs: torch.Tensor,
        next_obs: torch.Tensor,
        act: torch.Tensor,
        rewards: torch.Tensor,
        alpha: float | None = None,
    ):
        if alpha is not None:
            self.alpha = alpha

        obs_indices = tuple((self.all_observations_to_index(obs)).tolist())
        next_obs_indices = tuple((self.all_observations_to_index(next_obs)).tolist())
        action_indices = self._action_to_index(act.item())
        current_indices = obs_indices + (action_indices,)

        self.q_table[current_indices] += self.alpha * (
            rewards.item()
            + self.gamma * torch.max(self.q_table[next_obs_indices]).item()
            - self.q_table[current_indices]
        )

        # print(f"Q-table {current_indices} updated to {self.q_table[current_indices]}")

    def save(self, path: str):
        torch.save(self.q_table, path)

    def load(self, path: str):
        self.q_table = torch.load(path)


class QAgent:

    def __init__(self, env: ManagerBasedRLEnv, q_table: QTable, epsilon: float = 0.1):
        self.q_table = q_table
        self.epsilon = epsilon

    def get_action(self, obs: torch.Tensor, epsilon: float | None = None) -> torch.Tensor:
        if epsilon is not None:
            self.epsilon = epsilon

        obs_indices = tuple((self.q_table.all_observations_to_index(obs)).tolist())

        # epsilon-greedy policy
        if torch.rand(1) < self.epsilon:
            action_idx = torch.randint(0, self.q_table.n_actions, (1,))
        else:
            action_idx = torch.argmax(self.q_table.q_table[obs_indices])

        action = torch.tensor([[self.q_table._index_to_action(action_idx.item())]])
        return action
