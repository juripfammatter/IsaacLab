import torch
from torch import Tensor, dtype
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from utils import Transition


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent(object):

    def __init__(
        self,
        n_observations: int,
        n_action_steps: int,
        gamma: float,
        lr: float,
        device: str,
    ) -> None:
        self.device = device

        self.action_value_net = DQN(n_observations, n_action_steps).to(self.device)
        self.target_value_net = DQN(n_observations, n_action_steps).to(self.device)
        self.target_value_net.load_state_dict(self.action_value_net.state_dict())

        self.optimizer = optim.AdamW(self.action_value_net.parameters(), lr=lr, amsgrad=True)  # TODO: add scheduler
        self.gamma = gamma

    def get_action(self, state: Tensor, epsilon: float) -> Tensor:
        """epsilon greedy action selection"""
        # TODO: make dimensions parametric

        if torch.rand(1).item() > epsilon:
            with torch.no_grad():
                # return self.action_value_net(state).argmax().item()
                return self.action_value_net(state).max(1).indices.view(1, 1)
        else:
            return torch.randint(
                0, self.action_value_net.layer3.out_features, (1, 1), dtype=torch.long, device=self.device
            )

    def update(self, transition_batch: list[Transition]):
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transition_batch))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool
        )
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)  # Bx4
        action_batch = torch.cat(batch.action)  # Bx1
        reward_batch = torch.cat(batch.reward)  # B

        # Compute Q(s_t, a_t) given the action a_t taken at state s_t
        state_action_values = self.action_value_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(len(transition_batch), device=self.device)

        # compute the next state values from target net
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_value_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.action_value_net.parameters(), 100)
        self.optimizer.step()

    def sync_target_net(self):
        """Update target network"""
        tau = 0.005
        target_net_state_dict = self.target_value_net.state_dict()
        policy_net_state_dict = self.action_value_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
        self.target_value_net.load_state_dict(target_net_state_dict)

    def index_to_action(self, index: Tensor) -> Tensor:
        """Convert index to action"""
        action_range = (-2, 2)
        action = (
            (index / (self.action_value_net.layer3.out_features - 1)) * (action_range[1] - action_range[0])
        ) + action_range[0]
        return action
