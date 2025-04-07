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

    def __init__(self, action_value_net: DQN, target_value_net: DQN, gamma: float, lr: float) -> None:
        self.action_value_net = action_value_net
        self.target_value_net = target_value_net
        self.device = action_value_net.layer1.weight.device
        self.optimizer = optim.AdamW(self.action_value_net.parameters(), lr=lr, amsgrad=True)
        self.gamma = gamma

    def get_action(self, state: Tensor, epsilon: float) -> Tensor:
        """epsilon greedy action selection"""
        if torch.rand(1).item() > epsilon:
            with torch.no_grad():
                action_idx = self.action_value_net(state).argmax().item()
        else:
            action_idx = torch.randint(0, self.action_value_net.layer3.out_features, (1,)).item()

        return torch.tensor([[action_idx]], dtype=torch.int64, device=self.device)

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

        print(f"{state_batch = }")
        print(f"{action_batch = }")
        print(f"{reward_batch = }")

        # Compute Q(s_t, a_t) given the action a_t taken at state s_t
        state_action_values = self.action_value_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(len(transition_batch), device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_value_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        print(f"{state_action_values = }")
        print(f"{next_state_values = }")
        print(f"{expected_state_action_values.unsqueeze(1) = }")

        # TODO: fix error with missing gradients. Maybe some dimensions don't check out
        # state_action_values.requires_grad = True
        # expected_state_action_values.requires_grad = True
        #
        # print(f"{state_action_values.requires_grad = }")
        # print(f"{expected_state_action_values.requires_grad = }")

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.action_value_net.parameters(), 100)
        self.optimizer.step()

    def index_to_action(self, index: Tensor) -> Tensor:
        """Convert index to action"""
        action_range = (-2, 2)
        action = (
            (index / (self.action_value_net.layer3.out_features - 1)) * (action_range[1] - action_range[0])
        ) + action_range[0]
        return action
        # return torch.tensor([[action]], dtype=torch.float32, device=self.device)
