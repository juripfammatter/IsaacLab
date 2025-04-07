import torch
from torch import Tensor
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

    def __init__(self, action_value_net: DQN, target_value_net: DQN) -> None:
        self.action_value_net = action_value_net
        self.target_value_net = target_value_net

    def get_action(self, state: Tensor, epsilon: float):
        """epsilon greedy action selection"""
        if torch.rand(1).item() > epsilon:
            with torch.no_grad():
                action_idx = self.action_value_net(state).argmax().item()
        else:
            action_idx = torch.randint(0, self.action_value_net.layer3.out_features, (1,)).item()

        return self._index_to_action(action_idx)

    def update(self, transition_batch: list[Transition]):
        raise NotImplementedError

    def _index_to_action(self, index: int):
        """Convert index to action"""
        action_range = (-2, 2)
        action = (
            (index / (self.action_value_net.layer3.out_features - 1)) * (action_range[1] - action_range[0])
        ) + action_range[0]
        return torch.tensor([[action]], dtype=torch.float32)
