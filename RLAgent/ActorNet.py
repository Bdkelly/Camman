"""
ActorNet — DDPG policy (actor) network for continuous camera pan control.

The Actor maps a state observation to a deterministic continuous pan action.
It uses two fully-connected hidden layers with LayerNorm for training stability,
and scales the output to [−max_action, +max_action] via a tanh activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """Return uniform initialisation bounds for a linear layer.

    Uses the fan-in heuristic: bounds = ±1/sqrt(fan_in), which keeps
    initial activations in a stable range regardless of layer width.

    Args:
        layer (nn.Linear): The linear layer whose fan-in is used.

    Returns:
        tuple[float, float]: (lower, upper) initialisation bounds.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / (fan_in ** 0.5)
    return (-lim, lim)


class Actor(nn.Module):
    """Deterministic policy network for the DDPG agent.

    Architecture: state → FC(256)+LN+ReLU → FC(128)+LN+ReLU → FC(action_size)
    Output is scaled to [−max_action, +max_action] by multiplying by max_action
    after a tanh activation.

    Args:
        state_size (int): Dimensionality of the input state vector.
        action_size (int): Dimensionality of the output action vector.
        max_action (float): Maximum absolute value of each action component.
        fc1_units (int): Number of units in the first hidden layer. Default: 256.
        fc2_units (int): Number of units in the second hidden layer. Default: 128.
    """

    def __init__(self, state_size, action_size, max_action, fc1_units=256, fc2_units=128):
        super(Actor, self).__init__()
        self.max_action = max_action

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise weights using fan-in bounds; final layer uses ±3e-3."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Compute a deterministic action from the given state.

        Args:
            state (torch.Tensor): State tensor of shape (batch, state_size).

        Returns:
            torch.Tensor: Action tensor of shape (batch, action_size) with
                values in [−max_action, +max_action].
        """
        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        action = self.max_action * torch.tanh(self.fc3(x))
        return action