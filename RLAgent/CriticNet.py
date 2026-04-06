"""
CriticNet — DDPG Q-value (critic) network for continuous camera pan control.

The Critic estimates the expected cumulative return Q(s, a) for a given
state-action pair. The state is first processed independently through a
hidden layer, then concatenated with the action before the second hidden
layer — a standard DDPG practice that lets the network learn how actions
modulate value from a rich state representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """Return uniform initialisation bounds for a linear layer.

    Uses the fan-in heuristic: bounds = ±1/sqrt(fan_in).

    Args:
        layer (nn.Linear): The linear layer whose fan-in is used.

    Returns:
        tuple[float, float]: (lower, upper) initialisation bounds.
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / (fan_in ** 0.5)
    return (-lim, lim)


class Critic(nn.Module):
    """Q-value network for the DDPG agent.

    Architecture:
        state → FC(256)+LN+ReLU → concat(action) → FC(128)+LN+ReLU → FC(1)

    The action is injected after the first hidden layer so the state
    embedding is formed before conditioning on the action.

    Args:
        state_size (int): Dimensionality of the input state vector.
        action_size (int): Dimensionality of the input action vector.
        max_action (float): Stored for reference; not used in the forward pass.
            Default: 1.0.
        fcs1_units (int): Units in the first hidden layer (state branch). Default: 256.
        fc2_units (int): Units in the second hidden layer (merged branch). Default: 128.
    """

    def __init__(self, state_size, action_size, max_action=1.0, fcs1_units=256, fc2_units=128):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, fcs1_units)
        self.ln1 = nn.LayerNorm(fcs1_units)

        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.max_action = max_action

        self.reset_parameters()

    def reset_parameters(self):
        """Initialise weights using fan-in bounds; final layer uses ±3e-3."""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Estimate Q(state, action).

        Args:
            state (torch.Tensor): State tensor of shape (batch, state_size).
            action (torch.Tensor): Action tensor of shape (batch, action_size).

        Returns:
            torch.Tensor: Scalar Q-value estimates of shape (batch, 1).
        """
        xs = self.fc1(state)
        xs = self.ln1(xs)
        xs = F.relu(xs)

        x = torch.cat((xs, action), dim=1)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)

        q_value = self.fc3(x)
        return q_value