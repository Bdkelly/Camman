"""
RLAgent — Deep Deterministic Policy Gradient (DDPG) agent for camera pan control.

DDPG is an off-policy, model-free actor-critic algorithm suited for continuous
action spaces. This implementation maintains separate *local* and *target*
networks for both actor and critic. Target networks are updated via Polyak
(soft) averaging after every learning step to stabilise training.

Experience is stored in a fixed-size replay buffer (deque). A random mini-batch
is sampled each learning step to break temporal correlations.

Typical usage::

    agent = RLAgent(state_size=4, action_size=1, max_action=5.0, device=device)
    action = agent.choose_action(state)
    agent.add_experience(state, action, reward, next_state, done)
    agent.learn()
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque, namedtuple
import os
import sys
try:
    from ActorNet import Actor
    from CriticNet import Critic
except:
    from .ActorNet import Actor
    from .CriticNet import Critic
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
import config


Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class RLAgent:
    """DDPG agent for continuous camera pan control.

    Maintains actor and critic networks (local + target copies each), an
    experience replay buffer, and Adam optimisers. Learning follows the
    standard DDPG update rule:

    1. Sample a random mini-batch from the replay buffer.
    2. Compute Bellman targets using the *target* critic and *target* actor.
    3. Update the *local* critic by minimising MSE against those targets.
    4. Update the *local* actor by maximising the *local* critic's Q-value.
    5. Soft-update both target networks toward their local counterparts.

    Args:
        state_size (int): Dimensionality of the state vector (4 for this system).
        action_size (int): Dimensionality of the action vector (1 for pan-only).
        max_action (float): Maximum absolute pan angle; clips chosen actions.
        device (torch.device): Device on which all tensors and models are placed.

    Attributes:
        actor_local (Actor): The policy network used for action selection and training.
        actor_target (Actor): Slowly-updated copy of actor_local used for stable targets.
        critic_local (Critic): The Q-network used for TD-error computation.
        critic_target (Critic): Slowly-updated copy of critic_local used for stable targets.
        memory (deque): Fixed-size replay buffer storing ``Transition`` namedtuples.
        actor_loss (float | Tensor): Most recent actor loss value (for logging).
        critic_loss (float | Tensor): Most recent critic loss value (for logging).
    """

    def __init__(self, state_size, action_size, max_action, device):
        self.state_size = state_size
        self.action_size = action_size
        self.max_action = max_action
        self.device = device
        self.gamma = config.GAMMA
        self.softup = config.SOFT_UPDATE
        self.actor_local = Actor(state_size, action_size, max_action).to(device)
        self.actor_target = Actor(state_size, action_size, max_action).to(device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=config.LR_ACTOR)
        self.critic_local = Critic(state_size, action_size, max_action).to(device)
        self.critic_target = Critic(state_size, action_size, max_action).to(device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=config.LR_CRITIC)
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        self.batch_size = config.BATCH_SIZE
        self.actor_loss = 0
        self.critic_loss = 0

    def choose_action(self, state):
        """Select a deterministic action for the given state (no exploration noise).

        Temporarily switches the actor to eval mode so that LayerNorm
        behaves correctly for a single sample, then restores training mode.

        Args:
            state (np.ndarray): State vector of shape (state_size,).

        Returns:
            np.ndarray: Action vector of shape (action_size,) clipped to
                [−max_action, +max_action].
        """
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy().squeeze()
            if action.ndim == 0:
                action = np.array([action])
        self.actor_local.train()

        return np.clip(action, -self.max_action, self.max_action)

    def soft_update(self, local_model, target_model):
        """Polyak-average a target network toward a local network.

        Applies the update: θ_target ← τ·θ_local + (1−τ)·θ_target
        where τ = ``self.softup`` (config.SOFT_UPDATE).

        Args:
            local_model (nn.Module): Source model whose weights are blended in.
            target_model (nn.Module): Target model whose weights are updated.
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.softup * local_param.data + (1.0 - self.softup) * target_param.data)

    def add_experience(self, state, action, reward, next_state, done):
        """Store a single transition in the replay buffer.

        Converts all inputs to tensors before storing so that ``learn()``
        can stack them efficiently without repeated conversions.

        Args:
            state (np.ndarray): State at time t, shape (state_size,).
            action (np.ndarray): Action taken at time t, shape (action_size,).
            reward (float): Scalar reward received.
            next_state (np.ndarray): State at time t+1, shape (state_size,).
            done (bool | float): Episode termination flag (1.0 = terminal).
        """
        state = torch.from_numpy(state).float()
        if action.ndim == 0:
            action = np.array([action], dtype=np.float32)
        action = torch.from_numpy(action).float()
        reward = torch.tensor([reward], dtype=torch.float)
        next_state = torch.from_numpy(next_state).float()
        done = torch.tensor([done], dtype=torch.float)

        self.memory.append(Transition(state, action, reward, next_state, done))

    def learn(self):
        """Sample a mini-batch and perform one DDPG gradient update.

        Does nothing if the replay buffer contains fewer transitions than
        ``self.batch_size``.

        Update steps:
            1. Compute TD targets using target actor and target critic.
            2. Update critic_local by minimising MSE(Q_expected, Q_targets).
            3. Update actor_local by maximising critic_local(s, actor_local(s)).
            4. Soft-update both target networks.
        """
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        states = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.action).to(self.device)
        rewards = torch.stack(batch.reward).to(self.device)
        next_states = torch.stack(batch.next_state).to(self.device)
        dones = torch.stack(batch.done).to(self.device).float()
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic_local(states, actions)
        self.critic_loss = F.mse_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        actions_pred = self.actor_local(states)
        self.actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        self.actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target)
        self.soft_update(self.actor_local, self.actor_target)