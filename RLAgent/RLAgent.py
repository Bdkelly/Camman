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
        state = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy().squeeze()
            if action.ndim == 0:
                action = np.array([action])
        self.actor_local.train()
            
        return np.clip(action, -self.max_action, self.max_action)
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.softup * local_param.data + (1.0 - self.softup) * target_param.data)
    
    def add_experience(self, state, action, reward, next_state, done):
        state = torch.from_numpy(state).float()
        if action.ndim == 0:
            action = np.array([action],dtype=np.float32)
        action = torch.from_numpy(action).float() 
        reward = torch.tensor([reward], dtype=torch.float)
        next_state = torch.from_numpy(next_state).float()
        done = torch.tensor([done], dtype=torch.float)

        self.memory.append(Transition(state, action, reward, next_state, done))
        
    def learn(self):
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