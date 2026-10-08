"""DDPG with detached Bellman targets, bounded gradients and resumable replay."""

import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn.functional as F

from backend.actor import Actor
from training.reinforcement import config
from training.reinforcement.critic import Critic

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class ReplayBuffer:
    def __init__(self, capacity, state_size, action_size):
        self.capacity, self.size, self.cursor = capacity, 0, 0
        widths = [state_size, action_size, 1, state_size, 1]
        self.data = {
            key: torch.empty(capacity, width) for key, width in zip(Transition._fields, widths)
        }

    def __len__(self):
        return self.size

    def append(self, transition):
        for key, value in zip(Transition._fields, transition):
            self.data[key][self.cursor].copy_(value)
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, size):
        indices = torch.tensor(rng.sample(range(self.size), size))
        return Transition(*(self.data[key].index_select(0, indices) for key in Transition._fields))

    def state_dict(self):
        return {
            "size": self.size,
            "cursor": self.cursor,
            "data": {key: value[: self.size].clone() for key, value in self.data.items()},
        }

    def load_state_dict(self, state):
        self.size, self.cursor = state["size"], state["cursor"]
        if not 0 <= self.size <= self.capacity or not 0 <= self.cursor < self.capacity:
            raise ValueError("Invalid replay checkpoint")
        for key, value in state["data"].items():
            self.data[key][: self.size].copy_(value)


class RLAgent:
    def __init__(
        self,
        state_size,
        action_size,
        max_action,
        device,
        *,
        batch_size=config.BATCH_SIZE,
        memory_size=config.MEMORY_SIZE,
        lr_actor=config.LR_ACTOR,
        lr_critic=config.LR_CRITIC,
        gamma=config.GAMMA,
        tau=config.SOFT_UPDATE,
        grad_clip=1.0,
        seed=0,
    ):
        if batch_size < 1 or memory_size < batch_size or not 0 <= gamma <= 1 or not 0 < tau <= 1:
            raise ValueError("Invalid replay size, batch size, discount or target update rate")
        if min(lr_actor, lr_critic, grad_clip, max_action) <= 0:
            raise ValueError("Learning rates, gradient clip and action bound must be positive")
        self.state_size, self.action_size = state_size, action_size
        self.max_action, self.device = max_action, torch.device(device)
        self.gamma, self.softup, self.grad_clip = gamma, tau, grad_clip
        self.options = dict(
            batch_size=batch_size,
            memory_size=memory_size,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            grad_clip=grad_clip,
            seed=seed,
        )
        self.actor_local = Actor(state_size, action_size, max_action).to(self.device)
        self.actor_target = Actor(state_size, action_size, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor_local.state_dict())
        self.actor_target.eval().requires_grad_(False)
        self.critic_local = Critic(state_size, action_size, max_action).to(self.device)
        self.critic_target = Critic(state_size, action_size, max_action).to(self.device)
        self.critic_target.load_state_dict(self.critic_local.state_dict())
        self.critic_target.eval().requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=lr_critic)
        self.memory = ReplayBuffer(memory_size, state_size, action_size)
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.actor_loss = self.critic_loss = 0.0
        self.updates = 0

    def choose_action(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        if tensor.shape[1] != self.state_size or not torch.isfinite(tensor).all():
            raise ValueError("Invalid policy observation")
        with torch.inference_mode():
            action = self.actor_local(tensor).cpu().numpy().reshape(-1)
        if not np.isfinite(action).all():
            raise FloatingPointError("Actor returned non-finite actions")
        return np.clip(action, -self.max_action, self.max_action)

    @torch.no_grad()
    def soft_update(self, local_model, target_model):
        for target, local in zip(target_model.parameters(), local_model.parameters()):
            target.lerp_(local, self.softup)

    def add_experience(self, state, action, reward, next_state, done):
        values = [
            torch.as_tensor(v, dtype=torch.float32).reshape(-1).clone()
            for v in (state, action, [reward], next_state, [float(done)])
        ]
        if [v.numel() for v in values] != [
            self.state_size,
            self.action_size,
            1,
            self.state_size,
            1,
        ]:
            raise ValueError("Transition dimensions do not match the agent")
        if any(not torch.isfinite(v).all() for v in values):
            raise ValueError("Replay transitions must be finite")
        if torch.any(values[1].abs() > self.max_action + 1e-6):
            raise ValueError("Replay must store the executed bounded action")
        self.memory.append(Transition(*(v.cpu() for v in values)))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.rng, self.batch_size)
        states, actions, rewards, next_states, dones = (value.to(self.device) for value in batch)
        with torch.no_grad():
            q_next = self.critic_target(next_states, self.actor_target(next_states))
            targets = rewards + self.gamma * (1 - dones) * q_next
        critic_loss = F.mse_loss(self.critic_local(states, actions), targets)
        if not torch.isfinite(critic_loss):
            raise FloatingPointError("Non-finite critic loss")
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_local.parameters(), self.grad_clip, error_if_nonfinite=True
        )
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad(set_to_none=True)
        # Gradients must flow through Q's action input, not into Q's parameters.
        self.critic_local.requires_grad_(False)
        try:
            actor_loss = -self.critic_local(states, self.actor_local(states)).mean()
            if not torch.isfinite(actor_loss):
                raise FloatingPointError("Non-finite actor loss")
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor_local.parameters(), self.grad_clip, error_if_nonfinite=True
            )
            self.actor_optimizer.step()
        finally:
            self.critic_local.requires_grad_(True)
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)
        self.actor_loss, self.critic_loss = float(actor_loss.detach()), float(critic_loss.detach())
        self.updates += 1
        return {"actor_loss": self.actor_loss, "critic_loss": self.critic_loss}

    def state_dict(self):
        return {
            "state_size": self.state_size,
            "action_size": self.action_size,
            "max_action": self.max_action,
            "options": self.options,
            "actor": self.actor_local.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic": self.critic_local.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay": self.memory.state_dict(),
            "rng": self.rng.getstate(),
            "updates": self.updates,
            "actor_loss": self.actor_loss,
            "critic_loss": self.critic_loss,
        }

    def load_state_dict(self, state):
        if (state["state_size"], state["action_size"], state["max_action"], state["options"]) != (
            self.state_size,
            self.action_size,
            self.max_action,
            self.options,
        ):
            raise ValueError("Resume agent configuration does not match checkpoint")
        for name, model in (
            ("actor", self.actor_local),
            ("actor_target", self.actor_target),
            ("critic", self.critic_local),
            ("critic_target", self.critic_target),
        ):
            model.load_state_dict(state[name])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.memory.load_state_dict(state["replay"])
        self.rng.setstate(state["rng"])
        self.updates = state["updates"]
        self.actor_loss, self.critic_loss = state["actor_loss"], state["critic_loss"]
