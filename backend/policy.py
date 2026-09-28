"""Deployment-only actor policy; optimizers and replay stay in training/."""

import numpy as np
import torch

from backend.actor import Actor


class ActorPolicy:
    def __init__(self, model_path, device="cpu", *, state_size=4, action_size=1, max_action=1.0):
        self.device = device
        self.max_action = max_action
        self.actor = Actor(state_size, action_size, max_action)
        self.actor.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
        self.actor.to(device).eval()

    def choose_action(self, state):
        tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor(tensor).cpu().numpy().reshape(-1)
        return np.clip(action, -self.max_action, self.max_action)
