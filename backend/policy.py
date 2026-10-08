"""Versioned deployment actor: no critics, optimizers or replay buffers."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from backend.actor import Actor
from backend.control import ACTION_SCHEMA, STATE_FIELDS, ControlSpec

POLICY_FORMAT = "camman.pan-policy"
POLICY_VERSION = 1


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        torch.save(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def export_actor(actor, path, spec, *, training=None):
    if (
        actor.fc1.in_features != len(STATE_FIELDS)
        or actor.fc3.out_features != 1
        or actor.max_action != 1
    ):
        raise ValueError(
            "Deployment requires the Camman state schema and one normalized pan action"
        )
    checkpoint = {
        "format": POLICY_FORMAT,
        "version": POLICY_VERSION,
        "state_fields": list(STATE_FIELDS),
        "action_schema": ACTION_SCHEMA,
        "hidden_sizes": [actor.fc1.out_features, actor.fc2.out_features],
        "control": asdict(spec),
        "actor_state_dict": {
            key: value.detach().cpu().clone() for key, value in actor.state_dict().items()
        },
        "training": training or {},
    }
    atomic_save(checkpoint, path)
    return checkpoint


class ActorPolicy:
    def __init__(self, model_path, device="cpu"):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        if checkpoint.get("format") != POLICY_FORMAT:
            raise ValueError(
                "Actor lacks the Camman policy contract. Retrain with camman-train-agent; "
                "legacy raw weights do not encode their state/action scale."
            )
        if (
            checkpoint.get("version") != POLICY_VERSION
            or checkpoint.get("state_fields") != list(STATE_FIELDS)
            or checkpoint.get("action_schema") != ACTION_SCHEMA
        ):
            raise ValueError("Unsupported actor state/action schema or version")
        hidden = checkpoint.get("hidden_sizes", [])
        if len(hidden) != 2 or any(not isinstance(v, int) or not 8 <= v <= 2048 for v in hidden):
            raise ValueError("Invalid actor architecture")
        self.spec = ControlSpec(**checkpoint["control"])
        self.device = torch.device(device)
        self.actor = Actor(len(STATE_FIELDS), 1, 1.0, *hidden)
        self.actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
        if any(not torch.isfinite(p).all() for p in self.actor.parameters()):
            raise ValueError("Actor checkpoint contains non-finite weights")
        self.actor.to(self.device).eval().requires_grad_(False)
        self.metadata = {k: v for k, v in checkpoint.items() if k != "actor_state_dict"}

    def choose_action(self, state):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (len(STATE_FIELDS),) or not np.isfinite(state).all():
            raise ValueError(f"Actor state must contain {len(STATE_FIELDS)} finite values")
        tensor = torch.from_numpy(state).to(self.device).unsqueeze(0)
        with torch.inference_mode():
            action = self.actor(tensor).cpu().numpy().reshape(-1)
        if not np.isfinite(action).all():
            raise ValueError("Actor returned a non-finite action")
        return np.clip(action, -1.0, 1.0)
