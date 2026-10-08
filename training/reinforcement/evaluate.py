"""Deterministic closed-loop validation on trajectories excluded from training."""

import argparse
import json
from pathlib import Path

import numpy as np

from backend.policy import ActorPolicy
from training.reinforcement.environment import SIMULATOR_VERSION, CameraControlEnv, SimulationSpec
from training.reinforcement.reward import RewardSystem
from training.reinforcement.tracks import TrackSequence


def evaluate_policy(policy, sequences, spec, simulation, *, max_steps=1000, reward_weights=None):
    env = CameraControlEnv(
        sequences,
        spec,
        simulation,
        max_steps=max_steps,
        training=False,
        reward_system=RewardSystem(reward_weights),
    )
    totals = dict(
        reward=0.0, steps=0, known=0, visible=0, centered=0, error=0.0, effort=0.0, change=0.0
    )
    choose = policy.choose_action if hasattr(policy, "choose_action") else policy
    for index in range(len(sequences)):
        for offset in (-0.3, 0.0, 0.3):
            state, _ = env.reset(options={"sequence": index, "offset": offset})
            previous = 0.0
            while True:
                state, reward, terminated, truncated, info = env.step(choose(state))
                action = info["executed_action"]
                totals["reward"] += reward
                totals["steps"] += 1
                totals["known"] += info["known_target"]
                totals["visible"] += info["visible"]
                totals["centered"] += info["visible"] and abs(info["dx"]) < 0.1
                totals["error"] += abs(info["dx"]) if info["known_target"] else 0
                totals["effort"] += abs(action)
                totals["change"] += abs(action - previous)
                previous = action
                if terminated or truncated:
                    break
    steps, known = totals["steps"], max(totals["known"], 1)
    return {
        "mean_reward": totals["reward"] / steps,
        "visible_fraction": totals["visible"] / known,
        "centered_fraction": totals["centered"] / known,
        "mean_absolute_error_fov": totals["error"] / known,
        "mean_absolute_pan": totals["effort"] / steps,
        "mean_action_change": totals["change"] / steps,
        "steps": steps,
        "known_target_steps": totals["known"],
    }


def baselines(sequences, spec, simulation, *, max_steps=1000, reward_weights=None):
    return {
        "stationary": evaluate_policy(
            lambda state: [0.0],
            sequences,
            spec,
            simulation,
            max_steps=max_steps,
            reward_weights=reward_weights,
        ),
        "proportional": evaluate_policy(
            lambda state: [np.clip(3 * state[0], -1, 1) if state[3] else 0],
            sequences,
            spec,
            simulation,
            max_steps=max_steps,
            reward_weights=reward_weights,
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--tracks", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    policy = ActorPolicy(args.actor, args.device)
    simulation = SimulationSpec(**policy.metadata["training"].get("simulation", {}))
    reward_weights = policy.metadata["training"].get("reward_weights")
    sequences = [TrackSequence.load(path) for path in args.tracks]
    report = {
        "evaluation_simulator_version": SIMULATOR_VERSION,
        "training_simulator_version": policy.metadata["training"].get("simulator_version", 1),
        "actor": evaluate_policy(
            policy,
            sequences,
            policy.spec,
            simulation,
            max_steps=args.steps,
            reward_weights=reward_weights,
        ),
        "baselines": baselines(
            sequences, policy.spec, simulation, max_steps=args.steps, reward_weights=reward_weights
        ),
        "tracks": [str(Path(path).resolve()) for path in args.tracks],
    }
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
