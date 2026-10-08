"""Verify synthetic training, actor reload and serial output without opening hardware."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import serial
import torch

from backend.hardware.protocol import velocity_command
from backend.policy import ActorPolicy
from backend.tracking import TrackingController
from training.reinforcement.environment import SIMULATOR_VERSION, SimulationSpec
from training.reinforcement.evaluate import baselines, evaluate_policy
from training.reinforcement.tracks import synthetic_tracks
from training.reinforcement.train import train_agent


def check_serial_output(policy):
    """Exercise the actual runtime controller and pyserial's in-memory loopback.

    The only transport opened here is loop://. This does not emulate the MCU's
    handshake, watchdog or electrical output; those need firmware/hardware tests.
    """
    now = [10.0]
    controller = TrackingController(policy.spec, clock=lambda: now[0])
    commands = []
    actions = []
    with serial.serial_for_url("loop://", timeout=0.2, write_timeout=0.2) as connection:

        def receive(expected):
            wire = connection.readline()
            if wire != (expected + "\n").encode("ascii"):
                raise RuntimeError(f"Serial command mismatch: expected {expected!r}, got {wire!r}")
            commands.append(wire.decode("ascii").strip())

        def stop_and_check():
            controller.stop(connection, force=True, reset=True)
            receive("Stop")

        stop_and_check()
        moving_center = None
        try:
            # Fresh target observations at the checkpoint's actual control cadence.
            for x in (0.15, 0.3, 0.5, 0.7, 0.85, 0.65, 0.35):
                now[0] += policy.spec.period + 1e-6
                box = [{"box": (x * 1000 - 5, 495, x * 1000 + 5, 505)}]
                previous = controller.prev_action
                action = controller.update(
                    box, connection, 1000, 1000, agent=policy, observed_at=now[0]
                )
                if action is None or not math.isfinite(action) or not -1 <= action <= 1:
                    raise RuntimeError("Reloaded actor did not produce a bounded pan decision")
                actions.append(action)
                if action or previous:
                    receive(velocity_command(action))
                elif connection.in_waiting:
                    raise RuntimeError("Unexpected duplicate Stop in serial output")
                if action:
                    moving_center = x
            if moving_center is None:
                raise RuntimeError(
                    "Actor produced no motion in the verification observations; "
                    "increase --episodes or inspect the training output"
                )

            # Start from a fresh observation so each stop check begins in motion.
            def start_motion():
                stop_and_check()
                for x in (moving_center, 0.15, 0.85, 0.3, 0.7):
                    now[0] += policy.spec.period + 1e-6
                    action = controller.update(
                        [{"box": (x * 1000 - 5, 495, x * 1000 + 5, 505)}],
                        connection,
                        1000,
                        1000,
                        agent=policy,
                        observed_at=now[0],
                    )
                    if action:
                        receive(velocity_command(action))
                        return
                raise RuntimeError("Could not exercise the trained actor's moving-to-stopped path")

            start_motion()
            now[0] += policy.spec.lost_target_timeout_s + 1e-6
            controller.tick(connection)
            receive("Stop")
            if controller.prev_action != 0:
                raise RuntimeError("Lost-target timeout did not stop the actor")

            start_motion()
            controller.tick(connection, enabled=False)
            receive("Stop")
            if controller.prev_action != 0:
                raise RuntimeError("Disabling tracking did not stop the actor")
        finally:
            stop_and_check()
    return {
        "transport": "pyserial loop:// (no hardware)",
        "commands": commands,
        "policy_actions": actions,
        "lost_target_stop": True,
        "disabled_stop": True,
        "shutdown_stop": True,
        "hardware_handshake_tested": False,
    }


def verify_pipeline(output, *, device="cpu", episodes=30, steps=100):
    output = Path(output).resolve()
    if episodes < 1 or steps < 1:
        raise ValueError("Episodes and steps must be positive")
    # Never replace an existing model/run while running an installation check.
    output.mkdir(parents=True, exist_ok=False)
    train_path, validation_path = output / "train.npz", output / "validation.npz"
    synthetic_tracks(1800, seed=0).save(train_path)
    validation = synthetic_tracks(900, seed=91)
    validation.save(validation_path)
    result = train_agent(
        tracks_paths=[train_path],
        validation_paths=[validation_path],
        checkpoint_dir=output / "training",
        device=device,
        cpu_threads=1,
        num_episodes=episodes,
        max_t=steps,
        options={
            "batch_size": 32,
            "memory_size": 5000,
            "warmup_steps": 128,
            "eval_every": 5,
            "checkpoint_every": 5,
        },
    )
    policy = ActorPolicy(result["actor"], "cpu")
    if policy.metadata["training"]["updates"] <= 0:
        raise RuntimeError("Exported actor has no optimizer updates")
    simulation = SimulationSpec(**policy.metadata["training"]["simulation"])
    rewards = policy.metadata["training"]["reward_weights"]
    evaluation = evaluate_policy(
        policy, [validation], policy.spec, simulation, max_steps=steps, reward_weights=rewards
    )
    if not all(math.isfinite(value) for value in evaluation.values()):
        raise RuntimeError("Reloaded actor produced invalid evaluation metrics")
    # Compare saved metrics on the training device. Tiny GPU/CPU differences can
    # cross a deadband/centering threshold and change a closed-loop rollout.
    training_policy = (
        policy if torch.device(device).type == "cpu" else ActorPolicy(result["actor"], device)
    )
    reloaded_validation = (
        evaluation
        if training_policy is policy
        else evaluate_policy(
            training_policy,
            [validation],
            policy.spec,
            simulation,
            max_steps=steps,
            reward_weights=rewards,
        )
    )
    expected = policy.metadata["training"]["validation"]
    if any(
        not math.isclose(value, expected[key], rel_tol=1e-4, abs_tol=1e-5)
        for key, value in reloaded_validation.items()
    ):
        raise RuntimeError("Reloaded actor validation differs from its saved results")
    for dx in (-0.35, 0.0, 0.35):
        state = np.array([dx, 0, 0, 1, 0, 1, 0], dtype=np.float32)
        if not np.allclose(
            policy.choose_action(state), training_policy.choose_action(state), rtol=1e-4, atol=1e-5
        ):
            raise RuntimeError("CPU deployment actor differs from the training-device actor")
    comparison = baselines(
        [validation], policy.spec, simulation, max_steps=steps, reward_weights=rewards
    )
    wire = check_serial_output(policy)
    report = {
        "status": "passed",
        "scope": "Synthetic software verification; not game or physical hardware qualification",
        "simulator_version": SIMULATOR_VERSION,
        "training_device": str(device),
        "deployment_device": "cpu",
        "torch": str(torch.__version__),
        "episodes": episodes,
        "optimizer_updates": result["updates"],
        "actor": result["actor"],
        "actor_parameters": sum(parameter.numel() for parameter in policy.actor.parameters()),
        "actor_bytes": Path(result["actor"]).stat().st_size,
        "validation": evaluation,
        "reloaded_training_validation": reloaded_validation,
        "baselines": comparison,
        "beats_proportional_baseline": evaluation["mean_reward"]
        > comparison["proportional"]["mean_reward"],
        "serial": wire,
    }
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Pipeline verified. Report: {output / 'verification.json'}")
    print("Synthetic actor only: train on reviewed game trajectories before deployment.")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="New directory for the check and report")
    parser.add_argument(
        "--device", default="cpu", help="Training device; exported actor checked on CPU"
    )
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args(argv)
    try:
        verify_pipeline(args.output, device=args.device, episodes=args.episodes, steps=args.steps)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Pipeline verification failed: {exc}\n")


if __name__ == "__main__":
    main()
