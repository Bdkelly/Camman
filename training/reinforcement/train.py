"""Train/resume a DDPG camera policy; hardware is never opened during training."""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from backend.control import STATE_FIELDS, ControlSpec
from backend.models import load_model_from_path
from backend.policy import atomic_save, export_actor
from training.reinforcement import config
from training.reinforcement.agent import RLAgent
from training.reinforcement.environment import SIMULATOR_VERSION, CameraControlEnv, SimulationSpec
from training.reinforcement.evaluate import baselines, evaluate_policy
from training.reinforcement.reward import RewardSystem
from training.reinforcement.tracks import TrackSequence, extract_tracks


@dataclass(frozen=True)
class TrainingOptions:
    batch_size: int = 128
    memory_size: int = 50000
    lr_actor: float = 0.0001
    lr_critic: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    grad_clip: float = 1.0
    warmup_steps: int = 1000
    noise_sigma: float = 0.2
    noise_decay: float = 0.995
    noise_min: float = 0.02
    eval_every: int = 10
    checkpoint_every: int = 10
    validation_fraction: float = 0.2
    seed: int = 0

    def __post_init__(self):
        if any(not math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Training options must be finite")
        if self.batch_size < 1 or self.memory_size < self.batch_size or self.warmup_steps < 0:
            raise ValueError("Invalid batch, replay or warmup size")
        if min(self.lr_actor, self.lr_critic, self.grad_clip) <= 0:
            raise ValueError("Learning rates and gradient limit must be positive")
        if not 0 <= self.gamma <= 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid gamma or target update rate")
        if not 0 <= self.noise_min <= self.noise_sigma or not 0 < self.noise_decay <= 1:
            raise ValueError("Invalid exploration noise settings")
        if min(self.eval_every, self.checkpoint_every) < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("Invalid evaluation/checkpoint interval or holdout fraction")


def _configuration(cls, values, saved, key):
    base = saved.get(key, {}) if saved else {}
    override = asdict(values) if isinstance(values, cls) else (values or {})
    resolved = cls(**{**base, **override})
    if saved and asdict(resolved) != base:
        raise ValueError(
            f"Resume {key} differs from checkpoint; start a new run for changed settings"
        )
    return resolved


def _datasets(paths, validation_paths, fraction):
    full = [TrackSequence.load(path) for path in paths]
    if validation_paths:
        train, validation = full, [TrackSequence.load(path) for path in validation_paths]
        if {s.fingerprint() for s in train} & {s.fingerprint() for s in validation}:
            raise ValueError("Training and validation contain an identical trajectory")
        split = "separate trajectories (use different games)"
    else:
        train, validation = [], []
        for sequence in full:
            boundary = int(len(sequence.centers) * (1 - fraction))
            train.append(sequence.slice(0, boundary))
            validation.append(sequence.slice(boundary, len(sequence.centers)))
        split = "contiguous temporal holdout; separate games are preferred"
    # Validate both sets before allocating an agent or writing training artifacts.
    for data in (train, validation):
        if any(not s.detected[:-1].any() for s in data):
            raise ValueError(
                "A training/validation split has no detections; provide better tracks or another validation game"
            )
    signatures = {
        "train": [s.fingerprint() for s in train],
        "validation": [s.fingerprint() for s in validation],
    }
    return train, validation, signatures, split


def train_agent(
    videopth=None,
    model_path=None,
    num_episodes=config.NUM_EPISODES,
    max_t=None,
    *,
    checkpoint_dir=None,
    device=None,
    tracks_paths=None,
    validation_paths=None,
    control=None,
    simulation=None,
    options=None,
    resume=None,
    cpu_threads=None,
):
    output = Path(
        checkpoint_dir or (Path(resume).parent if resume else config.CHECKPOINT_DIR)
    ).resolve()
    if num_episodes < 1:
        raise ValueError("Episodes must be positive")
    if cpu_threads is not None:
        if cpu_threads < 1:
            raise ValueError("CPU threads must be positive")
        torch.set_num_threads(cpu_threads)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    saved = torch.load(resume, map_location="cpu", weights_only=True) if resume else None
    if resume and (
        not isinstance(saved, dict)
        or saved.get("format") != "camman.ddpg-training"
        or saved.get("version") != 1
    ):
        raise ValueError("Resume requires a complete Camman training checkpoint")
    if saved and saved.get("simulator_version", 1) != SIMULATOR_VERSION:
        raise ValueError(
            "Resume simulator version differs from this code. Start a new run; "
            "old replay used different camera stopping behavior."
        )
    if (not saved or Path(resume).resolve() != output / "training_latest.pth") and any(
        (output / name).exists()
        for name in ("training_latest.pth", "actor_episode_best.pth", "actor_episode_final.pth")
    ):
        raise ValueError(
            "Output already contains a run; use --resume or choose a new output directory"
        )
    spec = _configuration(ControlSpec, control, saved, "control")
    sim = _configuration(SimulationSpec, simulation, saved, "simulation")
    opts = _configuration(TrainingOptions, options, saved, "options")
    reward_weights = dict(
        saved.get("reward_weights", config.RWD_WEIGHTS) if saved else config.RWD_WEIGHTS
    )
    max_t = max_t if max_t is not None else saved["max_steps"] if saved else config.MAX_T
    if max_t < 1 or (saved and max_t != saved["max_steps"]):
        raise ValueError("Steps must be positive and match the resumed run")
    if saved and num_episodes <= saved["episode"]:
        raise ValueError(
            "--episodes is the total target; it must exceed the completed episode count"
        )
    if saved:
        tracks_paths = tracks_paths or saved["tracks_paths"]
        validation_paths = (
            validation_paths if validation_paths is not None else saved["validation_paths"]
        )
    if tracks_paths and videopth:
        raise ValueError("Choose cached tracks or a video, not both")
    if not tracks_paths:
        if not videopth or not model_path:
            raise ValueError("Provide --tracks, or --video together with --model")
        detector = load_model_from_path(model_path, device)
        tracks = extract_tracks(videopth, detector, device)
        del detector
        cache = output / "tracks" / "video.npz"
        tracks.metadata["model"] = str(Path(model_path).resolve())
        tracks.save(cache)
        tracks_paths = [cache]
    paths = [str(Path(path).resolve()) for path in tracks_paths]
    val_paths = [str(Path(path).resolve()) for path in (validation_paths or [])]
    train, validation, signatures, split = _datasets(paths, val_paths, opts.validation_fraction)
    if saved and signatures != saved["data_signatures"]:
        raise ValueError("Resume data differs from the saved training/validation trajectories")
    print(f"Validation: {split}")
    torch.manual_seed(opts.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(opts.seed)
    agent_keys = (
        "batch_size",
        "memory_size",
        "lr_actor",
        "lr_critic",
        "gamma",
        "tau",
        "grad_clip",
        "seed",
    )
    agent = RLAgent(
        len(STATE_FIELDS), 1, 1.0, device, **{key: getattr(opts, key) for key in agent_keys}
    )
    env = CameraControlEnv(
        train,
        spec,
        sim,
        max_steps=max_t,
        seed=opts.seed + 1,
        reward_system=RewardSystem(reward_weights),
    )
    rng = np.random.default_rng(opts.seed + 2)
    start_episode, total_steps, best_score = 0, 0, -float("inf")
    sigma = opts.noise_sigma
    best_actor = best_critic = None
    if saved:
        agent.load_state_dict(saved["agent"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda" and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        rng.bit_generator.state = json.loads(saved["exploration_rng"])
        env.rng.bit_generator.state = json.loads(saved["environment_rng"])
        start_episode, total_steps = saved["episode"], saved["total_steps"]
        sigma, best_score = saved["noise_sigma"], saved["best_score"]
        best_actor, best_critic = saved["best_actor"], saved["best_critic"]
    output.mkdir(parents=True, exist_ok=True)
    comparison = baselines(validation, spec, sim, max_steps=max_t, reward_weights=reward_weights)
    run = {
        "simulator_version": SIMULATOR_VERSION,
        "control": asdict(spec),
        "simulation": asdict(sim),
        "options": asdict(opts),
        "reward_weights": reward_weights,
        "max_steps": max_t,
        "tracks_paths": paths,
        "validation_paths": val_paths,
        "data_signatures": signatures,
        "validation_split": split,
        "baselines": comparison,
        "source_metadata": [s.metadata for s in train],
    }
    (output / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    if saved and (output / "metrics.jsonl").exists():
        # A crash can leave log entries beyond the last durable checkpoint.
        rows = []
        for line in (output / "metrics.jsonl").read_text().splitlines():
            try:
                if json.loads(line)["episode"] <= start_episode:
                    rows.append(line)
            except (ValueError, KeyError):
                continue  # Ignore a partial line left by an interrupted write.
        (output / "metrics.jsonl").write_text("".join(row + "\n" for row in rows))
    if best_actor is not None:
        atomic_save(best_actor, output / "actor_episode_best.pth")
        atomic_save(best_critic, output / "critic_episode_best.pth")
    latest_validation = None
    for episode in range(start_episode + 1, num_episodes + 1):
        state, _ = env.reset()
        score = 0.0
        count = 0
        for _ in range(max_t):
            action = (
                rng.uniform(-1, 1, 1)
                if total_steps < opts.warmup_steps
                else np.clip(agent.choose_action(state) + rng.normal(0, sigma, 1), -1, 1)
            )
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.add_experience(state, [info["executed_action"]], reward, next_state, terminated)
            total_steps += 1
            if total_steps >= opts.warmup_steps:
                agent.learn()
            state = next_state
            score += reward
            count += 1
            if terminated or truncated:
                break
        sigma = max(opts.noise_min, sigma * opts.noise_decay)
        metrics = {
            "episode": episode,
            "steps": count,
            "total_steps": total_steps,
            "updates": agent.updates,
            "mean_training_reward": score / count,
            "actor_loss": agent.actor_loss,
            "critic_loss": agent.critic_loss,
            "noise_sigma": sigma,
        }
        if agent.updates and (episode % opts.eval_every == 0 or episode == num_episodes):
            latest_validation = evaluate_policy(
                agent, validation, spec, sim, max_steps=max_t, reward_weights=reward_weights
            )
            metrics["validation"] = latest_validation
            if latest_validation["mean_reward"] > best_score:
                best_score = latest_validation["mean_reward"]
                metadata = {
                    "simulator_version": SIMULATOR_VERSION,
                    "episode": episode,
                    "updates": agent.updates,
                    "validation": latest_validation,
                    "baselines": comparison,
                    "simulation": asdict(sim),
                    "reward_weights": reward_weights,
                    "data_signatures": signatures,
                }
                best_actor = export_actor(
                    agent.actor_local, output / "actor_episode_best.pth", spec, training=metadata
                )
                best_critic = {
                    k: v.detach().cpu().clone() for k, v in agent.critic_local.state_dict().items()
                }
                atomic_save(best_critic, output / "critic_episode_best.pth")
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(
            f"Episode {episode}: mean_reward={score / count:.3f}, updates={agent.updates}, "
            f"actor_loss={agent.actor_loss:.4f}, critic_loss={agent.critic_loss:.4f}"
        )
        if episode % opts.checkpoint_every == 0 or episode == num_episodes:
            atomic_save(
                {
                    **run,
                    "format": "camman.ddpg-training",
                    "version": 1,
                    "episode": episode,
                    "total_steps": total_steps,
                    "noise_sigma": sigma,
                    "best_score": best_score,
                    "best_actor": best_actor,
                    "best_critic": best_critic,
                    "agent": agent.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                    "exploration_rng": json.dumps(rng.bit_generator.state),
                    "environment_rng": json.dumps(env.rng.bit_generator.state),
                },
                output / "training_latest.pth",
            )
    if not agent.updates:
        raise RuntimeError(
            "No optimizer updates occurred. No deployment actor was exported. "
            "Resume for more episodes or start a run with a smaller batch/warmup."
        )
    export_actor(
        agent.actor_local,
        output / "actor_episode_final.pth",
        spec,
        training={
            "simulator_version": SIMULATOR_VERSION,
            "episode": num_episodes,
            "updates": agent.updates,
            "validation": latest_validation,
            "simulation": asdict(sim),
            "reward_weights": reward_weights,
            "baselines": comparison,
            "data_signatures": signatures,
        },
    )
    atomic_save(agent.critic_local.state_dict(), output / "critic_episode_final.pth")
    report = {
        "completed_episodes": num_episodes,
        "total_steps": total_steps,
        "updates": agent.updates,
        "best_validation_reward": best_score,
        "beats_proportional_baseline": best_score > comparison["proportional"]["mean_reward"],
        "actor": str(output / "actor_episode_best.pth"),
        "baselines": comparison,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--tracks", nargs="+")
    data.add_argument("--video", help="One-pass extraction shortcut; requires --model")
    parser.add_argument("--model")
    parser.add_argument("--validation-tracks", nargs="+")
    parser.add_argument("--resume", help="Full training_latest.pth, not deployment actor weights")
    parser.add_argument(
        "--episodes", type=int, default=config.NUM_EPISODES, help="Total target episode count"
    )
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--output", help="New run: artifacts/reinforcement; resume: checkpoint directory"
    )
    parser.add_argument("--device")
    parser.add_argument("--cpu-threads", type=int)
    # Omitted settings inherit the resumed run, otherwise the dataclass defaults.
    for cls in (TrainingOptions, ControlSpec, SimulationSpec):
        for name, field in cls.__dataclass_fields__.items():
            parser.add_argument(
                "--" + name.replace("_", "-"), type=type(field.default), default=None
            )
    args = parser.parse_args(argv)
    kwargs = {}
    for key, cls in (
        ("options", TrainingOptions),
        ("control", ControlSpec),
        ("simulation", SimulationSpec),
    ):
        kwargs[key] = {
            name: getattr(args, name)
            for name in cls.__dataclass_fields__
            if getattr(args, name) is not None
        }
    try:
        train_agent(
            args.video,
            args.model,
            args.episodes,
            args.steps,
            checkpoint_dir=args.output,
            device=args.device,
            tracks_paths=args.tracks,
            validation_paths=args.validation_tracks,
            resume=args.resume,
            cpu_threads=args.cpu_threads,
            **kwargs,
        )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
