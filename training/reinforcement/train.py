"""Headless camera-policy training; run with python -m training.reinforcement.train."""

import argparse
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.detection import FrameTransform
from backend.models import load_model_from_path
from training.reinforcement import config
from training.reinforcement.agent import RLAgent
from training.reinforcement.environment import CameraControlEnv
from training.reinforcement.noise import OUNoise
from training.reinforcement.reward import RewardSystem


def vidget(videopth):
    cap = cv2.VideoCapture(str(videopth))
    if not cap.isOpened():
        cap.release()
        return None, 0, 0
    return cap, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


def save_checkpoint(agent, episode, checkpoint_dir=None):
    output = Path(checkpoint_dir or config.CHECKPOINT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.actor_local.state_dict(), output / f"actor_episode_{episode}.pth")
    torch.save(agent.critic_local.state_dict(), output / f"critic_episode_{episode}.pth")


def train_agent(
    videopth,
    model_path,
    num_episodes=config.NUM_EPISODES,
    max_t=config.MAX_T,
    *,
    checkpoint_dir=None,
    device=None,
):
    if num_episodes < 1 or max_t < 1:
        raise ValueError("Episodes and steps must be positive")
    device = torch.device(device or config.DEVICE)
    detection_model = load_model_from_path(model_path, device)
    cap, width, height = vidget(videopth)
    if cap is None:
        raise FileNotFoundError(f"Could not open video: {videopth}")
    try:
        agent = RLAgent(config.STATE_SIZE, config.ACTION_SIZE, config.MAX_ACTION, device)
        env = CameraControlEnv(
            cap,
            detection_model,
            FrameTransform(),
            device,
            width // 2,
            height // 2,
            config.MAX_ACTION,
            RewardSystem(config.RWD_WEIGHTS),
        )
        scores = deque(maxlen=100)
        noise = OUNoise(config.ACTION_SIZE, theta=config.NOISE_THETA, sigma=config.NOISE_SIGMA)
        best_score = -np.inf
        for episode in range(1, num_episodes + 1):
            state, _ = env.reset()
            noise.reset()
            score = 0.0
            for _ in range(max_t):
                action = np.clip(
                    agent.choose_action(state) + noise.sample(),
                    -config.MAX_ACTION,
                    config.MAX_ACTION,
                )
                next_state, reward, done, _, _ = env.step(action)
                agent.add_experience(state, action, reward, next_state, done)
                agent.learn()
                state = next_state
                score += reward
                if done:
                    break
            scores.append(score)
            average = float(np.mean(scores))
            if average > best_score:
                best_score = average
                save_checkpoint(agent, "best", checkpoint_dir)
            noise.sigma = max(config.NOISE_SIGMA_MIN, noise.sigma * config.NOISE_DECAY)
            print(f"Episode {episode}: score={score:.2f}, average={average:.2f}")
            if episode % 100 == 0:
                save_checkpoint(agent, episode, checkpoint_dir)
        save_checkpoint(agent, "final", checkpoint_dir)
        return agent.actor_local.state_dict()
    finally:
        cap.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train the camera-control actor on a server")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True, help="Trained ball detector .pth")
    parser.add_argument("--episodes", type=int, default=config.NUM_EPISODES)
    parser.add_argument("--steps", type=int, default=config.MAX_T)
    parser.add_argument("--output", default=config.CHECKPOINT_DIR)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    train_agent(
        args.video,
        args.model,
        args.episodes,
        args.steps,
        checkpoint_dir=args.output,
        device=args.device,
    )


if __name__ == "__main__":
    main()
