"""
agentTrain — DDPG training loop for the camera pan tracking agent.

Entry point: run ``python agentTrain.py`` after configuring paths and
hyperparameters in ``config.py``.

Training outline
----------------
For each episode:

1. Reset the ``CameraControlEnv`` (seek video to frame 0, re-centre window).
2. Reset OU exploration noise.
3. For each step up to ``MAX_T``:
   a. Choose a deterministic action from the actor.
   b. Add decayed OU noise for exploration.
   c. Step the environment to get the next state, reward, and done flag.
   d. Store the transition in the replay buffer.
   e. Perform one DDPG learning update.
4. Decay noise sigma by ``NOISE_DECAY`` (annealed exploration).
5. Save a checkpoint every 100 episodes and whenever a new best
   100-episode average score is achieved.

The ``best`` checkpoint (``actor_best.pth`` / ``critic_best.pth``) is
the recommended model to copy to ``guiapp/agentModel/agentModel.pth``
for deployment.
"""

#Agent Calls
try:
    from RLAgent import RLAgent
    from camController import CameraControlEnv
    from utils.models import get_fasterrcnn_model_single_class
    from reward import RewardSystem
    from utils.noise import OUNoise
    import config
except:
    from .RLAgent import RLAgent
    from .camController import CameraControlEnv
    from .utils.models import get_fasterrcnn_model_single_class
    from .reward import RewardSystem
    from .utils.noise import OUNoise
    from . import config

#Standard Calls
from albumentations.pytorch import ToTensorV2
import albumentations as A
import os
import cv2
import numpy as np
import torch
from collections import deque


def vidget(videopth):
    """Open a video file and return capture object plus frame dimensions.

    Args:
        videopth (str): Path to the video file.

    Returns:
        tuple[cv2.VideoCapture | None, int, int]:
            - ``cap``: Opened VideoCapture, or None if the file could not be opened.
            - ``W``: Frame width in pixels (0 on failure).
            - ``H``: Frame height in pixels (0 on failure).
    """
    cap = cv2.VideoCapture(videopth)
    if not cap.isOpened():
        print(f"Error: Could not open video file {videopth}")
        return None, 0, 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, W, H


def save_checkpoint(agent, episode):
    """Save actor and critic local network weights to the checkpoint directory.

    Creates ``config.CHECKPOINT_DIR`` if it does not exist. Files are named
    ``actor_episode_<episode>.pth`` and ``critic_episode_<episode>.pth``.
    Pass ``episode='best'`` to save as the best-so-far checkpoint.

    Args:
        agent (RLAgent): The agent whose weights are saved.
        episode (int | str): Episode number used in the filename, or ``'best'``.
    """
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)

    actor_path = os.path.join(config.CHECKPOINT_DIR, f'actor_episode_{episode}.pth')
    critic_path = os.path.join(config.CHECKPOINT_DIR, f'critic_episode_{episode}.pth')

    torch.save(agent.actor_local.state_dict(), actor_path)
    torch.save(agent.critic_local.state_dict(), critic_path)
    print(f"\nCheckpoint saved for episode {episode}")


def train_agent(videopth, model_path, num_episodes=config.NUM_EPISODES, max_t=config.MAX_T):
    """Run the full DDPG training loop and return the final actor weights.

    Sets up all components (detection model, environment, agent, noise),
    then iterates over episodes. A live OpenCV window shows the agent's
    tracking window (green rectangle) and per-step state/action info.
    Press **Q** at any time to interrupt training early.

    Checkpoints are saved to ``config.CHECKPOINT_DIR``:

    - Every ``100`` episodes: ``actor_episode_<N>.pth``
    - Whenever 100-episode average improves: ``actor_episode_best.pth``

    Args:
        videopth (str): Path to the training video file.
        model_path (str): Path to the pre-trained Faster R-CNN ``.pth`` weights.
        num_episodes (int): Total number of training episodes. Default: config.NUM_EPISODES.
        max_t (int): Maximum steps per episode before forced reset. Default: config.MAX_T.

    Returns:
        collections.OrderedDict | None: The final ``actor_local`` state dict on
            completion, or None if initialisation fails (bad video / model path).
    """
    num_classes = config.NUM_CLASSES
    device = config.DEVICE
    cap, W, H = vidget(videopth)

    if cap is None:
        return

    reward_system = RewardSystem(config.RWD_WEIGHTS)
    
    agent = RLAgent(config.STATE_SIZE, config.ACTION_SIZE, config.MAX_ACTION, device)
    
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    detection_model = get_fasterrcnn_model_single_class(num_classes).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Detection model file not found at {model_path}")
        return
    detection_model.load_state_dict(torch.load(model_path, map_location=device))
    detection_model.eval()
    
    env = CameraControlEnv(
        cap=cap, 
        detection_model=detection_model, 
        transform=transform, 
        device=device,
        frame_center_x=W // 2,
        frame_center_y=H // 2,
        max_action=config.MAX_ACTION,
        reward_system=reward_system,
    )

    print(f"Starting DDPG Training for {num_episodes} episodes...")
    scores_deque = deque(maxlen=100)
    ou_noise = OUNoise(config.ACTION_SIZE, theta=config.NOISE_THETA, sigma=config.NOISE_SIGMA)
    best_score = -np.inf
    
    for i_episode in range(1, num_episodes + 1):
        state, frame_with_detections = env.reset() 
        ou_noise.reset()
        score = 0
        
        for t in range(max_t):
            action = agent.choose_action(state)
            action_with_noise = action + ou_noise.sample()
            action_with_noise = np.clip(action_with_noise, -config.MAX_ACTION, config.MAX_ACTION)

            next_state, reward, done, action_taken, frame_with_detections = env.step(action_with_noise)
            
            agent.add_experience(state, action_with_noise, reward, next_state, done)
            agent.learn()
            
            state = next_state
            score += reward

            if len(action_taken) == 1:
                text = f"Pan: {action_taken[0]:.2f}, State: {state[0]:.2f}, , R={reward:.2f}"
            else:
                text = f"Pan: {action_taken[0]:.2f}, Tilt: {action_taken[1]:.2f}"
            print(f"State: dx={state[0]:.3f}, dy={state[1]:.3f}, Action: {action_with_noise[0]:.3f}")
            cv2.putText(frame_with_detections, text, 
                        (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('DDPG Tracking (Green Box is Agent Control Window)', frame_with_detections)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if done:
                break
        
        scores_deque.append(score)
        avg_score = np.mean(scores_deque)
        
        if avg_score > best_score:
            best_score = avg_score
            save_checkpoint(agent, 'best')
        
        
        ou_noise.sigma = max(config.NOISE_SIGMA_MIN, ou_noise.sigma * config.NOISE_DECAY)

        print(f'\rEpisode {i_episode}\tAverage Score (100 eps): {avg_score:.2f}\tScore: {score:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score (100 eps): {avg_score:.2f}')
            save_checkpoint(agent, i_episode)
            
    cap.release()
    cv2.destroyAllWindows()
    print("\nTraining complete.")
    return agent.actor_local.state_dict()

if __name__ == '__main__':
    trained_actor_weights = train_agent(config.VIDEO_PATH, config.MODEL_PATH)
    print("Actor weights trained.")
    pass
