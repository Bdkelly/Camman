import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
CHECKPOINT_DIR = "artifacts/reinforcement"
LOG_DIR = "artifacts/logs"

# Model parameters
STATE_SIZE = 7
ACTION_SIZE = 1
MAX_ACTION = 1.0
NUM_CLASSES = 2

# Training hyperparameters
NUM_EPISODES = 200
MAX_T = 5000
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
SOFT_UPDATE = 0.005
BATCH_SIZE = 128
MEMORY_SIZE = 50000

# Noise parameters
NOISE_SIGMA = 0.2
NOISE_THETA = 0.15
NOISE_DECAY = 0.999
NOISE_SIGMA_MIN = 0.01

# Reward weights
RWD_WEIGHTS = {
    "centering_peak": 1.0,
    "centering_decay": 8.0,
    "effort": 0.02,
    "stability": 0.1,
    "lost_ball_penalty": 1.0,
    "window_bonus": 0.25,
}
