import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# File paths
VIDEO_PATH = r'/Users/Ben/Documents/dever/python/ptorch/data/SmallVideo.mp4'
MODEL_PATH = r'/Users/Ben/Documents/dever/python/ptorch/models/fin_comb.pth'
CHECKPOINT_DIR = 'checkpoints'
LOG_DIR = 'logs'

# Model parameters
STATE_SIZE = 4
ACTION_SIZE = 1
MAX_ACTION = 5.0
NUM_CLASSES = 2 

# Training hyperparameters
NUM_EPISODES = 200
MAX_T = 5000
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
GAMMA = 0.99
SOFT_UPDATE = 1e-3
BATCH_SIZE = 128   
MEMORY_SIZE = 2048

# Noise parameters
NOISE_SIGMA = 0.2
NOISE_THETA = 0.15
NOISE_DECAY = 0.999
NOISE_SIGMA_MIN = 0.01

# Reward weights
RWD_WEIGHTS = {
    "centering": 100.0,
    "effort": 0.001,
    "stability": 10.0,
    "lost_ball_penalty": 100.0,
    "window_bonus": 100.0
}
