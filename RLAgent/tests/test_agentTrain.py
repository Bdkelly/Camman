import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from RLAgent.agentTrain import train_agent, vidget

@pytest.fixture
def mock_dependencies(mocker):
    mocker.patch('RLAgent.agentTrain.cv2')
    mocker.patch('RLAgent.agentTrain.vidget', return_value=(Mock(), 640, 480))
    mocker.patch('RLAgent.agentTrain.get_fasterrcnn_model_single_class', return_value=Mock(to=Mock(return_value=Mock())))
    mocker.patch('RLAgent.agentTrain.RewardSystem')
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('torch.load')
    mocker.patch('torch.save')

    mock_agent = mocker.patch('RLAgent.agentTrain.RLAgent').return_value
    mock_agent.choose_action.return_value = np.array([0.5], dtype=np.float32)

    mock_noise = mocker.patch('RLAgent.agentTrain.OUNoise').return_value
    mock_noise.sample.return_value = np.array([0.1], dtype=np.float32)
    mock_noise.sigma = 0.2

    mock_env_instance = Mock()
    mock_env_instance.reset.return_value = (np.array([0.1, 0.1, 0, 1], dtype=np.float32), np.zeros((480, 640, 3)))
    mock_env_instance.step.return_value = (np.array([0.0, 0.0, 0, 1], dtype=np.float32), 1.0, True, np.array([0.5]), np.zeros((480, 640, 3)))
    mocker.patch('RLAgent.agentTrain.CameraControlEnv', return_value=mock_env_instance)
    
    return mock_env_instance