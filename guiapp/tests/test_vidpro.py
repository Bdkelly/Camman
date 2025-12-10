import pytest
from unittest.mock import Mock, patch
import torch
import numpy as np
import cv2
import os
from guiapp.utils.vidpro import videorun, track_control, load_model_from_path, init_video_comp, get_ball_detection_external

@pytest.fixture
def mock_thread_instance():
    instance = Mock()
    instance._run_flag = True
    instance.agent_active = True
    instance.inference_active = True
    instance.mutex = Mock()
    instance.last_command_time = 0
    instance.command_log_signal = Mock()
    instance.change_pixmap_signal = Mock()
    return instance

@pytest.fixture
def mock_cap():
    cap = Mock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cap.read.return_value = (True, frame)
    cap.release = Mock()
    return cap

@pytest.fixture
def mock_model():
    model = Mock()
    output = [{
        'boxes': torch.tensor([[100, 100, 200, 200]], dtype=torch.float32),
        'labels': torch.tensor([1]),
        'scores': torch.tensor([0.99])
    }]
    model.return_value = output
    model.eval = Mock()
    model.to = Mock(return_value=model)
    return model

@pytest.fixture
def mock_transform():
    transform = Mock()
    transform.return_value = {'image': torch.zeros((3, 640, 640))}
    return transform

def test_load_model_from_path_success(mock_thread_instance):
    with patch('guiapp.utils.vidpro.fmodel') as mock_fmodel, \
         patch('torch.load') as mock_torch_load, \
         patch('os.path.exists', return_value=True):
        
        mock_model = Mock()
        mock_fmodel.return_value = mock_model
        mock_model.to.return_value = mock_model
        
        device = torch.device('cpu')
        model = load_model_from_path('dummy_path.pth', device, mock_thread_instance)
        
        assert model == mock_model
        mock_model.load_state_dict.assert_called_once()
        mock_thread_instance.command_log_signal.emit.assert_called()

def test_load_model_from_path_failure(mock_thread_instance):
    with patch('guiapp.utils.vidpro.fmodel') as mock_fmodel, \
         patch('os.path.exists', return_value=False):
        
        mock_model = Mock()
        mock_fmodel.return_value = mock_model
        mock_model.to.return_value = mock_model

        device = torch.device('cpu')
        model = load_model_from_path('dummy_path.pth', device, mock_thread_instance)

        assert model == mock_model

        mock_thread_instance.command_log_signal.emit.assert_called_with("Model file not found at dummy_path.pth. Using base model.")

def test_track_control_move_left(mock_thread_instance):
    mock_ser = Mock()
    W, H = 640, 480
    command_interval = 0.5
    detected_boxes = [{'box': (0, 100, 100, 200), 'label': 'Ball', 'score': 0.99}]

    with patch('guiapp.utils.vidpro.move_left') as mock_move_left:
        mock_thread_instance.last_command_time = 0
        result = track_control(mock_thread_instance, detected_boxes, mock_ser, W, H, command_interval)
        assert result is True
        mock_move_left.assert_called_once_with(mock_ser)

        assert result is True
        mock_ser.write.assert_called_with(b"Stop\n")

def test_get_ball_detection_external(mock_model, mock_transform):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    device = torch.device('cpu')
    
    detected_boxes, frame_out = get_ball_detection_external(mock_model, frame, mock_transform, device)
    assert len(detected_boxes) == 1
    assert detected_boxes[0]['label'] == 'Ball'
    assert detected_boxes[0]['score'] == 0.99
    assert isinstance(frame_out, np.ndarray)
