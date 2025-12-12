import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import torch
from PyQt5.QtGui import QImage
from guiapp.threads.video_threads  import VideoThread

@pytest.fixture
def mock_video_thread(qtbot):
    thread = VideoThread()
    qtbot.addWidget(thread)
    return thread

def test_initialization(mock_video_thread):
    assert mock_video_thread._run_flag is True
    assert mock_video_thread.inference_active is False
    assert mock_video_thread.agent_active is False
    assert mock_video_thread.command_interval == 1.0

def test_run_success(mock_video_thread):
    mock_model = Mock()
    mock_transform = Mock()
    mock_ser = Mock()
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.get.return_value = 640

    with patch('video_thread.init_video_comp', return_value=(True, mock_model, mock_transform, mock_ser)), \
         patch('cv2.VideoCapture', return_value=mock_cap), \
         patch('video_thread.RLAgent') as mock_rl_agent, \
         patch('os.path.exists', return_value=True), \
         patch('torch.load'), \
         patch('video_thread.videorun') as mock_videorun:
        
        mock_video_thread.run()

        assert mock_video_thread.model == mock_model
        assert mock_video_thread.ser == mock_ser
        mock_rl_agent.assert_called()
        mock_videorun.assert_called_once()

def test_run_initialization_failed(mock_video_thread):
    with patch('video_thread.init_video_comp', return_value=(False, None, None, None)), \
         patch('video_thread.videorun') as mock_videorun:
        
        with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
            mock_video_thread.run()
            
            assert mock_video_thread._run_flag is False
            mock_signal.assert_called_with("Initialization failed.")
            mock_videorun.assert_not_called()

def test_run_camera_open_failed(mock_video_thread):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch('video_thread.init_video_comp', return_value=(True, Mock(), Mock(), Mock())), \
         patch('cv2.VideoCapture', return_value=mock_cap), \
         patch('video_thread.videorun') as mock_videorun:

        with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
            mock_video_thread.run()
            
            mock_signal.assert_called_with("Error: Could not open video stream.")
            assert mock_video_thread._run_flag is False
            mock_videorun.assert_not_called()

def test_update_model_success(mock_video_thread):
    new_model = Mock()
    
    with patch('video_thread.load_model_from_path', return_value=new_model):
        with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
            
            mock_video_thread.update_model("new_model.pth")
            
            assert mock_video_thread.model == new_model
            mock_signal.assert_called_with("Model updated successfully.")

def test_update_model_failure(mock_video_thread):
    with patch('video_thread.load_model_from_path', return_value=None):
        with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
            
            mock_video_thread.update_model("bad_path.pth")
            
            assert mock_video_thread.model is None 
            mock_signal.assert_called_with("Failed to load new model.")

def test_toggle_inference(mock_video_thread):
    with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
        mock_video_thread.toggle_inference(True)
        assert mock_video_thread.inference_active is True
        mock_signal.assert_called_with("--- Inference STARTED ---")
        
        mock_video_thread.toggle_inference(False)
        assert mock_video_thread.inference_active is False
        mock_signal.assert_called_with("--- Inference STOPPED ---")

def test_toggle_agent(mock_video_thread):
    with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
        mock_video_thread.toggle_agent(True)
        assert mock_video_thread.agent_active is True
        mock_signal.assert_called_with("--- CamMan Agent STARTED ---")

        mock_video_thread.toggle_agent(False)
        assert mock_video_thread.agent_active is False
        mock_signal.assert_called_with("--- CamMan Agent STOPPED ---")

def test_set_command_interval(mock_video_thread):
    with patch.object(mock_video_thread.command_log_signal, 'emit') as mock_signal:
        mock_video_thread.set_command_interval(2.5)
        
        assert mock_video_thread.command_interval == 2.5
        mock_signal.assert_called_with("Command Interval set to: 2.50s")

def test_convert_cv_qt(mock_video_thread):
    cv_img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    q_image = mock_video_thread._convert_cv_qt(cv_img)
    
    assert isinstance(q_image, QImage)
    assert q_image.height() == 480 
    assert q_image.width() == 480

def test_stop(mock_video_thread):
    mock_video_thread._run_flag = True
    
    mock_video_thread.wait = Mock()
    
    mock_video_thread.stop()
    
    assert mock_video_thread._run_flag is False
    mock_video_thread.wait.assert_called_once()