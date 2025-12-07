import cv2
import time
import torch
import os
import serial
import numpy as np
from guiapp.utils.models import get_fasterrcnn_model_single_class as fmodel
from guiapp.utils.ser_con import move_left, move_right, find_esp32, set_command_signal, send_agent_command
import albumentations as A
from albumentations.pytorch import ToTensorV2

def track_control(thread_instance, detected_boxes, ser, W, H, command_interval, agent=None, state=None):
    if ser: 
        current_time = time.time()
        if current_time - thread_instance.last_command_time >= command_interval:
            if getattr(thread_instance, 'agent', None):
                dx = 0.0
                dy = 0.0
                is_detected = 0.0
                
                if detected_boxes:
                    ball_box = detected_boxes[0]['box']
                    ball_center_x = (ball_box[0] + ball_box[2]) / 2
                    ball_center_y = (ball_box[1] + ball_box[3]) / 2
                    
                    dx = (ball_center_x - (W / 2)) / W
                    dy = (ball_center_y - (H / 2)) / H
                    is_detected = 1.0

                prev_action = getattr(thread_instance, 'prev_action', 0.0)
                
                state = np.array([dx, dy, prev_action, is_detected], dtype=np.float32)
                action = thread_instance.agent.choose_action(state)
                
                pan_action = action[0]
                thread_instance.prev_action = pan_action
                
                command = f"P:{pan_action:.2f},T:{0.0:.2f}\n"
                ser.write(command.encode('utf-8'))
                thread_instance.command_log_signal.emit(f"RL Cmd: {command.strip()}")
                
                thread_instance.last_command_time = current_time
                return True


            elif detected_boxes:
                ball_box = detected_boxes[0]['box']
                ball_center_x = (ball_box[0] + ball_box[2]) / 2
                screen_center_x = W / 2
            
                if ball_center_x < screen_center_x - 50:
                    move_left(ser)
                elif ball_center_x > screen_center_x + 50:
                    move_right(ser)
                else:
                    ser.write(b"Stop\n") 
                    thread_instance.command_log_signal.emit("Stop") 
                    print("Sent command: Stop")
                thread_instance.last_command_time = current_time
                return None
        return None

def load_model_from_path(model_path, device, thread_instance=None):
    GLOBAL_CLASS_NAMES = ['__background__', 'Ball']
    num_classes = len(GLOBAL_CLASS_NAMES)
    model = fmodel(num_classes).to(device)
    if model_path and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            if thread_instance:
                thread_instance.command_log_signal.emit(f"Model loaded from {model_path}")
        except Exception as e:
            if thread_instance:
                thread_instance.command_log_signal.emit(f"Error loading model from {model_path}: {e}")
            print(f"Error loading model: {e}")
    else:
        if thread_instance:
             thread_instance.command_log_signal.emit(f"Model file not found at {model_path}. Using base model.")
    
    model.eval()
    return model

def init_video_comp(thread_instance, model_path=None):
    set_command_signal(thread_instance.command_log_signal)

    if not model_path:
        models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        if os.path.exists(models_dir):
             files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
             if files:
                 model_path = os.path.join(models_dir, files[0])
    
    if not model_path:
        legacy_path = r'/Users/Ben/Documents/dever/python/ptorch/models/fin_comb.pth'
        if os.path.exists(legacy_path):
            model_path = legacy_path


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    transform = None
    ser = None

    thread_instance.command_log_signal.emit(f"Using device: {device}")

    try:
        model = load_model_from_path(model_path, device, thread_instance)
    except Exception as e:
        thread_instance.command_log_signal.emit(f"Error initializing model: {e}")
        return False, None, None, None

    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    esp32_port = find_esp32()
    if esp32_port:
        try:
            ser = serial.Serial(esp32_port, 115200, timeout=1)
            thread_instance.command_log_signal.emit(f" Serial connected to {esp32_port}")
        except Exception as e:
            thread_instance.command_log_signal.emit(f" Failed to connect: {e}")
            ser = None
    
    return True, model, transform, ser


def get_ball_detection_external(model, frame, transform, device, confidence_threshold=0.98):
    """Performs ball detection using the PyTorch model and draws the bounding box on the frame."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=frame_rgb)
    image_tensor = transformed['image'].to(device).unsqueeze(0)

    with torch.no_grad():
        predictions = model(image_tensor)

    boxes_raw = predictions[0]['boxes']
    labels_raw = predictions[0]['labels']
    scores_raw = predictions[0]['scores']

    GLOBAL_CLASS_NAMES = ['__background__', 'Ball'] 
    
    keep_indices = torch.where(scores_raw >= confidence_threshold)[0]
    filtered_boxes = boxes_raw[keep_indices].cpu().numpy()
    filtered_labels = labels_raw[keep_indices].cpu().numpy()
    filtered_scores = scores_raw[keep_indices].cpu().numpy()

    original_size = (frame.shape[1], frame.shape[0])
    x_scale = original_size[0] / 640
    y_scale = original_size[1] / 640

    detected_boxes = []
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i]
        label_idx = filtered_labels[i]
        score = filtered_scores[i]

        x_min_orig = int(box[0] * x_scale)
        y_min_orig = int(box[1] * y_scale)
        x_max_orig = int(box[2] * x_scale)
        y_max_orig = int(box[3] * y_scale)
        
        detected_boxes.append({
            'box': (x_min_orig, y_min_orig, x_max_orig, y_max_orig),
            'label': GLOBAL_CLASS_NAMES[label_idx],
            'score': score
        })

        cv2.rectangle(frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0, 0, 255), 2)
        text = f"{GLOBAL_CLASS_NAMES[label_idx]} {score:.2f}"
        cv2.putText(frame, text, (x_min_orig, y_min_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        break

    return detected_boxes, frame

def videorun(thread_instance, cap, W, H, model, transform, device, ser):

    prev_action = np.zeros(1)
    screen_center_x = W / 2
    screen_center_y = H / 2

    while thread_instance._run_flag:
        ret, frame = cap.read()
        if ret:
            frame_to_display = frame.copy()

            thread_instance.mutex.lock()
            is_agent_active = thread_instance.agent_active
            is_inference_active = thread_instance.inference_active
            current_model = getattr(thread_instance, 'model', model)
            agent = getattr(thread_instance, 'agent', None)
            thread_instance.mutex.unlock()

            if is_inference_active and current_model:
                detected_boxes, frame_with_detections = get_ball_detection_external(
                    current_model, frame_to_display, transform, device
                )
                if is_agent_active:
                    is_detected = False
                    ball_x = screen_center_x
                    ball_y = screen_center_y

                    if detected_boxes:
                        box = detected_boxes[0]['box']
                        ball_x = (box[0] + box[2]) / 2
                        ball_y = (box[1] + box[3]) / 2
                        is_detected = True
                    dx = (ball_x - screen_center_x) / W
                    dy = (ball_y - screen_center_y) / H

                    state = np.array([dx, dy, prev_action.item(), float(is_detected)], dtype=np.float32)
                    returned_action = track_control(
                        thread_instance, 
                        detected_boxes, 
                        ser, 
                        W, 
                        H, 
                        command_interval=thread_instance.command_interval, 
                        agent=agent,
                        state=state
                    )
                    
                    if returned_action is not None:
                        prev_action = returned_action

                frame_to_display = frame_with_detections

            qt_image = thread_instance._convert_cv_qt(frame_to_display)
            thread_instance.change_pixmap_signal.emit(qt_image)
        
        time.sleep(0.03)

    cap.release()
    if thread_instance.ser:
        thread_instance.ser.close() 
        print("Serial connection closed.")
    print("VideoThread finished.")