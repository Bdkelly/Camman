import torch
import cv2

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']
def get_ball_detection(model, frame, transform, device, confidence_threshold=0.96):


    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=frame_rgb)
    image_tensor = transformed['image'].to(device).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    boxes_raw = predictions[0]['boxes']
    labels_raw = predictions[0]['labels']
    scores_raw = predictions[0]['scores']

    keep_indices = torch.where(scores_raw >= confidence_threshold)[0]
    filtered_boxes = boxes_raw[keep_indices].cpu().numpy()
    filtered_labels = labels_raw[keep_indices].cpu().numpy()
    filtered_scores = scores_raw[keep_indices].cpu().numpy()

    # Get original image dimensions from the frame
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