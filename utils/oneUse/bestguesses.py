import torch
import json
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from models import get_fasterrcnn_model_single_class as fmodel # Import your model function

# Define class names globally or pass them. Must match training order.
# Background is usually index 0, your first class is index 1.
GLOBAL_CLASS_NAMES = ['__background__', 'Ball'] #

def ballget(image_path, objects):
    boxer = {}
    blst = []
    framename = str(os.path.basename(image_path)).replace(".png", "") # Assuming .png, adjust if .jpg
    for num, obj in enumerate(objects):
        ball = {"Label": "Ball" + str(num + 1), "x_min": obj['box'][0], "y_min": obj['box'][1], "x_max": obj['box'][2], "y_max": obj['box'][3]}
        blst.append(ball)
    boxer[framename] = blst
    print(boxer)
    return boxer
        
def jsonwriter(data_list_for_json, jsonpath, batch_num=None):
    """
    Writes detection data to a JSON file.

    Args:
        data_list_for_json (list): A list of dictionaries, where each dict represents a frame's detections.
                                   Example: [{'frame_1': [...]}, {'frame_2': [...]}]
        jsonpath (str): The directory where the JSON file should be saved.
        batch_num (int, optional): An identifier for the batch number, used in the filename.
    """
    if not os.path.exists(jsonpath):
        os.makedirs(jsonpath)
        print(f"Created directory: {jsonpath}")

    filename = f"jdata_batch_{batch_num}.json" if batch_num is not None else "jdata_final.json"
    output_filepath = os.path.join(jsonpath, filename)

    try:
        with open(output_filepath, 'w') as f:
            json.dump(data_list_for_json, f, indent=4) # Use indent=4 for pretty-printing
        print(f"-----------------------Data successfully written to {output_filepath}")
    except IOError as e:
        print(f"Error writing to file {output_filepath}: {e}")

def run_inference(image_path, video_name, model_path='checkpoint_epoch_400.pth', confidence_threshold=0.89):
    """
    Performs inference on a single image using the trained model.

    Args:
        image_path (str): Path to the input image.
        video_name (str): Name of the video (currently unused in core logic but passed).
        model_path (str): Path to the saved .pth model file.
        confidence_threshold (float): Minimum confidence score to display a detection.
    """
    # 1. Load the Model
    num_classes = len(GLOBAL_CLASS_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = fmodel(num_classes).to(device) # Assuming fmodel is defined in models.py
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return {}, "" # Return empty dict and empty string for output_dir
    
    # --- FIX: Load the entire checkpoint and then extract model_state_dict ---
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    except KeyError:
        # If 'model_state_dict' key is missing, it implies an older save format
        # where model.state_dict() was saved directly.
        print(f"Warning: '{model_path}' does not contain 'model_state_dict' key. Attempting direct load.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        print("Please ensure the model file is valid and corresponds to the FasterRCNN architecture.")
        return {}, ""
    # --- END FIX ---
    
    model.eval() # Set the model to evaluation mode, crucial for inference

    # 2. Define Transformations for Inference (matching training as closely as possible)
    transform = A.Compose([
        A.Resize(640, 640), # Resize to the input size the model expects
        A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means
                    std=(0.229, 0.224, 0.225)),  # Standard ImageNet stds
        ToTensorV2() # Converts image to PyTorch tensor (HWC to CHW)
    ])

    # 3. Prepare the Image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)
    
    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].to(device)

    # Add a batch dimension (B, C, H, W) for the model input
    image_tensor = image_tensor.unsqueeze(0)

    # 4. Perform Inference
    with torch.no_grad(): # Disable gradient calculations for inference
        predictions = model(image_tensor)

    # 5. Process Predictions
    boxes_raw = predictions[0]['boxes'] # Keep as tensor initially
    labels_raw = predictions[0]['labels']
    scores_raw = predictions[0]['scores']

    # Filter detections by confidence threshold
    keep_indices = torch.where(scores_raw >= confidence_threshold)[0]
    
    # Convert to numpy *after* filtering
    filtered_boxes = boxes_raw[keep_indices].cpu().numpy()
    filtered_labels = labels_raw[keep_indices].cpu().numpy()
    filtered_scores = scores_raw[keep_indices].cpu().numpy()

    detected_objects = []
    if len(filtered_boxes) > 0: # Only proceed if detections exist after filtering
        for i in range(len(filtered_boxes)):
            box = filtered_boxes[i]
            label_idx = filtered_labels[i]
            score = filtered_scores[i]

            x_scale = original_size[0] / 640
            y_scale = original_size[1] / 640

            x_min, y_min, x_max, y_max = box
            
            x_min_orig = int(x_min * x_scale)
            y_min_orig = int(y_min * y_scale)
            x_max_orig = int(x_max * x_scale)
            y_max_orig = int(y_max * y_scale)
            
            # Add a check for degenerate boxes after scaling back to original size
            if x_max_orig <= x_min_orig or y_max_orig <= y_min_orig:
                print(f"WARNING: Degenerate box detected after scaling: ({x_min_orig}, {y_min_orig}, {x_max_orig}, {y_max_orig}). Skipping drawing this box.")
                continue # Skip adding this degenerate box to detected_objects

            detected_objects.append({
                'box': (x_min_orig, y_min_orig, x_max_orig, y_max_orig),
                'label': GLOBAL_CLASS_NAMES[label_idx],
                'score': score
            })
    
    print(f"Detected objects in {os.path.basename(image_path)}:")
    print(f"The number of objects: {len(detected_objects)}")
    for obj in detected_objects:
        print(f"  {obj['label']}: Score={obj['score']:.2f}, Box={obj['box']}")
    
    boxer = {}
    if len(detected_objects) > 0: # If any objects were detected
        boxer = ballget(image_path, detected_objects)

        # 6. Visualize Results (Optional) - Only if objects were detected
        draw = ImageDraw.Draw(image)
        try:
            # You might need to specify a full path like "C:/Windows/Fonts/arial.ttf"
            font = ImageFont.truetype("arial.ttf", 20) 
        except IOError:
            print("Warning: arial.ttf not found. Using default font.")
            font = ImageFont.load_default() # Fallback to default font
        
        count = 1
        for obj in detected_objects: # Iterate over detected_objects, which now excludes degenerate boxes
            box = obj['box']
            label = obj['label'] + str(count)
            score = obj['score']
            
            draw.rectangle(box, outline="red", width=3)
            draw.text((box[0] + 5, box[1] + 5), f"{label} {score:.2f}", fill="red", font=font)
            count += 1

        output_dir = "inferred_images/"+video_name
        output_img_dir = output_dir+"/imgs/"
        os.makedirs(output_img_dir, exist_ok=True) # Ensure the /imgs/ subdirectory exists
        output_image_name = f"inferred_{os.path.basename(image_path)}"
        image.save(os.path.join(output_img_dir, output_image_name)) # Use os.path.join here for robustness
        print(f"Inferred image saved to: {output_image_name}")
    else:
        # If no objects detected, print a message but don't save the image
        print(f"No balls detected for {os.path.basename(image_path)}. Not saving inferred image.")
        output_dir = "inferred_images/"+video_name # Still return the base output_dir for JSON pathing
    
    return boxer, output_dir


if __name__ == "__main__":
    # Example usage:
    # Make sure to place some unlabeled images in an 'unlabeled_images' directory
    video_name = "/mlsvideo"
    unlabeled_image_dir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes" + video_name + "/imgs/" #
    
    if not os.path.exists(unlabeled_image_dir):
        print(f"Please create a directory named '{unlabeled_image_dir}' and place some images there.")
        exit()

    # --- Batch Saving Configuration ---
    batch_save_interval = 1000  # Save every 1000 frames with detected balls
    current_batch_data = []     # Accumulates data for the current batch
    detected_frames_count = 0   # Counts frames where balls were detected
    batch_counter = 0           # Tracks how many batches have been saved

    # Get the output directory for inferred images (and thus for JSON)
    # This might fail if the first image processed has no detections or if unlabeled_image_dir is empty.
    
    # --- Handle potential empty directory or no images for first run ---
    image_files_in_dir = sorted([f for f in os.listdir(unlabeled_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])
    if not image_files_in_dir:
        print(f"Error: No image files found in '{unlabeled_image_dir}'. Cannot run inference.")
        exit()
    
    # Process the first image to get a base output_dir for JSON, assuming it contains at least one image.
    first_image_path_for_setup = os.path.join(unlabeled_image_dir, image_files_in_dir[0])
    
    # Use a dummy confidence threshold for this initial setup run, to ensure it doesn't filter out too much
    # and provides an output_dir. The actual confidence_threshold for real inference will be used in the loop.
    # It's important that 'trained_model_final.pth' exists for this part.
    _, json_output_base_dir_setup = run_inference(
        first_image_path_for_setup,
        video_name,
        confidence_threshold=0.01 # A very low threshold to ensure it runs
    )
    
    jsondir = os.path.join(json_output_base_dir_setup, "jdata/")
    
    # Ensure the JSON output directory exists
    os.makedirs(jsondir, exist_ok=True)


    # Sort image files to ensure consistent processing order
    image_files = sorted([f for f in os.listdir(unlabeled_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))])

    # Use the actual confidence threshold for the main loop
    main_confidence_threshold = 0.89 # As defined in run_inference default parameter

    for image_file in image_files:
        image_path = os.path.join(unlabeled_image_dir, image_file)
        print(f"\n--- Processing: {image_file} ---")
        
        # Run inference and get the detected box data for this frame
        # Pass the main_confidence_threshold here.
        box, _ = run_inference(image_path, video_name, confidence_threshold=main_confidence_threshold) 

        # Check if any balls were detected in this frame
        if box: # if box is not an empty dictionary (meaning detections were found and processed by ballget)
            current_batch_data.append(box)
            detected_frames_count += 1

            # Check if it's time to save a batch
            if detected_frames_count % batch_save_interval == 0:
                batch_counter += 1
                print(f"--- Saving batch {batch_counter} (after {detected_frames_count} frames with detections) ---")
                jsonwriter(current_batch_data, jsondir, batch_num=batch_counter)
                current_batch_data = [] # Reset for the next batch

    # After the loop, save any remaining data that didn't form a full batch
    if current_batch_data:
        batch_counter += 1
        print(f"--- Saving final batch {batch_counter} (remaining {len(current_batch_data)} frames) ---")
        jsonwriter(current_batch_data, jsondir, batch_num=batch_counter)

    print("\nAll images processed. JSON generation complete.")