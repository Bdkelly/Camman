import torch
import json
import cv2
from ..utils.jsonreader import extbondbox
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
import os
import shutil
import numpy as np
from ..utils.models import get_fasterrcnn_model_single_class as fmodel # Import your model function

GLOBAL_CLASS_NAMES = ['__background__', 'Ball']
def ballget(image_path,objects):
    boxer = {}
    blst = []
    framename = str(os.path.basename(image_path)).replace(".png","")
    for num,obj in enumerate(objects):
        ball = {"Label":"Ball"+str(num+1),"x_min":obj['box'][0],"y_min":obj['box'][1],"x_max":obj['box'][2],"y_max":obj['box'][3]}
        blst.append(ball)
    boxer[framename] = blst
    print(boxer)
    return boxer
        
def jsonwriter(boxlist,jsonpath):
    if not os.path.exists(jsonpath):
        os.makedirs(jsonpath)
    jsonpath = jsonpath+"jdata.json"
    try:
        with open(jsonpath, 'w') as f:
            json.dump(boxlist, f, indent=4) # Use indent=4 for pretty-printing
    except IOError as e:
        print(f"Error writing to file {jsonpath}: {e}")

def images_to_video(image_folder, video_name="/Users/Ben/Documents/dever/python/ptorch/ball_tracking/mlsSmall/smallmls.mp4", fps=30):
    """
    Converts all images in a folder into a video file.

    Args:
        image_folder (str): The path to the folder containing the images.
        video_name (str): The name of the output video file (e.g., 'output.mp4').
        fps (int): The frames per second for the output video.
    """
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Sorts the images to ensure correct frame order

    if not images:
        print("Error: No images found in the specified folder.")
        return

    # Read the first image to get the video dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read image {first_image_path}")
        return
        
    height, width, layers = frame.shape
    size = (width, height)

    # Define the video codec and create VideoWriter object
    # For MP4 format, 'mp4v' or 'avc1' are common codecs
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(video_name, fourcc, fps, size)

    # Write each image frame to the video
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Skipping {img_path} as it could not be read.")

    video.release()
    print(f"Successfully created video at '{video_name}' with {len(images)} frames.")

def copy_image_to_folder(source_image_path, destination_folder_path="/Users/Ben/Documents/dever/python/ptorch/ball_tracking/mlsSmall"): 
    """
    Copies an image from a source path to a destination folder.

    Args:
        source_image_path (str): The full path to the source image file.
        destination_folder_path (str): The path to the destination folder.
    """
    if not os.path.exists(source_image_path):
        print(f"Error: Source image not found at {source_image_path}")
        return

    # Create the destination folder if it does not exist
    os.makedirs(destination_folder_path, exist_ok=True)

    # Construct the full destination path
    image_name = os.path.basename(source_image_path)
    destination_image_path = os.path.join(destination_folder_path, image_name)

    # Copy the file
    try:
        shutil.copy(source_image_path, destination_image_path)
        print(f"Successfully copied '{image_name}' to '{destination_folder_path}'")
    except Exception as e:
        print(f"Error copying file: {e}")


def run_inference(image_path, video_name, model_path, confidence_threshold=0.98):
    num_classes = len(GLOBAL_CLASS_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = fmodel(num_classes).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    

    model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval() # Set the model to evaluation mode, crucial for inference


    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),  
        ToTensorV2() 
    ])

    # 3. Prepare the Image
    image = Image.open(image_path).convert("RGB")
    original_size = image.size # (width, height)
    
    print(f"DEBUG: Original Image Size: {original_size}") # Debug print

    transformed = transform(image=np.array(image))
    image_tensor = transformed['image'].to(device)

    image_tensor = image_tensor.unsqueeze(0)
    print(f"DEBUG: Model Input Tensor Shape: {image_tensor.shape}") # Debug print

    #Perform Inference
    with torch.no_grad():
        predictions = model(image_tensor)

   
    boxes_raw = predictions[0]['boxes'] 
    labels_raw = predictions[0]['labels']
    scores_raw = predictions[0]['scores']

    print(f"DEBUG: Raw predictions from model (all before confidence filter):") 
    if len(scores_raw) > 0:
        for i in range(len(scores_raw)):
            print(f"  Box: {boxes_raw[i].tolist()}, Label: {labels_raw[i].item()}, Score: {scores_raw[i].item():.4f}")
    else:
        print("No raw predictions found.")

    # Filter detections by confidence threshold
    keep_indices = torch.where(scores_raw >= confidence_threshold)[0]
    
    
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

            # Add more debug prints here:
            print(f"DEBUG: Filtered Box (640x640 coords) before scaling: ({x_min:.4f}, {y_min:.4f}, {x_max:.4f}, {y_max:.4f})")
            print(f"DEBUG: Scaling factors: x_scale={x_scale:.4f}, y_scale={y_scale:.4f}")

            x_min_orig = int(x_min * x_scale)
            y_min_orig = int(y_min * y_scale)
            x_max_orig = int(x_max * x_scale)
            y_max_orig = int(y_max * y_scale)



            print(f"DEBUG: Box after scaling (Original Image Coords): ({x_min_orig}, {y_min_orig}, {x_max_orig}, {y_max_orig})")
            
            # Add a check for degenerate boxes after scaling back to original size
            if x_max_orig <= x_min_orig or y_max_orig <= y_min_orig:
                print(f"WARNING: Degenerate box detected after scaling: ({x_min_orig}, {y_min_orig}, {x_max_orig}, {y_max_orig}). Skipping drawing this box.")
                continue 

            detected_objects.append({
                'box': (x_min_orig, y_min_orig, x_max_orig, y_max_orig),
                'label': GLOBAL_CLASS_NAMES[label_idx],
                'score': score
            })
            copy_image_to_folder(image_path)
    
    print(f"Detected objects in {os.path.basename(image_path)}:")
    print(f"The number of objects: {len(detected_objects)}")
    for obj in detected_objects:
        print(f"  {obj['label']}: Score={obj['score']:.2f}, Box={obj['box']}")
    boxer = {}
    if len(detected_objects) > 0:
        boxer = ballget(image_path,detected_objects)
    draw = ImageDraw.Draw(image)
    try:

        font = ImageFont.truetype("arial.ttf", 20) 
    except IOError:
        print("Warning: arial.ttf not found. Using default font.")
        font = ImageFont.load_default() 
    count = 1
    for obj in detected_objects: # Iterate over detected_objects, which now excludes degenerate boxes
        print("For Ball object"+str(count))
        box = obj['box']
        label = obj['label'] + str(count)
        score = obj['score']
        
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 5, box[1] + 5), f"{label} {score:.2f}", fill="red", font=font)
        count += 1

    output_dir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/"+video_name+"/inf_imgs"
    os.makedirs(output_dir, exist_ok=True)
    if detected_objects > 0 :
        output_image_path = os.path.join(output_dir, f"inferred_{os.path.basename(image_path)}")
        image.save(output_image_path)
        print(f"Inferred image saved to: {output_image_path}")
    return boxer,output_dir

def testhelper():
    bondbox = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/video1/jdata/video1p.json"
    imgdir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/video1/imgs"
    jsondata = extbondbox(bondbox)
    image_list = []
    for frame in jsondata:
        img_filen = f"frame_{frame:05d}.jpg"
        imgf = f"{imgdir}/{img_filen}"
        if os.path.exists(imgf):
            image_list.append(img_filen)
        else:
            print("No:",imgf)
    return image_list


if __name__ == "__main__":
    model_path = "/Users/Ben/Documents/dever/python/ptorch/ball_tracking/infer/dipmodel.pth"
    video_name = "dips/p10"
    unlabeled_image_dir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/"+video_name+"/imgs/"
    if not os.path.exists(unlabeled_image_dir):
        print(f"Please create a directory named '{unlabeled_image_dir}' and place some images there.")
    box_list = []
    img_list = os.listdir(unlabeled_image_dir)
    for image in img_list:
        image_path = os.path.join(unlabeled_image_dir, image)
        box,outd = run_inference(image_path,video_name,model_path)
        if box != {}:
            box_list.append(box)
    #images_to_video("/mlsvideoS")
    jsondir = (f"{outd}/jdata/")
    jsonwriter(box_list,jsondir)
    '''
    img_list = testhelper()
    for image_file in img_list:
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            image_path = os.path.join(unlabeled_image_dir, image_file)
            print(image_path)
            print(f"\n--- Processing: {image_file} ---")
            box,outd = run_inference(image_path,video_name,model_path)
            if box != {}:
                box_list.append(box)
    jsondir = (f"{outd}/jdata/")
    jsonwriter(box_list,jsondir)
    '''

