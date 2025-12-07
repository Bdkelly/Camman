import torch
from jsonreader import extbondbox
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np

class objectdata(Dataset):
    def __init__(self, images_base_dir, bbox, class_names, transform=None):
        self.images_base_dir = images_base_dir
        self.annotations_per_frame = bbox # Directly use the pre-parsed dict
        self.class_names = ['__background__'] + class_names # Ensure background class is at index 0
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}

        # Step 1: Get original video dimensions from an example image
        # This is CRUCIAL for de-normalizing bounding box coordinates from Label Studio
        # Assuming all extracted images have the same dimensions and that there's at least one image.
        image_files_in_dir = sorted([f for f in os.listdir(self.images_base_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if not image_files_in_dir:
            raise FileNotFoundError(f"No image files found in {self.images_base_dir}. Cannot determine original image size.")
        
        first_image_path = os.path.join(self.images_base_dir, image_files_in_dir[0])
        with Image.open(first_image_path) as img:
            self.original_video_width, self.original_video_height = img.size
        print(f"DEBUG: Detected original image resolution: {self.original_video_width}x{self.original_video_height}")

        # Get the list of all frame numbers that have annotations
        # These are the frame numbers we will iterate over
        self.annotated_frame_numbers = sorted(list(self.annotations_per_frame.keys()))
        
        if transform is None:
            self.transform = A.Compose([
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform

    def __len__(self):
        # The length of the dataset is the number of frames that have annotations
        return len(self.annotated_frame_numbers)
    
    def __getitem__(self, idx):
        # Get the original Label Studio frame number for this index
        frame_number_to_load = self.annotated_frame_numbers[idx]
        
        # Retrieve all bounding box annotations for this frame from the pre-parsed dictionary
        # This will be a list of dictionaries, e.g., [{'x':.., 'y':.., 'width':.., 'height':.., 'label_name':..}, ...]
        frame_annotations_raw = self.annotations_per_frame.get(frame_number_to_load, [])

        # Construct image filename using the Label Studio frame number directly
        # Assuming your extracted image files are named like 'frame_00XXX.jpg' where XXX matches Label Studio's frame number
        image_filename = f"frame_{frame_number_to_load:05d}.jpg" 
        img_path = os.path.join(self.images_base_dir, image_filename)
        
        # Initialize empty lists for boxes and labels to handle frames with no valid annotations
        boxes_for_frame = []
        labels_for_frame = []

        # Handle cases where the image file might genuinely be missing
        if not os.path.exists(img_path):
            print(f"Error: Image file not found for {img_path}. Returning dummy data.")
            # Create a black dummy image matching the original video's dimensions
            image = Image.new('RGB', (self.original_video_width, self.original_video_height), (0,0,0)) 
            # Leave boxes_for_frame and labels_for_frame empty
        else:
            image = Image.open(img_path).convert("RGB")
            
        # Process each raw annotation for the current frame
        for ann_raw in frame_annotations_raw:
            x_norm = ann_raw.get("x")
            y_norm = ann_raw.get("y")
            width_norm = ann_raw.get("width")
            height_norm = ann_raw.get("height")
            label_name = ann_raw.get("label_name")
            print(width_norm,height_norm)

            if any(val is None for val in [x_norm, y_norm, width_norm, height_norm]) or label_name is None:
                print(f"Warning: Incomplete annotation data for frame {frame_number_to_load}. Skipping this annotation: {ann_raw}")
                continue

            # Convert normalized (0-100%) JSON coordinates to absolute pixel coordinates
            # This is crucial as Label Studio typically exports 0-100% relative to the original image size
            xmin = (x_norm / 100.0) * self.original_video_width
            ymin = (y_norm / 100.0) * self.original_video_height
            xmax = ((x_norm + width_norm) / 100.0) * self.original_video_width
            ymax = ((y_norm + height_norm) / 100.0) * self.original_video_height

            # Sanity check for degenerate boxes (width/height <= 0) and clip to image bounds
            # A small epsilon prevents float precision issues from creating invalid boxes
            if (xmax - xmin) <= 1e-6 or (ymax - ymin) <= 1e-6:
                print(f"Warning: Degenerate bbox in frame {frame_number_to_load} after denormalization. Skipping. Coords: [{xmin:.2f}, {ymin:.2f}, {xmax:.2f}, {ymax:.2f}]")
                continue
            
            # Clip coordinates to actual image bounds to prevent errors from out-of-bounds boxes
            xmin = max(0.0, xmin)
            ymin = max(0.0, ymin)
            xmax = min(float(self.original_video_width), xmax)
            ymax = min(float(self.original_video_height), ymax)

            # Map label name to integer ID
            label_id = self.label_to_id.get(label_name)
            if label_id is None:
                print(f"Warning: Label '{label_name}' for frame {frame_number_to_load} not found in class_names. Skipping this annotation.")
                continue

            boxes_for_frame.append([xmin, ymin, xmax, ymax])
            labels_for_frame.append(label_id)

        # Convert lists to PyTorch tensors
        boxes_tensor = torch.as_tensor(boxes_for_frame, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels_for_frame, dtype=torch.int64)

        target = {'boxes': boxes_tensor, 'labels': labels_tensor}
        
        # Apply transformations (resize, normalize, ToTensorV2)
        # Albumentations expects image as a NumPy array for its transforms
        image_np = np.array(image)

        if self.transform:
            transformed = self.transform(
                image=image_np, 
                bboxes=target['boxes'].cpu().numpy(), # Albumentations expects NumPy for bboxes
                labels=target['labels'].cpu().numpy() # Albumentations expects NumPy for labels
            )
            image = transformed['image'] # Transformed image (now a PyTorch tensor)
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)
        else:
            # If no transform was explicitly provided, apply a basic default to ensure consistency
            # (ToTensorV2 and Normalize are usually essential for model input)
            default_transform_pipeline = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            transformed = default_transform_pipeline(
                image=image_np, 
                bboxes=target['boxes'].cpu().numpy(), 
                labels=target['labels'].cpu().numpy()
            )
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            target['labels'] = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        return image, target
        
        

if __name__ == "__main__":
    # IMPORTANT: Adjust these paths to your actual setup
    # This should be the directory containing your extracted 'frame_00XXX.jpg' images
    IMAGES_BASE_DIR = "/Users/Ben/Documents/dever/python/data/outframes/video1/imgs"
    parsed_json_annotations = extbondbox("/Users/Ben/Documents/dever/python/data/outframes/video1/jdata/video1p.json")
    
    # These class names MUST match the 'labels' used in your Label Studio annotations exactly.
    # Exclude '__background__', as it's added automatically by the dataset class.
    CLASS_NAMES_IN_PROJECT = ['Ball'] 

    # 1. Use extbondbox to parse the JSON and get the annotations dictionary

    if not parsed_json_annotations:
        print("Exiting: No annotations parsed from JSON file.")
        exit()

    # 2. Define your Albumentations transform pipeline for training.
    # This should match what you use in train.py exactly (including Resize, Normalize).
    training_transform_pipeline = A.Compose([
        A.Resize(640, 640), # Resize images to model input size
        A.HorizontalFlip(p=0.5), # Example augmentation
        A.RandomBrightnessContrast(p=0.2), # Example augmentation
        A.Normalize(mean=(0.485, 0.456, 0.406), # Standard ImageNet means
                    std=(0.229, 0.224, 0.225)),  # Standard ImageNet stds
        ToTensorV2() # Converts image to PyTorch tensor (HWC to CHW)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'])) # Crucial for bounding box transforms

    # 3. Create the dataset instance, passing the parsed annotations
    if os.path.exists(IMAGES_BASE_DIR):
        try:
            dataset = objectdata(
                images_base_dir=IMAGES_BASE_DIR, 
                bbox=parsed_json_annotations, # Pass the parsed dict here
                class_names=CLASS_NAMES_IN_PROJECT, # <--- THIS IS THE MISSING ARGUMENT YOU NEEDED TO ADD
                transform=training_transform_pipeline # Pass your training transforms
            )
            print(dataset[0])
            # 4. Create the DataLoader
            # num_workers > 0 requires the main script to be under if __name__ == "__main__" on Windows
            # For testing, 0 is fine. For actual training performance, use a higher number (e.g., 4 or os.cpu_count() - 1).
            dataloader = DataLoader(
                dataset, 
                batch_size=2, # Use a small batch size for testing
                shuffle=True, 
                num_workers=0, # Set to 0 for initial debugging to avoid multiprocessing issues
                collate_fn=lambda x: tuple(zip(*x)) # Custom collate_fn for object detection
            ) 
        except Exception as e:
            print(f"An error occurred during dataset/dataloader initialization or iteration: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for deeper debugging
            '''

            print(f"Dataset has {len(dataset)} annotated frames prepared for loading.")
            
            # 5. Iterate through a few batches to verify the output
            print("\n--- Verifying DataLoader output ---")
            for i, (images, targets) in enumerate(dataloader):
                print(f"\nBatch {i+1}:")
                print(f"  Number of images in batch: {len(images)}")
                print(f"  Number of targets in batch: {len(targets)}")

                if len(images) > 0:
                    print(f"  First image tensor shape: {images[0].shape}")
                
                if len(targets) > 0:
                    for j, target_dict in enumerate(targets):
                        print(f"  Target {j+1} (for image {j+1}) keys: {target_dict.keys()}")
                        if 'boxes' in target_dict and target_dict['boxes'].numel() > 0:
                            print(f"    Boxes shape: {target_dict['boxes'].shape}, Labels shape: {target_dict['labels'].shape}")
                            # Print first box's coordinates to check scaling and format
                            print(f"    First box content (pixel coords, after transforms): {target_dict['boxes'][0].tolist()}")
                            print(f"    First label content: {target_dict['labels'][0].item()} (raw ID)")
                        else:
                            print(f"    No bounding boxes found for this image (after filtering/processing).")

                if i >= 2: # Process 3 batches for testing
                    break

        except Exception as e:
            print(f"An error occurred during dataset/dataloader initialization or iteration: {e}")
            import traceback
            traceback.print_exc() # Print full traceback for deeper debugging
    else:
        print(f"Error: Images directory does not exist at {IMAGES_BASE_DIR}. Please check the path.")
        '''