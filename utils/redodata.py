import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .cleanCall import cleanCall
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
import numpy as np


class objectdata(Dataset):
    def __init__(self, images_dir, jsondata, class_names, transform):
        self.images_dir = images_dir
        self.jsondata = jsondata
        self.class_names = ['__background__'] + class_names
        self.label_to_id = {name: i for i, name in enumerate(self.class_names)}
        
        if not os.path.exists(self.images_dir):
            raise FileNotFoundError(f"No image files found in {self.images_dir}. Cannot determine original image size.")

        if transform is None:
            self.transform = A.Compose([
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.jsondata)
    
    def __getitem__(self, idx):
        frame = self.jsondata[idx]
        fimg = frame["frame"]
        img_path = f"{self.images_dir}/{fimg}"

        boxes_for_frame = []
        labels_for_frame = []

        if not os.path.exists(img_path):
            print(f"Error: Image file not found for {img_path}. Returning dummy data.")
            image = Image.new('RGB', (self.oVideo_width, self.oVideo_height), (0,0,0)) 
        else:
            image = Image.open(img_path).convert("RGB")
            img_width, img_height = image.size 
        
        xmin_pix = frame['x_min']
        ymin_pix = frame['y_min']
        xmax_pix = frame['x_max']
        ymax_pix = frame['y_max']
        label_name = "Ball"
        
        label_id = self.label_to_id.get(label_name)

        # Normalize coordinates
        xmin_norm = xmin_pix / img_width
        ymin_norm = ymin_pix / img_height
        xmax_norm = xmax_pix / img_width
        ymax_norm = ymax_pix / img_height
        
        # Clip coordinates to the [0.0, 1.0] range
        xmin_norm = max(0.0, xmin_norm)
        ymin_norm = max(0.0, ymin_norm)
        xmax_norm = min(1.0, xmax_norm)
        ymax_norm = min(1.0, ymax_norm)

        boxes_for_frame.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm])
        labels_for_frame.append(label_id)

        boxes_np = np.array(boxes_for_frame, dtype=np.float32)
        labels_np = np.array(labels_for_frame, dtype=np.int32)

        image_np = np.array(image)

        if self.transform:
            transformed = self.transform(
                image=image_np, 
                bboxes=boxes_np, 
                labels=labels_np 
            )
            image = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                 target_boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                 target_labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:
                 target_boxes = torch.empty((0, 4), dtype=torch.float32)
                 target_labels = torch.empty((0,), dtype=torch.int64)
        else:
            default_transform_pipeline = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
            transformed = default_transform_pipeline(
                image=image_np, 
                bboxes=boxes_np, 
                labels=labels_np
            )
            image = transformed['image']
            
            if len(transformed['bboxes']) > 0:
                 target_boxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
                 target_labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)
            else:
                 target_boxes = torch.empty((0, 4), dtype=torch.float32)
                 target_labels = torch.empty((0,), dtype=torch.int64)

        target = {'boxes': target_boxes, 'labels': target_labels}
        return image, target
    
if __name__ == "__main__":
    bondbox = "/Users/Ben/Documents/dever/python/ptorch/ball_tracking/inferred_images/mlsvideo/jdata/edited_jdata_100.json"
    imgdir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/mlsvideo/imgs"

    box,width,height = cleanCall(bondbox,imgdir)
    class_name = ['Ball'] 
    
    transform_pipeline = A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    dataset = objectdata(imgdir,box,class_name,transform_pipeline)
    dataloader = DataLoader(
        dataset, 
        batch_size=2,
        shuffle=True, 
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))
    )