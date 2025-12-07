import torch
import os
from ..utils.jsonreader import extbondbox, infbondbox
from ..utils.cleanCall import cleanCall
import torch.optim as optim
from torch.utils.data import DataLoader
from ..utils.redodata import objectdata
from ..utils.models import get_fasterrcnn_model_single_class as fmodel
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms 
from torch.optim.lr_scheduler import StepLR, LinearLR, SequentialLR

def collate_fn(batch):
    return tuple(zip(*batch))

def trainer(jsondata,imgdir,num_epochs):
    class_names = ['Ball']  
    num_classes = len(class_names) + 1 

    # Define your transformations
    transform = A.Compose([
        A.Resize(640, 640),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
    
    # Create datasets and dataloaders
    train_dataset = objectdata(imgdir,jsondata, class_names, transform=transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}") # Print which device is being used
    model = fmodel(num_classes).to(device)

    # Define the optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    warmup_epochs = 5
    warmup_factor = 1.0 / warmup_epochs
    lr_scheduler_warmup = LinearLR(optimizer, start_factor=warmup_factor, total_iters=warmup_epochs)

    
    lr_scheduler_step = StepLR(optimizer, step_size=10, gamma=0.1)

    
    lr_scheduler = SequentialLR(optimizer, schedulers=[lr_scheduler_warmup, lr_scheduler_step], milestones=[warmup_epochs])


    # Training loop
    for epoch in range(num_epochs):
        model.train() # Set model to training mode
        epoch_loss = 0.0

        # Use enumerate to get batch_idx for TensorBoard logging
        for batch_idx, (images, targets) in enumerate(train_dataloader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Zero the gradients before each batch
            optimizer.zero_grad()

            # Forward pass: model returns a dict of losses in training mode
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Backward pass: compute gradients
            losses.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) 

            # Optimizer step: update model parameters
            optimizer.step()

            # Accumulate total loss for the epoch
            epoch_loss += losses.item() 

          
        lr_scheduler.step()

       

        avg_epoch_loss = epoch_loss / len(train_dataloader) # Average loss per batch for the epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_epoch_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        if (epoch + 1) % 10 == 0 or (epoch + 1) == num_epochs:
            checkpoint_path = f'checkpoint_epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Save the final trained model state
    torch.save(model.state_dict(), 'trained_model_final.pth')
    print("Training finished and final model saved!")   
##########################
##########################
if __name__ == "__main__":
    jsonpath = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/dips/p5/dips005.json"
    imgdir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/dips/p5/imgs"
    #jsonpath = "/Users/Ben/Documents/dever/python/ptorch/ball_tracking/inferred_images/mlsvideo/jdata/jdata.json"
    #imgdir = "/Users/Ben/Documents/dever/python/ptorch/data/outframes/mlsvideo/imgs"
    num_epochs = 42
    boxlst,width,height = cleanCall(jsonpath,imgdir)
    trainer(boxlst,imgdir,num_epochs)