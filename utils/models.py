import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_fasterrcnn_model_single_class(num_classes=2):
    """
    Returns a Faster R-CNN model configured for a specified number of output classes.
    Defaults to 2 classes (your object class + background).
    """
    # Load pre-trained weights. DEFAULT uses the best available weights.
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    
    # Instantiate the model with pre-trained weights
    model = fasterrcnn_resnet50_fpn(weights=weights)
    
    # Get the number of input features for the classifier
    # This is the number of features coming from the ROI pooling layer
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one for your specific number of classes
    # num_classes includes the background class
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model