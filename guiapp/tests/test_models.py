import unittest
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from guiapp.utils.models import get_fasterrcnn_model_single_class

class TestModels(unittest.TestCase):
    def test_get_fasterrcnn_model_single_class(self):
        num_classes = 5
        model = get_fasterrcnn_model_single_class(num_classes)
        self.assertIsInstance(model.roi_heads.box_predictor, FastRCNNPredictor)
        self.assertEqual(model.roi_heads.box_predictor.cls_score.out_features, num_classes)

if __name__ == '__main__':
    unittest.main()