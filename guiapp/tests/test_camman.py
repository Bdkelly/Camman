import unittest
import numpy as np
from PyQt5.QtGui import QImage
from guiapp.camman import VideoThread

class TestCamman(unittest.TestCase):
    def test_convert_cv_qt(self):
        # Create dummy instance
        video_thread = VideoThread()
        cv_img = np.zeros((480, 640, 3), dtype=np.uint8)
        qt_img = video_thread._convert_cv_qt(cv_img)
        # Test dummy instance
        self.assertIsInstance(qt_img, QImage)
        self.assertEqual(qt_img.width(), 640)
        self.assertEqual(qt_img.height(), 480)

if __name__ == '__main__':
    unittest.main()
