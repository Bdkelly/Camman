import os
import shutil
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QListWidget, QPushButton, QLabel, QMessageBox, QFileDialog
)
from PyQt5.QtCore import pyqtSignal, Qt, QSize
from PyQt5.QtGui import QDragEnterEvent, QDropEvent


class DragDropListWidget(QListWidget):

    file_dropped_signal = pyqtSignal(str)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDrop)
        self.setSelectionMode(QListWidget.SingleSelection)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path.endswith('.pth'):
                    self.file_dropped_signal.emit(file_path)
                else:
                    # Optional: Emit error or just ignore
                    pass
        else:
            event.ignore()

class ModelWindow(QWidget):
    model_selected_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Model Manager")
        self.setGeometry(200, 200, 400, 300)
        
        self.models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)

        self._create_widgets()
        self._setup_layout()
        self._refresh_model_list()

    def _create_widgets(self):
        self.info_label = QLabel("Drag & Drop .pth files below or use 'Upload Model'")
        self.info_label.setWordWrap(True)
        self.info_label.setAlignment(Qt.AlignCenter)

        self.model_list = DragDropListWidget()
        self.model_list.file_dropped_signal.connect(self.handle_file_drop)

        self.load_button = QPushButton("Load Selected Model")
        self.load_button.clicked.connect(self.load_selected_model)

        self.upload_button = QPushButton("Upload Model")
        self.upload_button.clicked.connect(self.upload_model_dialog)

    def _setup_layout(self):
        layout = QVBoxLayout()
        layout.addWidget(self.info_label)
        layout.addWidget(self.model_list)
        layout.addWidget(self.upload_button)
        layout.addWidget(self.load_button)
        self.setLayout(layout)

    def _refresh_model_list(self):
        self.model_list.clear()
        if os.path.exists(self.models_dir):
            files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
            for f in files:
                self.model_list.addItem(f)

    def handle_file_drop(self, file_path):
        filename = os.path.basename(file_path)
        dest_path = os.path.join(self.models_dir, filename)
        
        if os.path.exists(dest_path):
            reply = QMessageBox.question(
                self, 'File Exists', 
                f"Model '{filename}' already exists. Overwrite?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                return

        try:
            shutil.copy2(file_path, dest_path)
            self._refresh_model_list()
            # Select the newly added item
            items = self.model_list.findItems(filename, Qt.MatchExactly)
            if items:
                self.model_list.setCurrentItem(items[0])
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to copy file: {e}")

    def upload_model_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pth)"
        )
        if file_path:
            self.handle_file_drop(file_path)

    def load_selected_model(self):
        current_item = self.model_list.currentItem()
        if current_item:
            filename = current_item.text()
            full_path = os.path.join(self.models_dir, filename)
            self.model_selected_signal.emit(full_path)
            QMessageBox.information(self, "Success", f"Model '{filename}' requested for loading.")
        else:
            QMessageBox.warning(self, "Warning", "Please select a model from the list.")