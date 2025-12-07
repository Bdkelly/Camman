from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QDialog, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
import asyncio
from guiapp.utils.bluecon import ESP32Controller
from guiapp.utils.ser_con import find_esp32

class WiredWorker(QThread):
    result_signal = pyqtSignal(str)

    def run(self):
        port = find_esp32()
        self.result_signal.emit(port if port else "None")

class BluetoothWorker(QThread):
    status_signal = pyqtSignal(bool)

    def __init__(self):
        super().__init__()
        self.controller = ESP32Controller()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        found = loop.run_until_complete(self.controller.scan_and_check())
        self.status_signal.emit(found)
        loop.close()

class StatusIndicator(QWidget):
    def __init__(self, color=Qt.red):
        super().__init__()
        self.color = color
        self.setFixedSize(20, 20)

    def set_color(self, color):
        self.color = color
        self.update()

    def paintEvent(self, event):
        from PyQt5.QtGui import QPainter, QBrush
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(self.color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 20, 20)

class PlatformWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Platform Connectivity")
        self.setFixedSize(400, 200)
        self._setup_ui()
        
        self.wired_worker = WiredWorker()
        self.wired_worker.result_signal.connect(self.update_wired_status)
        
        self.bt_worker = BluetoothWorker()
        self.bt_worker.status_signal.connect(self.update_bt_status)
        
        self.start_bt_scan()

    def _setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Wired Section
        wired_group = QFrame()
        wired_group.setFrameShape(QFrame.StyledPanel)
        wired_layout = QHBoxLayout(wired_group)
        
        self.wired_indicator = StatusIndicator(Qt.red)
        self.wired_label = QLabel("Wired Status: Unknown")
        self.scan_wired_btn = QPushButton("Scan Wired")
        self.scan_wired_btn.clicked.connect(self.start_wired_scan)
        
        wired_layout.addWidget(self.wired_indicator)
        wired_layout.addWidget(self.wired_label)
        wired_layout.addStretch()
        wired_layout.addWidget(self.scan_wired_btn)
        
        layout.addWidget(QLabel("<b>Wired Connection</b>"))
        layout.addWidget(wired_group)

        # Bluetooth Section
        bt_group = QFrame()
        bt_group.setFrameShape(QFrame.StyledPanel)
        bt_layout = QHBoxLayout(bt_group)
        
        self.bt_indicator = StatusIndicator(Qt.red)
        self.bt_label = QLabel("Bluetooth Status: Scanning...")
        
        bt_layout.addWidget(self.bt_indicator)
        bt_layout.addWidget(self.bt_label)
        bt_layout.addStretch()
        
        layout.addWidget(QLabel("<b>Bluetooth Connection</b>"))
        layout.addWidget(bt_group)
        
        layout.addStretch()

    def start_wired_scan(self):
        self.wired_label.setText("Wired Status: Scanning...")
        self.wired_indicator.set_color(Qt.yellow)
        self.wired_worker.start()

    def update_wired_status(self, port):
        if port != "None":
            self.wired_label.setText(f"Wired Status: Connected ({port})")
            self.wired_indicator.set_color(Qt.green)
        else:
            self.wired_label.setText("Wired Status: Disconnected")
            self.wired_indicator.set_color(Qt.red)

    def start_bt_scan(self):
        self.bt_label.setText("Bluetooth Status: Scanning...")
        self.bt_indicator.set_color(Qt.yellow)
        self.bt_worker.start()

    def update_bt_status(self, found):
        if found:
            self.bt_label.setText("Bluetooth Status: Connected/Found")
            self.bt_indicator.set_color(Qt.green)
        else:
            self.bt_label.setText("Bluetooth Status: Not Found")
            self.bt_indicator.set_color(Qt.red)