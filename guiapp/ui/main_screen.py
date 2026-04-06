"""
main_screen — Primary application window for the Camman GUI.

``MainWindow`` is the top-level PyQt5 window. It creates the ``VideoThread``,
builds the UI, and wires all signals between the main thread and the
background processing thread.

Layout (horizontal split):

- **Left panel**: 640×480 video feed label, inference toggle, agent toggle,
  platform connectivity button, model selector button.
- **Right panel**: Tracking settings group (command interval slider, manual
  pan buttons), scrollable command log.

Signal/slot connections:

- GUI buttons → ``MainWindow`` methods → signals → ``VideoThread`` slots
- ``VideoThread`` → ``change_pixmap_signal`` → ``update_image``
- ``VideoThread`` → ``command_log_signal`` → ``update_log_display``
"""

from PyQt5.QtWidgets import (
    QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLabel, QHBoxLayout, QTextEdit, QGroupBox, QSlider
)
from PyQt5.QtGui import QPixmap, QFont, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt

from guiapp.threads.video_threads import VideoThread
from guiapp.utils.ser_con import move_left, move_right
from guiapp.platform_screen import PlatformWindow
from guiapp.ui.model_menu import ModelWindow


class MainWindow(QMainWindow):
    """Main application window for CamMan.

    Owns the ``VideoThread`` and exposes controls for inference, agent
    activation, manual pan overrides, and model/platform management.

    Signals:
        inference_toggle_signal (bool): Emitted when the inference button is toggled.
        command_interval_update_signal (float): Emitted when the interval slider changes.
        agent_toggle_signal (bool): Emitted when the agent button is toggled.
        update_model_signal (str): Emitted with the new model path when a model is selected.
    """

    inference_toggle_signal = pyqtSignal(bool)
    command_interval_update_signal = pyqtSignal(float)
    agent_toggle_signal = pyqtSignal(bool)
    update_model_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("CamMan GUI")
        self.setGeometry(100, 100, 950, 650) 
        
        self.thread = VideoThread() 
        
        self._create_widgets()
        self._setup_layout()
        self._connect_signals()
        
        self.thread.start()

    def _create_widgets(self):
        self.image_label = QLabel("Video Feed will appear here.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        
        self.control_button = QPushButton("Start Inference")
        self.control_button.setCheckable(True)

        self.agent_control_button = QPushButton("Start CamMan Agent")
        self.agent_control_button.setCheckable(True)
        self.agent_control_button.setStyleSheet("background-color: lightblue")

        self.platform_button = QPushButton("Platform")
        self.platform_button.setStyleSheet("background-color: lightgray")

        self.models_button = QPushButton("Models")
        self.models_button.setStyleSheet("background-color: lightgreen")

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMinimumWidth(250)
        self.log_display.setMaximumHeight(550)
        self.log_display.setFont(QFont('Courier', 10))
        self.log_display.setText("--- ESP32 Command Log ---\n")
        
        self.settings_group = QGroupBox("Tracking Settings")
        
        self.interval_slider = QSlider(Qt.Horizontal)
        self.interval_slider.setRange(1, 20)
        self.interval_slider.setValue(10)
        
        self.interval_label = QLabel(f"Command Interval: {self.interval_slider.value() / 10.0}s")
        
        self.manual_left_button = QPushButton("Manual Left")
        self.manual_right_button = QPushButton("Manual Right")


    #Organizes the UI components into layouts
    def _setup_layout(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_h_layout = QHBoxLayout(central_widget)
        
        video_control_v_layout = QVBoxLayout()
        video_control_v_layout.addWidget(self.image_label)
        video_control_v_layout.addWidget(self.control_button)
        video_control_v_layout.addWidget(self.agent_control_button)
        video_control_v_layout.addWidget(self.platform_button)
        video_control_v_layout.addWidget(self.models_button)
        main_h_layout.addLayout(video_control_v_layout)
        
        right_v_layout = QVBoxLayout()
        
        settings_layout = QVBoxLayout(self.settings_group)
        settings_layout.addWidget(self.interval_label)
        settings_layout.addWidget(self.interval_slider)
        
        manual_h_layout = QHBoxLayout()
        manual_h_layout.addWidget(self.manual_left_button)
        manual_h_layout.addWidget(self.manual_right_button)
        settings_layout.addLayout(manual_h_layout)

        right_v_layout.addWidget(self.settings_group)
        right_v_layout.addWidget(QLabel("Command Log:"))
        right_v_layout.addWidget(self.log_display)
        
        main_h_layout.addLayout(right_v_layout)

    def _connect_signals(self):
        self.control_button.clicked.connect(self.toggle_inference)
        self.agent_control_button.clicked.connect(self.toggle_agent)
        self.platform_button.clicked.connect(self.open_platform_menu)
        self.models_button.clicked.connect(self.open_model_menu)
        self.interval_slider.valueChanged.connect(self.update_interval_ui)
        self.interval_slider.valueChanged.connect(self.update_interval_thread)

        self.manual_left_button.clicked.connect(lambda: move_left(self.thread.ser))
        self.manual_right_button.clicked.connect(lambda: move_right(self.thread.ser))
        
        self.inference_toggle_signal.connect(self.thread.toggle_inference)
        self.agent_toggle_signal.connect(self.thread.toggle_agent)
        self.command_interval_update_signal.connect(self.thread.set_command_interval)
        self.update_model_signal.connect(self.thread.update_model)
        
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.command_log_signal.connect(self.update_log_display)
    

    @pyqtSlot(str)
    def update_log_display(self, command_string):
        """Append a log message to the command log widget.

        Args:
            command_string (str): Message to append (typically a serial command
                or status update from ``VideoThread``).
        """
        self.log_display.append(command_string)

    @pyqtSlot(QImage)
    def update_image(self, qt_img):
        """Update the video feed label with a new frame.

        Args:
            qt_img (QImage): The latest processed frame from ``VideoThread``.
        """
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))

    def open_platform_menu(self):
        """Open the platform connectivity dialog."""
        self.platform_menu = PlatformWindow(self)
        self.platform_menu.show()

    def open_model_menu(self):
        """Open the model selection dialog and connect its signal."""
        self.model_menu = ModelWindow(self)
        self.model_menu.model_selected_signal.connect(self.emit_update_model)
        self.model_menu.show()

    def emit_update_model(self, model_path):
        """Forward a selected model path to the VideoThread.

        Args:
            model_path (str): Path to the new ``.pth`` model file.
        """
        self.update_model_signal.emit(model_path)

    def toggle_inference(self, checked):
        """Handle inference button toggle.

        Args:
            checked (bool): True when the button is pressed (inference starts).
        """
        if checked:
            self.control_button.setText("Stop Inference")
            self.inference_toggle_signal.emit(True)
        else:
            self.control_button.setText("Start Inference")
            self.inference_toggle_signal.emit(False)

    def toggle_agent(self, checked):
        """Handle agent button toggle.

        Args:
            checked (bool): True when the button is pressed (agent starts).
        """
        if checked:
            self.agent_control_button.setText("Stop CamMan Agent")
            self.agent_toggle_signal.emit(True)
        else:
            self.agent_control_button.setText("Start CamMan Agent")
            self.agent_toggle_signal.emit(False)

    def update_interval_ui(self, value):
        """Update the interval label when the slider moves.

        Args:
            value (int): Raw slider value (range 1–20); displayed as value/10 seconds.
        """
        self.interval_label.setText(f"Command Interval: {value / 10.0:.1f}s")

    def update_interval_thread(self, value):
        """Emit the new command interval to the VideoThread.

        Args:
            value (int): Raw slider value (range 1–20); converted to seconds.
        """
        self.command_interval_update_signal.emit(value / 10.0)

    def closeEvent(self, event):
        """Stop the VideoThread gracefully when the window is closed.

        Args:
            event (QCloseEvent): The close event (accepted unconditionally).
        """
        self.thread.stop()
        event.accept()