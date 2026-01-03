#!/usr/bin/python3
import sys

import librosa
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from scipy.io import wavfile

from assets.stemmy import roformer_models, roformer_separator

formats = ["FLAC", "MP3", "WAV"]
roformer_compute = "cpu"
device = roformer_compute


def number_convert(s):
    """
    Converts a string to an integer if it has an integer value (e.g., "123" or "123.0").
    Returns a float if it is a non-integer float value (e.g., "4.56").
    Returns None or handles the error if the input is not a valid number.
    Oh my god this IS FINALLY WORKING
    """
    try:
        # Attempt to convert directly to int (handles "123")
        return int(s)
    except ValueError:
        # If int() fails, try converting to float first (handles "4.56" and "123.0")
        try:
            f = float(s)
            # Check if the float has an integer value
            if f.is_integer():
                return int(f)
            else:
                return f
        except ValueError:
            # If float() fails, the string is not a valid number
            print(f"Error: '{s}' is not a valid number.")
            return None  # Or raise an exception, or return a default value like 0


class AudioSeparatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("stemmy")
        self.setGeometry(100, 100, 800, 500)

        # Main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Initialize components dictionary
        self.components = {}
        self.all_configurable_inputs = []

        # Create tabs
        self.create_roformer_tab()

    #        self.create_mdxnet_tab()
    #        self.create_vrarch_tab()
    #        self.create_demucs_tab()

    def create_roformer_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # Helper function to create slider with a value label
        def create_labeled_slider(min_val, max_val, default_val, divisor=1):
            container = QHBoxLayout()
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default_val)

            # Label to display value
            label = QLabel()

            def update_value(v):
                raw_val = v / divisor
                # Convert to int if it's a whole number, otherwise keep as float
                final_val = int(raw_val) if raw_val.is_integer() else raw_val

                label.setText(str(final_val))
                # Optional: call external function here with final_val
                # self.on_slider_changed(final_val)

            # Initialize label and connect signal
            update_value(default_val)
            slider.valueChanged.connect(update_value)

            label.setFixedWidth(40)
            container.addWidget(slider)
            container.addWidget(label)

            return slider, container

        # Model and format selection
        model_format_layout = QHBoxLayout()
        model_format_layout.addWidget(QLabel("Select the model:"))
        self.roformer_model = QComboBox()
        roformers_list = (list(roformer_models.keys()),)
        self.roformer_model.addItems(*roformers_list)
        model_format_layout.addWidget(self.roformer_model)

        model_format_layout.addWidget(QLabel("Output format:"))
        self.roformer_output_format = QComboBox()
        self.roformer_output_format.addItems(formats)
        model_format_layout.addWidget(self.roformer_output_format)

        model_format_layout.addWidget(QLabel("Processor:"))
        self.roformer_compute = QComboBox()
        self.roformer_compute.addItems(["cpu", "cuda"])
        model_format_layout.addWidget(self.roformer_compute)
        layout.addLayout(model_format_layout)

        # Configuration group
        config_group = QGroupBox("Configuration")
        config_layout = QVBoxLayout(config_group)

        # First row: Segment size
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Segment size:"))
        self.roformer_segment_size, s_layout = create_labeled_slider(32, 4096, 2048)
        self.roformer_segment_size.setTickInterval(32)
        row1.addLayout(s_layout)

        self.roformer_override_segment_size = QCheckBox("Override segment size")
        self.roformer_override_segment_size.setChecked(True)
        row1.addWidget(self.roformer_override_segment_size)
        config_layout.addLayout(row1)

        # Second row: Overlap and Batch Size
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Overlap:"))
        self.roformer_overlap, o_layout = create_labeled_slider(2, 99, 1)
        row2.addLayout(o_layout)

        row2.addWidget(QLabel("Batch size:"))
        self.roformer_batch_size, b_layout = create_labeled_slider(1, 128, 1)
        row2.addLayout(b_layout)
        config_layout.addLayout(row2)

        # Third row: Normalization and Amplification (using divisor 10.0 for floats)
        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Normalization threshold:"))
        self.roformer_normalization_threshold, n_layout = create_labeled_slider(
            1, 10, 9, 10.0
        )
        row3.addLayout(n_layout)

        row3.addWidget(QLabel("Amplification threshold:"))
        self.roformer_amplification_threshold, a_layout = create_labeled_slider(
            1, 10, 7, 10.0
        )
        row3.addLayout(a_layout)
        config_layout.addLayout(row3)

        layout.addWidget(config_group)

        # Audio input row
        row4 = QHBoxLayout()
        self.file_path_label = QLabel("No file selected")
        row4.addWidget(self.file_path_label)
        self.roformer_audio_button = QPushButton("Select Audio File")
        self.roformer_audio_button.clicked.connect(self.select_audio_file)
        row4.addWidget(self.roformer_audio_button)
        config_layout.addLayout(row4)

        # Separate button
        self.roformer_button = QPushButton("Separate!")
        # Note: Updated lambda to grab current values at the moment of clicking
        self.roformer_button.clicked.connect(
            lambda: roformer_separator(
                self.audio_path,
                self.roformer_model.currentText(),
                self.roformer_output_format.currentText(),
                self.roformer_segment_size.value(),
                self.roformer_override_segment_size.isChecked(),
                self.roformer_overlap.value(),
                self.roformer_batch_size.value(),
                self.roformer_normalization_threshold.value() / 10.0,
                self.roformer_amplification_threshold.value() / 10.0,
            )
        )
        layout.addWidget(self.roformer_button)

        # Output stems
        stems_layout = QHBoxLayout()
        for i in range(1, 3):
            stems_layout.addWidget(QLabel(f"Stem {i}:"))
            stem_frame = QFrame()
            stem_frame.setFrameStyle(QFrame.Shape.StyledPanel)
            stem_frame.setMinimumHeight(60)
            stems_layout.addWidget(stem_frame)
            setattr(self, f"roformer_stem{i}", stem_frame)
        layout.addLayout(stems_layout)

        self.tabs.addTab(tab, "Roformers")

    def select_audio_file(self):
        # Define supported audio file filters
        file_filter = "Audio Files (*.wav *.mp3 *.flac *.ogg);;All Files (*)"

        # Use static method to get file path
        # Returns tuple: (file_path, selected_filter)
        file_path, _ = QFileDialog.getOpenFileName(
            self,  # Parent widget
            "Select Audio File",  # Dialog title
            "",  # Initial directory (empty = last used)
            file_filter,  # File filter
        )

        # Store the file path in instance variable
        if file_path:  # If user didn't cancel the dialog
            self.audio_path = file_path
            self.file_path_label.setText(f"Selected: {file_path}")
            print(f"Audio file path stored: {self.audio_path}")
        else:
            # User cancelled the dialog
            self.audio_path = None
            self.file_path_label.setText("No file selected")
            print("File selection cancelled")

        return self.audio_path
        # Load as float (default)

    def wav2tuple(self, audio_path):
        y, sr = librosa.load(self.audio_path, sr=None)

        # Scale and convert to 16-bit integers
        y_int = (y * 32767).astype(np.int16)

        # Convert to tuple
        audio_as_tuple = tuple(y_int.tolist())
        return self.conv_tuple


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioSeparatorApp()
    window.show()
    sys.exit(app.exec())
