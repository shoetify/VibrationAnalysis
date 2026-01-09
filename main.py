# -*- coding: utf-8 -*-
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QTextEdit, QStatusBar, QProgressBar,
    QHBoxLayout, QLabel, QFileDialog, QLineEdit, QFormLayout, QGroupBox
)
from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt
from PySide6.QtGui import QIcon, QPixmap

import logic

# --- Constants for UI ---
APP_TITLE = "Vibration Analysis Tool"
WINDOW_ICON_PATH = "LOGO.png"  # Path to the icon file
DEFAULT_WINDOW_SIZE = (800, 700) # Increased height for new fields

# --- QSS Dark Theme (Material Design Inspired) ---
DARK_THEME_QSS = """
    QMainWindow {
        background-color: #2b2b2b;
    }
    QWidget {
        background-color: #3c3c3c;
        color: #f0f0f0;
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 10pt;
    }
    QLabel#LogoLabel {
        background-color: transparent;
        padding: 10px;
    }
    QLabel {
        font-size: 10pt;
        font-weight: bold;
    }
    QGroupBox {
        font-size: 12pt;
        font-weight: bold;
        border: 1px solid #555;
        border-radius: 8px;
        padding-top: 20px; /* Provides space for the title inside the border */
        margin-top: 10px;
        background-color: #3c3c3c;
    }
    QGroupBox::title {
        subcontrol-origin: padding;
        subcontrol-position: top left;
        padding: 0 10px;
        left: 10px; /* Indent the title from the left edge */
        color: #f0f0f0;
    }
    QTextEdit {
        background-color: #2b2b2b;
        border: 1px solid #555;
        border-radius: 8px;
        font-family: 'Consolas', 'Courier New', monospace;
        padding: 5px;
    }
    QLineEdit {
        background-color: #2b2b2b;
        border: 1px solid #555;
        padding: 6px;
        border-radius: 4px;
    }
    QLineEdit:focus {
        border: 1px solid #0078d7;
    }
    QPushButton {
        background-color: #0078d7;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 4px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #005a9e;
    }
    QPushButton:pressed {
        background-color: #004578;
    }
    QPushButton:disabled {
        background-color: #484848;
        color: #888;
    }
    QProgressBar {
        border: 1px solid #555;
        border-radius: 5px;
        text-align: center;
        background-color: #2b2b2b;
        color: white;
    }
    QProgressBar::chunk {
        background-color: #0078d7;
        border-radius: 4px;
    }
    QStatusBar {
        font-size: 9pt;
    }
"""

class Worker(QObject):
    """
    A worker object that runs a long task in a separate thread.
    Emits signals to communicate with the main UI thread.
    """
    progress = Signal(str)
    finished = Signal(bool)  # Indicates success or failure
    error = Signal(str)

    def __init__(
        self,
        log_file_path: str,
        sampling_frequency: float,
        cutoff_frequency: float,
        data_structure: list[int]
    ):
        super().__init__()
        self.log_file_path = log_file_path
        self.sampling_frequency = sampling_frequency
        self.cutoff_frequency = cutoff_frequency
        self.data_structure = data_structure

    @Slot()
    def run(self):
        """Execute the analysis logic."""
        try:
            analysis_generator = logic.run_analysis(
                log_file_path=self.log_file_path,
                sampling_frequency=self.sampling_frequency,
                cutoff_frequency_hz=self.cutoff_frequency,
                data_structure=self.data_structure,
            )
            for status_message in analysis_generator:
                self.progress.emit(status_message)
            self.finished.emit(True)
        except Exception as e:
            # Emit both error and finished signals on exception
            self.error.emit(str(e))
            self.finished.emit(False)


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        if WINDOW_ICON_PATH:
            self.setWindowIcon(QIcon(WINDOW_ICON_PATH))
        self.resize(*DEFAULT_WINDOW_SIZE)

        # --- UI Components ---
        self.logo_label = QLabel()
        self.logo_label.setObjectName("LogoLabel")
        pixmap = QPixmap(WINDOW_ICON_PATH)
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaledToHeight(80, Qt.SmoothTransformation))
            self.logo_label.setAlignment(Qt.AlignCenter)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)

        self.run_button = QPushButton("Run Analysis")
        self.select_log_button = QPushButton("Select Log File")
        self.log_file_label = QLabel("No log file selected.")
        self.log_file_label.setStyleSheet("font-size: 9pt; font-weight: normal; color: #bbb;")
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)

        # --- Settings Inputs ---
        self.sampling_freq_input = QLineEdit(str(logic.DEFAULT_SAMPLING_FREQUENCY))
        self.cutoff_freq_input = QLineEdit(str(logic.DEFAULT_CUTOFF_FREQUENCY_HZ))
        self.data_structure_input = QLineEdit(', '.join(map(str, logic.DEFAULT_DATA_STRUCTURE)))

        # --- Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        main_layout.addWidget(self.logo_label)

        # --- Settings Group ---
        settings_group = QGroupBox("Analysis Settings")
        settings_layout = QFormLayout(settings_group)
        settings_layout.addRow(QLabel("Sampling Frequency (Hz):"), self.sampling_freq_input)
        settings_layout.addRow(QLabel("Cutoff Frequency (Hz):"), self.cutoff_freq_input)
        settings_layout.addRow(QLabel("Data Structure:"), self.data_structure_input)
        main_layout.addWidget(settings_group)

        # --- File & Controls Group ---
        controls_group = QGroupBox("File & Execution")
        controls_layout = QVBoxLayout(controls_group)

        file_layout = QHBoxLayout()
        file_layout.addWidget(self.select_log_button)
        file_layout.addWidget(self.log_file_label)
        file_layout.addStretch()
        
        controls_layout.addLayout(file_layout)
        controls_layout.addWidget(self.run_button)

        main_layout.addWidget(controls_group)

        # --- Log Area ---
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_area)
        main_layout.addWidget(log_group, 1) # Give log area stretch factor

        # --- Status Bar ---
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.addPermanentWidget(self.progress_bar, 1)

        # --- Connections ---
        self.run_button.clicked.connect(self.start_analysis)
        self.select_log_button.clicked.connect(self.select_log_file)
        
        self.thread = None
        self.worker = None
        self.selected_log_file = None

    def select_log_file(self):
        """Open a file dialog to select the log.xlsx file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Log File", "", "Excel Files (*.xlsx)"
        )
        if file_path:
            self.selected_log_file = file_path
            self.log_file_label.setText(f"Using: {Path(file_path).name}")
            self.log_file_label.setStyleSheet("font-size: 9pt; font-weight: normal; color: #90ee90;") # Light green

    def start_analysis(self):
        """Prepare and start the worker thread for analysis."""
        self.log_area.clear()
        
        # --- Parameter Validation ---
        try:
            sampling_freq = float(self.sampling_freq_input.text())
            cutoff_freq = float(self.cutoff_freq_input.text())
            
            data_structure_str = self.data_structure_input.text()
            data_structure = [int(x.strip()) for x in data_structure_str.split(',')]
            
            if not all(isinstance(i, int) for i in data_structure):
                raise ValueError("Data structure must contain only integers.")

        except ValueError as e:
            self.log_area.setText(f"ERROR: Invalid input parameter.\n- Frequencies must be numbers.\n- Data structure must be a comma-separated list of integers (e.g., 1, 2, 3, 0).\nDetails: {e}")
            self.status_bar.showMessage("Invalid settings. Please correct them.", 5000)
            return
            
        self.log_area.append("Starting analysis...")
        self.run_button.setEnabled(False)
        self.select_log_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate progress bar

        # Create and start the thread
        self.thread = QThread()
        self.worker = Worker(
            log_file_path=self.selected_log_file,
            sampling_frequency=sampling_freq,
            cutoff_frequency=cutoff_freq,
            data_structure=data_structure
        )
        self.worker.moveToThread(self.thread)

        # Connect worker signals to main thread slots
        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @Slot(str)
    def update_log(self, message: str):
        """Append a message to the log area."""
        self.log_area.append(message)
        if "ERROR" in message:
            self.status_bar.showMessage(f"Error occurred. Check log.", 5000)
        else:
            self.status_bar.showMessage(message, 3000)

    @Slot(str)
    def on_analysis_error(self, error_message: str):
        """Show an error message."""
        self.log_area.append(f"--- FATAL ERROR ---\n{error_message}\n--------------------")


    @Slot(bool)
    def on_analysis_finished(self, success: bool):
        """Clean up after the analysis is complete."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100) # Reset
        self.run_button.setEnabled(True)
        self.select_log_button.setEnabled(True)
        
        if success:
            self.log_area.append("\nAnalysis completed successfully!")
            self.status_bar.showMessage("Analysis completed successfully!", 5000)
        else:
            self.log_area.append("\nAnalysis failed. See log for details.")
            self.status_bar.showMessage("Analysis failed. See log for details.", 5000)
        
        # Clean up the thread
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None


if __name__ == "__main__":
    # Add Path to the imports
    from pathlib import Path
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
