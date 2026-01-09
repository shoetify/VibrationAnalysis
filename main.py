# -*- coding: utf-8 -*-
import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QPushButton, QTextEdit, QStatusBar, QProgressBar,
    QHBoxLayout, QLabel, QFileDialog, QLineEdit, QFormLayout, QGroupBox,
    QFrame, QSizePolicy
)
from PySide6.QtCore import QThread, QObject, Signal, Slot, Qt, QMimeData
from PySide6.QtGui import QIcon, QPixmap, QDragEnterEvent, QDropEvent

import logic

import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# --- Constants for UI ---
APP_TITLE = "Vibration Analysis Tool"
#WINDOW_ICON_PATH = "assets/LOGO.png"
DEFAULT_WINDOW_SIZE = (1000, 700) # Wider for sidebar

# --- QSS Dark Theme (Fluent/Material Inspired) ---
DARK_THEME_QSS = """
    /* Main Window & Backgrounds */
    QMainWindow, QWidget#MainContent {
        background-color: #202020;
    }
    QWidget#Sidebar {
        background-color: #2b2b2b;
        border-right: 1px solid #3c3c3c;
    }

    /* Typography */
    QLabel {
        color: #e0e0e0;
        font-family: 'Segoe UI', sans-serif;
        font-size: 10pt;
    }
    QLabel#TitleLabel {
        font-size: 12pt;
        font-weight: bold;
        color: #ffffff;
        margin-bottom: 0px;
        padding: 0 5px;
    }
    QLabel#SubtitleLabel {
        font-size: 10pt;
        color: #aaaaaa;
        margin-bottom: 20px;
        padding: 0 5px;
    }

    /* Inputs */
    QLineEdit {
        background-color: #333333;
        border: 1px solid #444444;
        border-radius: 4px; /* Slightly smaller radius for inputs */
        padding: 8px;
        color: #f0f0f0;
        font-family: 'Segoe UI', sans-serif;
    }
    QLineEdit:focus {
        border: 1px solid #60cdff; /* Fluent Blue */
        background-color: #3a3a3a;
    }

    /* Group Box */
    QGroupBox {
        font-family: 'Segoe UI', sans-serif;
        font-size: 11pt;
        font-weight: bold;
        border: 1px solid #444444;
        border-radius: 8px;
        margin-top: 20px; /* Space for title */
        padding-top: 25px;
        color: #e0e0e0;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 5px;
        left: 10px;
    }

    /* Buttons */
    QPushButton {
        background-color: #60cdff;
        color: #000000;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        font-family: 'Segoe UI', sans-serif;
        font-weight: bold;
        font-size: 10pt;
    }
    QPushButton:hover {
        background-color: #72d5ff;
    }
    QPushButton:pressed {
        background-color: #4cbbf4;
    }
    QPushButton:disabled {
        background-color: #3c3c3c;
        color: #777777;
        border: 1px solid #444444;
    }

    /* Drag & Drop Area */
    QLabel#DragDropArea {
        background-color: #252525;
        border: 2px dashed #555555;
        border-radius: 8px;
        color: #aaaaaa;
        font-size: 12pt;
    }
    QLabel#DragDropArea:hover {
        border-color: #60cdff;
        background-color: #2a2a2a;
        color: #e0e0e0;
    }

    /* Console Log */
    QTextEdit {
        background-color: #1e1e1e;
        border: 1px solid #333333;
        border-radius: 8px;
        font-family: 'Consolas', 'Courier New', monospace;
        font-size: 10pt;
        color: #cccccc;
        padding: 10px;
    }

    /* Progress Bar */
    QProgressBar {
        border: none;
        background-color: #333333;
        border-radius: 4px;
        height: 8px;
        text-align: center;
    }
    QProgressBar::chunk {
        background-color: #60cdff;
        border-radius: 4px;
    }
    
    /* ScrollBar */
    QScrollBar:vertical {
        border: none;
        background: #2b2b2b;
        width: 10px;
        margin: 0px 0px 0px 0px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical {
        background: #555555;
        min-height: 20px;
        border-radius: 5px;
    }
    QScrollBar::handle:vertical:hover {
        background: #666666;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        height: 0px;
    }
"""

class Worker(QObject):
    """
    A worker object that runs a long task in a separate thread.
    """
    progress = Signal(str)
    finished = Signal(bool)
    error = Signal(str)

    def __init__(self, log_file_path, sampling_frequency, diameter, natural_frequency, data_structure):
        super().__init__()
        self.log_file_path = log_file_path
        self.sampling_frequency = sampling_frequency
        self.diameter = diameter
        self.natural_frequency = natural_frequency
        self.data_structure = data_structure

    @Slot()
    def run(self):
        try:
            analysis_generator = logic.run_analysis(
                log_file_path=self.log_file_path,
                sampling_frequency=self.sampling_frequency,
                data_structure=self.data_structure,
                diameter=self.diameter,
                natural_frequency=self.natural_frequency
            )
            for status_message in analysis_generator:
                self.progress.emit(status_message)
            self.finished.emit(True)
        except Exception as e:
            self.error.emit(str(e))
            self.finished.emit(False)

class DragDropWidget(QLabel):
    """
    Custom widget for Drag & Drop file selection.
    """
    fileDropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("DragDropArea")
        self.setText("Drag & Drop Log File (.xlsx) Here\n\nor Click to Browse")
        self.setAlignment(Qt.AlignCenter)
        self.setAcceptDrops(True)
        self.setMinimumHeight(150)
        self.setCursor(Qt.PointingHandCursor)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path.lower().endswith('.xlsx'):
                self.fileDropped.emit(file_path)
            else:
                # Optional: Signal invalid file type or handle visually
                self.setText("Invalid file type! Please drop an .xlsx file.")
                
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Log File", "", "Excel Files (*.xlsx)"
            )
            if file_path:
                self.fileDropped.emit(file_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_TITLE)
        logo_path = resource_path("assets/logo.png")
        icon_path = resource_path("assets/icon.ico")
        if Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))
        self.resize(*DEFAULT_WINDOW_SIZE)

        # State
        self.selected_log_file = None
        self.thread = None
        self.worker = None

        # --- Main Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ================= Sidebar =================
        self.sidebar = QWidget()
        self.sidebar.setObjectName("Sidebar")
        self.sidebar.setFixedWidth(300)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(20, 30, 20, 30)
        sidebar_layout.setSpacing(20)

        # Branding
        self.logo_label = QLabel()
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            self.logo_label.setPixmap(pixmap.scaledToHeight(80, Qt.SmoothTransformation))
        self.logo_label.setAlignment(Qt.AlignCenter)
        
        self.title_label = QLabel("Supervisor: Tongming Zhou")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignLeft)
        self.title_label.setWordWrap(True)
        
        self.subtitle_label = QLabel("Supported email:\ndifei.xiao@research.uwa.edu.au")
        self.subtitle_label.setObjectName("SubtitleLabel")
        self.subtitle_label.setAlignment(Qt.AlignLeft)
        self.subtitle_label.setWordWrap(True)

        sidebar_layout.addWidget(self.logo_label)
        
        # Info Group (to remove gap between title and subtitle)
        info_container = QWidget()
        info_layout = QVBoxLayout(info_container)
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(0)
        
        info_layout.addWidget(self.title_label)
        info_layout.addWidget(self.subtitle_label)
        
        sidebar_layout.addWidget(info_container)
 
        # Inputs
        settings_group = QGroupBox("Configuration")
        settings_layout = QFormLayout(settings_group)
        settings_layout.setLabelAlignment(Qt.AlignLeft)
        settings_layout.setVerticalSpacing(15)

        self.sampling_freq_input = QLineEdit(str(logic.DEFAULT_SAMPLING_FREQUENCY))
        # New Inputs
        self.diameter_input = QLineEdit(str(logic.DEFAULT_DIAMETER))
        self.natural_freq_input = QLineEdit(str(logic.DEFAULT_NATURAL_FREQUENCY))
        
        self.data_structure_input = QLineEdit(', '.join(map(str, logic.DEFAULT_DATA_STRUCTURE)))

        settings_layout.addRow(QLabel("Sampling Freq (Hz):"), self.sampling_freq_input)
        settings_layout.addRow(QLabel("Diameter (mm):"), self.diameter_input)
        settings_layout.addRow(QLabel("Natural Freq (Hz):"), self.natural_freq_input)
        settings_layout.addRow(QLabel("Data Structure:"), self.data_structure_input)
        
        # Helper text for structure
        structure_help = QLabel("1=Time, 2=Disp, 3=Acc, 0=Ignore")
        structure_help.setStyleSheet("color: #777; font-size: 8pt;")
        structure_help.setWordWrap(True)
        settings_layout.addRow(structure_help)

        sidebar_layout.addWidget(settings_group)
        sidebar_layout.addStretch() # Push everything up

        # Version/Footer
        version_label = QLabel("v1.3.0")
        version_label.setStyleSheet("color: #555; font-size: 8pt;")
        version_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(version_label)

        # ================= Main Content =================
        self.main_content = QWidget()
        self.main_content.setObjectName("MainContent")
        content_layout = QVBoxLayout(self.main_content)
        content_layout.setContentsMargins(30, 30, 30, 30)
        content_layout.setSpacing(20)

        # Header for Main Content
        header_label = QLabel("Analysis Dashboard")
        header_label.setStyleSheet("font-size: 18pt; font-weight: bold; color: #fff;")
        content_layout.addWidget(header_label)

        # Drag & Drop Area
        self.drag_drop_area = DragDropWidget()
        self.drag_drop_area.fileDropped.connect(self.on_file_selected)
        content_layout.addWidget(self.drag_drop_area)

        # Selected File Indicator
        self.file_label = QLabel("No file selected")
        self.file_label.setStyleSheet("color: #888; font-style: italic; margin-left: 5px;")
        content_layout.addWidget(self.file_label)

        # Action Buttons
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setEnabled(False) # Disabled initially
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self.start_analysis)
        content_layout.addWidget(self.run_button)

        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        content_layout.addWidget(self.progress_bar)

        # Console Log
        log_label = QLabel("System Log")
        log_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        content_layout.addWidget(log_label)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        content_layout.addWidget(self.log_area, 1) # Expand to fill space

        # Add widgets to main Split
        main_layout.addWidget(self.sidebar)
        main_layout.addWidget(self.main_content)

        # Status Bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

    def on_file_selected(self, file_path):
        self.selected_log_file = file_path
        self.file_label.setText(f"Selected: {Path(file_path).name}")
        self.file_label.setStyleSheet("color: #60cdff; font-weight: bold; margin-left: 5px;")
        self.drag_drop_area.setText(f"File Loaded:\n{Path(file_path).name}")
        self.drag_drop_area.setStyleSheet("border-color: #60cdff; color: #e0e0e0;")
        self.run_button.setEnabled(True)
        self.log_area.append(f"> File selected: {file_path}")

    def start_analysis(self):
        self.log_area.clear()
        
        # --- Parameter Validation ---
        try:
            sampling_freq = float(self.sampling_freq_input.text())
            # Read new inputs
            diameter = float(self.diameter_input.text())
            natural_freq = float(self.natural_freq_input.text())
            
            data_structure_str = self.data_structure_input.text()
            data_structure = [int(x.strip()) for x in data_structure_str.split(',')]
            
            if not all(isinstance(i, int) for i in data_structure):
                raise ValueError("Data structure must contain only integers.")

        except ValueError as e:
            self.log_area.setText(f"ERROR: Invalid input parameter.\nDetails: {e}")
            self.status_bar.showMessage("Invalid settings.", 3000)
            return
            
        self.log_area.append("> Starting analysis...")
        self.run_button.setEnabled(False)
        self.drag_drop_area.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0) # Indeterminate

        self.thread = QThread()
        self.worker = Worker(
            log_file_path=self.selected_log_file,
            sampling_frequency=sampling_freq,
            diameter=diameter,
            natural_frequency=natural_freq,
            data_structure=data_structure
        )
        self.worker.moveToThread(self.thread)

        self.worker.progress.connect(self.update_log)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error.connect(self.on_analysis_error)
        
        self.thread.started.connect(self.worker.run)
        self.thread.start()

    @Slot(str)
    def update_log(self, message: str):
        self.log_area.append(message)
        # Auto-scroll
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    @Slot(str)
    def on_analysis_error(self, error_message: str):
        self.log_area.append(f"!!! ERROR: {error_message} !!!")

    @Slot(bool)
    def on_analysis_finished(self, success: bool):
        self.progress_bar.setVisible(False)
        self.run_button.setEnabled(True)
        self.drag_drop_area.setEnabled(True)
        
        if success:
            self.log_area.append("\n> Analysis completed successfully.")
            self.status_bar.showMessage("Done.", 5000)
        else:
            self.log_area.append("\n> Analysis stopped due to error.")
            self.status_bar.showMessage("Error.", 5000)
        
        if self.thread:
            self.thread.quit()
            self.thread.wait()
            self.thread = None
            self.worker = None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_THEME_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
