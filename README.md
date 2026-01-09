# Vibration Analysis Tool

This application is a desktop tool for running vibration analysis on signal data, built with Python and PySide6.

## 1. Prerequisites

- **Python**: Ensure you have Python 3.8+ installed. You can download it from [python.org](https://www.python.org/downloads/).
- **Pip**: The Python package installer, which comes with modern Python installations.

## 2. Installation

1.  **Clone the repository or download the source code** to a local directory.

2.  **Open a terminal or command prompt** in the project's root directory.

3.  **Create and activate a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    # On Windows
    .venv\\Scripts\\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

4.  **Install the required packages** using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## 3. Running the Application from Source

With the virtual environment activated and dependencies installed, you can run the application directly:

```bash
python main.py
```

This will launch the application window. You can then select a `log.xlsx` file and run the analysis.

## 4. Packaging into a Single .exe File

This project uses **PyInstaller** to bundle the application and all its dependencies into a single executable file (`.exe`) for easy distribution on Windows.

### The Command

To create the executable, run the following command from the project's root directory in your terminal:

```bash
pyinstaller --name "VibrationAnalysis" --onefile --windowed --icon="NONE" main.py
```

### Command Explained:

-   `--name "VibrationAnalysis"`: Sets the name of the final executable file (`VibrationAnalysis.exe`).
-   `--onefile`: Bundles everything into a single `.exe` file. When the user runs it, it will create a temporary folder to unpack the dependencies, which is automatically cleaned up when the app closes. This is the simplest way to distribute the application.
-   `--windowed`: Prevents a console window from appearing in the background when the application is run. This is essential for GUI applications.
-   `--icon="NONE"`: If you have a `.ico` file for your application, you can specify its path here (e.g., `--icon="path/to/your/icon.ico"`). Using "NONE" results in the default executable icon.
-   `main.py`: The entry-point to your application.

### After Running the Command:

-   PyInstaller will create a few folders (`build`, `dist`) and a `.spec` file (`VibrationAnalysis.spec`).
-   The final executable file will be located in the **`dist`** folder.
-   You can now share **`VibrationAnalysis.exe`** from the `dist` folder with other students. They will not need to install Python or any packages to run it.

### Important Notes on Dependencies:

-   This project relies on libraries like `pandas` and `numpy`, which have complex dependencies. PyInstaller is generally good at finding them, but if you encounter `ModuleNotFoundError` issues when running the `.exe`, you may need to edit the `.spec` file to explicitly include hidden imports. For this project's current dependencies, the command above should be sufficient.
