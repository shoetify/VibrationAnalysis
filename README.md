# Vibration Analysis App User Guide

This guide explains how to use the Vibration Analysis application to process your experiment data.

## 1. Prepare the Log File

Before running the application, you need to prepare a log file (Excel format). You can refer to `example_log.xlsx` included with the app for the correct format.

The log file must contain **4 columns** in the following order:

1.  **Wind Speed (Motor Hz):** The frequency of the motor or wind speed setting.
2.  **Start Time (s):** The time shown on the screen when the cylinder starts to become stable at the specific wind speed.
3.  **End Time (s):** The time shown on the screen at the end of the experiment for that specific wind speed.
4.  **File Name:** The name of the raw data file.
    *    The program will automatically search for this file in the current folder and its subfolders.
    *    It will automatically append `.txt` to the file name you provide.
    *    **Note:** If a row uses the same data file as the previous row, you can leave this cell empty.

## 2. Load Your Data

1.  Open the application (`.exe`).
2.  Drag and drop your prepared **Log File** into the file box area of the app.
3.  Ensure your **Raw Data** files (the `.txt` files referenced in your log) are located in the same folder as the app or in a subfolder.

## 3. Adjust Configuration

Before running the analysis, adjust the configuration settings if necessary:

*   **Sampling Freq:** The sampling frequency of the LVDT used in your experiment.
*   **Diameter:** The diameter of the cylinder.
*   **Natural Freq:** The natural frequency of the vibration system, obtained from a free vibration test.
*   **Data Structure:** Defines how the columns in your raw data files are interpreted. Use the following codes:
    *   `1`: Time data column.
    *   `2`: Displacement data column.
    *   `3`: Acceleration data column (this will be automatically converted to displacement data before analysis).
    *   `0`: Ignore this column.

    *(Example: If your file has Time, Unused, and Acceleration, you would set the structure to identify column 1 as Time and column 3 as Acceleration).*

## 4. Run Analysis

1.  Once the files are loaded and settings are configured, click the **"Run Analysis"** button.
2.  The app will begin processing. You can check the log window within the app to see the progress and identify any errors that occur during analysis.
3.  The application will output one result data file for each processed data column.