# -*- coding: utf-8 -*-
"""
This file contains the core analysis logic for the vibration analysis application.
It is designed to be independent of the user interface.
"""
import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Iterator

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    import pandas as pd
except ImportError:
    pd = None

# --- Constants ---
DEFAULT_SAMPLING_FREQUENCY = 333.3333
# DEFAULT_CUTOFF_FREQUENCY_HZ was removed per user request. 
# A hardcoded value (e.g. 0.5) is used internally where filtering is strictly required.
DEFAULT_DATA_STRUCTURE = [1, 2]
DEFAULT_DIAMETER = 60
DEFAULT_NATURAL_FREQUENCY = 8

COLUMN_ALIASES: dict[str, List[str]] = {
    "wind_speed": [
        "wind_speed", "wind speed", "wind speed (hz)", "wind_speed (hz)",
        "wind speed hz", "windspeed", "windspeed (hz)", "windspeedhz",
    ],
    "start_time": ["start_time", "start time", "start", "start (s)", "start_time (s)"],
    "end_time": ["end_time", "end time", "end", "end (s)", "end_time (s)"],
    "file_name": ["file_name", "file name", "file", "filename"],
}


# --- Utility Functions (from util.py) ---

def _is_missing(value: Any) -> bool:
    """Return True when the cell is considered empty."""
    if pd is not None:
        try:
            return bool(pd.isna(value))
        except Exception:
            pass
    if value is None:
        return True
    return isinstance(value, str) and not value.strip()


def _normalize_text(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    return value


def _normalize_column_name(value: str) -> str:
    """Lowercase and strip non-alphanumeric characters to match flexible headers."""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def find_log_file(base_dir: Optional[Path] = None) -> Path:
    """
    Locate the most recently modified Excel file whose name ends with 'log.xlsx'.
    Searches recursively starting from base_dir (or the current working directory).
    """
    search_root = base_dir or Path.cwd()
    candidates = [p for p in search_root.rglob("*log.xlsx") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No log.xlsx file found under {search_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_rows_with_pandas(path: Path) -> Optional[List[Mapping[str, Any]]]:
    if pd is None:
        raise ImportError("Pandas is required to read Excel files. Please install it.")

    df = pd.read_excel(path, engine='openpyxl')
    normalized_lookup = {_normalize_column_name(str(col)): col for col in df.columns}

    def resolve_column(key: str) -> str:
        for candidate in COLUMN_ALIASES[key]:
            normalized = _normalize_column_name(candidate)
            if normalized in normalized_lookup:
                return normalized_lookup[normalized]
        raise KeyError(key)

    resolved: Dict[str, str] = {}
    missing: List[str] = []
    for required in ("wind_speed", "start_time", "end_time", "file_name"):
        try:
            resolved[required] = resolve_column(required)
        except KeyError:
            missing.append(required)

    if missing:
        raise ValueError(f"Missing expected columns in log: {', '.join(missing)}")

    rows: List[Mapping[str, Any]] = []
    for idx, row in df.iterrows():
        rows.append({
            "row_number": idx + 2,
            "wind_speed": row[resolved["wind_speed"]],
            "start_time": row[resolved["start_time"]],
            "end_time": row[resolved["end_time"]],
            "file_name": row[resolved["file_name"]],
        })
    return rows


def read_log(log_path: Path | str) -> Dict[Any, Dict[Any, List[Any]]]:
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    rows = _load_rows_with_pandas(path)
    if rows is None:
        raise ImportError("Could not load rows. 'pandas' and 'openpyxl' are required.")

    result: Dict[Any, Dict[Any, List[Any]]] = {}
    last_file_name: Optional[Any] = None

    for row in rows:
        row_number = row["row_number"]
        wind_speed = row["wind_speed"]
        start_time = row["start_time"]
        end_time = row["end_time"]
        file_name = row["file_name"]

        if _is_missing(wind_speed):
            raise ValueError(f"Missing wind_speed in row {row_number}")
        if _is_missing(start_time):
            raise ValueError(f"Missing start_time in row {row_number}")
        if _is_missing(end_time):
            raise ValueError(f"Missing end_time in row {row_number}")

        if _is_missing(file_name):
            if last_file_name is None:
                raise ValueError("The first row must provide a file_name.")
            file_name = last_file_name
        else:
            file_name = _normalize_text(file_name)
            last_file_name = file_name

        wind_speed = _normalize_text(wind_speed)
        start_time = float(_normalize_text(start_time))
        end_time = float(_normalize_text(end_time))

        file_bucket = result.setdefault(file_name, {})
        file_bucket[wind_speed] = [start_time, end_time]

    return result


def load_signal_data(
    file_key: str, data_structure: Sequence[int]
) -> tuple[np.ndarray, list[tuple[int, np.ndarray]], list[tuple[int, np.ndarray]], Path]:
    file_name = file_key if file_key.endswith(".txt") else f"{file_key}.txt"
    candidate = Path(file_name)
    if not candidate.is_file():
        matches = list(Path.cwd().rglob(file_name))
        if not matches:
            raise FileNotFoundError(f"Could not find data file for '{file_name}'")
        candidate = matches[0]

    data = np.loadtxt(candidate)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D data from {candidate}, got shape {data.shape}")
    if data.shape[1] < len(data_structure):
        raise ValueError(
            f"DATA_STRUCTURE expects {len(data_structure)} columns but {candidate} has {data.shape[1]}"
        )

    time: Optional[np.ndarray] = None
    displacements: list[tuple[int, np.ndarray]] = []
    accelerations: list[tuple[int, np.ndarray]] = []

    for col_idx, marker in enumerate(data_structure):
        column = data[:, col_idx]
        if marker == 0:
            continue
        if marker == 1:
            time = column
        elif marker == 2:
            displacements.append((col_idx + 1, column))
        elif marker == 3:
            accelerations.append((col_idx + 1, column))
        else:
            raise ValueError(f"Unsupported DATA_STRUCTURE marker '{marker}' at position {col_idx}")

    if time is None:
        raise ValueError("No time column defined by DATA_STRUCTURE")

    return time, displacements, accelerations, candidate


def find_data_index(time: np.ndarray, t: float) -> int:
    idx = int(np.searchsorted(time, t, side="right"))
    if idx >= time.size:
        raise ValueError(f"Time {t} exceeds available range (max {time.max()})")
    return idx


def compute_fft(
    signal: Sequence[float], sample_rate: float,
) -> Tuple[Sequence[float], Sequence[float], float]:
    array = np.asarray(signal, dtype=float)
    if array.size == 0:
        raise ValueError("Cannot compute FFT of an empty signal.")

    fluctuations = array - np.mean(array)
    fft_result = np.fft.rfft(fluctuations)
    magnitude = np.abs(fft_result) ** 2
    frequency = np.fft.rfftfreq(array.size, d=1.0 / sample_rate)

    resolution = frequency[1] - frequency[0] if frequency.size > 1 else 0.0
    return frequency.tolist(), magnitude.tolist(), float(resolution)


def export_result_to_csv(
    loaded_signals: Dict[str, List[float]], 
    file_name: str, 
    output_filename: str,
    diameter: float,
    natural_frequency: float
) -> Path:
    output_path = Path.cwd() / output_filename
    headers = [
        "File Name", 
        "Wind Speed", 
        "Top 10% Peak Value", 
        "Top 10% Bottom Value", 
        "Amplitude",
        "A*",
        "Ur"
    ]

    rows = []
    for wind_speed, values in loaded_signals.items():
        if len(values) != 2:
            raise ValueError(f"Expected two values for '{wind_speed}'")
        top_peak, bottom_bottom = values
        amplitude = float(top_peak) - float(bottom_bottom)
        
        # Parse Wind Speed for calculation
        try:
            ws_val = float(str(wind_speed).replace("m/s", "").strip())
        except (ValueError, TypeError):
            # Fallback if wind_speed is not numeric (though expected to be numeric per logic)
            ws_val = 0.0

        # Calculations
        # A* = Amplitude / diameter / 2
        # Ur = Wind Speed / diameter / natural_frequency
        ws_val = ws_val * 0.1726 - 0.06956
        a_star = amplitude / diameter / 2.0 if diameter != 0 else 0.0
        ur = ws_val / diameter / natural_frequency * 1000.0 if (diameter * natural_frequency) != 0 else 0.0

        rows.append([
            file_name, 
            wind_speed, 
            float(top_peak), 
            float(bottom_bottom), 
            amplitude,
            a_star,
            ur
        ])

    write_header = not output_path.exists()
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerows(rows)

    return output_path


# --- Core Analysis Functions (from main.py) ---

def high_pass_filter(signal: np.ndarray, cutoff_hz: float, sample_rate: float, order: int = 6) -> np.ndarray:
    nyquist = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyquist, btype="highpass")
    return filtfilt(b, a, np.asarray(signal, dtype=float))


def acceleration_to_displacement(
    acceleration: np.ndarray,
    sampling_frequency: float
) -> np.ndarray:
    # Use a hardcoded cutoff since the parameter was removed from UI/logic
    cutoff_frequency_hz = 0.5 
    
    filtered_acc = high_pass_filter(acceleration, cutoff_frequency_hz, sampling_frequency)
    dt = 1.0 / sampling_frequency
    speed = np.cumsum(filtered_acc) * dt
    filtered_speed = high_pass_filter(speed, cutoff_frequency_hz, sampling_frequency)
    displacement = np.cumsum(filtered_speed) * dt
    displacement_mm = displacement * 1000.0
    return high_pass_filter(displacement_mm, cutoff_frequency_hz, sampling_frequency)


def analyze_displacement(
    signal: np.ndarray,
    start_index: int,
    end_index: int,
    sampling_frequency: float
) -> tuple[float, float]:
    segment = np.asarray(signal[start_index:end_index], dtype=float)
    if segment.size == 0:
        raise ValueError("Selected signal segment is empty.")

    freq, mag, _ = compute_fft(segment, sampling_frequency)
    try:
        freq_1_index = find_data_index(np.asarray(freq, dtype=float), 1)
    except ValueError:
        freq_1_index = 1 

    if (freq_1_index <= 0) or (freq_1_index >= len(freq)):
        raise ValueError("Issue found: FFT result abnormal")

    freq = freq[freq_1_index:]
    mag = mag[freq_1_index:]
    mag_max_index = int(np.argmax(mag))
    peak_freq = freq[mag_max_index]
    peaks_width = max(1, int((1 / peak_freq) * 0.9 * sampling_frequency))

    peaks_indices, _ = find_peaks(segment, distance=peaks_width)
    bottoms_indices, _ = find_peaks(-segment, distance=peaks_width)
    if peaks_indices.size == 0 or bottoms_indices.size == 0:
        raise ValueError("No peaks detected in the selected segment.")

    peaks_values = segment[peaks_indices]
    bottoms_values = segment[bottoms_indices]

    sorted_peaks = np.sort(peaks_values)
    sorted_bottoms = np.sort(bottoms_values)

    top_n = max(1, int(0.1 * len(sorted_peaks)))
    bottom_n = max(1, int(0.1 * len(sorted_bottoms)))

    top_10_percent_peaks = sorted_peaks[-top_n:]
    bottom_10_percent_bottoms = sorted_bottoms[:bottom_n]

    avg_top_10_peaks = np.mean(top_10_percent_peaks)
    avg_bottom_10_bottoms = np.mean(bottom_10_percent_bottoms)

    return avg_top_10_peaks, avg_bottom_10_bottoms


# --- Main Orchestration Function ---

def run_analysis(
    log_file_path: Optional[str],
    sampling_frequency: float,
    data_structure: List[int],
    diameter: float = DEFAULT_DIAMETER,
    natural_frequency: float = DEFAULT_NATURAL_FREQUENCY,
) -> Iterator[str]:
    """
    Main analysis workflow. Finds and processes data based on a log file.
    Yields status messages for the UI.
    """
    if pd is None:
        raise ImportError("This application requires 'pandas' and 'openpyxl'. Please install them.")

    try:
        if log_file_path:
            log_path = Path(log_file_path)
            if not log_path.exists():
                raise FileNotFoundError(f"Specified log file not found: {log_path}")
        else:
            log_path = find_log_file()
        
        yield f"Loading log file: {log_path.name}"
        log_data = read_log(log_path)
        yield f"Successfully loaded log file."
    except (FileNotFoundError, ValueError, ImportError) as e:
        yield f"ERROR: {e}"
        return

    file_keys = list(log_data.keys())
    total_files = len(file_keys)

    for i, file_key in enumerate(file_keys):
        yield f"Processing file {i + 1}/{total_files}: '{file_key}'"
        try:
            time, disp_cols, acc_cols, path = load_signal_data(file_key, data_structure)
            yield f"  - Loaded signal data from {path.name}"

            displacement_series: list[tuple[str, int, np.ndarray]] = []
            for col_num, disp in disp_cols:
                displacement_series.append(("disp", col_num, disp))
            for col_num, acc in acc_cols:
                yield f"  - Converting acceleration to displacement for column {col_num}..."
                # Updated call: no cutoff_frequency_hz passed
                disp_from_acc = acceleration_to_displacement(acc, sampling_frequency)
                displacement_series.append(("acce", col_num, disp_from_acc))

            if not displacement_series:
                yield f"  - WARNING: No displacement or acceleration columns found for '{file_key}'"
                continue

            column_results: dict[tuple[str, int], dict[str, list[float]]] = {}
            logs = log_data[file_key]
            
            total_logs = len(logs)
            for j, log_key in enumerate(logs):
                yield f"  - Analyzing segment {j + 1}/{total_logs}: '{log_key}'"
                start_time, end_time = logs[log_key]
                start_index = find_data_index(time, start_time)
                end_index = find_data_index(time, end_time)

                for kind, col_num, signal in displacement_series:
                    avg_top, avg_bottom = analyze_displacement(signal, start_index, end_index, sampling_frequency)
                    key = (kind, col_num)
                    per_column = column_results.setdefault(key, {})
                    per_column[log_key] = [avg_top, avg_bottom]

            yield f"  - Exporting results for '{file_key}'..."
            for (kind, col_num), results in column_results.items():
                output_filename = f"output_{file_key}_{kind}_{col_num}.csv"
                # Pass diameter and natural_frequency to export
                export_result_to_csv(results, file_key, output_filename, diameter, natural_frequency)
                yield f"    - Saved results to {output_filename}"

        except (FileNotFoundError, ValueError, IndexError) as e:
            yield f"  - ERROR processing '{file_key}': {e}"
            continue
    
    yield "Analysis complete."