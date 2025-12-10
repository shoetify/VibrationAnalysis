import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None


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


COLUMN_ALIASES: dict[str, List[str]] = {
    "wind_speed": [
        "wind_speed",
        "wind speed",
        "wind speed (hz)",
        "wind_speed (hz)",
        "wind speed hz",
        "windspeed",
        "windspeed (hz)",
        "windspeedhz",
    ],
    "start_time": [
        "start_time",
        "start time",
        "start",
        "start (s)",
        "start_time (s)",
    ],
    "end_time": [
        "end_time",
        "end time",
        "end",
        "end (s)",
        "end_time (s)",
    ],
    "file_name": [
        "file_name",
        "file name",
        "file",
        "filename",
    ],
}


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
        return None

    df = pd.read_excel(path)
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
        rows.append(
            {
                "row_number": idx + 2,  # excel row number (account for header)
                "wind_speed": row[resolved["wind_speed"]],
                "start_time": row[resolved["start_time"]],
                "end_time": row[resolved["end_time"]],
                "file_name": row[resolved["file_name"]],
            }
        )
    return rows


def read_log(log_path: Path | str) -> Dict[Any, Dict[Any, List[Any]]]:
    """
    Read the log.xlsx file and build a nested mapping.

    Returns:
        Dict where each key is a file_name and the value is a dict mapping
        wind_speed -> [start_time, end_time].
    """
    path = Path(log_path)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    rows = _load_rows_with_pandas(path)
    if pd is None or rows is None:
        raise ImportError("Please install 'pandas' to read Excel log files.")

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
        start_time = _normalize_text(start_time)
        end_time = _normalize_text(end_time)

        file_bucket = result.setdefault(file_name, {})
        file_bucket[wind_speed] = [start_time, end_time]

    return result


def load_signal_data(
    file_key: str, data_structure: Sequence[int]
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], Path]:
    """
    Load a txt file by file_key and map its columns according to data_structure.

    data_structure defines the semantic meaning of each column index:
        1 -> time, 2 -> displacement, 3 -> acceleration, 0 -> ignore.
    """
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
    displacements: list[np.ndarray] = []
    accelerations: list[np.ndarray] = []

    for col_idx, marker in enumerate(data_structure):
        column = data[:, col_idx]
        if marker == 0:
            continue
        if marker == 1:
            time = column
        elif marker == 2:
            displacements.append(column)
        elif marker == 3:
            accelerations.append(column)
        else:
            raise ValueError(f"Unsupported DATA_STRUCTURE marker '{marker}' at position {col_idx}")

    if time is None:
        raise ValueError("No time column defined by DATA_STRUCTURE")

    return time, displacements, accelerations, candidate


def find_data_index(time: np.ndarray, t: float) -> int:
    """
    Return the index of the first element in `time` that occurs after `t`.

    Assumes `time` is sorted in ascending order. Raises ValueError if no
    element is greater than `t`.
    """
    idx = int(np.searchsorted(time, t, side="right"))
    if idx >= time.size:
        raise ValueError(f"Time {t} exceeds available range (max {time.max()})")
    return idx


def load_log_data() -> Dict[Any, Dict[Any, List[Any]]]:
    log_path = find_log_file()
    log_data = read_log(log_path)
    print(f"Loaded {log_path}")
    return log_data


def compute_fft(
    signal: Sequence[float],
    sample_rate: float,
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
    loaded_signals: Dict[str, List[float]], file_name: str
) -> Path:
    """
    Append measurement summary rows to output.csv (created if missing).

    Columns: File Name, Wind Speed, Top 10% Peak Value,
    Top 10% Bottom Value, Amplitude.
    """
    output_path = Path(__file__).resolve().parent / "output.csv"
    headers = [
        "File Name",
        "Wind Speed",
        "Top 10% Peak Value",
        "Top 10% Bottom Value",
        "Amplitude",
    ]

    rows = []
    for wind_speed, values in loaded_signals.items():
        if len(values) != 2:
            raise ValueError(
                f"Expected two values (top10_peak, bottom10_bottom) for '{wind_speed}'"
            )
        top_peak, bottom_bottom = values
        amplitude = top_peak - bottom_bottom
        rows.append(
            [file_name, wind_speed, float(top_peak), float(bottom_bottom), amplitude]
        )

    write_header = not output_path.exists()
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerows(rows)

    return output_path
