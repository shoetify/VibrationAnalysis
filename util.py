import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

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


def load_signal_data(file_key: str) -> tuple[np.ndarray, np.ndarray, Path]:
    """Load a two-column txt file by file_key; columns map to (time, signal)."""
    file_name = file_key if file_key.endswith(".txt") else f"{file_key}.txt"
    candidate = Path(file_name)
    if not candidate.is_file():
        matches = list(Path.cwd().rglob(file_name))
        if not matches:
            raise FileNotFoundError(f"Could not find data file for '{file_name}'")
        candidate = matches[0]

    data = np.loadtxt(candidate)
    time = data[:, 0]
    signal = data[:, 1]
    return time, signal, candidate


def load_log_data() -> Dict[Any, Dict[Any, List[Any]]]:
    log_path = find_log_file()
    log_data = read_log(log_path)
    print(f"Loaded {log_path}")
    return log_data
