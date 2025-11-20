from pathlib import Path

import numpy as np
from scipy.signal import find_peaks
from util import main as util_main


def load_signal_file(file_key: str) -> tuple[np.ndarray, np.ndarray, Path]:
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


def main() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    log_data = util_main()

    loaded_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for file_key in log_data:
        time, signal, path = load_signal_file(file_key)
        print(f"Loaded {path} for key '{file_key}'")
        




if __name__ == "__main__":
    main()


# FILE_NAME = 'tri_23_04_2025_12.53.04 PM.txt'

# # Load the data from the text file
# data = np.loadtxt(FILE_NAME)

# # Separate the columns
# time = data[:, 0]            # First column: Time
# signal = data[:, 1]          # Second column: Vibration signal

# # Find the indices of the peaks
# peaks_indices, _ = find_peaks(signal)

# # Find the indices of the bottoms (by finding peaks of the inverted signal)
# bottoms_indices, _ = find_peaks(-signal)

# # Extract peak and bottom values
# peaks_values = np.array(signal[peaks_indices].tolist())
# bottoms_values = np.array(signal[bottoms_indices].tolist())

# # Sort the values
# sorted_peaks = np.sort(peaks_values)
# sorted_bottoms = np.sort(bottoms_values)

# # Calculate 10% count
# top_n = max(1, int(0.1 * len(sorted_peaks)))
# bottom_n = max(1, int(0.1 * len(sorted_bottoms)))

# # Get top and bottom 10% values
# top_10_percent_peaks = sorted_peaks[-top_n:]
# bottom_10_percent_bottoms = sorted_bottoms[:bottom_n]

# # Calculate averages
# avg_top_10_peaks = np.mean(top_10_percent_peaks)
# avg_bottom_10_bottoms = np.mean(top_10_percent_bottoms)

# print(f"Average of Top 10% Peak Values: {avg_top_10_peaks}")
# print(f"Average of Bottom 10% Bottom Values: {avg_bottom_10_bottoms}")
