import numpy as np
from scipy.signal import find_peaks

import util


def main() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    log_data = util.load_log_data()

    loaded_signals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for file_key in log_data:
        time, signal, path = util.load_signal_data(file_key)
        print(f"Loaded {path} for key '{file_key}'")
        logs = log_data[file_key]
        for log in logs:
            start_index = util.find_data_index(time, logs[log][0])
            end_index = util.find_data_index(time, logs[log][1])

            peaks_indices, _ = find_peaks(signal[start_index:end_index])
            peak_times = util.find_time_index_from_peaks(
                peaks_indices, time[start_index:end_index], signal[start_index:end_index]
            )
            peak_signals = signal[start_index:end_index][peaks_indices]

            # Save in the same two-column, tab-delimited format as the raw data.
            np.savetxt(
                f"{log}.txt",
                np.column_stack((peak_times, peak_signals)),
                delimiter="\t",
                fmt="%.6f\t%.6f",
            )

            



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
