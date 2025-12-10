import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import util

SAMPLING_FREQUENCY = 333.3333
CUTOFF_FREQUENCY_HZ = 0.5
DATA_STRUCTURE = [1, 2, 3, 3, 3, 0, 0, 0, 0]


def high_pass_filter(signal: np.ndarray, cutoff_hz: float, sample_rate: float, order: int = 6) -> np.ndarray:
    nyquist = sample_rate / 2.0
    b, a = butter(order, cutoff_hz / nyquist, btype="highpass")
    return filtfilt(b, a, np.asarray(signal, dtype=float))


def acceleration_to_displacement(acceleration: np.ndarray) -> np.ndarray:
    """Convert acceleration data to displacement using chained high-pass filters and integration."""
    filtered_acc = high_pass_filter(acceleration, CUTOFF_FREQUENCY_HZ, SAMPLING_FREQUENCY)
    dt = 1.0 / SAMPLING_FREQUENCY

    speed = np.cumsum(filtered_acc) * dt
    filtered_speed = high_pass_filter(speed, CUTOFF_FREQUENCY_HZ, SAMPLING_FREQUENCY)

    displacement = np.cumsum(filtered_speed) * dt
    displacement_mm = displacement * 1000.0  # convert m to mm to match displacement units
    return high_pass_filter(displacement_mm, CUTOFF_FREQUENCY_HZ, SAMPLING_FREQUENCY)


def analyze_displacement(signal: np.ndarray, start_index: int, end_index: int) -> tuple[float, float]:
    """Compute peak and bottom metrics for the specified segment of a displacement signal."""
    segment = np.asarray(signal[start_index:end_index], dtype=float)
    if segment.size == 0:
        raise ValueError("Selected signal segment is empty.")

    freq, mag, _ = util.compute_fft(segment, SAMPLING_FREQUENCY)
    freq_1_index = util.find_data_index(np.asarray(freq, dtype=float), 1)
    if (freq_1_index <= 0) or (freq_1_index >= len(freq)):
        raise ValueError("Issue found: FFT result abnormal")

    freq = freq[freq_1_index:]
    mag = mag[freq_1_index:]
    mag_max_index = int(np.argmax(mag))
    peak_freq = freq[mag_max_index]
    peaks_width = max(1, int((1 / peak_freq) * 0.9 * SAMPLING_FREQUENCY))

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


def main() -> dict[str, dict[str, list[float]]]:
    log_data = util.load_log_data()
    all_results: dict[str, dict[str, list[float]]] = {}

    for file_key in log_data:  # For every file inside the log
        time, displacement_columns, acceleration_columns, path = util.load_signal_data(
            file_key, DATA_STRUCTURE
        )
        print(f"Loaded {path} for key '{file_key}'")

        displacement_series: list[tuple[str, int, np.ndarray]] = []
        for col_number, displacement in displacement_columns:
            displacement_series.append(("disp", col_number, displacement))
        for col_number, acceleration in acceleration_columns:
            displacement_series.append(
                ("acce", col_number, acceleration_to_displacement(acceleration))
            )

        if not displacement_series:
            raise ValueError("No displacement or acceleration columns found based on DATA_STRUCTURE.")

        column_results: dict[tuple[str, int], dict[str, list[float]]] = {}
        logs = log_data[file_key]
        for log in logs:  # For every wind speed inside the files
            start_index = util.find_data_index(time, logs[log][0])
            end_index = util.find_data_index(time, logs[log][1])

            for kind, col_number, signal in displacement_series:
                avg_top_10_peaks, avg_bottom_10_bottoms = analyze_displacement(
                    signal, start_index, end_index
                )
                key = (kind, col_number)
                per_column = column_results.setdefault(key, {})
                per_column[log] = [avg_top_10_peaks, avg_bottom_10_bottoms]

        for (kind, col_number), results in column_results.items():
            output_filename = (
                f"output_disp_{col_number}.csv" if kind == "disp" else f"output_acce_{col_number}.csv"
            )
            util.export_result_to_csv(results, file_key, output_filename)
            all_results[f"{file_key}_{kind}_{col_number}"] = results

    return all_results


if __name__ == "__main__":
    main()
