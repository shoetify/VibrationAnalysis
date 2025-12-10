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
    return high_pass_filter(displacement, CUTOFF_FREQUENCY_HZ, SAMPLING_FREQUENCY)


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


def main() -> dict[str, list[float]]:
    log_data = util.load_log_data()
    loaded_signals: dict[str, list[float]] = {}

    for file_key in log_data:  # For every file inside the log
        time, displacement_columns, acceleration_columns, path = util.load_signal_data(
            file_key, DATA_STRUCTURE
        )
        print(f"Loaded {path} for key '{file_key}'")

        displacement_series: list[tuple[str, np.ndarray]] = []
        for idx, displacement in enumerate(displacement_columns, start=1):
            displacement_series.append((f"disp_c{idx}", displacement))
        for idx, acceleration in enumerate(acceleration_columns, start=1):
            displacement_series.append((f"acc_c{idx}_disp", acceleration_to_displacement(acceleration)))

        if not displacement_series:
            raise ValueError("No displacement or acceleration columns found based on DATA_STRUCTURE.")

        logs = log_data[file_key]
        for log in logs:  # For every wind speed inside the files
            start_index = util.find_data_index(time, logs[log][0])
            end_index = util.find_data_index(time, logs[log][1])

            for label, signal in displacement_series:
                avg_top_10_peaks, avg_bottom_10_bottoms = analyze_displacement(
                    signal, start_index, end_index
                )
                result_key = log if len(displacement_series) == 1 else f"{log}_{label}"
                loaded_signals[result_key] = [avg_top_10_peaks, avg_bottom_10_bottoms]

        util.export_result_to_csv(loaded_signals, file_key)

    return loaded_signals


if __name__ == "__main__":
    main()
