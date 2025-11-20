import numpy as np
from scipy.signal import find_peaks
from util import main as util_main


if __name__ == "__main__":
    log_data = util_main()
 

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
# avg_bottom_10_bottoms = np.mean(bottom_10_percent_bottoms)

# print(f"Average of Top 10% Peak Values: {avg_top_10_peaks}")
# print(f"Average of Bottom 10% Bottom Values: {avg_bottom_10_bottoms}")

