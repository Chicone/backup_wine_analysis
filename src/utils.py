import numpy as np
import re
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt

def collapse_lists(d):
    for key, value in d.items():
        if isinstance(value, list) and all(isinstance(sublist, list) for sublist in value):
            d[key] = [item for sublist in value for item in sublist]
        elif not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' is not a list or a list of lists.")
    return d

def concatenate_dict_values(d1, d2):
    result = {}
    all_keys = set(d1.keys()).union(set(d2.keys()))
    for key in all_keys:
        result[key] = d1.get(key, []) + d2.get(key, [])
    return result


def find_first_and_last_position(s):
    # Define the regex pattern to find numbers
    pattern = r'\d+'

    # Find the first occurrence
    first_match = re.search(pattern, s)
    first_pos = first_match.start() if first_match else -1

    # Find the last occurrence
    last_match = None
    for match in re.finditer(pattern, s):
        last_match = match
    last_pos = last_match.start() if last_match else -1

    return first_pos, last_pos

def normalize_dict(data, scaler='standard'):
    # Normalise dictionary values
    keys = list(data.keys())
    values = np.array(list(data.values()))
    values_scaled = None
    if scaler == 'standard':
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
    elif scaler == 'minmax':
        scaler = MinMaxScaler()
        values_scaled = scaler.fit_transform(values)
    elif scaler == 'robust':
        scaler = RobustScaler()
        values_scaled = scaler.fit_transform(values)
    norm_data = {key: values_scaled[idx, :].tolist() for idx, key in enumerate(keys)}

    return norm_data


def smooth_remove_peak(signal, peak_idx, window_size=5):
    """
    Smoothly removes a peak from a signal using interpolation.

    Parameters:
    signal (np.ndarray): The input signal.
    peak_idx (int): The index of the peak to remove.
    window_size (int): The size of the window around the peak to use for interpolation.

    Returns:
    np.ndarray: The signal with the peak removed.
    """
    left_idx = max(0, peak_idx - window_size)
    right_idx = min(len(signal), peak_idx + window_size + 1)

    # Interpolate over the region including the peak
    x = np.array([left_idx, right_idx])
    y = signal[x]
    interpolated_values = np.interp(np.arange(left_idx, right_idx), x, y)

    # Replace the values in the signal with the interpolated values
    signal_smooth = np.copy(signal)
    signal_smooth[left_idx:right_idx] = interpolated_values

    return signal_smooth


def plot_data_from_dict(data_dict, title):
    """
    Plots all lists of data contained in a dictionary.

    Parameters:
    data_dict (dict): A dictionary with labels as keys and lists of values as values.
    """
    plt.figure(figsize=(10, 6))

    for label, data in data_dict.items():
        plt.plot(data, label=label)

    plt.xlabel('Retention time')
    plt.ylabel('Intensity')
    plt.title(title)
    # plt.legend()
    plt.grid(True)
    plt.show()
