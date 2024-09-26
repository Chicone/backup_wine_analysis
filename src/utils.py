import numpy as np
import re
import os

import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter


def collapse_lists(d):
    """
    Collapse nested lists into a single list for each dictionary entry.

    This function is useful when the values in the dictionary are lists of lists, and you want to flatten them into a
    single list.

    Parameters
    ----------
    d : dict
        The dictionary with list or list of lists as values.

    Returns
    -------
    dict
        The dictionary with the lists collapsed into single lists.

    Raises
    ------
    ValueError
        If a value in the dictionary is not a list or a list of lists.
    """
    for key, value in d.items():
        # Check if the value is a list of lists
        if isinstance(value, list) and all(isinstance(sublist, list) for sublist in value):
            # Flatten the list of lists into a single list
            d[key] = [item for sublist in value for item in sublist]
        elif not isinstance(value, list):
            raise ValueError(f"Value for key '{key}' is not a list or a list of lists.")

    return d


def concatenate_dict_values(d1, d2):
    """
    Concatenate the values of two dictionaries.

    If a key exists in both dictionaries, their values are concatenated. If a key exists in only one dictionary, its
    value is included as is.

    Parameters
    ----------
    d1 : dict
        The first dictionary.
    d2 : dict
        The second dictionary.

    Returns
    -------
    dict
        A new dictionary with concatenated values.
    """
    result = {}

    # Union of keys from both dictionaries
    all_keys = set(d1.keys()).union(set(d2.keys()))

    for key in all_keys:
        # Concatenate values for each key
        result[key] = d1.get(key, []) + d2.get(key, [])

    return result


def find_first_and_last_position(s):
    """
    Find the positions of the first and last numbers in a string.

    This function searches for numbers in the string and returns the starting positions of the first and last occurrences.

    Parameters
    ----------
    s : str
        The input string to search for numbers.

    Returns
    -------
    tuple
        A tuple containing the start positions of the first and last numbers. If no numbers are found, returns (-1, -1).
    """
    # Define the regex pattern to find numbers
    pattern = r'\d+'

    # Find the first occurrence of the pattern
    first_match = re.search(pattern, s)
    first_pos = first_match.start() if first_match else -1

    # Find the last occurrence of the pattern
    last_match = None
    for match in re.finditer(pattern, s):
        last_match = match
    last_pos = last_match.start() if last_match else -1

    return first_pos, last_pos


def load_chromatograms(file_path, normalize=False):
    """
    Load chromatogram data from a given file path.

    Parameters
    ----------
    file_path : str
        The path to the file containing the chromatogram data.
    normalize : bool, optional
        Whether to normalize the chromatogram data. Default is False.

    Returns
    -------
    dict
        A dictionary containing the loaded chromatogram data.
    """
    from wine_analysis import WineAnalysis

    # Expand the user path to the full file path
    analysis = WineAnalysis(os.path.expanduser(file_path))

    # Return the data loaded by the WineAnalysis instance
    return analysis.data_loader.data


def min_max_normalize(data, min_range=0, max_range=1):
    """
    Perform Min-Max normalization on data.

    This function scales the input data to a specified range [min_range, max_range] using Min-Max normalization.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be normalized.
    min_range : float, optional
        The minimum value of the desired range. Default is 0.
    max_range : float, optional
        The maximum value of the desired range. Default is 1.

    Returns
    -------
    numpy.ndarray
        The normalized data within the specified range.

    Notes
    -----
    - Min-Max normalization rescales the data to a fixed range, typically [0, 1], based on the minimum and maximum values in the data.
    - This function normalizes the data across all values (i.e., sample-wise normalization) in the input array.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = min_range + ((data - min_val) * (max_range - min_range) / (max_val - min_val))

    return normalized_data


def normalize_amplitude_minmax(signal):
    """
    Normalize the signal using min-max normalization.

    This method scales the input signal so that its values range between 0 and 1. The minimum value
    of the signal becomes 0, and the maximum value becomes 1.

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be normalized.

    Returns
    -------
    numpy.ndarray
        The normalized signal, where the values are scaled to the range [0, 1].
    """
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)

    return normalized_signal


def normalize_amplitude_zscore(signal):
    """
    Normalize the signal using standard normalization (z-score normalization).

    Parameters
    ----------
    signal : numpy.ndarray
        The input signal to be normalized.

    Returns
    -------
    numpy.ndarray
        The normalized signal.
    """
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    normalized_signal = (signal - mean_val) / std_val

    return normalized_signal


def normalize_data(data, scaler='standard', feature_range=(0,1)):
    """
    Normalize data which can be either a dictionary or an array.

    This function normalizes each feature (column) across all samples (rows) using the specified scaler.
    It ensures that the normalization is consistent for each feature across the dataset.

    Parameters
    ----------
    data : dict or np.ndarray
        The data to be normalized. If dict, values should be arrays. If np.ndarray, it should be a 2D array.
    scaler : str, optional
        The type of scaler to use for normalization. Options are 'standard', 'minmax', 'robust'. Default is 'standard'.

    Returns
    -------
    dict or np.ndarray
        The normalized data. If input was a dict, returns a dict with the same keys and normalized values.
        If input was an array, returns a normalized array.
    """
    values = None

    if isinstance(data, dict):
        keys = list(data.keys())
        values = np.array(list(data.values()))
    elif isinstance(data, np.ndarray):
        values = data
    else:
        raise ValueError("Input data must be either a dictionary or a numpy array.")

    if scaler == 'standard':
        scaler = StandardScaler()
    elif scaler == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif scaler == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError("Unsupported scaler type. Choose from 'standard', 'minmax', or 'robust'.")

    values_scaled = scaler.fit_transform(values)

    if isinstance(data, dict):
        norm_data = {key: values_scaled[idx, :].tolist() for idx, key in enumerate(keys)}
        return norm_data, scaler
    elif isinstance(data, np.ndarray):
        return values_scaled, scaler


def normalize_dict(data, scaler='standard'):
    """
    Normalize the values in a dictionary across samples.

    This function normalizes each feature (column) in the dictionary across all samples (rows) using the specified scaler.
    The dictionary keys remain the same, while the values are normalized.

    Parameters
    ----------
    data : dict
        The input dictionary to be normalized. The values should be lists or numpy arrays representing the features.
    scaler : str, optional
        The type of scaler to use for normalization. Options are 'standard' (Z-score normalization),
        'minmax' (Min-Max scaling), and 'robust' (RobustScaler). Default is 'standard'.

    Returns
    -------
    dict
        A dictionary with the same keys and normalized values.

    Notes
    -----
    - Normalization is performed feature-wise across all samples in the dictionary, meaning each feature (value associated with a key)
      is normalized across all samples (rows) in the dataset.
    """
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

def filter_dict_by_keys(original_dict, keys_to_keep):
    return {key: original_dict[key] for key in keys_to_keep if key in original_dict}

def filter_dict_by_first_letter(original_dict, letters):
    return {key: original_dict[key] for key in original_dict if key[0] in letters}



def plot_data_from_dict(data_dict, title, legend=False):
    """
    Plots all lists of data contained in a dictionary.

    Parameters
    ----------
    data_dict : dict
        A dictionary with labels as keys and lists of values as values.
    title : str
        The title of the plot.
    legend : bool, optional
        Whether to display the legend. Default is False.

    Raises
    ------
    ValueError
        If the input dictionary is empty.
    """
    from itertools import cycle

    if not data_dict:
        raise ValueError("The input data dictionary is empty.")

    plt.figure(figsize=(10, 6))

    # Create a color cycle to avoid repeating colors
    colors = plt.cm.get_cmap('tab10', len(data_dict)).colors
    color_cycle = cycle(colors)

    for label, data in data_dict.items():
        plt.plot(data, label=label, color=next(color_cycle))

    plt.xlabel('Retention time')
    plt.ylabel('Intensity')
    plt.title(title)
    if legend:
        plt.legend()
    plt.grid(True)
    plt.show()


def plot_lag(lags_loc, lags, title='Lag as a function of retention time'):
    """
    Plot the lag as a function of retention time.

    Parameters
    ----------
    lags_loc : numpy.ndarray
        Locations of the lags.
    lags : numpy.ndarray
        Lags for each datapoint.
    title : str, optional
        Title of the plot. Default is 'Lag as a function of retention time'.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lags_loc, lags, label='Lag')
    plt.xlabel('Retention time')
    plt.ylabel('Lag')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def remove_peak(signal, peak_idx, window_size=5):
    """
    Smoothly removes a peak from a signal using interpolation.

    Parameters
    ----------
    signal : np.ndarray
        The input signal.
    peak_idx : int
        The index of the peak to remove.
    window_size : int, optional
        The size of the window around the peak to use for interpolation. Default is 5.

    Returns
    -------
    np.ndarray
        The signal with the peak removed.
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
