import numpy as np
import re
import os

import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter


def load_chromatograms(file_path, normalize=False):
    from wine_analysis import WineAnalysis
    analysis = WineAnalysis(os.path.expanduser(file_path))
    return analysis.data_loader.data

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

def normalize_data(data, scaler='standard', feature_range=(0,1)):
    """
    Normalize data which can be either a dictionary or an array.

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


def min_max_normalize(data, min_range=0, max_range=1):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = min_range + ((data - min_val) * (max_range - min_range) / (max_val - min_val))
    return normalized_data


def remove_peak(signal, peak_idx, window_size=5):
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


def plot_data_from_dict(data_dict, title, legend=False):
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
    if legend:
        plt.legend()
    plt.grid(True)
    plt.show()



def calculate_lag_profile(c1, c2, segment_length, hop=1, sigma=20, lag_range=10, distance_metric='l2', init_min_dist=0.1):
    """
    Calculate the lag of a segment ahead for each datapoint in c2 against c1 using L1 or L2 distance.

    Parameters
    ----------
    c1 : numpy.ndarray
        Reference signal.
    c2 : numpy.ndarray
        Signal to be compared.
    segment_length : int
        Length of the segment ahead.
    hop : int, optional
        Step size for sliding the window. Default is 1.
    sigma : float, optional
        Standard deviation for Gaussian filter. Default is 20.
    lag_range : int, optional
        The range of lags to test on both sides. Default is 10.
    distance_metric : str, optional
        The distance metric to use for lag calculation ('L1' or 'L2'). Default is 'L2'.

    Returns
    -------
    numpy.ndarray
        Lags for each datapoint in c2.
    """

    def initial_global_alignment(c1, c2):
        """Compute initial global alignment using cross-correlation."""
        corr = correlate(c1, c2)
        lag = np.argmax(corr) - len(c2) + 1
        return lag

    # Initial global alignment
    initial_lag = initial_global_alignment(c1, c2)
    c2 = np.roll(c2, initial_lag)  # Apply the global shift

    lags = []
    lags_loc = []
    for i in range(0, len(c2) - segment_length + 1, hop):
        segment_c2 = c2[i:i + segment_length]
        start = max(0, i)
        end = min(len(c1), i + segment_length)
        segment_c1 = c1[start:end]

        # Normalize and apply Gaussian filter
        segment_c1_filtered = normalize_standard(gaussian_filter(segment_c1, sigma))
        segment_c2_filtered = normalize_standard(gaussian_filter(segment_c2, sigma))

        if distance_metric == 'corr':
            # Calculate cross-correlation between the segments
            corr = correlate(segment_c1_filtered, segment_c2_filtered)
            best_lag = np.argmax(corr) - len(segment_c2_filtered) + 1
            if np.abs(best_lag) > lag_range:
                continue
        else:
            # Initialize the minimum distance and best lag
            min_distance = init_min_dist
            best_lag = 0

            # Calculate distance for each possible lag within the specified range
            for lag in range(-lag_range, lag_range + 1):
                shifted_segment_c2 = np.roll(segment_c2_filtered, lag)
                if distance_metric == 'l2':
                    distance = np.sum((segment_c1_filtered[:len(shifted_segment_c2)] - shifted_segment_c2) ** 2)
                elif distance_metric == 'l1':
                    distance = np.mean(np.abs(segment_c1_filtered[:len(shifted_segment_c2)] - shifted_segment_c2))
                elif distance_metric == 'mse':
                    distance = np.mean((segment_c1_filtered[:len(shifted_segment_c2)] - shifted_segment_c2) ** 2)
                else:
                    raise ValueError("Invalid distance metric. Use 'l1' or 'l2' or 'mse'.")

                if distance < min_distance:
                    best_shifted = shifted_segment_c2
                    min_distance = distance
                    best_lag = lag

        lags.append(best_lag + initial_lag)
        lags_loc.append(i)
        # print(min_distance)

    # # Add one last point equal to the last one at 30000
    lags.append(lags[-1])
    lags_loc.append(30000)

    return np.array(lags_loc), np.array(lags)


def lag_profile_from_peaks(self, c1, c2, proximity_threshold, peak_prominence, nsegments):
    c1_peaks, _ = find_peaks(c1)
    #  remove peak for standard in reference
    index_to_remove = np.where(c1_peaks == 8918)[0]
    c1_peaks = np.delete(c1_peaks, index_to_remove)
    c2_peaks, _ = find_peaks(c2)
    c1_peaks, valid = self.ensure_peaks_in_segments(c1, c1_peaks, num_segments=nsegments)
    c2_peaks, valid = self.ensure_peaks_in_segments(c2, c2_peaks, num_segments=nsegments)
    if not valid:
        raise ValueError(
            "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks


    c2_aligned = c2.copy()
    initial_npeaks = len(c2_peaks)

    # Add zero to also scale the first part of the chromatogram
    c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
    c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
    prev_length_change = 0

    for i in range(1, initial_npeaks):
        c2p_prev = c2_peaks[i - 1]
        c2p = c2_peaks[i]
        c1p_prev = c2p_prev
        c1p = c1_peaks[np.argmin(np.abs(c1_peaks - c2p))]

        # # Dynamic proximity threshold between peaks
        # if c2p >  len(c2) / 2:
        #     proximity_threshold = 100 * c2p / len(c2)

        # If the current peak in c2 is too far from the closest peak in c1, skip it
        if np.abs(c1p - c2p) > proximity_threshold or c1p <= c1p_prev:
            continue

        # Scale the interval
        interval_c2 = c2p - c2p_prev
        interval_c1 = c1p - c1p_prev
        if interval_c2 == 0:
            continue
        # scale0 = interval_c1 / interval_c2
        # scale = interval_c1 / (interval_c2 + prev_length_change)
        scale = interval_c1 / interval_c2

        start = min(c2p_prev, c2p)
        end = max(c2p_prev, c2p)
        c2_segment = c2_aligned[start:end]
        scaled_segment = self.scale_chromatogram(c2_segment, scale)
        # prev_length_change = int(len(c2_segment) * scale0) - len(c2_segment)

        temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], scaled_segment, c2_aligned[end:]])
        temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
        c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments)
        if not valid:
            raise ValueError(
                "Please change parameters, some segments do not have peaks.")
        c2_peaks = np.concatenate([np.array((0,)), c2_peaks])

        if len(temp_c2_peaks_1) < initial_npeaks: # or len(c2_peaks) != nsegments:
            continue
            break


        c2_aligned = temp_c2_aligned_1

    return c2_aligned


def calculate_lag_corr(c1, c2, segment_length, hop=1, sigma=20, extend=10):
    """
    Calculate the lag of a segment ahead for each datapoint in c2 against c1.

    Parameters
    ----------
    c1 : numpy.ndarray
        Reference signal.
    c2 : numpy.ndarray
        Signal to be compared.
    segment_length : int
        Length of the segment ahead.
    hop : int, optional
        Step size for sliding the window. Default is 1.
    sigma : float, optional
        Standard deviation for Gaussian filter. Default is 20.
    extend : int, optional
        Number of points to extend segment_c1 on both sides. Default is 10.

    Returns
    -------
    numpy.ndarray
        Lags for each datapoint in c2.
    """
    lags = []
    lags_loc = []
    for i in range(extend, len(c2) - segment_length - extend + 1, hop):
        segment_c2 = c2[i:i + segment_length]
        start = max(0, i - extend)
        end = min(len(c1), i + segment_length + extend)
        segment_c1 = c1[start:end]

        # Calculate the cross-correlation between the segment and the extended segment_c1
        corr = correlate(
            normalize_standard(gaussian_filter(segment_c1, sigma)),
            normalize_standard(gaussian_filter(segment_c2, sigma)),
        )
        lag = np.argmax(corr) - len(segment_c2) + 1
        lags.append(lag)
        lags_loc.append(i)

    # # Add one last point equal to the last one at 300000
    lags.append(lags[-1])
    lags_loc.append(30000)

    return np.array(lags_loc), np.array(lags)


def plot_lag(lags_loc, lags, title='Lag as a function of retention time'):
    """
    Plot the lag as a function of x.

    Parameters
    ----------
    lags : numpy.ndarray
        Lags for each datapoint.
    title : str, optional
        Title of the plot. Default is 'Lag as a function of x'.
    """
    x = np.arange(len(lags))
    plt.figure(figsize=(10, 6))
    plt.plot(lags_loc, lags, label='Lag')
    plt.xlabel('Retention time')
    plt.ylabel('Lag')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def normalize_signal_standard(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    return normalized_signal

def normalize_standard(signal):
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

