import numpy as np
import re
import os
import subprocess
import pandas as pd
import utils
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from matplotlib import pyplot as plt
from scipy.signal import correlate, find_peaks
from scipy.ndimage import gaussian_filter
from collections import Counter


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
    # plt.xlabel('m/z')
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


def calculate_chance_accuracy_with_priors(labels):
    """
    Calculate the chance accuracy with priors based on class distribution.

    This function computes the chance accuracy by using the class priors.
    The chance accuracy is the sum of squared class proportions, where the
    proportion of each class is based on its occurrence in the dataset.

    Parameters
    ----------
    labels : list of str
        A list of labels representing the classes of the samples.

    Returns
    -------
    float
        The calculated chance accuracy with priors.

    Examples
    --------
    >>> labels = ['E', 'L', 'Q', 'D', 'P', 'E', 'E', 'Q', 'D', 'P', 'L', 'E', 'R', 'X', 'M']
    >>> calculate_chance_accuracy_with_priors(labels)
    0.06842374493702534
    """
    # Create a Counter object to count occurrences of each class
    class_counts = Counter(labels)

    # Calculate the total number of samples
    total_samples = sum(class_counts.values())

    # Calculate chance accuracy by summing the square of the class proportions
    chance_accuracy = sum((count / total_samples) ** 2 for count in class_counts.values())

    return chance_accuracy


def load_ms_data_from_directories(directory, columns, row_start, row_end):
    """
    Reads CSV files from all .D directories in the specified directory and extracts specific columns and row ranges.

    Args:
        directory (str): The path to the main directory containing .D directories.
        columns (list of int): A list of column indices to extract from each CSV file.
        row_start (int): The starting row index to extract (inclusive).
        row_end (int): The ending row index to extract (exclusive).

    Returns:
        dict of numpy arrays: A dictionary where each key is a .D directory name (without the .D suffix),
                              and each value is a numpy array containing the extracted data from each CSV file.
    """
    data_dict = {}

    # Loop through all .D directories in the specified directory
    for subdir in sorted(os.listdir(directory)):
        if subdir.endswith('.D'):
            # Remove the '.D' part for use as the dictionary key
            dir_name = subdir[:-2]

            # Construct the expected CSV file path
            # csv_file_path = os.path.join(directory, subdir, f"{dir_name}_MS.csv")
            csv_file_path = os.path.join(directory, subdir, f"{dir_name}.csv")

            # Check if the expected CSV file exists
            if os.path.isfile(csv_file_path):
                try:
                    # Read the CSV file with pandas
                    df = pd.read_csv(csv_file_path)

                    # Select the specified columns and row range
                    extracted_data = df.iloc[row_start:row_end, columns].to_numpy()

                    # Store the extracted data in the dictionary with the directory name as the key
                    data_dict[dir_name] = extracted_data

                except Exception as e:
                    print(f"Error processing file {csv_file_path}: {e}")
            else:
                print(f"No matching CSV file found in {subdir}.")

    return data_dict

def find_data_margins_in_csv(directory):
    """
       Finds the first CSV file in a .D directory within the specified directory,
       returns the number of rows, and identifies the indices of the first and last columns
       with integer headers.

       Args:
           directory (str): Path to the main directory containing .D directories.

       Returns:
           tuple: (int, int, int) where:
               - The first element is the row count in the CSV,
               - The second element is the index of the first integer-named column,
               - The third element is the index of the last integer-named column.
           Returns (None, None, None) if no valid CSV file is found.
       """
    # Traverse through each .D subdirectory
    for subdir in sorted(os.listdir(directory)):
        if subdir.endswith('.D'):
            dir_name = subdir[:-2]
            csv_file_path = os.path.join(directory, subdir, f"{dir_name}.csv")

            # Check if the expected CSV file exists
            if os.path.isfile(csv_file_path):
                try:
                    # Load the entire CSV to get an accurate row count and column information
                    df = pd.read_csv(csv_file_path)

                    # Row count, excluding the header row
                    row_count = len(df)

                    # Load only the header to get column names
                    df = pd.read_csv(csv_file_path, nrows=5)

                    # Identify columns with integer names
                    integer_columns = [i for i, col in enumerate(df.columns) if col.isdigit()]

                    if integer_columns:
                        # Get the first and last indices
                        first_integer_col = integer_columns[0]
                        last_integer_col = integer_columns[-1]

                        return row_count + 1, first_integer_col, last_integer_col

                except Exception as e:
                    print(f"Error processing file {csv_file_path}: {e}")
                    return None, None, None

    print("No CSV file found in any .D directory.")
    return None, None, None


def sum_data_in_data_dict(data_dict, axis=1):
    """
    Takes a dictionary with arrays as values and returns a new dictionary with the same keys.
    Depending on the axis parameter, each value is a list of either row sums or column sums.

    Args:
        data_dict (dict): A dictionary where keys are directory names, and values are 2D numpy arrays.
        axis (int): Axis along which to sum:
                    - 1 for summing all columns in a row (default)
                    - 0 for summing all rows in a column
    Returns:
        dict: A dictionary with the same keys as data_dict. Each value is a list of sums along the specified axis.
    """
    sum_dict = {}

    # Loop through each key and array in the input dictionary
    for key, array in data_dict.items():
        # Calculate the sum along the specified axis and convert to a list
        sums = array.sum(axis=axis).tolist()
        # Store the list of sums in the new dictionary
        sum_dict[key] = sums

    return sum_dict


def string_to_latex_confusion_matrix(data_str, headers):
    # Convert string to numpy array
    data_str = re.sub(r'\s+', ' ', data_str.replace('\n', ' '))  # Clean up whitespace
    data = np.array([list(map(float, row.split())) for row in data_str[2:-2].split('] [')])

    # Multiply by 100 and convert to integer
    data = np.round(data * 100).astype(int)

    # Begin LaTeX table string
    latex_string = "\\begin{table}[h!]\n\\centering\n\\begin{tabular}{|c|" + "c|" * len(headers) + "}\n    \\hline\n"

    # Add column headers with rotated labels
    latex_string += "    & " + " & ".join(f"\\rotatebox{{90}}{{{header}}}" for header in headers) + " \\\\\\hline\n"

    # Populate rows with cell color and no display value
    for i, row in enumerate(data):
        row_name = headers[i]
        row_cells = " & ".join(f"\\cellcolorval{{{value}}}" for value in row)
        latex_string += f"    {row_name} & {row_cells} \\\\\\hline\n"

    # Complete LaTeX table
    latex_string += "\\end{tabular}\n\\caption{Confusion Matrix in LaTeX}\n\\end{table}"

    # Print without escape characters
    print(latex_string)


def plot_image(image_tensor):
    """
    Plot a 3, 224, 224 image tensor using Matplotlib.

    Parameters:
    -----------
    image_tensor : torch.Tensor
        A tensor of shape (3, H, W) representing an RGB image.
    """
    if image_tensor.shape[0] == 3:  # Ensure it's 3-channel
        image_tensor = image_tensor.permute(1, 2, 0)  # Rearrange to (H, W, C)
    else:
        raise ValueError("Input tensor should have 3 channels (RGB).")

    # Convert tensor to numpy array
    image = image_tensor.cpu().numpy()

    # De-normalize if needed (ImageNet stats)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = std * image + mean  # De-normalize
    image = image.clip(0, 1)  # Clip values to [0, 1] for display

    # Plot the image
    plt.imshow(image)
    plt.axis('off')  # Turn off axes
    plt.show()