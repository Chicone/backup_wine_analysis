"""
Wine Analysis Overview
====================

This module provides tools and methods for analyzing wine-related data, particularly focusing on chromatogram analysis and dimensionality reduction techniques.

Key Features:
-------------
- **WineAnalysis Class**: Facilitates the analysis of wine datasets, including the ability to run dimensionality reduction techniques such as t-SNE and UMAP, as well as train classifiers.
- **ChromatogramAnalysis Class**: Handles the loading, normalization, and merging of chromatogram data, as well as synchronization and scaling of chromatograms.
- **SyncChromatograms Class**: Provides advanced techniques for aligning chromatograms, including methods for adjusting retention times and synchronizing chromatograms based on peak alignment.

Usage:
------
This module is intended for wine researchers and data scientists who need to analyze complex chromatographic data. The classes and methods provided can be used to preprocess, analyze, and visualize data, making it easier to identify patterns and trends in wine samples.

Dependencies:
-------------
The module relies on several third-party libraries, including:
- `numpy` for numerical operations
- `pandas` for data manipulation
- `scikit-learn` for machine learning and dimensionality reduction
- `matplotlib` for plotting and visualization
- `scipy` for signal processing and interpolation

Example:
--------
An example of using the `WineAnalysis` class to run a t-SNE analysis:

```
python from wine_analysis import WineAnalysis

# Initialize the WineAnalysis class with a data file
analysis = WineAnalysis(file_path='wine_data.npy', normalize=True)

# Run t-SNE on the dataset and plot the results
analysis.run_tsne(perplexity=30, random_state=42, plot=True)
"""
import numpy as np
import pandas as pd
import os
import math
from dcor import distance_correlation

import utils
from data_loader import DataLoader
from sklearn.preprocessing import StandardScaler
from classification import Classifier
from dimensionality_reduction import DimensionalityReducer
from visualizer import Visualizer
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks, peak_prominences
from scipy.stats import pearsonr
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d
from utils import normalize_dict, remove_peak, min_max_normalize, normalize_data
from scipy.ndimage import gaussian_filter
from sklearn.metrics import mutual_info_score
from scipy.stats import spearmanr, kendalltau
import utils

class WineAnalysis:
    """
        WineAnalysis Class
        ==================

        The `WineAnalysis` class is designed to facilitate the analysis of wine-related chromatographic data.
        It provides methods for loading data from various sources, performing dimensionality reduction, and running
        classification algorithms to extract meaningful insights from the data.

        Key Features:
        -------------
        - **Data Loading**: Load wine chromatographic data from .npy or .xlsx files, or directly from a dictionary.
        - **Dimensionality Reduction**: Apply techniques like PCA, t-SNE, and UMAP to reduce the dimensionality of the data.
        - **Classification**: Train classifiers such as LDA on the data to evaluate and compare different wine samples.
"""
    def __init__(self, file_path=None, data_dict=None, normalize=False):
        if file_path:
            script_dir = os.path.dirname(__file__)
            self.file_path = os.path.join(script_dir, file_path)
            if not os.path.isfile(self.file_path):
                raise FileNotFoundError(f"The file {self.file_path} does not exist.")
            self.data_loader = DataLoader(self.file_path, normalize=normalize)
            self.data = np.array(self.data_loader.df)
            # self.data = self.data_loader.get_standardized_data()
            self.labels = np.array(list(self.data_loader.df.index))
            self.chem_name = os.path.splitext(self.file_path)[0].split('/')[-1]
            print(self.file_path)
        elif data_dict:
            self.data_loader = None
            if normalize:
                self.data = StandardScaler().fit_transform(pd.DataFrame(data_dict).T)
            else:
                self.data = np.array(pd.DataFrame(data_dict).T)
            self.labels = np.array(list(data_dict.keys()))
            self.chem_name = "Not available"
        else:
            raise ValueError("Either file_path or data_dict must be provided.")

    def train_classifier(self, classifier_type='LDA', vintage=False, n_splits=50, test_size=None):
        data = self.data
        clf = Classifier(data, self.labels, classifier_type=classifier_type)
        return clf.train_and_evaluate(n_splits, vintage=vintage, test_size=test_size)

    def run_tsne(self, perplexity=60, random_state=16, best_score=None, plot=None):
        """
        Runs t-Distributed Stochastic Neighbor Embedding (t-SNE) on the data and plots the results.
        """
        reducer = DimensionalityReducer(self.data)
        tsne_result = reducer.tsne(components=2, perplexity=perplexity, random_state=random_state)
        # tsne_result = reducer.tsne(components=2, perplexity=15, random_state=10)
        tsne_result = -tsne_result  # change the sign of the axes to show data like in the paper
        tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE Component 1', 't-SNE Component 2'], index=self.labels)
        title = f'tSNE on {self.chem_name}; {len(self.data)} wines\nSilhouette score: {best_score} '
        if plot:
            Visualizer.plot_2d_results(tsne_df, title, 't-SNE Component 1', 't-SNE Component 2')

    def run_umap(self, n_neighbors=60, random_state=16, best_score=None, plot=None, title=None):
        """
        Runs Uniform Manifold Approximation and Projection (UMAP) on the data and plots the results.
        """

        reducer = DimensionalityReducer(self.data)
        umap_result = reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state)  # for concat
        # umap_result = reducer.umap(components=2, n_neighbors=60, random_state=20)  # for oak
        # umap_result = reducer.umap(components=2, n_neighbors=50, random_state=0)  # for 7 estates 2018+2022
        # umap_result = reducer.umap(components=2, n_neighbors=75, random_state=70)  # from searchgrid
        umap_result = -umap_result  # change the sign of the axes to show data like in the paper
        umap_df = pd.DataFrame(data=umap_result, columns=['UMAP Component 1', 'UMAP Component 2'], index=self.labels)
        title = f'{title}; {len(self.data)} wines\n Score: {best_score} '
        if plot:
            Visualizer.plot_2d_results(umap_df, title, 'UMAP Component 1', 'UMAP Component 2')

        return umap_df


class ChromatogramAnalysis:
    """
        ChromatogramAnalysis Class
        ==========================

        The `ChromatogramAnalysis` class is responsible for managing chromatogram data in wine analysis. It provides methods to load, resample, synchronize, and merge chromatograms from different datasets, making it a vital tool for comparative analysis of wine samples.

        Key Features:
        -------------
        - **Data Loading**: Load chromatograms from various file formats.
        - **Resampling and Merging**: Resample chromatograms to a common time axis and merge data from different sources.
        - **Synchronization**: Synchronize chromatograms by aligning peaks and adjusting retention times.

        """

    def __init__(self, file_path1, file_path2):
        self.file_path1 = file_path1
        self.file_path2 = file_path2

    def load_chromatogram(self, file_path):
        return np.load(file_path, allow_pickle=True).item()

    def load_chromatograms(self):
        chromatograms = {}
        for file_path in self.file_paths:
            chrom_data = self.load_chromatogram(file_path)
            chromatograms.update(chrom_data)
        return chromatograms


    def normalize_chromatogram(self, chromatogram):
        """
        Normalizes the chromatogram data using StandardScaler.

        Parameters:
        chromatogram (np.ndarray): The input chromatogram data.

        Returns:
        np.ndarray: The normalized chromatogram.
        """
        scaler = StandardScaler()
        chromatogram = np.array(chromatogram).reshape(-1, 1)  # Reshape for StandardScaler
        normalized_chromatogram = scaler.fit_transform(chromatogram).flatten()  # Fit and transform, then flatten back
        return normalized_chromatogram


    def calculate_mean_chromatogram(self, chromatograms):
        all_data = np.array(list(chromatograms.values()))
        mean_chromatogram = np.mean(all_data, axis=0)
        return mean_chromatogram


    def merge_chromatograms(self, chrom1, chrom2, norm=False):
        """
        Merges two chromatogram dictionaries, normalizes them, and handles duplicate sample names.

        Parameters:
        chrom1 (dict): The first chromatogram dictionary.
        chrom2 (dict): The second chromatogram dictionary.

        Returns:
        dict: The merged and normalized chromatogram data.
        """

        merged_chromatograms = {}

        for key, value in chrom1.items():
            # # remove extra peak from standard
            # value = sc_inst.remove_peak(self, value, peak_idx=8910, window_size=30)
            if norm:
                merged_chromatograms[key] = self.normalize_chromatogram(value)
            else:
                merged_chromatograms[key] = value

        for key, value in chrom2.items():
            if key in merged_chromatograms or key[0] + '_' + key[1:] in merged_chromatograms:
                key = f"{key}b"
            if norm:
                merged_chromatograms[key] = self.normalize_chromatogram(value)
            else:
                merged_chromatograms[key] = value

        return merged_chromatograms

    def resample_chromatogram(self, chrom, new_length):
        """
        Resamples a chromatogram to a new length using interpolation.

        Parameters:
        chrom (np.ndarray): The chromatogram to be resampled.
        new_length (int): The new length for the chromatogram.

        Returns:
        np.ndarray: The resampled chromatogram.
        """
        x_old = np.linspace(0, 1, len(chrom))
        x_new = np.linspace(0, 1, new_length)
        f = interp1d(x_old, chrom, kind='linear')
        return f(x_new)

    def tsne_analysis(self, data_dict, vintage, chem_name):
        analysis = WineAnalysis(data_dict=data_dict, normalize=False)
        cls = Classifier(analysis.data, analysis.labels)
        perplexity, random_state, best_score = run_tsne_and_evaluate(
            analysis,
            cls._process_labels(vintage),
            chem_name,
            perplexities=range(10, 60, 10),
            random_states=range(0, 96, 16)
        )
        analysis.run_tsne(perplexity=perplexity, random_state=random_state, best_score=best_score, plot=True)
        # analysis.run_umap(n_neighbors=10, random_state=10, best_score=10)
#
    def umap_analysis(self, data_dict, vintage, chem_name, neigh_range=range(10, 60, 10), random_states=range(0, 96, 16)):
        analysis = WineAnalysis(data_dict=data_dict, normalize=False)
        cls = Classifier(analysis.data, analysis.labels)
        n_neighbors, random_state, best_score = run_umap_and_evaluate(
            analysis,
            cls._process_labels(vintage),
            chem_name,
            neigh_range=neigh_range,
            random_states=random_states
        )
        title = f'UMAP on {chem_name}; neigh={n_neighbors}, random state={random_state}'
        analysis.run_umap(n_neighbors=n_neighbors, random_state=random_state, best_score=best_score, plot=True, title=title)
        # analysis.run_umap(n_neighbors=10, random_state=10, best_score=10)
#

    def sync_individual_chromatograms(self, reference_chromatogram, input_chromatograms, scales, initial_lag=300):
        """
        Synchronize individual chromatograms with a reference chromatogram.

        Parameters
        ----------
        reference_chromatogram : array-like
            The reference chromatogram to which the individual chromatograms will be synchronized.
        input_chromatograms : dict
            A dictionary where keys are labels for each chromatogram, and values are the chromatograms to be synchronized.
        scales : array-like
            Scaling factors to be applied during synchronization.
        initial_lag : int, optional
            The initial lag to be considered for synchronization, by default 300.

        Returns
        -------
        dict
            A dictionary with the same keys as `input_chromatograms`, where each value is the synchronized chromatogram.

        Notes
        -----
        This function uses the `SyncChromatograms` class from the `wine_analysis` module to perform the synchronization.

        Examples
        --------
        >>> synced_chromatograms = obj.sync_individual_chromatograms(reference_chromatogram, input_chromatograms, scales)
        >>> print(synced_chromatograms['label1'])

        """

        from wine_analysis import SyncChromatograms
        def plot_average_profile_with_std(data, num_points=500, title='Average Profile with Standard Deviation'):
            """
            Plots the average profile with standard deviation for a list of (x, y) tuples.

            Parameters:
            - data: list of tuples, where each tuple contains two numpy arrays (x, y).
            - num_points: int, the number of points in the common x grid for interpolation (default is 500).

            Returns:
            - None
            """
            # Define a common x grid based on the min and max x values across all profiles
            x_common = np.linspace(min([min(x) for x, _ in data]), max([max(x) for x, _ in data]), num_points)

            # Interpolate the y values to the common x grid
            interpolated_y_values = []
            for x, y in data:
                # Create an interpolation function
                f = interp1d(x, y, bounds_error=False, fill_value="extrapolate")
                # Interpolate y values to the common x grid
                y_interp = f(x_common)
                interpolated_y_values.append(y_interp)

            # Convert the list to a numpy array for easier computation
            interpolated_y_values = np.array(interpolated_y_values)

            # Calculate the mean and standard deviation across the y values
            y_mean = np.mean(interpolated_y_values, axis=0)
            y_std = np.std(interpolated_y_values, axis=0)

            # Plot the mean profile with the standard deviation as a shaded region
            plt.figure(figsize=(10, 6))
            plt.plot(x_common, y_mean, label='Average Profile', color='blue')
            plt.fill_between(x_common, y_mean - y_std, y_mean + y_std, color='blue', alpha=0.3,
                             label='±1 Standard Deviation')
            plt.xlabel('Retention time')
            plt.ylabel('Local retention time correction')
            plt.title(title)
            plt.legend()
            plt.show()

        lag_profiles = []
        synced_chromatograms = {}
        for i, key in enumerate(input_chromatograms.keys()):
            print(i, key)
            chrom = input_chromatograms[key]
            sync_chrom = SyncChromatograms(
                reference_chromatogram, chrom, 1, scales, 1E6, threshold=0.00, max_sep_threshold=50,
                peak_prominence=0.00
            )
            optimized_chrom = sync_chrom.adjust_chromatogram()
            lag_profiles.append(sync_chrom.lag_res[0:2])
            synced_chromatograms[key] = optimized_chrom

        # plot_average_profile_with_std(lag_profiles, title='Lag distribution 2018 dataset')
        return synced_chromatograms

    def stacked_2D_plots_3D(self, data_dict):
        """
        Creates stacked 2D plots in a 3D space for the given dictionary of data.

        Parameters:
        data_dict (dict): A dictionary with labels as keys and lists of values as values.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        labels = list(data_dict.keys())
        values = list(data_dict.values())

        num_plots = len(labels)
        x = np.arange(len(values[0]))
        offset = 10  # Offset between plots in the z direction

        for i, (label, value) in enumerate(zip(labels, values)):
            z = np.full_like(x, i * offset)  # Set the z value to stack plots
            ax.plot(x, value, zs=z, zdir='z', label=label)

            # Show legend for the first and last plots
            if i == 0 or i == num_plots - 1:
                ax.legend(loc='upper left')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Plot index')
        ax.set_title('Stacked 2D Plots in 3D')

        # Set the initial view to make y-axis vertical
        ax.view_init(elev=150, azim=-90)
        ax.set_xlim([0, 15000])
        # ax.set_ylim([0, 0.2])

        def on_key(event):
            if event.key == 'r':
                ax.view_init(elev=90, azim=-90)
                fig.canvas.draw()

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

    def stacked_plot(self, data_dict):
        """
        Creates a stacked plot for the given dictionary of data.

        Parameters:
        data_dict (dict): A dictionary with labels as keys and lists of values as values.

        """
        labels = list(data_dict.keys())
        values = list(data_dict.values())

        # Ensure all lists have the same length
        max_length = max(len(lst) for lst in values)
        values = [np.pad(lst, (0, max_length - len(lst)), mode='constant') for lst in values]

        # Create the stack plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')

        # ax.stackplot(range(max_length), *values, labels=labels)

        # Customize the plot
        ax.legend(loc='upper left')
        ax.set_title('Stacked Plot')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Values')

        # Show the plot
        plt.show()

    def resample_chromatograms(self, chrom1, chrom2, start=None, length=None):
        """
        Resamples two chromatograms to the same length using interpolation.

        Parameters:
        chrom1 (dict or np.ndarray): The first chromatogram dictionary or array to be resampled.
        chrom2 (dict or np.ndarray): The second chromatogram dictionary or array to be resampled.

        Returns:
        tuple: The resampled chromatograms.
        """
        if isinstance(chrom1, dict) and isinstance(chrom2, dict):
            # Use the minimum number of chromatogram values in any sample
            if not length:
                length = min(min(len(value) for value in chrom1.values()), min(len(value) for value in chrom2.values()))
            resampled_chrom1 = {key: self.resample_chromatogram(value[start:], length) for key, value in chrom1.items()}
            resampled_chrom2 = {key: self.resample_chromatogram(value[start:], length) for key, value in chrom2.items()}
        else:
            length = min(len(chrom1), len(chrom2))  # Use the minimum length if chrom1 and chrom2 are arrays
            resampled_chrom1 = self.resample_chromatogram(chrom1, length)
            resampled_chrom2 = self.resample_chromatogram(chrom2, length)

        return resampled_chrom1, resampled_chrom2


class SyncChromatograms:
    """
        SyncChromatograms Class
        =======================

        The `SyncChromatograms` class provides methods for synchronizing chromatographic data, focusing on aligning peaks and adjusting retention times across different chromatograms. This is essential for comparing chromatographic profiles from different wine samples.

        Key Features:
        -------------
        - **Peak Alignment**: Automatically detects and aligns peaks across chromatograms.
        - **Retention Time Adjustment**: Corrects differences in retention times to synchronize chromatograms.
        - **Segmented Analysis**: Allows for synchronization across multiple segments for greater accuracy.

        Usage:
        ------
        This class is ideal for researchers who need to compare chromatographic data across multiple wine samples, ensuring that peaks and retention times are aligned for accurate analysis.
        """

    def __init__(self, c1, c2, n_segments, scales, min_peaks=5, max_iterations=100, threshold=0.5, max_sep_threshold=50, peak_prominence=0.1):
        self.c1 = c1
        self.c2 = c2
        self.n_segments = n_segments
        self.scales = scales
        self.min_peaks = min_peaks
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.max_sep_threshold = max_sep_threshold
        self.peak_prominence = peak_prominence
        self.lag_res = None

    def scale_chromatogram(self, chrom, scale):
        x = np.arange(len(chrom))
        scaled_length = int(len(chrom) * scale)
        scaled_x = np.linspace(0, len(chrom) - 1, num=scaled_length)
        f = interp1d(x, chrom, bounds_error=False, fill_value="extrapolate")
        return f(scaled_x)

    def correct_segment(self, segment, scale, lag):
        if scale is None or lag is None:
            return segment

        scaled_segment = self.scale_chromatogram(segment, scale)
        if lag > 0:
            corrected_segment = np.roll(scaled_segment, lag)
            corrected_segment[:lag] = scaled_segment[0]
        elif lag < 0:
            corrected_segment = np.roll(scaled_segment, lag)
            corrected_segment[lag:] = scaled_segment[-1]
        else:
            corrected_segment = scaled_segment

        return corrected_segment

    def find_best_scale_and_lag_corr(self, chrom1, chrom2, scales, max_lag=300):
        """
        Finds the best scale and lag for cross-correlation between two chromatograms.

        Parameters:
        chrom1 (np.array): The first chromatogram.
        chrom2 (np.array): The second chromatogram.
        scales (list of float): The list of scaling factors to try.
        max_lag (int): The maximum lag to consider for cross-correlation.

        Returns:
        float: The best scaling factor.
        int: The best lag.
        float: The highest cross-correlation value.
        """
        best_scale = None
        best_lag = None
        best_corr = -np.inf

        for scale in scales:

            # Scale the first chromatogram
            scaled_chrom2 = self.scale_chromatogram(chrom2, scale)

            # Compute the cross-correlation
            corr = correlate(chrom1, scaled_chrom2, mode='full', method='auto')
            center = len(corr) // 2
            corr = corr[center - max_lag:center + max_lag + 1]
            # lags = np.arange(-len(chrom1) + 1, len(scaled_chrom2))
            lags = np.arange(-max_lag, max_lag + 1)

            # Find the lag with the highest correlation within the specified range
            max_lag_idx = np.argmax(corr)
            lag = lags[max_lag_idx]
            corr_value = corr[max_lag_idx]

            # Update the best scale and lag if the correlation is higher
            if corr_value > best_corr:
                best_corr = corr_value
                best_scale = scale
                best_lag = lag

        return best_scale, best_lag, best_corr

    def calculate_lag_profile(self, c1, c2, segment_length, hop=1, sigma=20, lag_range=10, distance_metric='l2',
                              init_min_dist=0.1):
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
        lags_location = []
        for i in range(0, len(c2) - segment_length + 1, hop):
            segment_c2 = c2[i:i + segment_length]
            start = max(0, i)
            end = min(len(c1), i + segment_length)
            segment_c1 = c1[start:end]

            # Normalize and apply Gaussian filter
            segment_c1_filtered = utils.normalize_amplitude_zscore(gaussian_filter(segment_c1, sigma))
            segment_c2_filtered = utils.normalize_amplitude_zscore(gaussian_filter(segment_c2, sigma))

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
            lags_location.append(i)
            # print(min_distance)

        # # Add one last point equal to the last one at 30000
        lags.append(lags[-1])
        lags_location.append(30000)

        return np.array(lags_location), np.array(lags)


    def lag_profile_from_peaks(self, reference_chromatogram, target_chromatogram, alignment_tolerance, num_segments,
                               apply_global_alignment=True):
        """
        Generate a lag profile from peaks in two chromatograms by aligning segments.

        This function aligns peaks in `target_chromatogram` to peaks in `reference_chromatogram` and adjusts the chromatogram
        `target_chromatogram` accordingly. It then computes the lag profile based on the difference in peak positions between
        the two chromatograms.

        Parameters
        ----------
        reference_chromatogram : array-like
            The reference chromatogram (e.g. the mean chromatogram of a given dataset).
        target_chromatogram : array-like
            The chromatogram to be aligned with `reference_chromatogram`.
        alignment_tolerance : float
            The maximum allowed distance between aligned peaks in `reference_chromatogram` and `target_chromatogram`. Peaks in
            `target_chromatogram` that are too far from the corresponding peak in `reference_chromatogram` will be skipped.
        num_segments : int
            The number of segments to divide the chromatograms into for peak alignment. The highest peak from each segment
            is typically selected.
        apply_global_alignment : bool, optional
            If True, apply a global time shift to align the chromatograms before local adjustments. Default is True.

        Returns
        -------
        lags_location : numpy.ndarray
            The locations of the lags in `target_chromatogram` after alignment.
        lags : numpy.ndarray
            The lag values corresponding to each peak after alignment.
        target_chromatogram_aligned : numpy.ndarray
            The aligned chromatogram `target_chromatogram` after applying the per-section adjustments.

        Raises
        ------
        ValueError
            If some segments in `target_chromatogram` do not have peaks.

        Notes
        -----
        This method uses a combination of global and local alignment to synchronize the chromatograms. The global alignment
        is applied first (if `apply_global_alignment` is True), followed by local adjustments in each segment.

        Examples
        --------
        >>> lags_location, lags, target_chromatogram_aligned = obj.lag_profile_from_peaks(reference_chromatogram, target_chromatogram, alignment_tolerance=50, num_segments=50)
        """

        if apply_global_alignment:
            # Apply global alignment by finding the best scale (currently turned off) and lag
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(reference_chromatogram[:10000], 50),
                gaussian_filter(target_chromatogram[:10000], 50),
                np.linspace(1.0, 1.0, 1)
            )
            target_chromatogram_aligned = gaussian_filter(
                self.correct_segment(target_chromatogram, best_scale, best_lag), 5
            )
            accum_lag = best_lag
        else:
            target_chromatogram_aligned = target_chromatogram
            accum_lag = 0

        # Find peaks in the reference chromatogram
        reference_peaks, _ = find_peaks(reference_chromatogram)

        # Remove peak at a known standard position in the reference chromatogram
        reference_peaks = np.delete(reference_peaks, np.where(reference_peaks == 8918))

        # Find peaks in the aligned target chromatogram
        target_peaks, _ = find_peaks(target_chromatogram_aligned)

        # Ensure there is a maximum of one peak per segment (the highest) in both chromatograms
        reference_peaks, valid = self.ensure_peaks_in_segments(
            reference_chromatogram, reference_peaks, num_segments=num_segments
        )
        target_peaks, valid = self.ensure_peaks_in_segments(
            target_chromatogram_aligned, target_peaks, num_segments=num_segments
        )

        if not valid:
            raise ValueError("Please change parameters, some segments in chromatograms do not have peaks.")

        # Add zero to scale the first part of the chromatogram up to the first peak
        reference_peaks = np.concatenate([np.array([0]), reference_peaks])
        target_peaks = np.concatenate([np.array([0]), target_peaks])
        num_initial_peaks = min(len(target_peaks), len(reference_peaks))

        lags = []
        lags_location = []

        for i in range(1, num_initial_peaks):
            # Select the previous and current peaks from the target chromatogram
            target_peak_prev = target_peaks[i - 1]
            target_peak = target_peaks[i]

            # Initialize the corresponding previous peak in the reference chromatogram
            reference_peak_prev = target_peak_prev

            # Find the closest peak in the reference chromatogram to the current target peak
            reference_peak = reference_peaks[np.argmin(np.abs(reference_peaks - target_peak))]

            # Skip if the current peak in target is too far from the reference or in the wrong order
            if np.abs(reference_peak - target_peak) > alignment_tolerance or reference_peak <= reference_peak_prev:
                if i >= len(target_peaks) - 1:
                    break
                else:
                    continue

            # Accumulate the lag
            accum_lag += reference_peak - target_peak

            # Calculate the scaling factor and adjust the target segment
            interval_target = target_peak - target_peak_prev
            interval_reference = reference_peak - reference_peak_prev
            if interval_target == 0:
                continue
            scale = interval_reference / interval_target
            start = min(target_peak_prev, target_peak)
            end = max(target_peak_prev, target_peak)
            target_segment = target_chromatogram_aligned[start:end]
            scaled_segment = self.scale_chromatogram(target_segment, scale)

            # Update the aligned target chromatogram with the scaled segment
            target_chromatogram_aligned = np.concatenate([
                target_chromatogram_aligned[:start], scaled_segment, target_chromatogram_aligned[end:]
            ])

            # Recalculate peaks in the updated target chromatogram
            target_peaks, _ = find_peaks(target_chromatogram_aligned)
            target_peaks, valid = self.ensure_peaks_in_segments(
                target_chromatogram_aligned, target_peaks, num_segments=num_segments
            )

            if not valid:
                raise ValueError("Please change parameters, some segments do not have peaks.")
            target_peaks = np.concatenate([np.array([0]), target_peaks])

            if len(target_peaks) < num_initial_peaks and i >= len(target_peaks) - 1:
                break

            # Skip peaks that would move backwards in retention time
            if len(lags_location) > 0 and any(target_peak + accum_lag <= loc for loc in lags_location):
                continue

            # Append the accumulated lag and its location
            lags.append(accum_lag)
            lags_location.append(target_peak + accum_lag)

        # Ensure the first lag is consistent with the rest
        if len(lags) > 0:
            lags.insert(0, lags[0])
            lags_location.insert(0, 0)
            lags.append(lags[-1])
            lags_location.append(30000)

        return np.array(lags_location), np.array(lags), target_chromatogram_aligned

    def lag_profile_moving_peaks_individually(
            self, reference_chromatogram, target_chromatogram, alignment_tolerance, num_segments, apply_global_alignment=True,
            scan_range=1, peak_order=0, interval_after=500, min_avg_peak_distance=50):
        """
        Align peaks between a reference signal and a target signal by moving target peaks independently,
        and calculate the lag profile. The peaks are moved by rescaling the intervals with their adjacent peaks to both
         left and right of each peak.
        .

        Parameters
        ----------
        reference_chromatogram : array-like
            The reference chromatogram used for peak alignment.
        target_chromatogram : array-like
            The target chromatogram to be aligned with the reference signal.
        alignment_tolerance : float
            The maximum allowed distance between aligned peaks.
        num_segments : int
            Number of segments to divide the signal into.
        apply_global_alignment : bool, optional
            If True, apply a global time shift before peak alignment. Default is True.
        scan_range : int, optional
            The range within which to scan for matching peaks. Default is 1.
        peak_order : int, optional
            The rank of the peak to select within each segment. Default is 0 (highest peak).
        interval_after : int, optional
            Interval length after the peak to improve similarity measure. Default is 500.
        min_avg_peak_distance : float, optional
            Minimum average peak distance required for accepting the scaling. Default is 50.

        Returns
        -------
        numpy.ndarray
            Locations of the lags in the target signal after alignment.
        numpy.ndarray
            The lag values corresponding to each peak after alignment.
        """

        #### Local functions ####
        def calculate_average_peak_distance(signal1, signal2, prominence=1E-6, distance_type='mean'):
            """
            Calculate the average distance between peaks in two signals.

            Parameters
            ----------
            signal1, signal2 : array-like
                The signals for which peak distances are calculated.
            prominence : float, optional
                The prominence of peaks to consider. Default is 1E-6.
            distance_type : str, optional
                Type of distance to calculate ('mean' or 'mean_square'). Default is 'mean'.

            Returns
            -------
            float
                The average distance between peaks.
            list
                List of distances between matched peaks.
            numpy.ndarray
                Peaks in the first signal.
            numpy.ndarray
                Peaks in the second signal.
            """
            peaks1, _ = find_peaks(signal1, prominence=prominence)
            peaks2, _ = find_peaks(signal2, prominence=prominence)

            if len(peaks1) == 0 or len(peaks2) == 0:
                raise ValueError("No peaks found in one of the signals")

            distances = []
            for peak1 in peaks1:
                closest_peak2 = peaks2[np.argmin(np.abs(peaks2 - peak1))]
                distances.append(abs(peak1 - closest_peak2))

            if distance_type == 'mean':
                average_distance = np.mean(distances)
            elif distance_type == 'mean_square':
                average_distance = np.mean(np.square(distances))
            else:
                raise ValueError("Invalid distance_type. Choose 'mean' or 'mean_square'.")

            return average_distance, distances, peaks1, peaks2

        def calculate_weighted_average_peak_distance(signal1, signal2, prominence=1E-6, distance_type='mean',
                                                     weight_power=1):
            """
            Calculate the weighted average distance between peaks in two signals, with weights based on the relative heights of both peaks.

            Parameters
            ----------
            signal1, signal2 : array-like
                The signals for which peak distances are calculated.
            prominence : float, optional
                The prominence of peaks to consider. Default is 1E-6.
            distance_type : str, optional
                Type of distance to calculate ('mean' or 'mean_square'). Default is 'mean'.
            weight_power : float, optional
                The power to raise the relative height ratio, allowing for non-linearity in the weighting. Default is 1 (linear).

            Returns
            -------
            float
                The weighted average distance between peaks.
            list
                List of distances between matched peaks.
            numpy.ndarray
                Peaks in the first signal.
            numpy.ndarray
                Peaks in the second signal.
            """
            peaks1, properties1 = find_peaks(signal1, prominence=prominence)
            peaks2, properties2 = find_peaks(signal2, prominence=prominence)

            if len(peaks1) == 0 or len(peaks2) == 0:
                raise ValueError("No peaks found in one of the signals")

            distances = []
            weights = []  # Store weights based on relative peak heights
            for peak1, height1 in zip(peaks1, properties1["prominences"]):
                # Find the closest peak in signal2 to peak1
                closest_peak2_idx = np.argmin(np.abs(peaks2 - peak1))
                closest_peak2 = peaks2[closest_peak2_idx]
                height2 = properties2["prominences"][closest_peak2_idx]  # Get height of the closest peak2

                # Calculate the distance between the two peaks
                distance = abs(peak1 - closest_peak2)
                distances.append(distance)

                # Calculate the relative weight based on peak heights
                relative_height_ratio = min(height1, height2) / max(height1, height2)

                # Apply the weight_power to introduce non-linearity
                weight = relative_height_ratio ** weight_power
                weights.append(weight)

            # Normalize the weights so they sum to 1
            normalized_weights = np.array(weights) / np.sum(weights)

            # Compute the weighted average distance
            if distance_type == 'mean':
                weighted_average_distance = np.average(distances, weights=normalized_weights)
            elif distance_type == 'mean_square':
                weighted_average_distance = np.average(np.square(distances), weights=normalized_weights)
            else:
                raise ValueError("Invalid distance_type. Choose 'mean' or 'mean_square'.")

            return weighted_average_distance, distances, peaks1, peaks2

        #### End local functions ####

        if apply_global_alignment:
            # Apply a global time shift using the best scale and lag
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(reference_chromatogram[:10000], 50),
                gaussian_filter(target_chromatogram[:10000], 50),
                np.linspace(1., 1.1, 1)
            )
            target_chromatogram_aligned = self.correct_segment(target_chromatogram, best_scale, best_lag)
            accumulated_lag = best_lag
        else:
            target_chromatogram_aligned = target_chromatogram
            accumulated_lag = 0

        reference_peaks, _ = find_peaks(reference_chromatogram)
        all_target_peaks, _ = find_peaks(target_chromatogram_aligned)

        # Ensure peaks are within segments in the target signal
        target_peaks, valid = self.ensure_peaks_in_segments(
            target_chromatogram_aligned, all_target_peaks, num_segments=num_segments, peak_ord=peak_order
        )
        if not valid:
            raise ValueError("Please change parameters, peaks found in target.")

        # Add zero to scale the first part of the chromatogram
        reference_peaks = np.concatenate([np.array([0]), reference_peaks])
        target_peaks = np.concatenate([np.array([0]), target_peaks])

        # Initialize lists to store the calculated lags and their corresponding locations
        lags = []
        lags_location = []

        num_initial_peaks = min(len(target_peaks), len(reference_peaks))

        # Loop over each peak, starting from the second peak (index 1) to the last peak
        for i in range(1, num_initial_peaks):
            try:
                # Find the previous peak in the target signal (this is to scale interval on the left of target)
                target_peak_prev = all_target_peaks[np.where(all_target_peaks == target_peaks[i])[0] - 1][0]
            except IndexError:
                print('Error finding previous peak in target signal.')
                continue  # If the previous peak cannot be found, skip this iteration

            # Get the current target peak
            target_peak = target_peaks[i]
            reference_peak_prev = target_peak_prev  # INitialize the previous reference peak to the previous target peak

            # Find the closest peak in the reference signal to the current target peak
            closest_index = np.argmin(np.abs(reference_peaks - target_peaks[i]))
            start_index = max(0, closest_index - scan_range)  # Start of scanning from a range around the closest peak
            end_index = min(len(reference_peaks), closest_index + scan_range + 1)  # End of scanning

            # Initialize minimum distance to the parameter-passed minimum average peak distance
            min_distance = min_avg_peak_distance

            # Define the start and end retention times of the segment in the target signal that will be adjusted
            start = min(target_peak_prev, target_peak)
            end = max(target_peak_prev, target_peak)

            # Explore the surroundings of the reference peak within the specified range
            for idx in range(start_index, end_index):
                reference_peak = reference_peaks[idx]  # Get the current reference peak within the scan range

                # Check if the current peaks fall within the valid range of the chromatogram
                if ((reference_peak + interval_after > len(reference_chromatogram)) or
                        (target_peak + interval_after > len(target_chromatogram_aligned))):
                    continue

                # Calculate the intervals between the current and previous peaks for both signals
                interval_target = target_peak - target_peak_prev
                interval_reference = reference_peak - reference_peak_prev

                # Ensure the intervals are valid and within the allowed alignment tolerance
                if (
                        abs(reference_peak - target_peak) > alignment_tolerance or
                        interval_target <= 0 or
                        interval_reference <= 0 or
                        (end - start) <= 0
                ):
                    continue

                # Calculate the scaling factor based on the ratio of intervals
                scale = interval_reference / interval_target

                # Extract the segment from the target signal and apply the scaling
                target_segment = target_chromatogram_aligned[start:end]
                scaled_segment = self.scale_chromatogram(target_segment, scale)

                if len(scaled_segment) == 0:
                    continue  # Skip if the scaled segment is empty

                # Normalize the segments of both reference and target signals for comparison
                norm_reference_segment = utils.normalize_amplitude_minmax(
                    reference_chromatogram[reference_peak_prev:reference_peak + interval_after]
                )
                norm_target_segment = utils.normalize_amplitude_minmax(
                    np.concatenate([scaled_segment, target_chromatogram[end:end + interval_after]])
                )

                try:
                    # Calculate the average peak distance between the normalized segments of target and reference
                    # avg_peak_dist = calculate_average_peak_distance(
                    #     norm_reference_segment, norm_target_segment, prominence=1E-6, distance_type='mean'
                    # )[0]
                    avg_peak_dist = calculate_weighted_average_peak_distance(
                        norm_reference_segment, norm_target_segment, prominence=1E-6, distance_type='mean'
                    )[0]
                except Exception:
                    print('Error calculating average peak distance.')
                    continue

                # If the calculated average peak distance is less than the current minimum, update the best segment
                if avg_peak_dist < min_distance:
                    min_distance = avg_peak_dist
                    best_scaled_segment = scaled_segment
                    best_end = end
                    best_reference_peak = reference_peak

            # Ensure a best segment was found; otherwise, default to the original segment
            try:
                best_scaled_segment
            except NameError:
                best_end = max(target_peak_prev, target_peak)
                best_reference_peak = reference_peak
                best_scaled_segment = target_chromatogram_aligned[start:end]

            # Check if the best reference peak is within the allowable alignment tolerance
            if np.abs(
                    best_reference_peak - target_peak) > alignment_tolerance or best_reference_peak <= reference_peak_prev:
                if i >= len(target_peaks) - 1:
                    break  # If the last peak is reached and conditions are not met, break the loop
                else:
                    continue  # Skip to the next peak

            # Calculate the stretch required to align the peaks
            stretch = best_reference_peak - target_peak

            # Identify the next peak in the target signal (this is to scale interval on the right of target)
            next_peak = all_target_peaks[np.where(all_target_peaks == target_peaks[i])[0] + 1][0]

            # If there is another peak after and sufficient space after the peak to apply the stretch
            if i + 1 < len(target_peaks) and len(target_chromatogram_aligned[best_end:next_peak]) > np.abs(stretch):
                next_segment = target_chromatogram_aligned[best_end:next_peak]  # Extract the section to be scaled
                compensate_factor = len(next_segment) / (len(next_segment) + stretch)
                try:
                    # Scale the section using the compensating factor
                    next_segment_corrected = self.scale_chromatogram(next_segment, compensate_factor)
                except Exception:
                    print('Error scaling next segment.')
                    continue

                # Update the aligned target signal with the corrected segment
                temp_target_chromatogram_aligned = np.concatenate(
                    [
                        target_chromatogram_aligned[:start],
                        best_scaled_segment,
                        next_segment_corrected,
                        target_chromatogram_aligned[next_peak:]
                     ]
                )
            else:
                # If compensation is not needed or possible, just update the aligned target signal with the best segment
                temp_target_chromatogram_aligned = np.concatenate(
                    [target_chromatogram_aligned[:start], best_scaled_segment, target_chromatogram_aligned[best_end:]]
                )

            # Update peaks after realignment and ensure they remain in their respective segments
            temp_target_peaks, _ = find_peaks(temp_target_chromatogram_aligned)
            all_target_peaks = temp_target_peaks
            target_peaks, valid = self.ensure_peaks_in_segments(
                temp_target_chromatogram_aligned, temp_target_peaks, num_segments=num_segments, peak_ord=peak_order
            )
            if not valid:
                raise ValueError("Please change parameters, no peaks found.")

            target_peaks = np.concatenate([np.array([0]), target_peaks])

            # If not enough peaks are left after alignment, break the loop
            if len(target_peaks) < num_initial_peaks and i >= len(target_peaks) - 1:
                break

            # Update the aligned target signal
            target_chromatogram_aligned = temp_target_chromatogram_aligned

            # Avoid the adjusted peak position (target_peak + stretch) ending up at or before any of the existing lag
            # locations
            if len(lags_location) > 0 and any(target_peak + stretch <= loc for loc in lags_location):
                continue

            # Append the calculated stretch and its location to the lags list
            lags.append(stretch)
            lags_location.append(target_peak + stretch)

        # Add one last point equal to the last one at a specific position (e.g., 30000) to avoid spline growing too large
        if len(lags) > 0:
            lags.append(lags[-1])
            lags_location.append(30000)

        # Return the arrays of lag locations and lag values
        return np.array(lags_location), np.array(lags), target_chromatogram_aligned

    def ensure_peaks_in_segments(self, signal, peaks_in_signal, num_segments=10, last_segment=None, peak_ord=0):
        """
        Ensure that each segment of the signal has at least one peak by selecting the most prominent peaks.

        Parameters
        ----------
        signal : array-like
            The signal data from which peaks have been identified.
        peaks_in_signal : array-like
            Indices of the peaks identified in the signal.
        num_segments : int, optional
            The number of segments to divide the signal into. Default is 10.
        last_segment : int or None, optional
            The index of the last segment to process. If None, all segments are processed. Default is None.
        peak_ord : int, optional
            The ordinal rank of the peak to select in each segment when sorted by prominence. Default is 0 (highest peak).

        Returns
        -------
        numpy.ndarray
            The indices of the selected peaks across all segments.
        bool
            True if peaks were successfully found in all segments, False otherwise.
        """
        segment_length = len(signal) // num_segments

        # Initialize an empty list to store the selected peaks from each segment
        new_peaks = []

        # Iterate through each segment
        for i in range(num_segments):
            # Calculate the start and end indices for the current segment
            start = i * segment_length
            end = (i + 1) * segment_length if i < num_segments - 1 else len(signal)

            # Find peaks that fall within the current segment
            segment_peaks = [peak for peak in peaks_in_signal if start <= peak < end]

            # If no peaks are found in this segment, skip to the next segment
            if not segment_peaks:
                continue

            # Sort the peaks in the segment based on their prominence (or signal value at the peak) in descending order
            sorted_peaks = sorted(segment_peaks, key=lambda x: signal[x], reverse=True)

            try:
                # Select the peak based on the specified ordinal rank (e.g., highest, second highest, etc.)
                selected_peak = sorted_peaks[peak_ord]
            except IndexError:  # Catch exception if peak_ord is out of range
                continue  # If the peak_ord is invalid, skip to the next segment

            # Add the selected peak to the list of new peaks
            new_peaks.append(selected_peak)

            # If a specific last segment is defined and reached, break out of the loop
            if last_segment is not None and i == last_segment:
                break

        # If no peaks were added to the new_peaks list, return an empty array and False
        if not new_peaks:
            return np.array(new_peaks), False

        # Return the array of selected peaks and True, indicating successful peak selection
        return np.array(new_peaks), True

    def adjust_chromatogram(self):
        """
        Adjusts the chromatogram to match a reference chromatogram.

        Returns
        -------
        numpy.ndarray
            The adjusted chromatogram after applying the synchronization algorithm.
        """

        #### Local functions ####
        def apply_shift_spline(c2, t, spline):
            """
            Apply a spline-based retention time shift to a chromatogram signal.

            Parameters
            ----------
            c2 : array-like
                The chromatogram to be adjusted.
            t : array-like
                The time indices in the chromatogram.
            spline : UnivariateSpline
                The spline function representing the time shift to be applied.

            Returns
            -------
            numpy.ndarray
                The chromatogram after the shift has been applied.
            """
            shifted_t = t - spline(t)
            interpolator = interp1d(t, c2, fill_value="extrapolate", bounds_error=False)
            return interpolator(shifted_t)

        def objective_function_spline(params, c1, c2, t, spline, loss='l2'):
            """
            Objective function to minimize the difference between the reference and adjusted chromatograms.

            Parameters
            ----------
            params : array-like
                Parameters to optimize, typically the smoothing factor for the spline.
            c1 : array-like
                The reference chromatogram.
            c2 : array-like
                The chromatogram to be adjusted.
            t : array-like
                The time points corresponding to the chromatograms.
            spline : UnivariateSpline
                The spline function representing the shift to be applied.
            loss : str, optional
                The loss function to be used in the optimization. Options are:
                - 'l1': Mean absolute error
                - 'l2': Mean squared error (default)
                - 'corr': Negative maximum cross-correlation
                - 'mse': Mean squared error

            Returns
            -------
            float
                The value of the loss function for the given parameters.
            """
            try:
                spline.set_smoothing_factor(params[0])
            except:
                print('Error in spline.set_smoothing_factor(params[0])')
            c2_shifted = apply_shift_spline(c2, t, spline)
            if loss == 'l1':
                return np.sum(np.abs(c1 - c2_shifted))
            elif loss == 'l2':
                return np.sum((c2_shifted - c1) ** 2)
            elif loss == 'corr':
                cross_corr = correlate(c1, c2_shifted, mode='valid')
                return -np.max(cross_corr)
            elif loss == 'mse':
                return np.mean((c2_shifted - c1) ** 2)

        def correct_with_spline(corrected_c2, s, k, normalize=True, plot=False):
            """
            Correct the chromatogram using a spline-based shift.

            Parameters
            ----------
            corrected_c2 : array-like
                The chromatogram to be corrected.
            s : float
                Smoothing factor for the spline.
            k : int
                Degree of the spline.
            normalize : bool, optional
                Whether to normalize the chromatograms before correction. Default is True.
            plot : bool, optional
                Whether to plot the results. Default is False.

            Returns
            -------
            numpy.ndarray
                The corrected chromatogram after applying the spline-based shift.
            """
            min_len = min(len(self.c1), len(corrected_c2))
            ref = min_max_normalize(self.c1, 0, 1)[:min_len]
            if normalize:
                chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
            else:
                chrom = corrected_c2[:min_len]

            try:
                spline = UnivariateSpline(self.lag_res[0], self.lag_res[1], s=s, k=k)
            except:
                print("Error creating spline")
                return chrom

            t = np.arange(len(ref))

            initial_guess = [s]
            bounds = [(0, None)]
            result = minimize(
                objective_function_spline, initial_guess, args=(ref, chrom, t, spline, 'mse'), bounds=bounds
            )

            optimized_smoothing_factor = result.x[0]
            spline.set_smoothing_factor(optimized_smoothing_factor)
            corrected_c2 = apply_shift_spline(chrom, t, spline)
            if plot:
                self.plot_signal_and_fit(
                    t, ref, chrom, corrected_c2, fit_type='spline', fit_params=spline, data_points=self.lag_res
                )

            return corrected_c2
        ##### End of local functions #####


        corrected_c2 = self.c2

        # Start first adjustment based on scaling of retention times between main peaks
        for prox in [40]:
            self.lag_res = self.lag_profile_from_peaks(
                self.c1, corrected_c2, alignment_tolerance=prox, num_segments=50, apply_global_alignment=True
            )
            corrected_c2 = correct_with_spline(corrected_c2, 50, 1, normalize=True, plot=False)
            corrected_c2_sharp = correct_with_spline(self.c2, 50, 1, normalize=True, plot=False)

        # Start second adjustment based on moving individual peaks to match the reference's
        c1 = self.c1.copy()
        # Apply Gaussian smoothing to the chromatograms (sigma value found experimentally)
        c1 = gaussian_filter(c1, 10)
        corrected_c2 = gaussian_filter(corrected_c2, 10)

        plot = False
        cnt = 0

        # Iterate over the specified peak order sequence to refine the alignment. Iteratively doing it several times for
        # each order improves accuracy
        for ord in [0, 0, 0, 1, 1, 1, 1, 1]:
            self.lag_res = self.lag_profile_moving_peaks_individually(
                c1, corrected_c2, alignment_tolerance=250, num_segments=10, apply_global_alignment=False, scan_range=3,
                peak_order=ord, interval_after=3000, min_avg_peak_distance=10
            )
            print(self.lag_res[0], self.lag_res[1])

            # # Apply spline-based correction to the chromatogram based on the current lag profile
            # corrected_c2 = correct_with_spline(corrected_c2, 20, 1, normalize=False, plot=plot)
            # corrected_c2_sharp = correct_with_spline(corrected_c2_sharp, 20, 1, normalize=False, plot=plot)
            corrected_c2 = self.lag_res[2]
            cnt += 1
        # corrected_c2 = corrected_c2_sharp

        return np.array(corrected_c2)

    # Function to plot the results
    def plot_signal_and_fit(self, t, ref, chrom, corrected_c2, fit_type='spline', fit_params=None, data_points=None):
        """
        Plots the original and corrected signals, as well as the fitting curve (spline, quadratic, or linear).

        Parameters:
        - t: Time axis or index.
        - ref: Reference signal.
        - chrom: Original signal before correction.
        - corrected_c2: Corrected signal after applying the shift.
        - fit_type: Type of fitting ('spline', 'quadratic', or 'linear').
        - fit_params: Parameters for the fit. Could be the spline object, quadratic coefficients, or linear coefficients.
        - data_points: Data points related to the lag or shift used for plotting (optional).
        """
        plt.figure(figsize=(10, 8))

        # Plot the original and corrected signals
        plt.subplot(2, 1, 1)
        plt.plot(t, ref, label='Reference Signal c1')
        plt.plot(t, chrom, label='Original Signal c2')
        plt.plot(t, corrected_c2, label='Shifted Signal c2')
        plt.legend()
        plt.title('Signals')
        plt.xlabel('Retention time')
        plt.ylabel('Intensity')

        # Plot the fitting curve along with the data points
        plt.subplot(2, 1, 2)
        if data_points is not None:
            plt.plot(data_points[0], data_points[1], 'o', label='Time shifts')

        if fit_type == 'spline' and fit_params is not None:
            spline = fit_params
            plt.plot(t, spline(t), label='Fitted Spline')

        elif fit_type == 'quadratic' and fit_params is not None:
            a, b, c = fit_params
            quadratic_fit = c + b * data_points[0] + a * data_points[0] ** 2
            plt.plot(data_points[0], quadratic_fit, label='Quadratic Fit', color='orange')

        elif fit_type == 'linear' and fit_params is not None:
            a, b = fit_params
            linear_shift = a * t + b
            plt.plot(t, linear_shift, label='Linear Fit')

        plt.legend()
        plt.title(f'{fit_type.capitalize()} Fit')
        plt.xlabel('Retention time')
        plt.ylabel('Lag')
        plt.tight_layout()
        plt.show()





