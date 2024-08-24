import numpy as np
import pandas as pd
import os
import re

import utils
from data_loader import DataLoader
from sklearn.preprocessing import StandardScaler
from classification import Classifier
from dimensionality_reduction import DimensionalityReducer
from visualizer import Visualizer
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks, peak_prominences
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.ndimage import gaussian_filter1d
from utils import normalize_dict, remove_peak, min_max_normalize, normalize_data
from scipy.ndimage import gaussian_filter
import utils

class WineAnalysis:
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

    def normalize_all_chromatograms(self, chromatograms):
        normalized_chromatograms = {}
        for key, value in chromatograms.items():
            normalized_chromatograms[key] = self.normalize_chromatogram(value)
        return normalized_chromatograms

    def calculate_mean_chromatogram(self, chromatograms):
        all_data = np.array(list(chromatograms.values()))
        mean_chromatogram = np.mean(all_data, axis=0)
        return mean_chromatogram

    def shift_chromatogram(self, chrom, lag):
        shifted_chrom = np.roll(chrom, lag)
        if lag > 0:
            shifted_chrom[:lag] = 0  # Zero out the elements shifted in from the right
        elif lag < 0:
            shifted_chrom[lag:] = 0  # Zero out the elements shifted in from the left
        return shifted_chrom


    def scale_time_series(self, data, scale):
        original_time_points = np.arange(len(data))
        scaled_length = int(len(data) * scale)
        scaled_time_points = np.linspace(0, len(data) - 1, num=scaled_length)
        interp_func = interp1d(original_time_points, data, bounds_error=False, fill_value="extrapolate")
        scaled_data = interp_func(scaled_time_points)
        return scaled_data

    def find_optimal_offset(self, chrom_ref, chrom2):
        cross_corr = correlate(chrom_ref, chrom2, mode='full')
        lag = np.argmax(cross_corr) - len(chrom2) + 1
        return lag

    def find_optimal_scale_and_offset(self, chrom_ref, chrom2, scale_range, lag_range):
        best_scale = None
        best_lag = None
        best_corr = -np.inf

        for scale in scale_range:
            scaled_chrom2 = self.scale_time_series(chrom2, scale)
            cross_corr = correlate(chrom_ref, scaled_chrom2, mode='full')
            lag = np.argmax(cross_corr) - len(scaled_chrom2) + 1
            max_corr = np.max(cross_corr)

            if max_corr > best_corr:
                best_corr = max_corr
                best_scale = scale
                best_lag = lag

        return best_scale, best_lag


    def find_largest_peak(self, chromatogram, height=0.01):
        peaks, _ = find_peaks(chromatogram, height=height)
        largest_peak = peaks[np.argmax(chromatogram[peaks])]
        return largest_peak

    def find_second_highest_common_peak(self, chromatograms, tolerance=5):
        """
        Finds the second highest peak that is present in every chromatogram in chromatograms
        within a specified tolerance.

        Parameters:
        chromatograms (dict): Dictionary of chromatograms.
        tolerance (int): The tolerance within which peaks are considered the same.

        Returns:
        float: The second highest common peak intensity.
        int: The position of the second highest common peak.
        """
        peak_positions = []
        peak_values = []

        for label, chrom in chromatograms.items():
            # Find peaks
            peaks, properties = find_peaks(chrom, height=0.01)
            peak_positions.append(peaks)
            peak_values.append(properties['peak_heights'])

        if not peak_positions:
            return None, None

        # Find common peaks within the tolerance
        common_peaks = set(peak_positions[0])
        common_peak_values = {pos: peak_values[0][i] for i, pos in enumerate(peak_positions[0])}

        for i in range(1, len(peak_positions)):
            new_common_peaks = set()
            new_common_peak_values = {}
            for pos in peak_positions[i]:
                # Check if there is a peak within the tolerance in the common peaks
                for common_pos in common_peaks:
                    if abs(pos - common_pos) <= tolerance:
                        new_common_peaks.add(common_pos)
                        new_common_peak_values[common_pos] = max(common_peak_values[common_pos], peak_values[i][
                            np.where(peak_positions[i] == pos)[0][0]])
                        break

            common_peaks = new_common_peaks
            common_peak_values = new_common_peak_values

        if not common_peaks:
            return None, None

        # Sort the common peaks by their intensity
        sorted_common_peaks = sorted(common_peak_values.items(), key=lambda item: item[1], reverse=True)

        if len(sorted_common_peaks) < 2:
            return None, None

        # Get the second highest peak
        second_highest_common_peak_position = sorted_common_peaks[1][0]
        second_highest_common_peak_value = sorted_common_peaks[1][1]

        return second_highest_common_peak_value, second_highest_common_peak_position

    def find_highest_common_peak(self, chromatograms, tolerance=5):
        """
        Finds the highest peak that is present in every chromatogram in chromatograms
        within a specified tolerance.

        Parameters:
        chromatograms (dict): Dictionary of chromatograms.
        tolerance (int): The tolerance within which peaks are considered the same.

        Returns:
        float: The highest common peak intensity.
        int: The position of the highest common peak.
        """
        peak_positions = []
        peak_values = []

        for label, chrom in chromatograms.items():
            # Find peaks
            peaks, properties = find_peaks(chrom, height=0.01)
            peak_positions.append(peaks)
            peak_values.append(properties['peak_heights'])

        if not peak_positions:
            return None, None

        # Find common peaks within the tolerance
        common_peaks = set(peak_positions[0])
        common_peak_values = {pos: peak_values[0][i] for i, pos in enumerate(peak_positions[0])}

        for i in range(1, len(peak_positions)):
            new_common_peaks = set()
            new_common_peak_values = {}
            for pos in peak_positions[i]:
                # Check if there is a peak within the tolerance in the common peaks
                for common_pos in common_peaks:
                    if abs(pos - common_pos) <= tolerance:
                        new_common_peaks.add(common_pos)
                        new_common_peak_values[common_pos] = max(common_peak_values[common_pos], peak_values[i][
                            np.where(peak_positions[i] == pos)[0][0]])
                        break

            common_peaks = new_common_peaks
            common_peak_values = new_common_peak_values

        if not common_peaks:
            return None, None

        # Find the highest peak among the common peaks
        highest_common_peak_position = max(common_peak_values, key=common_peak_values.get)
        highest_common_peak_value = common_peak_values[highest_common_peak_position]

        return highest_common_peak_value, highest_common_peak_position

    def scale_chromatogram_to_reference(self, intensities, target_peak_index, reference_peak_index):
        """
        Scales the chromatogram to align the target peak index with the reference peak index.

        Parameters:
        intensities (np.array): The intensity values of the chromatogram.
        target_peak_index (int): The index of the target peak in the chromatogram.
        reference_peak_index (int): The desired index for the target peak.

        Returns:
        np.array: The scaled intensity values.
        """
        # Calculate the scaling factor
        scaling_factor = reference_peak_index / target_peak_index

        # Create an array of the original indices
        original_indices = np.arange(len(intensities))

        # Calculate the new indices after scaling
        scaled_indices = original_indices * scaling_factor

        # Interpolate the intensities to the new scaled indices
        interpolator = interp1d(scaled_indices, intensities, kind='linear', fill_value="extrapolate")
        scaled_intensities = interpolator(original_indices)

        return scaled_intensities

    def sync_chromatograms(self, reference_chrom, chrom2, ref_peak_pos=None):

        # Rough alignment using cross-correlation
        # best_scale, best_lag = self.find_optimal_scale_and_offset(mean_c1, chrom2, scale_range, lag_range)
        best_lag = int(self.find_optimal_offset(reference_chrom, chrom2))

        # scaled_chrom2 = self.scale_time_series(chrom2, best_scale)
        shifted_chrom2 = np.roll(chrom2, best_lag)

        if best_lag > 0:
            shifted_chrom2[:best_lag] = 0  # Zero out the elements shifted in from the right
        elif best_lag < 0:
            shifted_chrom2[best_lag:] = 0  # Zero out the elements shifted in from the left

        # Find the largest peak in mean_c1
        largest_peak_chrom1 = self.find_largest_peak(reference_chrom, 0.01)

        # Find the closest peak in shifted_chrom2 to the largest peak in mean_c1
        peaks_chrom2, _ = find_peaks(shifted_chrom2, height=0.01)
        # closest_peak_chrom2 = min(peaks_chrom2, key=lambda p: abs(p - largest_peak_chrom1))
        closest_peak_chrom2 = min(peaks_chrom2, key=lambda p: abs(p - ref_peak_pos))

        # Calculate the scale factor to align the peaks
        # peak_distance = closest_peak_chrom2 - largest_peak_chrom1
        peak_distance = closest_peak_chrom2 - ref_peak_pos
        fine_scaled_chrom2 = self.scale_chromatogram_to_reference(chrom2, closest_peak_chrom2 - best_lag, ref_peak_pos)
        # fine_scaled_chrom2 = self.scale_chromatogram(chrom2, closest_peak_chrom2 - best_lag, ref_peak_pos)
        # fine_scaled_chrom2 = self.scale_time_series(shifted_chrom2, 1 - peak_distance / len(chrom2))
        # fine_scaled_chrom2 = self.scale_time_series(chrom2, 1 - (peak_distance - best_lag) / len(chrom2))
        fine_scaled_chrom2 = self.resample_chromatogram(fine_scaled_chrom2, len(reference_chrom))

        return fine_scaled_chrom2

    def sync_and_plot_chromatograms(
            self, chromatograms1, chromatograms2, label_to_plot=None, extra_label=None, window_size=5000, overlap=0):

        if not label_to_plot or label_to_plot not in chromatograms1 or label_to_plot not in chromatograms2:
            print(f"Label '{label_to_plot}' not found in both chromatograms.")
            return

        chrom1 = chromatograms1[label_to_plot]
        chrom2 = chromatograms2[label_to_plot]

        lag_range = range(-500, 501)

        #  Estimate the lag of the first part of the chromatogram to sync both chrom from the beginning
        best_scale, best_lag, _ = self.find_best_scale_and_lag(chrom1[:5000], chrom2[:5000], np.array((1,)), 500)
        chrom2_shifted = self.shift_chromatogram(chrom2, best_lag)

        scale_range = np.linspace(0.9, 1.1, 1000)
        best_scale, _, best_corr = self.find_best_scale_and_lag(chrom1[:15000], chrom2_shifted[:15000], scale_range, 500)
        chrom2_sync = self.scale_chromatogram(chrom2_shifted, 0.998)

        # num_sections = 20  # Number of sections to divide the chromatograms
        # mean_c1 = self.normalize_chromatogram(self.calculate_mean_chromatogram(chromatograms1))
        # mean_c1 = self.calculate_mean_chromatogram(chromatograms1)
        # optimized_chrom2 = self.sync_chromatograms(chrom1, chrom2)
        # scale_range = np.linspace(0.7, 1.3, 100)
        # optimized_chrom2, lag, best_scale = self.sync_chromatograms(chrom1, chrom2, scale_range)

        # optimized_chrom2 = self.align_chromatogram_with_windows(chrom1, chrom2, window_size=window_size, overlap=overlap)

        # num_iterations = 10
        # for iteration in range(num_iterations):
        #     optimized_chrom2 = self.align_chromatogram_with_windows(chrom1, optimized_chrom2, 5000, 0)
        #     print(f"Iteration {iteration + 1}/{num_iterations} complete")

        # # Optimize offset
        # offset = self.find_optimal_offset_for_pair(chrom1, chrom2)
        # shifted_chrom2 = self.shift_chromatogram(chrom2, offset)

        # # Optimize offset and scaling factor
        # offset, scaling_factor, optimized_chrom2 = self.optimize_offset_and_scaling(chrom1, chrom2)

        # # Apply DTW to find the best alignment path
        # path = self.dtw_chromatogram(chrom1, shifted_chrom2, window=None)
        # optimized_chrom2 = self.warp_chromatogram(shifted_chrom2, path, len(chrom1))

        # # Smooth the chromatograms to reduce sharp peaks
        # optimized_chrom2 = self.smooth_chromatogram(optimized_chrom2, sigma=2)

        # Plot original and synchronized chromatograms
        plt.figure(figsize=(12, 8))

        # Original chromatograms
        plt.subplot(2, 1, 1)
        plt.plot(chrom1, label=f'{label_to_plot} (2018)', color='blue')
        plt.plot(chrom2, label=f'{label_to_plot} (2022)', color='red')
        if extra_label and extra_label in chromatograms1:
            extra_chrom1 = chromatograms1[extra_label]
            plt.plot(extra_chrom1, label=f'{extra_label} (2018)', color='green')
        plt.title(f'Original Chromatograms for {label_to_plot} and {extra_label if extra_label else ""}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Intensity')
        plt.legend()

        # Synchronized chromatograms
        plt.subplot(2, 1, 2)
        plt.plot(chrom1, label=f'{label_to_plot} (2018)', color='blue')
        # plt.plot(chrom2_shifted, label=f'{label_to_plot} (2022, Synchronized)', color='red')
        plt.plot(chrom2_sync, label=f'{label_to_plot} (2022, Synchronized)', color='red')
        if extra_label and extra_label in chromatograms1:
            extra_chrom1 = self.normalize_chromatogram(chromatograms1[extra_label])
            plt.plot(extra_chrom1, label=f'{extra_label} (File 1)', color='green')
        plt.title(f'Synchronized Chromatograms for {label_to_plot} and {extra_label if extra_label else ""}')
        # plt.title(f'Synchronized Chromatograms for {label_to_plot} and {extra_label if extra_label else ""}\n(Offset: {offset}')
        # plt.title(f'Synchronized Chromatograms for {label_to_plot} and {extra_label if extra_label else ""}\n(Offset: {offset}, Scaling Factor: {scaling_factor:.2f})')
        plt.xlabel('Time')
        plt.ylabel('Normalized Intensity')
        plt.legend()

        plt.tight_layout(pad=3.0)
        plt.show()

    def merge_chromatograms(self, chrom1, chrom2, norm=False):
        """
        Merges two chromatogram dictionaries, normalizes them, and handles duplicate sample names.

        Parameters:
        chrom1 (dict): The first chromatogram dictionary.
        chrom2 (dict): The second chromatogram dictionary.

        Returns:
        dict: The merged and normalized chromatogram data.
        """
        sc_inst = SyncChromatograms
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


    def shift_chromatograms(self, chromatograms, offset):
        """
        Shifts all chromatograms by the given offset.

        Parameters:
        chromatograms (dict): The chromatogram data as a dictionary.
        offset (int): The offset by which to shift the chromatograms.

        Returns:
        dict: The shifted chromatogram data.
        """
        shifted_chromatograms = {key: np.roll(value, int(offset)) for key, value in chromatograms.items()}
        return shifted_chromatograms

    def find_scaling_factor(self, mean_chromatogram1, mean_chromatogram2):
        x1 = np.linspace(0, 1, len(mean_chromatogram1))
        x2 = np.linspace(0, 1, len(mean_chromatogram2))

        best_scale = 1.0
        best_diff = np.inf
        for scale in np.linspace(0.9, 1.1, 100):
            scaled_x2 = x2 * scale
            f = interp1d(scaled_x2, mean_chromatogram2, bounds_error=False, fill_value="extrapolate")
            scaled_chromatogram2 = f(x1)
            valid_indices = (scaled_x2 >= 0) & (scaled_x2 <= 1)  # Only consider valid range
            diff = np.sum((mean_chromatogram1[valid_indices] - scaled_chromatogram2[valid_indices]) ** 2)
            if diff < best_diff:
                best_diff = diff
                best_scale = scale

        return best_scale
#
    def scale_chromatograms(self, chromatograms, scale):
        """
        Scales the x-dimension of all chromatograms by the given scaling factor.

        Parameters:
        chromatograms (dict): The chromatogram data as a dictionary.
        scale (float): The scaling factor by which to scale the x-dimension of the chromatograms.

        Returns:
        dict: The scaled chromatogram data.
        """
        scaled_chromatograms = {}
        for key, value in chromatograms.items():
            x_old = np.linspace(0, 1, len(value))
            scaled_x = x_old * scale
            f = interp1d(scaled_x, value, bounds_error=False, fill_value="extrapolate")
            scaled_chromatograms[key] = f(x_old)
        return scaled_chromatograms

    def plot_chromatograms(self, chromatogram1, chromatogram2, file_name1, file_name2, cl):
        """
        Plots multiple chromatograms for comparison and includes the mean chromatograms.

        Parameters:
        chromatograms1 (dict): The first chromatogram data as a dictionary.
        chromatograms2 (dict): The second chromatogram data as a dictionary.
        file_name1 (str): The name of the first file.
        file_name2 (str): The name of the second file.
        cl (ChromatogramLoader): Instance of ChromatogramLoader to use its methods.
        """
        plt.figure(figsize=(12, 12))

        # Plot original mean chromatograms
        plt.subplot(3, 1, 1)
        plt.plot(chromatogram1, label=f'Mean {file_name1}', color='blue')
        plt.plot(chromatogram2, label=f'Mean {file_name2}', color='red')
        plt.title('Original Mean Chromatograms')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.legend()

        # Shift chromatograms and plot
        best_scale, best_lag, best_corr = self.find_best_scale_and_lag(
            chromatogram1[:5000], chromatogram2[:5000], np.array((1,)), 500
        )
        chrom2_shifted = self.shift_chromatogram(chromatogram2, best_lag)
        chrom2_sync = self.scale_chromatogram(chrom2_shifted, 0.998)


        # optimal_offset = cl.find_optimal_offset(chromatogram1, chromatogram2)
        # shifted_chromatograms2 = cl.shift_chromatograms(chromatogram2, optimal_offset)
        # shifted_mean_chromatogram2 = cl.calculate_mean_chromatogram(shifted_chromatograms2)

        plt.subplot(3, 1, 2)
        plt.plot(chromatogram1, label=f'Mean {file_name1}', color='blue')
        plt.plot(chrom2_shifted, label=f'Mean {file_name2} (shifted)', color='red')
        plt.title(f'Shifted Mean Chromatograms (Offset: {best_lag})')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.legend()

        # # Scale chromatograms and plot
        # scaling_factor = cl.find_scaling_factor(mean_chromatogram1, shifted_mean_chromatogram2)
        # scaled_chromatograms2 = cl.scale_chromatograms(shifted_chromatograms2, scaling_factor)
        # scaled_mean_chromatogram2 = cl.calculate_mean_chromatogram(scaled_chromatograms2)

        plt.subplot(3, 1, 3)
        plt.plot(chromatogram1, label=f'Mean {file_name1}', color='blue')
        plt.plot(chrom2_sync, label=f'Mean {file_name2} (shifted & scaled)', color='red')
        plt.title(f'Shifted and Scaled Mean Chromatograms (Offset: {best_lag}, Scale: {0.998})')
        plt.xlabel('Time')
        plt.ylabel('Intensity')
        plt.legend()

        plt.tight_layout()
        plt.show()


    def min_max_normalize(self, data, min_range=0, max_range=1):
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = min_range + ((data - min_val) * (max_range - min_range) / (max_val - min_val))
        return normalized_data

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
    def sync_and_scale_chromatograms(self, cl, chrom1, chrom2):

        mean_chromatogram1 = cl.calculate_mean_chromatogram(chrom1)
        mean_chromatogram2 = cl.calculate_mean_chromatogram(chrom2)
        optimal_offset = cl.find_optimal_offset(mean_chromatogram1, mean_chromatogram2)
        shifted_chromatograms2 = cl.shift_chromatograms(chrom2, optimal_offset)
        shifted_mean_chromatogram2 = cl.calculate_mean_chromatogram(shifted_chromatograms2)
        scaling_factor = cl.find_scaling_factor(mean_chromatogram1, shifted_mean_chromatogram2)
        scaled_chromatograms2 = cl.scale_chromatograms(shifted_chromatograms2, scaling_factor)
        return scaled_chromatograms2

    def find_optimal_offset_for_pair(self, chrom1, chrom2):
        cross_corr = np.correlate(chrom1, chrom2, mode='full')
        lag = np.argmax(cross_corr) - len(chrom2) + 1
        return lag



    def find_scaling_factor_for_pair(self, chrom1, chrom2):
        """
        Finds the scaling factor to further align the two chromatograms by adjusting the x-dimension.

        Parameters:
        chrom1 (np.ndarray): The first chromatogram data.
        chrom2 (np.ndarray): The second chromatogram data.

        Returns:
        float: The scaling factor.
        """
        x1 = np.linspace(0, 1, len(chrom1))
        x2 = np.linspace(0, 1, len(chrom2))

        def objective(scale):
            scaled_x2 = x2 * scale
            f = interp1d(scaled_x2, chrom2, bounds_error=False, fill_value="extrapolate")
            scaled_chrom2 = f(x1)
            if len(scaled_chrom2) != len(chrom1):
                return np.inf  # Return a high value if the lengths don't match
            valid_indices = (scaled_x2 >= 0) & (scaled_x2 <= 1)# Only consider valid range
            return np.sum((chrom1[valid_indices] - scaled_chrom2[valid_indices]) ** 2)

        result = minimize(objective, 1.0, method='Powell', bounds=[(0.5, 2.0)])
        return result.x[0]

    def scale_chromatogram(self, chrom, scale):
        x = np.arange(len(chrom))
        scaled_length = int(len(chrom) * scale)
        scaled_x = np.linspace(0, len(chrom) - 1, num=scaled_length)
        f = interp1d(x, chrom, bounds_error=False, fill_value="extrapolate")
        return f(scaled_x)  # Ensure the scaled array matches the original length

    def sync_individual_chromatograms(self, mean_c1, chromatograms, scales, algo=None, initial_lag=300, lag_res=None):
        from wine_analysis import SyncChromatograms
        synced_chromatograms = {}
        for i, key in enumerate(chromatograms.keys()):
            # print(i, key, end=" ")
            print(i, key)
            chrom = chromatograms[key]
            sync_chrom = SyncChromatograms(
                mean_c1, chrom, 1, scales, 1E6, threshold=0.00, max_sep_threshold=50, peak_prominence=0.00
                )
            # sync_chrom.lag_res = utils.calculate_lag_corr(mean_c1, chrom, 4000, extend=0, hop=2000, sigma=20)
            # sync_chrom.lag_res = utils.calculate_lag(mean_c1, chrom, 4000, lag_range=initial_lag, hop=1000, sigma=50, distance_metric='L2')

            if algo in [3, 4, 5]:
                if lag_res:
                    sync_chrom.lag_res = lag_res  # use shift profile from mean chromatogram
                # else:
                    # get individual shift profile
                    # sync_chrom.lag_res = utils.calculate_lag_profile(
                    #     mean_c1, chrom, 2000, lag_range=initial_lag, hop=100, sigma=20, distance_metric='mse',
                    #     init_min_dist=0.15
                    # )
                    # sync_chrom.lag_res = utils.calculate_lag_profile(
                    #     mean_c1, chrom, 2000, lag_range=initial_lag, hop=1000, sigma=20, distance_metric='mse',
                    #     init_min_dist=0.15
                    # )
                    # sync_chrom.lag_res = utils.calculate_lag_profile(
                    #     mean_c1, chrom, 4000, lag_range=initial_lag, hop=2000, sigma=20, distance_metric='corr',
                    #     init_min_dist=1E6
                    # )
                    # sync_chrom.lag_res = sync_chrom.lag_profile_from_peaks(
                    #     mean_c1, chrom, proximity_threshold=25, peak_prominence=sync_chrom.peak_prominence, nsegments=100
                    # )

            optimized_chrom = sync_chrom.adjust_chromatogram(algo=algo, initial_lag=initial_lag)

            synced_chromatograms[key] = optimized_chrom

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

    def scale_peaks_location(self, peaks, scale):
        return peaks * scale

    def find_largest_peaks(self, segment, num_peaks):
        peaks, _ = find_peaks(segment, height=self.threshold)
        if len(peaks) > num_peaks:
            largest_peaks_indices = np.argsort(segment[peaks])[-num_peaks:]
            largest_peaks = peaks[largest_peaks_indices]
        else:
            largest_peaks = peaks
        return np.sort(largest_peaks)

    def pad_with_zeros(self, array, pad_width):
        return np.pad(array, (pad_width, 0), 'constant')

    def find_best_scale_and_lag_l2(self, c1_segment, c2_segment, max_lag=2500):
        best_scale = None
        best_lag = None
        best_diff = np.inf

        for scale in self.scales:
            scaled_c2_segment = self.scale_chromatogram(c2_segment, scale)
            _, best_lag_for_scale = self.cross_correlation_l2_norm(c1_segment, scaled_c2_segment, max_lag)

            if best_lag_for_scale is None:
                continue

            shifted_scaled_c2_segment = np.roll(scaled_c2_segment, best_lag_for_scale)
            if best_lag_for_scale > 0:
                shifted_scaled_c2_segment[:best_lag_for_scale] = 0
            else:
                shifted_scaled_c2_segment[best_lag_for_scale:] = 0

            min_length = min(len(c1_segment), len(shifted_scaled_c2_segment))
            c1_segment_trimmed = c1_segment[:min_length]
            shifted_scaled_c2_segment_trimmed = shifted_scaled_c2_segment[:min_length]

            diff = np.sqrt(np.sum((c1_segment_trimmed - shifted_scaled_c2_segment_trimmed)**2))

            if diff < best_diff:
                best_diff = diff
                best_scale = scale
                best_lag = best_lag_for_scale

        return best_scale, best_lag


    def cross_correlation_l2_norm(self, signal1, signal2, max_lag):
        len1 = len(signal1)
        len2 = len(signal2)
        result = np.zeros(2 * max_lag + 1)

        for lag in range(-max_lag, max_lag + 1):
            if lag < 0:
                shifted_signal2 = np.concatenate([np.zeros(-lag), signal2[:len2 + lag]])
                signal1_part = signal1[:len1 + lag]
            else:
                shifted_signal2 = np.concatenate([signal2[lag:], np.zeros(lag)])
                signal1_part = signal1[lag:]

            min_length = min(len(shifted_signal2), len(signal1_part))
            shifted_signal2 = shifted_signal2[:min_length]
            signal1_part = signal1_part[:min_length]

            result[lag + max_lag] = np.sqrt(np.sum((shifted_signal2 - signal1_part) ** 2))

        min_lag_index = np.argmin(result)
        min_lag = min_lag_index - max_lag

        return result, min_lag

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

    def scale_and_shift_in_sections(self, c2, nsections, lag=300):
        def normalize_signal(signal):
            min_val = np.min(signal)
            max_val = np.max(signal)
            normalized_signal = (signal - min_val) / (max_val - min_val)
            return normalized_signal

        segment_length = len(c2) // nsections
        # final_corrected_c2 = []
        final_corrected_c2 = c2.copy()

        for i in range(nsections):
            start = i * segment_length
            end = (i + 1) * segment_length if i < nsections - 1 else len(c2)

            c1_segment = self.c1[start:end]
            c2_segment = final_corrected_c2[start:end]

            if len(c1_segment) == 0 or len(c2_segment) == 0:
                continue

            best_scale, best_lag, corr = self.find_best_scale_and_lag_corr(
                normalize_signal(gaussian_filter(c1_segment, 20)),
                normalize_signal(gaussian_filter(c2_segment, 20)),
                np.linspace(1.0, 1.1, 1),
                max_lag=self.max_sep_threshold
            )
            best_lag = len(c1_segment) - len(c2_segment) + best_lag

            print("    ",  best_scale, best_lag)

            # best_scale, best_lag = self.find_best_scale_and_lag(c1_segment, c2_segment, initial_lag=lag, nsegments=30)
            # best_scale, best_lag = self.find_best_scale_and_lag_dtw(c1_segment, c2_segment, initial_lag=50, nsegments=300)
            if best_scale is None or best_lag is None:
                continue
            if corr < 20 or np.abs(best_lag) >= self.max_sep_threshold:
                 best_scale, best_lag = 1, 0

            corrected_segment = self.correct_segment(c2_segment, best_scale, best_lag)

            if best_lag < 0:
                final_corrected_c2 = np.concatenate(
                    [final_corrected_c2[:start], corrected_segment[:best_lag], final_corrected_c2[end:]]
                )
            else:
                final_corrected_c2 = np.concatenate(
                    [final_corrected_c2[:start],
                     np.concatenate([np.repeat(corrected_segment[0], best_lag), corrected_segment]),
                     final_corrected_c2[end:]]
                )

            # final_corrected_c2 = np.concatenate([final_corrected_c2[:start], corrected_segment, final_corrected_c2[end:]])

            # final_corrected_c2.extend(corrected_segment)

        return np.array(final_corrected_c2)


    def find_best_scale_and_lag(self, c1_segment, c2_segment, scales=None, initial_lag=None, lag_hop=1, nsegments=10, dtw=None):
        best_scale = None
        best_lag = None
        best_diff = np.inf
        if not scales:
            scales = self.scales

        c1_peaks, _ = find_peaks(c1_segment)
        #  remove peak for standard in reference
        index_to_remove = np.where(c1_peaks == 8918)[0]
        c1_peaks = np.delete(c1_peaks, index_to_remove)
        c1_peaks, valid = self.ensure_peaks_in_segments(c1_segment, c1_peaks, num_segments=2*nsegments)
        if not valid:
            raise ValueError('Missing peaks in peaks_c1 (find_best_scale_and_lag())')

        c2_peaks_scaled, _ = find_peaks(c2_segment)
        c2_peaks_scaled, valid = self.ensure_peaks_in_segments(c2_segment, c2_peaks_scaled, num_segments=nsegments)
        if not valid:
            raise ValueError('Missing peaks in peaks_c2 (find_best_scale_and_lag())')

        if len(c1_peaks) == 0:
            return best_scale, best_lag

        peaks_c1 = np.sort(c1_peaks)

        lag_range = int(0.2 * len(c1_segment))

        if initial_lag is not None:
            lag_start = -initial_lag
            lag_end = initial_lag
        else:
            lag_start = -lag_range
            lag_end = lag_range

        c2_peaks_scaled_orig = c2_peaks_scaled.copy()
        for lag in range(lag_start, lag_end + 1, lag_hop):
            for scale in scales:
                c2_peaks_scaled = self.scale_peaks_location(c2_peaks_scaled_orig, scale)
                # scaled_c2_segment = self.scale_chromatogram(c2_segment, scale)
                # peaks_scaled_c2, _ = find_peaks(scaled_c2_segment)
                # peaks_scaled_c2, _ = self.ensure_peaks_in_segments(scaled_c2_segment, peaks_scaled_c2, num_segments=10)

                if len(c2_peaks_scaled) == 0:
                    continue

                c2_peaks_scaled = np.sort(c2_peaks_scaled)
                c2_shifted_peaks_scaled = c2_peaks_scaled + lag
                valid_indices = (c2_shifted_peaks_scaled >= 0) & (c2_shifted_peaks_scaled < len(c1_segment))
                c2_shifted_peaks_scaled = c2_shifted_peaks_scaled[valid_indices]

                if len(c2_shifted_peaks_scaled) == 0:
                    continue

                if dtw:
                    # Calculate DTW distance
                    distance, _ = fastdtw(c1_peaks, c2_shifted_peaks_scaled)
                else:
                    distances = []
                    for peak1 in c1_peaks:
                        closest_peak2 = c2_shifted_peaks_scaled[np.argmin(np.abs(c2_shifted_peaks_scaled - peak1))]
                        distances.append(abs(peak1 - closest_peak2))
                    if len(distances) == 0:
                        continue
                    distance = np.mean(distances)

                if distance < best_diff:
                    best_diff = distance
                    best_scale = scale
                    best_lag = lag

        return best_scale, best_lag


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




    def scale_between_peaks(self, c1, c2, proximity_threshold, peak_prominence, nsegments):
        # Find peaks
        threshold_c1 = min(c1) + np.std(c1) * self.threshold
        threshold_c2 = min(c2) + np.std(c2) * self.threshold
        # c1_peaks, _ = find_peaks(c1, height=threshold_c1, prominence=peak_prominence)
        # c2_peaks, _ = find_peaks(c2, height=threshold_c2, prominence=peak_prominence)
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


    def lag_profile_from_peaks(self, c1, c2, proximity_threshold, nsegments, global_align=True):

        if global_align:
            # Global time shift
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(c1[:10000], 50),
                gaussian_filter(c2[:10000], 50),
                np.linspace(1., 1.1, 1)
            )
            c2_aligned = gaussian_filter(self.correct_segment(c2, best_scale, best_lag), 5)
            # c2_aligned = self.correct_segment(c2, best_scale, best_lag)
            accum_lag = best_lag
        else:
            c2_aligned = c2
            accum_lag = 0

        c1_peaks, _ = find_peaks(c1)
        #  remove peak for standard in reference
        index_to_remove = np.where(c1_peaks == 8918)[0]
        c1_peaks = np.delete(c1_peaks, index_to_remove)
        c2_peaks, _ = find_peaks(c2_aligned)
        c1_peaks, valid = self.ensure_peaks_in_segments(c1, c1_peaks, num_segments=nsegments)
        c2_peaks, valid = self.ensure_peaks_in_segments(c2_aligned, c2_peaks, num_segments=nsegments)

        if not valid:
            raise ValueError(
                "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks

        # Add zero to also scale the first part of the chromatogram
        c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
        c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
        initial_npeaks = min(len(c2_peaks), len(c1_peaks))

        lags = []
        lags_loc = []
        # cnt = 0
        for i in range(1, initial_npeaks):
            c2p_prev = c2_peaks[i - 1]
            c2p = c2_peaks[i]
            c1p_prev = c2p_prev
            c1p = c1_peaks[np.argmin(np.abs(c1_peaks - c2p))]

            # If the current peak in c2 is too far from the closest peak in c1, skip it
            if np.abs(c1p - c2p) > proximity_threshold or c1p <= c1p_prev:
                if i >= len(c2_peaks) - 1:
                    break
                else:
                    continue

            accum_lag += c1p - c2p

            # Scale the interval
            interval_c2 = c2p - c2p_prev
            interval_c1 = c1p - c1p_prev
            if interval_c2 == 0:
                continue
            scale = interval_c1 / interval_c2
            start = min(c2p_prev, c2p)
            end = max(c2p_prev, c2p)
            c2_segment = c2_aligned[start:end]
            scaled_segment = self.scale_chromatogram(c2_segment, scale)
            # corr1 = correlate(min_max_normalize(c1[c1p_prev:c1p]), min_max_normalize(c2_segment))
            # corr2 = correlate(min_max_normalize(c1[c1p_prev:c1p]), min_max_normalize(scaled_segment))
            # if max(corr2) < max(corr1) + max(corr1) * 0.00:
            #     scaled_segment = c2_segment
            #     accum_lag -= c1p - c2p
            temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], scaled_segment, c2_aligned[end:]])

            # best_lag = np.argmax(corr) - len(c2_segment) + 1

            # cut = min(len(temp_c2_aligned_1), len(c1))
            # current_mse = np.mean((np.concatenate([c2_aligned[:start], scaled_segment]) - c1[:(start + len(scaled_segment))]) ** 2)
            # print(current_mse)
            # if current_mse > max_mse:
            #     continue

            # tol = 10
            # if np.abs(c1p - c2p) >= np.abs(best_lag) + tol:
            #     if len(lags_loc) > 0:
            #         lags.append(lags[-1])
            #         lags_loc.append(lags_loc[-1])
            #         continue


            temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
            c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments)
            if not valid:
                raise ValueError(
                    "Please change parameters, some segments do not have peaks.")
            c2_peaks = np.concatenate([np.array((0,)), c2_peaks])

            if len(c2_peaks) < initial_npeaks and i >= len(c2_peaks) - 1:  # or len(c2_peaks) != nsegments:
                # continue
                break

            c2_aligned = temp_c2_aligned_1

            #  Avoid repeated peak locations or update those that moved to less retention times
            # if len(lags_loc) > 0 and c2p + accum_lag <= lags_loc[-1]:
            if len(lags_loc) > 0 and any(c2p + accum_lag <= loc for loc in lags_loc):
                continue
                lags_loc = lags_loc[:-1]
                lags = lags[:-1]

            lags.append(accum_lag)
            lags_loc.append(c2p + accum_lag)

        # Add one last point equal to the last one at 300000
        if len(lags) > 0:
            lags.insert(0, lags[0])
            lags_loc.insert(0, 0)

        return np.array(lags_loc), np.array(lags)



    def lag_profile_from_peaks2(self, c1, c2, proximity_threshold, nsegments, global_align=True, scan_range=1):

        if global_align:
            # Global time shift
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(c1[:10000], 50),
                gaussian_filter(c2[:10000], 50),
                np.linspace(1., 1.1, 1)
            )
            c2_aligned = gaussian_filter(self.correct_segment(c2, best_scale, best_lag), 5)
            # c2_aligned = self.correct_segment(c2, best_scale, best_lag)
            accum_lag = best_lag
        else:
            c2_aligned = c2
            accum_lag = 0

        c1_peaks, _ = find_peaks(c1)
        #  remove peak for standard in reference
        index_to_remove = np.where(c1_peaks == 8918)[0]
        c1_peaks = np.delete(c1_peaks, index_to_remove)
        c2_peaks, _ = find_peaks(c2_aligned)
        # c1_peaks, valid = self.ensure_peaks_in_segments(c1, c1_peaks, num_segments=nsegments)
        c2_peaks, valid = self.ensure_peaks_in_segments(c2_aligned, c2_peaks, num_segments=nsegments)

        if not valid:
            raise ValueError(
                "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks

        # Add zero to also scale the first part of the chromatogram
        c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
        c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
        initial_npeaks = min(len(c2_peaks), len(c1_peaks))

        lags = []
        lags_loc = []
        # cnt = 0
        for i in range(1, initial_npeaks):
            c2p_prev = c2_peaks[i - 1]
            c2p = c2_peaks[i]
            c1p_prev = c2p_prev
            closest_index = np.argmin(np.abs(c1_peaks - c2_peaks[i]))
            start_index = max(0, closest_index - scan_range)
            end_index = min(len(c1_peaks), closest_index + scan_range + 1)
            min_distance = 1E6
            min_corr = 0
            for idx in range(start_index, end_index):
                c1p = c1_peaks[idx]
                # Scale the interval
                interval_c2 = c2p - c2p_prev
                interval_c1 = c1p - c1p_prev
                start = min(c2p_prev, c2p)
                end = max(c2p_prev, c2p)
                if np.abs(c1p - c2p) > proximity_threshold or interval_c2 <= 0 or interval_c1 <= 0 or (end - start) <= 0:
                    continue
                scale = interval_c1 / interval_c2
                if scale < 0:
                    print('error here')
                c2_segment = c2_aligned[start:end]
                scaled_segment = self.scale_chromatogram(c2_segment, scale)

                if len(scaled_segment) == 0:
                    continue
                # corr = correlate(min_max_normalize(c1[c1p_prev:c1p]), min_max_normalize(scaled_segment))
                # corr_score = max(correlate(c1[c1p_prev:c1p] / c1[c1p], scaled_segment / c2_aligned[c2p]))
                # cut = min(len(c1[c1p_prev:c1p]), len(scaled_segment)) - 2 * proximity_threshold
                cut = min(len(c1[c1p_prev:c1p]), len(scaled_segment)) - int(0.25 * min(len(c1[c1p_prev:c1p]), len(scaled_segment)))
                mse_dist = np.mean(((c1[c1p_prev:c1p] / c1[c1p])[:cut] - (scaled_segment / c2_aligned[c2p])[:cut]) ** 2)
                # distance = np.mean((min_max_normalize(c1[c1p_prev:c1p_prev + cut]) - min_max_normalize(scaled_segment[:cut])) ** 2)
                # distance = np.mean((min_max_normalize(c1[c1p_prev:c1p]) - min_max_normalize(scaled_segment)) ** 2)
                if mse_dist < min_distance:
                # if corr_score > min_corr:
                #     min_corr = corr_score
                    min_distance = mse_dist
                    best_scaled_segment = scaled_segment
                    best_end = end
                    best_c1p = c1p

            try:
                best_scaled_segment
            except:
                best_c1p = c1p
                best_scaled_segment = c2_aligned[start:end]

            # If the current peak in c2 is too far from the closest peak in c1, skip it
            if np.abs(best_c1p - c2p) > proximity_threshold or best_c1p <= c1p_prev:
                if i >= len(c2_peaks) - 1:
                    break
                else:
                    continue
            accum_lag += best_c1p - c2p

            temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, c2_aligned[best_end:]])

            # best_lag = np.argmax(corr) - len(c2_segment) + 1

            # cut = min(len(temp_c2_aligned_1), len(c1))
            # current_mse = np.mean((np.concatenate([c2_aligned[:start], scaled_segment]) - c1[:(start + len(scaled_segment))]) ** 2)
            # print(current_mse)
            # if current_mse > max_mse:
            #     continue

            # tol = 10
            # if np.abs(c1p - c2p) >= np.abs(best_lag) + tol:
            #     if len(lags_loc) > 0:
            #         lags.append(lags[-1])
            #         lags_loc.append(lags_loc[-1])
            #         continue


            temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
            c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments)
            if not valid:
                raise ValueError(
                    "Please change parameters, some segments do not have peaks.")
            c2_peaks = np.concatenate([np.array((0,)), c2_peaks])

            if len(c2_peaks) < initial_npeaks and i >= len(c2_peaks) - 1:  # or len(c2_peaks) != nsegments:
                # continue
                break

            c2_aligned = temp_c2_aligned_1

            #  Avoid repeated peak locations or update those that moved to less retention times
            # if len(lags_loc) > 0 and c2p + accum_lag <= lags_loc[-1]:
            if len(lags_loc) > 0 and any(c2p + accum_lag <= loc for loc in lags_loc):
                continue
                lags_loc = lags_loc[:-1]
                lags = lags[:-1]

            lags.append(accum_lag)
            lags_loc.append(c2p + accum_lag)

        # Add one last point equal to the last one at 300000
        if len(lags) > 0:
            lags.insert(0, lags[0])
            lags_loc.insert(0, 0)

        return np.array(lags_loc), np.array(lags)


    # def lag_profile_from_peaks3(self, c1, c2, proximity_threshold, nsegments, global_align=True, scan_range=1):
    #
    #     if global_align:
    #         # Global time shift
    #         best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
    #             gaussian_filter(c1[:10000], 50),
    #             gaussian_filter(c2[:10000], 50),
    #             np.linspace(1., 1.1, 1)
    #         )
    #         c2_aligned = gaussian_filter(self.correct_segment(c2, best_scale, best_lag), 5)
    #         # c2_aligned = self.correct_segment(c2, best_scale, best_lag)
    #         accum_lag = best_lag
    #     else:
    #         c2_aligned = gaussian_filter(c2, 5)
    #         accum_lag = 0
    #
    #     c1 = gaussian_filter(c1, 5)
    #     c1_peaks, _ = find_peaks(c1)
    #     #  remove peak for standard in reference
    #     index_to_remove = np.where(c1_peaks == 8918)[0]
    #     c1_peaks = np.delete(c1_peaks, index_to_remove)
    #     c2_peaks, _ = find_peaks(c2_aligned)
    #     # c1_peaks, valid = self.ensure_peaks_in_segments(c1, c1_peaks, num_segments=nsegments)
    #     c2_peaks, valid = self.ensure_peaks_in_segments(c2_aligned, c2_peaks, num_segments=nsegments)
    #
    #     if not valid:
    #         raise ValueError(
    #             "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks
    #
    #     # Add zero to also scale the first part of the chromatogram
    #     c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
    #     c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
    #     initial_npeaks = min(len(c2_peaks), len(c1_peaks))
    #
    #     lags = []
    #     lags_loc = []
    #     interval_after = 2000
    #     for i in range(1, initial_npeaks):
    #         c2p_prev = c2_peaks[i - 1]
    #         c2p = c2_peaks[i]
    #         c1p_prev = c2p_prev
    #         closest_index = np.argmin(np.abs(c1_peaks - c2_peaks[i]))
    #         start_index = max(0, closest_index - scan_range)
    #         end_index = min(len(c1_peaks), closest_index + scan_range + 1)
    #         min_distance = 1E6
    #         min_corr = 0
    #         start = min(c2p_prev, c2p)
    #         end = max(c2p_prev, c2p)
    #         for idx in range(start_index, end_index):
    #             c1p = c1_peaks[idx]
    #             if c1p > len(c1) - interval_after or c2p > len(c2_aligned) - interval_after:
    #                 continue
    #             # Scale the interval
    #             interval_c2 = c2p - c2p_prev
    #             interval_c1 = c1p - c1p_prev
    #             if np.abs(c1p - c2p) > proximity_threshold or interval_c2 <= 0 or interval_c1 <= 0 or (end - start) <= 0:
    #                 continue
    #             scale = interval_c1 / interval_c2
    #             if scale < 0:
    #                 print('error here')
    #             c2_segment = c2_aligned[start:end]
    #             scaled_segment = self.scale_chromatogram(c2_segment, scale)
    #
    #             if len(scaled_segment) == 0:
    #                 continue
    #
    #             # cut = min(len(c1[c1p_prev:c1p]), len(scaled_segment)) - int(0.1 * min(len(c1[c1p_prev:c1p]), len(scaled_segment)))
    #             # mse_dist = np.mean(((c1[c1p_prev:c1p] / c1[c1p])[:cut] - (scaled_segment / c2_aligned[c2p])[:cut]) ** 2)
    #             # mse_dist = np.mean((utils.normalize_signal_standard(c1[c1p_prev:c1p][:cut]) - utils.normalize_signal_standard(scaled_segment[:cut])) ** 2)
    #
    #             # c1_deriv = np.sign(np.diff(c1[c1p_prev:c1p + interval_after]))
    #             # c2_deriv = np.sign(np.diff((np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]]))))
    #             # cut = min(len(c1_deriv), len(c2_deriv))
    #             # mse_dist = np.mean(
    #             #     (np.sign(np.diff(c1[c1p_prev:c1p + interval_after]))[:cut]
    #             #      - np.sign(np.diff((np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]]))))[
    #             #        :cut]) ** 2)
    #             # cut = min(len(c1[c1p_prev:c1p + interval_after]) , len(np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]])))
    #             # mse_dist = np.mean(
    #             #     (utils.normalize_signal_standard(c1[c1p_prev:c1p + interval_after])[:cut]
    #             #      - utils.normalize_signal_standard(np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]]))[:cut]) ** 2)
    #
    #             corr = max(correlate(utils.normalize_signal_standard(c1[c1p_prev:c1p + interval_after]),
    #                           utils.normalize_signal_standard(
    #                               np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]]))))
    #             if corr > min_corr:
    #                 min_corr = corr
    #                 best_scaled_segment = scaled_segment
    #                 best_end = end
    #                 best_c1p = c1p
    #
    #             # cut = min(len(c1[c1p_prev:c1p]), len(scaled_segment))
    #             # mse_dist = np.mean((min_max_normalize(c1[c1p_prev:])[:cut] - min_max_normalize(np.concatenate([scaled_segment, c2_aligned[end:]]))[:cut]) ** 2)
    #
    #             # if mse_dist < min_distance:
    #             #     min_distance = mse_dist
    #             #     best_scaled_segment = scaled_segment
    #             #     best_end = end
    #             #     best_c1p = c1p
    #         try:
    #             best_scaled_segment
    #         except:
    #             best_end = max(c2p_prev, c2p)
    #             best_c1p = c1p
    #             best_scaled_segment = c2_aligned[start:end]
    #
    #         # If the current peak in c2 is too far from the closest peak in c1, skip it
    #         if np.abs(best_c1p - c2p) > proximity_threshold or best_c1p <= c1p_prev:
    #             if i >= len(c2_peaks) - 1:
    #                 break
    #             else:
    #                 continue
    #
    #         stretch = best_c1p - c2p
    #         # accum_lag += stretch
    #
    #         #  Contract or stretch next between-peak section to compensate for previous movement
    #         if i + 1 < len(c2_peaks) and len(c2_aligned[best_end:c2_peaks[i + 1]]) > np.abs(stretch):
    #             next_peak = c2_peaks[i + 1]
    #             next_segment = c2_aligned[best_end:next_peak]
    #             compensate_factor = len(next_segment) / (len(next_segment) + stretch)
    #             next_segment_corrected = self.scale_chromatogram(next_segment, compensate_factor)
    #             temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, next_segment_corrected, c2_aligned[next_peak:]])
    #         else:
    #             temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, c2_aligned[best_end:]])
    #         temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
    #         c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments)
    #         if not valid:
    #             raise ValueError(
    #                 "Please change parameters, some segments do not have peaks.")
    #         c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
    #
    #         if len(c2_peaks) < initial_npeaks and i >= len(c2_peaks) - 1:  # or len(c2_peaks) != nsegments:
    #             # continue
    #             break
    #
    #         c2_aligned = temp_c2_aligned_1
    #
    #         #  Avoid repeated peak locations or update those that moved to lower retention times
    #         # if len(lags_loc) > 0 and c2p + accum_lag <= lags_loc[-1]:
    #         if len(lags_loc) > 0 and any(c2p + stretch <= loc for loc in lags_loc):
    #             continue
    #             # lags_loc = lags_loc[:-1]
    #             # lags = lags[:-1]
    #
    #         # if min_distance > 0.1:
    #         #     lags.append(0)
    #         #     lags_loc.append(c2p + accum_lag)
    #         #     continue
    #         lags.append(stretch)
    #         lags_loc.append(c2p + stretch)
    #
    #     # Add one last point equal to the last one at 300000
    #     if len(lags) > 0:
    #         lags.append(lags[-1])
    #         lags_loc.append(30000)
    #     return np.array(lags_loc), np.array(lags)


    def lag_profile_from_peaks4(
            self, c1, c2, proximity_threshold, nsegments, global_align=True, scan_range=1, peak_ord=0, scale_dist=100,
            interval_after=500, min_avg_peak_dist=50):
        def calculate_average_peak_distance(signal1, signal2, prm=1E-6, distance_type='mean'):
            # Find peaks in both signals
            peaks1, _ = find_peaks(signal1, prominence=prm)
            peaks2, _ = find_peaks(signal2, prominence=prm)

            if len(peaks1) == 0 or len(peaks2) == 0:
                raise ValueError("No peaks found in one of the signals")

            # Match each peak in signal1 to the closest peak in signal2
            distances = []
            for peak1 in peaks1:
                closest_peak2 = peaks2[np.argmin(np.abs(peaks2 - peak1))]
                distance = abs(peak1 - closest_peak2)
                distances.append(distance)

            # Calculate the average distance
            if distance_type == 'mean':
                average_distance = np.mean(distances)
            elif distance_type == 'mean_square':
                average_distance = np.mean(np.square(distances))
            else:
                raise ValueError("Invalid distance_type. Choose 'mean' or 'mean_square'.")

            return average_distance, distances, peaks1, peaks2

        if global_align:
            # Global time shift
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(c1[:10000], 50),
                gaussian_filter(c2[:10000], 50),
                np.linspace(1., 1.1, 1)
            )
            c2_aligned = self.correct_segment(c2, best_scale, best_lag)
            # c2_aligned = self.correct_segment(c2, best_scale, best_lag)
            accum_lag = best_lag
        else:
            c2_aligned = c2
            accum_lag = 0

        c1_peaks, _ = find_peaks(c1)
        c2_peaks_all, _ = find_peaks(c2_aligned)
        # c2_peaks, properties = find_peaks(c2_aligned, prominence=0)
        # left_bases = properties['left_bases']
        # right_bases = properties['right_bases']
        # distances_to_minima = {
        #     "left_distance": c2_peaks - left_bases,
        #     "right_distance": right_bases - c2_peaks
        # }

        # c1_peaks, valid = self.ensure_peaks_in_segments(c1, c1_peaks, num_segments=nsegments)
        c2_peaks, valid = self.ensure_peaks_in_segments(c2_aligned, c2_peaks_all, num_segments=nsegments, peak_ord=peak_ord)
        # selected_peaks_idx = np.where(np.isin(find_peaks(c2_aligned, prominence=0)[0], c2_peaks))[0]
        # left_distances = distances_to_minima["left_distance"][selected_peaks_idx]
        # right_distances = distances_to_minima["right_distance"][selected_peaks_idx]
        if not valid:
            raise ValueError(
                "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks

        # Add zero to also scale the first part of the chromatogram
        c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
        c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
        initial_npeaks = min(len(c2_peaks), len(c1_peaks))

        lags = []
        lags_loc = []

        for i in range(1, initial_npeaks):
            # c2p_prev = c2_peaks[i - 1]
            try:
                c2_peaks_all, _ = find_peaks(c2_aligned)
                c2p_prev = c2_peaks_all[np.where(c2_peaks_all == c2_peaks[i])[0] - 1][0]
            except:
                print('''Error in c2_peaks_all''')
            c2p = c2_peaks[i]
            # c2p_prev = max(0, c2_peaks[i] - scale_dist)
            # c2p = c2_peaks[i]
            c1p_prev = c2p_prev
            closest_index = np.argmin(np.abs(c1_peaks - c2_peaks[i]))
            start_index = max(0, closest_index - scan_range)
            end_index = min(len(c1_peaks), closest_index + scan_range + 1)
            min_distance = min_avg_peak_dist  # Minimum average peak distance for accepting the scaling
            start = min(c2p_prev, c2p)
            end = max(c2p_prev, c2p)
            for idx in range(start_index, end_index):
                c1p = c1_peaks[idx]
                if c1p > len(c1) - interval_after or c2p > len(c2_aligned) - interval_after:
                    continue
                # Scale the interval
                interval_c2 = c2p - c2p_prev
                interval_c1 = c1p - c1p_prev
                if np.abs(c1p - c2p) > proximity_threshold or interval_c2 <= 0 or interval_c1 <= 0 or (end - start) <= 0:
                    continue
                scale = interval_c1 / interval_c2
                if scale < 0:
                    print('error here')
                c2_segment = c2_aligned[start:end]
                scaled_segment = self.scale_chromatogram(c2_segment, scale)

                if len(scaled_segment) == 0:
                    continue

                norm_c1_segment = utils.normalize_signal_standard(c1[c1p_prev:c1p + interval_after])
                norm_c2_segment = utils.normalize_signal_standard(
                                  # np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]])
                                  np.concatenate([scaled_segment, c2[end:end + interval_after]])
                )
                try:
                    avg_peak_dist = calculate_average_peak_distance(
                        norm_c1_segment, norm_c2_segment, prm=1E-6, distance_type='mean')[0]
                    # print(avg_peak_dist)
                except:
                    print('Error in avg_peak_dist')
                if avg_peak_dist < min_distance:
                    min_distance = avg_peak_dist
                    best_scaled_segment = scaled_segment
                    best_end = end
                    best_c1p = c1p

            try:
                best_scaled_segment
            except:
                best_end = max(c2p_prev, c2p)
                best_c1p = c1p
                best_scaled_segment = c2_aligned[start:end]

            # If the current peak in c2 is too far from the closest peak in c1, skip it
            if np.abs(best_c1p - c2p) > proximity_threshold or best_c1p <= c1p_prev:
                if i >= len(c2_peaks) - 1:
                    break
                else:
                    continue

            stretch = best_c1p - c2p
            # accum_lag += stretch

            #  Contract or stretch next between-peak section to compensate for previous movement
            next_peak = c2_peaks_all[np.where(c2_peaks_all == c2_peaks[i])[0] + 1][0]
            # if i + 1 < len(c2_peaks) and len(c2_aligned[best_end:c2_peaks[i + 1]]) > np.abs(stretch):
            if i + 1 < len(c2_peaks) and len(c2_aligned[best_end:next_peak]) > np.abs(stretch):
                # next_peak = c2_peaks[i + 1]
                # next_peak = c2_peaks[i] + scale_dist
                # next_peak = c2_peaks_all[np.where(c2_peaks_all == c2_peaks[i])[0] + 1][0]
                next_segment = c2_aligned[best_end:next_peak]
                compensate_factor = len(next_segment) / (len(next_segment) + stretch)
                try:
                    next_segment_corrected = self.scale_chromatogram(next_segment, compensate_factor)
                except:
                    print('next_segment_corrected error')
                temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, next_segment_corrected, c2_aligned[next_peak:]])
            else:
                temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, c2_aligned[best_end:]])
            temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
            c2_peaks_all = temp_c2_peaks_1
            c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments, peak_ord=peak_ord)
            if not valid:
                raise ValueError(
                    "Please change parameters, some segments do not have peaks.")
            c2_peaks = np.concatenate([np.array((0,)), c2_peaks])

            if len(c2_peaks) < initial_npeaks and i >= len(c2_peaks) - 1:  # or len(c2_peaks) != nsegments:
                # continue
                break

            c2_aligned = temp_c2_aligned_1

            #  Avoid repeated peak locations or update those that moved to lower retention times
            # if len(lags_loc) > 0 and c2p + accum_lag <= lags_loc[-1]:
            if len(lags_loc) > 0 and any(c2p + stretch <= loc for loc in lags_loc):
                continue
                # lags_loc = lags_loc[:-1]
                # lags = lags[:-1]

            # if min_distance > 0.1:
            #     lags.append(0)
            #     lags_loc.append(c2p + accum_lag)
            #     continue
            lags.append(stretch)
            lags_loc.append(c2p + stretch)

        # Add one last point equal to the last one at 300000
        if len(lags) > 0:
            lags.append(lags[-1])
            lags_loc.append(30000)
        return np.array(lags_loc), np.array(lags)

    def lag_profile_from_peaks_general(
            self, c1, c2, proximity_threshold, nsegments, global_align=True, move_and_scale=True, scan_range=1,
            peak_ord=0, scale_dist=100, interval_after=500, min_avg_peak_dist=50):
        def calculate_average_peak_distance(signal1, signal2, prm=1E-6, distance_type='mean', num_largest_peaks=None, height_ratio=0.5):
            # Find peaks in both signals
            peaks1, properties1 = find_peaks(signal1, prominence=prm)
            peaks2, properties2 = find_peaks(signal2, prominence=prm)

            if len(peaks1) == 0 or len(peaks2) == 0:
                raise ValueError("No peaks found in one of the signals")

            # If num_largest_peaks is specified, select the indices of the largest peaks in signal1
            if num_largest_peaks is not None:
                prominences = properties1['prominences']
                largest_peak_indices = np.argsort(prominences)[-num_largest_peaks:]  # Indices of the largest peaks
                peaks1 = peaks1[largest_peak_indices]  # Filter peaks1 to include only the largest specified peaks

            # Match each peak in signal1 to the closest peak in signal2 with similar height
            distances = []
            for peak1 in peaks1:
                # Get the height of the current peak in signal1
                height1 = signal1[peak1]

                # Find the closest peak in signal2
                closest_peak2 = peaks2[np.argmin(np.abs(peaks2 - peak1))]
                height2 = signal2[closest_peak2]

                # Check if the height ratio is within the acceptable range
                if min(height1, height2) / max(height1, height2) >= height_ratio:
                    distance = abs(peak1 - closest_peak2)
                    distances.append(distance)

            if len(distances) < num_largest_peaks:
                return np.nan, distances, peaks1, peaks2

            # Calculate the average distance
            if distance_type == 'mean':
                average_distance = np.mean(distances) if distances else float('inf')
            elif distance_type == 'mean_square':
                average_distance = np.mean(np.square(distances)) if distances else float('inf')
            else:
                raise ValueError("Invalid distance_type. Choose 'mean' or 'mean_square'.")

            return average_distance, distances, peaks1, peaks2

        if global_align:
            # Global time shift
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(c1[:10000], 50),
                gaussian_filter(c2[:10000], 50),
                np.linspace(1., 1.1, 1)
            )
            c2_aligned = self.correct_segment(c2, best_scale, best_lag)
            accum_lag = best_lag
        else:
            c2_aligned = c2
            accum_lag = 0

        # c2_global_aligned = c2_aligned.copy()
        c1_peaks, _ = find_peaks(c1)
        c2_peaks_all, _ = find_peaks(c2_aligned)

        c2_peaks, valid = self.ensure_peaks_in_segments(c2_aligned, c2_peaks_all, num_segments=nsegments, peak_ord=peak_ord)
        if not valid:
            raise ValueError(
                "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks

        # Add zero to also scale the first part of the chromatogram
        c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
        c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
        initial_npeaks = min(len(c2_peaks), len(c1_peaks))

        lags = []
        lags_loc = []
        for i in range(1, initial_npeaks):
            if move_and_scale:
                c2p_prev = c2_peaks[i - 1]
            else:
                c2_peaks_all, _ = find_peaks(c2_aligned)
                c2p_prev = c2_peaks_all[np.where(c2_peaks_all == c2_peaks[i])[0] - 1][0]

            c2p = c2_peaks[i]
            # c2p_prev = max(0, c2_peaks[i] - scale_dist)
            # c2p = c2_peaks[i]
            c1p_prev = c2p_prev
            closest_index = np.argmin(np.abs(c1_peaks - c2_peaks[i]))
            start_index = max(0, closest_index - scan_range)
            end_index = min(len(c1_peaks), closest_index + scan_range + 1)
            min_distance = min_avg_peak_dist  # Minimum average peak distance for accepting the scaling
            start = min(c2p_prev, c2p)
            end = max(c2p_prev, c2p)
            # if end > 15000:
            #     print('here')
            for idx in range(start_index, end_index):
                c1p = c1_peaks[idx]
                # Scale the interval
                interval_c2 = c2p - c2p_prev
                interval_c1 = c1p - c1p_prev
                if ((c1p > len(c1) - interval_after
                        or c2p > len(c2_aligned) - interval_after)
                        or np.abs(c1p - c2p) > proximity_threshold
                        or interval_c2 <= 0
                        or interval_c1 <= 0
                        or (end - start) <= 0):
                    continue
                scale = interval_c1 / interval_c2
                if scale < 0:
                    print('Error in scale')
                c2_segment = c2_aligned[start:end]
                scaled_segment = self.scale_chromatogram(c2_segment, scale)

                if len(scaled_segment) == 0:
                    print('Error in scale len(scaled_segment)')
                    continue

                norm_c1_segment = utils.normalize_signal_standard(c1[c1p_prev:c1p + interval_after])
                norm_c2_segment = utils.normalize_signal_standard(np.concatenate([scaled_segment, c2_aligned[end:end + interval_after]]))
                try:
                    avg_peak_dist = calculate_average_peak_distance(
                        norm_c1_segment, norm_c2_segment, prm=1E-4, distance_type='mean', num_largest_peaks=5, height_ratio=0.1)[0]
                    # print(avg_peak_dist)
                except:
                    print('Error in avg_peak_dist')
                if avg_peak_dist < min_distance:
                    min_distance = avg_peak_dist
                    best_scaled_segment = scaled_segment
                    best_end = end
                    best_c1p = c1p

            try:
                best_scaled_segment
            except:
                best_end = max(c2p_prev, c2p)
                best_c1p = c1p
                best_scaled_segment = c2_aligned[start:end]

            # If the current peak in c2 is too far from the closest peak in c1, skip it
            if np.abs(best_c1p - c2p) > proximity_threshold or best_c1p <= c1p_prev:
                if i >= len(c2_peaks) - 1:
                    break
                else:
                    continue

            if move_and_scale:
                accum_lag += best_c1p - c2p
            else:
                stretch = best_c1p - c2p

            if move_and_scale:
                # Scaling between peaks transmitting changes to the rest of chromatogram
                temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, c2_aligned[best_end:]])
            else:
                #  Contract or stretch next between-peak section to compensate for previous movement
                next_peak = c2_peaks_all[np.where(c2_peaks_all == c2_peaks[i])[0] + 1][0]
                # if i + 1 < len(c2_peaks) and len(c2_aligned[best_end:c2_peaks[i + 1]]) > np.abs(stretch):
                if i + 1 < len(c2_peaks) and len(c2_aligned[best_end:next_peak]) > np.abs(stretch):
                    next_segment = c2_aligned[best_end:next_peak]
                    compensate_factor = len(next_segment) / (len(next_segment) + stretch)
                    try:
                        next_segment_corrected = self.scale_chromatogram(next_segment, compensate_factor)
                    except:
                        print('next_segment_corrected error')
                    temp_c2_aligned_1 = np.concatenate(
                        [c2_aligned[:start], best_scaled_segment, next_segment_corrected, c2_aligned[next_peak:]]
                    )
                else:
                    temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, c2_aligned[best_end:]])
            temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
            c2_peaks_all = temp_c2_peaks_1
            c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments, peak_ord=peak_ord)
            if not valid:
                raise ValueError(
                    "Please change parameters, some segments do not have peaks.")
            c2_peaks = np.concatenate([np.array((0,)), c2_peaks])

            if len(c2_peaks) < initial_npeaks and i >= len(c2_peaks) - 1:  # or len(c2_peaks) != nsegments:
                # continue
                break

            c2_aligned = temp_c2_aligned_1

            #  Avoid repeated peak locations or update those that moved to lower retention times
            # if len(lags_loc) > 0 and c2p + accum_lag <= lags_loc[-1]:
            if move_and_scale:
                if len(lags_loc) > 0 and any(c2p + accum_lag <= loc for loc in lags_loc):
                    continue
            else:
                if len(lags_loc) > 0 and any(c2p + stretch <= loc for loc in lags_loc):
                    continue

            if move_and_scale:
                lags.append(accum_lag)
                lags_loc.append(c2p + accum_lag)
            else:
                lags.append(stretch)
                lags_loc.append(c2p + stretch)

        # Add one last point equal to the last one at 300000
        if len(lags) > 0:
            lags.append(lags[-1])
            lags_loc.append(30000)
        return np.array(lags_loc), np.array(lags)



    # def lag_profile_from_peaks2(self, c1, c2, proximity_threshold, nsegments, global_align=True, scan_range=0):
    #
    #     if global_align:
    #         # Global time shift
    #         best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
    #             gaussian_filter(c1[:10000], 50),
    #             gaussian_filter(c2[:10000], 50),
    #             np.linspace(1., 1.1, 1)
    #         )
    #         c2_aligned = gaussian_filter(self.correct_segment(c2, best_scale, best_lag), 5)
    #         accum_lag = best_lag
    #     else:
    #         c2_aligned = c2
    #         accum_lag = 0
    #
    #     c1 = min_max_normalize(c1)
    #     c2_aligned = min_max_normalize(c2_aligned)
    #     # c1, _ = normalize_data(c1.reshape(-1, 1), scaler='standard')
    #     # c1 = c1.flatten()
    #     # c2_aligned, _ = normalize_data(c2_aligned.reshape(-1, 1), scaler='standard')
    #     # c2_aligned = c2_aligned.flatten()
    #
    #     c1_peaks, properties1 = find_peaks(c1, prominence=0)
    #     #  remove peak for standard in reference
    #     index_to_remove = np.where(c1_peaks == 8918)[0]
    #     c1_peaks = np.delete(c1_peaks, index_to_remove)
    #     c2_peaks, properties2 = find_peaks(c2_aligned, prominence=0)
    #     # prominences1 = properties1['prominences']
    #     # prominences2 = properties2['prominences']
    #     c1_peaks, valid = self.ensure_peaks_in_segments(c1, c1_peaks, num_segments=nsegments)
    #     c2_peaks, valid = self.ensure_peaks_in_segments(c2_aligned, c2_peaks, num_segments=nsegments)
    #
    #     if not valid:
    #         raise ValueError(
    #             "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks
    #
    #     # Add zero to also scale the first part of the chromatogram
    #     c1_peaks = np.concatenate([np.array((0,)), c1_peaks])
    #     c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
    #     initial_npeaks = min(len(c2_peaks), len(c1_peaks))
    #
    #     def scale_peak_interval(c2_aligned, c1p_prev, c1p, c2p_prev, c2p, start, end):
    #         interval_c2 = c2p - c2p_prev
    #         interval_c1 = c1p - c1p_prev
    #         if interval_c2 == 0:
    #             return False
    #         scale = interval_c1 / interval_c2
    #         c2_segment = c2_aligned[start:end]
    #         return self.scale_chromatogram(c2_segment, scale)
    #
    #     lags = []
    #     lags_loc = []
    #     for i in range(1, initial_npeaks):
    #         c2p_prev = c2_peaks[i - 1]
    #         c2p = c2_peaks[i]
    #         c1p_prev = c2p_prev
    #         c1p = c1_peaks[np.argmin(np.abs(c1_peaks - c2p))]
    #         closest_index = np.argmin(np.abs(c1_peaks - c2_peaks[i]))
    #         start_index = max(0, closest_index - scan_range)
    #         end_index = min(len(c1_peaks), closest_index + scan_range + 1)
    #         best_peak = c1_peaks[closest_index]
    #         min_distance = 1E6
    #         start = min(c2p_prev, c2p)
    #         end = max(c2p_prev, c2p)
    #         # for idx in range(start_index, end_index):
    #         #     _c1p = c1_peaks[idx]
    #         #
    #         #     # If the current peak in c2 is too far from the closest peak in c1, skip it
    #         #     if np.abs(_c1p - c2p) > proximity_threshold or _c1p <= c1p_prev:
    #         #         if i >= len(c2_peaks) - 1:
    #         #             break
    #         #         else:
    #         #             continue
    #         #     c1p = _c1p
    #         #     end = max(c2p_prev, c2p)
    #         #     scaled_segment = scale_peak_interval(c2_aligned, c1p_prev, c1p, c2p_prev, c2p, start, end)
    #         #     if isinstance(scaled_segment, bool):
    #         #         continue
    #         #     cut = min(len(c1[c1p_prev:c1p]), len(scaled_segment))
    #         #     distance = np.mean((min_max_normalize(c1[c1p_prev:c1p_prev + cut]) - min_max_normalize(scaled_segment[:cut])) ** 2)
    #         #     if distance < min_distance:
    #         #         min_distance = distance
    #         #         best_peak = c1_peaks[idx]
    #         #         best_scaled_segment = scaled_segment
    #         #         best_end = end
    #         # try:
    #         #     best_scaled_segment
    #         # except:
    #         #     continue
    #
    #         # best_scaled_segment = scale_peak_interval(c2_aligned, c1p_prev, c1_peaks[closest_index], c2p_prev, c2p, min(c2p_prev, c2p), max(c2p_prev, c2p))
    #         temp_c2_aligned_1 = np.concatenate([c2_aligned[:start], best_scaled_segment, c2_aligned[end:]])
    #
    #
    #             # # Scale the interval
    #             # interval_c2 = c2p - c2p_prev
    #             # interval_c1 = c1p - c1p_prev
    #             # if interval_c2 == 0:
    #             #     continue
    #             # scale = interval_c1 / interval_c2
    #             # start = min(c2p_prev, c2p)
    #             # end = max(c2p_prev, c2p)
    #             # c2_segment = c2_aligned[start:end]
    #             # scaled_segment = self.scale_chromatogram(c2_segment, scale)
    #
    #             # corr1 = correlate(min_max_normalize(c1[c1p_prev:c1p]), min_max_normalize(c2_segment))
    #             # corr2 = correlate(min_max_normalize(c1[c1p_prev:c1p]), min_max_normalize(scaled_segment))
    #             # if max(corr2) < max(corr1) + max(corr1) * 0.00:
    #             #     scaled_segment = c2_segment
    #             #     accum_lag -= c1p - c2p
    #
    #             # best_lag = np.argmax(corr) - len(c2_segment) + 1
    #
    #             # cut = min(len(temp_c2_aligned_1), len(c1))
    #             # current_mse = np.mean((np.concatenate([c2_aligned[:start], scaled_segment]) - c1[:(start + len(scaled_segment))]) ** 2)
    #             # print(current_mse)
    #             # if current_mse > max_mse:
    #             #     continue
    #
    #             # tol = 10
    #             # if np.abs(c1p - c2p) >= np.abs(best_lag) + tol:
    #             #     if len(lags_loc) > 0:
    #             #         lags.append(lags[-1])
    #             #         lags_loc.append(lags_loc[-1])
    #             #         continue
    #
    #
    #         temp_c2_peaks_1, _ = find_peaks(temp_c2_aligned_1)
    #         c2_peaks, valid = self.ensure_peaks_in_segments(temp_c2_aligned_1, temp_c2_peaks_1, num_segments=nsegments)
    #         if not valid:
    #             raise ValueError(
    #                 "Please change parameters, some segments do not have peaks.")
    #         c2_peaks = np.concatenate([np.array((0,)), c2_peaks])
    #
    #         if len(c2_peaks) < initial_npeaks and i >= len(c2_peaks) - 1:  # or len(c2_peaks) != nsegments:
    #             # continue
    #             break
    #
    #         c2_aligned = temp_c2_aligned_1
    #         accum_lag += c1p - c2p
    #
    #         #  Avoid repeated peak locations or update those that moved to less retention times
    #         # if len(lags_loc) > 0 and c2p + accum_lag <= lags_loc[-1]:
    #         if len(lags_loc) > 0 and any(c2p + accum_lag <= loc for loc in lags_loc):
    #             continue
    #             lags_loc = lags_loc[:-1]
    #             lags = lags[:-1]
    #
    #         lags.append(accum_lag)
    #         lags_loc.append(c2p + accum_lag)
    #
    #     # Add one last point equal to the last one at 300000
    #     if len(lags) > 0:
    #         lags.insert(0, lags[0])
    #         lags_loc.insert(0, 0)
    #
    #     # lags.append(lags[-1])
    #     # lags_loc.append(30000)
    #
    #     return np.array(lags_loc), np.array(lags)


    def scale_between_peaks_dtw(self, c1, c2, proximity_threshold, peak_prominence, nsegments):
        # Find peaks
        c2_peaks, _ = find_peaks(c2)

        # choose 2 peaks from the first section
        c2_peaks_first_segment, valid = self.ensure_peaks_in_segments(c2, c2_peaks, num_segments=2*nsegments, last_segment=1)
        c2_peaks, valid = self.ensure_peaks_in_segments(c2, c2_peaks, num_segments=nsegments)
        if not valid:
            raise ValueError(
                "Please change parameters, some segments in c2_peaks do not have peaks (scale_between_peaks).")  # Stop if any segment lacks peaks

        #  Remove single peak from first section and add two peaks from first section split in two
        c2_peaks = np.concatenate((c2_peaks_first_segment, c2_peaks[1:]))

        dtw_path = self.get_dtw_path(c1, c2)
        corresponding_c1_peaks = self.find_corresponding_indices(c2_peaks, dtw_path)

        # c2_aligned = c2.copy()
        npeaks = len(c2_peaks)
        temp_c2_aligned = np.array([])

        # Add zero to also scale the first part of the chromatogram
        corresponding_c1_peaks = np.concatenate([np.array((0,)), corresponding_c1_peaks])
        c2_peaks = np.concatenate([np.array((0,)), c2_peaks])

        prev_length_change = 0
        for i in range(1, npeaks):
            c2p_prev = c2_peaks[i - 1]
            c2p = c2_peaks[i]
            c1p_prev = c2p_prev
            c1p = corresponding_c1_peaks[i]

            # Scale the interval
            interval_c2 = c2p - c2p_prev
            interval_c1 = c1p - c1p_prev


            if interval_c2 == 0:
                continue

            #  Scale factor also taking into account the contraction/extension of previous section
            scale0 = interval_c1 / interval_c2
            scale = interval_c1 / (interval_c2 + prev_length_change)

            start = min(c2p_prev, c2p)
            end = max(c2p_prev, c2p)
            c2_segment = c2[start:end]

            # If the current peak in c2 is too far from the closest peak in c1, skip it
            if np.abs(c1p - c2p) > proximity_threshold or c1p <= c1p_prev:
                temp_c2_aligned = np.concatenate([temp_c2_aligned, c2_segment])
                continue

            scaled_segment = self.scale_chromatogram(c2_segment, scale)
            temp_c2_aligned = np.concatenate([temp_c2_aligned, scaled_segment])
            prev_length_change = int(len(c2_segment) * scale0) - len(c2_segment)

        c2_aligned = np.concatenate([temp_c2_aligned, c2[c2_peaks[-1]:]])

        return c2_aligned


    def ensure_peaks_in_segments(self, c2, c2_peaks, num_segments=10, last_segment=None, peak_ord=0):
        segment_length = len(c2) // num_segments
        new_c2_peaks = []

        for i in range(num_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < num_segments - 1 else len(c2)
            segment_peaks = [peak for peak in c2_peaks if start <= peak < end]

            if not segment_peaks:
                continue
                # print(f"Segment {i + 1} of {num_segments} has no peaks. Please change parameters.")
                # return c2, False  # Return original c2 if any segment lacks peaks

            sorted_peaks = sorted(segment_peaks, key=lambda x: c2[x], reverse=True)
            try:
                highest_peak = sorted_peaks[peak_ord]
            except:
                continue
            # highest_peak = segment_peaks[np.argmax(c2[segment_peaks])]
            new_c2_peaks.append(highest_peak)

            if i == last_segment:
                break

        return np.array(new_c2_peaks), True


    def remove_peak(self, signal, peak_idx, window_size=5):
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


    def shift_chromatogram(self, chrom, lag, left_value=0, right_value=0):
        shifted_chrom = np.roll(chrom, lag)
        if lag > 0:
            shifted_chrom[:lag] = left_value  # Set the elements shifted in from the right
        elif lag < 0:
            shifted_chrom[-lag:] = right_value  # Set the elements shifted in from the left
        return shifted_chrom

    def find_optimal_offset(self, chrom1, chrom2):
        cross_corr = correlate(chrom1, chrom2, mode='full')
        lag = np.argmax(cross_corr) - len(chrom2) + 1
        return lag

    def optimize_shift(self, reference_signal, signal):
        """
        Optimize the shift parameters to align `signal` with `reference_signal`.

        Parameters:
        - reference_signal: array-like, the reference signal to align to
        - signal: array-like, the signal to be shifted

        Returns:
        - shifted_signal_opt: array-like, the optimally shifted signal
        - params_opt: list, the optimized parameters [a, b, c]
        """

        # Generate the time axis
        t = np.arange(len(reference_signal))

        # Objective function to minimize
        def objective(params, signal, reference_signal, t, loss='l2'):
            a, b, c = params
            shifted_t = t + a + b * t + c * t ** 2
            # Ensure the shifted_t is within the range of t
            shifted_t = np.clip(shifted_t, 0, len(t) - 1)
            shifted_signal = np.interp(shifted_t, t, signal)
            if loss == 'l1':
                return np.sum(np.abs(reference_signal - shifted_signal))
            elif loss == 'l2':
                return np.sum((shifted_signal - reference_signal) ** 2)
            elif loss == 'corr':
                cross_corr = correlate(reference_signal, shifted_signal, mode='valid')
                return -np.max(cross_corr)

        # Initial guess for the parameters
        initial_guess = [0, 0, 0]

        # Perform the optimization
        result = minimize(objective, initial_guess, args=(signal, reference_signal, t))

        # Extract the optimized parameters
        a_opt, b_opt, c_opt = result.x
        print(f"Optimized parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}")

        # Shift the signal using the optimized parameters
        shifted_t_opt = t + a_opt + b_opt * t + c_opt * t ** 2
        shifted_t_opt = np.clip(shifted_t_opt, 0, len(t) - 1)
        shifted_signal_opt = np.interp(shifted_t_opt, t, signal, left=signal[0], right=signal[-1])

        return shifted_signal_opt, [a_opt, b_opt, c_opt]

    def adjust_chromatogram(self, algo=None, initial_lag=300):
        """
        Adjusts the chromatogram by scaling and lagging the segments to match a reference chromatogram.

        Parameters
        ----------
        algo : int, optional
            The algorithm to be used for local synchronization. Default is 0.
            Options:
            - 0: Global shift and scaling (other algorithms are applied to the output of this one)
            - 1: Scale and shift local segments.
            - 2: Align and scale signals between the highest peaks.
            - 3: Model shift as a function of time using a quadratic function.
            - 4: Model shift as a function of time using a spline.
        initial_lag : int, optional
            The initial lag value to be used in synchronization. Default is 300.

        Returns
        -------
        numpy.ndarray
            The adjusted chromatogram after applying the specified synchronization algorithm.
        """
        if algo in [3, 4, 5]:
            corrected_c2 = self.c2
        else:
            # Global scale and time shift
            best_scale, best_lag, _ = self.find_best_scale_and_lag_corr(
                gaussian_filter(self.c1[:10000], 50),
                gaussian_filter(self.c2[:10000], 50),
                np.linspace(0.98, 1.02, 100)
            )
            print(best_scale, best_lag)
            corrected_c2 = self.correct_segment(self.c2, best_scale, best_lag)

        if algo == 0:
            return np.array(corrected_c2)

        # Scale and shift local segments
        elif algo == 1:
            for i in [10]:
                corrected_c2 = self.scale_and_shift_in_sections(corrected_c2, nsections=i, lag=initial_lag)

        # Align and scale signals between the highest peaks
        elif algo == 2:
            for seg in [10]:
                corrected_c2 = self.scale_between_peaks(
                    self.c1, corrected_c2,
                    proximity_threshold=self.max_sep_threshold,
                    peak_prominence=self.peak_prominence,
                    nsegments=seg,
                )

        # # Model shift as a function of t using quadratic function
        # elif algo == 3:
        #     min_len = min(len(self.c1), len(corrected_c2))
        #     ref = min_max_normalize(self.c1, 0, 1)[:min_len]
        #     chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
        #     corrected_c2, params_opt = self.optimize_shift(ref, chrom)
        #     t = np.arange(len(ref))
        #     self.plot_signal_and_fit(t, ref, chrom, corrected_c2, fit_type='quadratic', fit_params=params_opt, data_points=self.lag_res)

        # Model shift as a function of t using a quadratic function
        elif algo == 3:
            min_len = min(len(self.c1), len(corrected_c2))
            ref = min_max_normalize(self.c1, 0, 1)[:min_len]
            chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
            t = np.arange(len(ref))

            # Use np.polyfit to get the initial guess for quadratic coefficients [a, b, c] based on lag_res
            polyfit_coeffs = np.polyfit(self.lag_res[0], self.lag_res[1], 2)
            initial_guess = [polyfit_coeffs[0], polyfit_coeffs[1], polyfit_coeffs[2]]

            # Define the objective function for quadratic fitting
            def objective_function_quadratic(params, ref, chrom, t, loss='corr'):
                a, b, c = params
                quadratic_shift = a * t ** 2 + b * t + c
                shifted_signal = np.interp(t - quadratic_shift, t, chrom)
                if loss == 'l1':
                    return np.sum(np.abs(ref - shifted_signal))
                elif loss == 'l2':
                    return np.sum((shifted_signal - ref) ** 2)
                elif loss == 'corr':
                    cross_corr = correlate(ref, shifted_signal, mode='valid')
                    return -np.max(cross_corr)
                elif loss == 'mse':
                    return np.mean((shifted_signal - ref) ** 2)

            # Minimize the objective function to find the best-fitting quadratic parameters
            result = minimize(objective_function_quadratic, initial_guess, args=(ref, chrom, t, 'corr'))
            # result = minimize(
            #     objective_function_quadratic, initial_guess, args=(gaussian_filter(ref, sigma=5), gaussian_filter(chrom, sigma=5), t))

            # Extract the optimized parameters
            a_opt, b_opt, c_opt = result.x
            print(f"Optimized quadratic parameters: a = {a_opt}, b = {b_opt}, c = {c_opt}")

            # # or use the initial fit
            # a_opt, b_opt, c_opt = polyfit_coeffs[0], polyfit_coeffs[1], polyfit_coeffs[2]

            # Apply the optimized quadratic shift
            quadratic_shift = a_opt * t ** 2 + b_opt * t + c_opt
            corrected_c2 = np.interp(t - quadratic_shift, t, chrom, left=chrom[0], right=chrom[-1])


            # Plot the signals and the fitted quadratic curve
            # self.plot_signal_and_fit(t, ref, chrom, corrected_c2, fit_type='quadratic', fit_params=[a_opt, b_opt, c_opt], data_points=self.lag_res)

        # Model shift with custom function fit with shift datapoints
        elif algo == 4:
            from scipy.interpolate import UnivariateSpline
            # Function to apply the shift to the signal using a spline
            def apply_shift_spline(c2, t, spline):
                # Apply the shift to the time axis
                shifted_t = t - spline(t)
                # Interpolate the c2 signal at the shifted time points
                interpolator = interp1d(t, c2, fill_value="extrapolate", bounds_error=False)
                return interpolator(shifted_t)

            # Objective function to minimize (MSE in this case)
            def objective_function_spline(params, c1, c2, t, spline, loss='l2'):
                try:
                    spline.set_smoothing_factor(params[0])  # Update smoothing factor
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
            switch = True

            def correct_with_spline(corrected_c2, s, k, normalize=True, plot=False):
                min_len = min(len(self.c1), len(corrected_c2))
                # ref, _ = normalize_data(self.c1.reshape(-1, 1), scaler='minmax')[:min_len]
                # chrom, _ =  normalize_data(corrected_c2.reshape(-1, 1), scaler='minmax')[:min_len]
                # ref = ref.flatten()
                # chrom = chrom.flatten()
                ref = min_max_normalize(self.c1, 0, 1)[:min_len]
                if normalize:
                    chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
                else:
                    chrom = corrected_c2[:min_len]

                # Create the spline for initial guess
                try:
                    spline = UnivariateSpline(self.lag_res[0], self.lag_res[1], s=s, k=k)
                except:
                    return chrom
                # spline = UnivariateSpline(self.lag_res[0], gaussian_filter(self.lag_res[1], 3), s=50, k=1)
                t = np.arange(len(ref))

                # Optimize the smoothing factor of the spline
                initial_guess = [s]  # Initial guess for the smoothing factor
                bounds = [(0, None)]  # Ensure smoothing factor is non-negative
                result = minimize(objective_function_spline, initial_guess, args=(ref, chrom, t, spline, 'mse'),
                                  bounds=bounds)

                # Get the optimized smoothing factor
                optimized_smoothing_factor = result.x[0]
                spline.set_smoothing_factor(optimized_smoothing_factor)

                # Apply the optimized shift
                corrected_c2 = apply_shift_spline(chrom, t, spline)

                if plot:
                    self.plot_signal_and_fit(
                        t, ref, chrom, corrected_c2, fit_type='spline', fit_params=spline, data_points=self.lag_res
                    )

                return corrected_c2


            c1 = gaussian_filter(self.c1, 10)
            corrected_c2 = gaussian_filter(corrected_c2, 10)
            for prox in [50]:
                for seg in [10]:
                    for ord in [0]:
                        self.lag_res = self.lag_profile_from_peaks_general(
                            c1, corrected_c2, proximity_threshold=prox, nsegments=seg, global_align=True, scan_range=2,
                            peak_ord=ord, interval_after=2500, min_avg_peak_dist=40, move_and_scale=True
                            )
                        corrected_c2 = correct_with_spline(corrected_c2, 5, 1, normalize=True, plot=False)

            # for ord in [0, 1, 2, 3, 4]:
            #     self.lag_res = self.lag_profile_from_peaks_general(
            #         c1, corrected_c2, proximity_threshold=200, nsegments=30, global_align=False, scan_range=3,
            #         peak_ord=ord, interval_after=3000, min_avg_peak_dist=30, move_and_scale=False,
            #         )
            #     print(self.lag_res[1])
            #     corrected_c2 = correct_with_spline(corrected_c2, 0, 1, normalize=False, plot=False)

            # for prox in [40]:
            #     self.lag_res = self.lag_profile_from_peaks(
            #             self.c1, corrected_c2, proximity_threshold=prox,  nsegments=50, global_align=True
            #         )
            #     corrected_c2 = correct_with_spline(corrected_c2, 50, 1, normalize=True, plot=False)

            # # c1 = self.c1.copy()
            # # c1 = gaussian_filter(c1, 10)
            # # corrected_c2 = gaussian_filter(corrected_c2, 10)
            # plot = False
            # cnt = 0
            # # for ord in [0, 0, 0, 1, 1, 1, 1, 1]:  # [0, 0, 1, 1, 2, 2, 2]:  # [0, 0, 0, 1, 1 ]
            # for ord in [0, 0, 0, 1, 1, 1, 1, 1]:
            # # for seg in [10]:
            # #     if cnt == 0:
            # #         glb_alig = True
            #
            #     self.lag_res = self.lag_profile_from_peaks4(
            #         c1, corrected_c2, proximity_threshold=200, nsegments=10, global_align=False, scan_range=3,
            #         peak_ord=ord, scale_dist=250, interval_after=3000, min_avg_peak_dist=10
            #     )
            #     glb_alig = False
            #     print(self.lag_res[1])
            #     if cnt == len([0, 1, 2, 3, 4, 5]) - 1:
            #         plot = False
            #
            #     corrected_c2 = correct_with_spline(corrected_c2, 20, 1, normalize=False, plot=plot)
            #     cnt += 1

            # TODO: select peaks not only on closest distance to reference but based on some similarity metric

            # self.lag_res = utils.calculate_lag_profile(
            #     self.c1, corrected_c2, 2000, lag_range=50, hop=100, sigma=20, distance_metric='mse',
            #     init_min_dist=1E6
            # )
            # corrected_c2 = correct_with_spline(corrected_c2)

            switch = False
            # print(self.lag_res[1])


            # min_len = min(len(self.c1), len(corrected_c2))
            # # ref, _ = normalize_data(self.c1.reshape(-1, 1), scaler='minmax')[:min_len]
            # # chrom, _ =  normalize_data(corrected_c2.reshape(-1, 1), scaler='minmax')[:min_len]
            # # ref = ref.flatten()
            # # chrom = chrom.flatten()
            # ref = min_max_normalize(self.c1, 0, 1)[:min_len]
            # chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
            #
            # # Create the spline for initial guess
            # spline = UnivariateSpline(self.lag_res[0], self.lag_res[1], s=20, k=1)
            # # spline = UnivariateSpline(self.lag_res[0], gaussian_filter(self.lag_res[1], 3), s=50, k=1)
            # t = np.arange(len(ref))
            #
            # # Optimize the smoothing factor of the spline
            # initial_guess = [20]  # Initial guess for the smoothing factor
            # bounds = [(0, None)]  # Ensure smoothing factor is non-negative
            # result = minimize(objective_function_spline, initial_guess, args=(ref, chrom, t, spline, 'mse'),
            #                   bounds=bounds)
            #
            # # Get the optimized smoothing factor
            # optimized_smoothing_factor = result.x[0]
            # spline.set_smoothing_factor(optimized_smoothing_factor)
            #
            # # Apply the optimized shift
            # corrected_c2 = apply_shift_spline(chrom, t, spline)

            # self.plot_signal_and_fit(t,  ref, chrom, corrected_c2, fit_type='spline', fit_params=spline, data_points=self.lag_res)

            # min_len = min(len(self.c1), len(corrected_c2))
            # # ref, _ = normalize_data(self.c1.reshape(-1, 1), scaler='minmax')[:min_len]
            # # chrom, _ =  normalize_data(corrected_c2.reshape(-1, 1), scaler='minmax')[:min_len]
            # # ref = ref.flatten()
            # # chrom = chrom.flatten()
            # ref = min_max_normalize(self.c1, 0, 1)[:min_len]
            # chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
            #
            # # Create the spline for initial guess
            # spline = UnivariateSpline(self.lag_res[0], self.lag_res[1], s=0, k=3)
            # # spline = UnivariateSpline(self.lag_res[0], gaussian_filter(self.lag_res[1], 1), s=0, k=3)
            # t = np.arange(len(ref))
            #
            # # Optimize the smoothing factor of the spline
            # initial_guess = [0]  # Initial guess for the smoothing factor
            # bounds = [(0, None)]  # Ensure smoothing factor is non-negative
            # result = minimize(objective_function_spline, initial_guess, args=(ref, chrom, t, spline, 'mse'), bounds=bounds)
            #
            # # Get the optimized smoothing factor
            # optimized_smoothing_factor = result.x[0]
            # spline.set_smoothing_factor(optimized_smoothing_factor)
            #
            # # Apply the optimized shift
            # corrected_c2 = apply_shift_spline(chrom, t, spline)

            # self.plot_signal_and_fit(t,  ref, chrom, corrected_c2, fit_type='spline', fit_params=spline, data_points=self.lag_res)

        # Model shift as a function of t using a linear function (new algo 5)
        elif algo == 5:
            min_len = min(len(self.c1), len(corrected_c2))
            ref = min_max_normalize(self.c1, 0, 1)[:min_len]
            chrom = min_max_normalize(corrected_c2, 0, 1)[:min_len]
            t = np.arange(len(ref))

            # Use np.polyfit to get the initial guess for linear coefficients [a, b] based on lag_res
            polyfit_coeffs = np.polyfit(self.lag_res[0], self.lag_res[1], 1)
            initial_guess = [polyfit_coeffs[0], polyfit_coeffs[1]]

            # Define the objective function for linear fit
            def objective_function_linear(params, signal, reference_signal, t):
                a, b = params
                linear_shift = a * t + b
                shifted_signal = np.interp(t - linear_shift, t, signal)
                return np.sum((reference_signal - shifted_signal) ** 2)

            # Optimize the linear coefficients
            result = minimize(objective_function_linear, initial_guess, args=(chrom, ref, t))
            a_opt, b_opt = result.x

            # Apply the optimized linear shift
            linear_shift = a_opt * t + b_opt
            corrected_c2 = np.interp(t - linear_shift, t, chrom, left=chrom[0], right=chrom[-1])

            # Plot the result
            # self.plot_signal_and_fit(t, ref, chrom, corrected_c2, fit_type='linear', fit_params=[a_opt, b_opt],
            #                          data_points=self.lag_res)

        else:
            raise ValueError("Invalid value for 'algo'. Must be one of None, 1, 2, 3.")

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

    def get_dtw_path(self, c1, c2):
        """
        Computes path between two chromatogram signals using Dynamic Time Warping (DTW).

        Parameters:
        c1 (np.array): First chromatogram signal.
        c2 (np.array): Second chromatogram signal.

        Returns:
        np.array: The path of c2 to c1.
        """
        # Compute the DTW alignment
        distance, path = fastdtw(c1, c2, dist=2)

        return path

    def find_corresponding_indices(self, c2_indices, dtw_path):
        """
        Finds the corresponding indices in c1 for a given list of indices in c2 based on the DTW path.

        Parameters:
        c2_indices (list): List of indices in c2.
        dtw_path (list): DTW path.

        Returns:
        list: Corresponding indices in c1 for the given indices in c2.
        """
        corresponding_indices = []

        for c2_idx in c2_indices:
            # Find the index in the DTW path that corresponds to the given c2_idx
            corresponding_c1_idx = [idx_c1 for (idx_c1, idx_c2) in dtw_path if idx_c2 == c2_idx]

            if corresponding_c1_idx:
                corresponding_indices.append(corresponding_c1_idx[0])
            else:
                corresponding_indices.append(None)  # If there's no corresponding index, append None

        return corresponding_indices

    def plot_chromatograms(self, corrected_c2):
        plt.figure(figsize=(16, 4))
        plt.plot(self.c1, label='Chromatogram 1')
        # plt.plot(self.c2, label='Chromatogram 2 (Original)', linestyle='--')
        plt.plot(corrected_c2, label='Chromatogram 2 (Corrected)')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.title('Chromatogram Adjustment')
        plt.legend()
        plt.grid(True)
        plt.show()




