import numpy as np
import pandas as pd
import os
import re
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

class WineAnalysis:
    def __init__(self, file_path=None, data_dict=None, normalize=True):
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
            self.data = StandardScaler().fit_transform(pd.DataFrame(data_dict).T)
            # self.data = np.array(pd.DataFrame(data_dict).T)
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

    def run_umap(self, n_neighbors=60, random_state=16, best_score=None, plot=None):
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
        title = f'UMAP on {self.chem_name}; {len(self.data)} wines\nSilhouette score: {best_score} '
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

    def merge_chromatograms(self, chrom1, chrom2):
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
            merged_chromatograms[key] = self.normalize_chromatogram(value)

        for key, value in chrom2.items():
            if key in merged_chromatograms:
                key = f"{key}b"
            merged_chromatograms[key] = self.normalize_chromatogram(value)

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

    def tsne_analysis(self, data_dict, vintage, chem_name):
        analysis = WineAnalysis(data_dict=data_dict, normalize=False)
        cls = Classifier(analysis.data, analysis.labels)
        perplexity, random_state, best_score = run_tsne_and_evaluate(
            analysis,
            cls._process_labels(vintage),
            chem_name,
            perplexities=range(40, 100, 10),
            random_states=range(0, 96, 16)
        )
        analysis.run_tsne(perplexity=perplexity, random_state=random_state, best_score=best_score, plot=True)
        # analysis.run_umap(n_neighbors=10, random_state=10, best_score=10)
#
    def umap_analysis(self, data_dict, vintage, chem_name):
        analysis = WineAnalysis(data_dict=data_dict, normalize=False)
        cls = Classifier(analysis.data, analysis.labels)
        n_neighbors, random_state, best_score = run_umap_and_evaluate(
            analysis,
            cls._process_labels(vintage),
            chem_name,
            neigh_range=range(40, 100, 10),
            random_states=range(0, 96, 16)
        )
        analysis.run_umap(n_neighbors=n_neighbors, random_state=random_state, best_score=best_score, plot=True)
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


    def find_best_scale_and_lag(self, chrom1, chrom2, scales, max_lag):
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
            lags = np.arange(-len(chrom1) + 1, len(scaled_chrom2))

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

    def sync_individual_chromatograms(self, reference_chrom, chromatograms, ref_peak_pos=None):
        scales = np.linspace(0.85, 1.15, 500)  # Example scales to try
        max_lag = 500
        synced_chromatograms = {}
        for key in chromatograms.keys():
            # if key == 'B1990':
                print(key)
                chrom = chromatograms[key]

                if not ref_peak_pos:
                    # Example maximum lag
                    best_scale, best_lag, best_corr = self.find_best_scale_and_lag(
                        reference_chrom[:5000], chrom[:5000], np.array((1,)), 500
                    )
                    chrom_shifted = self.shift_chromatogram(chrom, best_lag)
                    chrom_sync = self.scale_chromatogram(chrom_shifted, 0.998)
                    optimized_chrom = chrom_sync
                else:
                    optimized_chrom = self.sync_chromatograms(reference_chrom, chrom, ref_peak_pos=ref_peak_pos)

                synced_chromatograms[key] = optimized_chrom
                # plt.figure(figsize=(24, 4))
                # plt.plot(reference_chrom)
                # plt.plot(optimized_chrom)
                # plt.show()
                # plt.close()

        return synced_chromatograms


    def resample_chromatograms(self, chrom1, chrom2, start=None):
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
            length = min(min(len(value) for value in chrom1.values()), min(len(value) for value in chrom2.values()))
            resampled_chrom1 = {key: self.resample_chromatogram(value[start:], length) for key, value in chrom1.items()}
            resampled_chrom2 = {key: self.resample_chromatogram(value[start:], length) for key, value in chrom2.items()}
        else:
            length = min(len(chrom1), len(chrom2))  # Use the minimum length if chrom1 and chrom2 are arrays
            resampled_chrom1 = self.resample_chromatogram(chrom1, length)
            resampled_chrom2 = self.resample_chromatogram(chrom2, length)

        return resampled_chrom1, resampled_chrom2

class SyncChromatograms:
    def __init__(self, c1, c2, n_segments, scales, min_peaks=5, max_iterations=100, threshold=0.5, max_sep_threshold=50):
        self.c1 = c1
        self.c2 = c2
        self.n_segments = n_segments
        self.scales = scales
        self.min_peaks = min_peaks
        self.max_iterations = max_iterations
        self.threshold = threshold
        self.max_sep_threshold = max_sep_threshold

    def scale_chromatogram(self, chrom, scale):
        x = np.arange(len(chrom))
        scaled_length = int(len(chrom) * scale)
        scaled_x = np.linspace(0, len(chrom) - 1, num=scaled_length)
        f = interp1d(x, chrom, bounds_error=False, fill_value="extrapolate")
        return f(scaled_x)

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

    def find_best_scale_and_lag(self, c1_segment, c2_segment, initial_lag=None):
        best_scale = None
        best_lag = None
        best_diff = np.inf

        peaks_c1 = self.find_largest_peaks(c1_segment, self.min_peaks)
        if len(peaks_c1) == 0:
            return best_scale, best_lag

        peaks_c1 = np.sort(peaks_c1)

        lag_range = int(0.2 * len(c1_segment))

        if initial_lag is not None:
            lag_start = -initial_lag
            lag_end = initial_lag
        else:
            lag_start = -lag_range
            lag_end = lag_range

        for lag in range(lag_start, lag_end + 1):
            for scale in self.scales:
                scaled_c2_segment = self.scale_chromatogram(c2_segment, scale)
                peaks_scaled_c2 = self.find_largest_peaks(scaled_c2_segment, self.min_peaks)
                if len(peaks_scaled_c2) == 0:
                    continue

                peaks_scaled_c2 = np.sort(peaks_scaled_c2)
                shifted_peaks_scaled_c2 = peaks_scaled_c2 + lag
                valid_indices = (shifted_peaks_scaled_c2 >= 0) & (shifted_peaks_scaled_c2 < len(c1_segment))
                shifted_peaks_scaled_c2 = shifted_peaks_scaled_c2[valid_indices]

                if len(shifted_peaks_scaled_c2) == 0:
                    continue

                distances = []
                for peak1, peak2 in zip(peaks_c1, shifted_peaks_scaled_c2):
                    distances.append(abs(peak1 - peak2))

                if len(distances) == 0:
                    continue

                diff = np.mean(distances)

                if diff < best_diff:
                    best_diff = diff
                    best_scale = scale
                    best_lag = lag

        return best_scale, best_lag

    def correct_segment(self, segment, scale, lag):
        if scale is None or lag is None:
            return segment

        scaled_segment = self.scale_chromatogram(segment, scale)
        if lag > 0:
            corrected_segment = np.roll(scaled_segment, lag)
            corrected_segment[:lag] = scaled_segment[0]
        else:
            corrected_segment = np.roll(scaled_segment, lag)
            corrected_segment[lag:] = scaled_segment[-1]
        return corrected_segment

    def align_maxima(self, c1_segment, c2_segment):
        threshold_c1 = min(c1_segment) + np.std(c1_segment) * self.threshold
        threshold_c2 = min(c2_segment) + np.std(c2_segment) * self.threshold
        peaks_c1, _ = find_peaks(c1_segment, height=threshold_c1, prominence=0.1)
        peaks_c2, _ = find_peaks(c2_segment, height=threshold_c2, prominence=0.1)

        # Return the original segment if no peaks are found in either c1 or c2
        if len(peaks_c1) == 0 or len(peaks_c2) == 0:
            return c2_segment, []

        # Sort peaks in c2 by their amplitude in descending order
        sorted_peaks_c2 = peaks_c2[np.argsort(c2_segment[peaks_c2])[::-1]]
        c2_segment_aligned = np.copy(c2_segment)
        aligned_maxima = []  # List to keep track of the aligned maxima

        # Align the highest peak in c2 with the closest peak in c1
        m1_c2 = sorted_peaks_c2[0]
        m1_c1 = peaks_c1[np.argmin(np.abs(peaks_c1 - m1_c2))]
        shift = m1_c1 - m1_c2
        c2_segment_aligned = self.shift_chromatogram(c2_segment_aligned, shift, c2_segment[0], c2_segment[-1])
        peaks_c2, _ = find_peaks(c2_segment_aligned, height=threshold_c2)
        sorted_peaks_c2 = peaks_c2[np.argsort(c2_segment_aligned[peaks_c2])[::-1]]
        aligned_maxima.append(sorted_peaks_c2[0])  # Track aligned maxima

        # Align subsequent peaks by padding or trimming
        initial_peaks = min(len(peaks_c1), len(sorted_peaks_c2))
        for i in range(1, initial_peaks):
            m2_c2 = sorted_peaks_c2[i]
            m1_c2 = aligned_maxima[np.argmin(np.abs(aligned_maxima - m2_c2))]  # closest previously aligned

            m1_c1 = peaks_c1[np.argmin(np.abs(peaks_c1 - m1_c2))]  # closest m1 in c1
            m2_c1 = peaks_c1[np.argmin(np.abs(peaks_c1 - m2_c2))]  # closest m1 in c1

            if m1_c1 == m2_c1 or np.abs(m2_c2 - m2_c1) > self.max_sep_threshold:# or m1_c2 == m2_c2:
                 # break
                continue

            aligned_maxima.append(m2_c1)  # track aligned maxima

            # Ensure m1_c1 and m2_c1 are in the correct order without altering their original values
            sorted_m1_c1, sorted_m2_c1 = sorted((m1_c1, m2_c1))
            sorted_m1_c2, sorted_m2_c2 = sorted((m1_c2, m2_c2))

            # Find the minimum value in the c2 segment between the peaks
            interval_c2_segment = c2_segment_aligned[sorted_m1_c2:sorted_m2_c2]
            min_pos_c2 = np.argmin(interval_c2_segment) + sorted_m1_c2
            min_value_c2 = np.min(interval_c2_segment)

            # Calculate the intervals between peaks in c1 and c2
            interval_c1 = sorted_m2_c1 - sorted_m1_c1
            interval_c2 = sorted_m2_c2 - sorted_m1_c2

            # Pad or trim c2 to match the interval in c1
            if interval_c1 > interval_c2:
                # Pad c2 with the minimum value found between the peaks
                pad_amount = interval_c1 - interval_c2
                m2_c2_idx = int(np.where(peaks_c2 == m2_c2)[0][0])
                peak_on_left = peaks_c2[m2_c2_idx - 1]
                min_pos_peak_left = np.argmin(c2_segment_aligned[peak_on_left:peaks_c2[m2_c2_idx]]) + peak_on_left
                c2_segment_aligned = np.concatenate([
                    c2_segment_aligned[:min_pos_peak_left - pad_amount],
                    c2_segment_aligned[min_pos_peak_left:min_pos_c2],
                    np.full(pad_amount, min_value_c2),
                    c2_segment_aligned[min_pos_c2:]
                ])
                #     c2_segment_aligned[:min_pos_c2],
                #     np.full(pad_amount, min_value_c2),
                #     c2_segment_aligned[min_pos_c2:]
                # ])
                # c2_segment_aligned = c2_segment_aligned[pad_amount:]  # remove from the beginning to align
            elif interval_c2 > interval_c1:
                # Trim c2 to match the interval in c1 by removing data around the minimum value
                trim_amount = interval_c2 - interval_c1
                left_trim = trim_amount // 2
                right_trim = trim_amount - left_trim
                m2_c2_idx = int(np.where(peaks_c2 == m2_c2)[0][0])   # index of m2_c2
                if m2_c2_idx == 0 or m2_c2_idx >= len(peaks_c2) - 1:
                    continue
                if m2_c2 < m1_c2:
                    peak_on_left = peaks_c2[m2_c2_idx - 1]
                    min_pos_peak_left = np.argmin(c2_segment_aligned[peak_on_left:peaks_c2[m2_c2_idx]]) + peak_on_left
                    c2_segment_aligned = np.concatenate([
                        c2_segment_aligned[:min_pos_peak_left],
                        np.full(trim_amount, c2_segment_aligned[min_pos_peak_left]),
                        c2_segment_aligned[min_pos_peak_left:min_pos_c2 - left_trim],
                        c2_segment_aligned[min_pos_c2 + right_trim:]
                    ])
                elif m2_c2 > m1_c2:
                    peak_on_right = peaks_c2[m2_c2_idx + 1]
                    min_pos_peak_right = np.argmin(c2_segment_aligned[m2_c2:peak_on_right]) + m2_c2
                    c2_segment_aligned = np.concatenate([
                        c2_segment_aligned[:min_pos_c2 - trim_amount],
                        c2_segment_aligned[min_pos_c2:min_pos_peak_right],
                        np.full(trim_amount, c2_segment_aligned[min_pos_peak_right]),
                        c2_segment_aligned[min_pos_peak_right:]
                    ])
                # c2_segment_aligned = np.concatenate([
                #     c2_segment_aligned[:min_pos_c2 - left_trim],
                #     c2_segment_aligned[min_pos_c2 + right_trim:]
                # ])
                # c2_segment_aligned = np.concatenate([np.full(trim_amount, c2_segment_aligned[0]), c2_segment_aligned])

            peaks_c2, _ = find_peaks(c2_segment_aligned, height=threshold_c2)
            sorted_peaks_c2 = peaks_c2[np.argsort(c2_segment_aligned[peaks_c2])[::-1]]
            if len(sorted_peaks_c2) < initial_peaks:
                break

        return c2_segment_aligned


    def align_and_scale_signals(self, c1, c2, proximity_threshold, peak_prominence):
        # Find peaks
        threshold_c1 = min(c1) + np.std(c1) * self.threshold
        threshold_c2 = min(c2) + np.std(c2) * self.threshold
        c1_peaks, _ = find_peaks(c1, height=threshold_c1, prominence=peak_prominence)
        c2_peaks, _ = find_peaks(c2, height=threshold_c2, prominence=peak_prominence)

        # Initial alignment of the first peak
        c2p1 = c2_peaks[0]
        c1p1 = c1_peaks[np.argmin(np.abs(c1_peaks - c2p1))]
        if np.abs(c1p1 - c2p1) <= proximity_threshold:
            shift = c1p1 - c2p1
            c2_aligned = self.shift_chromatogram(c2, shift, c2[0], c2[-1])
            c2_peaks, _ = find_peaks(c2_aligned, height=threshold_c2, prominence=peak_prominence)
        else:
            return c2  # If the first peak is out of the proximity threshold, return the original c2

        initial_npeaks = len(c2_peaks)
        for i in range(1, initial_npeaks):
            c2p_prev = c2_peaks[i - 1]
            c2p = c2_peaks[i]
            c1p_prev = c2p_prev
            c1p = c1_peaks[np.argmin(np.abs(c1_peaks - c2p))]

            # If the current peak in c2 is too far from the closest peak in c1, skip it
            if np.abs(c1p - c2p) > proximity_threshold:
                continue

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

            c2_aligned = np.concatenate([c2_aligned[:start], scaled_segment, c2_aligned[end:]])
            c2_peaks, _ = find_peaks(c2_aligned, height=threshold_c2, prominence=peak_prominence)

            if len(c2_peaks) < initial_npeaks:
                break

            # temp_c2_aligned = np.concatenate([c2_aligned[:start], scaled_segment, c2_aligned[end:]])
            # temp_c2_peaks, _ = find_peaks(temp_c2_aligned, height=threshold_c2, prominence=peak_prominence)
            #
            # if len(temp_c2_peaks) < initial_npeaks:
            #     break
            #
            # # Calculate the average separation between peaks for both current and scaled alignments
            # min_length = min(len(c1_peaks), len(c2_peaks), len(temp_c2_peaks))
            # current_avg_sep = np.mean(np.abs(np.diff(c2_peaks[:min_length])))
            # scaled_avg_sep = np.mean(np.abs(np.diff(temp_c2_peaks[:min_length])))
            #
            # # Skip the peak if the average separation with the current scaling is better
            # if current_avg_sep < scaled_avg_sep:
            #     continue
            #
            # c2_aligned = temp_c2_aligned
            # c2_peaks = temp_c2_peaks



        return c2_aligned


    # def align_maxima(self, c1_segment, c2_segment):
    #
    #     threshold_c1 = min(c1_segment) + np.std(c1_segment) * self.threshold
    #     threshold_c2 = min(c2_segment) + np.std(c2_segment) * self.threshold
    #     peaks_c1, _ = find_peaks(c1_segment, height=threshold_c1)
    #     peaks_c2, _ = find_peaks(c2_segment, height=threshold_c2)
    #
    #     # Return the original segment if no peaks are found in either c1 or c2
    #     if len(peaks_c1) == 0 or len(peaks_c2) == 0:
    #         return c2_segment
    #
    #     # Sort peaks in c2 by their amplitude in descending order
    #     sorted_peaks_c2 = peaks_c2[np.argsort(c2_segment[peaks_c2])[::-1]]
    #     c2_segment_aligned = np.copy(c2_segment)
    #
    #     # Align the highest peak in c2 with the closest peak in c1
    #     m1_c2 = sorted_peaks_c2[0]
    #     m1_c1 = peaks_c1[np.argmin(np.abs(peaks_c1 - m1_c2))]
    #     shift = m1_c1 - m1_c2
    #     c2_segment_aligned = self.shift_chromatogram(c2_segment_aligned, shift, c2_segment[0], c2_segment[-1])
    #     # sorted_peaks_c2[0] = m1_c1
    #
    #     # Align subsequent peaks by padding or trimming
    #     for i in range(1, min(len(peaks_c1), len(sorted_peaks_c2))):
    #         m1_c2 = sorted_peaks_c2[i - 1]
    #         m2_c2 = sorted_peaks_c2[i]
    #         m1_c1 = peaks_c1[np.argmin(np.abs(peaks_c1 - m1_c2))]
    #         m2_c1 = peaks_c1[np.argmin(np.abs(peaks_c1 - m2_c2))]
    #
    #         # Skip if peak order is incorrect
    #         if m1_c2 >= m2_c2 or m1_c1 >= m2_c1:
    #             continue
    #         # Ensure m1_c1 and m2_c1 are in the correct order
    #         if m1_c1 > m2_c1:
    #             m1_c1, m2_c1 = m2_c1, m1_c1
    #         if m1_c2 > m2_c2:
    #             m1_c2, m2_c2 = m2_c2, m1_c2
    #
    #         # Find the minimum value in the c2 segment between the peaks
    #         interval_c2_segment = c2_segment_aligned[m1_c2:m2_c2]
    #         min_pos_c2 = np.argmin(interval_c2_segment) + m1_c2
    #         min_value_c2 = np.min(interval_c2_segment)
    #
    #         # Calculate the intervals between peaks in c1 and c2
    #         interval_c1 = m2_c1 - m1_c1
    #         interval_c2 = m2_c2 - m1_c2
    #
    #         # Pad or trim c2 to match the interval in c1
    #         if interval_c1 > interval_c2:
    #             # Pad c2 with the minimum value found between the peaks
    #             pad_amount = interval_c1 - interval_c2
    #
    #             c2_segment_aligned = np.concatenate([
    #                 c2_segment_aligned[:min_pos_c2],
    #                 np.full(pad_amount, min_value_c2),
    #                 c2_segment_aligned[min_pos_c2:]
    #             ])
    #         elif interval_c2 > interval_c1:
    #             # Trim c2 to match the interval in c1 by removing data around the minimum value
    #             trim_amount = interval_c2 - interval_c1
    #             left_trim = trim_amount // 2
    #             right_trim = trim_amount - left_trim
    #             c2_segment_aligned = np.concatenate([
    #                 c2_segment_aligned[:min_pos_c2 - left_trim],
    #                 c2_segment_aligned[min_pos_c2 + right_trim:]
    #             ])
    #
    #     return c2_segment_aligned


    def shift_chromatogram(self, chrom, lag, left_value=0, right_value=0):
        shifted_chrom = np.roll(chrom, lag)
        if lag > 0:
            shifted_chrom[:lag] = left_value  # Set the elements shifted in from the right
        elif lag < 0:
            shifted_chrom[-lag:] = right_value  # Set the elements shifted in from the left
        return shifted_chrom


    def adjust_chromatogram(self):
        segment_length = len(self.c2) // self.n_segments
        final_corrected_c2 = []

        for i in range(self.n_segments):
            start = i * segment_length
            end = (i + 1) * segment_length if i < self.n_segments - 1 else len(self.c2)

            c1_segment = self.c1[start:end]


            if i == 0:
                c2_segment = self.c2[start:end]
                # best_scale, best_lag = self.find_best_scale_and_lag(c1_segment, c2_segment)
                best_scale, best_lag = self.find_best_scale_and_lag(self.c1[:3000], self.c2[:3000])
                corrected_c2_full = self.correct_segment(self.c2, best_scale, best_lag)
                # corrected_c2_full = self.correct_segment(self.c2, 0.998, best_lag)
                # c2_segment_aligned = self.align_maxima(c1_segment, corrected_c2_full[start:end])
                c2_segment_aligned = self.align_and_scale_signals(
                    c1_segment, corrected_c2_full[start:end], proximity_threshold=25, peak_prominence=0.9)

            else:
                c2_segment = corrected_c2_full[start:end]
                # c2_segment_aligned = self.align_maxima(c1_segment, c2_segment)
                c2_segment_aligned = self.align_and_scale_signals(
                    c1_segment, c2_segment, proximity_threshold=25, peak_prominence=0.9)

            final_corrected_c2.extend(c2_segment_aligned)

        return np.array(final_corrected_c2)

    def plot_chromatograms(self, corrected_c2):
        plt.figure(figsize=(16, 4))
        plt.plot(self.c1, label='Chromatogram 1')
        plt.plot(self.c2, label='Chromatogram 2 (Original)', linestyle='--')
        plt.plot(corrected_c2, label='Chromatogram 2 (Corrected)')
        plt.xlabel('Index')
        plt.ylabel('Intensity')
        plt.title('Chromatogram Adjustment')
        plt.legend()
        plt.grid(True)
        plt.show()


# class SyncChromatograms:
#     def __init__(self, c1, c2, n_segments, scales, min_peaks=5, max_iterations=100, threshold=0.1):
#         self.c1 = c1
#         self.c2 = c2
#         self.n_segments = n_segments
#         self.scales = scales
#         self.min_peaks = min_peaks
#         self.max_iterations = max_iterations
#         self.threshold = threshold
#
#     def scale_chromatogram(self, chrom, scale):
#         x = np.arange(len(chrom))
#         scaled_length = int(len(chrom) * scale)
#         scaled_x = np.linspace(0, len(chrom) - 1, num=scaled_length)
#         f = interp1d(x, chrom, bounds_error=False, fill_value="extrapolate")
#         return f(scaled_x)
#
#     def find_largest_peaks(self, segment, num_peaks):
#         peaks, _ = find_peaks(segment)
#         if len(peaks) > num_peaks:
#             largest_peaks_indices = np.argsort(segment[peaks])[-num_peaks:]
#             largest_peaks = peaks[largest_peaks_indices]
#         else:
#             largest_peaks = peaks
#         return np.sort(largest_peaks)
#
#     def find_best_scale_and_lag(self, c1_segment, c2_segment, initial_lag=None):
#         best_scale = None
#         best_lag = None
#         best_corr = -np.inf
#
#         c1_segment_padded = np.where(c1_segment > self.threshold, c1_segment, 0)
#
#         for scale in self.scales:
#             scaled_c2_segment = self.scale_chromatogram(c2_segment, scale)
#             scaled_c2_segment_padded = np.where(scaled_c2_segment > self.threshold, scaled_c2_segment, 0)
#
#             corr = correlate(c1_segment_padded, scaled_c2_segment_padded, mode='full')
#             lag = np.argmax(corr) - (len(c1_segment_padded) - 1)
#
#             if np.max(corr) > best_corr:
#                 best_corr = np.max(corr)
#                 best_scale = scale
#                 best_lag = lag
#
#         return best_scale, best_lag
#
#     def correct_segment(self, segment, scale, lag):
#         scaled_segment = self.scale_chromatogram(segment, scale)
#         if lag > 0:
#             corrected_segment = np.roll(scaled_segment, lag)
#             corrected_segment[:lag] = scaled_segment[0]
#         else:
#             corrected_segment = np.roll(scaled_segment, lag)
#             corrected_segment[lag:] = scaled_segment[-1]
#         return corrected_segment
#
#     def adjust_chromatogram(self):
#         segment_length = len(self.c2) // self.n_segments
#
#         start = 0
#         end = segment_length
#         c1_segment = self.c1[start:end]
#         c2_segment = self.c2[start:end]
#
#         for i in range(10):
#             best_scale, best_lag = self.find_best_scale_and_lag(c1_segment, c2_segment)
#             initial_best_lag = best_lag
#             corrected_c2_full = self.correct_segment(self.c2, best_scale, best_lag)
#
#         final_corrected_c2 = []
#
#         for i in range(self.n_segments):
#             start = i * segment_length
#             end = (i + 1) * segment_length if i < self.n_segments - 1 else len(corrected_c2_full)
#
#             c1_segment = self.c1[start:end]
#             c2_segment = corrected_c2_full[start:end]
#
#             best_scale, best_lag = self.find_best_scale_and_lag(c1_segment, c2_segment, initial_lag=initial_best_lag)
#             corrected_segment = self.correct_segment(c2_segment, best_scale, best_lag)
#
#             final_corrected_c2.extend(corrected_segment)
#
#         return np.array(final_corrected_c2)
#
#     def plot_chromatograms(self, corrected_c2):
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.c1, label='Chromatogram 1')
#         plt.plot(self.c2, label='Chromatogram 2 (Original)', linestyle='--')
#         plt.plot(corrected_c2, label='Chromatogram 2 (Corrected)', linestyle='-.')
#         plt.xlabel('Index')
#         plt.ylabel('Intensity')
#         plt.title('Chromatogram Adjustment')
#         plt.legend()
#         plt.grid(True)
#         plt.show()


# class SyncChromatograms:
#     def __init__(self, c1, c2, n_segments, scales, min_peaks=5, max_iterations=100):
#         self.c1 = c1
#         self.c2 = c2
#         self.n_segments = n_segments
#         self.scales = scales
#         self.min_peaks = min_peaks
#         self.max_iterations = max_iterations
#
#     def scale_chromatogram(self, chrom, scale):
#         x = np.arange(len(chrom))
#         scaled_length = int(len(chrom) * scale)
#         scaled_x = np.linspace(0, len(chrom) - 1, num=scaled_length)
#         f = interp1d(x, chrom, bounds_error=False, fill_value="extrapolate")
#         return f(scaled_x)  # Ensure the scaled array matches the original length
#
#     def find_largest_peaks(self, segment, num_peaks):
#         peaks, _ = find_peaks(segment)
#         if len(peaks) > num_peaks:
#             # Sort peaks by their heights and select the largest ones
#             largest_peaks_indices = np.argsort(segment[peaks])[-num_peaks:]
#             largest_peaks = peaks[largest_peaks_indices]
#         else:
#             largest_peaks = peaks
#         return np.sort(largest_peaks)  # Ensure peaks are ordered by their indices
#
#     def find_best_scale_and_lag(self, c1_segment, c2_segment, initial_lag=None):
#         best_scale = None
#         best_lag = None
#         best_diff = np.inf
#
#         peaks_c1 = self.find_largest_peaks(c1_segment, self.min_peaks)
#         peaks_c1 = np.sort(peaks_c1)  # Ensure peaks are ordered by their indices
#
#         lag_range = int(0.2 * len(c1_segment))  # 20% of the segment length
#
#         # Set the initial lag range for the first segment or use a narrower range for subsequent segments
#         if initial_lag is not None:
#             lag_start = -initial_lag
#             lag_end = initial_lag
#             # lag_start = max(-lag_range, initial_lag - lag_range)
#             # lag_end = min(lag_range, initial_lag + lag_range)
#         else:
#             lag_start = -lag_range
#             lag_end = lag_range
#
#         # Scan lags within the specified range
#         for lag in range(lag_start, lag_end + 1):
#             for scale in self.scales:
#                 scaled_c2_segment = self.scale_chromatogram(c2_segment, scale)
#                 peaks_scaled_c2 = self.find_largest_peaks(scaled_c2_segment, self.min_peaks)
#                 peaks_scaled_c2 = np.sort(peaks_scaled_c2)  # Ensure peaks are ordered by their indices
#
#                 if len(peaks_scaled_c2) == 0:
#                     continue
#
#                 shifted_peaks_scaled_c2 = peaks_scaled_c2 + lag
#                 valid_indices = (shifted_peaks_scaled_c2 >= 0) & (shifted_peaks_scaled_c2 < len(c1_segment))
#                 shifted_peaks_scaled_c2 = shifted_peaks_scaled_c2[valid_indices]
#
#                 if len(shifted_peaks_scaled_c2) == 0:
#                     continue
#
#                 # Calculate the average absolute distance between peaks in order
#                 distances = []
#                 for peak1, peak2 in zip(peaks_c1, shifted_peaks_scaled_c2):
#                     distances.append(abs(peak1 - peak2))
#
#                 diff = np.mean(distances)
#
#                 if diff < best_diff:
#                     best_diff = diff
#                     best_scale = scale
#                     best_lag = lag
#
#         return best_scale, best_lag
#
#
#     def correct_segment(self, segment, scale, lag):
#         scaled_segment = self.scale_chromatogram(segment, scale)
#         if lag > 0:
#             corrected_segment = np.roll(scaled_segment, lag)
#             corrected_segment[:lag] = scaled_segment[0]  # Repeat the first value
#         else:
#             corrected_segment = np.roll(scaled_segment, lag)
#             corrected_segment[lag:] = scaled_segment[-1]  # Repeat the last value
#         return corrected_segment
#
#     def adjust_chromatogram(self):
#         segment_length = len(self.c2) // self.n_segments
#
#         # Process the first segment to determine the best lag and scale
#         start = 0
#         end = segment_length
#         c1_segment = self.c1[start:end]
#         c2_segment = self.c2[start:end]
#
#         best_scale, best_lag = self.find_best_scale_and_lag(c1_segment, c2_segment)
#         initial_best_lag = best_lag
#
#         # Correct the entire c2 chromatogram using the best scale and lag from the first segment
#         corrected_c2_full = self.correct_segment(self.c2, best_scale, best_lag)
#
#         # Now process segments from the corrected c2
#         final_corrected_c2 = []
#
#         for i in range(self.n_segments):
#             start = i * segment_length
#             end = (i + 1) * segment_length if i < self.n_segments - 1 else len(corrected_c2_full)
#
#             c1_segment = self.c1[start:end]
#             c2_segment = corrected_c2_full[start:end]
#
#             best_scale, best_lag = self.find_best_scale_and_lag(c1_segment, c2_segment, initial_lag=initial_best_lag)
#             corrected_segment = self.correct_segment(c2_segment, best_scale, best_lag)
#
#             final_corrected_c2.extend(corrected_segment)
#
#         return np.array(final_corrected_c2)
#
#
#     def plot_chromatograms(self, corrected_c2):
#         plt.figure(figsize=(10, 6))
#         plt.plot(self.c1, label='Chromatogram 1')
#         plt.plot(self.c2, label='Chromatogram 2 (Original)', linestyle='--')
#         plt.plot(corrected_c2, label='Chromatogram 2 (Corrected)', linestyle='-.')
#         plt.xlabel('Index')
#         plt.ylabel('Intensity')
#         plt.title('Chromatogram Adjustment')
#         plt.legend()
#         plt.grid(True)
#         plt.show()