import os.path
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from scipy.signal import correlate, find_peaks, peak_prominences

class DataLoader:
    def __init__(self, file_path, normalize=True):
        self.file_path = file_path
        self.data, self.df = self.load_data()
        if normalize:
            # Normalise dictionary values
            self.normalize_dict()
            self.df = pd.DataFrame(self.data).T

    def load_data(self):
        print('Loading data...')
        if self.file_path.endswith('.npy'):
            data = np.load(self.file_path, allow_pickle=True).item()
            df = pd.DataFrame(data).T
        elif self.file_path.endswith('.xlsx'):
            df = pd.read_excel(self.file_path)
            data = self.process_xlsx(df, self.file_path)
            df = pd.DataFrame(data).T
        else:
            raise ValueError("Unsupported file format")
        return data, df

    def get_standardized_data(self):
        scaler = StandardScaler()
        return scaler.fit_transform(self.df)

    def normalize_dict(self):
        # Normalise dictionary values
        keys = list(self.data.keys())
        values = np.array(list(self.data.values())).T
        scaler = StandardScaler()
        values_scaled = scaler.fit_transform(values)
        self.data = {key: values_scaled[:, idx].tolist() for idx, key in enumerate(keys)}


    def process_xlsx(self, df, file_path):
        """Implements the logic to appropriately process each dataset based on its file name"""

        files = ['2018 7 chateaux Ester Old vintages Masse 5.xlsx',                 #  0
                 '2018 7 chateaux Oak Old vintages Masse 5.xlsx',                   #  1
                 '2018 7 chateaux Off Old vintages Masse 5.xlsx',                   #  2
                 '2022 01 11 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx', #  3
                 '2022 4 new bordeaux Oak Masse 5 NORMALIZED 052022 SM2 .xlsx',     #  4
                 '2022 01 7 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx',  #  5
                 '2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.xlsx',  #  6
                 ]
        file_name = os.path.basename(file_path)

        if file_name in files[0:3]:
            # Remove first few rows with text
            for i in range(len(df)):
                if df.iloc[i].count() > 1:
                    df = pd.read_excel(self.file_path, skiprows=i)
                    break
            # Remove the last row to avoid nan
            df = df.iloc[:-1]

            data = {}
            for col in df.columns:
                label = df[col][0]  # detect the header
                if not isinstance(label, str):
                    continue
                if "Ab" in label:
                    pattern = r'Ab \S*[_ ]?([A-Z][ _]?\d{4})'
                    key = label.replace('\n', '')
                    key = re.sub(pattern, r'\1', key)  # use re.sub to remove the matching prefix
                    data[key] = [float(value) for value in df[col].tolist()[1:]]

        if file_name in [files[3], files[5], files[6]]:
            # Remove first row (contains text) and last one (for nan)
            df = df.iloc[1:]
            df = df.iloc[:-1]
            df.columns = df.iloc[0]

            data = {}
            for col in df.columns:
                if not isinstance(col, str):
                    continue
                key = col  # detect the header
                data[key] = [float(value) for value in df[col].tolist()[2:]]

        if file_name in files[4]:
            # Remove last 15 rows (zero value)
            df = df.iloc[:-15]

            data = {}
            for col in df.columns:
                if "Unnamed" in col:
                    continue
                key = col  # detect the header
                data[key] =[float(value) for value in df[col].tolist()[1:]]


        return data


class ChromatogramLoader:
    def __init__(self, file_path1, file_path2):
        """
        Initializes the ChromatogramLoader with the paths to the .npy files.

        Parameters:
        file_path1 (str): The path to the first .npy file.
        file_path2 (str): The path to the second .npy file.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2

    def load_chromatogram(self, file_path):
        """
        Loads chromatograms from a .npy file.

        Parameters:
        file_path (str): The path to the .npy file.

        Returns:
        dict: The loaded chromatogram data as a dictionary.
        """
        return np.load(file_path, allow_pickle=True).item()

    def normalize_chromatogram(self, chromatogram):
        """
        Normalizes a chromatogram to the range [0, 1].

        Parameters:
        chromatogram (list): The chromatogram data.

        Returns:
        list: The normalized chromatogram data.
        """
        chromatogram = np.array(chromatogram)
        min_val = np.min(chromatogram)
        max_val = np.max(chromatogram)
        return (chromatogram - min_val) / (max_val - min_val)

    def calculate_mean_chromatogram(self, chromatograms):
        """
        Calculates the mean chromatogram from a dictionary of chromatograms.

        Parameters:
        chromatograms (dict): The chromatogram data as a dictionary.

        Returns:
        np.ndarray: The mean chromatogram data.
        """
        all_data = np.array(list(chromatograms.values()))
        mean_chromatogram = np.mean(all_data, axis=0)
        return mean_chromatogram

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


    def find_initial_offset(self, mean_chromatogram1, mean_chromatogram2):
        """
        Finds the initial offset to align the two mean chromatograms using cross-correlation.

        Parameters:
        mean_chromatogram1 (np.ndarray): The first mean chromatogram data.
        mean_chromatogram2 (np.ndarray): The second mean chromatogram data.

        Returns:
        int: The initial offset value.
        """
        cross_corr = correlate(mean_chromatogram1, mean_chromatogram2)
        lag = np.argmax(cross_corr) - (len(mean_chromatogram2) - 1)
        return lag

    def fine_tune_offset(self, mean_chromatogram1, mean_chromatogram2, initial_offset):
        """
        Fine-tunes the offset to align the two mean chromatograms with weight on the largest peaks.

        Parameters:
        mean_chromatogram1 (np.ndarray): The first mean chromatogram data.
        mean_chromatogram2 (np.ndarray): The second mean chromatogram data.
        initial_offset (int): The initial offset value.

        Returns:
        float: The fine-tuned offset value.
        """
        peaks1, _ = find_peaks(mean_chromatogram1)
        prominences1 = peak_prominences(mean_chromatogram1, peaks1)[0]

        def objective(offset):
            shifted_mean_chromatogram2 = np.roll(mean_chromatogram2, int(offset))
            diff = mean_chromatogram1[peaks1] - shifted_mean_chromatogram2[peaks1]
            return np.sum((diff ** 2) * prominences1)

        result = minimize(objective, initial_offset, method='Powell')
        return result.x

    def find_optimal_offset(self, mean_chromatogram1, mean_chromatogram2):
        """
        Finds the optimal offset to align the two mean chromatograms.

        Parameters:
        mean_chromatogram1 (np.ndarray): The first mean chromatogram data.
        mean_chromatogram2 (np.ndarray): The second mean chromatogram data.

        Returns:
        float: The optimal offset value.
        """
        # Resample chromatograms to the same length
        length = min(len(mean_chromatogram1), len(mean_chromatogram2))
        mean_chromatogram1 = self.resample_chromatogram(mean_chromatogram1, length)
        mean_chromatogram2 = self.resample_chromatogram(mean_chromatogram2, length)

        # Find initial offset using cross-correlation
        initial_offset = self.find_initial_offset(mean_chromatogram1, mean_chromatogram2)

        # Fine-tune the initial offset
        optimal_offset = self.fine_tune_offset(mean_chromatogram1, mean_chromatogram2, initial_offset)
        return optimal_offset

    def plot_chromatograms(self, chromatograms1, chromatograms2, file_name1, file_name2):
        """
        Plots multiple chromatograms for comparison and includes the mean chromatograms.

        Parameters:
        chromatograms1 (dict): The first chromatogram data as a dictionary.
        chromatograms2 (dict): The second chromatogram data as a dictionary.
        file_name1 (str): The name of the first file.
        file_name2 (str): The name of the second file.
        """
        plt.figure(figsize=(12, 12))

        # Plot chromatograms from the first dictionary
        plt.subplot(4, 1, 1)
        for label, data in chromatograms1.items():
            normalized_data = self.normalize_chromatogram(data)
            plt.plot(normalized_data, alpha=0.5)
        plt.title(f'Chromatograms from {file_name1}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Intensity')

        # Plot chromatograms from the second dictionary
        plt.subplot(4, 1, 2)
        for label, data in chromatograms2.items():
            normalized_data = self.normalize_chromatogram(data)
            plt.plot(normalized_data, alpha=0.5)
        plt.title(f'Chromatograms from {file_name2}')
        plt.xlabel('Time')
        plt.ylabel('Normalized Intensity')

        # Calculate and plot original mean chromatograms
        mean_chromatogram1 = self.calculate_mean_chromatogram(chromatograms1)
        mean_chromatogram2 = self.calculate_mean_chromatogram(chromatograms2)

        plt.subplot(4, 1, 3)
        plt.plot(self.normalize_chromatogram(mean_chromatogram1), label=f'Mean {file_name1}', color='blue')
        plt.plot(self.normalize_chromatogram(mean_chromatogram2), label=f'Mean {file_name2}', color='red')
        plt.title('Original Mean Chromatograms')
        plt.xlabel('Time')
        plt.ylabel('Normalized Intensity')

        # Find and plot adjusted mean chromatograms
        optimal_offset = self.find_optimal_offset(mean_chromatogram1, mean_chromatogram2)
        shifted_mean_chromatogram2 = np.roll(mean_chromatogram2, int(optimal_offset))

        plt.subplot(4, 1, 4)
        plt.plot(self.normalize_chromatogram(mean_chromatogram1), label=f'Mean {file_name1}', color='blue')
        plt.plot(self.normalize_chromatogram(shifted_mean_chromatogram2), label=f'Mean {file_name2} (shifted)', color='red')
        plt.title(f'Mean Chromatograms (Optimal Offset: {optimal_offset})')
        plt.xlabel('Time')
        plt.ylabel('Normalized Intensity')

        plt.tight_layout(pad=3.0)
        plt.show()

    def run(self):
        """
        Loads the chromatograms from the provided file paths and plots them.
        """
        chromatograms1 = self.load_chromatogram(self.file_path1)
        chromatograms2 = self.load_chromatogram(self.file_path2)
        file_name1 = self.file_path1.split('/')[-1]
        file_name2 = self.file_path2.split('/')[-1]
        self.plot_chromatograms(chromatograms1, chromatograms2, file_name1, file_name2)
