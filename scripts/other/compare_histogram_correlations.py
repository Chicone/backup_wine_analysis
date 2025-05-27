"""
This script calculates pairwise correlations between multiple histogram files resulting from
different greedy feature selection methods applied to wine data (Merlot and Cabernet Sauvignon).
Each histogram file represents the frequency of selected (5) channels across multiple repeats for
various greedy selection strategies. The script reads the histogram data, computes correlation
coefficients between every pair of histogram files, and organizes the resulting correlations
into a structured correlation matrix. This matrix is then saved as a CSV file and printed for
immediate inspection, facilitating an analysis of similarity between feature selection methods
and wine types.
"""

from plots import *

import os
import numpy as np
import itertools
import pandas as pd

base_path = "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/"

file_names = [
    "hist_5ch_greedy_add_ranked_merlot.csv",
    "hist_5ch_greedy_remove_ranked_merlot.csv",
    "hist_5ch_greedy_add_merlot.csv",
    "hist_5ch_greedy_remove_merlot.csv",
    "hist_5ch_greedy_add_ranked_cab_sauv.csv",
    "hist_5ch_greedy_remove_ranked_cab_sauv.csv",
    "hist_5ch_greedy_add_cab_sauv.csv",
    "hist_5ch_greedy_remove_cab_sauv.csv"
]

short_labels = [chr(ord('A') + i) for i in range(len(file_names))]

# Initialize empty matrix for correlations
correlation_matrix = np.zeros((len(file_names), len(file_names)))

# Compute correlations
for (i, file1), (j, file2) in itertools.product(enumerate(file_names), repeat=2):
    correlation = plot_histogram_correlation(
        os.path.join(base_path, file1),
        os.path.join(base_path, file2),
        wine1=file1,
        wine2=file2,
        show_plots=False
    )
    correlation_matrix[i, j] = round(correlation, 3)

# Convert to DataFrame for easy visualization and CSV export
correlation_df = pd.DataFrame(correlation_matrix, index=short_labels, columns=short_labels)

# Save correlation matrix to CSV
correlation_df.to_csv(os.path.join(base_path, "histogram_correlation_matrix.csv"))

# Print the matrix for quick inspection
print(correlation_df)

# Print the matrix as plain array (no headers or labels)
print("\nMatrix as NumPy array (no headers):")
print(correlation_df.to_numpy())
