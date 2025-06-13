"""
Ridge Regression - One Model Per Taster
----------------------------------------

This script trains one Ridge regression model per taster.
Each model uses only that taster's samples (chromatograms + sensory attributes).
Cross-validation is used to estimate performance.
Chromatogram feature weights are saved per taster for later comparison.

Author: Luis G Camara
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from gcmswine import utils  # Your custom module
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.metrics import r2_score
import os
import yaml
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from gcmswine.wine_analysis import ChromatogramAnalysis, GCMSDataProcessor
from gcmswine.helpers import (
    load_config, get_model, load_chromatograms_decimated, load_and_clean_metadata, build_feature_target_arrays
)

if __name__ == "__main__":
    # Load configuration parameters from YAML
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))

    # --- Parameters ---
    N_DECIMATION = 10
    N_SPLITS = 5  # CV folds
    N_REPEATS = 20
    # RANDOM_SEED = 42
    random_seed = 42                   # For reproducibility
    row_start, row_end = 0, None
    directory = "/home/luiscamara/Documents/datasets/Champagnes/HETEROCYC"
    metadata_path = "/home/luiscamara/Documents/datasets/Champagnes/sensory_scores.csv"

    # Load user-defined options
    normalize = config.get("normalize", True)
    retention_time_range = config.get("rt_range", None)
    show_taster_predictions = config.get("show_predicted_profiles", False)
    taster_scaling = config.get("taster_scaling", False)
    model_name = config.get("regressor", "ridge").lower()
    num_repeats = config.get("num_repeats", 10)
    shuffle_labels = config.get("shuffle_labels", False)

    # Select the model based on user input
    model = get_model(model_name, random_seed=random_seed)
    print(f'Regressor is {model_name}')

    # --- Load chromatograms ---
    data_dict, chome_length = load_chromatograms_decimated(
        directory, row_start, row_end, N_DECIMATION, retention_time_range=retention_time_range
    )

    # --- Load metadata ---
    metadata = load_and_clean_metadata(metadata_path)

    # --- Explore variability of sensory attributes ---
    known_metadata = ['code vin', 'taster', 'prod area', 'variety', 'cave', 'age']
    sensory_cols = [col for col in metadata.columns if
                    col not in known_metadata and pd.api.types.is_numeric_dtype(metadata[col])]

    # Convert sensory columns to numeric
    metadata[sensory_cols] = metadata[sensory_cols].apply(pd.to_numeric, errors='coerce')

    print("\nVariability of sensory attributes (standard deviation):")
    for col in sensory_cols:
        std_dev = metadata[col].std()
        min_val = metadata[col].min()
        max_val = metadata[col].max()
        print(f"{col:>20s}: Std = {std_dev:6.2f}   Range = ({min_val:.1f}–{max_val:.1f})")

    # # --- Prepare sensory columns ---
    # known_metadata = ['code vin', 'taster', 'prod area', 'variety', 'cave', 'age']
    # sensory_cols = [col for col in metadata.columns if
    #                 col not in known_metadata and pd.api.types.is_numeric_dtype(metadata[col])]
    # metadata[sensory_cols] = metadata[sensory_cols].apply(pd.to_numeric, errors='coerce')

    # --- Group replicates ---
    metadata = metadata.groupby(['code vin', 'taster'], as_index=False)[sensory_cols].mean()

    # --- Build input (X) and output (y) ---
    X_raw, y, taster_ids, sample_ids = [], [], [], []
    skipped_count = 0

    X_raw, y, taster_ids, sample_ids = build_feature_target_arrays(metadata, sensory_cols, data_dict)

    # --- Group samples by taster ---
    taster_data = defaultdict(list)
    for i, taster in enumerate(taster_ids):
        taster_data[taster].append(i)

    # --- Train one Ridge model per taster ---
    per_taster_mae = {}
    per_taster_rmse = {}
    per_taster_weights = {}
    per_taster_r2 = {}

    print("\nTraining Ridge models per taster...")

    for taster, indices in taster_data.items():
        if len(indices) < N_SPLITS:
            print(f"Skipping taster {taster} (not enough samples for CV)")
            continue

        X_taster = X_raw[indices]
        y_taster = y[indices]
        mae_list = []
        rmse_list = []
        coef_list = []  # <--- save all coefficients here
        r2_list = []

        n_splits = 5
        for repeat in range(num_repeats):
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed + repeat)

            for train_idx, test_idx in kf.split(X_taster):
                X_train, X_test = X_taster[train_idx], X_taster[test_idx]
                y_train, y_test = y_taster[train_idx], y_taster[test_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # model = Ridge()
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
                rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

                mae_list.append(mae)
                rmse_list.append(rmse)

                coef_list.append(model.coef_)  # <--- save model coefficients

                # Compute multivariate R² as:
                # 1 - (sum of squared errors) / (total variance of true values)
                ss_res = np.sum((y_test - y_pred) ** 2)
                y_test_mean = np.mean(y_test, axis=0)
                ss_tot = np.sum((y_test - y_test_mean) ** 2)
                r2_multivariate = 1 - ss_res / ss_tot
                r2_list.append(r2_multivariate)

        # Average errors
        mean_mae = np.mean(mae_list, axis=0)
        mean_rmse = np.mean(rmse_list, axis=0)

        # Average Ridge weights across splits
        mean_coef = np.mean(np.array(coef_list), axis=0)  # <--- average all coefs

        per_taster_mae[taster] = mean_mae
        per_taster_rmse[taster] = mean_rmse
        per_taster_weights[taster] = mean_coef  # <--- save average coefficients here

        per_taster_r2[taster] = np.mean(r2_list)  # mean across repeats, 1 scalar per taster

    print("\nDone training per taster!")

    # --- Example: Print summary of per-taster performance ---
    print("\nPer-taster average MAE (across all sensory attributes):")
    for taster, mae in per_taster_mae.items():
        print(f"Taster {taster}: MAE = {np.mean(mae):.2f}")

    print("\nR² score per taster (multivariate):\n")
    print(f"{'Taster':<10} R²")
    print("-" * 20)
    for taster, r2_val in per_taster_r2.items():
        print(f"{taster:<10} {r2_val:6.3f}")


    attributes_to_plot = ['fruity', 'citrus', 'maturated fruits', 'candied citrus', 'toasted',
           'nuts', 'spicy', 'petroleum', 'undergroth', 'babery', 'honey', 'diary',
           'herbal', 'tobaco', 'texture', 'acid', 'ageing']

    # attributes_to_plot  = ['fruity', 'citrus', 'maturated fruits', 'candied citrus', 'toasted',  'nuts', 'spicy', 'petroleum', 'undergroth']
    # attributes_to_plot  = ['undergroth', 'babery', 'honey', 'diary', 'herbal', 'tobaco', 'texture', 'acid', 'ageing']

    def plot_top_peaks_selected_attributes(per_taster_weights, sensory_cols, X_raw_shape, attributes_to_plot, maxima_rank=1):
        window_size = 5
        use_absolute = True
        n_chromatogram_features = X_raw_shape[1]
        unique_tasters = sorted(per_taster_weights.keys())
        colors = cm.get_cmap('tab20', len(unique_tasters))

        n_cols = 3
        n_rows = int(np.ceil(len(attributes_to_plot) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        for i, attribute_to_plot in enumerate(attributes_to_plot):
            if attribute_to_plot not in sensory_cols:
                print(f"Warning: Attribute '{attribute_to_plot}' not found, skipping...")
                continue

            ax = axes[i]
            attr_idx = sensory_cols.index(attribute_to_plot)

            for idx, taster in enumerate(unique_tasters):
                weights = per_taster_weights[taster]
                chromatogram_weights = weights[attr_idx, :n_chromatogram_features]

                if use_absolute:
                    chromatogram_weights = np.abs(chromatogram_weights)

                sorted_indices = np.argsort(chromatogram_weights)[-maxima_rank:][::-1]
                if len(sorted_indices) < maxima_rank:
                    top_idx = sorted_indices[-1]
                else:
                    top_idx = sorted_indices[maxima_rank-1]

                top_value = chromatogram_weights[top_idx]

                # Scatter point
                ax.scatter(top_idx, top_value, label=f"Taster {taster}", s=60, alpha=0.7, color=colors(idx))

                # Label above the dot
                ax.text(top_idx, top_value + 0.01 * np.max(chromatogram_weights), taster, fontsize=10, ha='center', va='bottom')

            ax.set_title(f"{attribute_to_plot} (Top-{maxima_rank} peak)", fontsize=14)
            ax.set_xlabel("Chromatogram position")
            ax.set_ylabel("Abs Ridge weight")
            ax.grid(True)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Top-{maxima_rank} chromatogram peaks across tasters (selected attributes)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)
    # plot_top_peaks_selected_attributes(
    #     per_taster_weights=per_taster_weights,
    #     sensory_cols=sensory_cols,
    #     X_raw_shape=X_raw.shape,
    #     attributes_to_plot=attributes_to_plot,
    #     maxima_rank=1
    # )

    def plot_strongest_focus_per_attribute(per_taster_weights, sensory_cols, X_raw_shape, attributes_to_plot, maxima_rank=1):
        n_chromatogram_features = X_raw_shape[1]
        unique_tasters = sorted(per_taster_weights.keys())

        n_cols = 3
        n_rows = int(np.ceil(len(attributes_to_plot) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        axes = axes.flatten()

        cmap = cm.get_cmap('tab20', len(unique_tasters))
        taster_to_color_idx = {taster: idx for idx, taster in enumerate(unique_tasters)}

        for i, attribute_to_plot in enumerate(attributes_to_plot):
            if attribute_to_plot not in sensory_cols:
                print(f"Warning: Attribute '{attribute_to_plot}' not found, skipping...")
                continue

            ax = axes[i]
            attr_idx = sensory_cols.index(attribute_to_plot)

            peak_positions = []
            taster_labels = []

            for taster in unique_tasters:
                weights = per_taster_weights[taster]
                chromatogram_weights = weights[attr_idx, :n_chromatogram_features]
                chromatogram_weights = np.abs(chromatogram_weights)

                sorted_peak_indices = np.argsort(chromatogram_weights)[-maxima_rank:][::-1]

                if len(sorted_peak_indices) < maxima_rank:
                    peak_idx = sorted_peak_indices[-1]
                else:
                    peak_idx = sorted_peak_indices[maxima_rank-1]

                peak_positions.append(peak_idx)
                taster_labels.append(taster)

            # Sort tasters by peak position
            peak_positions = np.array(peak_positions)
            taster_labels = np.array(taster_labels)
            sort_idx = np.argsort(peak_positions)
            peak_positions = peak_positions[sort_idx]
            taster_labels = taster_labels[sort_idx]

            # Plot points
            for idx, (pos, taster) in enumerate(zip(peak_positions, taster_labels)):
                color_idx = taster_to_color_idx[taster]
                ax.scatter(pos, idx, color=cmap(color_idx), s=40)
                ax.text(pos + 6, idx, taster, fontsize=10, va='center', ha='left')

            ax.set_title(f"{attribute_to_plot} (Top-{maxima_rank} peak)", fontsize=10)
            # ax.set_xlabel("Chromatogram position (proxy for retention time)")
            ax.set_yticks([])
            ax.set_ylim(-1, len(taster_labels))
            ax.grid(True)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Strongest chromatogram focus per attribute – Top-{maxima_rank} peak", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)
    # plot_strongest_focus_per_attribute(
    #     per_taster_weights=per_taster_weights,
    #     sensory_cols=sensory_cols,
    #     X_raw_shape=X_raw.shape,
    #     attributes_to_plot=attributes_to_plot,
    #     maxima_rank=1
    # )


    def plot_vertical_focus_map_multiple_attributes(per_taster_weights, sensory_cols, X_raw_shape, attributes_to_plot,
                                                    maxima_rank=1):
        """
        Plot vertical focus maps for multiple attributes (one subplot per attribute).

        Args:
            per_taster_weights: dict {taster: Ridge coefficients matrix (n_attributes x n_features)}
            sensory_cols: list of attribute names
            X_raw_shape: shape of the X_raw matrix
            attributes_to_plot: list of attributes to plot
            maxima_rank: number of top peaks to show (default = 1)
        """
        n_chromatogram_features = X_raw_shape[1]
        unique_tasters = sorted(per_taster_weights.keys())

        n_cols = 3
        n_rows = int(np.ceil(len(attributes_to_plot) / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = axes.flatten()

        colors = cm.get_cmap('tab20', len(unique_tasters))
        taster_to_y = {taster: i for i, taster in enumerate(unique_tasters)}

        for i, attribute_to_plot in enumerate(attributes_to_plot):
            if attribute_to_plot not in sensory_cols:
                print(f"Warning: Attribute '{attribute_to_plot}' not found, skipping...")
                continue

            attr_idx = sensory_cols.index(attribute_to_plot)

            taster_positions = []
            taster_labels = []
            taster_peak_ranks = []

            for taster in unique_tasters:
                weights = per_taster_weights[taster]
                chromatogram_weights = weights[attr_idx, :n_chromatogram_features]
                chromatogram_weights = np.abs(chromatogram_weights)

                # Find top maxima_rank peaks
                sorted_peak_indices = np.argsort(chromatogram_weights)[-maxima_rank:][::-1]

                for rank, peak_idx in enumerate(sorted_peak_indices):
                    taster_positions.append(peak_idx)
                    taster_labels.append(taster)
                    taster_peak_ranks.append(rank + 1)

            ax = axes[i]

            for taster, peak_pos, rank in zip(taster_labels, taster_positions, taster_peak_ranks):
                y = taster_to_y[taster]
                color_idx = unique_tasters.index(taster)
                ax.scatter(peak_pos, y, s=80 / (rank ** 0.7), color=colors(color_idx), alpha=0.8, edgecolors='black')

            ax.set_yticks(range(len(unique_tasters)))
            ax.set_yticklabels(unique_tasters, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{attribute_to_plot}", fontsize=12)
            ax.grid(True, axis='x')
            ax.grid(False, axis='y')
            ax.set_xlim(0, n_chromatogram_features)
            if i % n_cols == 0:
                ax.set_ylabel("Tasters")
            else:
                ax.set_yticklabels([])  # Hide Y-ticks for internal plots
            ax.set_xlabel("Chromatogram position")

        # Delete empty subplots if needed
        for j in range(len(attributes_to_plot), len(axes)):
            fig.delaxes(axes[j])

        plt.suptitle(f"Vertical focus maps (Top-{maxima_rank} peaks per taster)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    # plot_vertical_focus_map_multiple_attributes(
    #     per_taster_weights=per_taster_weights,
    #     sensory_cols=sensory_cols,
    #     X_raw_shape=X_raw.shape,
    #     attributes_to_plot=attributes_to_plot,  # your list: fruity, citrus, toasted, etc.
    #     maxima_rank=5  # Top-3 peaks
    # )

    from scipy.ndimage import gaussian_filter1d


    def plot_focus_heatmap(per_taster_weights, sensory_cols, X_raw_shape, attributes_to_plot, maxima_rank=3, n_bins=50,
                           smoothing_sigma=1.5, rt_min=None, rt_max=None):
        """
        Plot smooth heatmaps of focus density across chromatogram positions for multiple attributes.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        n_chromatogram_features = X_raw_shape[1]
        unique_tasters = sorted(per_taster_weights.keys())

        # Determine chromatogram plotting limits
        if rt_min is None:
            rt_min = 0
        if rt_max is None:
            rt_max = n_chromatogram_features

        # Prepare binning
        bin_edges = np.linspace(0, n_chromatogram_features, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])


        n_cols = 3
        n_rows = int(np.ceil(len(attributes_to_plot) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))

        axes = axes.flatten()


        # Prepare to store one last image for colorbar
        im = None

        for i, attribute_to_plot in enumerate(attributes_to_plot):
            if attribute_to_plot not in sensory_cols:
                print(f"Warning: Attribute '{attribute_to_plot}' not found, skipping...")
                continue

            attr_idx = sensory_cols.index(attribute_to_plot)
            heatmap_bins = np.zeros(n_bins)

            for taster in unique_tasters:
                weights = per_taster_weights[taster]
                chromatogram_weights = weights[attr_idx, :n_chromatogram_features]
                chromatogram_weights = np.abs(chromatogram_weights)

                sorted_peak_indices = np.argsort(chromatogram_weights)[-maxima_rank:][::-1]

                for rank, peak_idx in enumerate(sorted_peak_indices):
                    contribution = 1.0 / (rank + 1)
                    bin_idx = np.searchsorted(bin_edges, peak_idx, side='right') - 1
                    if 0 <= bin_idx < n_bins:
                        heatmap_bins[bin_idx] += contribution

            # Smoothing
            if smoothing_sigma > 0:
                heatmap_bins_smoothed = gaussian_filter1d(heatmap_bins, sigma=smoothing_sigma)
            else:
                heatmap_bins_smoothed = heatmap_bins.copy()  # no smoothing

            # Select bins within the desired retention time range
            valid_bins = (bin_centers >= rt_min) & (bin_centers <= rt_max)
            heatmap_bins_to_plot = heatmap_bins_smoothed[valid_bins]
            bin_centers_to_plot = bin_centers[valid_bins]

            # Plot
            ax = axes[i]
            extent = [rt_min, rt_max, 0, 1]
            heatmap = heatmap_bins_to_plot[np.newaxis, :]
            im = ax.imshow(heatmap, cmap='viridis', aspect='auto', extent=extent, origin='lower')
            ax.set_title(f"{attribute_to_plot}", fontsize=12)
            # ax.set_xlabel("Chromatogram position")
            ax.set_yticks([])

            # Remove unused axes
        for j in range(len(attributes_to_plot), len(axes)):
            fig.delaxes(axes[j])

        # --- Correct space for colorbar ---
        fig.subplots_adjust(right=0.85)  # Make room on the right side
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax)

        plt.suptitle(f"Focus density heatmaps (Top-{maxima_rank} peaks per taster) – Smoothed", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96], h_pad=5.0)  # Don't overlap the colorbar
        plt.show(block=False)
    plot_focus_heatmap(
        per_taster_weights=per_taster_weights,
        sensory_cols=sensory_cols,
        X_raw_shape=X_raw.shape,
        attributes_to_plot=attributes_to_plot,  # your list
        maxima_rank=5,                           # Top-3 peaks
        n_bins=800,                                # 50 bins across chromatogram
        smoothing_sigma=3,
        rt_min=0, rt_max=2500
    )


    def plot_focus_heatmap_per_taster_multiple_attributes(
            per_taster_weights,
            sensory_cols,
            X_raw_shape,
            attributes_to_plot,
            maxima_rank=3,
            n_bins=50,
            smoothing_sigma=1.5,
            cmap='viridis',
            rt_min=None,
            rt_max=None
    ):
        """
        Plot heatmaps per taster for multiple attributes.
        Each subplot = one attribute.

        Args:
            per_taster_weights: dict {taster: Ridge coefficients matrix}
            sensory_cols: list of attribute names
            X_raw_shape: shape of X_raw (n_samples, n_features)
            attributes_to_plot: list of attributes to plot
            maxima_rank: number of top peaks per taster
            n_bins: number of bins across chromatogram
            smoothing_sigma: Gaussian smoothing parameter
            cmap: colormap
            rt_min: minimum retention time (position)
            rt_max: maximum retention time (position)
        """
        n_chromatogram_features = X_raw_shape[1]
        unique_tasters = sorted(per_taster_weights.keys())
        n_tasters = len(unique_tasters)

        if rt_min is None:
            rt_min = 0
        if rt_max is None:
            rt_max = n_chromatogram_features

        bin_edges = np.linspace(0, n_chromatogram_features, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # Subplot layout
        n_cols = 3
        n_rows = int(np.ceil(len(attributes_to_plot) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 5))
        axes = axes.flatten()

        im = None  # store the last imshow for colorbar

        for i, attribute_to_plot in enumerate(attributes_to_plot):
            if attribute_to_plot not in sensory_cols:
                print(f"Warning: Attribute '{attribute_to_plot}' not found, skipping...")
                continue

            heatmap_array = np.zeros((n_tasters, n_bins))
            attr_idx = sensory_cols.index(attribute_to_plot)

            for t, taster in enumerate(unique_tasters):
                weights = per_taster_weights[taster]
                chromatogram_weights = weights[attr_idx, :n_chromatogram_features]
                chromatogram_weights = np.abs(chromatogram_weights)

                sorted_peak_indices = np.argsort(chromatogram_weights)[-maxima_rank:][::-1]

                for rank, peak_idx in enumerate(sorted_peak_indices):
                    contribution = 1.0 / (rank + 1)
                    bin_idx = np.searchsorted(bin_edges, peak_idx, side='right') - 1
                    if 0 <= bin_idx < n_bins:
                        heatmap_array[t, bin_idx] += contribution

            # Smooth if needed
            if smoothing_sigma > 0:
                for t in range(n_tasters):
                    heatmap_array[t, :] = gaussian_filter1d(heatmap_array[t, :], sigma=smoothing_sigma)

            # Select bins within retention time range
            valid_bins = (bin_centers >= rt_min) & (bin_centers <= rt_max)
            heatmap_array = heatmap_array[:, valid_bins]
            bin_centers_to_plot = bin_centers[valid_bins]

            # Plot into correct subplot
            ax = axes[i]
            extent = [rt_min, rt_max, -0.5, n_tasters - 0.5]
            im = ax.imshow(heatmap_array, cmap=cmap, aspect='auto', extent=extent, origin='lower')

            ax.set_yticks(np.arange(n_tasters))
            ax.set_yticklabels(unique_tasters, fontsize=7)
            ax.invert_yaxis()
            ax.set_title(f"{attribute_to_plot}", fontsize=10)
            # ax.set_xlabel("Chromatogram position")

        # Remove unused axes
        for j in range(len(attributes_to_plot), len(axes)):
            fig.delaxes(axes[j])

        # Adjust layout and colorbar
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.suptitle(f"Taster focus heatmaps per attribute (Top-{maxima_rank} peaks)", fontsize=16)
        plt.tight_layout(rect=[0, 0, 0.85, 0.96], h_pad=5.0)

        print(f"Heatmaps plotted: Top-{maxima_rank} peaks per taster used for each attribute.")

        plt.show(block=False)
    plot_focus_heatmap_per_taster_multiple_attributes(
        per_taster_weights=per_taster_weights,
        sensory_cols=sensory_cols,
        X_raw_shape=X_raw.shape,
        attributes_to_plot=attributes_to_plot,
        maxima_rank=5,
        n_bins=800,
        smoothing_sigma=1.5,
        cmap='viridis',
        rt_min=0,
        rt_max=800
    )

    plt.show()

    # print("\nTraining one Ridge model per taster...")
    #
    # # --- Settings ---
    # attribute_to_plot = 'toasted'   # << Attribute you want to analyze
    # window_size = 5                 # << Smoothing window size
    # top_k = 5                       # << Number of important peaks to highlight
    # min_samples_per_taster = 5       # << Minimum samples per taster to be considered
    #
    # # Find attribute index
    # if attribute_to_plot not in sensory_cols:
    #     print(f"Attribute {attribute_to_plot} not found!")
    # else:
    #     attr_idx = sensory_cols.index(attribute_to_plot)
    #
    #     # Group data by taster
    #     taster_data = defaultdict(list)
    #     for i, taster in enumerate(taster_ids):
    #         taster_data[taster].append(i)
    #
    #     # Prepare plot
    #     unique_tasters = sorted(taster_data.keys())
    #     colors = cm.get_cmap('tab20', len(unique_tasters))
    #
    #     plt.figure(figsize=(14, 7))
    #
    #     for idx, taster in enumerate(unique_tasters):
    #         sample_indices = taster_data[taster]
    #
    #         if len(sample_indices) < min_samples_per_taster:
    #             continue  # Skip tasters with too few samples
    #
    #         # Prepare data for this taster
    #         X_taster = X_input[sample_indices]
    #         y_taster = y[sample_indices]
    #
    #         # Train/test split (not really needed, just train on all their samples)
    #         X_train = X_taster
    #         y_train = y_taster
    #
    #         # Scale
    #         scaler = StandardScaler()
    #         X_train_scaled = scaler.fit_transform(X_train)
    #
    #         # Train Ridge
    #         model = Ridge()
    #         model.fit(X_train_scaled, y_train)
    #
    #         # Extract chromatogram weights
    #         n_chromatogram_features = X_raw.shape[1]
    #         chromatogram_weights_taster = model.coef_[attr_idx, :n_chromatogram_features]
    #
    #         # Take absolute value if desired
    #         chromatogram_weights_taster = np.abs(chromatogram_weights_taster)
    #
    #         # Optional: smoothing
    #         smoothed_weights = np.convolve(chromatogram_weights_taster, np.ones(window_size)/window_size, mode='same')
    #
    #         # Plot
    #         plt.plot(smoothed_weights, label=f"Taster {taster}", color=colors(idx), linewidth=2, alpha=0.8)
    #
    #     # Highlight strongest peaks based on one random taster (optional)
    #     plt.title(f"ABSOLUTE Chromatogram feature importance for '{attribute_to_plot}' across tasters")
    #     # plt.xlabel("Chromatogram position (proxy for retention time)")
    #     plt.ylabel("Absolute Ridge Weight")
    #     plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    #     plt.grid(True)
    #     plt.tight_layout()
    #     plt.show()