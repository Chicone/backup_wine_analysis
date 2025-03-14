"""
Main Analysis Script Overview
==================================

This script serves as the primary entry point for analyzing wine-related chromatographic data. It integrates various
components of the `wine_analysis` framework, including data loading, dimensionality reduction, synchronization, and
classification.

Key Features:
-------------
- **Data Loading**: Loads and normalizes chromatographic data from `.npy` or `.xlsx` files using the `DataLoader` class.
- **Chromatogram Analysis**: Uses the `ChromatogramAnalysis` class to resample, synchronize, and merge chromatograms from different datasets.
- **Dimensionality Reduction**: Applies techniques such as PCA, t-SNE, and UMAP to reduce the dimensionality of the data for easier visualization and analysis.
- **Classification**: Implements various classification strategies using the `Classifier` class, including leave-one-out cross-validation and cross-dataset training and testing.
- **Visualization**: Generates visualizations of chromatograms, synchronization results, and dimensionality reduction outputs.

"""
import time

from traitlets.config import get_config

from config import (
    DATA_DIRECTORY,
    CHEMICAL_NAME,
    DATA_DIRECTORY_2,
    CHEMICAL_NAME_2,
    DATA_DIRECTORY_3,
    CHEMICAL_NAME_3,
    JOIN_DATASETS,
    ROW_START,
    NUM_SPLITS,
    SYNC_STATE,
    CH_TREAT,
    CHROM_CAP,
    N_DECIMATION,
    VINTAGE,
    DATA_TYPE,
    CNN_DIM,
    GCMS_DIRECTION,
    NUM_AGGR_CHANNELS,
    DELAY,
    CLASS_BY_YEAR,
    CHANNEL_METHOD,
    FEATURE_TYPE,
    CONCATENATE_TICS,
    PCA_STATE,
    WINDOW,
    STRIDE,
    WINE_KIND,
    REGION,
    BAYES_OPTIMIZE,
    NUM_SPLITS_BAYES,
    BAYES_CALLS,
)
import os
import numpy as np
import argparse
from data_loader import DataLoader
from classification import (Classifier, process_labels, assign_country_to_pinot_noir, assign_origin_to_pinot_noir,
                            assign_continent_to_pinot_noir, assign_winery_to_pinot_noir, assign_year_to_pinot_noir,
                            assign_north_south_to_beaune, assign_category_to_press_wine, BayesianParamOptimizer,
                            greedy_channel_selection,
                            classify_all_channels,
                            remove_highly_correlated_channels, split_by_split_greedy_channel_selection,
                            greedy_nested_cv_channel_selection, greedy_nested_cv_channel_elimination,
                            greedy_nested_cv_channel_selection_snr, assign_composite_label_to_press_wine,
                            extract_year_from_samples
                            )
from wine_analysis import WineAnalysis, ChromatogramAnalysis, GCMSDataProcessor
from visualizer import visualize_confusion_matrix_3d, plot_accuracy_vs_channels
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import utils
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from plots import (plot_channel_selection_performance_changins, plot_channel_selection_performance_isvv,
                   plot_channel_selection_thresholds, plot_accuracy_all_methods, plot_accuracy_vs_decimation,
                   plot_press_wines_accuracies)

if __name__ == "__main__":
    time.sleep(DELAY)

    # ##### Utils and plots #####
    # plot_classification_accuracy()
    # plot_accuracy_vs_channels()
    # plot_channel_selection_performance_changins()
    # plot_channel_selection_performance_isvv()
    # plot_channel_selection_thresholds("""""")
    # plot_accuracy_all_methods()
    # utils.copy_files_to_matching_directories("/home/luiscamara/kk/", "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/MERLOT/")
    # plot_accuracy_vs_decimation('merlot')
    # plot_accuracy_vs_decimation('cabernet_sauvignon')
    # plot_press_wines_accuracies()
    # utils.rename_directories("/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters21/011222/")
    ###########################

    cl = ChromatogramAnalysis()

    # utils.convert_ms_files_to_cdf_in_place(DATA_DIRECTORY)

    # Set rows in columns to read
    row_start = 1
    row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(DATA_DIRECTORY)
    column_indices = list(range(fc_idx_1, lc_idx_1 + 1))

    # Load primary dataset
    data_dict = utils.load_ms_csv_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end_1)
    dataset_origins = {key: "dataset1" for key in data_dict.keys()}

    # For 2 datasets
    if JOIN_DATASETS:
        row_end_2, fc_idx_2, lc_idx_2 = utils.find_data_margins_in_csv(DATA_DIRECTORY_2)

        # Ensure column indices match
        assert fc_idx_1 == fc_idx_2 and lc_idx_1 == lc_idx_2, "Mismatch in column indices between datasets."

        # Load secondary dataset
        data_dict_2 = utils.load_ms_csv_data_from_directories(DATA_DIRECTORY_2, column_indices, row_start, row_end_2)
        dataset_origins_2 = {key: "dataset2" for key in data_dict_2.keys()}


        # Trim samples to the shortest length for each dataset
        min_length_1 = min(array.shape[0] for array in data_dict.values())
        min_length_2 = min(array.shape[0] for array in data_dict_2.values())
        min_length = min(min_length_1, min_length_2)

        data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
        data_dict_2 = {key: array[:min_length, :] for key, array in data_dict_2.items()}

        # Apply decimation
        data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}
        data_dict_2 = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict_2.items()}

        # Merge datasets without checking keys
        merged_data = {}
        merged_origins = {}
        for key in set(data_dict.keys()).union(data_dict_2.keys()):
            array_1 = data_dict.get(key,np.empty((0, data_dict[next(iter(data_dict))].shape[1])))  # Handle missing keys
            array_2 = data_dict_2.get(key, np.empty((0, data_dict_2[next(iter(data_dict_2))].shape[1])))

            merged_data[key] = np.vstack((array_1, array_2))
            merged_origins[key] = dataset_origins.get(key, "dataset2")

        data_dict = merged_data
        dataset_origins = merged_origins

    # # for  3 datasets
    # if JOIN_DATASETS:
    #     row_end_2, fc_idx_2, lc_idx_2 = utils.find_data_margins_in_csv(DATA_DIRECTORY_2)
    #     row_end_3, fc_idx_3, lc_idx_3 = utils.find_data_margins_in_csv(DATA_DIRECTORY_3)
    #
    #     # Ensure column indices match across all datasets
    #     assert fc_idx_1 == fc_idx_2 == fc_idx_3 and lc_idx_1 == lc_idx_2 == lc_idx_3, "Mismatch in column indices between datasets."
    #
    #     # Load secondary and tertiary datasets
    #     data_dict_2 = utils.load_ms_csv_data_from_directories(DATA_DIRECTORY_2, column_indices, row_start, row_end_2)
    #     data_dict_3 = utils.load_ms_csv_data_from_directories(DATA_DIRECTORY_3, column_indices, row_start, row_end_3)
    #
    #     # Trim samples to the shortest length for each dataset
    #     min_length_1 = min(array.shape[0] for array in data_dict.values())
    #     min_length_2 = min(array.shape[0] for array in data_dict_2.values())
    #     min_length_3 = min(array.shape[0] for array in data_dict_3.values())
    #     min_length = min(min_length_1, min_length_2, min_length_3)
    #
    #     data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
    #     data_dict_2 = {key: array[:min_length, :] for key, array in data_dict_2.items()}
    #     data_dict_3 = {key: array[:min_length, :] for key, array in data_dict_3.items()}
    #
    #     # Apply decimation
    #     data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}
    #     data_dict_2 = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict_2.items()}
    #     data_dict_3 = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict_3.items()}
    #
    #     # Merge datasets without checking keys
    #     merged_data = {}
    #     for key in set(data_dict.keys()).union(data_dict_2.keys()).union(data_dict_3.keys()):
    #         array_1 = data_dict.get(key,
    #                                 np.empty((0, data_dict[next(iter(data_dict))].shape[1])))  # Handle missing keys
    #         array_2 = data_dict_2.get(key, np.empty((0, data_dict_2[next(iter(data_dict_2))].shape[1])))
    #         array_3 = data_dict_3.get(key, np.empty((0, data_dict_3[next(iter(data_dict_3))].shape[1])))
    #
    #         merged_data[key] = np.vstack((array_1, array_2, array_3))
    #     data_dict = merged_data

    else:
        # If not joining datasets, apply decimation only to primary dataset
        min_length = min(array.shape[0] for array in data_dict.values())
        data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
        data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}

    # import scipy.signal as signal
    # data_dict = {key: signal.decimate(matrix, 20, axis=0, zero_phase=True)
    #                       for key, matrix in data_dict.items()}
    CHROM_CAP = CHROM_CAP // N_DECIMATION
    # CHROM_CAP = None
    data_dict, valid_channels = utils.remove_zero_variance_channels(data_dict)
    # utils.plot_snr_per_channel(data_dict)

    gcms = GCMSDataProcessor(data_dict)

    # GCMS-Specific Handling
    if DATA_TYPE == "GCMS":
        if GCMS_DIRECTION == "RT_DIRECTION":
            print("Analyzing GCMS data in retention time direction")
            if SYNC_STATE:
                tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)

            gcms.data = utils.reduce_columns_in_dict(data_dict, NUM_AGGR_CHANNELS, normalize=False)
            chrom_length = len(list(data_dict.values())[0])
            # data, labels = gcms.create_overlapping_windows(chrom_length, chrom_length)  # with chrom_length is one window only
            data, labels = np.array(list(gcms.data.values())),  np.array(list(gcms.data.keys())) # with chrom_length is one window only
            if SYNC_STATE and CONCATENATE_TICS:
                data = np.concatenate((data, np.asarray(list(tics.values()))), axis=0)
                labels = np.concatenate((labels, np.asarray(list(tics.keys()))), axis=0)
    else:  # Handling Other Data Types
        if DATA_TYPE == "TIC":
            if SYNC_STATE:
                tics, _ = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
            else:
                # norm_tics = utils.normalize_dict(gcms.compute_tics(), scaler='standard')
                tics = gcms.compute_tics()
            # chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tics.items()}
            chromatograms = {key: signal for key, signal in tics.items()}
        elif DATA_TYPE == "TIS":
            chromatograms = gcms.compute_tiss()
        elif DATA_TYPE == "TIC-TIS":
            if SYNC_STATE:
                tics, _ = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
            else:
                # norm_tics = utils.normalize_dict(gcms.compute_tics(), scaler='standard')
                tics = gcms.compute_tics()
            # norm_tics, _ = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
            # tics = {key: utils.normalize_amplitude_zscore(signal) for key, signal in norm_tics.items()}
            tiss = gcms.compute_tiss()
            # chromatograms = utils.concatenate_dict_values(
            #     {key: list(value) for key, value in tics.items()},
            #     {key: list(value) for key, value in tiss.items()}
            # )
            chromatograms = {key: signal for key, signal in tics.items()}
            chromatograms_tic = {key: signal for key, signal in tics.items()}
            chromatograms_tis = {key: signal for key, signal in tiss.items()}
        else:
            raise ValueError("Invalid 'data_type' option.")
        data = chromatograms.values()
        labels = chromatograms.keys()

    if WINE_KIND == "bordeaux":
        region = 'bordeaux_chateaux'
        labels = process_labels(labels, vintage=VINTAGE)
    elif WINE_KIND == "pinot_noir":
        region = REGION
        # region = 'origin'
        if region == 'continent':
            labels = assign_continent_to_pinot_noir(labels)
        elif region == 'country':
            labels = assign_country_to_pinot_noir(labels)
        elif region == 'origin':
            labels = assign_origin_to_pinot_noir(labels)
        elif region =='winery':
            labels = assign_winery_to_pinot_noir(labels)
        elif region == 'year':
            labels = assign_year_to_pinot_noir(labels)
        elif region == 'beaume':
            labels = assign_north_south_to_beaune(labels)
        else:
            raise ValueError("Invalid region. Options are 'continent', 'country', 'origin', 'winery', or 'year'")
    elif WINE_KIND == "press":
        year_labels = extract_year_from_samples(chromatograms.keys()) if CLASS_BY_YEAR else None
        # labels = assign_category_to_press_wine(labels)
        labels = assign_composite_label_to_press_wine(labels)

    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(
    #     chromatograms, vintage, "Bordeaux", neigh_range=range(10, 11, 4),
    #     random_states=range(0, 1, 4), wk=wine_kind, region=region
    # )

    if DATA_TYPE == "GCMS": # TIS, GCMS
        # data_train, data_val, labels_train, labels_val = train_test_split(
        #     np.array(list(data)), np.array(list(labels)), test_size=0.5, random_state=42, stratify=np.array(list(labels))
        # )

        # Use all data without pre-splitting
        data_train = data
        labels_train = labels
        data_val = data
        labels_val = labels

        # # Generate shuffled indices for the first dimension (samples)
        # shuffled_indices = np.random.permutation(np.array(list(data)).shape[0])
        #
        # data_train = np.array(list(data))[shuffled_indices]
        # labels_train = np.array(labels)[shuffled_indices]
        # data_val = np.array(list(data))[shuffled_indices]
        # labels_val = np.array(labels)[shuffled_indices]
        cls = Classifier(
            np.array(list(data_val)), np.array(list(labels_val)), classifier_type='RGC', wine_kind=WINE_KIND,
            window_size=WINDOW, stride=STRIDE,
            year_labels=np.array(year_labels)
        )
        cls_type = 'RGC'
        alpha = 1

        if BAYES_OPTIMIZE:
            n_channels = data_train.shape[2]
            optimizer = BayesianParamOptimizer(data_train, list(labels_train), n_channels=n_channels)
            result = optimizer.optimize_gcms(
                n_calls=BAYES_CALLS, random_state=42, num_splits=NUM_SPLITS_BAYES, ch_treat=CH_TREAT
            )
            num_total_channels = n_channels // result.x[0]
            alpha = result.x[1]

            print(f"Optimal channel aggregation number: {result.x[0]}")
            print(f"Optimal number of channels: {num_total_channels}")
            print(f"Optimal alpha: {result.x[1]}")
            print(f"Best score: {-result.fun}")
            print("")
            print(f'sync {SYNC_STATE}')
            print(f"Estimating LOO accuracy on dataset {CHEMICAL_NAME}...")

            # data_copy = cls.data.copy()
            cls.data = utils.reduce_columns_to_final_channels(cls.data, num_total_channels)

        if CH_TREAT == 'concatenated':

            if CHANNEL_METHOD == 'all_channels':

                # Concatenate all channels
                data = data.transpose(0, 2, 1).reshape(data.shape[0], data.shape[1] * data.shape[2])

                cls = Classifier(
                    np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=WINE_KIND,
                    window_size=WINDOW, stride=STRIDE, alpha=alpha
                )

                cls.train_and_evaluate_all_channels(
                    num_repeats=NUM_SPLITS,
                    num_outer_repeats=1,
                    random_seed=42,
                    test_size=0.2, normalize=True, scaler_type='standard',
                    use_pca=False, vthresh=0.97, region=None,
                    print_results=True,
                    n_jobs=100
                )

            elif CHANNEL_METHOD == "greedy_remove":
                cls.train_and_evaluate_greedy_remove(
                # cls.train_and_evaluate_greedy_remove_batch(
                    num_repeats=50,
                    num_outer_repeats=1,
                    n_inner_repeats=50,
                    random_seed=42,
                    test_size=0.2,
                    normalize=True,
                    scaler_type='standard',
                    use_pca=False,
                    vthresh=0.97,
                    region=None,
                    print_results=True,
                    n_jobs=50,
                    feature_type=FEATURE_TYPE,
                    # batch_size = 3,
                    # selection_mode = "correlation"
                )
            elif CHANNEL_METHOD == "greedy_remove_batch":
                cls.train_and_evaluate_greedy_remove_batch(
                    num_repeats=50,
                    num_outer_repeats=1,
                    n_inner_repeats=50,
                    random_seed=42,
                    test_size=0.2,
                    normalize=True,
                    scaler_type='standard',
                    use_pca=False,
                    vthresh=0.97,
                    region=None,
                    print_results=True,
                    n_jobs=50,
                    feature_type=FEATURE_TYPE,
                    batch_size = 1,
                    selection_mode = 'correlation'  # None, correlation, snr
                )
            elif CHANNEL_METHOD == "greedy_add":
                cls.train_and_evaluate_greedy_add(
                    num_repeats=50,
                    num_outer_repeats=1,
                    n_inner_repeats=50,
                    random_seed=42,
                    test_size=0.2,
                    normalize=True,
                    scaler_type='standard',
                    use_pca=False,
                    vthresh=0.97,
                    region=None,
                    print_results=True,
                    n_jobs=50,
                    feature_type=FEATURE_TYPE
                )
            elif CHANNEL_METHOD == "greedy_ranked":
                cls.train_and_evaluate_greedy_ranked(
                    num_repeats=50,
                    num_outer_repeats=1,
                    n_inner_repeats=50,
                    random_seed=42,
                    test_size=0.2, normalize=True, scaler_type='standard',
                    use_pca=False, vthresh=0.97, region=None,
                    print_results=True,
                    n_jobs=50,
                    num_top_channels=139,
                    feature_type=FEATURE_TYPE
                )

        elif CH_TREAT == 'independent':
            classif_res = cls.train_and_evaluate_balanced_with_best_alpha2(
                n_splits=NUM_SPLITS, test_size=0.25, normalize=False, scaler_type='standard', use_pca=False,
                region=REGION, vthresh=0.97,
                best_alpha=alpha,
            )

    if DATA_TYPE == "TIC" or DATA_TYPE == "TIS":
        cls_type = 'RGC'
        alpha = 1

        dataset_origins_array = np.array([dataset_origins[key] for key in tics.keys()])

        cls = Classifier(
            np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=WINE_KIND,
            window_size=WINDOW, stride=STRIDE,
            year_labels=np.array(year_labels)
        )
        # cls.train_and_evaluate_tic(
        cls.train_and_evaluate_tic_diff_origins(
            num_repeats=NUM_SPLITS,
            random_seed=42,
            test_size=0.2, normalize=True, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None,
            print_results=True,
            n_jobs=10,
            dataset_origins = dataset_origins_array

        )
    elif DATA_TYPE == "TIC-TIS":
        cls_type = 'RGC'
        alpha = 1
        tic_data = chromatograms_tic.values()
        tis_data = chromatograms_tis.values()

        cls = Classifier(
            np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=WINE_KIND,
            window_size=WINDOW, stride=STRIDE
        )
        cls.train_and_evaluate_tic_tis(
            np.array(list(tic_data)),
            np.array(list(tis_data)),
            num_repeats=NUM_SPLITS,
            num_outer_repeats=1,
            random_seed=42,
            test_size=0.2, normalize=True, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None,
            print_results=True,
        )