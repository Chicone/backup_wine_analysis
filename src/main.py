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

import numpy as np
import argparse
import os
from data_loader import DataLoader
# from dimensionality_reduction import DimensionalityReducer
from wine_analysis import WineAnalysis, ChromatogramAnalysis
from classification import (Classifier, process_labels, assign_country_to_pinot_noir, assign_origin_to_pinot_noir,
                            assign_continent_to_pinot_noir, assign_winery_to_pinot_noir, assign_year_to_pinot_noir,
                            assign_north_south_to_beaune)
from visualizer import Visualizer, plot_all_acc_LDA, plot_stacked_chromatograms, plot_classification_accuracy
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import matplotlib.pyplot as plt
import utils
from scipy.ndimage import gaussian_filter
from wine_analysis import SyncChromatograms

if __name__ == "__main__":
    # plot_classification_accuracy()
    n_splits = 1
    chrom_cap = 25000
    vintage = False
    use_3d = False
    cnn = True
    collapse_axis = "TIC-TIS"  # TIS, TIC-TIS
    sync_state = True  # True to use retention time alignment
    pca_state = [False]   # True for classification on PCA-reduced data
    n = 5  # decimation factor for the 3D data

    # For CNN
    crop_size = (181, 181)  # Example crop size
    stride = (90, 181)     # Example stride for overlapping crops
    batch_size = 32
    num_epochs = 100
    learning_rate = 0.001


    # main_directory = "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/LLE_SCAN/"
    # chem_name = 'PINOT_NOIR_LLE_SCAN'
    main_directory = "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/DLLME_SCAN/"
    chem_name = 'PINOT_NOIR_LLE_SCAN'
    # main_directory = "/home/luiscamara/Documents/datasets/3D_data/BORDEAUX_OAK_PAPER/OAK_WOOD/"
    # chem_name = 'BORDEAUX_OAK_PAPER_OAK_WOOD'

    cl = ChromatogramAnalysis()
    if 'bordeaux' in chem_name.lower():
        wine_kind = 'bordeaux'
    elif 'pinot_noir' in chem_name.lower():
        wine_kind = 'pinot_noir'

    row_start = 1
    row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(main_directory)
    column_indices = list(range(fc_idx, lc_idx + 1))
    # column_indices = list(range(3, 10))

    data_dict = utils.load_ms_data_from_directories(main_directory, column_indices, row_start, row_end)
    min_length = min(array.shape[0] for array in data_dict.values())
    data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}

    # Function to align TIC chromatograms using MS data
    def align_tics():
        col_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=1)
        tics = col_sum_dict
        mean_c = cl.calculate_mean_chromatogram(tics)
        closest_to_mean = cl.closest_to_mean_chromatogram(tics, mean_c)
        synced_tics = cl.sync_individual_chromatograms(
            closest_to_mean[1], data_dict[closest_to_mean[0]], tics, data_dict,
            np.linspace(0.980, 1.020, 80), initial_lag=25
        )
        synced_tics = {key: value[:chrom_cap] for key, value in synced_tics.items()}
        norm_tics = utils.normalize_dict(synced_tics, scaler='standard')
        return norm_tics

    def generate_crops(data_dict, crop_size, stride):
        """
        Generate crops from each sample in the data dictionary.

        Parameters:
        -----------
        data_dict : dict
            A dictionary where keys are sample names and values are 2D numpy arrays (matrices).
        crop_size : tuple
            A tuple specifying the (height, width) of each crop.
        stride : tuple
            A tuple specifying the (vertical, horizontal) stride for the sliding window.

        Returns:
        --------
        cropped_data_dict : dict
            A dictionary where keys are sample names and values are lists of cropped matrices.
        """
        cropped_data_dict = {}

        for sample, matrix in data_dict.items():
            crops = []
            num_rows, num_cols = matrix.shape
            crop_height = min(crop_size[0], num_rows)
            crop_width = min(crop_size[1], num_cols)
            stride_vertical, stride_horizontal = stride

            for row_start in range(0, num_rows - crop_height + 1, stride_vertical):
                for col_start in range(0, num_cols - crop_width + 1, stride_horizontal):
                    crop = matrix[row_start:row_start + crop_height, col_start:col_start + crop_width]
                    crops.append(crop)

            cropped_data_dict[sample] = crops

        return cropped_data_dict

    if cnn:  # uses all MS data in crops
        data_dict = {key: array[::n, :] for key, array in data_dict.items()}
        # Create a dictionary containing crops for each sample
        cropped_data_dict = generate_crops(data_dict, crop_size, stride)
        print(f'Num. Crops = {len(cropped_data_dict)}')

        # # Decimate the retention times by a factor n while keeping the MS data intact
        # data_dict = {key: array[::n, :].flatten(order='C') for key, array in data_dict.items()}
        # data_dict = {key: value[:chrom_cap] for key, value in data_dict.items()}

    else:  # use the collapsed accumulated data
        if collapse_axis == "TIC":
            if sync_state:
                norm_tics = align_tics()
            else:
                # This is the MS sum of all ions for each retention time (TIC)
                col_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=1)
                tics = col_sum_dict
                norm_tics = utils.normalize_dict(tics, scaler='standard')
            chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in norm_tics.items()}
        elif collapse_axis == "TIS":
            # This is the m/z sum of all retention times
            row_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=0)
            chromatograms = row_sum_dict
        elif collapse_axis == "TIC-TIS":
            norm_tics = align_tics()
            # Amplitude normalize TIC
            norm_ampl_tics = {key: utils.normalize_amplitude_zscore(signal) for key, signal in norm_tics.items()}
            tics = {key: list(value) for key, value in norm_ampl_tics.items()}

            # Compute normalized TIS
            tiss = utils.sum_data_in_data_dict(data_dict, axis=0)
            norm_ampl_tiss = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tiss.items()}
            tiss = {key: list(value) for key, value in norm_ampl_tiss.items()}

            # Concatenate normalized TIC and TIS
            chromatograms = utils.concatenate_dict_values(tics, tiss)
        else:
            raise ValueError("Invalid 'collapse_axis' option. Options are 'TIC', 'TIS', and 'TIC-TIS'")


    if cnn:
        data = cropped_data_dict.values()
        labels = cropped_data_dict.keys()
    else:
        # Choose the chromatograms to analyze
        data = chromatograms.values()
        # data = synced_chromatograms.values()
        labels = chromatograms.keys()


    if wine_kind == "bordeaux":
        region = 'bordeaux_chateaux'
        labels = process_labels(labels, vintage=vintage)
    elif wine_kind == "pinot_noir":
        region = 'winery'
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

    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(
    #     chromatograms, vintage, "Bordeaux", neigh_range=range(10, 11, 4),
    #     random_states=range(0, 1, 4), wk=wine_kind, region=region
    # )

    for pca in pca_state:
        # if not sync_chroms and not pca:
        #     continue
        # for cls_type in ['LDA', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB']:
        # for cls_type in ['LDA', 'LR', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB']:
        for cls_type in ['CNN']:
            print("")
            print (f'sync {sync_state}')
            print (f'PCA {pca}')
            print(f"Estimating LOO accuracy on dataset {chem_name}...")
            cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=wine_kind)
            # cls.train_and_evaluate(
            #     n_splits, vintage=vintage, test_size=None, normalize=True, scaler_type='standard', use_pca=pca,
            #     vthresh=0.995, region=region
            # )
            # TODO add parameter to pass second set of features
            # cls.train_and_evaluate_balanced(
            #     n_splits, vintage=vintage, test_size=None, normalize=True, scaler_type='standard', use_pca=pca,
            #     vthresh=0.995, region=region, cnn=cnn
            # )
            cls.train_and_evaluate_balanced(
                n_splits, vintage=vintage, test_size=None, normalize=not cnn, scaler_type='standard', use_pca=pca,
                vthresh=0.995, region=region,
                cnn=cnn, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate
            )


    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms1, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(chromatograms1, vintage, "Pinot Noir Origins in CH", neigh_range=range(20, 101, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)  # continent, country, origin


    # for cls_type in ['LDA', 'LR', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB', 'GBC']:
