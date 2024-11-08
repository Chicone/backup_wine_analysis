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
    n_splits = 100
    chrom_cap = 25000
    vintage = False
    use_3d = False
    collapse_axis = "TIC-TIS"  # TIS, TIC-TIS
    sync_state = True  # True to use retention time alignment
    pca_state = [False]   # True for classification on PCA-reduced data
    n = 20  # decimation factor for the 3D data

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

    if use_3d:  # use the entire set of data
        data_dict = {key: value[:chrom_cap] for key, value in data_dict.items()}
        # Decimate the retention times by a factor n while keeping the MS data intact
        norm_chromatograms = {key: array[::n, :].flatten(order='C') for key, array in data_dict.items()}
    else:  # use the collapsed accumulated data
        if collapse_axis == "TIC":
            # This is the MS sum of all ions for each retention time (TIC)
            col_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=1)
            chromatograms = col_sum_dict
            if sync_state:
                mean_c = cl.calculate_mean_chromatogram(chromatograms)
                closest_to_mean = cl.closest_to_mean_chromatogram(chromatograms, mean_c)
                synced_chromatograms = cl.sync_individual_chromatograms(
                    closest_to_mean[1], data_dict[closest_to_mean[0]], chromatograms, data_dict,
                    np.linspace(0.980, 1.020, 80), initial_lag=25
                )
                synced_chromatograms = {key: value[:chrom_cap] for key, value in synced_chromatograms.items()}
                norm_chromatograms = utils.normalize_dict(synced_chromatograms, scaler='standard')
            else:
                norm_chromatograms = utils.normalize_dict(chromatograms, scaler='standard')
            norm_chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in
                                  norm_chromatograms.items()}
        elif collapse_axis == "TIS":
            # This is the m/z sum of all retention times
            row_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=0)
            chromatograms = row_sum_dict
            norm_chromatograms = chromatograms
        elif collapse_axis == "TIC-TIS":
            col_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=1)
            chromatograms = col_sum_dict
            mean_c = cl.calculate_mean_chromatogram(chromatograms)
            closest_to_mean = cl.closest_to_mean_chromatogram(chromatograms, mean_c)
            synced_chromatograms = cl.sync_individual_chromatograms(
                    closest_to_mean[1], data_dict[closest_to_mean[0]], chromatograms, data_dict,
                    np.linspace(0.980, 1.020, 80), initial_lag=25
                )
            synced_chromatograms = {key: value[:chrom_cap] for key, value in synced_chromatograms.items()}
            norm_chromatograms = utils.normalize_dict(synced_chromatograms, scaler='standard')
            norm_chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in norm_chromatograms.items()}
            norm_chromatograms = {key: list(value) for key, value in norm_chromatograms.items()}

            row_sum_dict = utils.sum_data_in_data_dict(data_dict, axis=0)
            row_sum_dict = {key: utils.normalize_amplitude_zscore(signal) for key, signal in row_sum_dict.items()}
            row_sum_dict = {key: list(value) for key, value in row_sum_dict.items()}

            norm_chromatograms = utils.concatenate_dict_values(norm_chromatograms, row_sum_dict)
        else:
            raise ValueError("Invalid 'collapse_axis' option. Options are 'TIC', 'TIS', and 'TIC-TIS'")


    # Choose the chromatograms to analyze
    data = norm_chromatograms.values()
    # data = synced_chromatograms.values()
    labels = norm_chromatograms.keys()


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
        for cls_type in ['RGC']:
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
            cls.train_and_evaluate_balanced(
                n_splits, vintage=vintage, test_size=None, normalize=True, scaler_type='standard', use_pca=pca,
                vthresh=0.995, region=region
            )


    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms1, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(chromatograms1, vintage, "Pinot Noir Origins in CH", neigh_range=range(20, 101, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)  # continent, country, origin


    # for cls_type in ['LDA', 'LR', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB', 'GBC']:
