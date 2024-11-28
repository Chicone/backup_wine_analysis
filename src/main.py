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
    ROW_START,
    N_SPLITS,
    SYNC_STATE,
    CHROM_CAP,
    N_DECIMATION,
    VINTAGE,
    DATA_TYPE,
    CNN_DIM,
    GCMS_DIRECTION,
    NUM_AGGR_CHANNELS,
    DELAY,
    CONCATENATE_TICS,
    PCA_STATE,
    WINDOW,
    STRIDE,
    CROP_SIZE,
    CROP_STRIDE,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    NCONV,
    MULTICHANNEL,
    WINE_KIND,
    REGION
)
import os
import numpy as np
import argparse
from data_loader import DataLoader
from classification import (Classifier, process_labels, assign_country_to_pinot_noir, assign_origin_to_pinot_noir,
                            assign_continent_to_pinot_noir, assign_winery_to_pinot_noir, assign_year_to_pinot_noir,
                            assign_north_south_to_beaune)
from wine_analysis import WineAnalysis, ChromatogramAnalysis, GCMSDataProcessor
from classification import Classifier, assign_winery_to_pinot_noir
from visualizer import visualize_confusion_matrix_3d, plot_accuracy_vs_channels
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import utils
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F

if __name__ == "__main__":
    time.sleep(DELAY)
    # plot_classification_accuracy()
    # plot_accuracy_vs_channels()
    # input_dir = "/home/luiscamara/Documents/datasets/3D_data/220322_Pinot_Noir_Tom_CDF"
    # utils.convert_cdf_directory_to_csv(input_dir, mz_min=40, mz_max=220)

    cl = ChromatogramAnalysis()

    # Set rows in columns to read
    row_start = 1
    row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(DATA_DIRECTORY)
    column_indices = list(range(fc_idx, lc_idx + 1))
    # column_indices = list(range(3, 10))

    data_dict = utils.load_ms_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end)
    min_length = min(array.shape[0] for array in data_dict.values())
    data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
    data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}

    gcms = GCMSDataProcessor(data_dict)

    # GCMS-Specific Handling
    if DATA_TYPE == "GCMS":
        if CNN_DIM:
            if CNN_DIM == 1:
                print("Analizing GCMS using 1d CNNs")
                if SYNC_STATE:
                    norm_tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
                data_dict = utils.normalize_mz_profiles_amplitude(data_dict, method='min-max')
                chromatograms = {sample_name: np.hsplit(matrix, matrix.shape[1]) for sample_name, matrix in data_dict.items()}
                data = chromatograms.values()
                labels = chromatograms.keys()
            elif CNN_DIM == 2:
                print("Analizing GCMS using 2d CNNs")
                data_dict = gcms.resample_gcms_pytorch(original_size=181, new_size=500)
                aggregated_dict = {
                    sample_name: utils.aggregate_retention_times(image, N_DECIMATION, method='max')
                    for sample_name, image in data_dict.items()
                }
                gcms.data = aggregated_dict
                cropped_data_dict = gcms.generate_crops(crop_size=CROP_SIZE, stride=CROP_STRIDE)
                print(f'Num. Crops = {len(cropped_data_dict[list(cropped_data_dict)[0]])}')
                data = cropped_data_dict.values()
                labels = cropped_data_dict.keys()
        else:
            if GCMS_DIRECTION == "RT_DIRECTION":
                print("Analyzing GCMS data in retention time direction")
                if SYNC_STATE:
                    tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)

                gcms.data = utils.reduce_columns_in_dict(gcms.data, NUM_AGGR_CHANNELS)

                # gcms.data = utils.normalize_dict_multichannel(gcms.data, scaler='standard')

                chrom_length = len(list(data_dict.values())[0])
                data, labels = gcms.create_overlapping_windows(chrom_length, chrom_length)  # with chrom_length is one window only
                labels = np.repeat(labels, data.shape[2])
                data = data.transpose(2, 0, 1).reshape(-1, data.shape[1])
                if SYNC_STATE and CONCATENATE_TICS:
                    data = np.concatenate((data, np.asarray(list(tics.values()))), axis=0)
                    labels = np.concatenate((labels, np.asarray(list(tics.keys()))), axis=0)
            # elif GCMS_DIRECTION == "MS_DIRECTION":
            #     print("Analyzing GCMS data in m/z direction")
            #     chrom_length = len(list(data_dict.values())[0])
            #     data, labels = gcms.create_overlapping_windows(chrom_length, chrom_length)
            #     labels = np.repeat(labels, data.shape[1])
            #     data = data.reshape(-1, data.shape[2])

    else:  # Handling Other Data Types
        if DATA_TYPE == "TIC":
            if SYNC_STATE:
                tics, _ = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
            else:
                # norm_tics = utils.normalize_dict(gcms.compute_tics(), scaler='standard')
                tics = gcms.compute_tics()
            chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tics.items()}
        elif DATA_TYPE == "TIS":
            chromatograms = gcms.compute_tiss()
        elif DATA_TYPE == "TIC-TIS":
            norm_tics, _ = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
            tics = {key: utils.normalize_amplitude_zscore(signal) for key, signal in norm_tics.items()}
            tiss = gcms.compute_tiss()
            chromatograms = utils.concatenate_dict_values(
                {key: list(value) for key, value in tics.items()},
                {key: list(value) for key, value in tiss.items()}
            )
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

    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(
    #     chromatograms, vintage, "Bordeaux", neigh_range=range(10, 11, 4),
    #     random_states=range(0, 1, 4), wk=wine_kind, region=region
    # )

    for pca in PCA_STATE:
        # if not sync_chroms and not pca:
        #     continue
        # for cls_type in ['LDA', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB']:
        # for cls_type in ['LDA', 'LR', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB']:
        for cls_type in ['RGC']:  # 'CNN', 'CNN1D'
        # for cls_type in ['LDA', 'RFC', 'PAC', 'PER', 'RGC', 'SGD']:
            # if DATA_TYPE == 'GCMS':
            #     cls_type = 'CNN1D'
            print("")
            print (f'sync {SYNC_STATE}')
            print (f'PCA {pca}')
            print(f"Estimating LOO accuracy on dataset {CHEMICAL_NAME}...")
            cls = Classifier(
                np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=WINE_KIND,
                cnn_dim=CNN_DIM, multichannel=MULTICHANNEL,
                window_size=WINDOW, stride=STRIDE, nconv=NCONV
            )
            classif_res = cls.train_and_evaluate_balanced_with_alpha(
                N_SPLITS, vintage=VINTAGE, test_size=None, normalize=not CNN_DIM, scaler_type='standard', use_pca=pca,
                vthresh=0.995, region=region,
                batch_size=BATCH_SIZE, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE,
            )
            if region == 'winery':
                class_labels = [
                    'Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot',
                    'Les Petits Monts', 'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach',
                    'Brunner', 'Vin des Croisés', 'Villard et Fils', 'République',
                    'Maladaires', 'Marimar', 'Drouhin'
                ]
            elif region == 'origin':
                class_labels = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
            # visualize_confusion_matrix_3d(classif_res['mean_confusion_matrix'], class_labels=class_labels,
            #                               title="MDS on wineries for ISVV LLE")


    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms1, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(chromatograms1, vintage, "Pinot Noir Origins in CH", neigh_range=range(20, 101, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)  # continent, country, origin


    # for cls_type in ['LDA', 'LR', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB', 'GBC']:
