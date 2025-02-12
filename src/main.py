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
    CHANNEL_METHOD,
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
                            assign_north_south_to_beaune, BayesianParamOptimizer, greedy_channel_selection,
                            classify_all_channels,
                            remove_highly_correlated_channels, split_by_split_greedy_channel_selection,
                            greedy_nested_cv_channel_selection, greedy_nested_cv_channel_elimination,
                            greedy_nested_cv_channel_selection_snr
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
                   plot_channel_selection_thresholds, plot_accuracy_all_methods)

if __name__ == "__main__":
    time.sleep(DELAY)
    # plot_classification_accuracy()
    # plot_accuracy_vs_channels()
    # plot_channel_selection_performance_changins()
    # plot_channel_selection_performance_isvv()
    # plot_channel_selection_thresholds("""""")
    # plot_accuracy_all_methods()

    cl = ChromatogramAnalysis()

    # Set rows in columns to read
    row_start = 1
    row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(DATA_DIRECTORY)
    column_indices = list(range(fc_idx, lc_idx + 1))

    data_dict = utils.load_ms_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end)
    min_length = min(array.shape[0] for array in data_dict.values())
    data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
    data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}
    CHROM_CAP = CHROM_CAP // N_DECIMATION
    gcms = GCMSDataProcessor(data_dict)

    # GCMS-Specific Handling
    if DATA_TYPE == "GCMS":
        if GCMS_DIRECTION == "RT_DIRECTION":
            print("Analyzing GCMS data in retention time direction")
            if SYNC_STATE:
                tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)

            gcms.data = utils.reduce_columns_in_dict(data_dict, NUM_AGGR_CHANNELS)
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
            chromatograms = {key: utils.normalize_amplitude_zscore(signal) for key, signal in tics.items()}
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

    if DATA_TYPE == "GCMS":
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
            multichannel=MULTICHANNEL,
            window_size=WINDOW, stride=STRIDE, nconv=NCONV
        )
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
                results = classify_all_channels(
                    data=data,
                    labels=labels,
                    alpha=1.0,
                    test_size=0.25,
                    num_splits=5,
                    use_pca=False,  # Enable PCA
                    n_components=50  # Retain n PCA components
                )
                # Print the results in a formatted way
                print("\n--- Classification Results ---")
                print(f"Mean Accuracy: {results['mean_accuracy']:.4f}")
                print(f"Mean Balanced Accuracy: {results['mean_balanced_accuracy']:.4f}")
                if "mean_explained_variance" in results:
                    print(f"Mean Explained Variance (PCA): {results['mean_explained_variance']:.4f}")

            elif CHANNEL_METHOD == 'greedy':
                correlation_thresholds = [1.0]
                # correlation_thresholds = np.linspace(0.5, 0.9, num=5)  # Adjust the number of steps if needed
                accuracy_progressions = {}  # Store accuracies for each threshold
                for correlation_threshold in correlation_thresholds:
                    print(f"Processing correlation_threshold = {correlation_threshold:.2f}")

                    # selected_channels, val_accuracies, test_accuracies = greedy_nested_cv_channel_selection(
                    selected_channels, val_accuracies, test_accuracies = greedy_nested_cv_channel_selection_snr(
                        # data[:, :, [10, 13, 27]],
                        data,
                        np.array(labels),
                        alpha=alpha,
                        # test_size=0.2,
                        # num_splits=NUM_SPLITS,
                        # tolerated_no_improvement_steps=10,
                        max_channels=181,
                        num_outer_repeats=100,
                        # num_outer_splits=1,
                        inner_cv_folds=15,
                        normalize=False,
                        scaler_type='standard',
                        random_seed=42,  # 42
                        parallel=True,
                        n_jobs=15,
                        method='average', # average, concatenation
                        # selection_mode='add',
                        # min_frequency=3
                        )

                    # Store the accuracies for this threshold
                    accuracy_progressions[correlation_threshold] = test_accuracies

                # # Print results
                # print("Selected Channels (in order):", selected_channels)
                # print("Accuracies at each step:", accuracies)

                # Plot the incremental performance
                import matplotlib.pyplot as plt
                # Plot the accuracy progression for each correlation threshold
                plt.figure(figsize=(12, 8))
                for correlation_threshold, test_accuracies in accuracy_progressions.items():
                    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, marker='o', linestyle='-', label=f'Threshold {correlation_threshold:.2f}')

                plt.xlabel("Number of Selected Channels")
                plt.ylabel("Balanced Test Accuracy")
                plt.title("Incremental Channel Selection Performance Across Correlation Thresholds")
                plt.legend(title="Correlation Threshold", loc="lower right")
                plt.grid()
                plt.tight_layout()
                plt.show()

            elif CHANNEL_METHOD == "greedy_remove":
                selected_channels, val_accuracies, test_accuracies = greedy_nested_cv_channel_elimination(
                        # data[:, :, [10, 13, 27]],
                        data,
                        np.array(labels),
                        alpha=alpha,
                        # test_size=0.2,
                        # num_splits=NUM_SPLITS,
                        # tolerated_no_improvement_steps=10,
                        max_channels=10,
                        num_outer_repeats=10,
                        # num_outer_splits=1,
                        inner_cv_folds=10,
                        normalize=False,
                        scaler_type='standard',
                        random_seed=42,  # 42
                        parallel=True,
                        n_jobs=10,
                        method='average', # average, concatenation
                    )



            elif CHANNEL_METHOD == "individual":
                mean_accuracies = []
                cls_data = cls.data.copy()
                num_channels = cls_data.shape[2]

                # Step 1: Compute mean accuracies for individual channels
                for ch_idx in range(num_channels):
                    print(f"Processing channel {ch_idx + 1}/{num_channels}...")

                    # Extract only the current channel
                    single_channel_data = cls_data[:, :, ch_idx].reshape(cls_data.shape[0], cls_data.shape[1], 1)

                    # Update cls.data temporarily with the single channel
                    cls.data = single_channel_data

                    # Run the training and evaluation function
                    results = cls.train_and_evaluate_all_mz_per_sample(
                        NUM_SPLITS, vintage=VINTAGE, test_size=0.2, normalize=False,
                        scaler_type='standard', use_pca=False, vthresh=0.995, region=region, best_alpha=alpha
                    )

                    # Store mean accuracy
                    mean_accuracies.append(results['mean_accuracy'])

                # Plot the mean accuracy for each channel (following your logic)
                plt.figure(figsize=(12, 6))
                plt.bar(range(1, num_channels + 1), mean_accuracies, color='skyblue', edgecolor='black')
                plt.xlabel("Channel Index")
                plt.ylabel("Mean Accuracy")
                plt.title("Mean Accuracy for Each Individual Channel")
                plt.xticks(range(0, num_channels + 1, 5), rotation=45, size=8)
                plt.tight_layout()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.show()

                # Step 2: Sort channels by mean accuracy (descending order)
                sorted_indices = np.argsort(mean_accuracies)[::-1]  # Indices of channels sorted by accuracy

                # Step 3: Incrementally concatenate top n channels and evaluate
                incremental_accuracies = []

                # for n in range(1, len(sorted_indices) + 1):
                for n in range(1, 51):
                    print(f"Concatenating top {n} channels...")

                    # Concatenate the top n channels
                    top_n_indices = sorted_indices[:n]
                    concatenated_data = np.concatenate([cls_data[:, :, idx].reshape(cls_data.shape[0], -1) for idx in top_n_indices], axis=1)

                    # Update cls.data temporarily with concatenated channels
                    cls.data = np.expand_dims(concatenated_data, axis=2)

                    # Evaluate
                    results = cls.train_and_evaluate_all_mz_per_sample(
                        NUM_SPLITS, vintage=VINTAGE, test_size=0.2, normalize=False,
                        scaler_type='standard', use_pca=False, vthresh=0.995, region=region, best_alpha=alpha
                    )

                    # Store mean balanced accuracy
                    incremental_accuracies.append(results['mean_balanced_accuracy'])

                # Plot incremental accuracies using your provided style
                plt.figure(figsize=(12, 6))
                plt.bar(range(1, len(incremental_accuracies) + 1), incremental_accuracies, color='lightcoral',
                        edgecolor='black')
                plt.xlabel("Number of Best Concatenated Channels (n)")
                plt.ylabel("Mean Balanced Accuracy")
                plt.title("Performance with Incrementally Concatenated Channels")
                plt.xticks(range(0, len(incremental_accuracies) + 1, 5), rotation=45, size=8)
                plt.tight_layout()
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.show()

        # if CH_TREAT == 'concatenated':
        #     classif_res = cls.train_and_evaluate_all_mz_per_sample(
        #         NUM_SPLITS, vintage=VINTAGE, test_size=None, normalize=False, scaler_type='standard', use_pca=False,
        #         vthresh=0.995, region=region,
        #         best_alpha=alpha
        #     )
        elif CH_TREAT == 'independent':
            classif_res = cls.train_and_evaluate_balanced_with_best_alpha2(
                n_splits=NUM_SPLITS, test_size=0.25, normalize=False, scaler_type='standard', use_pca=False,
                region=REGION, vthresh=0.97,
                best_alpha=alpha,
            )

    # elif DATA_TYPE == "TIC":
    elif DATA_TYPE in ["TIC", "TIS", "TIC-TIS"]:
        cls_type = 'RGC'
        # data_train, data_val, labels_train, labels_val = train_test_split(
        #     np.array(list(data)), np.array(list(labels)), test_size=0.3, random_state=42, stratify=np.array(list(labels))
        # )
        data_train, data_val, data_test, labels_train, labels_val, labels_test = utils.split_train_val_test(
            np.array(list(data)), np.array(list(labels)), test_size=0.2, validation_size=0.2)


        # # Generate shuffled indices for the first dimension (samples)
        # shuffled_indices = np.random.permutation(np.array(list(data)).shape[0])
        #
        # data_train = np.array(list(data))[shuffled_indices]
        # labels_train = np.array(labels)[shuffled_indices]
        # data_val = np.array(list(data))[shuffled_indices]
        # labels_val = np.array(labels)[shuffled_indices]

        alpha = 1

        if BAYES_OPTIMIZE:
            optimizer = BayesianParamOptimizer(data_train, list(labels_train), n_channels=None)
            result = optimizer.optimize_tic(n_calls=BAYES_CALLS, random_state=42, num_splits=NUM_SPLITS_BAYES)

            alpha = result.x[0]
            score = -result.fun
            print(f"Optimal alpha: {alpha}")
            print(f"Best score: {score}")

            print("")
            print (f'sync {SYNC_STATE}')
            print(f"Estimating LOO accuracy on dataset {CHEMICAL_NAME}...")

        # cls = Classifier(
        #     np.array(list(data_val)), np.array(list(labels_val)), classifier_type=cls_type, wine_kind=WINE_KIND,
        #     cnn_dim=CNN_DIM, multichannel=MULTICHANNEL,
        #     window_size=WINDOW, stride=STRIDE, nconv=NCONV,
        #     alpha=alpha,
        # )
        cls = Classifier(
            np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=WINE_KIND,
            multichannel=MULTICHANNEL,
            window_size=WINDOW, stride=STRIDE, nconv=NCONV,
            alpha=alpha,
        )

        # classif_res = cls.train_and_evaluate_balanced(
        #     n_splits=NUM_SPLITS, vintage=False, random_seed=42, test_size=0.2, normalize=True,
        #     scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
        #     batch_size=32, num_epochs=10, learning_rate=0.001
        # )

        classif_res = cls.train_and_evaluate_balanced(
            num_outer_repeats=10,  # Repeat the outer stratified split 5 times.
            n_inner_repeats=50,  # Use 20 inner CV repeats per outer repetition.
            random_seed=42,
            test_size=0.2,
            normalize=True,  # Enable normalization.
            scaler_type='standard',
            use_pca=False,  # Apply PCA.
            vthresh=0.95,  # Variance threshold for PCA.
            region='winery'  # Set region to 'winery' for a custom confusion matrix order.
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
