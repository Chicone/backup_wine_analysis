import time
import os
import numpy as np
import argparse
import torch
from traitlets.config import get_config
from config import *
from data_loader import DataLoader
from classification import *
from wine_analysis import *
import utils
from plots import *


if __name__ == "__main__":
    time.sleep(DELAY)

    # ##### Utils and plots #####
    # plot_classification_accuracy()
    # plot_accuracy_vs_channels()
    # plot_channel_selection_performance_changins()
    # plot_channel_selection_performance_isvv()
    # plot_channel_selection_thresholds()
    # plot_accuracy_all_methods()
    # utils.copy_files_to_matching_directories("/home/luiscamara/kk/", 
    #    "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/MERLOT/")
    # plot_accuracy_vs_decimation('merlot')
    # plot_accuracy_vs_decimation('cabernet_sauvignon')
    # plot_press_wines_accuracies()
    # utils.rename_directories("/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters21/011222/")
    # plot_histogram_correlation(
    #     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_first_five_channels_merlot.csv",
    #     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_first_five_channels_cab_sauv.csv"
    # )
    # import matplotlib
    # matplotlib.use('TkAgg')
    # plot_histogram_correlation(
    #     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_add_ranked_cab_sauv.csv",
    #     "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_ranked_cab_sauv.csv",
    #     wine1="hist_5ch_greedy_add_ranked_cab_sauv",
    #     wine2="hist_5ch_greedy_remove_ranked_cab_sauv"
    # )
    ###########################

    cl = ChromatogramAnalysis()
    selected_paths = {name: DATASET_DIRECTORIES[name] for name in SELECTED_DATASETS}
    first_path = next(iter(selected_paths.values()), "").lower()
    wine_kind = "pinot_noir" if "pinot_noir" in first_path else "press" if "press" in first_path else "bordeaux"
    data_dict, dataset_origins = utils.join_datasets(SELECTED_DATASETS, DATASET_DIRECTORIES, N_DECIMATION)
    dataset_origins = np.array([dataset_origins[key] for key in dataset_origins.keys()])
    # print(dataset_origins)
    # time.sleep(1000)

    data_dict, valid_channels = utils.remove_zero_variance_channels(data_dict)

    # Optionally select specific or random m/z channels
    selected_channels = None # [4, 5, 9, 12, 16, 21, 24, 26, 27] # [19, 27]# [2, 19, 24, 27, 28] # [14, 21, 27]    # or set to None  [14, 16, 19, 27, 31]
    # selected_channels = [7, 18, 24, 25, 31]  # or set to None
    num_random_channels = None  # e.g. 10

    # Determine number of channels
    sample_matrix = next(iter(data_dict.values()))
    total_channels = sample_matrix.shape[1]

    if num_random_channels is not None:
        # np.random.seed(42)
        channels_to_use = np.random.choice(total_channels, num_random_channels, replace=False)
        print(f"Selected random channels: {sorted(channels_to_use)}")
    elif selected_channels is not None:
        channels_to_use = selected_channels
    else:
        channels_to_use = slice(None)

    # Filter data_dict to only keep selected channels
    data_dict = {k: v[:, channels_to_use] for k, v in data_dict.items()}

    gcms = GCMSDataProcessor(data_dict)

    chromatograms = None
    if DATA_TYPE == "GCMS":
        if GCMS_DIRECTION == "RT_DIRECTION":
            print("Analyzing GCMS data in retention time direction")
            if SYNC_STATE:
                tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
            gcms.data = utils.reduce_columns_in_dict(data_dict, NUM_AGGR_CHANNELS, normalize=False)
            chrom_length = len(list(data_dict.values())[0])
            data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
            if SYNC_STATE and CONCATENATE_TICS:
                data = np.concatenate((data, np.asarray(list(tics.values()))), axis=0)
                labels = np.concatenate((labels, np.asarray(list(tics.keys()))), axis=0)
    else:
        chromatograms_result = process_chromatograms(gcms, DATA_TYPE, SYNC_STATE, CHROM_CAP)

        if DATA_TYPE == "TIC-TIS":
            chromatograms_tic, chromatograms_tis = chromatograms_result
            data_tic = np.array(list(chromatograms_tic.values()))
            data_tis = np.array(list(chromatograms_tis.values()))
            labels = np.array(list(chromatograms_tic.keys()))

            # Concatenation of TIC and TIS along the feature axis
            data = np.hstack((data_tic, data_tis))

            # Ensure chromatograms is available for label processing
            chromatograms = chromatograms_tic  # Use TIC dictionary since keys are identical

        elif DATA_TYPE in ["TIC", "TIS"]:
            chromatograms = chromatograms_result
            data = np.array(list(chromatograms.values()))
            labels = np.array(list(chromatograms.keys()))
        else:
            raise ValueError("Invalid 'DATA_TYPE'. Must be 'TIC', 'TIS', or 'TIC-TIS'.")


    region = REGION if wine_kind == "pinot_noir" else "bordeaux_chateaux"
    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, VINTAGE, CLASS_BY_YEAR, chromatograms)

    cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type='RGC', wine_kind=wine_kind, 
                     window_size=WINDOW, stride=STRIDE, year_labels=np.array(year_labels))
    cls_type, alpha = 'RGC', 1

    if BAYES_OPTIMIZE:
        alpha, num_total_channels = optimize_bayesian_params(data, labels, NUM_SPLITS_BAYES, CH_TREAT)

    if DATA_TYPE == "GCMS":
        if CHANNEL_METHOD == "all_channels":
            cls.train_and_evaluate_all_channels(
                num_repeats=50, num_outer_repeats=1,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=50
            )

        elif CHANNEL_METHOD == "greedy_add_ranked":
            cls.train_and_evaluate_greedy_add_ranked(
                num_repeats=200, num_outer_repeats=1, n_inner_repeats=20,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=20, num_top_channels=139, feature_type=FEATURE_TYPE
            )

        elif CHANNEL_METHOD == "greedy_remove_ranked":
            cls.train_and_evaluate_greedy_remove_ranked(
                num_repeats=200, n_inner_repeats=20,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=20, feature_type=FEATURE_TYPE
            )

        elif CHANNEL_METHOD == "greedy_add":
            cls.train_and_evaluate_greedy_add(
                num_repeats=200, num_outer_repeats=1, n_inner_repeats=20,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=20, feature_type=FEATURE_TYPE
            )

        elif CHANNEL_METHOD == "greedy_remove":
            cls.train_and_evaluate_greedy_remove(
                num_repeats=200, num_outer_repeats=1, n_inner_repeats=50,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=50, feature_type=FEATURE_TYPE
            )
        elif CHANNEL_METHOD == "greedy_remove_diff_origins":
            cls.train_and_evaluate_greedy_remove_diff_origins(
                num_repeats=50, n_inner_repeats=10, random_seed=42, test_size=0.2, normalize=True,
                scaler_type='standard', use_pca=False, region=None, print_results=True, n_jobs=10,
                feature_type=FEATURE_TYPE, dataset_origins=dataset_origins, target_origin="cab_sauv_2022"
            )

        elif CHANNEL_METHOD == "greedy_remove_batch":
            cls.train_and_evaluate_greedy_remove_batch(
                num_repeats=50, num_outer_repeats=1, n_inner_repeats=50,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=50, feature_type=FEATURE_TYPE,
                batch_size=1, selection_mode='correlation'
            )

        elif CHANNEL_METHOD == "random_subset":
            cls.train_and_evaluate_random_channel_subsets(
                num_repeats=200, n_inner_repeats=20,
                random_seed=42, test_size=0.2, normalize=True, scaler_type='standard',
                use_pca=False, vthresh=0.97, region=None, print_results=True,
                n_jobs=20, feature_type=FEATURE_TYPE
            )





    # elif CH_TREAT == 'independent':
    #     cls.train_and_evaluate_balanced_with_best_alpha2(
    #         n_splits=NUM_SPLITS, test_size=0.25, normalize=False,
    #         scaler_type='standard', use_pca=False, region=REGION, vthresh=0.97,
    #         best_alpha=alpha
    #     )

    elif DATA_TYPE in "TIC":
        cls.train_and_evaluate_tic(num_repeats=NUM_SPLITS, random_seed=42, test_size=0.2, normalize=True,
                                   scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True,
                                   n_jobs=10)

    elif DATA_TYPE == "TIC-TIS":
        tic_data, tis_data = chromatograms_tic.values(), chromatograms_tis.values()
        cls.train_and_evaluate_tic_tis(np.array(list(tic_data)), np.array(list(tis_data)),
                                       num_repeats=NUM_SPLITS, num_outer_repeats=1, random_seed=42, test_size=0.2,
                                       normalize=True, scaler_type='standard', use_pca=False, vthresh=0.97,
                                       region=None, print_results=True,
                                       )

    # cl.stacked_2D_plots_3D(synced_chromatograms)
    # cl.tsne_analysis(chromatograms, vintage,"Pinot Noir", perplexity_range=range(10, len(labels1) - 1, 2),
    #                  random_states=range(0, 121, 32), wk=wine_kind, region=region)
    # cl.umap_analysis(
    #     chromatograms, vintage, "Bordeaux", neigh_range=range(10, 11, 4),
    #     random_states=range(0, 1, 4), wk=wine_kind, region=region
    # )
