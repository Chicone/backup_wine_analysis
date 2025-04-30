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
    # plot_accuracy_vs_decimation('pinot_noir')
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
    # plot_accuracy_histogram_correlation(
    #     "channel_accuracy_histogram_pinot_noir_isvv_lle_RGC.csv",
    #     # "channel_accuracy_histogram_pinot_noir_changins_lle_RGC.csv",
    #     "channel_accuracy_histogram_pinot_noir_changins_LDA.csv",
    #     label1="ISVV", label2="Changins")

    # utils.create_dir_of_samples_from_champagnes("/home/luiscamara/Documents/datasets/Champagnes/DMS/champagnes_DMS.csv")
    ###########################

    cl = ChromatogramAnalysis()
    selected_paths = {name: DATASET_DIRECTORIES[name] for name in SELECTED_DATASETS}
    first_path = next(iter(selected_paths.values()), "").lower()
    if "pinot_noir" in first_path:
        wine_kind = "pinot_noir"
    elif "press" in first_path:
        wine_kind = "press"
    elif "bordeaux" in first_path:
        wine_kind = "bordeaux"
    elif "champagnes" in first_path:
        wine_kind = "champagnes"
    else:
        wine_kind = "unknown"
    data_dict, dataset_origins = utils.join_datasets(SELECTED_DATASETS, DATASET_DIRECTORIES, N_DECIMATION)
    dataset_origins = np.array([dataset_origins[key] for key in dataset_origins.keys()])
    # print(dataset_origins)
    # time.sleep(1000)

    data_dict, valid_channels = utils.remove_zero_variance_channels(data_dict)

    # Optionally select specific or random m/z channels
    selected_channels = None # [4, 5, 9, 12, 16, 21, 24, 26, 27] # [19, 27]# [2, 19, 24, 27, 28] # [14, 21, 27]    # or set to None  [14, 16, 19, 27, 31]
    # selected_channels = [7, 18, 24, 25, 31]  # or set to None
    # selected_channels = list(range(26))
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

    if SYNC_STATE:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=CHROM_CAP)
    gcms.data = utils.reduce_columns_in_dict(data_dict, NUM_AGGR_CHANNELS, normalize=False)
    chrom_length = len(list(data_dict.values())[0])
    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    if SYNC_STATE and CONCATENATE_TICS:
        data = np.concatenate((data, np.asarray(list(tics.values()))), axis=0)
        labels = np.concatenate((labels, np.asarray(list(tics.keys()))), axis=0)

    region = REGION if wine_kind == "pinot_noir" else "bordeaux_chateaux"
    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, VINTAGE, CLASS_BY_YEAR, chromatograms)

    cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type='RGC', wine_kind=wine_kind, 
                     window_size=WINDOW, stride=STRIDE, year_labels=np.array(year_labels))
    alpha = 1

    if BAYES_OPTIMIZE:
        alpha, num_total_channels = optimize_bayesian_params(data, labels, NUM_SPLITS_BAYES, CH_TREAT)

    if CHANNEL_METHOD == 'independent':
        cls.train_and_evaluate_balanced_with_best_alpha2(
            n_splits=NUM_SPLITS, test_size=0.2, normalize=NORMALIZE,
            scaler_type='standard', use_pca=False, region=REGION, vthresh=0.97,
            best_alpha=alpha
        )
    elif CHANNEL_METHOD == "individual":
        cls.train_and_evaluate_individual_channels(
            num_repeats=200, random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None, print_results=True,
            n_jobs=20, dataset=SELECTED_DATASETS, classifier_type=CLASSIFIER
        )

        def load_ms_csv_data_from_directories(directory, columns, row_start, row_end):
            """
            Reads CSV files from all .D directories in the specified directory and extracts specific columns and row ranges.
            Uses .npy caching to speed up repeated loading.

            Args:
                directory (str): The path to the main directory containing .D directories.
                columns (list of int): A list of column indices to extract from each CSV file.
                row_start (int): The starting row index to extract (inclusive).
                row_end (int): The ending row index to extract (exclusive).

            Returns:
                dict of numpy arrays: A dictionary where each key is a .D directory name (without the .D suffix),
                                      and each value is a numpy array containing the extracted data from each CSV file.
            """
            import os
            import numpy as np
            import pandas as pd

            data_dict = {}

            # Loop through all .D directories in the specified directory
            for subdir in sorted(os.listdir(directory)):
                if subdir.endswith('.D'):
                    dir_name = subdir[:-2]  # without '.D'
                    csv_file_path = os.path.join(directory, subdir, f"{dir_name}.csv")
                    cache_file_path = os.path.join(directory, subdir, f"{dir_name}_cached.npy")

                    if os.path.isfile(cache_file_path):
                        try:
                            extracted_data = np.load(cache_file_path)
                            data_dict[dir_name] = extracted_data
                            print(f"Loaded cached data from {cache_file_path}")
                        except Exception as e:
                            print(f"Error loading cache for {dir_name}: {e}")
                    elif os.path.isfile(csv_file_path):
                        try:
                            df = pd.read_csv(csv_file_path)
                            extracted_data = df.iloc[row_start:row_end, columns].to_numpy()
                            np.save(cache_file_path, extracted_data)
                            data_dict[dir_name] = extracted_data
                            print(f"Processed and cached data from {csv_file_path}")
                        except Exception as e:
                            print(f"Error processing file {csv_file_path}: {e}")
                    else:
                        print(f"No matching CSV file found in {subdir}.")

            return data_dict

    elif CHANNEL_METHOD == "all_channels":
        cls.train_and_evaluate_all_channels(
            num_repeats=NUM_SPLITS, num_outer_repeats=1,
            random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=region, print_results=True,
            n_jobs=20, feature_type=FEATURE_TYPE, classifier_type=CLASSIFIER, LOOPC=True
        )

    elif CHANNEL_METHOD == "greedy_add_ranked":
        cls.train_and_evaluate_greedy_add_ranked(
            num_repeats=200, num_outer_repeats=1, n_inner_repeats=20,
            random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None, print_results=True,
            n_jobs=20, num_top_channels=139, feature_type=FEATURE_TYPE
        )

    elif CHANNEL_METHOD == "greedy_remove_ranked":
        cls.train_and_evaluate_greedy_remove_ranked(
            num_repeats=NUM_SPLITS, n_inner_repeats=10,
            random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None, print_results=True,
            n_jobs=5, feature_type=FEATURE_TYPE, classifier_type=CLASSIFIER
        )
    elif CHANNEL_METHOD == "greedy_remove_ranked_bin_profiles":
        cls.train_and_evaluate_greedy_remove_ranked_bin_profiles(bin_size=50,
            num_repeats=NUM_SPLITS, n_inner_repeats=10,
            random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None, print_results=True,
            n_jobs=5, pool_method='mean', classifier_type=CLASSIFIER, combine_method='sum',
            num_min_profiles=1
        )

    elif CHANNEL_METHOD == "greedy_add":
        cls.train_and_evaluate_greedy_add(
            num_repeats=200, num_outer_repeats=1, n_inner_repeats=20,
            random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None, print_results=True,
            n_jobs=20, feature_type=FEATURE_TYPE
        )

    # elif CHANNEL_METHOD == "greedy_remove":
    #     cls.train_and_evaluate_greedy_remove(
    #         num_repeats=200, num_outer_repeats=1, n_inner_repeats=50,
    #         random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
    #         use_pca=False, vthresh=0.97, region=None, print_results=True,
    #         n_jobs=50, feature_type=FEATURE_TYPE
    #     )

    # keeps track of the origins (dataset) in case we want to predict only on one dataset
    elif CHANNEL_METHOD == "greedy_remove_diff_origins":
        cls.train_and_evaluate_greedy_remove_diff_origins(
            num_repeats=NUM_SPLITS, n_inner_repeats=10, random_seed=42, test_size=0.2, normalize=NORMALIZE,
            scaler_type='standard', use_pca=False, region=None, print_results=True, n_jobs=10,
            feature_type=FEATURE_TYPE, dataset_origins=dataset_origins, target_origin="pinot_noir_isvv_lle",
            classifier_type=CLASSIFIER
        )
    elif CHANNEL_METHOD == "greedy_remove_true_bin_profiles":
        cls.train_and_evaluate_greedy_remove_true_bin_profiles(bin_size=10,
              num_repeats=NUM_SPLITS, n_inner_repeats=10,
              random_seed=42, test_size=0.2, normalize=NORMALIZE,
              scaler_type='standard',
              use_pca=False, vthresh=0.97, region=None,
              print_results=True, n_jobs=5,
              pool_method='mean', combine_method='sum',
              classifier_type=CLASSIFIER)
    elif CHANNEL_METHOD == "random_subset":
        cls.train_and_evaluate_random_channel_subsets(
            num_repeats=200, n_inner_repeats=20,
            random_seed=42, test_size=0.2, normalize=NORMALIZE, scaler_type='standard',
            use_pca=False, vthresh=0.97, region=None, print_results=True,
            n_jobs=20, feature_type=FEATURE_TYPE
        )

