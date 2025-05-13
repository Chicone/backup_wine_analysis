"""
This script evaluates the impact of selecting different numbers of random m/z channels on the classification accuracy of
GC-MS wine data.
It performs a Monte Carlo simulation where, for each channel count (from 1 to 32), multiple random subsets are selected
and evaluated using a classifier.
In parallel, it also evaluates a fixed set of 5 channels derived from a prior histogram-based greedy method.
The script produces:
  - Accuracy curves (mean and maximum) as a function of the number of channels,
  - A comparison point for the fixed 5-channel selection,
  - Histograms showing the frequency of channels appearing in the top 10% most accurate subsets.
It concludes by printing and saving the best-performing channel set for each channel count.
"""
import matplotlib.pyplot as plt
import numpy as np
from config import *
from classification import Classifier
from wine_analysis import GCMSDataProcessor, process_labels_by_wine_kind
import utils
import time
import csv
import os
from collections import Counter


if __name__ == "__main__":
    time.sleep(DELAY)

    # Join and clean GC-MS datasets
    selected_paths = {name: DATASET_DIRECTORIES[name] for name in SELECTED_DATASETS}
    first_path = next(iter(selected_paths.values()), "").lower()
    wine_kind = "pinot_noir" if "pinot_noir" in first_path else "press" if "press" in first_path else "bordeaux"
    region = REGION if wine_kind == "pinot_noir" else "bordeaux_chateaux"


    data_dict, dataset_origins = utils.join_datasets(SELECTED_DATASETS, DATASET_DIRECTORIES, N_DECIMATION)
    data_dict, valid_channels = utils.remove_zero_variance_channels(data_dict)

    # Parameters
    NUM_RANDOM_REPEATS = 500           # How many random selections to evaluate per setting
    NUM_SPLITS = 50                     # Repeats for classifier evaluation
    DATA_TYPE = "TIC-TIS"               # Choose between "TIC", "TIS", or "TIC-TIS"

    # Determine total available channels
    sample_matrix = next(iter(data_dict.values()))
    total_channels = sample_matrix.shape[1]

    results = []
    max_results = []

    # === Evaluate predefined best channels from histogram ===
    best_5_histogram_channels_merlot = [8, 11, 14, 19, 27]  # Merlot
    best_5_histogram_channels_cab_sauv = [8, 11, 14, 26, 27]  # Cab Sauv

    if "merlot" in first_path:
        current_best_5 = best_5_histogram_channels_merlot
        other_best_5 = best_5_histogram_channels_cab_sauv
    else:
        current_best_5 = best_5_histogram_channels_cab_sauv
        other_best_5 = best_5_histogram_channels_merlot

    def evaluate_channel_set(channel_list):
        filtered_dict = {k: v[:, channel_list] for k, v in data_dict.items()}
        gcms = GCMSDataProcessor(filtered_dict)
        if DATA_TYPE == "TIC-TIS":
            chromatograms_tic = gcms.compute_tics()
            chromatograms_tis = gcms.compute_tiss()
            data_tic = np.array(list(chromatograms_tic.values()))
            data_tis = np.array(list(chromatograms_tis.values()))
            data = np.hstack((data_tic, data_tis))
            labels = np.array(list(chromatograms_tic.keys()))
            chromatograms = chromatograms_tic
        elif DATA_TYPE == "TIC":
            chromatograms = gcms.compute_tics()
            data = np.array(list(chromatograms.values()))
            labels = np.array(list(chromatograms.keys()))
        elif DATA_TYPE == "TIS":
            chromatograms = gcms.compute_tiss()
            data = np.array(list(chromatograms.values()))
            labels = np.array(list(chromatograms.keys()))
        else:
            raise ValueError("Invalid DATA_TYPE")

        labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, VINTAGE, CLASS_BY_YEAR, chromatograms)
        cls = Classifier(np.array(data), np.array(labels), classifier_type='RGC', wine_kind=wine_kind, window_size=WINDOW, stride=STRIDE,
                         year_labels=np.array(year_labels))

        return cls.train_and_evaluate_tic(
            num_repeats=NUM_SPLITS, random_seed=42, test_size=0.2, normalize=True,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=False, n_jobs=10
        )


    print("\nEvaluating best 5 channels from current wine...")
    hist_acc_mean, hist_acc_std = evaluate_channel_set(current_best_5)

    print("\nEvaluating best 5 channels from other wine...")
    other_hist_acc_mean, other_hist_acc_std = evaluate_channel_set(other_best_5)

    # === Run Monte Carlo experiments ===
    all_channel_sets = []
    all_accuracies = []

    for num_random_channels in range(1, 33):
        print(f"\n### Testing {num_random_channels} random channels ###")
        temp_accuracies = []
        temp_channel_sets = []

        for run_idx in range(NUM_RANDOM_REPEATS):
            print(f"\n--- Random channel run {run_idx + 1}/{NUM_RANDOM_REPEATS} ---")
            channels_to_use = np.random.choice(total_channels, num_random_channels, replace=False)
            print(f"Selected random channels: {sorted(channels_to_use)}")

            filtered_dict = {k: v[:, channels_to_use] for k, v in data_dict.items()}
            gcms = GCMSDataProcessor(filtered_dict)

            if DATA_TYPE == "TIC-TIS":
                chromatograms_tic = gcms.compute_tics()
                chromatograms_tis = gcms.compute_tiss()
                data_tic = np.array(list(chromatograms_tic.values()))
                data_tis = np.array(list(chromatograms_tis.values()))
                data = np.hstack((data_tic, data_tis))
                labels = np.array(list(chromatograms_tic.keys()))
                chromatograms = chromatograms_tic
            elif DATA_TYPE == "TIC":
                chromatograms = gcms.compute_tics()
                data = np.array(list(chromatograms.values()))
                labels = np.array(list(chromatograms.keys()))
            elif DATA_TYPE == "TIS":
                chromatograms = gcms.compute_tiss()
                data = np.array(list(chromatograms.values()))
                labels = np.array(list(chromatograms.keys()))
            else:
                raise ValueError("Invalid DATA_TYPE. Must be 'TIC', 'TIS', or 'TIC-TIS'.")

            labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, VINTAGE, CLASS_BY_YEAR, chromatograms)
            cls = Classifier(np.array(data), np.array(labels), classifier_type='RGC', wine_kind=wine_kind,
                             window_size=WINDOW, stride=STRIDE, year_labels=np.array(year_labels))

            acc_mean, acc_std = cls.train_and_evaluate_tic(
                num_repeats=NUM_SPLITS, random_seed=42, test_size=0.2, normalize=True,
                scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=False, n_jobs=10
            )

            temp_accuracies.append(acc_mean)
            temp_channel_sets.append(channels_to_use)

        mean_acc = np.mean(temp_accuracies)
        std_acc = np.std(temp_accuracies)
        max_idx = np.argmax(temp_accuracies)
        max_acc = temp_accuracies[max_idx]
        best_channels = sorted(temp_channel_sets[max_idx])

        results.append((num_random_channels, mean_acc, std_acc))
        max_results.append((num_random_channels, max_acc, best_channels))

        all_accuracies.extend(temp_accuracies)
        all_channel_sets.extend(temp_channel_sets)

        print("\n###################################################")
        print(f"Channels: {num_random_channels} | Accuracy: {mean_acc:.3f} ± {std_acc:.3f} | Max: {max_acc:.3f}")
        print(f"Best channel set: {best_channels}")
        print("###################################################\n")

    # Plotting results
    x = [r[0] for r in results]
    y = [r[1] for r in results]
    yerr = [r[2] for r in results]
    y_max = [r[1] for r in max_results]

    plt.figure(figsize=(10, 5))
    line, = plt.plot(x, y, '-o', label='Mean Accuracy Random Channels')
    plt.fill_between(x, np.array(y) - np.array(yerr), np.array(y) + np.array(yerr),
                 color=line.get_color(), alpha=0.3, label='± 1 Std Dev')

    # plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, label='Mean Accuracy')
    plt.plot(x, y_max, 's--', color='orange', label='Max Accuracy Random Channels')
    plt.scatter(5, hist_acc_mean, color='green', marker='D', label='Best 5 Histogram Channels (this wine)', zorder=10)
    plt.scatter(5, other_hist_acc_mean, color='blue', marker='X', label='Best 5 Channels Other Wine', zorder=10)
    plt.title("Accuracy vs Number of Random Channels")
    plt.xlabel("Number of Random Channels")
    plt.ylabel("Balanced Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Analyze distribution of channels in top 10% performing subsets
    threshold = np.percentile(all_accuracies, 90)
    top_sets = [s for s, acc in zip(all_channel_sets, all_accuracies) if acc >= threshold]
    channel_counter = Counter()
    for ch_set in top_sets:
        channel_counter.update(ch_set)

    most_common = channel_counter.most_common()
    print("\nMost frequently appearing channels in top 10% subsets:")
    for ch, count in most_common:
        print(f"Channel {ch}: {count} times")

    if most_common:
        channels, counts = zip(*most_common)
        plt.figure(figsize=(12, 5))
        plt.bar(channels, counts)
        plt.xlabel("Channel index")
        plt.ylabel("Frequency in top 10% runs")
        plt.title("Channel Importance Based on Top Accuracies")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # === Analyze top 10% of 5-channel subsets ===
    five_ch_indices = [i for i, s in enumerate(all_channel_sets) if len(s) == 5]
    five_ch_accuracies = [all_accuracies[i] for i in five_ch_indices]
    five_ch_sets = [all_channel_sets[i] for i in five_ch_indices]

    if five_ch_accuracies:
        five_threshold = np.percentile(five_ch_accuracies, 90)
        top_five_sets = [s for s, acc in zip(five_ch_sets, five_ch_accuracies) if acc >= five_threshold]
        five_counter = Counter()
        for s in top_five_sets:
            five_counter.update(s)

        print("\nMost frequent channels in top 10% of 5-channel subsets:")
        for ch, count in five_counter.most_common():
            print(f"Channel {ch}: {count} times")

        if five_counter:
            chs, cts = zip(*five_counter.most_common())
            plt.figure(figsize=(10, 4))
            plt.bar(chs, cts)
            plt.title("Top 10% - 5-channel subsets: Channel Frequency")
            plt.xlabel("Channel index")
            plt.ylabel("Count")
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    else:
        print("\n⚠️ No 5-channel subsets found — double check the loop logic.")

    # Final report: Best channels for each number of channels
    print("\n\n==========================")
    print(" BEST CHANNELS BY COUNT")
    print("==========================")
    for num_channels, max_acc, best_ch in sorted(max_results, key=lambda x: x[0]):
        ch_list = ', '.join(map(str, best_ch))
        print(f"{num_channels:2d} channel(s): Accuracy = {max_acc:.3f} | Best channels: [{ch_list}]")

    # output_path = "best_channels_per_count_merlot22.csv"
    output_path = "best_channels_per_count_cab_sauv22.csv"
    with open(output_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["num_channels", "max_accuracy", "best_channels"])
        for num_channels, max_acc, best_ch in sorted(max_results, key=lambda x: x[0]):
            writer.writerow([num_channels, f"{max_acc:.4f}", "[" + ", ".join(map(str, best_ch)) + "]"])

    print(f"\n✅ Best channels saved to '{os.path.abspath(output_path)}'")


# if __name__ == "__main__":
#     time.sleep(DELAY)
#
#     # Join and clean GC-MS datasets
#     selected_paths = {name: DATASET_DIRECTORIES[name] for name in SELECTED_DATASETS}
#     first_path = next(iter(selected_paths.values()), "").lower()
#     wine_kind = "pinot_noir" if "pinot_noir" in first_path else "press" if "press" in first_path else "bordeaux"
#
#     data_dict, dataset_origins = utils.join_datasets(SELECTED_DATASETS, DATASET_DIRECTORIES, N_DECIMATION)
#     data_dict, valid_channels = utils.remove_zero_variance_channels(data_dict)
#
#     # Parameters
#     NUM_RANDOM_REPEATS = 500             # How many random selections to evaluate per setting
#     NUM_SPLITS = 50                     # Repeats for classifier evaluation
#     DATA_TYPE = "TIC-TIS"               # Choose between "TIC", "TIS", or "TIC-TIS"
#
#     # Determine total available channels
#     sample_matrix = next(iter(data_dict.values()))
#     total_channels = sample_matrix.shape[1]
#
#     results = []
#     max_results = []
#
#     # === Evaluate predefined best channels from histogram ===
#     best_5_histogram_channels = [8, 11, 14, 19, 27]  # Merlot
#     # best_5_histogram_channels = [8, 11, 14, 26, 27]  # Cab Sauv
#     print("\nEvaluating best 5 channels from histogram...")
#     filtered_dict = {k: v[:, best_5_histogram_channels] for k, v in data_dict.items()}
#     gcms = GCMSDataProcessor(filtered_dict)
#     if DATA_TYPE == "TIC-TIS":
#         chromatograms_tic = gcms.compute_tics()
#         chromatograms_tis = gcms.compute_tiss()
#         data_tic = np.array(list(chromatograms_tic.values()))
#         data_tis = np.array(list(chromatograms_tis.values()))
#         data = np.hstack((data_tic, data_tis))
#         labels = np.array(list(chromatograms_tic.keys()))
#         chromatograms = chromatograms_tic
#     elif DATA_TYPE == "TIC":
#         chromatograms = gcms.compute_tics()
#         data = np.array(list(chromatograms.values()))
#         labels = np.array(list(chromatograms.keys()))
#     elif DATA_TYPE == "TIS":
#         chromatograms = gcms.compute_tiss()
#         data = np.array(list(chromatograms.values()))
#         labels = np.array(list(chromatograms.keys()))
#     else:
#         raise ValueError("Invalid DATA_TYPE. Must be 'TIC', 'TIS', or 'TIC-TIS'.")
#
#     region = REGION if wine_kind == "pinot_noir" else "bordeaux_chateaux"
#     labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, VINTAGE, CLASS_BY_YEAR, chromatograms)
#     cls = Classifier(np.array(data), np.array(labels), classifier_type='RGC', wine_kind=wine_kind,
#                      window_size=WINDOW, stride=STRIDE, year_labels=np.array(year_labels))
#     hist_acc_mean, hist_acc_std = cls.train_and_evaluate_tic(
#         num_repeats=NUM_SPLITS, random_seed=42, test_size=0.2, normalize=True,
#         scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=False, n_jobs=10
#     )
#
#     # === Run Monte Carlo experiments ===
#     for num_random_channels in range(1, 33):
#         print(f"\n### Testing {num_random_channels} random channels ###")
#         all_accuracies = []
#         all_channel_sets = []
#
#         for run_idx in range(NUM_RANDOM_REPEATS):
#             print(f"\n--- Random channel run {run_idx + 1}/{NUM_RANDOM_REPEATS} ---")
#             channels_to_use = np.random.choice(total_channels, num_random_channels, replace=False)
#             print(f"Selected random channels: {sorted(channels_to_use)}")
#
#             filtered_dict = {k: v[:, channels_to_use] for k, v in data_dict.items()}
#             gcms = GCMSDataProcessor(filtered_dict)
#
#             if DATA_TYPE == "TIC-TIS":
#                 chromatograms_tic = gcms.compute_tics()
#                 chromatograms_tis = gcms.compute_tiss()
#                 data_tic = np.array(list(chromatograms_tic.values()))
#                 data_tis = np.array(list(chromatograms_tis.values()))
#                 data = np.hstack((data_tic, data_tis))
#                 labels = np.array(list(chromatograms_tic.keys()))
#                 chromatograms = chromatograms_tic
#             elif DATA_TYPE == "TIC":
#                 chromatograms = gcms.compute_tics()
#                 data = np.array(list(chromatograms.values()))
#                 labels = np.array(list(chromatograms.keys()))
#             elif DATA_TYPE == "TIS":
#                 chromatograms = gcms.compute_tiss()
#                 data = np.array(list(chromatograms.values()))
#                 labels = np.array(list(chromatograms.keys()))
#             else:
#                 raise ValueError("Invalid DATA_TYPE. Must be 'TIC', 'TIS', or 'TIC-TIS'.")
#
#             labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, VINTAGE, CLASS_BY_YEAR, chromatograms)
#             cls = Classifier(np.array(data), np.array(labels), classifier_type='RGC', wine_kind=wine_kind,
#                              window_size=WINDOW, stride=STRIDE, year_labels=np.array(year_labels))
#
#             acc_mean, acc_std = cls.train_and_evaluate_tic(
#                 num_repeats=NUM_SPLITS, random_seed=42, test_size=0.2, normalize=True,
#                 scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=False, n_jobs=10
#             )
#
#             all_accuracies.append(acc_mean)
#             all_channel_sets.append(channels_to_use)
#
#         mean_acc = np.mean(all_accuracies)
#         std_acc = np.std(all_accuracies)
#         max_idx = np.argmax(all_accuracies)
#         max_acc = all_accuracies[max_idx]
#         best_channels = sorted(all_channel_sets[max_idx])
#
#         results.append((num_random_channels, mean_acc, std_acc))
#         max_results.append((num_random_channels, max_acc, best_channels))
#
#         print("\n###################################################")
#         print(f"Channels: {num_random_channels} | Accuracy: {mean_acc:.3f} ± {std_acc:.3f} | Max: {max_acc:.3f}")
#         print(f"Best channel set: {best_channels}")
#         print("###################################################\n")
#
#     # Plotting results
#     x = [r[0] for r in results]
#     y = [r[1] for r in results]
#     yerr = [r[2] for r in results]
#     y_max = [r[1] for r in max_results]
#
#     plt.figure(figsize=(10, 5))
#     plt.errorbar(x, y, yerr=yerr, fmt='-o', capsize=5, label='Mean Accuracy')
#     plt.plot(x, y_max, 's--', color='orange', label='Max Accuracy')
#     plt.axhline(hist_acc_mean, color='green', linestyle='dashed', label='Best 5 Channels (histogram)')
#     plt.title("Accuracy vs Number of Random Channels")
#     plt.xlabel("Number of Random Channels")
#     plt.ylabel("Balanced Accuracy")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
#     # Final report: Best channels for each number of channels
#     print("\n\n==========================")
#     print(" BEST CHANNELS BY COUNT")
#     print("==========================")
#     for num_channels, max_acc, best_ch in sorted(max_results, key=lambda x: x[0]):
#         ch_list = ', '.join(map(str, best_ch))
#         print(f"{num_channels:2d} channel(s): Accuracy = {max_acc:.3f} | Best channels: [{ch_list}]")
#
#     # output_path = "best_channels_per_count_merlot22.csv"
#     output_path = "best_channels_per_count_cab_sauv22.csv"
#     with open(output_path, mode="w", newline="") as file:
#         writer = csv.writer(file)
#         writer.writerow(["num_channels", "max_accuracy", "best_channels"])
#         for num_channels, max_acc, best_ch in sorted(max_results, key=lambda x: x[0]):
#             writer.writerow([num_channels, f"{max_acc:.4f}", "[" + ", ".join(map(str, best_ch)) + "]"])
#
#     print(f"\n✅ Best channels saved to '{os.path.abspath(output_path)}'")