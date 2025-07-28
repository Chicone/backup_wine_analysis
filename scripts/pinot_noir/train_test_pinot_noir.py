"""
To train and test classification of Pinot Noir wines, we use the script **train_test_pinot_noir.py**.
The goal is to classify wine samples based on their GC-MS chemical fingerprint, using geographic labels
at different levels of granularity (e.g., winery, region, country, north-south of Burgundy, or continent).

The script implements a complete machine learning pipeline including data loading, preprocessing,
region-based label extraction, feature computation, and repeated classifier evaluation.

Configuration Parameters
------------------------

The script reads configuration parameters from a file (`config.yaml`) located at the root of the repository.
Below is a description of the key parameters:

- **datasets**: Dictionary mapping dataset names to local paths. Each path must contain `.D` folders for each chromatogram.

- **selected_datasets**: The list of datasets to use for the analysis. Must be compatible in terms of m/z channels.

- **feature_type**: Defines how chromatograms are converted into features for classification:

  - ``tic``: Use the Total Ion Chromatogram only.
  - ``tis``: Use individual Total Ion Spectrum channels.
  - ``tic_tis``: Concatenate TIC and TIS.
  - ``concatenated``: Flatten raw chromatograms across all channels.

- **classifier**: Classification model to apply. Available options:

  - ``DTC``: Decision Tree Classifier
  - ``GNB``: Gaussian Naive Bayes
  - ``KNN``: K-Nearest Neighbors
  - ``LDA``: Linear Discriminant Analysis
  - ``LR``: Logistic Regression
  - ``PAC``: Passive-Aggressive Classifier
  - ``PER``: Perceptron
  - ``RFC``: Random Forest Classifier
  - ``RGC``: Ridge Classifier
  - ``SGD``: Stochastic Gradient Descent
  - ``SVM``: Support Vector Machine

- **num_splits**: Number of repeated train/test splits to run.

- **normalize**: Whether to apply standard scaling before classification. Normalization is fit on training data only.

- **n_decimation**: Downsampling factor along the retention time axis to reduce dimensionality.

- **sync_state**: Whether to align chromatograms using retention time synchronization (useful for Pinot Noir samples with retention drift).

- **region**: Defines the classification target. Available options:

  - ``winery``: Classify by individual wine producer
  - ``origin``: Group samples by geographic region (e.g., Beaune, Alsace)
  - ``country``: Group by country (e.g., France, Switzerland, USA)
  - ``continent``: Group by continent
  - ``north_south_burgundy``: Binary classification of northern vs southern Burgundy subregions

- **wine_kind**: Internally inferred from dataset paths. Should not be set manually.

Script Overview
---------------

This script performs classification of **Pinot Noir wine samples** using GC-MS data and a configurable
classification pipeline. It allows for flexible region-based classification using a strategy abstraction.

The main workflow is:

1. **Configuration Loading**:

   - Loads classifier, region, feature type, and dataset settings from `config.yaml`.
   - Confirms that all dataset paths are compatible (must contain `'pinot'`).

2. **Data Loading and Preprocessing**:

   - Chromatograms are loaded and decimated.
   - Channels with zero variance are removed.
   - If `sync_state=True`, samples are aligned by retention time.

3. **Label Processing**:

   - Region-based labels are extracted using `process_labels_by_wine_kind()` and the `WineKindStrategy` abstraction.
   - Granularity is determined by the `region` parameter (e.g., `"winery"` or `"country"`).

4. **Classification**:

   - Initializes a `Classifier` instance with the chosen feature representation and classifier model.
   - Runs repeated evaluation via `train_and_evaluate_all_channels()` using the selected splitting strategy.

5. **Cross-Validation and Replicate Handling**:

   - If `LOOPC=True`, one sample is randomly selected per class along with all of its replicates, then used as the test set. This ensures that each test fold contains exactly one unique wine per class, and no sample is split across train and test. The rest of the data is used for training.
   - If `LOOPC=False`, stratified shuffling is used while still preventing replicate leakage.

6. **Evaluation**:

   - Prints average and standard deviation of balanced accuracy across splits.
   - Displays label ordering and sample distribution.
   - Set `show_confusion_matrix=True` to visualize the averaged confusion matrix with matplotlib.

Requirements
------------

- Properly structured Pinot Noir GC-MS dataset folders
- All dependencies installed (see `README.md`)
- Valid paths and regions configured in `config.yaml`

Usage
-----

From the root of the repository, run:

.. code-block:: bash

   python scripts/pinot_noir/train_test_pinot_noir.py
"""
import numpy as np
import os
import yaml
import matplotlib.pyplot as plt
from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified
from gcmswine.logger_setup import logger, logger_raw
from sklearn.preprocessing import normalize
from gcmswine.dimensionality_reduction import DimensionalityReducer
from scripts.pinot_noir.plotting_pinot_noir import plot_pinot_noir
from distinctipy import distinctipy

if __name__ == "__main__":

    # # Use this function to convert the printed confusion matrix to a latex confusion matrix
    # # Copy the matrix into data_str using """ """ and create the list of headers, then call the function
    # headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
    #             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
    #             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
    # headers = ["Beaune", "Alsace", "Neuchatel", "Genève", "Valais", "Californie", "Oregon"]
    # headers = ["France", "Switzerland", "US"]
    # string_to_latex_confusion_matrix_modified(data_str, headers)

    # Load dataset paths from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parameters from config file
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]

    # Get the paths corresponding to the selected datasets
    selected_paths = [config["datasets"][name] for name in selected_datasets]

    # Check if all selected dataset contains "pinot"
    if not all("pinot_noir" in path.lower() for path in selected_paths):
        raise ValueError("Please use this script for Pinot Noir datasets.")

    # Infer wine_kind from selected dataset paths
    wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])

    # Plot parameters
    plot_projection = config.get("plot_projection", False)
    projection_method = config.get("projection_method", "UMAP").upper()
    # projection_source = config.get("projection_source", False)
    projection_source = config.get("projection_source", False) if plot_projection else False
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)
    color_by_country = config["color_by_country"]
    show_sample_names = config["show_sample_names"]
    invert_x =  config["invert_x"]
    invert_y =  config["invert_y"]
    sample_display_mode = config["sample_display_mode"]
    show_year = True if sample_display_mode == "years" else False
    show_sample_names = True if sample_display_mode == "names" else False
    color_by_winery = config.get("color_by_winery", False)
    color_by_origin = config.get("color_by_origin", False)
    exclude_us = config.get("exclude_us", False)
    density_plot = config.get("density_plot", False)

    # Run Parameters
    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_repeats = config["num_repeats"]
    normalize_flag = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    class_by_year = config['class_by_year']
    region = config["region"]
    # Enforce exclusivity logic
    if region == "origin" and not color_by_winery:
        color_by_origin = True
    elif region == "winery" and not color_by_origin:
        color_by_winery = True
    else:
        color_by_origin = False
        color_by_winery = False

    # wine_kind = config["wine_kind"]
    show_confusion_matrix = config['show_confusion_matrix']
    retention_time_range = config['rt_range']
    cv_type = config['cv_type']
    task="classification"  # hard-coded for now
    split_burgundy_ns = True  # config.get("split_burgundy_north_south", False)
    burg_by_year = True
    summary = {
        "Task": task,
        "Wine kind": wine_kind,
        "Datasets": ", ".join(selected_datasets),
        "Feature type": config["feature_type"],
        "Classifier": config["classifier"],
        "Repeats": config["num_repeats"],
        "Normalize": config["normalize"],
        "Decimation": config["n_decimation"],
        "Sync": config["sync_state"],
        "Year Classification": config["class_by_year"],
        "Region": config["region"],
        "CV type": config["cv_type"],
        "RT range": config["rt_range"],
        "Confusion matrix": config["show_confusion_matrix"]
    }

    logger_raw("\n")# Blank line with no timestamp
    logger.info('------------------------ RUN SCRIPT -------------------------')
    logger.info("Configuration Parameters")
    for k, v in summary.items():
        logger_raw(f"{k:>20s}: {v}")

    # strategy = get_strategy_by_wine_kind(wine_kind, get_custom_order_func=utils.get_custom_order_for_pinot_noir_region())
    strategy = get_strategy_by_wine_kind(
        wine_kind=wine_kind,
        region=region,
        get_custom_order_func=utils.get_custom_order_for_pinot_noir_region,
    )

    # Create ChromatogramAnalysis instance for optional alignment
    cl = ChromatogramAnalysis(ndec=n_decimation)

    # Load dataset, removing zero-variance channels
    selected_paths = {name: dataset_directories[name] for name in selected_datasets}
    data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
    chrom_length = len(list(data_dict.values())[0])
    # print(f'Chromatogram length: {chrom_length}')

    if retention_time_range:
        min_rt = retention_time_range['min'] // n_decimation
        raw_max_rt = retention_time_range['max'] // n_decimation
        max_rt = min(raw_max_rt, chrom_length)
        logger.info(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}

    data_dict, _ = utils.remove_zero_variance_channels(data_dict)

    gcms = GCMSDataProcessor(data_dict)
    if sync_state:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
        gcms = GCMSDataProcessor(data_dict)


    def split_into_bins(data, n_bins):
        """
        Split TIC into uniform bins (segments).
        Returns a list of (start_idx, end_idx) for each bin.
        """
        total_points = data.shape[1]
        bin_edges = np.linspace(0, total_points, n_bins + 1, dtype=int)
        return [(bin_edges[i], bin_edges[i+1]) for i in range(n_bins)]


    def remove_bins(data, bins_to_remove, bin_ranges):
        """
        Remove (zero out) specified bins in TIC by index.
        """
        data_copy = data.copy()
        for b in bins_to_remove:
            start, end = bin_ranges[b]
            data_copy[:, start:end] = 0  # Mask out that bin segment
        return data_copy


    # === Extract data matrix (samples × channels) and associated labels ===
    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    raw_sample_labels = labels.copy()

    # Extract only Burgundy if region is "burgundy"
    if wine_kind == "pinot_noir" and region == "burgundy":
        burgundy_prefixes = ('D', 'P', 'R', 'Q', 'Z', 'E')
        mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
        data = data[mask]
        labels = labels[mask]
        raw_sample_labels = raw_sample_labels[mask]

    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)

    # === Define binning ===
    n_bins = 50
    min_bins = 1
    bin_ranges = split_into_bins(data, n_bins)
    active_bins = list(range(n_bins))

    survival_mode = True

    # === Iteration logic ===
    if survival_mode:
        n_iterations = n_bins - min_bins + 1
    else:
        n_iterations = 1

    # === Progressive plot setup (only for survival mode) ===
    if survival_mode:
        cv_label = "LOO" if cv_type == "LOO" else "LOOPC" if cv_type == "LOOPC" else "Stratified"
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot([], [], marker='o')
        ax.set_xlabel("Percentage of TIC Data Remaining (%)")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Greedy Survival: Accuracy vs TIC Data Remaining ({cv_label} CV)")
        ax.grid(True)
        ax.set_xlim(100, 0)  # start at 100% and decrease

    accuracies = []
    percent_remaining = []
    baseline_nonzero = np.count_nonzero(data)

    # === Main loop ===
    for step in range(n_iterations):
        logger.info(f"=== Iteration {step + 1}/{n_iterations} | Active bins: {len(active_bins)} ===")

        # --- Mask inactive bins ---
        masked_data = remove_bins(
            data,
            bins_to_remove=[b for b in range(n_bins) if b not in active_bins],
            bin_ranges=bin_ranges
        )

        # Compute % TIC data remaining
        pct_data = (np.count_nonzero(masked_data) / baseline_nonzero) * 100
        percent_remaining.append(pct_data)

        # Instantiate classifier
        cls = Classifier(
            masked_data,
            labels,
            classifier_type=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            year_labels=np.array(year_labels),
            strategy=strategy,
            sample_labels=np.array(raw_sample_labels),
            dataset_origins=dataset_origins,
        )

        # Train & evaluate
        if cv_type in ["LOOPC", "stratified"]:
            loopc = (cv_type == "LOOPC")
            mean_acc, std_acc, *_ = cls.train_and_evaluate_all_channels(
                num_repeats=num_repeats,
                random_seed=42,
                test_size=0.2,
                normalize=normalize_flag,
                scaler_type='standard',
                use_pca=False,
                vthresh=0.97,
                region=region,
                print_results=False,
                n_jobs=20,
                feature_type=feature_type,
                classifier_type=classifier,
                LOOPC=loopc,
                projection_source=projection_source,
                show_confusion_matrix=show_confusion_matrix,
            )
        elif cv_type == "LOO":
            mean_acc, std_acc, *_ = cls.train_and_evaluate_leave_one_out_all_samples(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                feature_type=feature_type,
                classifier_type=classifier,
                projection_source=projection_source,
                show_confusion_matrix=show_confusion_matrix,
            )
        else:
            raise ValueError(f"Invalid cross-validation type: '{cv_type}'.")

        accuracies.append(mean_acc)
        logger.info(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

        # === Update live plot (only survival mode) ===
        if survival_mode:
            line.set_data(percent_remaining, accuracies)
            ax.set_xlim(100, min(percent_remaining) - 5)
            ax.set_ylim(0, 1)
            plt.draw()
            plt.pause(0.2)

        # === Greedy bin removal ===
        if survival_mode and len(active_bins) > min_bins:
            candidate_accuracies = []
            for b in active_bins:
                temp_bins = [x for x in active_bins if x != b]
                temp_masked_data = remove_bins(
                    data,
                    bins_to_remove=[x for x in range(n_bins) if x not in temp_bins],
                    bin_ranges=bin_ranges
                )

                temp_cls = Classifier(
                    temp_masked_data,
                    labels,
                    classifier_type=classifier,
                    wine_kind=wine_kind,
                    class_by_year=class_by_year,
                    year_labels=np.array(year_labels),
                    strategy=strategy,
                    sample_labels=np.array(raw_sample_labels),
                    dataset_origins=dataset_origins,
                )

                temp_acc, _, *_ = temp_cls.train_and_evaluate_all_channels(
                    num_repeats=3,  # fewer repeats for speed
                    random_seed=42,
                    test_size=0.2,
                    normalize=normalize_flag,
                    scaler_type='standard',
                    use_pca=False,
                    vthresh=0.97,
                    region=region,
                    print_results=False,
                    n_jobs=10,
                    feature_type=feature_type,
                    classifier_type=classifier,
                    LOOPC=(cv_type == "LOOPC"),
                    projection_source=projection_source,
                    show_confusion_matrix=False,
                )
                candidate_accuracies.append((b, temp_acc))

            # Pick bin whose removal gives best accuracy
            best_bin, best_candidate_acc = max(candidate_accuracies, key=lambda x: x[1])
            logger.info(f"Removing bin {best_bin}: next accuracy would be {best_candidate_acc:.3f}")
            active_bins.remove(best_bin)

    # === Finalize plot ===
    if survival_mode:
        plt.ioff()
        plt.show()
    else:
        logger.info(f"Final Accuracy (no survival): {accuracies[0]:.3f}")


    if plot_projection:
        ordered_labels = [
            "D", "E", "Q", "P", "R", "Z", "C", "W", "Y", "M", "N", "J", "L", "H", "U", "X"
        ]
        if region == "winery":
            legend_labels = {
                "D": "D = Clos Des Mouches Drouhin (FR)",
                "E": "E = Vigne de l’Enfant Jésus Bouchard (FR)",
                "Q": "Q = Nuit Saint Georges - Les Cailles Bouchard (FR)",
                "P": "P = Bressandes Jadot (FR)",
                "R": "R = Les Petits Monts Jadot (FR)",
                "Z": "Z = Nuit Saint Georges - Les Boudots Drouhin (FR)",
                "C": "C = Domaine Schlumberger (FR)",
                "W": "W = Domaine Jean Sipp (FR)",
                "Y": "Y = Domaine Weinbach (FR)",
                "M": "M = Domaine Brunner (CH)",
                "N": "N = Vin des Croisés (CH)",
                "J": "J = Domaine Villard et Fils (CH)",
                "L": "L = Domaine de la République (CH)",
                "H": "H = Les Maladaires (CH)",
                "U": "U = Marimar Estate (US)",
                "X": "X = Domaine Drouhin (US)"
        }
        elif region == "origin":
            legend_labels = {
                "A": "Alsace",
                "B": "Burgundy",
                "N": "Neuchatel",
                "G": "Geneva",
                "V": "Valais",
                "C": "California",
                "O": "Oregon",
            }
        elif region == "country":
            legend_labels = {
                "F": "France",
                "S": "Switzerland",
                "U": "US"
            }
        elif region == "continent":
            legend_labels = {
                "E": "Europe",
                "N": "America"
            }
        elif region == "burgundy":
            legend_labels = {
                "N": "Côte de Nuits (north)",
                "S": "Côte de Beaune (south)"
            }
        else:
            legend_labels = utils.get_custom_order_for_pinot_noir_region(region)

        # Manage plot titles
        if projection_source == "scores":
            data_for_umap = normalize(scores)
            projection_labels = all_labels
        elif projection_source in {"tic", "tis", "tic_tis"}:
            channels = list(range(data.shape[2]))  # use all channels
            data_for_umap = utils.compute_features(data, feature_type=projection_source)
            data_for_umap = normalize(data_for_umap)
            projection_labels = labels  # use raw labels from data
        else:
            raise ValueError(f"Unknown projection source: {projection_source}")

        # Generate title dynamically
        pretty_source = {
            "scores": "Classification Scores",
            "tic": "TIC",
            "tis": "TIS",
            "tic_tis": "TIC + TIS"
        }.get(projection_source, projection_source)

        pretty_method = {
            "UMAP": "UMAP",
            "T-SNE": "t-SNE",
            "PCA": "PCA"
        }.get(projection_method, projection_method)

        pretty_region = {
            "winery": "Winery",
            "origin": "Origin",
            "country": "Country",
            "continent": "Continent",
            "burgundy": "N/S Burgundy"
        }.get(region, region)

        if region:
            plot_title = f"{pretty_method} of {pretty_source} ({pretty_region})"
        else:
            plot_title = f"{pretty_method} of {pretty_source}"

        # Disable showing sample names
        if not show_sample_names:
            test_samples_names = None


        legend_labels = {
            "D": "D = Clos Des Mouches Drouhin (FR)",
            "E": "E = Vigne de l’Enfant Jésus Bouchard (FR)",
            "Q": "Q = Nuit Saint Georges - Les Cailles Bouchard (FR)",
            "P": "P = Bressandes Jadot (FR)",
            "R": "R = Les Petits Monts Jadot (FR)",
            "Z": "Z = Nuit Saint Georges - Les Boudots Drouhin (FR)",
            "C": "C = Domaine Schlumberger (FR)",
            "W": "W = Domaine Jean Sipp (FR)",
            "Y": "Y = Domaine Weinbach (FR)",
            "M": "M = Domaine Brunner (CH)",
            "N": "N = Vin des Croisés (CH)",
            "J": "J = Domaine Villard et Fils (CH)",
            "L": "L = Domaine de la République (CH)",
            "H": "H = Les Maladaires (CH)",
            "U": "U = Marimar Estate (US)",
            "X": "X = Domaine Drouhin (US)"
        }

        if data_for_umap is not None:
            reducer = DimensionalityReducer(data_for_umap)
            if projection_method == "UMAP":
                plot_pinot_noir(
                    reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state),
                    plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names,
                    unique_samples_only=False, n_neighbors=n_neighbors, random_state=random_state,
                    invert_x=invert_x, invert_y=invert_y,
                    raw_sample_labels=raw_sample_labels, show_year=show_year,
                    color_by_origin=color_by_origin, color_by_winery=color_by_winery, highlight_burgundy_ns=True,
                    exclude_us=exclude_us, density_plot=density_plot,
                    region=region
                )
            elif projection_method == "PCA":
                plot_pinot_noir(reducer.pca(components=projection_dim),plot_title, projection_labels, legend_labels,
                                color_by_country, test_sample_names=test_samples_names)
            elif projection_method == "T-SNE":
                plot_pinot_noir(reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state),
                        plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names
                        )
            # if projection_method == "UMAP":
            #     plot_pinot_noir(
            #         reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state),
            #         plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names,
            #         unique_samples_only=False, n_neighbors=n_neighbors, random_state=random_state,
            #         invert_x=invert_x, invert_y=invert_y,
            #         only_europe=False, split_burgundy_north_south=False, raw_sample_labels=raw_sample_labels, region=region
            #     )
            # elif projection_method == "PCA":
            #     plot_pinot_noir(reducer.pca(components=projection_dim),plot_title, projection_labels, legend_labels,
            #                     color_by_country, test_sample_names=test_samples_names,
            #             unique_samples_only = False, n_neighbors = n_neighbors, random_state = random_state,
            #             invert_x = invert_x, invert_y = invert_y
            #             )
            # elif projection_method == "T-SNE":
            #     plot_pinot_noir(reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state),
            #             plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names,
            #             unique_samples_only = False, n_neighbors = n_neighbors, random_state = random_state,
            #             invert_x = invert_x, invert_y = invert_y
            #             )
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")


        plt.show()


