"""
To train and test classification of Bordeaux wines, we use the script **train_test_bordeaux.py**.
The goal is to classify Bordeaux wine samples based on their GC-MS chemical fingerprint, using either
sample-level identifiers (e.g., A2022) or vintage year labels (e.g., 2022) depending on the configuration.

The script implements a complete machine learning pipeline including data loading, label parsing,
feature extraction, classification, and repeated evaluation using replicate-safe splitting.

Configuration Parameters
------------------------

The script reads configuration parameters from a file (`config.yaml`) located at the root of the repository.
Below is a description of the key parameters:

- **datasets**: A dictionary mapping dataset names to paths on your local machine. Each path should contain `.D` folders for raw GC-MS samples.

- **selected_datasets**: The list of datasets to include. All selected datasets must be compatible in terms of m/z channels.

- **feature_type**: Determines how chromatographic data are aggregated for classification.

  - ``tic``: Use the Total Ion Chromatogram only.
  - ``tis``: Use individual Total Ion Spectrum channels.
  - ``tic_tis``: Concatenates TIC and TIS into a joint feature vector.

- **classifier**: The classification algorithm to use. Options include:

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

- **num_splits**: Number of repetitions for train/test evaluation. Higher values yield more robust statistics.

- **normalize**: Whether to apply standard scaling to features. Scaling is fitted on the training set and applied to test.

- **n_decimation**: Downsampling factor for chromatograms along the retention time axis.

- **sync_state**: Enables retention time alignment between samples (typically not needed for Bordeaux).

- **region**: Not used in Bordeaux classification, but required for other pipelines such as Pinot Noir.

- **class_by_year**: If `True`, samples are classified by vintage year (e.g., 2020, 2021). If `False`, samples are classified by composite label (e.g., A2022).

- **wine_kind**: Internally inferred from the dataset path (should include `bordeaux`). Should not be set manually.

Script Overview
---------------

This script performs classification of **Bordeaux wine samples** using GC-MS data and a configurable machine learning pipeline.

All parameters are loaded from a central `config.yaml` file, enabling reproducibility and flexibility.

The main steps include:

1. **Configuration Loading**:

   - Loads paths, classifier settings, and feature types from the config file.
   - Verifies that all selected datasets are Bordeaux-type (i.e., paths contain `'bordeaux'`).

2. **Data Loading and Preprocessing**:

   - Loads and optionally decimates GC-MS chromatograms using `GCMSDataProcessor`.
   - Removes channels with zero variance.
   - Optional retention time synchronization can be enabled with `sync_state=True`.

3. **Label Processing**:

   - Labels are parsed based on `class_by_year`:
     - If `True`, classification is done by year (e.g., 2021).
     - If `False`, composite labels like `A2022` are used.
   - Label extraction and grouping are managed by the `WineKindStrategy` abstraction layer.

4. **Classification**:

   - A `Classifier` object is initialized with the processed data and selected classifier.
   - The `train_and_evaluate_all_channels()` method runs repeated evaluations across all channels or selected feature types.

5. **Cross-Validation and Replicate Handling**:

   - If `LOOPC=True`, one sample is randomly selected per class along with all of its replicates, then used as the test set. This ensures that each test fold contains exactly one unique wine per class, and no sample is split across train and test. The rest of the data is used for training.
   - If `LOOPC=False`, stratified shuffling is used, still preserving replicate integrity using group logic.

6. **Evaluation**:

   - Prints mean and standard deviation of balanced accuracy.
   - Displays label counts and ordering used for confusion matrix construction.
   - Set `show_confusion_matrix=True` to visualize the averaged confusion matrix with matplotlib.

Requirements
------------

- Properly structured GC-MS dataset folders
- All required Python dependencies installed (see `README.md`)
- Dataset paths correctly specified in `config.yaml`

Usage
-----

From the root of the repository, run:

.. code-block:: bash

   python scripts/bordeaux/train_test_bordeaux.py
"""


if __name__ == "__main__":
    import numpy as np
    import os
    import sys
    import yaml
    import matplotlib.pyplot as plt
    from gcmswine.classification import Classifier
    from gcmswine import utils
    from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
    from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
    from gcmswine.logger_setup import logger, logger_raw
    from sklearn.preprocessing import normalize
    from gcmswine.dimensionality_reduction import DimensionalityReducer
    from scripts.bordeaux.plotting_bordeaux import plot_bordeaux

    # from gcmswine.utils import create_dir_of_samples_from_bordeaux_oak, create_dir_of_samples_from_bordeaux_ester
    # create_dir_of_samples_from_bordeaux_ester(
    # # create_dir_of_samples_from_bordeaux_oak(
    #     "/home/luiscamara/Documents/datasets/BordeauxData/older data/Datat for paper Sept 2022/2018 7 chateaux Ester Old vintages Masse 5.csv",
    #     output_root="/home/luiscamara/Documents/datasets/BordeauxData/ester_paper")

    # === Utility Functions ===
    def split_into_bins(data, n_bins):
        total_points = data.shape[1]
        bin_edges = np.linspace(0, total_points, n_bins + 1, dtype=int)
        return [(bin_edges[i], bin_edges[i+1]) for i in range(n_bins)]

    def remove_bins(data, bins_to_remove, bin_ranges):
        """Remove (delete) specified bins in TIC by index."""
        mask = np.ones(data.shape[1], dtype=bool)
        for b in bins_to_remove:
            start, end = bin_ranges[b]
            mask[start:end] = False
        return data[:, mask]

    # === Load Config ===
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parameters from config file
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]
    selected_paths = [dataset_directories[name] for name in selected_datasets]

    if not all("bordeaux" in path.lower() for path in selected_paths):
        raise ValueError("Please use this script for Bordeaux datasets.")

    wine_kind = utils.infer_wine_kind(selected_datasets, dataset_directories)
    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_repeats = config["num_repeats"]
    normalize_flag = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    class_by_year = config["class_by_year"]
    region = config["region"]
    show_confusion_matrix = config["show_confusion_matrix"]
    retention_time_range = config["rt_range"]
    cv_type = config["cv_type"]

    # Survival parameters
    n_bins = 50
    min_bins = 1
    survival_mode = True

    # Projection plotting parameters
    plot_projection = config.get("plot_projection", False)
    projection_method = config.get("projection_method", "UMAP").upper()
    projection_source = config.get("projection_source", False) if plot_projection else False
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)
    color_by_country = config.get("color_by_country", False)
    show_sample_names = config.get("show_sample_names", False)
    invert_x = config.get("invert_x", False)
    invert_y = config.get("invert_y", False)

    # === Load and preprocess data ===
    cl = ChromatogramAnalysis(ndec=n_decimation)
    data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
    chrom_length = len(list(data_dict.values())[0])

    if retention_time_range:
        min_rt = retention_time_range['min'] // n_decimation
        raw_max_rt = retention_time_range['max'] // n_decimation
        max_rt = min(raw_max_rt, chrom_length)
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}

    data_dict, _ = utils.remove_zero_variance_channels(data_dict)
    gcms = GCMSDataProcessor(data_dict)
    if sync_state:
        _, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
        gcms = GCMSDataProcessor(data_dict)

    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    raw_sample_labels = labels.copy()
    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)
    strategy = get_strategy_by_wine_kind(wine_kind, class_by_year=class_by_year)

    # === Define binning ===
    bin_ranges = split_into_bins(data, n_bins)
    active_bins = list(range(n_bins))
    n_iterations = n_bins - min_bins + 1 if survival_mode else 1

    # Plot setup for survival mode
    if survival_mode:
        cv_label = "LOO" if cv_type == "LOO" else "LOOPC" if cv_type == "LOOPC" else "Stratified"
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 5))
        line, = ax.plot([], [], marker='o')
        ax.set_xlabel("Percentage of TIC Data Remaining (%)")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Greedy Survival: Accuracy vs TIC Data Remaining ({cv_label} CV)")
        ax.grid(True)
        ax.set_xlim(100, 0)

    accuracies = []
    percent_remaining = []
    baseline_nonzero = np.count_nonzero(data)

    # === Main loop ===
    for step in range(n_iterations):
        logger.info(f"=== Iteration {step+1}/{n_iterations} | Active bins: {len(active_bins)} ===")

        masked_data = remove_bins(
            data,
            bins_to_remove=[b for b in range(n_bins) if b not in active_bins],
            bin_ranges=bin_ranges
        )

        pct_data = (np.count_nonzero(masked_data) / baseline_nonzero) * 100
        percent_remaining.append(pct_data)

        cls = Classifier(
            masked_data,
            labels,
            classifier_type=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            year_labels=np.array(year_labels),
            strategy=strategy,
            sample_labels=raw_sample_labels,
            dataset_origins=dataset_origins,
        )

        if cv_type in ["LOOPC", "stratified"]:
            loopc = (cv_type == "LOOPC")
            mean_acc, std_acc, scores, all_labels, test_samples_names = cls.train_and_evaluate_all_channels(
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
                projection_source=False,
                show_confusion_matrix=False,
            )
        elif cv_type == "LOO":
            mean_acc, std_acc, scores, all_labels, test_samples_names = cls.train_and_evaluate_leave_one_out_all_samples(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                feature_type=feature_type,
                classifier_type=classifier,
                projection_source=False,
                show_confusion_matrix=False,
            )
        else:
            raise ValueError(f"Invalid CV type: {cv_type}")

        accuracies.append(mean_acc)
        logger.info(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

        if survival_mode:
            line.set_data(percent_remaining, accuracies)
            ax.set_xlim(100, min(percent_remaining) - 5)
            ax.set_ylim(0, 1)
            plt.draw()
            plt.pause(0.2)

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
                    sample_labels=raw_sample_labels,
                    dataset_origins=dataset_origins,
                )
                temp_acc, _, *_ = temp_cls.train_and_evaluate_all_channels(
                    num_repeats=3,
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
                    projection_source=False,
                    show_confusion_matrix=False,
                )
                candidate_accuracies.append((b, temp_acc))

            best_bin, best_candidate_acc = max(candidate_accuracies, key=lambda x: x[1])
            logger.info(f"Removing bin {best_bin}: next accuracy would be {best_candidate_acc:.3f}")
            active_bins.remove(best_bin)

    if survival_mode:
        plt.ioff()
        plt.show()

    # === Projection Plotting ===
    if plot_projection:
        if projection_source == "scores":
            data_for_projection = normalize(scores)
            projection_labels = all_labels
        elif projection_source in {"tic", "tis", "tic_tis"}:
            data_for_projection = utils.compute_features(data, feature_type=projection_source)
            data_for_projection = normalize(data_for_projection)
            projection_labels = year_labels if year_labels is not None else labels
        else:
            raise ValueError(f"Unknown projection source: {projection_source}")

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

        plot_title = f"{pretty_method} of {pretty_source}"
        sys.stdout.flush()


        if data_for_projection is not None:
            reducer = DimensionalityReducer(data_for_projection)
            if projection_method == "UMAP":
                plot_bordeaux(
                    reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state),
                    plot_title, projection_labels, labels, color_by_country, invert_x=invert_x, invert_y=invert_y
                )
            elif projection_method == "PCA":
                plot_bordeaux(
                    reducer.pca(components=projection_dim),
                    plot_title, projection_labels, labels, color_by_country, invert_x=invert_x, invert_y=invert_y
                )
            elif projection_method == "T-SNE":
                plot_bordeaux(
                    reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state),
                        plot_title, projection_labels, labels, color_by_country, invert_x=invert_x, invert_y=invert_y
                        )
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")
        plt.show()
