"""

To train and test classification of press wines, we use the script **train_test_press_wines.py**. The
goal is to predict wine press class membership (e.g., A, B, or C) based on GC-MS data collected from Merlot and
Cabernet Sauvignon wines across multiple vintages. The script implements a full processing pipeline including
data loading, preprocessing, feature extraction, and classification.


Configuration Parameters
------------------------

The script reads configuration parameters from a file (`config.yaml`) located at the root of the repository.
Below is a description of the key parameters:

- **dataset**: Each dataset must be specified with a name and its corresponding path on your local machine. The paths should point to directories containing `.D` folders for each sample.

- **selected_datasets**: Selects the datasets to be used. You can join more than one but must be compatible in terms of m/z channels

- **feature_type**: Determines how the chromatogram channels are aggregated for classification.

  - ``tic``: Use the Total Ion Chromatogram only.
  - ``tis``: Use individual Total Ion Spectrum channels.
  - ``tic_tis``: Combines TIC and TIS features by concatenation.

- **classifier**: Specifies the classification model used for training. Available options include:

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

- **num_splits**: Number of random train/test splits to evaluate model stability. Higher values improve statistical confidence.

- **normalize**: Whether to apply feature scaling (standard normalization) before classification. It is learned on training splits and applied to test split, so no leakage

- **n_decimation**: Factor by which chromatograms are downsampled along the retention time axis to reduce dimensionality.

- **sync_state**: Enables or disables retention time synchronization between samples using peak alignment algorithms.

- **region**: This parameter defines the classification granularity for **Pinot Noir** datasets, where samples can be grouped by winery, origin, country, or continent.

  This option is **not applicable** to press wine classification and can be ignored when using `train_test_press_wines.py`.

- **wine_kind**: This parameter is used internally to distinguish the type of wine (e.g., ``pinot_noir``, ``press``, ``champagne``) and to apply appropriate label parsing and evaluation logic.
  **This field is now automatically inferred from the dataset path and should not be set manually.**

These parameters allow users to flexibly configure the pipeline without modifying the script itself.

Script Overview
---------------

This script performs classification of **press wine samples** using GC-MS data and a configurable machine learning pipeline.

All key parameters (dataset paths, classifier settings, preprocessing options, etc.) are loaded from a `config.yaml` file,
allowing users to customize experiments without editing the script.

The main steps include:

1. **Configuration Loading**:

   - Loads parameters from `config.yaml`, including dataset paths, number of evaluation splits, classifier type, and feature extraction settings.
   - Automatically infers `wine_kind` based on the dataset path (e.g., `press_wines`), eliminating the need to set it manually.
   - Verifies that all selected dataset paths are compatible with the script (must contain `'press_wines'` in the path).

2. **Data Loading and Preprocessing**:

   - GC-MS chromatograms are loaded using `GCMSDataProcessor`.
   - Datasets are joined and downsampled using the `n_decimation` factor.
   - Channels with zero variance are automatically removed.
   - If `sync_state` is enabled, chromatograms are aligned using peak-based retention time synchronization (although for press wines this is typically discouraged).

3. **Label Processing**:

   - Sample labels are parsed using `process_labels_by_wine_kind()`, which groups samples by predefined categories (A, B, C) based on sample naming conventions.
   - Label parsing and split logic are dynamically handled via a strategy abstraction (WineKindStrategy), which adapts behavior for press, Bordeaux, or Pinot Noir samples.

4. **Classification**:

   - A `Classifier` object is initialized using the processed data and selected classifier type (e.g., `RGC`, `SVM`, etc.).
   - The `train_and_evaluate_all_channels()` method extracts features (e.g., TIC, TIS, or both) and evaluates model performance across multiple splits.

5. **Cross-Validation and Replicate Handling**:

   - The script uses repeated train/test splits (default: 20% test) across `num_splits` repetitions.
   - When `LOOPC=True`, one sample is randomly selected per class along with all of its replicates, then used as the test set. This ensures that each test fold contains exactly one unique wine per class, and no sample is split across train and test.  The rest of the data is used for training.
   - When `LOOPC=False`, a **stratified shuffle split** is used with a test fraction of 0.2 (i.e., 80/20 split). Even in this case, **replicates of the same sample are always grouped**, ensuring that no replicate of a given sample appears in both training and test sets.

6. **Evaluation**:

   - The script prints the mean and standard deviation of the balanced accuracy across all splits.
   - A normalized confusion matrix is computed and printed.
   - Set `show_confusion_matrix=True` to display the averaged confusion matrix using matplotlib.

This script provides a robust, reproducible workflow to evaluate the classification accuracy of press wine samples
based on their chemical fingerprints, while properly accounting for technical replicates and dataset structure.

Requirements
------------

- Properly structured GC-MS data directories
- Required dependencies installed (see `README.md`)
- Adjust paths in `DATASET_DIRECTORIES` to match your local setup

Usage
-----

From the root of the repository, run:

.. code-block:: bash

   python scripts/press_wines/train_test_press_wines.py
   """
if __name__ == "__main__":
    import numpy as np
    import os
    import yaml
    from gcmswine.classification import Classifier
    from gcmswine import utils
    from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
    from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified
    from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind


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
    if not all("press_wines" in path.lower() for path in selected_paths):
        raise ValueError("This script is for press wines. Some selected paths do not match.")

    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_splits = config["num_splits"]
    normalize = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    region = config["region"]
    # wine_kind = config["wine_kind"]
    show_confusion_matrix = config['show_confusion_matrix']
    class_by_year = config['class_by_year']
    # Infer wine_kind from selected dataset paths
    wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])
    strategy = get_strategy_by_wine_kind(wine_kind)

    cl = ChromatogramAnalysis(ndec=n_decimation)
    selected_paths = {name: dataset_directories[name] for name in selected_datasets}
    data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
    data_dict, _ = utils.remove_zero_variance_channels(data_dict)
    chrom_length = len(list(data_dict.values())[0])
    gcms = GCMSDataProcessor(data_dict)

    if sync_state:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)

    data = np.array(list(gcms.data.values()))
    labels_raw = np.array(list(gcms.data.keys()))
    labels, year_labels = process_labels_by_wine_kind(labels_raw, wine_kind, region, class_by_year, data_dict)

    cls = Classifier(
        data=data,
        labels=labels,
        classifier_type=classifier,
        wine_kind=wine_kind,
        year_labels=year_labels,
        dataset_origins=np.array(dataset_origins),
        strategy=strategy,
        class_by_year=class_by_year,
        labels_raw=labels_raw
    )

    cls.train_and_evaluate_all_channels(
        num_repeats=num_splits,
        random_seed=42,
        test_size=0.2,
        normalize=normalize,
        scaler_type='standard',
        use_pca=False,
        vthresh=0.97,
        region=region,
        print_results=True,
        n_jobs=20,
        feature_type=feature_type,
        classifier_type=classifier,
        LOOPC=True,
        umap_source=False,
        show_confusion_matrix=show_confusion_matrix
    )
