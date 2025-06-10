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
    import yaml
    from gcmswine.classification import Classifier
    from gcmswine import utils
    from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
    from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind, WineKindStrategy
    from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified

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

    # Check if all selected dataset contains "bordeaux"
    if not all("bordeaux" in path.lower() for path in selected_paths):
        raise ValueError("Please use this script for Bordeaux datasets.")

    # Infer wine_kind from selected dataset paths
    wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])
    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_repeats = config["num_repeats"]
    normalize = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    region = config["region"]
    # wine_kind = config["wine_kind"]
    class_by_year = config['class_by_year']
    show_confusion_matrix = config['show_confusion_matrix']
    retention_time_range = config['rt_range']

    # Load dataset, removing zero-variance channels
    cl = ChromatogramAnalysis(ndec=n_decimation)
    selected_paths = {name: dataset_directories[name] for name in selected_datasets}
    data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
    chrom_length = len(list(data_dict.values())[0])
    print(f'Chromatogram length: {chrom_length}')
    if retention_time_range:
        min_rt = retention_time_range['min'] // n_decimation
        raw_max_rt = retention_time_range['max'] // n_decimation
        max_rt = min(raw_max_rt, chrom_length)
        print(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}
    data_dict, _ = utils.remove_zero_variance_channels(data_dict)

    gcms = GCMSDataProcessor(data_dict)
    if sync_state:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
        gcms = GCMSDataProcessor(data_dict)

    # Extract data matrix (samples × channels) and associated labels
    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    labels_raw=labels
    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)

    # Strategy setup
    strategy = get_strategy_by_wine_kind(wine_kind, class_by_year=class_by_year)

    # Instantiate classifier with data and labels
    cls = Classifier(
        np.array(list(data)),
        np.array(list(labels)),
        classifier_type=classifier,
        wine_kind=wine_kind,
        year_labels=np.array(year_labels),
        strategy=strategy,
        class_by_year=class_by_year,
        labels_raw=labels_raw
    )

    # Train and evaluate on all channels. Parameter "feature_type" decides how to aggregate channels
    cls.train_and_evaluate_all_channels(
        num_repeats=num_repeats,
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
        LOOPC=True , # whether to use stratified splitting (False) or Leave One Out Per Class (True),
        projection_source=False,
        show_confusion_matrix=show_confusion_matrix
    )

