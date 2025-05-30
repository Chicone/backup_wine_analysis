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

if __name__ == "__main__":
    import numpy as np
    import os
    import yaml
    from gcmswine.classification import Classifier
    from gcmswine import utils
    from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
    from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
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

    # Check if all selected dataset contains "pinot"
    if not all("pinot_noir" in path.lower() for path in selected_paths):
        raise ValueError("Please use this script for Pinot Noir datasets.")

    # Infer wine_kind from selected dataset paths
    wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])

    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_splits = config["num_splits"]
    normalize = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    region = config["region"]
    # wine_kind = config["wine_kind"]
    show_confusion_matrix = config['show_confusion_matrix']

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
    data_dict, _ = utils.remove_zero_variance_channels(data_dict)
    chrom_length = len(list(data_dict.values())[0])

    gcms = GCMSDataProcessor(data_dict)

    if sync_state:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)

    # Extract data matrix (samples × channels) and associated labels
    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))

    # Extract only Burgundy if region is "burgundy"
    if wine_kind == "pinot_noir" and region == "burgundy":
        burgundy_prefixes = ('D', 'P', 'R', 'Q', 'Z', 'E')
        mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
        data = data[mask]
        labels = labels[mask]

    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, None, None)

    # Instantiate classifier with data and labels
    cls = Classifier(
        np.array(list(data)),
        np.array(list(labels)),
        classifier_type=classifier,
        wine_kind=wine_kind,
        year_labels=np.array(year_labels),
        strategy=strategy
    )

    # Train and evaluate on all channels. Parameter "feature_type" decides how to aggregate channels
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
        LOOPC=True , # whether to use stratified splitting (False) or Leave One Out Per Class (True),
        umap_source=False,
        show_confusion_matrix=show_confusion_matrix
    )

