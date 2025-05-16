"""

To train and test classification of Pinot Noir wine samples we use the script **train_test_pinot_noir.py**.


Configuration Parameters
-----------------------

The script reads analysis parameters from a configuration file (`config.yaml`) located at the root of the repository.
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

- **region**: Indicates the classification granularity, such as:

  - ``winery``: Group samples by producer.
  - ``origin``: Group samples by geographical origin or region of production.
  - ``country``: Group samples by country.
  - ``continent``: Group samples by continent.

- **wine_kind**: This parameter is used internally to distinguish the type of wine (e.g., ``pinot_noir``, ``press``, ``champagne``) and to apply appropriate label parsing and evaluation logic.
  **This field is now automatically inferred from the dataset path and should not be set manually.**

These parameters allow users to flexibly configure the pipeline without modifying the script itself.

Script Overview
---------------

The script performs classification of Pinot Noir wine samples using GC-MS data and a configurable machine learning pipeline.
It loads all key parameters and dataset paths from a separate configuration file. To modify the experiment and the
location of your dataset, simply edit ``config.yaml`` according to your needs.


The main steps include:

1. **Configuration loading**:

   - Loads experiment settings from ``config.yaml``.
   - This includes paths to the datasets, the number of evaluation splits, the classifier type, and other parameters.

2. **Data Loading and Preprocessing**:

   - GC-MS chromatograms are loaded using `GCMSDataProcessor`.
   - Datasets are joined and decimated according to the defined factor.
   - Channels with zero variance are removed.
   - Optionally, retention time alignment (synchronization) is performed if `sync_state` is enabled in the config.
   - Optionally, data normalization (recommended), using training-set statistics only to avoid leakage.

3. **Label Processing**:

   - Sample labels are extracted and grouped according to the selected `region` (e.g., winery, origin, country or continent).
   - These labels are prepared for supervised classification.

4. **Classification**:

   - The `Classifier` class is used to train a machine learning model on the processed data.
   - The `train_and_evaluate_all_channels()` method evaluates model performance across multiple splits.
   - Classification features are aggregated as specified by the `feature_type` parameter (e.g., TIC, TIS, or both).

5. **Evaluation**:

   - Accuracy results are printed.
   - Optionally, confusion matrices can be converted to LaTeX using provided helper functions for reporting.

This script provides a complete, reproducible workflow to test classification accuracy of Pinot Noir wines using chemical
profiles extracted from GC-MS data.


Requirements
------------

- Properly structured GC-MS data directories
- Required dependencies installed (see `README.md`)
- Adjust paths in `DATASET_DIRECTORIES` to match your local setup

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
    from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified

    # # Use this function to convert the printed confusion matrix to a latex confusion matrix
    # # Copy the matrix into data_str using """ """ and create the list of headers, then call the function
    # headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
    #             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
    #             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
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
        raise ValueError(
            "The datasets selected in the config.yaml file do not seem to be compatible with this script. "
            "At least one of the selected paths does not contain 'pinot_noir'."
        )

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
    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, None, None, None)

    # Instantiate classifier with data and labels
    cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=classifier, wine_kind=wine_kind,
                     year_labels=np.array(year_labels))

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
        LOOPC=True  # whether to use stratified splitting (False) or Leave One Out Per Class (True)
    )

