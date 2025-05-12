Pinot Noir Classification
=========================

This script demonstrates a full pipeline for classifying Pinot Noir wine samples using GC-MS data. It covers dataset loading, preprocessing, and classification using a selected model.

Script Overview
---------------

The script `scripts/pinot_noir/train_test_pinot_noir.py` performs the following key steps:

1. **Import Required Modules**
   Core tools from the `gcmswine` package are used to handle data processing, chromatogram alignment, and classification.

2. **Specify Dataset Paths**
   Three datasets are defined:
   - `pinot_noir_isvv_lle`
   - `pinot_noir_isvv_dllme`
   - `pinot_noir_changins`

   These datasets are expected to follow the folder structure explained in the README, with each sample in its own `.D` directory.

3. **Parameter Configuration**
   Important configuration options include:
   - `FEATURE_TYPE`: Defines how features are extracted (`'tic'`, `'tis'`, or `'tic_tis'`)
   - `CLASSIFIER`: The machine learning classifier used (e.g., `'RGC'`)
   - `NUM_SPLITS`: Number of random train/test splits
   - `N_DECIMATION`: Downsampling factor to reduce signal length
   - `REGION` and `WINE_KIND`: Used to parse and group labels

4. **Data Loading**
   The datasets are loaded and concatenated using `utils.join_datasets()` and filtered to remove channels with no variance.

5. **Optional Alignment**
   If `SYNC_STATE` is set to `True`, chromatograms are aligned using `align_tics()`. This is skipped by default in this script.

6. **Label Processing**
   Labels are parsed to extract sample metadata (e.g., winery, year) for grouping and evaluation.

7. **Classification**
   A `Classifier` object is instantiated and evaluated using:
   - Stratified train/test splits
   - Standard scaling
   - Repeated evaluation
   - Optionally applying PCA or custom feature aggregations (based on `FEATURE_TYPE`)

8. **Output**
   Classification accuracy and metrics are printed to the console. If enabled, confusion matrices can be converted to LaTeX using the utility functions.

Confusion Matrix Export
-----------------------

To convert the printed confusion matrix to LaTeX format, you can use:

.. code-block:: python

   headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', ...]
   string_to_latex_confusion_matrix_modified(data_str, headers)

Replace `data_str` with the string version of the confusion matrix printed by the script.

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
