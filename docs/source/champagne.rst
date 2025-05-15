Champagne Classification
=========================

In this section, we focus on the classification of Champagne wine samples using the script **train_test_champagne.py**.
The dataset includes sensory profiles from 12 tasters across 50 wines of the same vintage, with each wine rated along
187 sensory dimensions: fruity, citrus, mature, candied, toasted, nuts, spicy, petrol, undergrowth, bakery, honey,
dairy, herbal, tobacco, texture, acid, and aging.

There is no 3D GC-MS data for this dataset, meaning we have to deal directly with TICs as individual m/z channels are not available.


Sensory Data Visualization
--------------------------

The script **champagne_projection.py** provides a flexible tool to visualize sensory data for Champagne wine samples.
It enables exploratory analysis of sensory descriptors using unsupervised methods to uncover latent patterns related to
ageing, variety, production site, or taster.

This script operates on a dataset of 50 wines, each evaluated across 187 sensory dimensions by 12 different tasters.
Unlike other datasets in the project, this one does **not include full 3D GC-MS chromatograms**. Therefore, the
analysis is performed directly on the available **Total Ion Current (TIC)** data aggregated at the sensory level.

The main steps of the script are as follows:

1. **Data Loading and Cleaning**:

   - Loads a CSV file containing sensory ratings for Champagne wines.
   - Cleans column headers, removes duplicated rows, and ensures numerical consistency.
   - Averages replicate measurements for each wine and taster.

2. **Label Handling**:

   - Allows the user to select a column (e.g., `age`, `variety`, `cave`, `taster`) to use as the target label for coloring or annotating the plots.
   - Handles missing labels and ensures one label per wine-taster pair.

3. **Standardization and Dimensionality Reduction**:

   - Standardizes all sensory features using `StandardScaler`.
   - Applies three different dimensionality reduction techniques:

     - **PCA** (Principal Component Analysis)
     - **t-SNE** (t-Distributed Stochastic Neighbor Embedding)
     - **UMAP** (Uniform Manifold Approximation and Projection)

4. **Clustering (Optional)**:

   - Optionally applies **KMeans clustering** to the PCA-reduced data.
   - Evaluates silhouette scores to automatically determine the optimal number of clusters (k).

5. **Plotting and Visualization**:

   - Generates 2D or 3D scatter plots based on the chosen dimensionality reduction method.
   - Colors points either by cluster membership (if clustering is enabled) or by known labels (e.g., ageing).
   - Optionally displays sample labels on the plot for better interpretability.

6. **Interactivity**:

   - Modifiable parameters at the top of the script allow for easy reconfiguration:

     - `label_column`: determines what label to use for coloring
     - `plot_3d`: toggles 3D vs 2D visualization
     - `do_kmeans`: enables/disables clustering
     - `show_point_labels`: controls whether labels are shown on points

This visualization script is particularly useful for:

- Identifying latent structure in sensory data
- Evaluating whether tasters are consistent
- Comparing sensory signatures across caves, ageing conditions, or grape varieties
- Exploring whether known labels correspond to natural clustering in the data

Usage
~~~~~

To run the script:

.. code-block:: bash

   python scripts/champagne/champagne_projection.py

Before running, modify the top of the script to select the appropriate label column:

.. code-block:: python

   label_column = 'ageing'  # or 'variety', 'cave', 'taster', etc.

Output
~~~~~~

The script generates plots showing how wines are distributed in reduced feature space, helping to visually assess
whether sensory profiles group according to the chosen label (e.g., ageing style). It also enables exploratory use
of clustering to identify potential new categories or sensory trends.