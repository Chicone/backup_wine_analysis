import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os
import yaml
from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified
from umap import UMAP
from sklearn.preprocessing import normalize
from gcmswine.dimensionality_reduction import DimensionalityReducer
from gcmswine.visualizer import plot_2d, plot_3d

if __name__ == "__main__":
    # Load dataset paths from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    plot_umap = config.get("plot_umap", False)
    umap_source = config.get("umap_source", False)
    umap_dim = config.get("umap_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)

    # Parameters from config file
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]

    # Get the paths corresponding to the selected datasets
    selected_paths = [config["datasets"][name] for name in selected_datasets]

    # Check if all selected dataset contains "bordeaux"
    if not all("bordeaux" in path.lower() for path in selected_paths):
        raise ValueError("Please select a script for Bordeaux..."
        )

    # Infer wine_kind from selected dataset paths
    wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])

    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_splits = config["num_splits"]
    normalize_flag = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    region = config["region"]
    # wine_kind = config["wine_kind"]
    color_by_country = config["color_by_country"]


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

    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, None, None)

    # Instantiate classifier with data and labels
    cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=classifier, wine_kind=wine_kind,
                     year_labels=np.array(year_labels))

    # Run training and collect score vectors
    mean_acc, std_acc, scores, all_umap_labels = cls.train_and_evaluate_all_channels(
        num_repeats=num_splits,
        test_size=0.2,
        normalize=normalize_flag,
        use_pca=False,
        classifier_type=classifier,
        feature_type=feature_type,
        region=region,
        LOOPC=True,
        umap_source=umap_source
    )



    # reducer = DimensionalityReducer(scores)




    if umap_source == "scores":
        data_for_umap = normalize(scores)
        umap_labels = all_umap_labels
        umap_title = f"UMAP of Origin Classification Scores"
    elif umap_source in {"tic", "tis", "tic_tis"}:
        channels = list(range(data.shape[2]))  # use all channels
        data_for_umap = utils.compute_features(data, feature_type=umap_source)  # shape: (n_samples, n_timepoints)
        data_for_umap = normalize(data_for_umap)
        umap_labels = labels  # use raw labels from data
        umap_title = f"UMAP of {umap_source.upper()}"
    else:
        data_for_umap = None
        umap_labels = None

    if data_for_umap is not None:
        reducer = DimensionalityReducer(data_for_umap)
        if umap_dim == 2:
            plot_2d(
                reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state),
                umap_title, umap_labels, label_dict=None, group_by_country=color_by_country
            )
        elif umap_dim == 3:
            plot_3d(
                reducer.umap(components=3, n_neighbors=n_neighbors, random_state=random_state),
                umap_title, region, umap_labels, legend_labels, color_by_country
            )
    plt.show()


    #
    # # Define distinct markers and colors
    # markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '8', 'p', '+', 'x']
    # color_map = plt.cm.get_cmap("tab20", len(legend_labels))
    #
    #
    #
    #
    # # if plot_umap:
    # #     if umap_dim == 2:
    # #         plot_2d(
    # #             reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state),
    # #             f"UMAP Model Decision Scores ({region})", region, all_umap_labels, legend_labels, group_by_country
    # #         )
    # #     elif umap_dim == 3:
    # #         plot_3d(
    # #             reducer.umap(components=3, n_neighbors=n_neighbors, random_state=random_state),
    # #             f"UMAP Model Decision Scores ({region})", region, all_umap_labels, legend_labels, group_by_country
    # #         )
    #
    # # --- 2D ---
    # # plot_2d(reducer.pca(components=2), "PCA of Model Decision Scores (2D)", region, all_umap_labels,
    # #         winery_legend_labels, group_by_country)
    # # plot_2d(reducer.tsne(components=2, perplexity=5, random_state=42), "t-SNE of Model Decision Scores (2D)", region,
    # #         all_umap_labels, winery_legend_labels, group_by_country)
    # plot_2d(reducer.umap(components=2, n_neighbors=30, random_state=42), f"UMAP Model Decision Scores ({region})", region,
    #         all_umap_labels, legend_labels, group_by_country)
    #
    # # # --- 3D ---
    # # plot_3d(reducer.pca(components=3), "PCA of Model Decision Scores (3D)", region, all_umap_labels,
    # #         winery_legend_labels, group_by_country)
    # # plot_3d(reducer.tsne(components=3, perplexity=5, random_state=42), "t-SNE of Model Decision Scores (3D)", region,
    # #         all_umap_labels, winery_legend_labels, group_by_country)
    # # plot_3d(reducer.umap(components=3, n_neighbors=15, random_state=42), "UMAP of Model Decision Scores (3D)", region,
    # #         all_umap_labels, winery_legend_labels, group_by_country)
    #
    # plt.show()