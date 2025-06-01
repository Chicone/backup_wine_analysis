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
from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind

if __name__ == "__main__":
    # Load dataset paths from config.yaml
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    plot_projection = config.get("plot_projection", False)
    projection_method = config.get("projection_method", "UMAP").upper()
    projection_source = config.get("projection_source", False)
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)

    # Parameters from config file
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]

    # Get the paths corresponding to the selected datasets
    selected_paths = [config["datasets"][name] for name in selected_datasets]

    # Check if all selected dataset contains "pinot"
    if not all("press" in path.lower() for path in selected_paths):
        raise ValueError("Please select a script for Press Wines...")

    # Infer wine_kind from selected dataset paths
    wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])
    strategy = get_strategy_by_wine_kind(wine_kind)

    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_splits = config["num_splits"]
    normalize_flag = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    region = config["region"]
    # wine_kind = config["wine_kind"]
    color_by_country = config["color_by_country"]
    class_by_year = config['class_by_year']

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

    data = np.array(list(gcms.data.values()))
    labels_raw = np.array(list(gcms.data.keys()))
    labels, year_labels = process_labels_by_wine_kind(labels_raw, wine_kind, region, class_by_year, data_dict)

    # Instantiate classifier with data and labels
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
    # cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=classifier, wine_kind=wine_kind,
    #                  year_labels=np.array(year_labels))

    # Run training and collect score vectors
    mean_acc, std_acc, scores, all_labels = cls.train_and_evaluate_all_channels(
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
        projection_source=projection_source,
    )
    if projection_source == "scores":
        data_for_projection = normalize(scores)
        projection_labels = all_labels
    elif projection_source in {"tic", "tis", "tic_tis"}:
        channels = list(range(data.shape[2]))  # use all channels
        data_for_projection = utils.compute_features(data, feature_type=projection_source)
        data_for_projection = normalize(data_for_projection)
        if year_labels:
            projection_labels = year_labels
        else:
            projection_labels = labels
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

    plot_title = f"{pretty_method} of {pretty_source}"

    if data_for_projection is not None:
        reducer = DimensionalityReducer(data_for_projection)
        if projection_dim == 2:
            if projection_method == "UMAP":
                plot_2d(
                    reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state),
                    plot_title, projection_labels, labels, color_by_country
                )
            elif projection_method == "PCA":
                plot_2d(reducer.pca(components=2), plot_title, projection_labels, labels, color_by_country)
            elif projection_method == "T-SNE":
                plot_2d(reducer.tsne(components=2, perplexity=5, random_state=42),
                        plot_title, projection_labels, labels, color_by_country
                        )
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")

        elif projection_dim == 3:
            if projection_method == "UMAP":
                plot_3d(
                    reducer.umap(components=3, n_neighbors=n_neighbors, random_state=random_state),
                    plot_title, projection_labels, labels, color_by_country
                )
            elif projection_method == "PCA":
                plot_3d(reducer.pca(components=3), plot_title, projection_labels, labels, color_by_country)
            elif projection_method == "T-SNE":
                plot_3d(reducer.tsne(components=3, perplexity=5, random_state=42),
                        plot_title, projection_labels, labels, color_by_country
                        )
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")
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