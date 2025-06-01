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
    projection_method = config.get("projection_method", "UMAP").upper()

    projection_source = config.get("projection_source", False)
    umap_dim = config.get("umap_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)

    # Parameters from config file
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]

    # Get the paths corresponding to the selected datasets
    selected_paths = [config["datasets"][name] for name in selected_datasets]

    # Check if all selected dataset contains "pinot"
    if not all("pinot_noir" in path.lower() for path in selected_paths):
        raise ValueError(
            "Please select a script for Pinot Noir. The datasets selected in the config.yaml file do not seem to be compatible with this script. "
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

    # Extract only Burgundy if region is "burgundy"
    if wine_kind == "pinot_noir" and region == "burgundy":
        burgundy_prefixes = ('D', 'P', 'R', 'Q', 'Z', 'E')
        mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
        data = data[mask]
        labels = labels[mask]

    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, None, None)

    # Instantiate classifier with data and labels
    cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=classifier, wine_kind=wine_kind,
                     year_labels=np.array(year_labels))


    # Run training and collect score vectors
    mean_acc, std_acc, scores, all_labels = cls.train_and_evaluate_all_channels(
        num_repeats=num_splits,
        test_size=0.2,
        normalize=normalize_flag,
        use_pca=False,
        classifier_type=classifier,
        feature_type=feature_type,
        region=region,
        LOOPC=True,
        projection_source=projection_source
    )



    # reducer = DimensionalityReducer(scores)

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
    elif region == "burgundy":
        legend_labels = {
            "NB": "Côte de Nuits (north)",
            "SB": "Côte de Beaune (south)"
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

    if data_for_umap is not None:
        reducer = DimensionalityReducer(data_for_umap)
        if umap_dim == 2:
            if projection_method == "UMAP":
                plot_2d(
                    reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state),
                    plot_title, projection_labels, legend_labels, color_by_country
                )
            elif projection_method == "PCA":
                plot_2d(reducer.pca(components=2),plot_title, projection_labels, legend_labels, color_by_country)
            elif projection_method == "T-SNE":
                plot_2d(reducer.tsne(components=2, perplexity=5, random_state=42),
                        plot_title, projection_labels, legend_labels, color_by_country
                        )
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")

        elif umap_dim == 3:
            plot_3d(
                reducer.umap(components=3, n_neighbors=n_neighbors, random_state=random_state),
                plot_title, region, projection_labels, legend_labels, color_by_country
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