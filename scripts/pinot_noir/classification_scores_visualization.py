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
from matplotlib import colormaps

if __name__ == "__main__":
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

    plot_umap = config.get("plot_umap", False)
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
    return_umap_data = config['plot_umap']


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
    mean_acc, std_acc, scores, all_umap_labels = cls.train_and_evaluate_all_channels(
        num_repeats=num_splits,
        test_size=0.2,
        normalize=normalize_flag,
        use_pca=False,
        classifier_type=classifier,
        feature_type=feature_type,
        region=region,
        LOOPC=True,
        return_umap_data=return_umap_data
    )

    # Normalize
    scores = normalize(scores)

    reducer = DimensionalityReducer(scores)

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

    # Define distinct markers and colors
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'P', 'X', '*', 'h', 'H', '8', 'p', '+', 'x']
    color_map = plt.cm.get_cmap("tab20", len(legend_labels))


    def plot_2d(embedding, title, region, labels, label_dict, group_by_country=False):
        plt.figure(figsize=(8, 6))
        markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
        color_map = colormaps.get_cmap("tab20")

        if region == "winery" or region == "burgundy":
            label_keys = list(label_dict.keys())
            if group_by_country:
                countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
                country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}

            for i, code in enumerate(label_keys):
                mask = labels == code
                readable_label = label_dict[code]
                marker = markers[i % len(markers)]
                color = (country_colors[readable_label.split("(")[-1].strip(")")]
                         if group_by_country else color_map(i / len(label_keys)))
                plt.scatter(*embedding[mask].T, label=readable_label, alpha=0.9, s=80,
                            color=color, marker=marker)
        else:
            for i, label in enumerate(np.unique(labels)):
                mask = labels == label
                plt.scatter(*embedding[mask].T, label=str(label), alpha=0.9, s=80,
                            marker=markers[i % len(markers)])

        plt.title(title, fontsize=16)
        plt.legend(fontsize='large', loc='best')
        plt.tight_layout()
        plt.show(block=False)


    def plot_3d(embedding, title, region, labels, label_dict, group_by_country=False):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
        color_map = colormaps.get_cmap("tab20")

        if region == "winery":
            label_keys = list(label_dict.keys())
            if group_by_country:
                countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
                country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}

            for i, code in enumerate(label_keys):
                mask = labels == code
                readable_label = label_dict[code]
                marker = markers[i % len(markers)]
                color = (country_colors[readable_label.split("(")[-1].strip(")")]
                         if group_by_country else color_map(i / len(label_keys)))
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                           label=readable_label, alpha=0.9, s=80, color=color, marker=marker)
        else:
            for i, label in enumerate(np.unique(labels)):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                           label=str(label), alpha=0.9, s=80, marker=markers[i % len(markers)])

        ax.set_title(title)
        ax.set_xlabel(f"{title.split()[0]} 1")
        ax.set_ylabel(f"{title.split()[0]} 2")
        ax.set_zlabel(f"{title.split()[0]} 3")
        ax.legend(fontsize='medium', loc='best')
        plt.tight_layout()
        plt.show(block=False)

    # ---------- PCA ----------
    # Set this True or False depending on your need
    group_by_country = False

    # if plot_umap:
    #     if umap_dim == 2:
    #         plot_2d(
    #             reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state),
    #             f"UMAP Model Decision Scores ({region})", region, all_umap_labels, legend_labels, group_by_country
    #         )
    #     elif umap_dim == 3:
    #         plot_3d(
    #             reducer.umap(components=3, n_neighbors=n_neighbors, random_state=random_state),
    #             f"UMAP Model Decision Scores ({region})", region, all_umap_labels, legend_labels, group_by_country
    #         )

    # --- 2D ---
    # plot_2d(reducer.pca(components=2), "PCA of Model Decision Scores (2D)", region, all_umap_labels,
    #         winery_legend_labels, group_by_country)
    # plot_2d(reducer.tsne(components=2, perplexity=5, random_state=42), "t-SNE of Model Decision Scores (2D)", region,
    #         all_umap_labels, winery_legend_labels, group_by_country)
    plot_2d(reducer.umap(components=2, n_neighbors=30, random_state=42), f"UMAP Model Decision Scores ({region})", region,
            all_umap_labels, legend_labels, group_by_country)

    # # --- 3D ---
    # plot_3d(reducer.pca(components=3), "PCA of Model Decision Scores (3D)", region, all_umap_labels,
    #         winery_legend_labels, group_by_country)
    # plot_3d(reducer.tsne(components=3, perplexity=5, random_state=42), "t-SNE of Model Decision Scores (3D)", region,
    #         all_umap_labels, winery_legend_labels, group_by_country)
    # plot_3d(reducer.umap(components=3, n_neighbors=15, random_state=42), "UMAP of Model Decision Scores (3D)", region,
    #         all_umap_labels, winery_legend_labels, group_by_country)

    plt.show()