import numpy as np
from matplotlib import pyplot as plt
from gcmswine.data_provider import AccuracyDataProvider
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from matplotlib import cm

def plot_pinot_noir(embedding, title, labels, label_dict, group_by_country=False):
    """
    Plot a 2D or 3D scatter plot of embedded data with labeled points.

    Parameters
    ----------
    embedding : np.ndarray
        2D or 3D coordinates of points (n_samples, 2 or 3).

    title : str
        Title for the plot.

    labels : np.ndarray
        Array of label codes corresponding to each point.

    label_dict : dict
        Mapping from label codes to human-readable names.

    group_by_country : bool
        If True, color by country (only used for winery/burgundy-style labels).

    Returns
    -------
    None. Displays a matplotlib plot.
    """
    labels = np.array(labels)
    is_3d = embedding.shape[1] == 3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    if isinstance(label_dict, list):
        label_dict = {label: label for label in label_dict}

    markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
    color_map = colormaps.get_cmap("tab20")

    if group_by_country:
        countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
        country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}

    label_categories = np.array([lbl[0] for lbl in labels])
    category_keys = list(label_dict.keys())  # preserve the order passed in label_dict

    for i, cat in enumerate(category_keys):
        mask = label_categories == cat
        if not np.any(mask):
            continue

        readable_label = label_dict[cat]
        marker = markers[i % len(markers)]

        if group_by_country:
            country = readable_label.split("(")[-1].strip(")")
            color = country_colors[country]
        else:
            color = color_map(i / len(category_keys))

        coords = embedding[mask]
        if is_3d:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
                       alpha=0.9, s=80, color=color, marker=marker)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
                       alpha=0.9, s=80, color=color, marker=marker)

    ax.set_title(title, fontsize=16)
    ax.legend(fontsize='large', loc='best')
    plt.tight_layout()
    plt.show(block=False)





