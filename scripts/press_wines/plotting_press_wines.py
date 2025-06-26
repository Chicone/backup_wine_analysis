import numpy as np
from matplotlib import pyplot as plt
from gcmswine.data_provider import AccuracyDataProvider
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from matplotlib import cm

def plot_press_wines(
    embedding,
    title,
    labels,
    label_dict,
    group_by_country=False,
    invert_x=False,
    invert_y=False
):
    """
    Plot a 2D or 3D scatter plot of embedded data by press wine categories ('A', 'B', 'C').

    Parameters
    ----------
    embedding : np.ndarray
        2D or 3D coordinates of points (n_samples, 2 or 3).

    title : str
        Title for the plot.

    labels : np.ndarray
        Array of press category codes ('A', 'B', 'C') for each sample.

    label_dict : dict
        Mapping from press category codes to human-readable names.

    group_by_country : bool
        If True, color by country extracted from label_dict values in format 'Name (Country)'.

    invert_x : bool
        If True, inverts the x-axis (multiplies x-coordinates by -1).

    invert_y : bool
        If True, inverts the y-axis (multiplies y-coordinates by -1).

    Returns
    -------
    None. Displays a matplotlib plot.
    """
    labels = np.array(labels)
    is_3d = embedding.shape[1] == 3

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    if label_dict is None:
        label_dict = {label: label for label in np.unique(labels)}
    elif isinstance(label_dict, list):
        label_dict = {label: label for label in label_dict}

    # Apply axis inversion if requested
    if invert_x:
        embedding[:, 0] *= -1
    if invert_y:
        embedding[:, 1] *= -1

    markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8']
    color_map = colormaps.get_cmap("tab20")

    all_digits = all(str(lbl).isdigit() for lbl in labels)
    mode = "year" if all_digits else "press"
    categories = sorted(set(labels))

    if group_by_country:
        countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
        country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}

    for i, category in enumerate(categories):
        mask = labels == category
        if not np.any(mask):
            continue

        readable_label = label_dict.get(category, category)
        marker = markers[i % len(markers)]

        if group_by_country:
            country = readable_label.split("(")[-1].strip(")")
            color = country_colors.get(country, color_map(i / len(categories)))
        else:
            color = color_map(i / len(categories))

        coords = embedding[mask]
        if is_3d:
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
                       alpha=0.9, s=80, color=color, marker=marker)
        else:
            ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
                       alpha=0.9, s=80, color=color, marker=marker)

    ax.set_title(title, fontsize=16)
    ax.legend(title="Year" if mode == "year" else "Press", fontsize='large', loc='best')
    plt.tight_layout()
    # plt.show(block=False)
    import matplotlib
    matplotlib.use('Agg')
    plt.savefig("frontend-build/static/plot.png")
    plt.close(fig)





