import numpy as np
from matplotlib import pyplot as plt
from gcmswine.data_provider import AccuracyDataProvider
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from matplotlib import cm



def plot_bordeaux(
        embedding, title, labels, label_dict, group_by_country=False, invert_x=False, invert_y=False
):
    """
    Plot a 2D or 3D scatter plot of embedded data with labeled points.
    Automatically applies Bordeaux-specific coloring if labels match Bordeaux format.

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

    if isinstance(label_dict, list):
        label_dict = {label: label for label in label_dict}

    color_map = colormaps.get_cmap("tab20")
    all_digits = all(str(label).isdigit() for label in labels)
    sorted_indices = np.argsort(labels)
    labels = labels[sorted_indices]
    embedding = embedding[sorted_indices]
    used_colors = set()
    color_lookup = {}

    for i, label in enumerate(labels):
        coords = embedding[i].copy()
        if invert_x:
            coords[0] *= -1
        if invert_y:
            coords[1] *= -1
        if not all_digits:
            mod_label, color = change_letter_and_color_bordeaux(label)
            if color not in used_colors:
                (ax if is_3d else plt).scatter(*([[]]*embedding.shape[1]), color=color, label=mod_label[0])
                used_colors.add(color)
            (ax if is_3d else plt).scatter(*coords, color=color, s=60, alpha=0.9)
            (ax if is_3d else plt).text(*coords, mod_label, fontsize=8, color=color)
        else:
            if label not in color_lookup:
                color_lookup[label] = color_map(len(color_lookup) / 20)
            color = color_lookup[label]
            if color not in used_colors:
                (ax if is_3d else plt).scatter(*([[]]*embedding.shape[1]), color=color, label=str(label))
                used_colors.add(color)
            (ax if is_3d else plt).scatter(*coords, color=color, s=60, alpha=0.9)
            (ax if is_3d else plt).text(*coords, str(label), fontsize=8, color=color)

    ax.set_title(title, fontsize=16)
    # Sort legend labels
    handles, labels_ = ax.get_legend_handles_labels()
    if handles and labels_:
        sorted_pairs = sorted(zip(labels_, handles), key=lambda x: x[0])
        labels_, handles = zip(*sorted_pairs)
        ax.legend(handles, labels_, title="Mapped group", loc="best", fontsize='large')
        plt.tight_layout()
    # plt.show(block=False)
    # Ensure static dir exists
    import os
    import matplotlib
    matplotlib.use('Agg')
    plt.savefig("frontend-build/static/plot.png")
    plt.close(fig)


def change_letter_and_color_bordeaux(label):
    """
    Modifies the first letter of a given label and assigns a corresponding color code.

    This function is used to remap the initial character of a label to a different character,
    typically for anonymization or categorization purposes, and to assign a color used in plotting
    or visual grouping.

    Parameters
    ----------
    label : str
        The original string label. The function only considers the first character of this string.

    Returns
    -------
    tuple of (str, str)
        - modified_label : str
            The updated label with the first character changed based on predefined rules.
        - color : str
            A color code (matplotlib-compatible string) associated with the original first character.
            For example:
            - 'b' for blue
            - 'g' for green
            - 'r' for red
            - 'm' for magenta
            - 'k' for black
            - 'y' for yellow
            - 'c' for cyan
            - or specific color names like 'limegreen', 'cornflowerblue', etc.

    Examples
    --------
    >>> change_letter_and_color_bordeaux("Vin123")
    ('Ain123', 'b')

    >>> change_letter_and_color_bordeaux("Apples")
    ('Bpples', 'g')

    Notes
    -----
    - The function does not validate whether `label` is empty. Use with non-empty strings only.
    - The mapping is hardcoded and may require updates for different datasets or applications.
    """
    s = list(label)
    if label[0] == 'V':
        s[0] = 'A'
        color = 'b'
    elif label[0] == 'A':
        s[0] = 'B'
        color = 'g'
    elif label[0] == 'S':
        s[0] = 'C'
        color = 'r'
    elif label[0] == 'F':
        s[0] = 'D'
        color = 'm'
    elif label[0] == 'T':
        s[0] = 'E'
        color = 'k'
    elif label[0] == 'G':
        s[0] = 'F'
        color = 'y'
    elif label[0] == 'B':
        s[0] = 'G'
        color = 'c'
    elif label[0] == 'M':
        s[0] = 'F'
        color = 'y'
    elif label[0] == 'H':
        s[0] = 'H'
        color = 'limegreen'
    elif label[0] == 'I':
        s[0] = 'I'
        color = 'cornflowerblue'
    elif label[0] == 'K':
        s[0] = 'K'
        color = 'olive'
    elif label[0] == 'O':
        s[0] = 'O'
        color = 'tomato'
    return "".join(s), color