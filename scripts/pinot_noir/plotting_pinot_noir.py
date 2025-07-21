import numpy as np
from matplotlib import pyplot as plt
from gcmswine.data_provider import AccuracyDataProvider
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from matplotlib import cm
from distinctipy import distinctipy


def plot_pinot_noir(
    embedding,
    title,
    labels,
    label_dict,
    group_by_country=False,
    test_sample_names=None,
    unique_samples_only=False,
    n_neighbors=None,
    random_state=None,
    invert_x=False,
    invert_y=False,
    color_by_winery=False,
    raw_sample_labels=None,
    color_by_origin=False,
    highlight_burgundy_ns=False,
    exclude_us=False,
    region="origin",
    show_year: bool = False
):
    """
    Plot a 2D or 3D scatter plot of embedded data with labeled points.
    """

    letter_to_origin = {
        'M': 'Neuchatel', 'N': 'Neuchatel', 'J': 'Genève', 'L': 'Genève', 'H': 'Valais',
        'U': 'Californie', 'X': 'Oregon',
        'D': 'Burgundy', 'E': 'Burgundy', 'Q': 'Burgundy', 'P': 'Burgundy', 'R': 'Burgundy', 'Z': 'Burgundy',
        'C': 'Alsace', 'K': 'Alsace', 'W': 'Alsace', 'Y': 'Alsace'
    }

    letter_to_burgundy_ns = {
        'D': 'North', 'E': 'North', 'Q': 'North',
        'P': 'South', 'R': 'South', 'Z': 'South'
    }

    labels = np.array(labels)
    is_3d = embedding.shape[1] == 3

    # Apply exclusion filter early
    if exclude_us:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when exclude_us=True")
        us_letters = {"U", "X"}
        keep_mask = np.array([s[0] not in us_letters for s in raw_sample_labels])

        embedding = embedding[keep_mask]
        labels = labels[keep_mask]
        raw_sample_labels = [s for i, s in enumerate(raw_sample_labels) if keep_mask[i]]
        if test_sample_names is not None:
            test_sample_names = [s for i, s in enumerate(test_sample_names) if keep_mask[i]]

    if invert_x:
        embedding[:, 0] *= -1
    if invert_y:
        embedding[:, 1] *= -1

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
    default_color_map = colormaps.get_cmap("tab20")

    if color_by_origin:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when color_by_origin=True")
        winery_codes = np.array([s[0] for s in raw_sample_labels])
        origins = np.array([letter_to_origin.get(c, "Unknown") for c in winery_codes])
        unique_codes = np.array([s[0] for s in raw_sample_labels])
        unique_labels = sorted(set(unique_codes))
        unique_origins = sorted(set(origins))
        # origin_colors = distinctipy.get_colors(len(unique_origins))
        # origin_to_color = dict(zip(unique_origins, origin_colors))
        def soften_color(rgb, blend=0.5):
            # Blend each RGB channel with white (1.0)
            return tuple(blend * c + (1 - blend) * 1.0 for c in rgb)

        import random
        random.seed(42)  # Ensures same palette every time
        base_colors = distinctipy.get_colors(len(unique_origins))
        soft_colors = [soften_color(c, blend=0.7) for c in base_colors]
        origin_to_color = dict(zip(unique_origins, soft_colors))

        for i, code in enumerate(unique_labels):
            mask = unique_codes == code
            if not np.any(mask):
                continue
            coords = embedding[mask]
            marker = markers[i % len(markers)]
            origin = letter_to_origin.get(code, "Unknown")
            base_color = origin_to_color.get(origin, (0.5, 0.5, 0.5))

            if highlight_burgundy_ns and origin == "Burgundy":
                ns = letter_to_burgundy_ns.get(code)
                if ns == "North":
                    color = tuple(c * 0.80 for c in base_color)  # darken
                elif ns == "South":
                    color = tuple(min(c * 1.2, 1.0) for c in base_color)  # brighten
                else:
                    color = base_color
            else:
                color = base_color

            readable_label = label_dict[code]

            if is_3d:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
                           alpha=0.9, s=80, color=color, marker=marker)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
                           alpha=0.9, s=80, color=color, marker=marker)

    elif color_by_winery:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when color_by_winery=True")
        winery_codes = np.array([s[0] for s in raw_sample_labels])
        unique_codes = sorted(set(winery_codes))
        distinct_colors = distinctipy.get_colors(len(unique_codes))

        for i, code in enumerate(unique_codes):
            mask = winery_codes == code
            if not np.any(mask):
                continue
            coords = embedding[mask]
            marker = markers[i % len(markers)]
            color = distinct_colors[i]
            readable_label = label_dict[code]

            if is_3d:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
                           alpha=0.9, s=80, color=color, marker=marker)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
                           alpha=0.9, s=80, color=color, marker=marker)

    else:
        if isinstance(label_dict, list):
            label_dict = {label: label for label in label_dict}
        category_keys = list(label_dict.keys())
        label_categories = np.array([lbl[0] for lbl in labels])

        if group_by_country:
            countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
            country_colors = {country: default_color_map(i / len(countries)) for i, country in enumerate(countries)}

        for i, cat in enumerate(category_keys):
            mask = label_categories == cat
            if not np.any(mask):
                continue
            readable_label = label_dict[cat]
            marker = markers[i % len(markers)]
            color = (
                country_colors[readable_label.split("(")[-1].strip(")")]
                if group_by_country
                else default_color_map(i / len(category_keys))
            )
            coords = embedding[mask]
            if is_3d:
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
                           alpha=0.9, s=80, color=color, marker=marker)
            else:
                ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
                           alpha=0.9, s=80, color=color, marker=marker)

    if test_sample_names is not None:
        for i, name in enumerate(test_sample_names):
            ax.annotate(
                name,
                (embedding[i, 0], embedding[i, 1]),
                fontsize=10,
                alpha=0.6,
                xytext=(2, 2),
                textcoords="offset points"
            )
    subtitle = ""
    if n_neighbors is not None:
        subtitle += f"n_neighbors={n_neighbors}  "
    if random_state is not None:
        subtitle += f"random_state={random_state}"

    full_title = title if subtitle == "" else f"{title}\n{subtitle.strip()}"
    ax.set_title(full_title, fontsize=16)

    if color_by_winery or color_by_origin:
        handles, labels_found = ax.get_legend_handles_labels()
        label_to_handle = dict(zip(labels_found, handles))

        ordered_handles = []
        ordered_labels = []

        for key in label_dict:
            readable = label_dict[key]
            if readable in labels_found:
                ordered_handles.append(label_to_handle[readable])
                ordered_labels.append(readable)

        ax.legend(ordered_handles, ordered_labels, fontsize='large', loc='best')
    else:
        ax.legend(fontsize='large', loc='best')

    # Annotate years if requested
    # if show_year and raw_sample_labels is None:
    if show_year and raw_sample_labels is not None:
        for i, sample in enumerate(raw_sample_labels):
            if len(sample) >= 3 and sample[1:3].isdigit():
                year = sample[1:3]
                ax.annotate(
                    year,
                    (embedding[i, 0], embedding[i, 1]),
                    fontsize=10,
                    alpha=0.7,
                    xytext=(2, 2),
                    textcoords="offset points"
                )

    plt.tight_layout()
    plt.show(block=False)

# def plot_pinot_noir(
#     embedding,
#     title,
#     labels,
#     label_dict,
#     group_by_country=False,
#     test_sample_names=None,
#     unique_samples_only=False,
#     n_neighbors=None,              # ← new
#     random_state=None,
#     invert_x=False,
#     invert_y=False
# ):
#     """
#     Plot a 2D or 3D scatter plot of embedded data with labeled points.
#
#     Parameters
#     ----------
#     embedding : np.ndarray
#         2D or 3D coordinates of points (n_samples, 2 or 3).
#
#     title : str
#         Title for the plot.
#
#     labels : np.ndarray
#         Array of label codes corresponding to each point.
#
#     label_dict : dict
#         Mapping from label codes to human-readable names.
#
#     group_by_country : bool
#         If True, color by country (only used for winery/burgundy-style labels).
#
#     test_sample_names : list of str, optional
#         Sample names to annotate and (optionally) deduplicate.
#
#     unique_samples_only : bool
#         If True, plot and annotate only the first occurrence of each sample name.
#
#     Returns
#     -------
#     None. Displays a matplotlib plot.
#     """
#     labels = np.array(labels)
#     is_3d = embedding.shape[1] == 3
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d' if is_3d else None)
#
#     if isinstance(label_dict, list):
#         label_dict = {label: label for label in label_dict}
#
#     markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
#     color_map = colormaps.get_cmap("tab20")
#
#     if group_by_country:
#         countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
#         country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}
#
#     # Deduplicate if enabled
#     if test_sample_names is not None and unique_samples_only:
#         seen_names = set()
#         filtered_coords = []
#         filtered_labels = []
#         filtered_names = []
#
#         for i, name in enumerate(test_sample_names):
#             if name in seen_names:
#                 continue
#             seen_names.add(name)
#             filtered_coords.append(embedding[i])
#             filtered_labels.append(labels[i])
#             filtered_names.append(name)
#
#         embedding = np.array(filtered_coords)
#         labels = np.array(filtered_labels)
#         test_sample_names = filtered_names
#
#     label_categories = np.array([lbl[0] for lbl in labels])
#     category_keys = list(label_dict.keys())
#
#     for i, cat in enumerate(category_keys):
#         mask = label_categories == cat
#         if not np.any(mask):
#             continue
#
#         readable_label = label_dict[cat]
#         marker = markers[i % len(markers)]
#
#         if group_by_country:
#             country = readable_label.split("(")[-1].strip(")")
#             color = country_colors[country]
#         else:
#             color = color_map(i / len(category_keys))
#
#         coords = embedding[mask]
#         if is_3d:
#             ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
#                        alpha=0.9, s=80, color=color, marker=marker)
#         else:
#             ax.scatter(-coords[:, 0], -coords[:, 1], label=readable_label,
#                        alpha=0.9, s=80, color=color, marker=marker)
#
#     if test_sample_names is not None:
#         for i, name in enumerate(test_sample_names):
#             ax.annotate(
#                 name,
#                 (-embedding[i, 0], -embedding[i, 1]),
#                 fontsize=10,
#                 alpha=0.6,
#                 xytext=(2, 2),
#                 textcoords="offset points"
#             )
#
#     # ax.set_title(title, fontsize=16)
#     subtitle = ""
#     if n_neighbors is not None:
#         subtitle += f"n_neighbors={n_neighbors}  "
#     if random_state is not None:
#         subtitle += f"random_state={random_state}"
#
#     full_title = title if subtitle == "" else f"{title}\n{subtitle.strip()}"
#     ax.set_title(full_title, fontsize=16)
#     ax.legend(fontsize='large', loc='best')
#     plt.tight_layout()
#     plt.show(block=False)

# def plot_pinot_noir(embedding, title, labels, label_dict, group_by_country=False, test_sample_names=None):
#     """
#     Plot a 2D or 3D scatter plot of embedded data with labeled points.
#
#     Parameters
#     ----------
#     embedding : np.ndarray
#         2D or 3D coordinates of points (n_samples, 2 or 3).
#
#     title : str
#         Title for the plot.
#
#     labels : np.ndarray
#         Array of label codes corresponding to each point.
#
#     label_dict : dict
#         Mapping from label codes to human-readable names.
#
#     group_by_country : bool
#         If True, color by country (only used for winery/burgundy-style labels).
#
#     Returns
#     -------
#     None. Displays a matplotlib plot.
#     """
#     labels = np.array(labels)
#     is_3d = embedding.shape[1] == 3
#
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d' if is_3d else None)
#
#     if isinstance(label_dict, list):
#         label_dict = {label: label for label in label_dict}
#
#     markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
#     color_map = colormaps.get_cmap("tab20")
#
#     if group_by_country:
#         countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
#         country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}
#
#     label_categories = np.array([lbl[0] for lbl in labels])
#     category_keys = list(label_dict.keys())  # preserve the order passed in label_dict
#
#     for i, cat in enumerate(category_keys):
#         mask = label_categories == cat
#         if not np.any(mask):
#             continue
#
#         readable_label = label_dict[cat]
#         marker = markers[i % len(markers)]
#
#         if group_by_country:
#             country = readable_label.split("(")[-1].strip(")")
#             color = country_colors[country]
#         else:
#             color = color_map(i / len(category_keys))
#
#         coords = embedding[mask]
#         if is_3d:
#             ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
#                        alpha=0.9, s=80, color=color, marker=marker)
#         else:
#             # ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
#             ax.scatter(coords[:, 0], -coords[:, 1], label=readable_label,
#                        alpha=0.9, s=80, color=color, marker=marker)
#
#     if test_sample_names is not None:
#         n = min(len(test_sample_names), len(embedding))
#         for i in range(n):
#             ax.annotate(
#                 test_sample_names[i],
#                 (embedding[i, 0], -embedding[i, 1]),
#                 fontsize=10,
#                 alpha=0.6,
#                 xytext=(2, 2),
#                 textcoords="offset points"
#             )
#
#     ax.set_title(title, fontsize=16)
#     ax.legend(fontsize='large', loc='best')
#     plt.tight_layout()
#     plt.show(block=False)





