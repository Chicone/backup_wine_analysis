import numpy as np
from matplotlib import pyplot as plt
from gcmswine.data_provider import AccuracyDataProvider
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from matplotlib import cm
from distinctipy import distinctipy
from scipy.stats import multivariate_normal


def plot_gaussian_cloud_2d(ax, coords, base_color, marker, readable_label, zorder_base=1):
    """
    Plot a 2D Gaussian cloud using imshow, based on the empirical mean and covariance of coords.

    Parameters
    ----------
    ax : matplotlib Axes
        The axes to plot on.
    coords : np.ndarray
        Coordinates of the cluster points (n_samples, 2).
    base_color : tuple
        RGB tuple for the cluster color (3 floats).
    marker : str
        Marker symbol for scatter plot.
    readable_label : str
        Label to use in legend.
    zorder_base : int
        Z-order base value (will add +1 and +2 for scatter and contour).
    """
    # === 2. Compute empirical Gaussian ===
    mean = np.mean(coords, axis=0)
    cov = np.cov(coords.T)
    cov *= 0.4  # optional scaling
    cov += np.eye(cov.shape[0]) * 1e-3  # regularization

    margin = 5
    x_min, x_max = mean[0] - margin, mean[0] + margin
    y_min, y_max = mean[1] - margin, mean[1] + margin
    x = np.linspace(x_min, x_max, 500)
    y = np.linspace(y_min, y_max, 500)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))
    rv = multivariate_normal(mean, cov)
    Z = rv.pdf(pos)
    Z = Z / Z.max()

    # RGBA image
    rgba = np.ones((*Z.shape, 4))
    rgba[..., :3] = base_color
    rgba[..., 3] = Z * 0.7  # opacity

    # Plot Gaussian density as image
    ax.imshow(rgba, extent=(x_min, x_max, y_min, y_max),
              origin='lower', aspect='auto', zorder=zorder_base)

    # Optional: faint contour lines
    contour_levels = np.linspace(0.0, 1.0, 10)
    ax.contour(X, Y, Z, levels=contour_levels,
               colors=[base_color], linewidths=0.2, alpha=0.1, zorder=zorder_base + 2)

    # === 1. Plot the actual points AFTER cloud ===
    ax.scatter(coords[:, 0], coords[:, 1],
               label=readable_label,
               alpha=1.0,
               s=80,
               color=base_color,
               marker=marker,
               zorder=zorder_base + 1)

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
    rot_axes=False,
    raw_sample_labels=None,
    region="origin",
    show_year: bool = False,
    color_by_origin=False,
    color_by_winery=False,
    highlight_burgundy_ns=False,
    exclude_us=False,
    density_plot=False,
):
    """
    Plot a 2D or 3D scatter plot of embedded data with labeled points.
    """

    letter_to_origin = {
        'M': 'Neuchâtel', 'N': 'Neuchâtel', 'J': 'Geneva', 'L': 'Geneva', 'H': 'Valais',
        'U': 'California', 'X': 'Oregon',
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
        us_letters = {"U", "X",}
        # us_letters = {"C", "H"}
        # us_letters = {"U", "X", "C", "H"}
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
    if rot_axes:
        # Clockwise 90° rotation
        rotation_matrix_cw = np.array([[0, 1], [-1, 0]])
        embedding = embedding @ rotation_matrix_cw  # or rotation_matrix_cw


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d' if is_3d else None)

    markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']

    # === Burgundy state grouping ===
    burgundy_states = {
        "Drouhin": ["D", "R"],
        "Bouchard": ["E", "Q"],
        "Jadot": ["P", "Z"]
    }
    state_base_colors = {
        "Drouhin": (0.7, 0.1, 0.1),    # deep red
        "Bouchard": (0.1, 0.5, 0.7),   # blue
        "Jadot": (0.1, 0.6, 0.2)       # green
    }

    def adjust_color(color, factor):
        """Lighten/darken a base color."""
        return tuple(min(1, c + (1 - c) * factor) for c in color)

    def build_burgundy_color_map(unique_codes):
        """Return a dict {code: color} for Burgundy states + others."""
        winery_to_color = {}

        # Assign Burgundy state colors (shaded pairs)
        for state, codes in burgundy_states.items():
            base = state_base_colors[state]
            for i, code in enumerate(codes):
                if code in unique_codes:
                    factor = 0.2 if i == 0 else 0.5  # first code darker, second lighter
                    winery_to_color[code] = adjust_color(base, factor)

        return winery_to_color

    if color_by_origin:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when color_by_origin=True")
        import random
        from matplotlib.colors import to_rgb
        random.seed(42)

        winery_codes = np.array([s[0] for s in raw_sample_labels])
        origins = np.array([letter_to_origin.get(c, "Unknown") for c in winery_codes])
        unique_codes = np.array([s[0] for s in raw_sample_labels])
        unique_labels = sorted(set(unique_codes))
        unique_origins = sorted(set(origins))

        def soften_color(rgb, blend=0.5):
            # Blend each RGB channel with white (1.0)
            return tuple(blend * c + (1 - blend) * 1.0 for c in rgb)

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
                if density_plot:
                    plot_gaussian_cloud_2d(ax, coords, color, marker, readable_label)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
                               alpha=0.9, s=80, color=color, marker=marker)

    elif color_by_winery:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when color_by_winery=True")
        import seaborn as sns
        from scipy.stats import multivariate_normal
        from matplotlib.colors import to_rgb
        from matplotlib.cm import get_cmap
        winery_codes = np.array([s[0] for s in raw_sample_labels])
        unique_codes = sorted(set(winery_codes))
        winery_to_color = build_burgundy_color_map(unique_codes)
        non_burgundy = [c for c in unique_codes if c not in sum(burgundy_states.values(), [])]
        num_non_burgundy = len(non_burgundy)
        if num_non_burgundy > 0:
            cmap = get_cmap("tab20", num_non_burgundy if num_non_burgundy > 0 else 1)
            for i, code in enumerate(non_burgundy):
                winery_to_color[code] = to_rgb(cmap(i))
        if density_plot:
            for i, code in enumerate(unique_codes):
                mask = labels == code
                coords = embedding[mask]
                if coords.shape[0] < 2:
                    continue
                marker = markers[i % len(markers)]
                base_color = winery_to_color[code]
                readable_label = label_dict[code]
                if not is_3d:
                    plot_gaussian_cloud_2d(ax, coords, base_color, marker, readable_label)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label, alpha=0.8, color=base_color, s=90, marker=marker)
        else:
            for i, code in enumerate(unique_codes):
                mask = labels == code
                if not np.any(mask):
                    continue
                coords = embedding[mask]
                marker = markers[i % len(markers)]
                color = winery_to_color[code]
                readable_label = label_dict[code]
                if is_3d:
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label, alpha=0.9, s=80, color=color, marker=marker)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], label=readable_label, alpha=0.9, s=80, color=color, marker=marker)

    else:
        # Default: Plot according to the selected region
        if region == "burgundy":
            # Burgundy North (NB) vs South (SB) is already encoded in `labels`
            unique_burgundy = sorted(set(labels))  # should be ["NB", "SB"]

            colors = distinctipy.get_colors(len(unique_burgundy))
            burgundy_to_color = dict(zip(unique_burgundy, colors))

            for i, burg_label in enumerate(unique_burgundy):
                mask = np.array(labels) == burg_label
                if not np.any(mask):
                    continue
                coords = embedding[mask]
                color = burgundy_to_color[burg_label]
                marker = markers[i % len(markers)]

                if density_plot and not is_3d:
                    # Plot Gaussian density cloud
                    plot_gaussian_cloud_2d(ax, coords, color, marker, burg_label)
                else:
                    # Standard scatter
                    ax.scatter(coords[:, 0], coords[:, 1],
                               label=burg_label,
                               alpha=0.9,
                               s=80,
                               color=color,
                               marker=marker)

        elif region == "winery":
            # === Winery mode with Burgundy state grouping ===
            winery_codes = np.array([s[0] for s in raw_sample_labels])
            unique_codes = sorted(set(winery_codes))

            # Build Burgundy colors
            winery_to_color = build_burgundy_color_map(unique_codes)

            # Add distinct colors for non-Burgundy wineries
            non_burgundy = [c for c in unique_codes if c not in sum(burgundy_states.values(), [])]
            rng = distinctipy.random.Random(42)
            distinct_colors = distinctipy.get_colors(len(non_burgundy), rng=rng)
            for code, color in zip(non_burgundy, distinct_colors):
                winery_to_color[code] = color

            # === Plotting ===
            for i, code in enumerate(unique_codes):
                mask = labels == code
                if not np.any(mask):
                    continue
                coords = embedding[mask]
                color = winery_to_color[code]
                marker = markers[i % len(markers)]
                readable_label = label_dict[code]

                if density_plot and not is_3d:
                    plot_gaussian_cloud_2d(ax, coords, color, marker, readable_label)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1],
                               label=readable_label,
                               alpha=0.9,
                               s=80,
                               color=color,
                               marker=marker)

        elif isinstance(label_dict, list):
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

        ax.legend(ordered_handles, ordered_labels, fontsize='medium', loc='best')
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





