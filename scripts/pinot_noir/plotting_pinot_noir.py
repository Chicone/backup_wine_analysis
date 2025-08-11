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
from scipy.interpolate import make_interp_spline
import random
import distinctipy

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

    # --- Calculate transparency based on density ---
    density_measure = np.linalg.det(cov)  # determinant of covariance matrix
    k = 1.5 # sensitivity to dispersion
    min_alpha = 0.1 # minimum transparency
    alpha = 1.0 / (1.0 + k * density_measure)  # Base scaling
    alpha = max(alpha, min_alpha)

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
    rgba[..., 3] = Z * alpha  # opacity

    # Plot Gaussian density as image
    ax.imshow(rgba, extent=(x_min, x_max, y_min, y_max),
              origin='lower', aspect='auto', zorder=zorder_base)

    # Optional: faint contour lines
    contour_levels = np.linspace(0.0, 1.0, 10)
    ax.contour(X, Y, Z, levels=contour_levels,
               colors=[base_color], linewidths=0.2, alpha=0.1, zorder=zorder_base + 2)


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

    def build_estate_color_map(unique_codes):
        """
        Return a dict {wine_code: color} where all wines from the same estate share the same color.
        Includes Burgundy and non-Burgundy estates (e.g., US wines).
        Ensures reproducible and perceptually distinct colors.
        """

        winery_to_color = {}

        # Define estates and their wine codes (extendable)
        estate_mapping = {
            "Drouhin": ["D", "R", "X"],  # Burgundy D, R + US X
            "Bouchard": ["E", "Q"],
            "Jadot": ["P", "Z"]
            # Add other estates here if needed
        }

        # Assign base colors per estate (fixed, distinct)
        estate_base_colors = {
            "Drouhin": (0.7, 0.1, 0.1),  # deep red
            "Bouchard": (0.1, 0.5, 0.7),  # blue
            "Jadot": (0.1, 0.6, 0.2)  # green
        }

        rng = random.Random(42)  # Ensure reproducibility

        # Assign colors for predefined estates
        for estate in sorted(estate_mapping.keys()):
            base = estate_base_colors.get(estate, (0.5, 0.5, 0.5))  # fallback grey
            for code in sorted(estate_mapping[estate]):
                if code in unique_codes:
                    winery_to_color[code] = base

        # Handle remaining codes with reproducible distinct colors
        remaining = sorted([c for c in unique_codes if c not in winery_to_color])
        if remaining:
            extra_colors = distinctipy.get_colors(len(remaining), rng=rng)
            for code, color in zip(remaining, extra_colors):
                winery_to_color[code] = color

        return winery_to_color

    if color_by_origin:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when color_by_origin=True")
        from matplotlib.colors import to_rgb

        # Extract winery codes and map to origins
        winery_codes = np.array([s[0] for s in raw_sample_labels])
        origins = np.array([letter_to_origin.get(c, "Unknown") for c in winery_codes])
        unique_origins = sorted(set(origins))
        unique_wineries = sorted(set(winery_codes))

        # Pride flag color palette (repeated if more origins than colors)
        pride_colors = [
            (0.89, 0.0, 0.05),  # Red
            (1.0, 0.55, 0.0),  # Orange
            (0.85, 0.65, 0.0),    # Gold (replaces yellow, more visible)
            (0.0, 0.6, 0.28),  # Green
            (0.0, 0.45, 0.7),  # Blue
            (0.6, 0.2, 0.7)  # Purple
        ]
        origin_to_color = {
            origin: pride_colors[i % len(pride_colors)] for i, origin in enumerate(unique_origins)
        }

        origin_centers = {}  # store centers for later annotation

        # === 1) DENSITY PLOTS per ORIGIN ===
        for origin in unique_origins:
            mask_origin = origins == origin
            coords = embedding[mask_origin]
            if not np.any(mask_origin):
                continue
            color = origin_to_color[origin]
            marker = 'o'  # density blobs don't need marker differentiation

            # Store center for annotation later
            origin_centers[origin] = np.mean(coords, axis=0)

            if highlight_burgundy_ns and origin == "Burgundy":
                # Split Burgundy densities by N/S shading
                for code in set(winery_codes[mask_origin]):
                    ns = letter_to_burgundy_ns.get(code)
                    submask = (winery_codes == code)
                    ns_color = tuple(c * 0.8 if ns == "North" else min(c * 1.2, 1.0) for c in color)
                    plot_gaussian_cloud_2d(ax, embedding[submask], ns_color, marker, f"{origin} ({ns})")
            else:
                if density_plot:
                    plot_gaussian_cloud_2d(ax, coords, color, marker='o', readable_label=origin)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1], color=color, alpha=0.2, s=50, label=origin)

        # === 2) OVERLAY WINERIES WITH DISTINCT MARKERS ===
        for i, code in enumerate(unique_wineries):
            mask_winery = winery_codes == code
            coords = embedding[mask_winery]
            origin = letter_to_origin.get(code, "Unknown")
            color = origin_to_color[origin]
            marker = markers[i % len(markers)]  # unique marker per winery
            readable_label = label_dict[code]

            # Overlay scatter points for winery with marker shape
            ax.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, s=80, alpha=0.9, label=readable_label)

        # === 3) ANNOTATE ORIGIN CENTERS ===
        for origin, center in origin_centers.items():
            ax.annotate(
                origin,
                (center[0], center[1]),
                fontsize=12,
                fontweight='bold',
                color='black',
                ha='center',
                va='center',
                zorder=1000,  # on top of everything
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.3')
            )

    # if color_by_origin:
    #     if raw_sample_labels is None:
    #         raise ValueError("raw_sample_labels must be provided when color_by_origin=True")
    #     from matplotlib.colors import to_rgb
    #
    #     # Extract winery codes and map to origins
    #     winery_codes = np.array([s[0] for s in raw_sample_labels])
    #     origins = np.array([letter_to_origin.get(c, "Unknown") for c in winery_codes])
    #     unique_origins = sorted(set(origins))
    #     unique_wineries = sorted(set(winery_codes))
    #
    #     cmap = get_cmap("tab10") if len(unique_origins) <= 10 else get_cmap("tab20")
    #     origin_to_color = {
    #         origin: to_rgb(cmap(i % cmap.N)) for i, origin in enumerate(unique_origins)
    #     }
    #
    #     # === 1) DENSITY PLOTS per ORIGIN ===
    #     for origin in unique_origins:
    #         mask_origin = origins == origin
    #         coords = embedding[mask_origin]
    #         if not np.any(mask_origin):
    #             continue
    #         color = origin_to_color[origin]
    #         marker = 'o'  # density blobs don't need marker differentiation
    #         if highlight_burgundy_ns and origin == "Burgundy":
    #             # Split Burgundy densities by N/S shading
    #             for code in set(winery_codes[mask_origin]):
    #                 ns = letter_to_burgundy_ns.get(code)
    #                 submask = (winery_codes == code)
    #                 ns_color = tuple(c * 0.8 if ns == "North" else min(c * 1.2, 1.0) for c in color)
    #                 plot_gaussian_cloud_2d(ax, embedding[submask], ns_color, marker, f"{origin} ({ns})")
    #         else:
    #             if density_plot:
    #                 plot_gaussian_cloud_2d(ax, coords, color, marker='o', readable_label=origin)
    #                 # plot_gaussian_cloud_2d(ax, coords, color)
    #
    #
    #                 origin_centers = []
    #                 for origin in unique_origins:
    #                     mask_origin = origins == origin
    #                     coords = embedding[mask_origin]
    #                     if coords.shape[0] > 0:
    #                         origin_centers.append(np.mean(coords, axis=0))
    #
    #                 # Sort centers by X (left to right)
    #                 origin_centers = np.array(origin_centers)
    #                 origin_centers = origin_centers[np.argsort(origin_centers[:, 0])]
    #
    #                 # if len(origin_centers) > 1:
    #                 #     x_c, y_c = origin_centers[:, 0], origin_centers[:, 1]
    #                 #     spline = make_interp_spline(x_c, y_c, k=2 if len(origin_centers) < 4 else 3)
    #                 #     x_smooth = np.linspace(x_c.min(), x_c.max(), 300)
    #                 #     y_smooth = spline(x_smooth)
    #                 #
    #                 #     # Plot spline LAST (on top of everything)
    #                 #     ax.plot(x_smooth, y_smooth, color="black", linewidth=3, linestyle='-', alpha=0.8, zorder=999)
    #             else:
    #                 ax.scatter(coords[:, 0], coords[:, 1], color=color, alpha=0.2, s=50, label=origin)
    #
    #     # === 2) OVERLAY WINERIES WITH DISTINCT MARKERS ===
    #     for i, code in enumerate(unique_wineries):
    #         mask_winery = winery_codes == code
    #         coords = embedding[mask_winery]
    #         origin = letter_to_origin.get(code, "Unknown")
    #         color = origin_to_color[origin]
    #         marker = markers[i % len(markers)]  # unique marker per winery
    #         readable_label = label_dict[code]
    #
    #         # Overlay scatter points for winery with marker shape
    #         ax.scatter(coords[:, 0], coords[:, 1], color=color, marker=marker, s=80, alpha=0.9, label=readable_label)
    #

    # if color_by_origin:
    #     if raw_sample_labels is None:
    #         raise ValueError("raw_sample_labels must be provided when color_by_origin=True")
    #     import random
    #     from matplotlib.colors import to_rgb
    #     random.seed(42)
    #
    #     winery_codes = np.array([s[0] for s in raw_sample_labels])
    #     origins = np.array([letter_to_origin.get(c, "Unknown") for c in winery_codes])
    #     unique_codes = np.array([s[0] for s in raw_sample_labels])
    #     unique_labels = sorted(set(unique_codes))
    #     unique_origins = sorted(set(origins))
    #
    #     def soften_color(rgb, blend=0.5):
    #         # Blend each RGB channel with white (1.0)
    #         return tuple(blend * c + (1 - blend) * 1.0 for c in rgb)
    #
    #     base_colors = distinctipy.get_colors(len(unique_origins))
    #     soft_colors = [soften_color(c, blend=0.7) for c in base_colors]
    #     origin_to_color = dict(zip(unique_origins, soft_colors))
    #
    #     for i, code in enumerate(unique_labels):
    #         mask = unique_codes == code
    #         if not np.any(mask):
    #             continue
    #         coords = embedding[mask]
    #         marker = markers[i % len(markers)]
    #         origin = letter_to_origin.get(code, "Unknown")
    #         base_color = origin_to_color.get(origin, (0.5, 0.5, 0.5))
    #
    #         if highlight_burgundy_ns and origin == "Burgundy":
    #             ns = letter_to_burgundy_ns.get(code)
    #             if ns == "North":
    #                 color = tuple(c * 0.80 for c in base_color)  # darken
    #             elif ns == "South":
    #                 color = tuple(min(c * 1.2, 1.0) for c in base_color)  # brighten
    #             else:
    #                 color = base_color
    #         else:
    #             color = base_color
    #
    #         readable_label = label_dict[code]
    #
    #         if is_3d:
    #             ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], label=readable_label,
    #                        alpha=0.9, s=80, color=color, marker=marker)
    #         else:
    #             if density_plot:
    #                 plot_gaussian_cloud_2d(ax, coords, color, marker, readable_label)
    #             else:
    #                 ax.scatter(coords[:, 0], coords[:, 1], label=readable_label,
    #                            alpha=0.9, s=80, color=color, marker=marker)

    elif color_by_winery:
        if raw_sample_labels is None:
            raise ValueError("raw_sample_labels must be provided when color_by_winery=True")
        from scipy.stats import multivariate_normal
        from matplotlib.colors import to_rgb
        from scipy.spatial.distance import pdist

        winery_codes = np.array([s[0] for s in raw_sample_labels])
        unique_codes = sorted(set(winery_codes))
        winery_to_color = build_estate_color_map(unique_codes)
        non_estate = [c for c in unique_codes if c not in winery_to_color]
        if non_estate:
            cmap = get_cmap("tab20", len(non_estate))
            for i, code in enumerate(non_estate):
                winery_to_color[code] = to_rgb(cmap(i))

        # === Compute ICD for selected wineries ===
        # selected_wineries = ['D', 'R', 'X', 'E', 'Q', 'P']  # Example: ['A', 'J'] or leave empty for all samples
        selected_wineries = None  # Example: ['A', 'J'] or leave empty for all samples

        if not selected_wineries:
            selected_wineries = unique_codes
            selected_coords = embedding
            label_desc = "all samples"
        else:
            selected_mask = np.isin(winery_codes, selected_wineries)
            selected_coords = embedding[selected_mask]
            label_desc = f"wineries {selected_wineries}"

        if selected_coords.shape[0] >= 2:
            icd = pdist(selected_coords).mean()
            print(f"Intra-cluster distance (ICD) for {label_desc}: {icd:.3f}")
        else:
            print(f"⚠️ Not enough samples to compute ICD for {label_desc}")

        # === PLOT GAUSSIANS + SYMBOLS ===
        for i, code in enumerate(unique_codes):
            if code not in selected_wineries:
                continue  # Skip wineries not in the selected list
            mask = labels == code
            coords = embedding[mask]
            if coords.shape[0] < 2:
                continue
            marker = markers[i % len(markers)]
            base_color = winery_to_color[code]
            readable_label = label_dict[code]

            if density_plot and not is_3d:
                # Plot Gaussian cloud
                plot_gaussian_cloud_2d(ax, coords, base_color, marker, readable_label)
                # Overlay winery symbols
                ax.scatter(coords[:, 0], coords[:, 1], color=base_color, marker=marker,
                           s=80, alpha=0.9, label=readable_label, zorder=999)
            else:
                if is_3d:
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                               label=readable_label, alpha=0.9, s=80,
                               color=base_color, marker=marker)
                else:
                    ax.scatter(coords[:, 0], coords[:, 1],
                               label=readable_label, alpha=0.9, s=80,
                               color=base_color, marker=marker)
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
    # ax.set_title(full_title, fontsize=16)

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





