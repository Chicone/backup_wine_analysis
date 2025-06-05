import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import numpy as np


def compute_projection(X, method="UMAP", dim=2, random_state=42, n_neighbors=15, perplexity=30):
    """
    Compute 2D or 3D projection using UMAP, t-SNE, or PCA.

    Parameters
    ----------
    X : ndarray
        Input data (features or classification scores).
    method : str
        Projection method: 'UMAP', 't-SNE', or 'PCA'.
    dim : int
        Output dimension: 2 or 3.
    random_state : int
        Seed for reproducibility.
    n_neighbors : int
        UMAP parameter.
    perplexity : float
        t-SNE parameter.

    Returns
    -------
    X_proj : ndarray
        Projected data with shape (n_samples, dim).
    """
    if method == "UMAP":
        reducer = umap.UMAP(n_components=dim, n_neighbors=n_neighbors, random_state=random_state)
        X_proj = reducer.fit_transform(X)
    elif method == "t-SNE":
        reducer = TSNE(n_components=dim, perplexity=perplexity, random_state=random_state)
        X_proj = reducer.fit_transform(X)
    elif method == "PCA":
        reducer = PCA(n_components=dim, random_state=random_state)
        X_proj = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unsupported projection method: {method}")

    return X_proj

import re
def plot_projection_with_labels(X, labels, method="UMAP", label_encoder=None, title=None,
                                n_components=2, random_state=42, n_neighbors=15, perplexity=30):
    """
    Plot a 2D or 3D projection (UMAP, t-SNE, or PCA) with color-coded labels and a sorted legend.

    Parameters:
    - X: np.ndarray, shape (n_samples, n_features)
    - labels: array-like, integer-encoded or string labels
    - method: str, one of {"UMAP", "TSNE", "PCA"}
    - label_encoder: LabelEncoder, optional, to decode label integers to strings
    - title: str, optional title (defaults to method name)
    - n_components: int, 2 or 3 for dimensionality
    - random_state: int, for reproducibility
    - n_neighbors: int, for UMAP only
    - perplexity: int, for t-SNE only
    """

    def natural_key(label):
        match = re.search(r'\d+', str(label))
        return int(match.group()) if match else float('inf')

    if label_encoder:
        class_names = label_encoder.inverse_transform(np.unique(labels))
        label_map = dict(zip(np.unique(labels), class_names))
        color_labels = np.vectorize(label_map.get)(labels)
    else:
        color_labels = labels
        label_map = {lbl: str(lbl) for lbl in np.unique(labels)}

    method = method.upper()
    if method == "UMAP":
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, random_state=random_state)
    elif method == "TSNE":
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    elif method == "PCA":
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    X_proj = reducer.fit_transform(X)

    fig = plt.figure(figsize=(8, 6))

    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], X_proj[:, 2], c=labels, cmap="tab20", s=40, alpha=0.8)
    elif n_components == 2:
        ax = fig.add_subplot(111)
        scatter = ax.scatter(X_proj[:, 0], X_proj[:, 1], c=labels, cmap="tab20", s=40, alpha=0.8)
    else:
        raise ValueError("Only 2D and 3D plots are supported (n_components must be 2 or 3)")

    # Manual legend
    legend_items = []
    for lbl in np.unique(labels):
        name = label_map[lbl]
        color = scatter.cmap(scatter.norm(lbl))
        legend_items.append((name, color))

    legend_items.sort(key=lambda x: natural_key(x[0]))
    for name, color in legend_items:
        if n_components == 3:
            ax.scatter([], [], [], c=[color], label=name)
        else:
            ax.scatter([], [], c=[color], label=name)

    ax.legend(title="Classes", loc="best", fontsize=9)
    ax.set_title(title or f"{method} Projection")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    if n_components == 3:
        ax.set_zlabel(f"{method} 3")

    plt.tight_layout()
    plt.show()