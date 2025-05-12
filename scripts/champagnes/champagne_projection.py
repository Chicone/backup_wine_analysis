import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import umap

"""
Champagne Sensory Data Visualization Script

This script loads a dataset of sensory attributes in Champagne wine samples, performs preprocessing and dimensionality 
reduction, and generates visualizations to explore patterns based on selected labels (e.g., variety, ageing, taster,
 etc.).

Main Features:
--------------
1. Loads and processes a CSV file containing sensory or chemical data for Champagne wines.
2. Groups and averages replicate measurements per wine and per taster.
3. Standardizes the data and applies three dimensionality reduction techniques:
   - PCA (Principal Component Analysis)
   - t-SNE (t-distributed Stochastic Neighbor Embedding)
   - UMAP (Uniform Manifold Approximation and Projection)
4. Optionally performs KMeans clustering on PCA-reduced data.
5. Visualizes the results in 2D or 3D with optional per-point labels and coloring by:
   - Clusters (if `do_kmeans` is True), or
   - True labels such as ageing, variety, or taster.
6. Useful for exploring structure in the data, visualizing clusters, and understanding whether
   known labels correspond to chemical features in the dataset.

Parameters (modifiable at the top of the script):
-------------------------------------------------
- label_column        : the column name to use for coloring points (e.g., 'ageing', 'taster', etc.)
- plot_3d             : if True, renders 3D plots; if False, uses 2D
- do_kmeans           : if True, applies KMeans clustering on PCA output and colors by cluster
- show_point_labels   : if True, annotates plots with sample labels

Dependencies:
-------------
pandas, numpy, matplotlib, scikit-learn, umap-learn

"""


# --- Parameters ---
label_column = 'age'  # <-- CHANGE THIS to 'prod area', 'variety',  'cave', 'age', 'ageing', etc.
plot_3d = False  # <-- Toggle this to switch between 2D and 3D plots
do_kmeans = False  # Set to True to enable KMeans clustering after PCA
show_point_labels = True  # Set to False to hide text labels on plot points

# --- Load CSV and preprocess ---
df = pd.read_csv("/home/luiscamara/Documents/datasets/Champagnes/test.csv", skiprows=1)
df = df.iloc[1:]  # remove second header row if needed
df.columns = [col.strip().lower() for col in df.columns]  # lowercase and clean headers

# Extract relevant columns and average replicates
df_selected = df.loc[:, 'code vin':'acid']
df_selected['taster'] = df['taster']  # add the taster info so we can group by it
df_selected.iloc[:, 1:-1] = df_selected.iloc[:, 1:-1].apply(pd.to_numeric, errors='coerce') # all except first and last ('code vin' and 'taster')
df_grouped = df_selected.groupby(['code vin', 'taster']).mean().dropna()

# Prepare labels
labels_df = df[['code vin', label_column, 'taster']].dropna()
labels_df = labels_df.drop_duplicates(subset=['code vin', 'taster'])
labels_df.columns = ['code vin', 'label', 'taster']
df_grouped = df_grouped.reset_index().merge(labels_df, on=['code vin', 'taster'])
wine_labels = df_grouped['label'].values
df_grouped = df_grouped.drop(columns=['code vin', 'taster', 'label'])

# --- Standardize and PCA ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_grouped)

n_components_pca = 3 if plot_3d else 2
pca = PCA(n_components=n_components_pca)
X_pca = pca.fit_transform(X_scaled)

# --- PCA Plot ---
# fig = plt.figure(figsize=(10, 8))
if do_kmeans:
    # --- KMeans on PCA ---
    best_k = 2
    best_score = -1
    for k in range(2, 10):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_pca)
        score = silhouette_score(X_pca, labels)
        print(f'k: {k}, silhouette score: {score:.3f}')
        if score > best_score:
            best_k = k
            best_score = score

    # Final clustering
    kmeans_final = KMeans(n_clusters=best_k, random_state=42)
    # kmeans_final = KMeans(n_clusters=10, random_state=42)
    final_labels = kmeans_final.fit_predict(X_pca)

    colors = final_labels
    legend_label = "Cluster"
    title_suffix = f"with KMeans Clusters (k={best_k})"
else:
    # Just color by known labels
    label_encoder = LabelEncoder()
    label_ids = label_encoder.fit_transform(wine_labels)

    colors = label_ids
    legend_label = label_column
    title_suffix = f"(Colored by {label_column})"

# 3D or 2D PCA plot
if plot_3d:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=colors, cmap='tab20', s=60)
    if show_point_labels:
        for i, label in enumerate(wine_labels):
            ax.text(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2], label, fontsize=9, alpha=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")

else:
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=colors, cmap='tab20', s=60)
    if show_point_labels:
        for i, label in enumerate(wine_labels):
            plt.annotate(label, (X_pca[i, 0], X_pca[i, 1]), fontsize=11, alpha=1)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.title(f"PCA Projection – {title_suffix}")

# --- t-SNE Projection and Plot ---
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(wine_labels)

n_components_tsne = 3 if plot_3d else 2
tsne = TSNE(n_components=n_components_tsne, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))
if plot_3d:
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=label_ids, cmap='tab20', s=60)
    if show_point_labels:
        for i, label in enumerate(wine_labels):
            ax.text(X_pca[i, 0], X_tsne[i, 1], X_tsne[i, 2], label, fontsize=9, alpha=0.8)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_zlabel("t-SNE 3")
else:
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=label_ids, cmap='tab20', s=60)
    if show_point_labels:
        for i, label in enumerate(wine_labels):
            plt.annotate(label, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=11, alpha=1)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
plt.title(f"t-SNE of Champagne Sensory Data (Label: {label_column})")
plt.grid(True)
plt.colorbar(scatter, label=label_column)
plt.tight_layout()
plt.show()

# --- UMAP Projection and Plot ---
reducer = umap.UMAP(n_components=3 if plot_3d else 2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

fig = plt.figure(figsize=(10, 8))

if plot_3d:
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], X_umap[:, 2], c=label_ids, cmap='tab20', s=60)
    if show_point_labels:
        for i, label in enumerate(wine_labels):
            ax.text(X_pca[i, 0], X_umap[i, 1], X_umap[i, 2], label, fontsize=9, alpha=0.8)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_zlabel("UMAP 3")
else:
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=label_ids, cmap='tab20', s=60)
    if show_point_labels:
        for i, label in enumerate(wine_labels):
            plt.annotate(label, (X_umap[i, 0], X_umap[i, 1]), fontsize=9, alpha=1)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

plt.title(f"UMAP of Champagne Sensory Data (Label: {label_column})")
plt.colorbar(scatter, label=label_column)
plt.grid(True)
plt.tight_layout()
plt.show()