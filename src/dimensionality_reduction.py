import time

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from classification import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from visualizer import Visualizer
import pandas as pd
from sklearn.metrics import silhouette_score

class DimensionalityReducer:
    def __init__(self, data):
        self.data = data

    def pca(self, components=2):
        pca = PCA(n_components=components)
        return pca.fit_transform(self.data)

    def tsne(self, components=2, perplexity=30, random_state=8):
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data)
        tsne = TSNE(n_components=components, perplexity=perplexity, random_state=random_state)
        return tsne.fit_transform(scaled_features)
        # return tsne.fit_transform(self.data)

    def umap(self, components=2, n_neighbors=60, random_state=8):
        # Standardize the features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(self.data)
        reducer = umap.UMAP(n_components=components, n_neighbors=n_neighbors, random_state=random_state)
        return reducer.fit_transform(scaled_features)
        # return reducer.fit_transform(self.data)

    def perform_pca_on_dict(self, labels, n_components=None):
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.data)

        # Create a new dictionary with the same labels and the transformed values
        pca_dict = {label: pca_transformed[i].tolist() for i, label in enumerate(labels)}

        return pca_dict

    def cumulative_variance(self, labels, variance_threshold=0.95, plot = False, dataset_name=None):
        # Perform PCA
        pca = PCA()
        pca_transformed = pca.fit_transform(self.data)

        # Calculate cumulative variance
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Find the number of components that explain the variance_threshold
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        # print(f'{n_components} components at {variance_threshold*100}% variance')

        # Transform the data with the optimal number of components
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.data)

        # Create a new dictionary with the same labels and the transformed values
        pca_dict = {label: pca_transformed[i].tolist() for i, label in enumerate(labels)}

        if plot:
            # Plot cumulative variance
            plt.figure(figsize=(8, 6))
            plt.plot(cumulative_variance, marker='o')
            plt.axhline(y=variance_threshold, color='r', linestyle='--')
            plt.axvline(x=n_components - 1, color='r', linestyle='--')
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Explained Variance')
            title = 'Cumulative Explained Variance vs. Number of Components'
            if dataset_name:
               title = f"{title}\n{dataset_name} dataset"
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.annotate(f'{n_components} components',
                     xy=(n_components - 1, cumulative_variance[n_components - 1]),
                     xytext=(n_components + 1, cumulative_variance[n_components - 1] - 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05))
            plt.show()

        return pca_dict, cumulative_variance, n_components

    def cross_validate_pca_classification(self, processed_labels, n_splits=50, vthresh=0.97, test_size=None):
        accuracies = []
        print(f'Using PCA at {vthresh} accumulated variance')
        print('Split', end=' ', flush=True)
        for i in range(n_splits):
            test_indices = []
            train_indices = []

            for label in np.unique(processed_labels):
                label_indices = np.where(np.array(processed_labels) == label)[0]
                np.random.shuffle(label_indices)
                if test_size is None:
                    test_indices.extend(label_indices[:1])
                    train_indices.extend(label_indices[1:])
                else:
                    split_point = int(len(label_indices) * test_size)
                    test_indices.extend(label_indices[:split_point])
                    train_indices.extend(label_indices[split_point:])

            test_indices = np.array(test_indices)
            train_indices = np.array(train_indices)
            X_train, X_test = self.data[train_indices], self.data[test_indices]
            y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]
            reducer = DimensionalityReducer(X_train)
            pca_dict, cumulative_variance, n_components = reducer.cumulative_variance(y_train, variance_threshold=vthresh, plot=False)
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            classifier = LinearDiscriminantAnalysis()
            classifier.fit(X_train_pca, y_train)
            y_pred = classifier.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            print(i, end=' ', flush=True) if i % 5 == 0 else None
        print()
        avg_accuracy = np.mean(accuracies)
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (avg_accuracy, np.std(accuracies) * 2) + "\033[0m")

        return avg_accuracy

def run_tsne_and_evaluate(analysis, labels, chem_name, neigh_range=range(30, 100, 5), random_states=range(0, 100, 4)):
    data = analysis.data
    reducer = DimensionalityReducer(data)

    best_score = -1
    best_params = None

    # neighbours = range(30, 100, 5)
    # random_states = range(0, 100, 4)

    for neighbour in neigh_range:
        for random_state in random_states:
            tsne_df = analysis.run_tsne(n_neighbors=neighbour, random_state=random_state, plot=False)

            # Calculate the clustering score (e.g., silhouette score)
            score = silhouette_score(tsne_df, labels)
            print(f"Perplexity: {neighbour}, Random State: {random_state}, Silhouette Score: {score}")
            print(f"Best score {best_score}. Best parameters  so far: {best_params}")

            if score > best_score:
                best_score = score
                best_params = (neighbour, random_state)

    print(f"Best Perplexity: {best_params[0]}, Best Random State: {best_params[1]}, Best Silhouette Score: {best_score}")
    return best_params[0], best_params[1], best_score


def run_umap_and_evaluate(analysis, labels, chem_name, neigh_range=range(30, 100, 5), random_states=range(0, 100, 4)):
    data = analysis.data
    reducer = DimensionalityReducer(data)

    best_score = -1
    best_params = None

    # neighbours = range(30, 100, 5)
    # random_states = range(0, 100, 4)

    for neighbour in neigh_range:
        for random_state in random_states:
            umap_df = analysis.run_umap(n_neighbors=neighbour, random_state=random_state, plot=False)
            # umap_result = reducer.umap(components=2, n_neighbors=neighbour, random_state=random_state)
            # umap_df = pd.DataFrame(data=umap_result, columns=['UMAP Component 1', 'UMAP Component 2'], index=labels)
            # title = f'UMAP on {chem_name}; {len(data)} wines with  {neighbour} neighbors and random_state {random_state}'
            #
            # # Plot the results
            # Visualizer.plot_2d_results(umap_df, title, 'UMAP Component 1', 'UMAP Component 2')

            # Calculate the clustering score (e.g., silhouette score)
            score = silhouette_score(umap_df, labels)
            print(f"Neighbors: {neighbour}, Random State: {random_state}, Silhouette Score: {score}")
            print(f"Best score {best_score}. Best parameters  so far: {best_params}")

            if score > best_score:
                best_score = score
                best_params = (neighbour, random_state)

    print(f"Best number neighbors: {best_params[0]}, Best Random State: {best_params[1]}, Best Silhouette Score: {best_score}")
    return best_params[0], best_params[1], best_score

def run_tsne_and_evaluate(analysis, labels, chem_name, perplexities=range(30, 100, 5), random_states=range(0, 100, 4)):
    data = analysis.data
    reducer = DimensionalityReducer(data)

    best_score = -1
    best_params = None

    # perplexities = range(5, 80, 10)
    # random_states = range(0, 100, 10)

    for perplexity in perplexities:
        for random_state in random_states:
            tsne_result = reducer.tsne(components=2, perplexity=perplexity, random_state=random_state)
            tsne_result = -tsne_result  # change the sign of the axes to show data like in the paper
            tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE Component 1', 't-SNE Component 2'], index=labels)

            # Calculate the clustering score (e.g., silhouette score)
            score = silhouette_score(tsne_df, labels)
            print(f"Perplexity: {perplexity}, Random State: {random_state}, Silhouette Score: {score}")
            print(f"Best score {best_score}. Best parameters  so far: {best_params}")

            if score > best_score:
                best_score = score
                best_params = (perplexity, random_state)

    print(f"Best Perplexity: {best_params[0]}, Best Random State: {best_params[1]}, Best Silhouette Score: {best_score}")
    return best_params[0], best_params[1], best_score