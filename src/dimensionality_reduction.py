import time

import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# from classification import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from visualizer import Visualizer
import pandas as pd
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score

class DimensionalityReducer:
    """
    The DimensionalityReducer class provides tools for performing dimensionality reduction on datasets.

    This class offers various methods for reducing the dimensionality of data, making it easier to visualize and
    analyze complex datasets. It supports popular techniques such as PCA (Principal Component Analysis), t-SNE,
    and UMAP, and includes functionality for cumulative variance analysis and cross-validation of PCA for
    classification tasks. The class is designed to handle high-dimensional data efficiently and helps in
    identifying the most important features or components.

    Attributes
    ----------
    data : numpy.ndarray
        The dataset to be reduced in dimensionality.

    Methods
    -------
    pca(components=2)
        Performs Principal Component Analysis (PCA) to reduce the dataset to the specified number of components.
    tsne(components=2, perplexity=30, random_state=8)
        Performs t-SNE to reduce the dataset to the specified number of components.
    umap(components=2, n_neighbors=60, random_state=8)
        Performs UMAP to reduce the dataset to the specified number of components.
    perform_pca_on_dict(labels, n_components=None)
        Applies PCA to the data and returns a dictionary with transformed values.
    cumulative_variance(labels, variance_threshold=0.95, plot=False, dataset_name=None)
        Performs PCA and calculates the cumulative variance explained by the components, optionally plotting the results.
    cross_validate_pca_classification(processed_labels, n_splits=50, vthresh=0.97, test_size=None)
        Performs cross-validation of a PCA-based classification model, reporting the average accuracy.
    """

    def __init__(self, data):
        """Initialize the DimensionalityReducer with a dataset."""
        self.data = data

    def pca(self, components=2):
        """
        Perform Principal Component Analysis (PCA) on the dataset.

        Parameters
        ----------
        components : int, optional
            The number of principal components to keep. Default is 2.

        Returns
        -------
        numpy.ndarray
            The dataset transformed into the specified number of principal components.
        """
        pca = PCA(n_components=components)
        return pca.fit_transform(self.data)

    def tsne(self, components=2, perplexity=30, random_state=8):
        """
        Perform t-SNE (t-distributed Stochastic Neighbor Embedding) on the dataset.

        Parameters
        ----------
        components : int, optional
            The number of dimensions to reduce the dataset to. Default is 2.
        perplexity : float, optional
            The perplexity parameter for t-SNE, which affects the number of nearest neighbors considered. Default is 30.
        random_state : int, optional
            The random seed for reproducibility. Default is 8.

        Returns
        -------
        numpy.ndarray
            The dataset transformed into the specified number of dimensions using t-SNE.
        """
        # Initialize and fit the t-SNE model to the data
        tsne = TSNE(n_components=components, perplexity=perplexity, random_state=random_state)
        return tsne.fit_transform(self.data)

    def umap(self, components=2, n_neighbors=60, random_state=8):
        """
        Perform UMAP (Uniform Manifold Approximation and Projection) on the dataset.

        Parameters
        ----------
        components : int, optional
            The number of dimensions to reduce the dataset to. Default is 2.
        n_neighbors : int, optional
            The number of neighboring points used in local approximations of the manifold structure. Default is 60.
        random_state : int, optional
            The random seed for reproducibility. Default is 8.

        Returns
        -------
        numpy.ndarray
            The dataset transformed into the specified number of dimensions using UMAP.
        """
        # Initialize and fit the UMAP model to the data
        reducer = umap.UMAP(n_components=components, n_neighbors=n_neighbors, random_state=random_state)
        return reducer.fit_transform(self.data)

    def perform_pca_on_dict(self, labels, n_components=None):
        """
        Perform PCA on the dataset and return a dictionary of the transformed data.

        Parameters
        ----------
        labels : list
            A list of labels corresponding to the rows in the dataset.
        n_components : int, optional
            The number of principal components to keep. If None, all components are kept. Default is None.

        Returns
        -------
        dict
            A dictionary where keys are the labels and values are the transformed data in the reduced space.
        """
        # Perform PCA on the dataset
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.data)

        # Create a dictionary mapping labels to transformed data
        pca_dict = {label: pca_transformed[i].tolist() for i, label in enumerate(labels)}

        return pca_dict

    def cumulative_variance(self, labels, variance_threshold=0.95, plot=False, dataset_name=None):
        """
        Perform PCA and calculate the cumulative variance explained by the components.

        Parameters
        ----------
        labels : list
            A list of labels corresponding to the rows in the dataset.
        variance_threshold : float, optional
            The threshold of cumulative variance to be explained by the principal components. Default is 0.95.
        plot : bool, optional
            Whether to plot the cumulative variance explained by the components. Default is False.
        dataset_name : str, optional
            The name of the dataset, used for titling the plot. Default is None.

        Returns
        -------
        dict
            A dictionary where keys are the labels and values are the transformed data in the reduced space.
        numpy.ndarray
            The cumulative variance explained by each component.
        int
            The number of components that explain at least the specified variance threshold.
        """
        # Perform PCA to calculate variance
        pca = PCA()
        pca_transformed = pca.fit_transform(self.data)

        # Calculate the cumulative variance explained by the components
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        # Determine the number of components needed to reach the variance threshold
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

        # Recalculate PCA with the optimal number of components
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.data)

        # Create a dictionary mapping labels to transformed data
        pca_dict = {label: pca_transformed[i].tolist() for i, label in enumerate(labels)}

        if plot:
            # Plot the cumulative variance curve if plotting is requested
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

    def cross_validate_pca_classification(self, cls, n_splits=50, vthresh=0.97, test_size=None):
        """
        Perform cross-validation on a PCA-based classification model.

        Parameters
        ----------
        cls : classifier instance
            The labels associated with the dataset, used for classification.
        n_splits : int, optional
            The number of cross-validation splits. Default is 50.
        vthresh : float, optional
            The variance threshold to be explained by the PCA components. Default is 0.97.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, one sample per label is used. Default is None.

        Returns
        -------
        float
            The average accuracy of the model across the cross-validation splits.
        """
        processed_labels = cls._process_labels()

        accuracies = []
        print(f'Using PCA at {vthresh} accumulated variance')
        print('Split', end=' ', flush=True)
        for i in range(n_splits):
            # Initialize lists to store test and train indices for the current split
            test_indices = []
            train_indices = []

            # Perform stratified splitting by label
            for label in np.unique(processed_labels):
                label_indices = np.where(np.array(processed_labels) == label)[0]
                np.random.shuffle(label_indices)
                if test_size is None:
                    # If test_size is not specified, use one sample per label for testing
                    test_indices.extend(label_indices[:1])
                    train_indices.extend(label_indices[1:])
                else:
                    # If test_size is specified, split the data according to the test_size proportion
                    split_point = int(len(label_indices) * test_size)
                    test_indices.extend(label_indices[:split_point])
                    train_indices.extend(label_indices[split_point:])

            test_indices = np.array(test_indices)
            train_indices = np.array(train_indices)

            # Split the data into training and testing sets based on the calculated indices
            X_train, X_test = self.data[train_indices], self.data[test_indices]
            y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]

            # Perform PCA and transform the training and testing data
            reducer = DimensionalityReducer(X_train)
            pca_dict, cumulative_variance, n_components = reducer.cumulative_variance(y_train, variance_threshold=vthresh, plot=False)
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Train a classifier on the PCA-transformed training data
            # classifier = LinearDiscriminantAnalysis()
            # classifier.fit(X_train_pca, y_train)
            cls.classifier.fit(X_train_pca, y_train)

            # Predict and evaluate accuracy on the PCA-transformed testing data
            # y_pred = classifier.predict(X_test_pca)
            y_pred = cls.classifier.predict(X_test_pca)
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)

            # Print the progress of cross-validation
            print(i, end=' ', flush=True) if i % 5 == 0 else None
        print()

        # Calculate and print the average accuracy
        avg_accuracy = np.mean(accuracies)
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (avg_accuracy, np.std(accuracies) * 2) + "\033[0m")

        return avg_accuracy

def run_tsne_and_evaluate(analysis, labels, chem_name, perplexities=range(30, 100, 5), random_states=range(0, 100, 4)):
    """
    Run t-SNE on the dataset and evaluate the results using the Silhouette Score. Iterate over the range of perplexities
    and random states to find the best combination.

    Parameters
    ----------
    analysis : object
        The analysis object containing the dataset to be reduced.
    labels : list
        The labels corresponding to the data points in the dataset.
    chem_name : str
        The name of the chemical or dataset being analyzed (used for plot titles).
    perplexities : range, optional
        The range of perplexity values to explore for t-SNE. Default is range(30, 100, 5).
    random_states : range, optional
        The range of random state values to explore for t-SNE. Default is range(0, 100, 4).

    Returns
    -------
    tuple
        A tuple containing the best perplexity, best random state, and the best Silhouette Score.
    """
    data = analysis.data
    reducer = DimensionalityReducer(data)

    best_score = -1
    best_params = None

    # Iterate over the range of perplexities and random states
    for perplexity in perplexities:
        for random_state in random_states:
            # Perform t-SNE with the current parameters
            tsne_result = reducer.tsne(components=2, perplexity=perplexity, random_state=random_state)
            tsne_result = -tsne_result  # Change the sign of the axes to match the desired orientation
            tsne_df = pd.DataFrame(data=tsne_result, columns=['t-SNE Component 1', 't-SNE Component 2'], index=labels)

            # Calculate the Silhouette Score for the current t-SNE result
            score = silhouette_score(tsne_df, labels)
            print(f"Perplexity: {perplexity}, Random State: {random_state}, Silhouette Score: {score}")
            print(f"Best score {best_score}. Best parameters so far: {best_params}")

            # Update the best score and parameters if the current score is better
            if score > best_score:
                best_score = score
                best_params = (perplexity, random_state)

    print(f"Best Perplexity: {best_params[0]}, Best Random State: {best_params[1]}, Best Silhouette Score: {best_score}")
    return best_params[0], best_params[1], best_score


def run_umap_and_evaluate(analysis, labels, chem_name, neigh_range=range(30, 100, 5), random_states=range(0, 100, 4)):
    """
    Run UMAP on the dataset and evaluate the results using a combined score of Silhouette, Calinski-Harabasz, and
    Adjusted Rand Index.

    Parameters
    ----------
    analysis : object
        The analysis object containing the dataset to be reduced.
    labels : list
        The labels corresponding to the data points in the dataset.
    chem_name : str
        The name of the chemical or dataset being analyzed (used for plot titles).
    neigh_range : range, optional
        The range of neighbor values to explore for UMAP. Default is range(30, 100, 5).
    random_states : range, optional
        The range of random state values to explore for UMAP. Default is range(0, 100, 4).

    Returns
    -------
    tuple
        A tuple containing the best number of neighbors, the best random state, and the best combined score.
    """
    data = analysis.data
    reducer = DimensionalityReducer(data)

    best_score = -1
    best_params = None

    calinski_min = 0
    calinski_max = 3000  # Threshold found by running on an example dataset

    # Iterate over the range of neighbors and random states
    for neighbour in neigh_range:
        for random_state in random_states:
            # Perform UMAP with the current parameters
            umap_df = analysis.run_umap(n_neighbors=neighbour, random_state=random_state, plot=False, labels=labels)

            # Create cluster assignments for UMAP results
            umap_clusters = pd.cut(umap_df.iloc[:, 0], bins=len(set(labels)))
            umap_cluster_codes = umap_clusters.codes if hasattr(umap_clusters, 'codes') else umap_clusters.cat.codes

            # Calculate the clustering scores
            silhouette = silhouette_score(umap_df, labels)
            calinski_harabasz = calinski_harabasz_score(umap_df, labels)
            adjusted_rand = adjusted_rand_score(labels, umap_cluster_codes)

            # Normalize the scores so they can be combined
            norm_silhouette = (silhouette - (-1)) / (1 - (-1))  # Normalize silhouette score [-1, 1]
            norm_calinski_harabasz = (calinski_harabasz - calinski_min) / (calinski_max - calinski_min)
            norm_adjusted_rand = (adjusted_rand - (-1)) / (1 - (-1))  # Normalize adjusted rand index [-1, 1]

            # Calculate the combined score by averaging the normalized scores
            combined_score = (norm_silhouette + norm_calinski_harabasz + norm_adjusted_rand) / 3

            print(f"Neighbors: {neighbour}, Random State: {random_state}, Combined Score: {combined_score}")
            print(f"Best score {best_score}. Best parameters so far: {best_params}")

            # Update the best score and parameters if the current combined score is better
            if combined_score > best_score:
                best_score = combined_score
                best_params = (neighbour, random_state)

    print(f"Best number neighbors: {best_params[0]}, Best Random State: {best_params[1]}, Best Combined Score: {best_score}")
    return best_params[0], best_params[1], best_score


def run_tsne_and_evaluate(analysis, labels, chem_name, perplexity_range=range(30, 100, 5), random_states=range(0, 100, 4)):
    """
    Run t-SNE on the dataset and evaluate the results using the Silhouette Score.

    Parameters
    ----------
    analysis : object
        The analysis object containing the dataset to be reduced.
    labels : list
        The labels corresponding to the data points in the dataset.
    chem_name : str
        The name of the chemical or dataset being analyzed (used for plot titles).
    neigh_range : range, optional
        The range of neighbor values to explore for t-SNE. Default is range(30, 100, 5).
    random_states : range, optional
        The range of random state values to explore for t-SNE. Default is range(0, 100, 4).

    Returns
    -------
    tuple
        A tuple containing the best number of neighbors, the best random state, and the best Silhouette Score.
    """
    data = analysis.data
    reducer = DimensionalityReducer(data)

    best_score = -1
    best_params = None

    calinski_min = 0
    calinski_max = 3000  # Threshold found by running on an example dataset

    # Iterate over the range of neighbors and random states
    for perplexity in perplexity_range:
        for random_state in random_states:
            # Perform t-SNE with the current parameters
            tsne_df = analysis.run_tsne(perplexity=perplexity, random_state=random_state, plot=False, labels=labels)

            # Create cluster assignments for t-SNE results
            tsne_clusters = pd.cut(tsne_df.iloc[:, 0], bins=len(set(labels)))
            tsne_cluster_codes = tsne_clusters.codes if hasattr(tsne_clusters, 'codes') else tsne_clusters.cat.codes

            # Calculate the clustering scores
            silhouette = silhouette_score(tsne_df, labels)
            calinski_harabasz = calinski_harabasz_score(tsne_df, labels)
            adjusted_rand = adjusted_rand_score(labels, tsne_cluster_codes)

            # Normalize the scores so they can be combined
            norm_silhouette = (silhouette - (-1)) / (1 - (-1))  # Normalize silhouette score [-1, 1]
            norm_calinski_harabasz = (calinski_harabasz - calinski_min) / (calinski_max - calinski_min)
            norm_adjusted_rand = (adjusted_rand - (-1)) / (1 - (-1))  # Normalize adjusted rand index [-1, 1]

            # Calculate the combined score by averaging the normalized scores
            combined_score = (norm_silhouette + norm_calinski_harabasz + norm_adjusted_rand) / 3

            print(f"Perplexity: {perplexity}, Random State: {random_state}, Score: {combined_score}")
            print(f"Best score {best_score}. Best parameters so far: {best_params}")

            # Update the best score and parameters if the current score is better
            if combined_score > best_score:
                best_score = combined_score
                best_params = (perplexity, random_state)

    print(f"Best Perplexity: {best_params[0]}, Best Random State: {best_params[1]}, Best Silhouette Score: {best_score}")

    return best_params[0], best_params[1], best_score