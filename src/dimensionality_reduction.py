import numpy as np
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from classification import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

class DimensionalityReducer:
    def __init__(self, data):
        self.data = data

    def pca(self, components=2):
        pca = PCA(n_components=components)
        return pca.fit_transform(self.data)

    def tsne(self, components=2, perplexity=30, random_state=8):
        tsne = TSNE(n_components=components, perplexity=perplexity, random_state=random_state)
        return tsne.fit_transform(self.data)

    def umap(self, components=2, n_neighbors=60, random_state=8):
        reducer = umap.UMAP(n_components=components, n_neighbors=n_neighbors, random_state=random_state)
        return reducer.fit_transform(self.data)

    def perform_pca_on_dict(self, labels, n_components=None):
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca_transformed = pca.fit_transform(self.data)

        # Create a new dictionary with the same labels and the transformed values
        pca_dict = {label: pca_transformed[i].tolist() for i, label in enumerate(labels)}

        return pca_dict

    def cumulative_variance(self, labels, variance_threshold=0.95, plot =False):
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
            plt.title('Cumulative Explained Variance vs. Number of Components')
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

        avg_accuracy = np.mean(accuracies)
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (avg_accuracy, np.std(accuracies) * 2) + "\033[0m")

        return avg_accuracy


