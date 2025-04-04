import numpy as np
from pynndescent.optimal_transport import total_cost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.utils import resample
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_classif

import utils
from dimensionality_reduction import DimensionalityReducer
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import RidgeClassifierCV
from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, ShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut, LeavePOut

from utils import find_first_and_last_position, normalize_dict, normalize_data
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DataParallel
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torchvision import models
from torchvision import transforms
import torch.nn.functional as F

from tqdm import tqdm  # For the progress bar
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from joblib import Parallel, delayed
import random
import matplotlib.pyplot as plt

from collections import Counter, defaultdict
import csv



class CustomDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Classifier:
    """
    A classifier class that wraps around various machine learning algorithms
    provided by scikit-learn. This class allows for easy switching between different classifiers
    and provides methods for training and evaluating the models using cross-validation or separate datasets.

    Parameters
    ----------
    data : numpy.ndarray
        The input data to be used for training and evaluation.
    labels : numpy.ndarray
        The labels corresponding to the input data.
    classifier_type : str, optional
        The type of classifier to use. Default is 'LDA'.
        Supported values:
        - 'LDA': Linear Discriminant Analysis
        - 'LR': Logistic Regression
        - 'RFC': Random Forest Classifier
        - 'PAC': Passive Aggressive Classifier
        - 'PER': Perceptron
        - 'RGC': Ridge Classifier
        - 'SGD': Stochastic Gradient Descent Classifier
        - 'SVM': Support Vector Machine
        - 'KNN': K-Nearest Neighbors
        - 'DTC': Decision Tree Classifier
        - 'GNB': Gaussian Naive Bayes
        - 'GBC': Gradient Boosting Classifier
    """
    def __init__(self, data, labels, classifier_type='LDA', wine_kind='bordeaux', window_size=5000, stride=2500,
                 alpha=1, year_labels= None, dataset_origins=None):
        self.data = data
        self.labels = labels
        self.window_size = window_size
        self.stride = stride
        self.wine_kind = wine_kind
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.alpha=alpha
        self.classifier = self._get_classifier(classifier_type)
        self.year_labels = year_labels
        self.dataset_origins = dataset_origins

    def shuffle_split_without_splitting_duplicates(self, X, y, test_size=0.2, random_state=None,
                                                   group_duplicates=True,
                                                   dataset_origins=None):
        """
        Perform ShuffleSplit on samples while ensuring:
        - Duplicates of the same sample (including dataset origin) are kept together (if enabled).
        - Each class is represented in the test set.
        """
        rng = np.random.default_rng(random_state)

        if group_duplicates:
            unique_samples = {}  # {(origin, label): [indices]}
            class_samples = defaultdict(list)  # {class_label: [(origin, label), ...]}

            for idx, label in enumerate(y):
                origin = dataset_origins[idx]
                key = (origin, label)
                unique_samples.setdefault(key, []).append(idx)
                class_label = label[0]  # e.g. 'A' from 'A1'
                class_samples[class_label].append(key)

            sample_keys = list(unique_samples.keys())
            rng.shuffle(sample_keys)

            # Step 1: Randomly select test samples
            num_test_samples = int(len(sample_keys) * test_size)
            test_sample_keys = set(sample_keys[:num_test_samples])

            # Step 2: Ensure all class labels are represented
            test_classes = {key[1][0] for key in test_sample_keys}  # e.g. 'A' from ('merlot', 'A1')
            missing_classes = [cls for cls in class_samples if cls not in test_classes]

            for class_label in missing_classes:
                candidate_keys = class_samples[class_label]
                additional_key = tuple(rng.choice(candidate_keys))
                test_sample_keys.add(additional_key)

            # Step 3: Translate keys into train/test indices
            test_indices = [idx for key in test_sample_keys for idx in unique_samples[key]]
            train_indices = [idx for key in sample_keys if key not in test_sample_keys for idx in
                             unique_samples[key]]

        else:
            # Fallback: treat each instance independently
            indices = np.arange(len(y))
            rng.shuffle(indices)
            num_test_samples = int(len(indices) * test_size)
            test_indices = indices[:num_test_samples]
            train_indices = indices[num_test_samples:]

        return np.array(train_indices), np.array(test_indices)

    def _get_classifier(self, classifier_type):
        """
        Return the classifier object based on the classifier type.

        Parameters
        ----------
        classifier_type : str
            The type of classifier to initialize. Supported types include 'LDA', 'LR', 'RFC',
            'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB', and 'GBC'.

        Returns
        -------
        sklearn.base.BaseEstimator
            An instance of the selected scikit-learn classifier.
        """
        print(f'Classifier: {classifier_type}')
        if classifier_type == 'LDA':
            return LinearDiscriminantAnalysis()
        elif classifier_type == 'LR':
            return LogisticRegression(C=1.0, random_state=0, n_jobs=-1, max_iter=10000)
        elif classifier_type == 'RFC':
            return RandomForestClassifier(n_estimators=100)
        elif classifier_type == 'PAC':
            return PassiveAggressiveClassifier()
        elif classifier_type == 'PER':
            return Perceptron()
        elif classifier_type == 'RGC':
            return RidgeClassifier(alpha=self.alpha)
        elif classifier_type == 'SGD':
            return SGDClassifier()
        elif classifier_type == 'SVM':
            return SVC(kernel='rbf', random_state=0)
        elif classifier_type == 'KNN':
            return KNeighborsClassifier(n_neighbors=3)
        elif classifier_type == 'DTC':
            return DecisionTreeClassifier()
        elif classifier_type == 'GNB':
            return GaussianNB()
        elif classifier_type == 'GBC':
            return GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
        elif classifier_type == 'HGBC':
            return HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.2, max_iter=50, max_bins=128)

    def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None, normalize=False,
                           scaler_type='standard', use_pca=False, vthresh=0.97, region=None):
        """
        Train and evaluate the classifier using cross-validation, and calculate the mean confusion matrix.
        Can also perform PCA-based classification when `pca=True`.

        Parameters
        ----------
        n_splits : int, optional
            The number of splits for cross-validation. Default is 50.
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        random_seed : int, optional
            The random seed for reproducibility. Default is 42.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, only one sample is used for testing. Default is None.
        normalize : bool, optional
            Whether to normalize the data. Default is False.
        scaler_type : str, optional
            The type of scaler to use for normalization if `normalize` is True. Default is 'standard'.
        pca : bool, optional
            Whether to apply PCA to the data. Default is False.
        vthresh : float, optional
            The variance threshold to be explained by the PCA components. Default is 0.97.

        Returns
        -------
        dict
            A dictionary containing mean accuracy, precision, recall, F1-score, and the mean confusion matrix.
        """

        from sklearn.utils.class_weight import compute_sample_weight

        # Set the random seed for reproducibility
        np.random.seed(random_seed)

        # Initialize metrics accumulators
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        confusion_matrix_sum = None
        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region  == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
           custom_order = None
        # elif region  == 'bordeaux_chateaux':
        #     custom_order = ['D', 'E', 'F', 'G', 'A', 'B', 'C']


        n_components = 100


        if use_pca:
            # Estimate best number of components (on all data)
            reducer = DimensionalityReducer(self.data)
            _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
            pca = PCA(n_components=n_components, svd_solver='randomized')

            print(f'Applying PCA with {vthresh} variance threshold')
            print(f'PCA components= {n_components}')

        print('Split', end=' ', flush=True)
        # Perform cross-validation over the specified number of splits
        for i in range(n_splits):
            # Split the data into training and testing sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                vintage=vintage, test_size=test_size
            )
            # sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            # Normalize the data if normalization is enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)  # Fit scaler on training data
                X_test = scaler.transform(X_test)  # Transform test data using the train scaler

            # Apply PCA if enabled
            if use_pca:
               # Perform PCA on the training data
                X_train = pca.fit_transform(X_train[:, ::10])
                X_test = pca.transform(X_test[:, ::10])

            # Train the classifier on the (optionally PCA-transformed) training data
            # self.classifier.fit(X_train, y_train, sample_weight=sample_weights)
            self.classifier.fit(X_train, y_train)

            # Print the current split number every 5 iterations to show progress
            print(i, end=' ', flush=True) if i % 5 == 0 else None

            # Make predictions on the test set
            y_pred = self.classifier.predict(X_test)

            # Calculate accuracy and other metrics
            accuracy_scores.append(self.classifier.score(X_test, y_test))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall_scores.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute the confusion matrix for the current split
            if custom_order is not None:
                cm = confusion_matrix(y_test, y_pred, labels=custom_order)
            else:
                cm = confusion_matrix(y_test, y_pred)

            # Accumulate the confusion matrix
            if confusion_matrix_sum is None:
                confusion_matrix_sum = np.zeros_like(cm)  # Initialize the accumulator with zeros the same shape as `cm`

            confusion_matrix_sum += cm  # Add the confusion matrix from the current split

        # Print a new line after the loop completes
        print()

        # Convert lists of scores to numpy arrays for easier statistical calculations
        accuracy_scores = np.asarray(accuracy_scores)
        precision_scores = np.asarray(precision_scores)
        recall_scores = np.asarray(recall_scores)
        f1_scores = np.asarray(f1_scores)

        # Calculate the mean confusion matrix by dividing the accumulated matrix by the number of splits
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary of results
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (
        accuracy_scores.mean(), accuracy_scores.std() * 2) + "\033[0m")
        print("\033[96m" + "Precision: %0.3f (+/- %0.3f)" % (
        precision_scores.mean(), precision_scores.std() * 2) + "\033[0m")
        print("\033[96m" + "Recall: %0.3f (+/- %0.3f)" % (recall_scores.mean(), recall_scores.std() * 2) + "\033[0m")
        print("\033[96m" + "F1 Score: %0.3f (+/- %0.3f)" % (f1_scores.mean(), f1_scores.std() * 2) + "\033[0m")

        # Optionally print the mean confusion matrix
        print("\033[96m" + "Mean Confusion Matrix (over all splits):" + "\033[0m")
        print(mean_confusion_matrix)

        # Return the mean scores and the mean confusion matrix
        return {
            'mean_accuracy': accuracy_scores.mean(),
            'mean_precision': precision_scores.mean(),
            'mean_recall': recall_scores.mean(),
            'mean_f1_score': f1_scores.mean(),
            'mean_confusion_matrix': mean_confusion_matrix
        }

    def train_and_evaluate_balanced(self, n_inner_repeats=50, random_seed=42,
                                    test_size=0.2, normalize=False, scaler_type='standard',
                                    use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                    test_on_discarded=False):
        """
        Train and evaluate the classifier using repeated outer stratified splits. For each outer repetition,
        the training set is further split using RepeatedLeaveOneFromEachClassCV and validation metrics are computed.
        The outer test set is ignored; only the training set is used for inner CV.

        Parameters
        ----------
        n_inner_repeats : int
            Number of inner CV repeats (passed to RepeatedLeaveOneFromEachClassCV).
        random_seed : int, optional
            Seed for reproducibility.
        test_size : float, optional
            Fraction of data to hold out (unused in metric computations).
        normalize : bool, optional
            Whether to normalize the data.
        scaler_type : str, optional
            Which scaler to use ('standard' or 'minmax').
        use_pca : bool, optional
            Whether to apply PCA.
        vthresh : float, optional
            Variance threshold for PCA.
        region : str, optional
            Determines custom ordering for confusion matrix.
        batch_size, num_epochs, learning_rate : not used in this snippet.

        Returns
        -------
        dict
            A dictionary with the average inner validation metrics across all outer repeats.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.decomposition import PCA
        from sklearn.utils.class_weight import compute_sample_weight

        class RepeatedLeaveOneSamplePerClassCV(BaseCrossValidator):
            """
            Custom cross-validator that randomly selects one sample per class as the test set,
            ensuring that all replicates of the same sample stay together, or selects individual
            samples if standard mode is chosen. Uses composite labels to preserve replicate information.

            Parameters
            ----------
            n_repeats : int, optional (default=50)
                Number of repetitions.
            shuffle : bool, optional (default=True)
                Whether to shuffle the samples before splitting.
            random_state : int, optional (default=None)
                Seed for reproducibility.
            use_groups : bool, optional (default=True)
                If True, replicates of the same sample stay together (group-like behavior).
                If False, selects individual samples without considering replicates.

            Attributes
            ----------
            n_repeats : int
                Number of repetitions.
            shuffle : bool
                Whether to shuffle samples.
            random_state : int
                Random seed.
            use_groups : bool
                Determines whether to use group-like splitting or standard splitting.
            """

            def __init__(self, n_repeats=50, shuffle=True, random_state=None, use_groups=True):
                self.n_repeats = n_repeats
                self.shuffle = shuffle
                self.random_state = random_state
                self.use_groups = use_groups

            def get_n_splits(self, X, y, groups=None):
                """Returns the number of splits."""
                return self.n_repeats

            def split(self, X, y):
                """
                Splits the dataset into training and test sets using either:
                - Group-like splitting (all replicates of the same sample stay together).
                - Standard splitting (individual samples are selected without considering replicates).

                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    Feature matrix or sample labels.
                y : array-like, shape (n_samples,)
                    Composite labels (e.g., 'A1', 'B9') that preserve replicate information.

                Yields
                ------
                train_indices : ndarray
                    Training indices.
                test_indices : ndarray
                    Test indices.
                """

                # Step 1: Extract category from composite labels (e.g., 'A1' -> 'A')
                def get_category(label):
                    """Extracts the category (A, B, or C) from the composite label."""
                    match = re.match(r'([A-C])\d+', label)
                    return match.group(1) if match else label  # Fallback if no match

                rng = np.random.default_rng(self.random_state)

                # ✅ OPTION 1: Group-Like Splitting (use_groups=True)
                if self.use_groups:
                    # Group indices by composite labels
                    indices_by_sample = {}
                    for idx, label in enumerate(y):
                        indices_by_sample.setdefault(label, []).append(idx)

                    # Generate splits
                    for _ in range(self.n_repeats):
                        test_indices = []
                        for category in set(map(get_category, y)):
                            # Select all samples that belong to the current category
                            class_samples = [label for label in indices_by_sample.keys() if
                                             get_category(label) == category]

                            # Randomly choose one sample
                            chosen_sample = rng.choice(class_samples, size=1, replace=False)[0] if self.shuffle else \
                            class_samples[0]

                            # Add all indices of the chosen sample to the test set
                            test_indices.extend(indices_by_sample[chosen_sample])

                        test_indices = np.array(test_indices)
                        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

                        yield train_indices, test_indices

                # ✅ OPTION 2: Standard Splitting (use_groups=False)
                else:
                    # Collect indices by category
                    indices_by_category = {}
                    for idx, label in enumerate(y):
                        category = get_category(label)
                        indices_by_category.setdefault(category, []).append(idx)

                    # Generate splits
                    for _ in range(self.n_repeats):
                        test_indices = []
                        for category, indices in indices_by_category.items():
                            chosen = rng.choice(indices, size=1, replace=False) if self.shuffle else [indices[0]]
                            test_indices.extend(chosen)

                        test_indices = np.array(test_indices)
                        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

                        yield train_indices, test_indices


        def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None, group_duplicates=True):
            """
                Perform ShuffleSplit on samples while ensuring that:
                - Duplicates of the same sample are kept together (if enabled).
                - Each class is represented in the test set.

            Args:
                X (array-like): Feature matrix or sample labels.
                y (array-like): Composite labels (e.g., ['A1', 'A1', 'B2', 'B2']).
                test_size (float): Fraction of samples to include in the test set.
                random_state (int): Random seed for reproducibility.
                group_duplicates (bool): If True, duplicates of the same sample are kept together.
                                         If False, duplicates are treated independently.

            Returns:
                tuple: train_indices, test_indices (numpy arrays)
            """
            import numpy as np

            rng = np.random.default_rng(random_state)

            if group_duplicates:
                unique_samples = {}  # {sample_label: [indices]}
                class_samples = defaultdict(list)  # {class_label: [sample_label1, sample_label2, ...]}

                for idx, label in enumerate(y):
                    unique_samples.setdefault(label, []).append(idx)
                    class_samples[label[0]].append(label)  # Assuming class is the first character (e.g., 'A1' -> 'A')

                sample_labels = list(unique_samples.keys())
                rng.shuffle(sample_labels)

                # Step 1: Randomly select test samples
                num_test_samples = int(len(sample_labels) * test_size)
                test_sample_labels = set(sample_labels[:num_test_samples])

                # Step 2: Ensure each class is represented in the test set
                test_classes = {label[0] for label in test_sample_labels}
                missing_classes = [c for c in class_samples if c not in test_classes]

                # Step 3: Force at least one sample from missing classes
                for class_label in missing_classes:
                    additional_sample = rng.choice(class_samples[class_label])
                    test_sample_labels.add(additional_sample)

                # Step 4: Convert sample labels to index lists
                test_indices = [idx for label in test_sample_labels for idx in unique_samples[label]]
                train_indices = [idx for label in sample_labels if label not in test_sample_labels for idx in
                                 unique_samples[label]]

            else:
                # Treat each instance independently
                indices = np.arange(len(y))
                rng.shuffle(indices)

                # Calculate the number of test samples
                num_test_samples = int(len(indices) * test_size)

                # Split into train and test sets
                test_indices = indices[:num_test_samples]
                train_indices = indices[num_test_samples:]

            return np.array(train_indices), np.array(test_indices)


        def extract_category_labels(composite_labels):
            """
            Convert composite labels (e.g., 'A1', 'B9', 'C2') to category labels ('A', 'B', 'C').

            Args:
                composite_labels (array-like): List or array of composite labels.

            Returns:
                list: List of category labels.
            """
            return [re.match(r'([A-C])', label).group(1) if re.match(r'([A-C])', label) else label
                    for label in composite_labels]

        def process_fold(inner_train_idx, inner_val_idx, X_train_full, y_train_full, normalize, scaler_type, use_pca,
                         vthresh, custom_order):
            X_train = X_train_full[inner_train_idx]
            y_train = extract_category_labels(y_train_full[inner_train_idx])
            X_val = X_train_full[inner_val_idx]
            y_val = extract_category_labels(y_train_full[inner_val_idx])

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_val = scaler.transform(X_val)

            if use_pca:
                pca = PCA(n_components=None, svd_solver='randomized')
                pca.fit(X_train)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.searchsorted(cumulative_variance, vthresh) + 1
                n_components = min(n_components, len(np.unique(y_train)))
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_val = pca.transform(X_val)

            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_val)

            acc = self.classifier.score(X_val, y_val)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            sw = compute_sample_weight(class_weight='balanced', y=y_val)
            w_acc = np.average(y_pred == y_val, weights=sw)
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_val, y_pred, labels=custom_order if custom_order else None)

            return acc, bal_acc, w_acc, prec, rec, f1, cm


        category_labels = extract_category_labels(self.labels)

        # Compute class distribution
        if self.year_labels.size > 0 and np.any(self.year_labels != None):
            class_counts = Counter(self.year_labels)
        else:
            class_counts = Counter(category_labels)
        # class_counts = Counter(category_labels)

        total_samples = sum(class_counts.values())

        # Compute Correct Chance Accuracy (Taking Class Distribution into Account)
        class_probabilities = np.array([count / total_samples for count in class_counts.values()])
        chance_accuracy = np.sum(class_probabilities ** 2)  # Sum of squared class probabilities


        # Set up a custom order for the confusion matrix if a region is specified.
        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
            custom_order = None

        # Initialize accumulators for outer-repetition averaged metrics.
        eval_accuracy = []
        eval_balanced_accuracy = []
        eval_weighted_accuracy = []
        eval_precision = []
        eval_recall = []
        eval_f1 = []
        eval_cm = []

        # Use a reproducible RNG.
        if random_seed is None:
            random_seed = np.random.randint(0, int(1e6))
        rng = np.random.default_rng(random_seed)


        # Choose whether to use groups based on wine type.
        use_groups = True if self.wine_kind == "press" else False
        # use_groups = False

        # Outer split: use StratifiedShuffleSplit to split the data.
        # # sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.integers(0, int(1e6)))
        # sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.integers(0, int(1e6)))
        # train_idx, _ = next(sss.split(self.data, self.labels))

        # train_idx, test_idx = shuffle_split_without_splitting_duplicates(
        #     self.data, self.labels, test_size=test_size, random_state=rng.integers(0, int(1e6)),
        #     group_duplicates=use_groups
        # )

        train_idx, test_idx = shuffle_split_without_splitting_duplicates(
            self.data, self.labels, test_size=test_size, random_state=random_seed,
            group_duplicates=use_groups
        )

        X_train_full, X_test = self.data[train_idx], self.data[test_idx]
        if self.year_labels.size > 0 and np.any(self.year_labels != None):
            y_train_full, y_test = self.year_labels[train_idx], self.year_labels[test_idx]
        else:
            y_train_full, y_test = self.labels[train_idx], self.labels[test_idx]

        # y_train_full, y_test = self.labels[train_idx], self.labels[test_idx]

        if test_on_discarded:
            if normalize:
                X_train_full, scaler = normalize_data(X_train_full, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            if use_pca:
                pca = PCA(n_components=None, svd_solver='randomized')
                pca.fit(X_train_full)
                X_train_full = pca.transform(X_train_full)
                X_test = pca.transform(X_test)

            if self.year_labels.size > 0 and np.any(self.year_labels != None):
                self.classifier.fit(X_train_full, y_train_full)
            else:
                self.classifier.fit(X_train_full, np.array(extract_category_labels(y_train_full)))
                y_test = extract_category_labels(y_test)
            # self.classifier.fit(X_train_full, extract_category_labels(y_train_full))
            # y_test = extract_category_labels(y_test)

            y_pred = self.classifier.predict(X_test)


            eval_accuracy.append(accuracy_score(y_test, y_pred))
            eval_balanced_accuracy.append(balanced_accuracy_score(y_test, y_pred))
            eval_weighted_accuracy.append(
                np.average(y_pred == y_test, weights=compute_sample_weight('balanced', y_test)))
            eval_precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            eval_recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            eval_f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            eval_cm.append(confusion_matrix(y_test, y_pred))

            if print_results:
                print(f"  Test on discarded data metrics:")
                print(f"    Accuracy: {eval_accuracy[-1]:.3f}")
                print(f"    Balanced Accuracy: {eval_balanced_accuracy[-1]:.3f}")
                print(f"    Weighted Accuracy: {eval_weighted_accuracy[-1]:.3f}")
                print(f"    Precision: {eval_precision[-1]:.3f}")
                print(f"    Recall: {eval_recall[-1]:.3f}")
                print(f"    F1 Score: {eval_f1[-1]:.3f}")

        else:
            # cv = RepeatedLeaveOneFromEachClassCV(n_repeats=n_inner_repeats, shuffle=True, random_state=random_seed)
            cv = RepeatedLeaveOneSamplePerClassCV(
                n_repeats=n_inner_repeats,
                shuffle=True,
                random_state=random_seed,
                use_groups=use_groups
            )

            results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(process_fold)(inner_train_idx, inner_val_idx, X_train_full, y_train_full, normalize,
                                      scaler_type, use_pca, vthresh, custom_order)
                for inner_train_idx, inner_val_idx in cv.split(X_train_full, y_train_full)
            )

            inner_acc, inner_bal_acc, inner_w_acc, inner_prec, inner_rec, inner_f1, inner_cm = zip(*results)

            eval_accuracy.append(np.mean(inner_acc))
            eval_balanced_accuracy.append(np.mean(inner_bal_acc))
            eval_weighted_accuracy.append(np.mean(inner_w_acc))
            eval_precision.append(np.mean(inner_prec))
            eval_recall.append(np.mean(inner_rec))
            eval_f1.append(np.mean(inner_f1))
            eval_cm.append(np.mean(inner_cm, axis=0))

            if print_results:
                print(f"  Inner CV Averages:")
                print(f"    Accuracy: {eval_accuracy[-1]:.3f}")
                print(f"    Balanced Accuracy: {eval_balanced_accuracy[-1]:.3f}")
                print(f"    Weighted Accuracy: {eval_weighted_accuracy[-1]:.3f}")
                print(f"    Precision: {eval_precision[-1]:.3f}")
                print(f"    Recall: {eval_recall[-1]:.3f}")
                print(f"    F1 Score: {eval_f1[-1]:.3f}")

        # Compute the averaged confusion matrix across all repetitions
        overall_cm = np.mean(eval_cm, axis=0)

        # Normalize confusion matrix row-wise (true label-wise)
        overall_cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1, keepdims=True)

        overall_results = {
            'chance_accuracy': chance_accuracy,
            'overall_accuracy': np.mean(eval_accuracy),
            'overall_balanced_accuracy': np.mean(eval_balanced_accuracy),
            'overall_weighted_accuracy': np.mean(eval_weighted_accuracy),
            'overall_precision': np.mean(eval_precision),
            'overall_recall': np.mean(eval_recall),
            'overall_f1_score': np.mean(eval_f1),
            'overall_confusion_matrix': overall_cm,
            'overall_confusion_matrix_normalized': overall_cm_normalized,
        }

        if print_results:
            print("\nFinal Results:")
            print(f"  Overall Accuracy: {overall_results['overall_accuracy']:.3f}")
            print(f"  Overall Balanced Accuracy: {overall_results['overall_balanced_accuracy']:.3f}")
            print(f"  Overall Weighted Accuracy: {overall_results['overall_weighted_accuracy']:.3f}")
            print(f"  Overall Precision: {overall_results['overall_precision']:.3f}")
            print(f"  Overall Recall: {overall_results['overall_recall']:.3f}")
            print(f"  Overall F1 Score: {overall_results['overall_f1_score']:.3f}")
            # print("Overall Mean Confusion Matrix:")
            # print(overall_results['overall_confusion_matrix'])
            # print(overall_results['overall_confusion_matrix_normalized'])

        return overall_results

    def train_and_evaluate_balanced_diff_origins(self, n_inner_repeats=50, random_seed=42,
                                    test_size=0.2, normalize=False, scaler_type='standard',
                                    use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                    test_on_discarded=False):

        # def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None, group_duplicates=True):
        #     """
        #         Perform ShuffleSplit on samples while ensuring that:
        #         - Duplicates of the same sample are kept together (if enabled).
        #         - Each class is represented in the test set.
        #
        #     Args:
        #         X (array-like): Feature matrix or sample labels.
        #         y (array-like): Composite labels (e.g., ['A1', 'A1', 'B2', 'B2']).
        #         test_size (float): Fraction of samples to include in the test set.
        #         random_state (int): Random seed for reproducibility.
        #         group_duplicates (bool): If True, duplicates of the same sample are kept together.
        #                                  If False, duplicates are treated independently.
        #
        #     Returns:
        #         tuple: train_indices, test_indices (numpy arrays)
        #     """
        #     import numpy as np
        #
        #     rng = np.random.default_rng(random_state)
        #
        #     if group_duplicates:
        #         unique_samples = {}  # {sample_label: [indices]}
        #         class_samples = defaultdict(list)  # {class_label: [sample_label1, sample_label2, ...]}
        #
        #         for idx, label in enumerate(y):
        #             unique_samples.setdefault(label, []).append(idx)
        #             class_samples[label[0]].append(label)  # Assuming class is the first character (e.g., 'A1' -> 'A')
        #
        #         sample_labels = list(unique_samples.keys())
        #         rng.shuffle(sample_labels)
        #
        #         # Step 1: Randomly select test samples
        #         num_test_samples = int(len(sample_labels) * test_size)
        #         test_sample_labels = set(sample_labels[:num_test_samples])
        #
        #         # Step 2: Ensure each class is represented in the test set
        #         test_classes = {label[0] for label in test_sample_labels}
        #         missing_classes = [c for c in class_samples if c not in test_classes]
        #
        #         # Step 3: Force at least one sample from missing classes
        #         for class_label in missing_classes:
        #             additional_sample = rng.choice(class_samples[class_label])
        #             test_sample_labels.add(additional_sample)
        #
        #         # Step 4: Convert sample labels to index lists
        #         test_indices = [idx for label in test_sample_labels for idx in unique_samples[label]]
        #         train_indices = [idx for label in sample_labels if label not in test_sample_labels for idx in
        #                          unique_samples[label]]
        #
        #     else:
        #         # Treat each instance independently
        #         indices = np.arange(len(y))
        #         rng.shuffle(indices)
        #
        #         # Calculate the number of test samples
        #         num_test_samples = int(len(indices) * test_size)
        #
        #         # Split into train and test sets
        #         test_indices = indices[:num_test_samples]
        #         train_indices = indices[num_test_samples:]
        #
        #     return np.array(train_indices), np.array(test_indices)

        def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None,
                                                       group_duplicates=True):
            """
            Perform ShuffleSplit on samples while ensuring:
            - Duplicates of the same sample (including dataset origin) are kept together (if enabled).
            - Each class is represented in the test set.
            """
            rng = np.random.default_rng(random_state)

            if group_duplicates:
                unique_samples = {}  # {(origin, label): [indices]}
                class_samples = defaultdict(list)  # {class_label: [(origin, label), ...]}

                for idx, label in enumerate(y):
                    origin = self.dataset_origins[idx]
                    key = (origin, label)
                    unique_samples.setdefault(key, []).append(idx)
                    class_label = label[0]  # e.g. 'A' from 'A1'
                    class_samples[class_label].append(key)

                sample_keys = list(unique_samples.keys())
                rng.shuffle(sample_keys)

                # Step 1: Randomly select test samples
                num_test_samples = int(len(sample_keys) * test_size)
                test_sample_keys = set(sample_keys[:num_test_samples])

                # Step 2: Ensure all class labels are represented
                test_classes = {key[1][0] for key in test_sample_keys}  # e.g. 'A' from ('merlot', 'A1')
                missing_classes = [cls for cls in class_samples if cls not in test_classes]

                for class_label in missing_classes:
                    candidate_keys = class_samples[class_label]
                    additional_key = tuple(rng.choice(candidate_keys))
                    test_sample_keys.add(additional_key)

                # Step 3: Translate keys into train/test indices
                test_indices = [idx for key in test_sample_keys for idx in unique_samples[key]]
                train_indices = [idx for key in sample_keys if key not in test_sample_keys for idx in
                                 unique_samples[key]]

            else:
                # Fallback: treat each instance independently
                indices = np.arange(len(y))
                rng.shuffle(indices)
                num_test_samples = int(len(indices) * test_size)
                test_indices = indices[:num_test_samples]
                train_indices = indices[num_test_samples:]

            return np.array(train_indices), np.array(test_indices)

        def extract_category_labels(composite_labels):
            """
            Convert composite labels (e.g., 'A1', 'B9', 'C2') to category labels ('A', 'B', 'C').

            Args:
                composite_labels (array-like): List or array of composite labels.

            Returns:
                list: List of category labels.
            """
            return [re.match(r'([A-C])', label).group(1) if re.match(r'([A-C])', label) else label
                    for label in composite_labels]

        category_labels = extract_category_labels(self.labels)

        # Compute Correct Chance Accuracy
        class_counts = Counter(category_labels)
        total_samples = sum(class_counts.values())
        class_probabilities = np.array([count / total_samples for count in class_counts.values()])
        chance_accuracy = np.sum(class_probabilities ** 2)  # Sum of squared class probabilities

        results_by_origin = {}

        # Split the dataset into training (80%) and test (20%) without separating duplicates
        train_idx, test_idx = shuffle_split_without_splitting_duplicates(
            self.data, self.labels, test_size=test_size, random_state=random_seed, group_duplicates=True
        )

        X_train_full, X_test_full = self.data[train_idx], self.data[test_idx]
        y_train_full, y_test_full = np.array(category_labels)[train_idx], np.array(category_labels)[test_idx]
        dataset_origins_test = self.dataset_origins[test_idx]

        if normalize:
            X_train_full, scaler = normalize_data(X_train_full, scaler=scaler_type)
            X_test_full = scaler.transform(X_test_full)

        if use_pca:
            pca = PCA(n_components=None, svd_solver='randomized')
            pca.fit(X_train_full)
            X_train_full = pca.transform(X_train_full)
            X_test_full = pca.transform(X_test_full)

        self.classifier.fit(X_train_full, y_train_full)

        # if self.dataset_origins is not None:
        #     unique_origins = np.unique(self.dataset_origins)
        # else:
        #     unique_origins = [None]

        # Use all samples as one group if no origins or if only one origin exists
        if self.dataset_origins is None:
            unique_origins = [None]
            dataset_origins_test = None  # No filtering
        else:
            unique_origins = np.unique(self.dataset_origins)
            if len(unique_origins) == 1:
                # Ensure it's a proper list so the loop runs
                unique_origins = list(unique_origins)

        for origin in unique_origins:
            if dataset_origins_test is not None:
                origin_mask = dataset_origins_test == origin
                X_test = X_test_full[origin_mask]
                y_test = y_test_full[origin_mask]
            else:
                X_test = X_test_full
                y_test = y_test_full

            y_pred = self.classifier.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            weighted_acc = np.average(y_pred == y_test, weights=compute_sample_weight('balanced', y_test))
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred)

            results_by_origin[origin] = {
                'accuracy': acc,
                'balanced_accuracy': bal_acc,
                'weighted_accuracy': weighted_acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'confusion_matrix': cm,
            }

            if print_results:
                print(f"Results for {origin if origin is not None else 'All Data'}:")
                print(f"  Accuracy: {acc:.3f}")
                print(f"  Balanced Accuracy: {bal_acc:.3f}")
                print(f"  Weighted Accuracy: {weighted_acc:.3f}")
                print(f"  Precision: {prec:.3f}")
                print(f"  Recall: {rec:.3f}")
                print(f"  F1 Score: {f1:.3f}")
                print(f"  Confusion Matrix:\n{cm}")

        return results_by_origin

    def train_and_evaluate_balanced_target_origin(self, n_inner_repeats=50, random_seed=42,
                                                 test_size=0.2, normalize=False, scaler_type='standard',
                                                 use_pca=False, vthresh=0.97, region=None, print_results=True,
                                                 n_jobs=-1,
                                                 test_on_discarded=False, target_origin=None,
                                                 pre_split_indices=None):
        """
        Train and evaluate the classifier, reporting metrics per dataset origin.
        Supports both inner CV and test-on-discarded strategies.

        Parameters
        ----------
        target_origin : str or None
            If specified, restrict evaluation to this dataset origin only.
        """
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.decomposition import PCA
        from sklearn.utils.class_weight import compute_sample_weight
        from sklearn.model_selection import BaseCrossValidator
        from collections import defaultdict, Counter
        import numpy as np
        import re


        ############# FUNCTIONS #############
        class RepeatedLeaveOneSamplePerClassCV(BaseCrossValidator):
            def __init__(self, n_repeats=50, shuffle=True, random_state=None, use_groups=True):
                self.n_repeats = n_repeats
                self.shuffle = shuffle
                self.random_state = random_state
                self.use_groups = use_groups

            def get_n_splits(self, X, y, groups=None):
                return self.n_repeats

            def split(self, X, y):
                def get_category(label):
                    match = re.match(r'([A-C])\d+', label)
                    return match.group(1) if match else label

                rng = np.random.default_rng(self.random_state)

                if self.use_groups:
                    indices_by_sample = {}
                    for idx, label in enumerate(y):
                        indices_by_sample.setdefault(label, []).append(idx)

                    for _ in range(self.n_repeats):
                        test_indices = []
                        for category in set(map(get_category, y)):
                            class_samples = [label for label in indices_by_sample if get_category(label) == category]
                            chosen_sample = rng.choice(class_samples, size=1, replace=False)[0] if self.shuffle else \
                            class_samples[0]
                            test_indices.extend(indices_by_sample[chosen_sample])

                        test_indices = np.array(test_indices)
                        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                        yield train_indices, test_indices
                else:
                    indices_by_category = defaultdict(list)
                    for idx, label in enumerate(y):
                        category = get_category(label)
                        indices_by_category[category].append(idx)

                    for _ in range(self.n_repeats):
                        test_indices = []
                        for category, indices in indices_by_category.items():
                            chosen = rng.choice(indices, size=1, replace=False) if self.shuffle else [indices[0]]
                            test_indices.extend(chosen)

                        test_indices = np.array(test_indices)
                        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                        yield train_indices, test_indices


        def extract_category_labels(labels):
            return [re.match(r'([A-C])', lbl).group(1) if re.match(r'([A-C])', lbl) else lbl for lbl in labels]

        def process_fold(inner_train_idx, inner_val_idx, X_train_full, y_train_full, val_origins_full):
            X_train = X_train_full[inner_train_idx]
            X_val = X_train_full[inner_val_idx]
            y_train = extract_category_labels(y_train_full[inner_train_idx])
            y_val = extract_category_labels(y_train_full[inner_val_idx])
            val_origins = val_origins_full[inner_val_idx]

            if target_origin is not None:
                origin_mask = val_origins == target_origin
                X_val = X_val[origin_mask]
                y_val = np.array(y_val)[origin_mask]

            if len(y_val) == 0:
                return None

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_val = scaler.transform(X_val)
            if use_pca:
                pca = PCA(n_components=None, svd_solver='randomized')
                pca.fit(X_train)
                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                n_components = np.searchsorted(cumulative_variance, vthresh) + 1
                n_components = min(n_components, len(np.unique(y_train)))
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_val = pca.transform(X_val)

            self.classifier.fit(X_train, y_train)
            y_pred = self.classifier.predict(X_val)

            acc = np.mean(y_pred == y_val)
            bal_acc = balanced_accuracy_score(y_val, y_pred)
            w_acc = np.average(y_pred == y_val, weights=compute_sample_weight('balanced', y_val))
            prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            # cm = confusion_matrix(y_val, y_pred)
            cm = confusion_matrix(y_val, y_pred, labels=['A', 'B', 'C'])  # adjust to your full class set


            return {
                target_origin: {
                    'accuracy': acc,
                    'balanced_accuracy': bal_acc,
                    'weighted_accuracy': w_acc,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1,
                    'confusion_matrix': cm,
                }
            }
        ############# END FUNCTIONS #############

        category_labels = extract_category_labels(self.labels)
        use_groups = self.wine_kind == "press"

        if pre_split_indices is not None:
            train_idx, test_idx = pre_split_indices
        else:
            train_idx, test_idx = self.shuffle_split_without_splitting_duplicates(
                self.data, self.labels, test_size=test_size, random_state=random_seed, group_duplicates=True
            )

        X_train_full, X_test = self.data[train_idx], self.data[test_idx]
        y_train_full, y_test = self.labels[train_idx], self.labels[test_idx]
        origins_train = self.dataset_origins[train_idx] if self.dataset_origins is not None else None
        origins_test = self.dataset_origins[test_idx] if self.dataset_origins is not None else None

        if test_on_discarded:
            if normalize:
                X_train_full, scaler = normalize_data(X_train_full, scaler=scaler_type)
                X_test = scaler.transform(X_test)
            if use_pca:
                pca = PCA(n_components=None, svd_solver='randomized')
                pca.fit(X_train_full)
                X_train_full = pca.transform(X_train_full)
                X_test = pca.transform(X_test)

            self.classifier.fit(X_train_full, extract_category_labels(y_train_full))
            y_test_cat = extract_category_labels(y_test)
            y_pred = self.classifier.predict(X_test)

            results = {}
            for origin in np.unique(origins_test):
                if target_origin is not None and origin != target_origin:
                    continue
                mask = origins_test == origin
                y_true = np.array(y_test_cat)[mask]
                y_hat = np.array(y_pred)[mask]
                results[origin] = {
                    'accuracy': np.mean(y_hat == y_true),
                    'balanced_accuracy': balanced_accuracy_score(y_true, y_hat),
                    'weighted_accuracy': np.average(y_hat == y_true, weights=compute_sample_weight('balanced', y_true)),
                    'precision': precision_score(y_true, y_hat, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_hat, average='weighted', zero_division=0),
                    'f1_score': f1_score(y_true, y_hat, average='weighted', zero_division=0),
                    'confusion_matrix': confusion_matrix(y_true, y_hat),
                }
            return results

        else:
            # Initialize the cross-validation strategy
            cv = RepeatedLeaveOneSamplePerClassCV(
                n_repeats=n_inner_repeats,
                shuffle=True,
                random_state=random_seed,
                use_groups=use_groups  # keep replicates together if applicable
            )

            # Run CV in parallel: each fold calls `process_fold()` on train/val split
            split_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(process_fold)(train_idx, val_idx, X_train_full, y_train_full, origins_train)
                for train_idx, val_idx in cv.split(X_train_full, y_train_full)
            )

            # Merge metrics by origin: structure = {origin: {metric_name: [list of values across folds]}}
            merged = defaultdict(lambda: defaultdict(list))
            for res in split_results:
                if res is None:  # Some folds may not contain target_origin at all
                    continue
                for origin, metrics in res.items():
                    if target_origin is not None and origin != target_origin:
                        continue  # Skip irrelevant origins if a target is specified
                    for metric, val in metrics.items():
                        merged[origin][metric].append(val)  # Accumulate metric values

            # Compute averages across folds for each origin
            averaged_results = {}
            for origin, metrics in merged.items():
                averaged_results[origin] = {
                    metric: np.mean(values) if isinstance(values[0], (float, int)) else np.mean(values, axis=0)
                    for metric, values in metrics.items()
                }

            # Optionally print results per origin
            if print_results:
                for origin, metrics in averaged_results.items():
                    print(f"Results for {origin}:")
                    for key, val in metrics.items():
                        if isinstance(val, float):
                            print(f"  {key.replace('_', ' ').capitalize()}: {val:.3f}")

            # Return per-origin averaged results
            return averaged_results

    # def train_and_evaluate_balanced_diff_origins(self, n_inner_repeats=50, random_seed=42,
    #                                 test_size=0.2, normalize=False, scaler_type='standard',
    #                                 use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
    #                                 test_on_discarded=False,
    #                                 target_origin=None):
    #
    #     class RepeatedLeaveOneSamplePerClassCV(BaseCrossValidator):
    #         """
    #         Custom cross-validator that randomly selects one sample per class as the test set,
    #         ensuring that all replicates of the same sample stay together, or selects individual
    #         samples if standard mode is chosen. Uses composite labels to preserve replicate information.
    #
    #         Parameters
    #         ----------
    #         n_repeats : int, optional (default=50)
    #             Number of repetitions.
    #         shuffle : bool, optional (default=True)
    #             Whether to shuffle the samples before splitting.
    #         random_state : int, optional (default=None)
    #             Seed for reproducibility.
    #         use_groups : bool, optional (default=True)
    #             If True, replicates of the same sample stay together (group-like behavior).
    #             If False, selects individual samples without considering replicates.
    #
    #         Attributes
    #         ----------
    #         n_repeats : int
    #             Number of repetitions.
    #         shuffle : bool
    #             Whether to shuffle samples.
    #         random_state : int
    #             Random seed.
    #         use_groups : bool
    #             Determines whether to use group-like splitting or standard splitting.
    #         """
    #
    #         def __init__(self, n_repeats=50, shuffle=True, random_state=None, use_groups=True):
    #             self.n_repeats = n_repeats
    #             self.shuffle = shuffle
    #             self.random_state = random_state
    #             self.use_groups = use_groups
    #
    #         def get_n_splits(self, X, y, groups=None):
    #             """Returns the number of splits."""
    #             return self.n_repeats
    #
    #         def split(self, X, y):
    #             """
    #             Splits the dataset into training and test sets using either:
    #             - Group-like splitting (all replicates of the same sample stay together).
    #             - Standard splitting (individual samples are selected without considering replicates).
    #
    #             Parameters
    #             ----------
    #             X : array-like, shape (n_samples, n_features)
    #                 Feature matrix or sample labels.
    #             y : array-like, shape (n_samples,)
    #                 Composite labels (e.g., 'A1', 'B9') that preserve replicate information.
    #
    #             Yields
    #             ------
    #             train_indices : ndarray
    #                 Training indices.
    #             test_indices : ndarray
    #                 Test indices.
    #             """
    #
    #             # Step 1: Extract category from composite labels (e.g., 'A1' -> 'A')
    #             def get_category(label):
    #                 """Extracts the category (A, B, or C) from the composite label."""
    #                 match = re.match(r'([A-C])\d+', label)
    #                 return match.group(1) if match else label  # Fallback if no match
    #
    #             rng = np.random.default_rng(self.random_state)
    #
    #             # ✅ OPTION 1: Group-Like Splitting (use_groups=True)
    #             if self.use_groups:
    #                 # Group indices by composite labels
    #                 indices_by_sample = {}
    #                 for idx, label in enumerate(y):
    #                     indices_by_sample.setdefault(label, []).append(idx)
    #
    #                 # Generate splits
    #                 for _ in range(self.n_repeats):
    #                     test_indices = []
    #                     for category in set(map(get_category, y)):
    #                         # Select all samples that belong to the current category
    #                         class_samples = [label for label in indices_by_sample.keys() if
    #                                          get_category(label) == category]
    #
    #                         # Randomly choose one sample
    #                         chosen_sample = rng.choice(class_samples, size=1, replace=False)[0] if self.shuffle else \
    #                         class_samples[0]
    #
    #                         # Add all indices of the chosen sample to the test set
    #                         test_indices.extend(indices_by_sample[chosen_sample])
    #
    #                     test_indices = np.array(test_indices)
    #                     train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    #
    #                     yield train_indices, test_indices
    #
    #             # ✅ OPTION 2: Standard Splitting (use_groups=False)
    #             else:
    #                 # Collect indices by category
    #                 indices_by_category = {}
    #                 for idx, label in enumerate(y):
    #                     category = get_category(label)
    #                     indices_by_category.setdefault(category, []).append(idx)
    #
    #                 # Generate splits
    #                 for _ in range(self.n_repeats):
    #                     test_indices = []
    #                     for category, indices in indices_by_category.items():
    #                         chosen = rng.choice(indices, size=1, replace=False) if self.shuffle else [indices[0]]
    #                         test_indices.extend(chosen)
    #
    #                     test_indices = np.array(test_indices)
    #                     train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
    #
    #                     yield train_indices, test_indices
    #
    #     # def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None, group_duplicates=True):
    #     #     """
    #     #         Perform ShuffleSplit on samples while ensuring that:
    #     #         - Duplicates of the same sample are kept together (if enabled).
    #     #         - Each class is represented in the test set.
    #     #
    #     #     Args:
    #     #         X (array-like): Feature matrix or sample labels.
    #     #         y (array-like): Composite labels (e.g., ['A1', 'A1', 'B2', 'B2']).
    #     #         test_size (float): Fraction of samples to include in the test set.
    #     #         random_state (int): Random seed for reproducibility.
    #     #         group_duplicates (bool): If True, duplicates of the same sample are kept together.
    #     #                                  If False, duplicates are treated independently.
    #     #
    #     #     Returns:
    #     #         tuple: train_indices, test_indices (numpy arrays)
    #     #     """
    #     #     import numpy as np
    #     #
    #     #     rng = np.random.default_rng(random_state)
    #     #
    #     #     if group_duplicates:
    #     #         unique_samples = {}  # {sample_label: [indices]}
    #     #         class_samples = defaultdict(list)  # {class_label: [sample_label1, sample_label2, ...]}
    #     #
    #     #         for idx, label in enumerate(y):
    #     #             unique_samples.setdefault(label, []).append(idx)
    #     #             class_samples[label[0]].append(label)  # Assuming class is the first character (e.g., 'A1' -> 'A')
    #     #
    #     #         sample_labels = list(unique_samples.keys())
    #     #         rng.shuffle(sample_labels)
    #     #
    #     #         # Step 1: Randomly select test samples
    #     #         num_test_samples = int(len(sample_labels) * test_size)
    #     #         test_sample_labels = set(sample_labels[:num_test_samples])
    #     #
    #     #         # Step 2: Ensure each class is represented in the test set
    #     #         test_classes = {label[0] for label in test_sample_labels}
    #     #         missing_classes = [c for c in class_samples if c not in test_classes]
    #     #
    #     #         # Step 3: Force at least one sample from missing classes
    #     #         for class_label in missing_classes:
    #     #             additional_sample = rng.choice(class_samples[class_label])
    #     #             test_sample_labels.add(additional_sample)
    #     #
    #     #         # Step 4: Convert sample labels to index lists
    #     #         test_indices = [idx for label in test_sample_labels for idx in unique_samples[label]]
    #     #         train_indices = [idx for label in sample_labels if label not in test_sample_labels for idx in
    #     #                          unique_samples[label]]
    #     #
    #     #     else:
    #     #         # Treat each instance independently
    #     #         indices = np.arange(len(y))
    #     #         rng.shuffle(indices)
    #     #
    #     #         # Calculate the number of test samples
    #     #         num_test_samples = int(len(indices) * test_size)
    #     #
    #     #         # Split into train and test sets
    #     #         test_indices = indices[:num_test_samples]
    #     #         train_indices = indices[num_test_samples:]
    #     #
    #     #     return np.array(train_indices), np.array(test_indices)
    #
    #     def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None,
    #                                                    group_duplicates=True):
    #         """
    #         Perform ShuffleSplit on samples while ensuring:
    #         - Duplicates of the same sample (including dataset origin) are kept together (if enabled).
    #         - Each class is represented in the test set.
    #         """
    #         rng = np.random.default_rng(random_state)
    #
    #         if group_duplicates:
    #             unique_samples = {}  # {(origin, label): [indices]}
    #             class_samples = defaultdict(list)  # {class_label: [(origin, label), ...]}
    #
    #             for idx, label in enumerate(y):
    #                 origin = self.dataset_origins[idx]
    #                 key = (origin, label)
    #                 unique_samples.setdefault(key, []).append(idx)
    #                 class_label = label[0]  # e.g. 'A' from 'A1'
    #                 class_samples[class_label].append(key)
    #
    #             sample_keys = list(unique_samples.keys())
    #             rng.shuffle(sample_keys)
    #
    #             # Step 1: Randomly select test samples
    #             num_test_samples = int(len(sample_keys) * test_size)
    #             test_sample_keys = set(sample_keys[:num_test_samples])
    #
    #             # Step 2: Ensure all class labels are represented
    #             test_classes = {key[1][0] for key in test_sample_keys}  # e.g. 'A' from ('merlot', 'A1')
    #             missing_classes = [cls for cls in class_samples if cls not in test_classes]
    #
    #             for class_label in missing_classes:
    #                 candidate_keys = class_samples[class_label]
    #                 additional_key = tuple(rng.choice(candidate_keys))
    #                 test_sample_keys.add(additional_key)
    #
    #             # Step 3: Translate keys into train/test indices
    #             test_indices = [idx for key in test_sample_keys for idx in unique_samples[key]]
    #             train_indices = [idx for key in sample_keys if key not in test_sample_keys for idx in
    #                              unique_samples[key]]
    #
    #         else:
    #             # Fallback: treat each instance independently
    #             indices = np.arange(len(y))
    #             rng.shuffle(indices)
    #             num_test_samples = int(len(indices) * test_size)
    #             test_indices = indices[:num_test_samples]
    #             train_indices = indices[num_test_samples:]
    #
    #         return np.array(train_indices), np.array(test_indices)
    #
    #
    #     def process_fold(inner_train_idx, inner_val_idx, X_train_full, y_train_full, normalize, scaler_type, use_pca,
    #                      vthresh, custom_order=None):
    #         X_train = X_train_full[inner_train_idx]
    #         y_train = extract_category_labels(y_train_full[inner_train_idx])
    #         X_val = X_train_full[inner_val_idx]
    #         y_val = extract_category_labels(y_train_full[inner_val_idx])
    #
    #         if normalize:
    #             X_train, scaler = normalize_data(X_train, scaler=scaler_type)
    #             X_val = scaler.transform(X_val)
    #
    #         if use_pca:
    #             pca = PCA(n_components=None, svd_solver='randomized')
    #             pca.fit(X_train)
    #             cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    #             n_components = np.searchsorted(cumulative_variance, vthresh) + 1
    #             n_components = min(n_components, len(np.unique(y_train)))
    #             pca = PCA(n_components=n_components, svd_solver='randomized')
    #             X_train = pca.fit_transform(X_train)
    #             X_val = pca.transform(X_val)
    #
    #         self.classifier.fit(X_train, y_train)
    #         y_pred = self.classifier.predict(X_val)
    #
    #         acc = self.classifier.score(X_val, y_val)
    #         bal_acc = balanced_accuracy_score(y_val, y_pred)
    #         sw = compute_sample_weight(class_weight='balanced', y=y_val)
    #         w_acc = np.average(y_pred == y_val, weights=sw)
    #         prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    #         rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    #         f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    #         cm = confusion_matrix(y_val, y_pred, labels=custom_order if custom_order else None)
    #
    #         return acc, bal_acc, w_acc, prec, rec, f1, cm
    #
    #
    #     def extract_category_labels(composite_labels):
    #         """
    #         Convert composite labels (e.g., 'A1', 'B9', 'C2') to category labels ('A', 'B', 'C').
    #
    #         Args:
    #             composite_labels (array-like): List or array of composite labels.
    #
    #         Returns:
    #             list: List of category labels.
    #         """
    #         return [re.match(r'([A-C])', label).group(1) if re.match(r'([A-C])', label) else label
    #                 for label in composite_labels]
    #
    #
    #     if target_origin is None:
    #         raise ValueError("You must specify a target_origin if evaluating a single origin (e.g. 'merlot_2022).")
    #
    #     category_labels = extract_category_labels(self.labels)
    #
    #     # Compute Correct Chance Accuracy
    #     class_counts = Counter(category_labels)
    #     total_samples = sum(class_counts.values())
    #     class_probabilities = np.array([count / total_samples for count in class_counts.values()])
    #     chance_accuracy = np.sum(class_probabilities ** 2)  # Sum of squared class probabilities
    #
    #     results_by_origin = {}
    #
    #     # Split the dataset into training (80%) and test (20%) without separating duplicates
    #     train_idx, test_idx = shuffle_split_without_splitting_duplicates(
    #         self.data, self.labels, test_size=test_size, random_state=random_seed, group_duplicates=True
    #     )
    #
    #     X_train_full, X_test_full = self.data[train_idx], self.data[test_idx]
    #     y_train_full, y_test_full = np.array(category_labels)[train_idx], np.array(category_labels)[test_idx]
    #     dataset_origins_train = self.dataset_origins[train_idx]
    #     dataset_origins_test = self.dataset_origins[test_idx]
    #
    #     # Loop through each origin separately
    #     unique_origins = np.unique(dataset_origins_train)
    #     results_by_origin = {}
    #
    #     origin = target_origin
    #     print(f"\nEvaluating origin: {origin}")
    #
    #     origin_train_mask = dataset_origins_train == origin
    #     origin_test_mask = dataset_origins_test == origin
    #
    #     X_train_origin = X_train_full[origin_train_mask]
    #     y_train_origin = y_train_full[origin_train_mask]
    #
    #     X_test_origin = X_test_full[origin_test_mask]
    #     y_test_origin = y_test_full[origin_test_mask]
    #
    #     if test_on_discarded:
    #         # Single-train/test evaluation (existing behavior)
    #         if normalize:
    #             X_train_origin, scaler = normalize_data(X_train_origin, scaler=scaler_type)
    #             X_test_origin = scaler.transform(X_test_origin)
    #
    #         if use_pca:
    #             pca = PCA(n_components=None, svd_solver='randomized')
    #             pca.fit(X_train_origin)
    #             X_train_origin = pca.transform(X_train_origin)
    #             X_test_origin = pca.transform(X_test_origin)
    #
    #         self.classifier.fit(X_train_origin, y_train_origin)
    #         y_pred = self.classifier.predict(X_test_origin)
    #
    #         # Calculate metrics
    #         acc = accuracy_score(y_test_origin, y_pred)
    #         bal_acc = balanced_accuracy_score(y_test_origin, y_pred)
    #         prec = precision_score(y_test_origin, y_pred, average='weighted', zero_division=0)
    #         rec = recall_score(y_test_origin, y_pred, average='weighted', zero_division=0)
    #         f1 = f1_score(y_test_origin, y_pred, average='weighted', zero_division=0)
    #         cm = confusion_matrix(y_test_origin, y_pred)
    #
    #         results_by_origin[origin] = {
    #             'accuracy': acc,
    #             'balanced_accuracy': bal_acc,
    #             'precision': prec,
    #             'recall': rec,
    #             'f1_score': f1,
    #             'confusion_matrix': cm,
    #         }
    #
    #         if print_results:
    #             print(f"  Accuracy: {acc:.3f}")
    #             print(f"  Balanced Accuracy: {bal_acc:.3f}")
    #             print(f"  Precision: {prec:.3f}")
    #             print(f"  Recall: {rec:.3f}")
    #             print(f"  F1 Score: {f1:.3f}")
    #
    #     else:
    #         # Perform repeated cross-validation evaluation per origin
    #         cv = RepeatedLeaveOneSamplePerClassCV(
    #             n_repeats=n_inner_repeats,
    #             shuffle=True,
    #             random_state=random_seed,
    #             use_groups=True  # assuming you want group behavior here
    #         )
    #
    #         results = Parallel(n_jobs=n_jobs, backend='loky')(
    #             delayed(process_fold)(
    #                 inner_train_idx, inner_val_idx, X_train_origin, y_train_origin, normalize,
    #                 scaler_type, use_pca, vthresh
    #             )
    #             for inner_train_idx, inner_val_idx in cv.split(X_train_origin, y_train_origin)
    #         )
    #
    #         inner_acc, inner_bal_acc, inner_w_acc, inner_prec, inner_rec, inner_f1, inner_cm = zip(*results)
    #
    #         # Store the averaged results
    #         results_by_origin[origin] = {
    #             'accuracy': np.mean(inner_acc),
    #             'balanced_accuracy': np.mean(inner_bal_acc),
    #             'weighted_accuracy': np.mean(inner_w_acc),
    #             'precision': np.mean(inner_prec),
    #             'recall': np.mean(inner_rec),
    #             'f1_score': np.mean(inner_f1),
    #             'confusion_matrix': np.mean(inner_cm, axis=0),
    #         }
    #
    #         if print_results:
    #             print(f"  Inner CV Averages for origin {origin}:")
    #             print(f"    Accuracy: {results_by_origin[origin]['accuracy']:.3f}")
    #             print(f"    Balanced Accuracy: {results_by_origin[origin]['balanced_accuracy']:.3f}")
    #             print(f"    Precision: {results_by_origin[origin]['precision']:.3f}")
    #             print(f"    Recall: {results_by_origin[origin]['recall']:.3f}")
    #             print(f"    F1 Score: {results_by_origin[origin]['f1_score']:.3f}")
    #
    #     return results_by_origin



    def train_and_evaluate_balanced_tic_tis(self, tics, tiss, n_inner_repeats=50, random_seed=42,
                                    test_size=0.2, normalize=False, scaler_type='standard',
                                    use_pca=False, vthresh=0.97, region=None, print_results=True,
                                    normalize_before_concat=True):
        """
        Train and evaluate the classifier using repeated outer stratified splits. For each outer repetition,
        the training set is further split using RepeatedLeaveOneFromEachClassCV and validation metrics are computed.
        The outer test set is ignored; only the training set is used for inner CV.

        Parameters
        ----------
        n_inner_repeats : int
            Number of inner CV repeats (passed to RepeatedLeaveOneFromEachClassCV).
        random_seed : int, optional
            Seed for reproducibility.
        test_size : float, optional
            Fraction of data to hold out (unused in metric computations).
        normalize : bool, optional
            Whether to normalize the data.
        scaler_type : str, optional
            Which scaler to use ('standard' or 'minmax').
        use_pca : bool, optional
            Whether to apply PCA.
        vthresh : float, optional
            Variance threshold for PCA.
        region : str, optional
            Determines custom ordering for confusion matrix.
        batch_size, num_epochs, learning_rate : not used in this snippet.

        Returns
        -------
        dict
            A dictionary with the average inner validation metrics across all outer repeats.
        """
        from sklearn.model_selection import StratifiedShuffleSplit
        from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        from sklearn.decomposition import PCA
        from sklearn.utils.class_weight import compute_sample_weight

        class RepeatedLeaveOneSamplePerClassCV(BaseCrossValidator):
            """
            Custom cross-validator that randomly selects one sample per class as the test set,
            ensuring that all replicates of the same sample stay together, or selects individual
            samples if standard mode is chosen. Uses composite labels to preserve replicate information.

            Parameters
            ----------
            n_repeats : int, optional (default=50)
                Number of repetitions.
            shuffle : bool, optional (default=True)
                Whether to shuffle the samples before splitting.
            random_state : int, optional (default=None)
                Seed for reproducibility.
            use_groups : bool, optional (default=True)
                If True, replicates of the same sample stay together (group-like behavior).
                If False, selects individual samples without considering replicates.

            Attributes
            ----------
            n_repeats : int
                Number of repetitions.
            shuffle : bool
                Whether to shuffle samples.
            random_state : int
                Random seed.
            use_groups : bool
                Determines whether to use group-like splitting or standard splitting.
            """

            def __init__(self, n_repeats=50, shuffle=True, random_state=None, use_groups=True):
                self.n_repeats = n_repeats
                self.shuffle = shuffle
                self.random_state = random_state
                self.use_groups = use_groups

            def get_n_splits(self, X, y, groups=None):
                """Returns the number of splits."""
                return self.n_repeats

            def split(self, X, y):
                """
                Splits the dataset into training and test sets using either:
                - Group-like splitting (all replicates of the same sample stay together).
                - Standard splitting (individual samples are selected without considering replicates).

                Parameters
                ----------
                X : array-like, shape (n_samples, n_features)
                    Feature matrix or sample labels.
                y : array-like, shape (n_samples,)
                    Composite labels (e.g., 'A1', 'B9') that preserve replicate information.

                Yields
                ------
                train_indices : ndarray
                    Training indices.
                test_indices : ndarray
                    Test indices.
                """

                # Step 1: Extract category from composite labels (e.g., 'A1' -> 'A')
                def get_category(label):
                    """Extracts the category (A, B, or C) from the composite label."""
                    match = re.match(r'([A-C])\d+', label)
                    return match.group(1) if match else label  # Fallback if no match

                rng = np.random.default_rng(self.random_state)

                # ✅ OPTION 1: Group-Like Splitting (use_groups=True)
                if self.use_groups:
                    # Group indices by composite labels
                    indices_by_sample = {}
                    for idx, label in enumerate(y):
                        indices_by_sample.setdefault(label, []).append(idx)

                    # Generate splits
                    for _ in range(self.n_repeats):
                        test_indices = []
                        for category in set(map(get_category, y)):
                            # Select all samples that belong to the current category
                            class_samples = [label for label in indices_by_sample.keys() if
                                             get_category(label) == category]

                            # Randomly choose one sample
                            chosen_sample = rng.choice(class_samples, size=1, replace=False)[0] if self.shuffle else \
                                class_samples[0]

                            # Add all indices of the chosen sample to the test set
                            test_indices.extend(indices_by_sample[chosen_sample])

                        test_indices = np.array(test_indices)
                        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

                        yield train_indices, test_indices

                # ✅ OPTION 2: Standard Splitting (use_groups=False)
                else:
                    # Collect indices by category
                    indices_by_category = {}
                    for idx, label in enumerate(y):
                        category = get_category(label)
                        indices_by_category.setdefault(category, []).append(idx)

                    # Generate splits
                    for _ in range(self.n_repeats):
                        test_indices = []
                        for category, indices in indices_by_category.items():
                            chosen = rng.choice(indices, size=1, replace=False) if self.shuffle else [indices[0]]
                            test_indices.extend(chosen)

                        test_indices = np.array(test_indices)
                        train_indices = np.setdiff1d(np.arange(len(y)), test_indices)

                        yield train_indices, test_indices

        def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None, group_duplicates=True):
            """
                Perform ShuffleSplit on samples while ensuring that:
                - Duplicates of the same sample are kept together (if enabled).
                - Each class is represented in the test set.

            Args:
                X (array-like): Feature matrix or sample labels.
                y (array-like): Composite labels (e.g., ['A1', 'A1', 'B2', 'B2']).
                test_size (float): Fraction of samples to include in the test set.
                random_state (int): Random seed for reproducibility.
                group_duplicates (bool): If True, duplicates of the same sample are kept together.
                                         If False, duplicates are treated independently.

            Returns:
                tuple: train_indices, test_indices (numpy arrays)
            """
            import numpy as np

            rng = np.random.default_rng(random_state)

            if group_duplicates:
                unique_samples = {}  # {sample_label: [indices]}
                class_samples = defaultdict(list)  # {class_label: [sample_label1, sample_label2, ...]}

                for idx, label in enumerate(y):
                    unique_samples.setdefault(label, []).append(idx)
                    class_samples[label[0]].append(label)  # Assuming class is the first character (e.g., 'A1' -> 'A')

                sample_labels = list(unique_samples.keys())
                rng.shuffle(sample_labels)

                # Step 1: Randomly select test samples
                num_test_samples = int(len(sample_labels) * test_size)
                test_sample_labels = set(sample_labels[:num_test_samples])

                # Step 2: Ensure each class is represented in the test set
                test_classes = {label[0] for label in test_sample_labels}
                missing_classes = [c for c in class_samples if c not in test_classes]

                # Step 3: Force at least one sample from missing classes
                for class_label in missing_classes:
                    additional_sample = rng.choice(class_samples[class_label])
                    test_sample_labels.add(additional_sample)

                # Step 4: Convert sample labels to index lists
                test_indices = [idx for label in test_sample_labels for idx in unique_samples[label]]
                train_indices = [idx for label in sample_labels if label not in test_sample_labels for idx in
                                 unique_samples[label]]

            else:
                # Treat each instance independently
                indices = np.arange(len(y))
                rng.shuffle(indices)

                # Calculate the number of test samples
                num_test_samples = int(len(indices) * test_size)

                # Split into train and test sets
                test_indices = indices[:num_test_samples]
                train_indices = indices[num_test_samples:]

            return np.array(train_indices), np.array(test_indices)

        def extract_category_labels(composite_labels):
            """
            Convert composite labels (e.g., 'A1', 'B9', 'C2') to category labels ('A', 'B', 'C').

            Args:
                composite_labels (array-like): List or array of composite labels.

            Returns:
                list: List of category labels.
            """
            return [re.match(r'([A-C])', label).group(1) if re.match(r'([A-C])', label) else label
                    for label in composite_labels]

        category_labels = extract_category_labels(self.labels)

        # Compute class distribution
        class_counts = Counter(category_labels)
        total_samples = sum(class_counts.values())

        # Compute Correct Chance Accuracy (Taking Class Distribution into Account)
        class_probabilities = np.array([count / total_samples for count in class_counts.values()])
        chance_accuracy = np.sum(class_probabilities ** 2)  # Sum of squared class probabilities

        # Set up a custom order for the confusion matrix if a region is specified.
        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
            custom_order = None

        # Initialize accumulators for outer-repetition averaged metrics.
        outer_accuracy = []
        outer_balanced_accuracy = []
        outer_weighted_accuracy = []
        outer_precision = []
        outer_recall = []
        outer_f1 = []
        outer_cm = []

        # Use a reproducible RNG.
        if random_seed is None:
            random_seed = np.random.randint(0, int(1e6))
        rng = np.random.default_rng(random_seed)

        # Choose whether to use groups based on wine type.
        use_groups = True if self.wine_kind == "press" else False
        # use_groups = False

        # Outer split: use StratifiedShuffleSplit to split the data.
        # # sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.integers(0, int(1e6)))
        # sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.integers(0, int(1e6)))
        # train_idx, _ = next(sss.split(self.data, self.labels))
        train_idx, test_idx = shuffle_split_without_splitting_duplicates(
            tics, self.labels, test_size=test_size, random_state=rng.integers(0, int(1e6)),
            group_duplicates=use_groups
        )
        X_train_full_tics, X_test_tics = tics[train_idx], tics[test_idx]
        X_train_full_tiss, X_test_tiss = tiss[train_idx], tiss[test_idx]
        y_train_full, y_test = self.labels[train_idx], self.labels[test_idx]

        if normalize_before_concat:   # normalise separately before concatenation
            X_train_full_tics, scaler_tic = normalize_data(X_train_full_tics, scaler=scaler_type)
            X_test_tics = scaler_tic.transform(X_test_tics)

            X_train_full_tiss, scaler_tis = normalize_data(X_train_full_tiss, scaler=scaler_type)
            X_test_tiss = scaler_tis.transform(X_test_tiss)

            X_train_full = np.concatenate((X_train_full_tics, X_train_full_tiss), axis=1)
            X_test = np.concatenate((X_test_tics, X_test_tiss), axis=1)
            # X_train_full = X_train_full_tics
            # X_test = X_test_tics

        if not normalize_before_concat:  # concatenate before normalising
            X_train_full = np.concatenate((X_train_full_tics, X_train_full_tiss), axis=1)
            X_test = np.concatenate((X_test_tics, X_test_tiss), axis=1)
            # X_train_full, scaler = normalize_data(X_train_full, scaler=scaler_type)
            # X_test = scaler.transform(X_test)


        self.classifier.fit(X_train_full, extract_category_labels(y_train_full))
        y_pred = self.classifier.predict(X_test)
        y_test = extract_category_labels(y_test)

        outer_accuracy.append(accuracy_score(y_test, y_pred))
        outer_balanced_accuracy.append(balanced_accuracy_score(y_test, y_pred))
        outer_weighted_accuracy.append(
            np.average(y_pred == y_test, weights=compute_sample_weight('balanced', y_test)))
        outer_precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
        outer_recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
        outer_f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        outer_cm.append(confusion_matrix(y_test, y_pred))

        if print_results:
            print(f"  Test on discarded data metrics:")
            print(f"    Accuracy: {outer_accuracy[-1]:.3f}")
            print(f"    Balanced Accuracy: {outer_balanced_accuracy[-1]:.3f}")
            print(f"    Weighted Accuracy: {outer_weighted_accuracy[-1]:.3f}")
            print(f"    Precision: {outer_precision[-1]:.3f}")
            print(f"    Recall: {outer_recall[-1]:.3f}")
            print(f"    F1 Score: {outer_f1[-1]:.3f}")


        # Compute the averaged confusion matrix across all repetitions
        overall_cm = np.mean(outer_cm, axis=0)

        # Normalize confusion matrix row-wise (true label-wise)
        overall_cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1, keepdims=True)

        overall_results = {
            'chance_accuracy': chance_accuracy,
            'overall_accuracy': np.mean(outer_accuracy),
            'overall_balanced_accuracy': np.mean(outer_balanced_accuracy),
            'overall_weighted_accuracy': np.mean(outer_weighted_accuracy),
            'overall_precision': np.mean(outer_precision),
            'overall_recall': np.mean(outer_recall),
            'overall_f1_score': np.mean(outer_f1),
            'overall_confusion_matrix': overall_cm,
            'overall_confusion_matrix_normalized': overall_cm_normalized,
        }

        if print_results:
            print("\nFinal Results:")
            print(f"  Overall Accuracy: {overall_results['overall_accuracy']:.3f}")
            print(f"  Overall Balanced Accuracy: {overall_results['overall_balanced_accuracy']:.3f}")
            print(f"  Overall Weighted Accuracy: {overall_results['overall_weighted_accuracy']:.3f}")
            print(f"  Overall Precision: {overall_results['overall_precision']:.3f}")
            print(f"  Overall Recall: {overall_results['overall_recall']:.3f}")
            print(f"  Overall F1 Score: {overall_results['overall_f1_score']:.3f}")
            # print("Overall Mean Confusion Matrix:")
            # print(overall_results['overall_confusion_matrix'])
            # print(overall_results['overall_confusion_matrix_normalized'])

        return overall_results


    def train_and_evaluate_greedy_add_ranked(self, num_repeats=10, num_outer_repeats=1, n_inner_repeats=50,
                                         random_seed=42, test_size=0.2, normalize=False, scaler_type='standard',
                                         use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                         num_top_channels=50, feature_type="concatenated"):
        """
        Perform ranked greedy channel selection, starting with the best single channel and incrementally adding others.
        Dynamically updates the performance plot after each repeat.

        Parameters:
        ----------
        feature_type : str
            - "concatenated" (default) → Uses raw m/z channels.
            - "sum" → Sums selected channels into a single time-series vector per sample.
        """
        all_test_accuracies = []
        first_five_chosen_channels = []  # Track first 5 selected channels in each repeat

        cls_data = self.data.copy()  # Shape: (num_samples, num_timepoints, num_channels)
        num_samples, num_timepoints, num_channels = cls_data.shape
        labels = self.labels

        def compute_features(channels):
            """Compute features based on the chosen feature type."""
            print(f"Computing features for channels: {channels}")

            if feature_type == "concatenated":
                concatenated_data = np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
                return concatenated_data
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

            # if feature_type == "concatenated":
            #     return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            # elif feature_type == "sum":
            #     return np.sum(cls_data[:, :, channels],
            #                   axis=2)  # Sum across selected channels → shape (num_samples, num_timepoints)
            # else:
            #     raise ValueError("Invalid feature_type. Use 'concatenated' or 'sum'.")

        plt.ion()  # Enable interactive mode
        # fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plt.subplots()

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            # Step 1: Evaluate Accuracy for Each Individual Channel
            mean_accuracies = []
            for ch_idx in range(num_channels):
                feature_matrix = compute_features([ch_idx])  # Evaluate one channel at a time
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels
                                 )
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    n_jobs=n_jobs,
                    test_on_discarded=False
                )
                print(f'Channel = {ch_idx}')
                mean_accuracies.append(results['overall_balanced_accuracy'])

            # Step 2: Sort Channels by Accuracy
            unique_accuracies = np.unique(mean_accuracies)
            sorted_indices = []
            rng = np.random.default_rng(random_seed + repeat_idx)
            for acc in sorted(unique_accuracies, reverse=True):
                tied_channels = [idx for idx, val in enumerate(mean_accuracies) if val == acc]
                rng.shuffle(tied_channels)  # Shuffle tied channels randomly
                sorted_indices.extend(tied_channels)

            # Step 3: Incrementally Add Top Channels and Evaluate
            incremental_accuracies = []
            selected_indices = []
            for n in range(1, min(num_top_channels, len(sorted_indices)) + 1):
                selected_indices.append(sorted_indices[n - 1])  # Add next best channel
                feature_matrix = compute_features(selected_indices)  # Recompute features

                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels
                                 )
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    n_jobs=n_jobs,
                    test_on_discarded=True
                )
                incremental_accuracies.append(results['overall_balanced_accuracy'])

                if print_results:
                    print(f"After adding {n} channel(s): Test Accuracy = {results['overall_balanced_accuracy']:.3f}")

            first_five_chosen_channels.append(selected_indices[:5])
            all_test_accuracies.append(incremental_accuracies)

            # Compute Mean and Standard Deviation Across Repeats
            mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
            std_test_accuracies = np.std(all_test_accuracies, axis=0)

            # --- Dynamic Plot Update ---
            ax.clear()  # Clear the figure before replotting
            x = range(1, len(mean_test_accuracies) + 1)

            ax.plot(x, mean_test_accuracies, '-o', label='Mean Accuracy')
            ax.fill_between(x, mean_test_accuracies - std_test_accuracies,
                            mean_test_accuracies + std_test_accuracies, alpha=0.3, label='± 1 Std Dev')

            ax.set_xlabel("Number of Best Channels Added")
            ax.set_ylabel("Mean Balanced Accuracy")
            ax.set_title(f"Performance Trend with Greedy Ranked Add ({feature_type}; {repeat_idx + 1} / {num_repeats})")
            # ax.text(0.05, 0.95, f"Repeat {repeat_idx + 1}/{num_repeats}", transform=ax.transAxes,
            #         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
            ax.grid(True)
            ax.legend()

            plt.pause(0.5)  # Pause to update the figure

        # Keep the final plot open after all repeats
        plt.ioff()  # Disable interactive mode
        plt.show(block=True)

        # Save histogram data to CSV
        with open("/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_add_ranked_merlot.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Channel Index"])
            for ch in first_five_chosen_channels:
                writer.writerow([ch])

        # Plot histogram of channels selected in first 5 steps
        plt.figure(figsize=(10, 6))
        plt.hist(first_five_chosen_channels,
                 bins=np.arange(min(first_five_chosen_channels), max(first_five_chosen_channels) + 2) - 0.5,
                 edgecolor='black')
        plt.xlabel("Channel Index")
        plt.ylabel("Number of Repeats")
        plt.title("Histogram of Channels Chosen in First 5 Steps Across Repeats")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show(block=True)

        print(f"Final Mean Accuracy: {np.mean(mean_test_accuracies):.3f} ± {np.mean(std_test_accuracies):.3f}")

        return mean_test_accuracies, std_test_accuracies



    def train_and_evaluate_greedy_remove_ranked(self, num_repeats=10, n_inner_repeats=50,
                                                random_seed=42, test_size=0.2, normalize=False,
                                                scaler_type='standard', use_pca=False, vthresh=0.97,
                                                region=None, print_results=True, n_jobs=-1,
                                                num_min_channels=1, feature_type="concatenated"):
        """
        Perform greedy ranked channel removal starting from all channels.

        Args:
            num_repeats (int): Number of outer repetitions.
            n_inner_repeats (int): Repeats for inner evaluation.
            random_seed (int): Seed for reproducibility.
            test_size (float): Test split ratio.
            normalize (bool): Whether to normalize features.
            scaler_type (str): Type of scaler ('standard', etc).
            use_pca (bool): Whether to apply PCA.
            vthresh (float): Variance threshold for PCA.
            region (str or None): Region specifier.
            print_results (bool): Whether to print progress.
            n_jobs (int): Number of parallel jobs.
            num_min_channels (int): Minimum number of channels to stop removing.
            feature_type (str): Feature representation.
        """
        all_test_accuracies = []
        # final_removed_channels = []  # Track the last 5 removed channels per repeat
        final_remaining_channels = []  # Track the last 5 remaining channels per repeat
        cls_data = self.data.copy()
        num_samples, num_timepoints, num_channels = cls_data.shape
        labels = self.labels

        def compute_features(channels):
            if feature_type == "concatenated":
                return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type")

        plt.ion()
        fig, ax = plt.subplots()

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")
            rng = np.random.default_rng(random_seed + repeat_idx)
            mean_accuracies = []
            for ch_idx in range(num_channels):
                feature_matrix = compute_features([ch_idx])
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels
                                 )
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats, random_seed=random_seed + repeat_idx,
                    test_size=test_size, normalize=normalize, scaler_type=scaler_type, use_pca=use_pca,
                    vthresh=vthresh, region=region, print_results=False, n_jobs=n_jobs, test_on_discarded=False)
                mean_accuracies.append(results['overall_balanced_accuracy'])

            unique_accuracies = np.unique(mean_accuracies)
            sorted_indices = []
            for acc in sorted(unique_accuracies):
                tied_channels = [idx for idx, val in enumerate(mean_accuracies) if val == acc]
                rng.shuffle(tied_channels)
                sorted_indices.extend(tied_channels)

            selected_indices = list(sorted_indices)  # Start with all channels
            incremental_accuracies = []
            removed_channels = []


            for n in range(num_channels, num_min_channels - 1, -1):
                feature_matrix = compute_features(selected_indices)
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels
                                 )
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats, random_seed=random_seed + repeat_idx,
                    test_size=test_size, normalize=normalize, scaler_type=scaler_type, use_pca=use_pca,
                    vthresh=vthresh, region=region, print_results=False, n_jobs=n_jobs, test_on_discarded=True)
                incremental_accuracies.append(results['overall_balanced_accuracy'])

                if print_results:
                    print(f"Remaining {n} channel(s): Test Accuracy = {results['overall_balanced_accuracy']:.3f}")

                # Corrected code to save when exactly 5 channels remain:
                if len(selected_indices) == 5:
                    final_remaining_channels.append(selected_indices.copy())

                if n > num_min_channels:
                    removed_channels.append(selected_indices[0])
                    selected_indices.pop(0)  # Remove worst-ranked remaining channel

            # final_removed_channels.append(removed_channels[-5:])
            all_test_accuracies.append(incremental_accuracies)

            # Dynamic plot update
            ax.clear()
            x = range(0, num_channels - num_min_channels + 1)
            mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
            std_test_accuracies = np.std(all_test_accuracies, axis=0)
            ax.plot(x, mean_test_accuracies, '-o', label='Mean Accuracy')
            ax.fill_between(x, mean_test_accuracies - std_test_accuracies,
                            mean_test_accuracies + std_test_accuracies, alpha=0.3, label='± 1 Std Dev')
            ax.set_xlabel("Number of Worst Channels Removed")
            ax.set_ylabel("Mean Balanced Accuracy")
            ax.set_title(f"Performance Trend with Greedy Ranked Remove ({feature_type}); Repeat {repeat_idx + 1}/{num_repeats}")
            ax.grid(True)
            ax.legend()
            plt.pause(0.5)

        plt.ioff()
        plt.show(block=True)

        # Save histogram data to CSV
        with open("/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_ranked_merlot.csv",
                  'w', newline=''
                  ) as file:
            writer = csv.writer(file)
            writer.writerow(["Channel Index"])
            # writer.writerows([[ch] for ch in final_removed_channels])
            for channels in final_remaining_channels:
                writer.writerow([str(channels)])  # Convert list explicitly to string

        # Plot histogram
        plt.figure(figsize=(10, 6))
        # plt.hist(final_removed_channels,
        #          bins=np.arange(min(final_removed_channels), max(final_removed_channels) + 2) - 0.5,
        #          edgecolor='black')
        plt.hist(final_remaining_channels,
                 bins=np.arange(min(final_remaining_channels), max(final_remaining_channels) + 2) - 0.5,
                 edgecolor='black')
        plt.xlabel("Channel Index")
        plt.ylabel("Number of Occurrences")
        plt.title("Histogram of Last 5 Removed Channels Across Repeats")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show(block=True)

        print(f"Final Mean Accuracy: {np.mean(mean_test_accuracies):.3f} ± {np.mean(std_test_accuracies):.3f}")

        return mean_test_accuracies, std_test_accuracies



    def train_and_evaluate_greedy_add(self, num_repeats=10, num_outer_repeats=1, n_inner_repeats=50,
                                      random_seed=42, test_size=0.2, normalize=False, scaler_type='standard',
                                      use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                      feature_type="concatenated"):
        """
        Perform greedy channel addition with an option to use either:
          - "concatenated": Raw m/z channels concatenated as features.
          - "tic_tis": Computes and concatenates TIC (Total Ion Chromatogram) and TIS (Total Ion Spectrum).

        Starts with a single best-performing channel and adds channels iteratively to maximize accuracy.
        """

        all_test_accuracies = []
        first_five_chosen_channels = []  # Track channels chosen in the first 5 steps


        def compute_features(channels):
            """ Compute features based on chosen feature type. """
            if feature_type == "concatenated":
                concatenated_data = np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
                return concatenated_data
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls_data = self.data.copy()
            num_samples, num_timepoints, num_channels = cls_data.shape
            labels = self.labels

            # Start with no channels, then add iteratively
            available_indices = list(range(num_channels))
            selected_indices = []
            incremental_accuracies = []

            for step in range(min(num_channels, 140 + 1)):
                if not available_indices:
                    print("⚠️ No more channels left to add. Stopping early.")
                    break

                # Evaluate the accuracy of adding each channel
                def evaluate_channel(ch_idx):
                    print(f'Step {step}; Adding Channel = {ch_idx}')
                    temp_indices = selected_indices + [ch_idx]
                    feature_matrix = compute_features(temp_indices)

                    cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                     year_labels=self.year_labels)
                    results = cls.train_and_evaluate_balanced(
                        n_inner_repeats=n_inner_repeats,
                        random_seed=random_seed + repeat_idx,
                        test_size=test_size,
                        normalize=normalize,
                        scaler_type=scaler_type,
                        use_pca=use_pca,
                        vthresh=vthresh,
                        region=region,
                        print_results=False,
                        n_jobs=1,
                        test_on_discarded=False
                    )
                    return ch_idx, results['overall_balanced_accuracy']

                # Parallel execution to test adding each channel
                validation_accuracies = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(evaluate_channel)(ch_idx) for ch_idx in available_indices)

                # Select the best channel to add
                max_accuracy = max(validation_accuracies, key=lambda x: x[1])[1]
                candidates = [ch_idx for ch_idx, acc in validation_accuracies if acc == max_accuracy]
                best_channel_to_add = random.choice(candidates)
                selected_indices.append(best_channel_to_add)
                available_indices.remove(best_channel_to_add)

                if len(selected_indices) == 5:
                    first_five_chosen_channels.append(
                        list(selected_indices))  # Store only once when 5 channels have been added

                # Compute accuracy after adding the best channel
                feature_matrix = compute_features(selected_indices)
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels)
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    n_jobs=n_jobs,
                    test_on_discarded=True
                )

                incremental_accuracies.append(results['overall_balanced_accuracy'])

                if print_results:
                    print(
                        f"After adding {step + 1} channel(s): Test Accuracy = {results['overall_balanced_accuracy']:.3f}")

                plt.pause(0.1)
            all_test_accuracies.append(incremental_accuracies)

            # Dynamic plot after each repeat
            plt.ion()
            plt.clf()
            mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
            std_test_accuracies = np.std(all_test_accuracies, axis=0)

            x = range(len(mean_test_accuracies))
            plt.plot(x, mean_test_accuracies, '-o', label='Mean Accuracy')
            plt.fill_between(x, mean_test_accuracies - std_test_accuracies, mean_test_accuracies + std_test_accuracies,
                             alpha=0.3, label='± 1 Std Dev')

            plt.xlabel("Number of Channels Added")
            plt.ylabel("Mean Balanced Accuracy")
            plt.title(f"Performance Trend with Greedy Add ({feature_type}; {repeat_idx + 1} / {num_repeats})")
            plt.grid(True)
            plt.legend()
            if repeat_idx == num_repeats - 1:
                plt.show(block=True)
            else:
                plt.pause(1)

        # Save histogram data to file
        with open("/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_add_wine_type.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Channel Index"])
            writer.writerows([[ch] for ch in first_five_chosen_channels])

        # Plot histogram of the first 5 chosen channels across repeats
        plt.figure(figsize=(10, 6))
        plt.hist(first_five_chosen_channels, bins=np.arange(min(first_five_chosen_channels), max(first_five_chosen_channels) + 1) - 0.5, edgecolor='black')
        plt.xlabel("Channel Index")
        plt.ylabel("Number of Repeats")
        plt.title("Histogram of Channels Chosen in the First 5 Steps Across Repeats")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show(block=True)

        # Final Summary
        print(f"Final Mean Accuracy: {np.mean(mean_test_accuracies):.3f} ± {np.mean(std_test_accuracies):.3f}")

        return mean_test_accuracies, std_test_accuracies



    def train_and_evaluate_greedy_remove(self, num_repeats=10, num_outer_repeats=1, n_inner_repeats=50,
                                         random_seed=42, test_size=0.2, normalize=False, scaler_type='standard',
                                         use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                         feature_type="concatenated"):
        """
        Perform greedy channel removal with an option to use either:
          - "concatenated": Raw m/z channels concatenated as features.
          - "tic_tis": Computes and concatenates TIC (Total Ion Chromatogram) and TIS (Total Ion Spectrum).

        Parameters:
        ----------
        feature_type : str
            - "concatenated" (default) → Uses raw m/z channels.
            - "tic_tis" → Uses TIC (sum over m/z) and TIS (sum over time).
        """

        all_test_accuracies = []
        final_channel_distributions = []  # Track channels left in last 5 steps

        def compute_features(channels):
            """ Compute features based on chosen feature type. """
            if feature_type == "concatenated":
                try:
                    # Concatenate raw selected channels
                    concatenated_data = np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
                except Exception as e:
                    print(f"An error occurred while plotting: {e}")
                return concatenated_data
            elif feature_type == "tic_tis":
                # Compute TIC (sum over m/z) and TIS (sum over time)
                tic = np.sum(cls_data[:, :, channels], axis=2)  # (num_samples, num_timepoints)
                tis = np.sum(cls_data[:, :, channels], axis=1)  # (num_samples, num_channels)
                return np.hstack([tic, tis])  # Combine TIC and TIS
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

        # Evaluate the accuracy of removing each channel
        def evaluate_channel(ch_idx):
            print(f'Step {step}; Channel = {ch_idx}')
            temp_indices = [idx for idx in remaining_indices if idx != ch_idx]
            if not temp_indices:  # If no channels remain, skip this iteration
                print(f"⚠️ No remaining channels after removing {ch_idx}. Skipping evaluation.")
                return ch_idx, None  # Return None to avoid breaking max() call later

            feature_matrix = compute_features(temp_indices)

            if feature_matrix is None or feature_matrix.size == 0:  # Check if feature matrix is empty
                print(f"⚠️ Feature matrix is empty after removing {ch_idx}. Skipping.")
                return ch_idx, None

            cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                             year_labels=self.year_labels)
            results = cls.train_and_evaluate_balanced(
                n_inner_repeats=n_inner_repeats,
                random_seed=random_seed + repeat_idx,
                test_size=test_size,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                region=region,
                print_results=False,
                n_jobs=1,
                test_on_discarded=False
            )
            return ch_idx, results['overall_balanced_accuracy']

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls_data = self.data.copy()  # Shape: (num_samples, num_timepoints, num_channels)
            num_samples, num_timepoints, num_channels = cls_data.shape
            labels = self.labels

            # Start with all channels
            remaining_indices = list(range(num_channels))
            incremental_accuracies = [None]  # Store accuracy before removing any channels

            for step in range(min(num_channels, 137 + 1)):  # Limit number of removals for efficiency

                if not remaining_indices:  # Prevent issues when all channels are removed
                    print("⚠️ No more channels left to remove. Stopping early.")
                    break

                # Compute current accuracy before removing any channel this iteration
                feature_matrix = compute_features(remaining_indices)
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels)
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    n_jobs=n_jobs,
                    test_on_discarded=True
                )

                # Parallel execution to test removing each channel
                validation_accuracies = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(evaluate_channel)(ch_idx) for ch_idx in remaining_indices)

                # Store accuracy at current step (before removal)
                if incremental_accuracies[0] is None:
                    incremental_accuracies[0] = results['overall_balanced_accuracy']
                else:
                    incremental_accuracies.append(results['overall_balanced_accuracy'])

                # Remove the channel whose removal leads to the highest validation accuracy
                max_accuracy = max(validation_accuracies, key=lambda x: x[1])[1]
                candidates = [ch_idx for ch_idx, acc in validation_accuracies if acc == max_accuracy]
                best_channel_to_remove = random.choice(candidates)  # Randomly select one of the best
                remaining_indices.remove(best_channel_to_remove)

                # Track the last 5 steps' remaining channels
                if len(remaining_indices) == 5:
                # if len(remaining_indices) == 10:
                    final_channel_distributions.append(
                        list(remaining_indices))  # Only store when there are exactly 5 left

                if print_results:
                    print(
                        f"After removing {step + 1} channel(s): Test Accuracy = {results['overall_balanced_accuracy']:.3f}")

            all_test_accuracies.append(incremental_accuracies)

            # Dynamic plot after each repeat
            plt.ion()  # Enable interactive mode
            plt.clf()  # Clear the current figure
            mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
            std_test_accuracies = np.std(all_test_accuracies, axis=0)

            x = range(len(mean_test_accuracies))
            plt.plot(x, mean_test_accuracies, '-o', label='Mean Accuracy')
            plt.fill_between(x, mean_test_accuracies - std_test_accuracies, mean_test_accuracies + std_test_accuracies,
                             alpha=0.3, label='± 1 Std Dev')

            plt.xlabel("Number of Channels Removed")
            plt.ylabel("Mean Balanced Accuracy")
            plt.title(f"Performance Trend with Greedy Remove ({feature_type}; {repeat_idx + 1} / {num_repeats})")
            plt.grid(True)
            plt.legend()
            if repeat_idx == num_repeats - 1:
                plt.show(block=True)  # Keep the final plot open without closing the program
            else:
                plt.pause(1)  # Pause briefly to allow the plot to update

        # Print distribution of remaining channels
        print("\nDistribution of Remaining Channels in the Last 5 Steps:")
        unique_channels, counts = np.unique(final_channel_distributions, return_counts=True)
        for ch, count in zip(unique_channels, counts):
            print(f"Channel {ch}: {count} occurrences")


        # Save histogram data to file
        with open("/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_wine_type.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Channel Index"])
            writer.writerows([[ch] for ch in final_channel_distributions])

        # Plot histogram of remaining channels in the last 5 steps across repeats
        plt.figure(figsize=(10, 6))
        plt.hist(final_channel_distributions, bins=np.arange(min(final_channel_distributions), max(final_channel_distributions) + 1) - 0.5, edgecolor='black')
        plt.xlabel("Channel Index")
        plt.ylabel("Number of Repeats")
        plt.title("Histogram of Channels Remaining in the Last 5 Steps Across Repeats")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show(block=True)

        # Final Summary
        print(f"Final Mean Accuracy: {np.mean(mean_test_accuracies):.3f} ± {np.mean(std_test_accuracies):.3f}")

        return mean_test_accuracies, std_test_accuracies


    def train_and_evaluate_greedy_remove_diff_origins(self, num_repeats=10, n_inner_repeats=50,
                                         random_seed=42, test_size=0.2, normalize=False, scaler_type='standard',
                                         use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                         feature_type="concatenated",
                                         dataset_origins=None,
                                         target_origin=None):
        """
        Perform greedy channel removal with an option to use either:
          - "concatenated": Raw m/z channels concatenated as features.
          - "tic_tis": Computes and concatenates TIC (Total Ion Chromatogram) and TIS (Total Ion Spectrum).

        Parameters:
        ----------
        feature_type : str
            - "concatenated" (default) → Uses raw m/z channels.
            - "tic_tis" → Uses TIC (sum over m/z) and TIS (sum over time).
        """

        all_test_accuracies = []
        final_channel_distributions = []  # Track channels left in last 5 steps


        ############### FUNCTIONS ################
        def compute_features(channels):
            """ Compute features based on chosen feature type. """
            if feature_type == "concatenated":
                try:
                    # Concatenate raw selected channels
                    concatenated_data = np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
                except Exception as e:
                    print(f"An error occurred while plotting: {e}")
                return concatenated_data
            elif feature_type == "tic_tis":
                # Compute TIC (sum over m/z) and TIS (sum over time)
                tic = np.sum(cls_data[:, :, channels], axis=2)  # (num_samples, num_timepoints)
                tis = np.sum(cls_data[:, :, channels], axis=1)  # (num_samples, num_channels)
                return np.hstack([tic, tis])  # Combine TIC and TIS
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

        # Evaluate the accuracy of removing each channel

        ############### END FUNCTIONS ################


        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls_data = self.data.copy()  # Shape: (num_samples, num_timepoints, num_channels)
            num_samples, num_timepoints, num_channels = cls_data.shape
            labels = self.labels

            train_idx, test_idx = self.shuffle_split_without_splitting_duplicates(
                self.data, self.labels,
                test_size=test_size,
                random_state=random_seed + repeat_idx,
                group_duplicates=True,
                dataset_origins=dataset_origins
            )

            # Start with all channels
            remaining_indices = list(range(num_channels))
            incremental_accuracies = [None]  # Store accuracy before removing any channels

            def evaluate_channel(ch_idx):
                print(f'Step {step}; Channel = {ch_idx}')
                temp_indices = [idx for idx in remaining_indices if idx != ch_idx]
                if not temp_indices:  # If no channels remain, skip this iteration
                    print(f"⚠️ No remaining channels after removing {ch_idx}. Skipping evaluation.")
                    return ch_idx, None  # Return None to avoid breaking max() call later

                # Data with currently selected channels
                feature_matrix = compute_features(temp_indices)

                if feature_matrix is None or feature_matrix.size == 0:  # Check if feature matrix is empty
                    print(f"⚠️ Feature matrix is empty after removing {ch_idx}. Skipping.")
                    return ch_idx, None

                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels,
                                 dataset_origins=dataset_origins)
                # results = cls.train_and_evaluate_balanced_diff_origins(
                results = cls.train_and_evaluate_balanced_target_origin(
                    n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    n_jobs=n_inner_repeats,
                    test_on_discarded=False,
                    target_origin=target_origin,
                    pre_split_indices=(train_idx, test_idx)
                )
                # Extract just the result for the target origin
                origin_key = str(target_origin)

                # If target origin is missing but only one key exists, use that one
                if origin_key not in results:
                    if len(results) == 1:
                        origin_key = list(results.keys())[0]  # use the only available key
                        print(f"⚠️ Using fallback origin key: {origin_key}")
                    else:
                        print(f"❌ Target origin '{origin_key}' not found in results. Skipping.")
                        return None

                # Pull the accuracy for that specific dataset
                accuracy = results[origin_key]["balanced_accuracy"]
                return ch_idx, accuracy

            for step in range(min(num_channels, 137 + 1)):  # Limit number of removals for efficiency

                if not remaining_indices:  # Prevent issues when all channels are removed
                    print("⚠️ No more channels left to remove. Stopping early.")
                    break

                # Compute current accuracy before removing any channel this iteration
                feature_matrix = compute_features(remaining_indices)
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                                 year_labels=self.year_labels,
                                 dataset_origins=dataset_origins)
                # results = cls.train_and_evaluate_balanced_diff_origins(
                results = cls.train_and_evaluate_balanced_target_origin(
                    # n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    # n_jobs=1,
                    test_on_discarded=True,
                    target_origin=target_origin,
                    pre_split_indices=(train_idx, test_idx)
                )
                # Extract just the result for the target origin
                origin_key = str(target_origin)
                if origin_key not in results:
                    print(f"⚠️ Target origin '{origin_key}' not found in results. Skipping.")
                    return None  # Or continue safely

                # Parallel execution to test removing each channel
                validation_accuracies = Parallel(n_jobs=10, backend='loky')(
                    delayed(evaluate_channel)(ch_idx) for ch_idx in remaining_indices)

                # validation_accuracies = [evaluate_channel(ch_idx) for ch_idx in remaining_indices]

                # Store accuracy at current step (before removal)
                if incremental_accuracies[0] is None:
                    incremental_accuracies[0] = results[origin_key]["balanced_accuracy"]
                else:
                    incremental_accuracies.append(results[origin_key]['balanced_accuracy'])

                # Remove the channel whose removal leads to the highest validation accuracy
                max_accuracy = max(validation_accuracies, key=lambda x: x[1])[1]
                candidates = [ch_idx for ch_idx, acc in validation_accuracies if acc == max_accuracy]
                best_channel_to_remove = random.choice(candidates)  # Randomly select one of the best
                remaining_indices.remove(best_channel_to_remove)

                # Track the last 5 steps' remaining channels
                if len(remaining_indices) == 5:
                # if len(remaining_indices) == 10:
                    final_channel_distributions.append(
                        list(remaining_indices))  # Only store when there are exactly 5 left

                if print_results:
                    print(
                        f"After removing {step + 1} channel(s): Test Accuracy = {results[origin_key]['balanced_accuracy']:.3f}")

            all_test_accuracies.append(incremental_accuracies)

            # Dynamic plot after each repeat
            plt.ion()  # Enable interactive mode
            plt.clf()  # Clear the current figure
            mean_test_accuracies = np.mean(all_test_accuracies, axis=0)
            std_test_accuracies = np.std(all_test_accuracies, axis=0)

            x = range(len(mean_test_accuracies))
            plt.plot(x, mean_test_accuracies, '-o', label='Mean Accuracy')
            plt.fill_between(x, mean_test_accuracies - std_test_accuracies, mean_test_accuracies + std_test_accuracies,
                             alpha=0.3, label='± 1 Std Dev')

            plt.xlabel("Number of Channels Removed")
            plt.ylabel("Mean Balanced Accuracy")
            plt.ylim(0.5, 1.1)
            title_origin = f"{target_origin}" if target_origin else ""
            plt.title(f"Acc. Trend with Greedy Remove({title_origin}); {feature_type}; {repeat_idx + 1} / {num_repeats}")
            plt.grid(True)
            plt.legend()
            if repeat_idx == num_repeats - 1:
                plt.show(block=True)  # Keep the final plot open without closing the program
            else:
                plt.pause(1)  # Pause briefly to allow the plot to update

        # Print distribution of remaining channels
        print("\nDistribution of Remaining Channels in the Last 5 Steps:")
        unique_channels, counts = np.unique(final_channel_distributions, return_counts=True)
        for ch, count in zip(unique_channels, counts):
            print(f"Channel {ch}: {count} occurrences")


        # Save histogram data to file
        with open(f"/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_greedy_remove_JM22CS22({title_origin}).csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Channel Index"])
            writer.writerows([[ch] for ch in final_channel_distributions])

        # Plot histogram of remaining channels in the last 5 steps across repeats
        plt.figure(figsize=(10, 6))
        plt.hist(final_channel_distributions, bins=np.arange(min(final_channel_distributions), max(final_channel_distributions) + 1) - 0.5, edgecolor='black')
        plt.xlabel("Channel Index")
        plt.ylabel("Number of Repeats")
        plt.title("Histogram of Channels Remaining in the Last 5 Steps Across Repeats")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show(block=True)

        # Final Summary
        print(f"Final Mean Accuracy: {np.mean(mean_test_accuracies):.3f} ± {np.mean(std_test_accuracies):.3f}")

        return mean_test_accuracies, std_test_accuracies



    def train_and_evaluate_greedy_remove_batch(self, num_repeats=10, num_outer_repeats=1, n_inner_repeats=50,
                                               random_seed=42, test_size=0.2, normalize=False, scaler_type='standard',
                                               use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                               feature_type="concatenated", batch_size=50,
                                               selection_mode="correlation"):  # correlation or snr
        """
        Perform greedy channel removal in small batches based on the specified selection mode.
        Options for selection_mode:
        - "correlation": Orders channels by descending correlation similarity.
        - "snr": Orders channels by increasing signal-to-noise ratio (SNR) averaged across samples.
        """

        all_test_accuracies = []

        def compute_features(channels):
            """ Compute features based on chosen feature type. """
            if isinstance(channels, int):
                channels = [channels]  # Convert single integer to list
            if feature_type == "concatenated":
                return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")
            cls_data = self.data.copy()
            num_samples, num_timepoints, num_channels = cls_data.shape
            labels = self.labels

            if selection_mode == "none":
                remaining_indices = list(range(num_channels))
            # elif selection_mode == "correlation":
            #     # Compute correlation across all channels and sort them in descending order
            #     reshaped_data = cls_data.transpose(2, 0, 1).reshape(num_channels, -1)
            #     corr_matrix = np.corrcoef(reshaped_data)
            #     mean_correlation = np.abs(corr_matrix).mean(axis=1)
            #     sorted_indices = np.argsort(mean_correlation)[::-1]  # Sort in descending order
            #     remaining_indices = list(sorted_indices)

            elif selection_mode == "correlation":
                # Compute correlation across all channels
                reshaped_data = cls_data.transpose(2, 0, 1).reshape(num_channels,
                                                                    -1)  # Shape: (num_channels, total_samples)
                corr_matrix = np.corrcoef(reshaped_data)

                # Compute mean absolute correlation per channel
                mean_correlation = np.abs(corr_matrix).mean(axis=1)

                # Step 1: Identify groups of highly correlated channels
                threshold = 0.7  # Adjust threshold as needed
                correlated_groups = []
                visited = set()

                for i in range(num_channels):
                    if i in visited:
                        continue
                    # Find channels highly correlated with channel i
                    correlated_indices = np.where(np.abs(corr_matrix[i]) > threshold)[0].tolist()
                    correlated_groups.append(correlated_indices)
                    visited.update(correlated_indices)

                # Step 2: Select one "representative" per group
                representatives = []
                redundant_channels = []

                for group in correlated_groups:
                    if len(group) == 1:
                        representatives.append(group[0])  # Keep single-channel groups
                    else:
                        # Choose the least redundant (lowest mean correlation) as the representative
                        rep = min(group, key=lambda ch: mean_correlation[ch])
                        representatives.append(rep)
                        redundant_channels.extend([ch for ch in group if ch != rep])

                # Step 3: Interleave representatives and redundant channels
                ordered_indices = representatives + redundant_channels
                remaining_indices = list(ordered_indices)

            elif selection_mode == "snr":
                # Compute signal-to-noise ratio (SNR) for each channel
                mean_signal = np.mean(cls_data, axis=(0, 1))  # Shape: (num_channels,)
                std_noise = np.std(cls_data, axis=(0, 1))  # Shape: (num_channels,)
                snr_values = mean_signal / (std_noise + 1e-10)  # Avoid division by zero
                sorted_indices = np.argsort(snr_values)  # Sort in ascending order
                remaining_indices = list(sorted_indices)


            elif selection_mode == "tree_based":

                # Reshape the data so that each channel is treated as a separate feature
                num_samples, num_timepoints, num_channels = cls_data.shape

                # Flatten the retention times dimension for each sample so that (num_samples, num_channels)
                feature_matrix = cls_data.reshape(num_samples, -1, num_channels)  # Shape: (num_samples, timepoints, num_channels)
                feature_matrix = np.mean(feature_matrix, axis=1)  # Average across retention times → (num_samples, num_channels)

                # Ensure labels are 1D
                labels = np.array(labels).ravel()

                # Train a Random Forest model
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=None)
                rf.fit(feature_matrix, labels)

                # Extract feature importances and sort them
                feature_importances = rf.feature_importances_
                sorted_indices = np.argsort(feature_importances)  # Remove least important channels first
                remaining_indices = list(sorted_indices)


            incremental_accuracies = [None]

            for step in range(0, min(num_channels, 125 + 1), batch_size):
                if len(remaining_indices) < batch_size:
                    print("⚠️ Too few channels left to remove in batch. Stopping early.")
                    break

                feature_matrix = compute_features(remaining_indices)
                cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind)
                results = cls.train_and_evaluate_balanced(
                    n_inner_repeats=n_inner_repeats,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    use_pca=use_pca,
                    vthresh=vthresh,
                    region=region,
                    print_results=False,
                    n_jobs=n_jobs,
                    test_on_discarded=True
                )

                validation_accuracies = Parallel(n_jobs=5, backend='loky')(
                    delayed(lambda subset: (subset, cls.train_and_evaluate_balanced(
                        n_inner_repeats=n_inner_repeats,
                        random_seed=random_seed + repeat_idx,
                        test_size=test_size,
                        normalize=normalize,
                        scaler_type=scaler_type,
                        use_pca=use_pca,
                        vthresh=vthresh,
                        region=region,
                        print_results=False,
                        n_jobs=n_jobs,
                        test_on_discarded=True
                    )["overall_balanced_accuracy"]))(
                        remaining_indices[i:i + min(batch_size, len(remaining_indices) - i)])
                    for i in range(0, len(remaining_indices), batch_size)
                )

                if incremental_accuracies[0] is None:
                    incremental_accuracies[0] = results['overall_balanced_accuracy']
                else:
                    incremental_accuracies.append(results['overall_balanced_accuracy'])

                best_subset = min(validation_accuracies, key=lambda x: x[1])[0]
                for ch in best_subset:
                    remaining_indices.remove(ch)

                if print_results:
                    print(
                        f"After removing {step + batch_size} channels: Test Accuracy = {results['overall_balanced_accuracy']:.3f}")

            all_test_accuracies.append(incremental_accuracies)
            max_length = max(len(acc) for acc in all_test_accuracies)
            all_test_accuracies = [acc + [np.nan] * (max_length - len(acc)) for acc in all_test_accuracies]

            if repeat_idx == 0:
                plt.figure(figsize=(12, 6))
            plt.ion()
            plt.clf()
            mean_test_accuracies = np.nanmean(all_test_accuracies, axis=0)
            std_test_accuracies = np.nanstd(all_test_accuracies, axis=0)

            x = [i * batch_size for i in range(len(mean_test_accuracies))]
            plt.plot(x, mean_test_accuracies, '-o', label='Mean Accuracy')
            plt.fill_between(x, mean_test_accuracies - std_test_accuracies,
                             mean_test_accuracies + std_test_accuracies,
                             alpha=0.3, label='± 1 Std Dev')
            plt.xlabel(f"Number of Channels Removed (Batch {batch_size})")
            plt.ylabel("Mean Balanced Accuracy")
            plt.title(
                f"Performance Trend Greedy Remove ({selection_mode.capitalize()} Method) (Repeat {repeat_idx + 1} / {num_repeats})")
            plt.grid(True)
            plt.legend()
            if repeat_idx == num_repeats - 1:
                plt.show(block=True)
            else:
                plt.pause(1)

            print(f"Final Mean Accuracy: {np.mean(mean_test_accuracies):.3f} ± {np.mean(std_test_accuracies):.3f}")
        return mean_test_accuracies, std_test_accuracies

    def train_and_evaluate_random_channel_subsets(self, num_repeats=100, n_inner_repeats=20,
                                                  random_seed=42, test_size=0.2, normalize=False,
                                                  scaler_type='standard', use_pca=False, vthresh=0.97,
                                                  region=None, print_results=True, n_jobs=-1,
                                                  feature_type="tic_tis"):
        """
        Randomly selects 5 channels for each repeat and evaluates performance.
        Intended for comparison with greedy removal, for histogram generation.
        """
        all_test_accuracies = []
        final_channel_distributions = []

        def compute_features(channels):
            if feature_type == "concatenated":
                return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

        import time
        random_seed = int(time.time_ns())  # use time-based entropy
        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls_data = self.data.copy()
            labels = self.labels
            num_samples, num_timepoints, num_channels = cls_data.shape

            random.seed(random_seed + repeat_idx)
            selected_channels = sorted(random.sample(range(num_channels), 5))
            final_channel_distributions.append(selected_channels)

            feature_matrix = compute_features(selected_channels)

            cls = Classifier(feature_matrix, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                             year_labels=self.year_labels)

            results = cls.train_and_evaluate_balanced(
                n_inner_repeats=n_inner_repeats,
                random_seed=random_seed + repeat_idx,
                test_size=test_size,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                region=region,
                print_results=False,
                n_jobs=n_jobs,
                test_on_discarded=False
            )

            all_test_accuracies.append(results['overall_balanced_accuracy'])

            if print_results:
                print(f"Selected Channels: {selected_channels} | Accuracy = {results['overall_balanced_accuracy']:.3f}")

        # Save histogram data
        output_file = "/home/luiscamara/PycharmProjects/wine_analysis/data/press_wines/hist_5ch_random_channel_subset_merlot.csv"
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Selected Channels"])
            for subset in final_channel_distributions:
                writer.writerow([subset])

        # Plot histogram
        flat_channels = [ch for subset in final_channel_distributions for ch in subset]
        plt.figure(figsize=(10, 6))
        plt.hist(flat_channels, bins=np.arange(min(flat_channels), max(flat_channels) + 2) - 0.5, edgecolor='black')
        plt.xlabel("Channel Index")
        plt.ylabel("Frequency across repeats")
        plt.title("Histogram of Randomly Selected 5 Channels")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show(block=True)

        print(f"\nFinal Mean Accuracy: {np.mean(all_test_accuracies):.3f} ± {np.std(all_test_accuracies):.3f}")

        return np.mean(all_test_accuracies), np.std(all_test_accuracies)



    def train_and_evaluate_tic(
            self, num_repeats=10, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
            dataset_origins=None
    ):
        cls_data = self.data.copy()
        labels = self.labels
        accuracies = []
        confusion_matrices = []  # Store confusion matrices
        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls = Classifier(cls_data, labels, classifier_type="RGC", wine_kind=self.wine_kind,year_labels=self.year_labels)
            results = cls.train_and_evaluate_balanced(
                random_seed=random_seed + repeat_idx,
                test_size=test_size,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                region=region,
                print_results=True,
                n_jobs=n_jobs,
                test_on_discarded=True
            )
            accuracies.append(results['overall_balanced_accuracy'])
            confusion_matrices.append(results['overall_confusion_matrix_normalized'])  # Collect confusion matrices

        # Compute average performance across repeats
        mean_test_accuracy = np.mean(accuracies, axis=0)
        std_test_accuracy = np.std(accuracies, axis=0)
        print("\n##################################")
        print(f"Final Mean Accuracy: {mean_test_accuracy:.3f} ± {std_test_accuracy:.3f} "
              f"(chance {results['chance_accuracy']:.3f})")

        # Compute the averaged confusion matrix across all repetitions
        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

        # Print final normalized confusion matrix
        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        return mean_test_accuracy, std_test_accuracy


    def train_and_evaluate_tic_diff_origins(
            self, num_repeats=10, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
            dataset_origins=None
    ):
        cls_data = self.data.copy()
        labels = self.labels
        accuracies = {}
        confusion_matrices = {}
        # accuracies = []
        # confusion_matrices = []  # Store confusion matrices
        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls = Classifier(cls_data, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                             year_labels=self.year_labels,
                             dataset_origins=dataset_origins
                             )
            results = cls.train_and_evaluate_balanced_diff_origins(
                random_seed=random_seed + repeat_idx,
                test_size=test_size,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                region=region,
                print_results=True,
                n_jobs=n_jobs,
                test_on_discarded=True
            )
            # Store accuracy and confusion matrices per dataset origin
            for dataset, metrics in results.items():
                dataset_key = str(dataset)  # Convert numpy.str_ to a native Python string

                if dataset_key not in accuracies:
                    accuracies[dataset_key] = []
                    confusion_matrices[dataset_key] = []

                accuracies[dataset_key].append(metrics['balanced_accuracy'])
                confusion_matrices[dataset_key].append(metrics['confusion_matrix'])

        # Compute and print average performance per dataset origin
        print("\n##################################")
        for dataset, acc_list in accuracies.items():
            mean_acc = np.mean(acc_list)
            std_acc = np.std(acc_list)
            print(f"{dataset} Mean Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

        # Compute and print the averaged confusion matrix per dataset origin
        for dataset, cm_list in confusion_matrices.items():
            mean_cm = np.mean(cm_list, axis=0)
            print(f"\n{dataset} Final Averaged Normalized Confusion Matrix:")
            print(mean_cm)
        print("##################################")

        return accuracies, confusion_matrices


    def train_and_evaluate_tic_tis(
            self, tics, tiss, num_repeats=10, num_outer_repeats=1, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True
    ):
        cls_data = self.data.copy()
        labels = self.labels
        accuracies = []
        confusion_matrices = []  # Store confusion matrices
        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls = Classifier(cls_data, labels, classifier_type="RGC", wine_kind=self.wine_kind,
                             alpha=1)
            results = cls.train_and_evaluate_balanced_tic_tis(
                tics, tiss,
                random_seed=random_seed + repeat_idx,
                test_size=test_size,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                region=region,
                print_results=True,
                normalize_before_concat=True
            )

            accuracies.append(results['overall_balanced_accuracy'])
            confusion_matrices.append(results['overall_confusion_matrix_normalized'])  # Collect confusion matrices

        # Compute average performance across repeats
        mean_test_accuracy = np.mean(accuracies, axis=0)
        std_test_accuracy = np.std(accuracies, axis=0)
        print("\n##################################")
        print(f"Final Mean Balanced Accuracy: {mean_test_accuracy:.3f} ± {std_test_accuracy:.3f} "
              f"(chance {results['chance_accuracy']:.3f})")

        # Compute the averaged confusion matrix across all repetitions
        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

        # Print final normalized confusion matrix
        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        return mean_test_accuracy, std_test_accuracy



    def train_and_evaluate_all_channels(
            self, num_repeats=10, num_outer_repeats=1, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1
    ):
        cls_data = self.data.copy()
        labels = self.labels
        accuracies = []
        confusion_matrices = []  # Store confusion matrices
        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")

            cls = Classifier(cls_data, labels, classifier_type="RGC", wine_kind=self.wine_kind)
            results = cls.train_and_evaluate_balanced(
                random_seed=random_seed + repeat_idx,
                test_size=test_size,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                region=region,
                print_results=False,
                n_jobs=n_jobs,
                test_on_discarded=True
            )
            accuracies.append(results['overall_balanced_accuracy'])
            confusion_matrices.append(results['overall_confusion_matrix_normalized'])  # Collect confusion matrices

        # Compute average performance across repeats
        mean_test_accuracy = np.mean(accuracies, axis=0)
        std_test_accuracy = np.std(accuracies, axis=0)
        print("\n##################################")
        print(f"Mean Accuracy: {mean_test_accuracy:.3f} ± {std_test_accuracy:.3f}")

        # Compute the averaged confusion matrix across all repetitions
        mean_confusion_matrix = np.mean(confusion_matrices, axis=0)

        # Print final normalized confusion matrix
        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        return mean_test_accuracy, std_test_accuracy


    def train_and_evaluate_balanced_with_alpha(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
                                               normalize=False,
                                               scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
                                               batch_size=32, num_epochs=10, learning_rate=0.001,
                                               alpha_range=None, num_test=1, channel_weights=None):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------
        alpha_range : list or None
            Range of alpha values to test during cross-validation. If None, defaults to [0.1, 1.0, 10.0, 100.0].

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, and the mean optimal alpha value.
        """


        if channel_weights is not None:
            self.data = self.data * channel_weights.detach().numpy()

        # Separate all the m/z profiles and give them the label of the sample
        self.labels = np.repeat(self.labels, self.data.shape[2])
        self.data = self.data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])

        # Set default alpha range if not provided
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]

        # Initialize accumulators for metrics and alpha
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        best_alpha_values = []
        confusion_matrix_sum = None

        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(self.labels, self.data,
                vintage=vintage,test_size=test_size, num_test=num_test)

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Learn optimal alpha using RidgeClassifierCV
            ridge_classifier = RidgeClassifierCV(
                alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True
            )
            ridge_classifier.fit(X_train, y_train)
            best_alpha = ridge_classifier.alpha_
            best_alpha_values.append(best_alpha)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed. Best alpha: {best_alpha}")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits
        mean_alpha = np.mean(best_alpha_values)

        # Print summary
        print(f"Mean Alpha: {mean_alpha:.3f}")
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
            'mean_alpha': mean_alpha
        }


    def train_and_evaluate_balanced_with_passed_weights(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
                                               normalize=False,
                                               scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
                                               batch_size=32, num_epochs=10, learning_rate=0.001,
                                               alpha_range=None, num_test=1, channel_weights=None):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------
        alpha_range : list or None
            Range of alpha values to test during cross-validation. If None, defaults to [0.1, 1.0, 10.0, 100.0].

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, and the mean optimal alpha value.
        """

        if channel_weights is not None:
            self.data = self.data * channel_weights

        # Separate all the m/z profiles and give them the label of the sample
        self.labels = np.repeat(self.labels, self.data.shape[2])
        self.data = self.data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])

        # Set default alpha range if not provided
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]

        # Initialize accumulators for metrics and alpha
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        best_alpha_values = []
        confusion_matrix_sum = None

        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(self.labels, self.data,
                vintage=vintage,test_size=test_size, num_test=num_test)

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Learn optimal alpha using RidgeClassifierCV
            ridge_classifier = RidgeClassifier()
            ridge_classifier.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
        }


    def train_and_evaluate_balanced_with_best_alpha(
            self, n_splits=50, vintage=False, test_size=None, normalize=False, scaler_type='standard', use_pca=False,
            vthresh=0.97, best_alpha=1, region=None):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------


        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, and the mean optimal alpha value.
        """

        if region == 'winery':
            custom_order = ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X']
        elif region  == 'origin':
            custom_order = ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        else:
           custom_order = None

        # Separate all the m/z profiles and give them the label of the sample
        self.labels = np.repeat(self.labels, self.data.shape[2])
        # self.data = self.data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        self.data = self.data.transpose(0, 2, 1).reshape(-1, self.data.shape[1])

        # Initialize accumulators for metrics and alpha
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        ridge_classifier = RidgeClassifier(alpha=best_alpha)

        # Cross-validation loop
        for i in range(n_splits):
            # Split data into train and test sets
            # train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(self.labels, self.data,
            #     vintage=vintage,test_size=test_size, num_test=num_test)

            X_train, X_test, y_train, y_test = train_test_split(
                self.data, self.labels, test_size=0.1, stratify=self.labels
            )

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            ridge_classifier.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            # cm = confusion_matrix(y_test, y_pred, labels=custom_order)
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
        }


    def train_and_evaluate_balanced_with_best_alpha2(
            self, n_splits=50, test_size=0.2, normalize=False, scaler_type='standard',
            use_pca=False, vthresh=0.97, best_alpha=1, region=None
    ):
        """
        Train and evaluate the classifier using cross-validation, with stratified train-test splits at the sample level.

        Parameters
        ----------
        n_splits : int
            Number of cross-validation splits.
        test_size : float
            Proportion of the data to include in the test split.
        normalize : bool
            Whether to normalize the data.
        scaler_type : str
            Type of scaler to use for normalization.
        use_pca : bool
            Whether to apply PCA for dimensionality reduction.
        vthresh : float
            Variance threshold for PCA.
        best_alpha : float
            Regularization parameter for RidgeClassifier.
        region : str
            Custom label ordering for visualization.

        Returns
        -------
        dict
            A dictionary containing evaluation metrics and the confusion matrix.
        """

        # Custom label ordering for specific regions (if needed)
        custom_order = {
            'winery': ['D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'W', 'Y', 'M', 'N', 'J', 'L', 'H', 'U', 'X'],
            'origin': ['Beaune', 'Alsace', 'Neuchatel', 'Genève', 'Valais', 'Californie', 'Oregon']
        }.get(region, None)

        # Labels and data at the sample level
        sample_labels = self.labels
        sample_data = self.data  # Shape: (num_samples, features, channels)

        # Initialize accumulators for metrics
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        ridge_classifier = RidgeClassifier(alpha=best_alpha)

        # Cross-validation loop
        for i in range(n_splits):
            # Perform stratified split at the sample level
            train_idx, test_idx = train_test_split(
                np.arange(len(sample_labels)),
                test_size=test_size,
                stratify=sample_labels,
                random_state=i
            )

            # Prepare train and test data while keeping channels grouped by sample
            X_train = sample_data[train_idx].transpose(0, 2, 1).reshape(-1, sample_data.shape[1])  # Flatten channels
            X_test = sample_data[test_idx].transpose(0, 2, 1).reshape(-1, sample_data.shape[1])
            y_train = np.repeat(sample_labels[train_idx], sample_data.shape[2])  # Repeat labels for all channels
            y_test = np.repeat(sample_labels[test_idx], sample_data.shape[2])

            # Normalize data if required
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if required
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Train and predict
            ridge_classifier.fit(X_train, y_train)
            y_pred = ridge_classifier.predict(X_test)

            # Compute metrics
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_train)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(
            f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
        }


    def train_and_evaluate_balanced_with_minimize(
            self, n_splits=50, vintage=False, random_seed=42, test_size=None,
            normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
            alpha_range=None, num_test=1):
        """
        Train and evaluate the classifier using cross-validation, optimizing channel weights with numerical optimization.
        Learn the optimal alpha parameter for Ridge Classifier during training.

        Parameters
        ----------
        alpha_range : list or None
            Range of alpha values to test during cross-validation. If None, defaults to [0.1, 1.0, 10.0, 100.0].

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, weighted accuracy, precision, F1-score,
            mean confusion matrix, final weights, and the mean optimal alpha value.
        """

        # Default alpha range
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]
            alpha_range = [10.0]

        # Separate m/z profiles and repeat labels for each channel
        n_channels = self.data.shape[2]

        # Metrics accumulators
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        best_alpha_values = []
        confusion_matrix_sum = None

        def objective(weights):
            """
            Objective function to optimize weights. It computes the negative balanced accuracy
            of the Ridge ClassifierCV with the given weights applied to the m/z channels.
            """

            weighted_data = self.data * weights.reshape(1, 1, -1)
            reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
            labels_expanded = np.repeat(self.labels, n_channels)
            accuracy_scores_split = []

            # Train RidgeClassifierCV
            ridge_classifier = RidgeClassifierCV(alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True)

            for i in range(n_splits):
                train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                    labels_expanded, reshaped_data,  vintage=vintage, test_size=test_size, num_test=num_test
                )

                ridge_classifier.fit(X_train, y_train)
                best_alpha_values.append(ridge_classifier.alpha_)

                # Accuracy for this split
                y_pred = ridge_classifier.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                accuracy_scores_split.append(balanced_accuracy)

            # Return the negative mean balanced accuracy as the loss
            # print(np.mean(accuracy_scores_split))
            return -np.mean(accuracy_scores_split)

        # Initialize weights for m/z channels
        initial_weights = np.ones(n_channels)

        # Optimize weights
        result = minimize(objective, initial_weights, method='L-BFGS-B', bounds=[(0, None)] * n_channels,
                          options={'maxiter': 10, 'disp': True, 'iprint': 1}
                          )
        optimized_weights = result.x

        # Apply optimized weights to data
        print(f'Optimzed weights: {optimized_weights}')
        weighted_data = self.data * optimized_weights
        reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        labels_expanded = np.repeat(self.labels, n_channels)

        # Final evaluation
        for i in range(n_splits):
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                labels_expanded, reshaped_data, vintage=vintage, test_size=test_size, num_test=num_test
            )

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            ridge_classifier = RidgeClassifierCV(alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True)
            ridge_classifier.fit(X_train, y_train)

            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

        # Final metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits
        mean_alpha = np.mean(best_alpha_values)

        print(f"Optimized Weights: {optimized_weights}")
        print(f"Mean Alpha: {mean_alpha:.3f}")
        print(f"Mean Accuracy: {np.mean(accuracy_scores):.3f}")
        print(f"Mean Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f}")

        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
            'final_weights': optimized_weights,
            'mean_alpha': mean_alpha
        }


    def train_and_evaluate_balanced_with_weights_and_alpha0(
            self, n_splits=50, vintage=False, random_seed=42, test_size=None,
            normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
            batch_size=32, num_epochs=10, learning_rate=0.001, alpha_range=None, num_test=1):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.
        Learn the weights for each m/z channel and the optimal alpha parameter for Ridge Classifier.

        Returns
        -------
        dict
            Metrics including mean accuracy, balanced accuracy, precision, F1-score,
            mean confusion matrix, final weights, and mean alpha.
        """

        def ridge_loss(predictions, targets, weights, alpha):
            """
            Compute Ridge Loss: Mean Squared Error + Regularization on weights.

            Parameters
            ----------
            predictions : torch.Tensor
                Predicted values from the model.
            targets : torch.Tensor
                Ground truth labels.
            weights : torch.Tensor
                Learnable weights for channels.
            alpha : torch.Tensor
                Ridge regularization parameter.

            Returns
            -------
            torch.Tensor
                Computed Ridge Loss.
            """

            if predictions.dim() > 1:
                # Example: Taking the score for the correct class if multi-class
                predictions = predictions.gather(1, targets_train_tensor.long().unsqueeze(1)).squeeze()

            # Mean Squared Error Loss
            # mse_loss = torch.nn.functional.mse_loss(predictions, targets)
            ce_loss = torch.nn.functional.cross_entropy(predictions, targets)

            # Ridge Regularization (L2 Norm on Weights)
            # l2_reg = alpha * torch.sum(weights ** 2)
            l2_reg = optimizer.param_groups[0]['weight_decay'] * torch.sum(weights ** 2)

            # Combine MSE with L2 regularization
            return ce_loss + l2_reg

        # Set default alpha range if not provided
        if alpha_range is None:
            alpha_range = [0.1, 1.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 1500.0, 2000.0, 4000.0]

        # Number of channels in the original data
        n_channels = self.data.shape[2]

        # Initialize weights for each m/z channel
        channel_weights = torch.ones(n_channels, requires_grad=True, dtype=torch.float32)
        optimizer = torch.optim.Adam([channel_weights], lr=learning_rate, weight_decay=0.01)

        # Initialize accumulators for metrics and alpha
        accuracy_scores_epoch = []
        balanced_accuracy_scores_epoch = []
        precision_scores_epoch = []
        f1_scores_epoch = []
        confusion_matrix_epoch_sum = None
        best_alpha_values_epoch = []

        for epoch in range(num_epochs):
            print(f"Epoch {epoch + 1}/{num_epochs}")

            # Apply channel weights to the data
            weighted_data = self.data * channel_weights.detach().numpy()
            # weighted_data = self.data * channel_weights.unsqueeze(0).unsqueeze(2)

            # Reshape the data so channels become independent samples
            labels_expanded = np.repeat(self.labels, n_channels)
            data_expanded = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])

            # Initialize accumulators for this epoch
            accuracy_scores = []
            balanced_accuracy_scores = []
            precision_scores = []
            f1_scores = []
            best_alpha_values = []
            confusion_matrix_sum = None

            label_encoder = LabelEncoder()
            encoded_labels = label_encoder.fit_transform(labels_expanded)

            ridge_classifier = RidgeClassifierCV(alphas=alpha_range, scoring='balanced_accuracy', store_cv_values=True)

            for i in range(n_splits):
                # Split data into train and test sets
                train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                    encoded_labels, data_expanded, vintage=vintage, test_size=test_size, num_test=num_test
                )

                # Normalize data if enabled
                if normalize:
                    X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                    X_test = scaler.transform(X_test)

                # Apply PCA if enabled
                if use_pca:
                    reducer = DimensionalityReducer(self.data)
                    _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh,
                                                                     plot=False)
                    n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                    pca = PCA(n_components=n_components, svd_solver='randomized')
                    X_train = pca.fit_transform(X_train)
                    X_test = pca.transform(X_test)

                # Learn optimal alpha using RidgeClassifierCV
                ridge_classifier.fit(X_train, y_train)
                best_alpha = ridge_classifier.alpha_
                best_alpha_values.append(ridge_classifier.alpha_)

                # Predictions and metrics
                y_pred = ridge_classifier.predict(X_test)
                accuracy = ridge_classifier.score(X_test, y_test)
                accuracy_scores.append(accuracy)
                balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))
                precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
                f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

                # Compute confusion matrix
                cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
                confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

                print(f"Split {i + 1}/{n_splits}: Best Alpha = {best_alpha:.3f}, Accuracy = {accuracy:.4f}")

            # Compute metrics for the epoch
            mean_accuracy = np.mean(accuracy_scores)
            mean_balanced_accuracy = np.mean(balanced_accuracy_scores)
            mean_precision = np.mean(precision_scores)
            mean_f1_score = np.mean(f1_scores)
            mean_alpha = np.mean(best_alpha_values)

            # Append metrics for this epoch
            accuracy_scores_epoch.append(mean_accuracy)
            balanced_accuracy_scores_epoch.append(mean_balanced_accuracy)
            precision_scores_epoch.append(mean_precision)
            f1_scores_epoch.append(mean_f1_score)

            if confusion_matrix_epoch_sum is None:
                confusion_matrix_epoch_sum = confusion_matrix_sum
            else:
                confusion_matrix_epoch_sum += confusion_matrix_sum

            best_alpha_values_epoch.append(mean_alpha)

            # Train on the entire training data with the best alpha
            predictions_train = torch.tensor(ridge_classifier.decision_function(data_expanded), dtype=torch.float32)
            targets_train_tensor = torch.tensor(encoded_labels, dtype=torch.long)

            # Validate shapes
            assert predictions_train.shape[0] == targets_train_tensor.shape[0], \
                f"Shape mismatch: predictions {predictions_train.shape}, targets {targets_train_tensor.shape}"

            # Compute loss for the epoch
            # ridge_loss_value = ridge_loss(predictions_train, targets_train_tensor, channel_weights, mean_alpha)
            # predictions_train = torch.nn.functional.softmax(predictions_train, dim=1)  # Ensure normalized probabilities
            ce_loss = torch.nn.functional.cross_entropy(predictions_train, targets_train_tensor)

            # Add regularization
            # l2_reg = optimizer.param_groups[0]['weight_decay'] * torch.sum(channel_weights ** 2)
            ridge_loss_value = ce_loss # + l2_reg

            # Update weights
            optimizer.zero_grad()
            ridge_loss_value.backward()
            print("Gradients:", channel_weights.grad)
            optimizer.step()
            print(channel_weights)

            print(f"Epoch {epoch + 1} Summary: Accuracy = {mean_accuracy:.4f}, Learned Alpha = {mean_alpha:.3f}")

        # Final metrics
        mean_confusion_matrix = confusion_matrix_epoch_sum / n_splits
        final_weights = channel_weights.detach().numpy()
        mean_alpha_final = np.mean(best_alpha_values_epoch)

        print(f"Final Weights: {final_weights}")
        print(f"Final Learned Alpha: {mean_alpha_final:.3f}")

        return {
            'mean_accuracy': np.mean(accuracy_scores_epoch),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores_epoch),
            'mean_precision': np.mean(precision_scores_epoch),
            'mean_f1_score': np.mean(f1_scores_epoch),
            'mean_confusion_matrix': mean_confusion_matrix,
            'final_weights': final_weights,
            'mean_alpha': mean_alpha_final
        }


    def train_and_evaluate_separate_datasets(self, X_train, y_train, X_test, y_test, n_splits=50, vintage=False,
                                             random_seed=42, normalize=True, scaler_type='standard'):
        """
        Train the classifier on the provided training dataset and evaluate its performance on the testing dataset
        using cross-validation.
        Parameters
        ----------
        X_train : numpy.ndarray
            Training data.
        y_train : numpy.ndarray
            Training labels.
        X_test : numpy.ndarray
            Testing data.
        y_test : numpy.ndarray
            Testing labels.
        n_splits : int, optional
            The number of splits for cross-validation. Default is 50.
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        random_seed : int, optional
            The random seed for reproducibility. Default is 42.
        normalize : bool, optional
            Whether to normalize the data. Default is True.
        scaler_type : str, optional
            The type of scaler to use for normalization if `normalize` is True. Default is 'standard'.

        Returns
        -------
        float
            The mean accuracy score from cross-validation.

        Notes
        -----
        This function trains the classifier on the training data and evaluates it on the testing data.
        """

        # Set the random seed for reproducibility, ensuring that the data splits and other random processes are consistent
        np.random.seed(random_seed)

        # Normalize the training data if the normalize flag is set to True
        if normalize:
            X_train, scaler_train = normalize_data(X_train, scaler=scaler_type)

        # Train the classifier using the training data
        self.classifier.fit(X_train, y_train)

        # Initialize a new Classifier instance for the testing data
        test_cls = Classifier(X_test, y_test)

        # Initialize a list to store the accuracy scores from each split
        scores = []

        # Print 'Split' to indicate the start of cross-validation, keeping the output on the same line
        print('Split', end=' ', flush=True)

        # Perform cross-validation for the specified number of splits
        for i in range(n_splits):
            # Split the testing data into "in" and "out" samples for cross-validation
            in_indices, out_indices, X_in, X_out, y_in, y_out = test_cls.split_data(vintage=vintage, test_size=None)

            # Normalize the samples if normalization is enabled
            if normalize:
                X_in, scaler_test = normalize_data(X_in, scaler=scaler_type)
                # Use scaler fitted on X_in to transform X_out to ensure consistent scaling and prevent data leakage.
                X_out = scaler_test.transform(X_out)
            # X_out = scaler_train.transform(X_out)

            # Evaluate the classifier on the "out" sample and append the score to the list
            scores.append(self.classifier.score(X_out, y_out))
            print(self.classifier.score(X_out, y_out))

            # Print the current split number every 5 iterations to show progress
            print(i, end=' ', flush=True) if i % 5 == 0 else None

        # Convert the list of scores to a numpy array for easier statistical calculations
        scores = np.asarray(scores)

        # Print a new line after the loop completes
        print()

        # Print the mean accuracy and the standard deviation across the cross-validation splits
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")

        # Return the mean accuracy score as the final result
        return scores.mean()

    def _process_labels(self, vintage=False):
        """
        Process the labels to extract relevant parts based on whether the data is vintage or not.

        Parameters
        ----------
        vintage : bool
            If True, the function processes labels to extract a substring starting from the first digit
            found in the label (assuming vintage data formatting). If False, it processes labels to
            extract a single character or digit before the first digit found.

        Returns
        -------
        numpy.ndarray
            An array of processed labels.
        """
        if self.wine_kind == 'pinot_noir':
            processed_labels = self.labels
        elif self.wine_kind == 'bordeaux':
            processed_labels = []

            # Iterate over each label in the labels list
            for label in self.labels:
                # Search for the first digit in the label
                match = re.search(r'\d+', label)

                if vintage:
                    # If processing vintage data, extract the substring starting from the first digit
                    processed_labels.append(label[match.start():])
                else:
                    # If not vintage, extract the character before the first digit
                    if label[match.start() - 1] == '_':
                        # If the character before the digit is an underscore, take the character before the underscore
                        lb = label[match.start() - 2]
                    else:
                        # Otherwise, take the character directly before the first digit
                        lb = label[match.start() - 1]
                    processed_labels.append(lb)

        # Return the processed labels as a numpy array
        return np.array(processed_labels)

    def split_data(self, labels, data, vintage=False, test_size=None, num_test=1):
        """
        Split the data into training and testing sets based on labels.

        Parameters
        ----------
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, only one sample
            per unique label is used for testing. Default is None.

        Returns
        -------
        tuple
            A tuple containing the following elements:

            - train_indices : numpy.ndarray
                Indices of the training data samples.

            - test_indices : numpy.ndarray
                Indices of the testing data samples.

            - X_train : numpy.ndarray
                The training data.

            - X_test : numpy.ndarray
                The testing data.

            - y_train : numpy.ndarray
                The labels for the training data.

            - y_test : numpy.ndarray
                The labels for the testing data.

        Notes
        -----
        This function splits the dataset into training and testing sets by first processing the labels.
        The splitting is done in such a way that either one sample per unique label is reserved for testing
        (if test_size is None) or a specified proportion of samples per label is reserved for testing.
        The samples are randomly shuffled before splitting to ensure randomness in the selection.
        """

        # if self.wine_kind == 'bordeaux':
        #     # Process the labels according to whether they are vintage or not
        #     processed_labels = self._process_labels(vintage)
        # else:
        processed_labels = labels

        # Initialize lists to store indices for training and testing samples
        test_indices = []
        train_indices = []

        # Iterate over each unique label to perform stratified splitting
        for label in np.unique(processed_labels):
            # Find indices of all samples corresponding to the current label
            label_indices = np.where(np.array(processed_labels) == label)[0]

            # Shuffle these indices to ensure randomness in splitting
            np.random.shuffle(label_indices)

            if test_size is None:
                # If test_size is not specified, select one sample per label for testing
                test_indices.extend(label_indices[:num_test])  # Take the first shuffled index for testing
                train_indices.extend(label_indices[num_test:])  # The rest is for training
            else:
                # If test_size is specified, calculate the split point based on the test_size proportion
                split_point = int(len(label_indices) * test_size)
                test_indices.extend(label_indices[:split_point])  # The first part goes into testing
                train_indices.extend(label_indices[split_point:])  # The remaining is for training

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        # Split the data and labels into training and testing sets based on the calculated indices
        X_train, X_test = data[train_indices], data[test_indices]
        y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]

        # Return the indices, data, and labels for both training and testing sets
        return train_indices, test_indices, X_train, X_test, y_train, y_test


    def train_and_evaluate_all_mz_per_sample(self, n_splits=50, vintage=False, random_seed=42, test_size=None,
                                             normalize=False, scaler_type='standard', use_pca=False, vthresh=0.97,
                                             region=None, best_alpha=1.0):
        """
        Train and evaluate the classifier using cross-validation, with accuracy metrics for imbalanced classes.

        Parameters
        ----------
        alpha : float
            Regularization strength for RidgeClassifier.
        n_splits : int
            Number of cross-validation splits.
        vintage : bool
            Whether to use vintage data for splitting.
        random_seed : int
            Seed for reproducibility.
        test_size : float or None
            Proportion of the data to include in the test split.
        normalize : bool
            Whether to normalize the data.
        scaler_type : str
            Type of scaler to use for normalization ('standard' or 'minmax').
        use_pca : bool
            Whether to apply PCA for dimensionality reduction.
        vthresh : float
            Variance threshold for PCA.
        region : str or None
            Region-specific configurations for label ordering.

        Returns
        -------
        dict
            A dictionary containing mean accuracy, balanced accuracy, precision, F1-score,
            mean confusion matrix, and the fixed alpha value.
        """
        # Initialize accumulators for metrics
        accuracy_scores = []
        balanced_accuracy_scores = []
        precision_scores = []
        f1_scores = []
        confusion_matrix_sum = None

        # Use a dynamic seed if none is provided
        if random_seed is None:
            random_seed = np.random.randint(0, 1e6)
        rng = np.random.default_rng(random_seed)

        # Predefine splits for consistency
        predefined_splits = []
        for _ in range(n_splits):
            train_idx, temp_idx = train_test_split(
                np.arange(len(self.labels)), test_size=test_size + test_size, stratify=self.labels, random_state=rng.integers(0, 1e6)
            )
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=0.5, stratify=self.labels[temp_idx], random_state=rng.integers(0, 1e6)
            )
            predefined_splits.append((train_idx, val_idx, test_idx))

        # Cross-validation loop
        for i, (train_idx, val_idx, test_idx) in enumerate(predefined_splits):
            X_train = self.data[train_idx]
            X_val = self.data[val_idx]
            X_test = self.data[test_idx]
            y_train, y_val, y_test = self.labels[train_idx], self.labels[val_idx], self.labels[test_idx]

            # Concatenate the m/z profiles for each sample
            X_train = X_train.transpose(0, 2, 1).reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2])
            X_test = X_test.transpose(0, 2, 1).reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])

            # Normalize data if enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)

            # Apply PCA if enabled
            if use_pca:
                reducer = DimensionalityReducer(self.data)
                _, _, n_components = reducer.cumulative_variance(self.labels, variance_threshold=vthresh, plot=False)
                n_components = min(n_components, len(set(y_train)))  # Adjust PCA components based on class count
                pca = PCA(n_components=n_components, svd_solver='randomized')
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)

            # Train RidgeClassifier with the given alpha
            ridge_classifier = RidgeClassifier(alpha=best_alpha)
            ridge_classifier.fit(X_train, y_train)

            # Predictions and metrics
            y_pred = ridge_classifier.predict(X_test)
            accuracy_scores.append(ridge_classifier.score(X_test, y_test))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Compute precision and F1 score
            precision_scores.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            f1_scores.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=sorted(set(y_test)))
            confusion_matrix_sum = cm if confusion_matrix_sum is None else confusion_matrix_sum + cm

            # print(f"Split {i + 1}/{n_splits} completed.")

        # Calculate mean metrics
        mean_confusion_matrix = confusion_matrix_sum / n_splits

        # Print summary
        print(f"Alpha: {best_alpha:.3f}")
        print(f"Accuracy: {np.mean(accuracy_scores):.3f} (+/- {np.std(accuracy_scores) * 2:.3f})")
        print(f"Balanced Accuracy: {np.mean(balanced_accuracy_scores):.3f} (+/- {np.std(balanced_accuracy_scores) * 2:.3f})")
        print(f"Precision: {np.mean(precision_scores):.3f}")
        print(f"F1 Score: {np.mean(f1_scores):.3f}")
        print("Mean Confusion Matrix:\n", mean_confusion_matrix)

        # Return metrics
        return {
            'mean_accuracy': np.mean(accuracy_scores),
            'mean_balanced_accuracy': np.mean(balanced_accuracy_scores),
            'mean_precision': np.mean(precision_scores),
            'mean_f1_score': np.mean(f1_scores),
            'mean_confusion_matrix': mean_confusion_matrix,
            'alpha': best_alpha
        }


class BayesianParamOptimizer:
    def __init__(self, data, labels, n_channels=None, n_splits=5):
        """
        Initialize the Bayesian Weight Optimizer.

        Parameters
        ----------
        data : np.ndarray
            The input data with shape (samples, features, channels).
        labels : np.ndarray
            The labels for each sample.
        n_channels : int
            Number of m/z channels.
        n_splits : int
            Number of splits for cross-validation.
        """
        self.data = data
        self.labels = labels
        self.n_channels = n_channels
        self.n_splits = n_splits

        # Determine the base number with many multiples

        if self.n_channels is not None:
            self.base_channels = self._determine_base_channels()

    def _determine_base_channels(self):
        """
        Determine a base number close to n_channels that has many multiples.

        Returns
        -------
        int
            A base number with many divisors.
        """
        candidates = [i for i in range(self.n_channels, 1, -1) if 180 % i == 0]
        return candidates[0] if candidates else self.n_channels

    def evaluate_n_and_alpha(self, n, alpha, ch_treat, num_splits=5):
        """
        Evaluate the balanced accuracy for a given number of grouped channels and alpha.

        Parameters
        ----------
        n : int
            Number of channels grouped together.
        alpha : float
            Regularization strength for RidgeClassifier.
        ch_treat : string
            How to use the channels, independently or concatenated

        Returns
        -------
        float
            Negative mean balanced accuracy (to minimize).
        """
        num_groups = self.n_channels // n
        aggregated_data = np.zeros((self.data.shape[0], self.data.shape[1], num_groups))

        # Aggregate channels into groups
        for i in range(num_groups):
            start_idx = i * n
            end_idx = min((i + 1) * n, self.n_channels)
            aggregated_data[:, :, i] = np.mean(self.data[:, :, start_idx:end_idx], axis=-1)

        # # reshaped_data = aggregated_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        # reshaped_data = aggregated_data.transpose(0, 2, 1).reshape(-1, self.data.shape[1])
        # labels_expanded = np.repeat(self.labels, num_groups)

        # Evaluate performance using multiple train-test splits
        balanced_accuracies = []

        model = RidgeClassifier(alpha=alpha)
        for split in range(num_splits):
            # X_train, X_test, y_train, y_test = train_test_split(
            #     reshaped_data, labels_expanded, test_size=0.4, stratify=labels_expanded
            # )
            if ch_treat == 'individual':
                # Stratified split at the sample level
                train_idx, test_idx = train_test_split(
                    np.arange(len(self.labels)),
                    test_size=0.4,
                    stratify=self.labels,
                    random_state=split
                )

                # Prepare train and test data while keeping channels grouped
                X_train = aggregated_data[train_idx].transpose(0, 2, 1).reshape(-1, self.data.shape[1])
                X_test = aggregated_data[test_idx].transpose(0, 2, 1).reshape(-1, self.data.shape[1])
                y_train = np.repeat(np.array(self.labels)[train_idx], num_groups)
                y_test = np.repeat(np.array(self.labels)[test_idx], num_groups)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                balanced_accuracies.append(balanced_accuracy)

            elif ch_treat == 'concatenated':
                # Concatenate grouped channels for each sample
                concatenated_data = aggregated_data.transpose(0, 2, 1).reshape(self.data.shape[0], -1)

                # Stratified split at the sample level
                train_idx, test_idx = train_test_split(
                    np.arange(len(self.labels)),
                    test_size=0.4,
                    stratify=self.labels,
                    random_state=split
                )

                # Prepare train and test data with concatenated channels
                X_train, X_test = concatenated_data[train_idx], concatenated_data[test_idx]
                y_train, y_test = np.array(self.labels)[train_idx], np.array(self.labels)[test_idx]

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
                balanced_accuracies.append(balanced_accuracy)

            else:
                raise ValueError("Invalid value for ch_treat. Use 'individual' or 'concatenated'.")

        return -np.mean(balanced_accuracies)  # Negative score for minimization



    def evaluate_alpha(self, alpha, num_splits=5):
        """
        Evaluate the balanced accuracy for a given alpha.

        Parameters
        ----------
        alpha : float
            Regularization strength for RidgeClassifier.

        Returns
        -------
        float
            Negative mean balanced accuracy (to minimize).
        """
        balanced_accuracies = []

        model = RidgeClassifier(alpha=alpha)
        for _ in range(num_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                self.data, self.labels, test_size=0.4, stratify=self.labels, random_state=_
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            balanced_accuracies.append(balanced_accuracy)

        return -np.mean(balanced_accuracies)  # Negative score for minimization


    def optimize_gcms(self, n_calls=50, random_state=42, num_splits=5, ch_treat='concatenated'):
        """
        Optimize the number of channels grouped together and alpha using Bayesian Optimization.

        Parameters
        ----------
        n_calls : int
            Number of evaluations of the objective function.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Results of the optimization process.
        """
        # Generate multiples of base_channels that are valid for grouping
        n_values = [i for i in range(1, self.base_channels + 1) if self.base_channels % i == 0]
        # n_values = [180]

        # Define the search space
        space = [
            Categorical(n_values, name="n"),  # Multiples of the base number
            Real(0.1, 20000.0, name="alpha", prior="log-uniform")  # Alpha range for RidgeClassifier
        ]

        @use_named_args(space)
        def objective(**params):
            n = params["n"]
            alpha = params["alpha"]
            return self.evaluate_n_and_alpha(n, alpha, ch_treat, num_splits=num_splits)

        # Initialize progress bar
        with tqdm(total=n_calls, desc="Optimizing Channels and Alpha") as pbar:
            def progress_callback(res):
                """Update the progress bar after each iteration."""
                pbar.update(1)

            # Perform Bayesian Optimization with a callback for progress
            result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
                                 callback=[progress_callback])

        return result

    def optimize_tic(self, n_calls=50, random_state=42, num_splits=5):
        """
        Optimize the alpha parameter using Bayesian Optimization.

        Parameters
        ----------
        n_calls : int
            Number of evaluations of the objective function.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Results of the optimization process.
        """
        # Define the search space
        space = [Real(0.1, 20000.0, name="alpha", prior="log-uniform")]

        @use_named_args(space)
        def objective(**params):
            alpha = params["alpha"]
            return self.evaluate_alpha(alpha, num_splits=num_splits)

        # Initialize progress bar
        with tqdm(total=n_calls, desc="Optimizing Alpha") as pbar:
            def progress_callback(res):
                """Update the progress bar after each iteration."""
                pbar.update(1)

            # Perform Bayesian Optimization with a callback for progress
            result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
                                 callback=[progress_callback])

        return result



# class BayesianParamOptimizer:
#     def __init__(self, data, labels, n_channels, n_splits=5):
#         """
#         Initialize the Bayesian Weight Optimizer.
#
#         Parameters
#         ----------
#         data : np.ndarray
#             The input data with shape (samples, features, channels).
#         labels : np.ndarray
#             The labels for each sample.
#         n_channels : int
#             Number of m/z channels.
#         n_splits : int
#             Number of splits for cross-validation.
#         """
#         self.data = data
#         self.labels = labels
#         self.n_channels = n_channels
#         self.n_splits = n_splits
#
#     def evaluate_n(self, n):
#         """
#         Evaluate the balanced accuracy for a given number of aggregated channels.
#
#         Parameters
#         ----------
#         n : int
#             Number of aggregated channels.
#
#         Returns
#         -------
#         float
#             Negative mean balanced accuracy (to minimize).
#         """
#         group_size = self.n_channels // n
#         aggregated_data = np.zeros((self.data.shape[0], self.data.shape[1], n))
#
#         for i in range(n):
#             start_idx = i * group_size
#             end_idx = (i + 1) * group_size if i != n - 1 else self.n_channels
#             aggregated_data[:, :, i] = np.mean(self.data[:, :, start_idx:end_idx], axis=-1)
#
#         reshaped_data = aggregated_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
#         labels_expanded = np.repeat(self.labels, n)
#
#         accuracy_scores = []
#         ridge_classifier = RidgeClassifierCV(alphas=self.alpha_range, scoring='balanced_accuracy')
#
#         for _ in range(self.n_splits):
#             X_train, X_test, y_train, y_test = train_test_split(
#                 reshaped_data, labels_expanded, random_state=None
#             )
#             ridge_classifier.fit(X_train, y_train)
#             y_pred = ridge_classifier.predict(X_test)
#             balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#             accuracy_scores.append(balanced_accuracy)
#
#         return -np.mean(accuracy_scores)
#
#     def evaluate_n_and_alpha(self, n, alpha, num_splits=20):
#         """
#         Evaluate the balanced accuracy for a given number of aggregated channels and alpha.
#
#         Parameters
#         ----------
#         n : int
#             Number of aggregated channels.
#         alpha : float
#             Regularization strength for RidgeClassifier.
#
#         Returns
#         -------
#         float
#             Negative mean balanced accuracy (to minimize).
#         """
#         group_size = self.n_channels // n
#         remainder = self.n_channels % n  # Channels left after even grouping
#         aggregated_data = np.zeros((self.data.shape[0], self.data.shape[1], n))
#
#         for i in range(n):
#             start_idx = i * group_size
#             if i < remainder:
#                 # Distribute one extra channel to the first 'remainder' groups
#                 end_idx = start_idx + group_size + 1
#             else:
#                 # Remaining groups take only 'group_size' channels
#                 end_idx = start_idx + group_size
#
#             aggregated_data[:, :, i] = np.sum(self.data[:, :, start_idx:end_idx], axis=-1)
#
#         reshaped_data = aggregated_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
#         labels_expanded = np.repeat(self.labels, n)
#
#         balanced_accuracies = []
#
#         for _ in range(num_splits):
#             # Stratified train-test split
#             X_train, X_test, y_train, y_test = train_test_split(
#                 reshaped_data,
#                 labels_expanded,
#                 test_size=0.2,
#                 stratify=labels_expanded,
#                 random_state=None
#             )
#
#             # Train RidgeClassifier with the given alpha
#             ridge_classifier = RidgeClassifier(alpha=alpha)
#             ridge_classifier.fit(X_train, y_train)
#             y_pred = ridge_classifier.predict(X_test)
#
#             # Compute the balanced accuracy for this split
#             balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#             balanced_accuracies.append(balanced_accuracy)
#
#         # Return the negative mean balanced accuracy (to minimize)
#         return -np.mean(balanced_accuracies)
#
#     def optimize(self, n_calls=50, random_state=42, num_splits=5):
#         """
#         Optimize the number of channels and alpha using Bayesian Optimization.
#
#         Parameters
#         ----------
#         n_calls : int
#             Number of evaluations of the objective function.
#         random_state : int
#             Random seed for reproducibility.
#
#         Returns
#         -------
#         dict
#             Results of the optimization process.
#         """
#         space = [
#             Integer(1, self.n_channels, name="n"),
#             Real(0.1, 20000.0, name="alpha")
#         ]
#
#         @use_named_args(space)
#         def objective(**params):
#             n = params["n"]
#             alpha = params["alpha"]
#             return self.evaluate_n_and_alpha(n, alpha, num_splits=num_splits)
#
#         # Initialize progress bar
#         with tqdm(total=n_calls, desc="Optimizing Channels and Alpha") as pbar:
#             def progress_callback(res):
#                 """Update the progress bar after each iteration."""
#                 pbar.update(1)
#
#             # Perform Bayesian Optimization with a callback for progress
#             result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state,
#                                  callback=[progress_callback])
#
#         return result
#
#
#     # def evaluate_weights(self, weights):
#     #     """
#     #     Evaluate the balanced accuracy for given weights.
#     #
#     #     Parameters
#     #     ----------
#     #     weights : list
#     #         Weights for each channel.
#     #
#     #     Returns
#     #     -------
#     #     float
#     #         Negative mean balanced accuracy (to minimize).
#     #     """
#     #     weights = np.array(weights).reshape(1, 1, -1)
#     #     weighted_data = self.data * weights
#     #     reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
#     #     labels_expanded = np.repeat(self.labels, self.n_channels)
#     #
#     #     accuracy_scores = []
#     #     ridge_classifier = RidgeClassifierCV(alphas=self.alpha_range, scoring='balanced_accuracy')
#     #
#     #     for _ in range(self.n_splits):
#     #         X_train, X_test, y_train, y_test = train_test_split(
#     #             reshaped_data, labels_expanded, random_state=None
#     #         )
#     #         ridge_classifier.fit(X_train, y_train)
#     #         y_pred = ridge_classifier.predict(X_test)
#     #         balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
#     #         accuracy_scores.append(balanced_accuracy)
#     #
#     #     # Return negative accuracy (to minimize)
#     #     return -np.mean(accuracy_scores)
#
#     # def optimize(self, n_calls=50, random_state=42):
#     #     """
#     #     Optimize weights using Bayesian Optimization.
#     #
#     #     Parameters
#     #     ----------
#     #     n_calls : int
#     #         Number of evaluations of the objective function.
#     #     random_state : int
#     #         Random seed for reproducibility.
#     #
#     #     Returns
#     #     -------
#     #     dict
#     #         Results of the optimization process.
#     #     """
#     #     # Define the search space for weights (0 to 2 for each channel)
#     #     space = [Real(0, 2, name=f"w{i}") for i in range(self.n_channels)]
#     #
#     #     # Decorate the objective function with the search space
#     #     @use_named_args(space)
#     #     def objective(**weights):
#     #         weight_array = np.array([weights[f"w{i}"] for i in range(self.n_channels)])
#     #         return self.evaluate_weights(weight_array)
#     #
#     #     # Perform Bayesian Optimization
#     #     result = gp_minimize(objective, space, n_calls=n_calls, random_state=random_state)
#     #
#     #     return result



class CoordinateDescentOptimizer:
    def __init__(self, data, labels, alpha_range=None, n_splits=5, max_iter=100, tol=1e-4):
        """
        Initialize the optimizer.

        Parameters
        ----------
        data : np.ndarray
            The input data of shape (samples, features, channels).
        labels : np.ndarray
            The corresponding labels.
        alpha_range : list
            The range of alphas to test with RidgeClassifierCV.
        n_splits : int
            Number of cross-validation splits.
        max_iter : int
            Maximum number of iterations for coordinate descent.
        tol : float
            Convergence tolerance.
        """
        self.data = data
        self.labels = labels
        self.alpha_range = alpha_range if alpha_range is not None else [0.1, 1.0, 10.0, 100.0]
        self.n_splits = n_splits
        self.max_iter = max_iter
        self.tol = tol
        self.weights = np.ones(data.shape[2])  # Initialize weights to 1 (one per channel)

    def evaluate_weight(self, channel_idx):
        """
        Evaluate the balanced accuracy for a specific channel's weight.

        Parameters
        ----------
        channel_idx : int
            The index of the channel to optimize.

        Returns
        -------
        float
            Negative mean balanced accuracy (to minimize).
        """
        # Create a copy of weights, setting all other weights to 1
        channel_weights = np.ones(self.weights.shape)
        channel_weights[channel_idx] = self.weights[channel_idx]

        # Apply weights to the data for the specified channel
        weighted_data = self.data * channel_weights.reshape(1, 1, -1)
        reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
        labels_expanded = np.repeat(self.labels, self.data.shape[2])

        # Ridge Classifier with cross-validation
        ridge_classifier = RidgeClassifierCV(alphas=self.alpha_range, scoring='balanced_accuracy')
        accuracy_scores = []

        for _ in range(self.n_splits):
            X_train, X_test, y_train, y_test = train_test_split(
                reshaped_data, labels_expanded, random_state=None, stratify=labels_expanded
            )

            ridge_classifier.fit(X_train, y_train)
            y_pred = ridge_classifier.predict(X_test)
            balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
            accuracy_scores.append(balanced_accuracy)

        # Return negative accuracy to minimize
        return -np.mean(accuracy_scores)


    def optimize(self):
        """
        Optimize weights using coordinate descent.

        Returns
        -------
        np.ndarray
            The optimized weights.
        """
        for iteration in range(self.max_iter):
            previous_weights = self.weights.copy()

            for channel_idx in range(len(self.weights)):
                # Optimize weight for this channel
                best_weight = self.find_best_weight(channel_idx)
                self.weights[channel_idx] = best_weight

            # Check convergence
            weight_change = np.linalg.norm(self.weights - previous_weights)
            # Calculate accuracy with current weights
            weighted_data = self.data * self.weights.reshape(1, 1, -1)
            reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
            labels_expanded = np.repeat(self.labels, self.data.shape[2])
            accuracy_scores = []

            for _ in range(self.n_splits):
                X_train, X_test, y_train, y_test = train_test_split(reshaped_data, labels_expanded, random_state=None)
                ridge_classifier = RidgeClassifier()
                ridge_classifier.fit(X_train, y_train)
                y_pred = ridge_classifier.predict(X_test)
                accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            # Calculate accuracy with current weights
            weighted_data = self.data * self.weights.reshape(1, 1, -1)
            reshaped_data = weighted_data.transpose(2, 0, 1).reshape(-1, self.data.shape[1])
            labels_expanded = np.repeat(self.labels, self.data.shape[2])
            accuracy_scores = []

            for _ in range(self.n_splits):
                X_train, X_test, y_train, y_test = train_test_split(reshaped_data, labels_expanded, random_state=None)
                ridge_classifier = RidgeClassifier()
                ridge_classifier.fit(X_train, y_train)
                y_pred = ridge_classifier.predict(X_test)
                accuracy_scores.append(balanced_accuracy_score(y_test, y_pred))

            mean_accuracy = np.mean(accuracy_scores)

            print(f"Iteration {iteration + 1}, Weight Change: {weight_change:.6f}, Weights: {self.weights}, "
                  f"Mean Accuracy: {mean_accuracy:.4f}")

            if weight_change < self.tol:
                print("Converged!")
                break

        return self.weights

    def find_best_weight(self, channel_idx):
        """
        Perform a line search to find the best weight for a given channel.

        Parameters
        ----------
        channel_idx : int
            The index of the channel to optimize.

        Returns
        -------
        float
            The best weight for the given channel.
        """

        best_weight = self.weights[channel_idx]
        best_score = self.evaluate_weight(channel_idx)

        # Search over a range of weights (e.g., [-5, 5])
        for candidate_weight in np.linspace(0, 1, 20):
            self.weights[channel_idx] = candidate_weight
            score = self.evaluate_weight(channel_idx)

            if score < best_score:  # Minimize negative balanced accuracy
                best_weight = candidate_weight
                best_score = score

        return best_weight



def process_labels(labels, vintage):
    """
    Process a list of labels to extract relevant parts based on whether the data is vintage or not.

    Parameters
    ----------
    labels : list of str
        A list of label strings to be processed.
    vintage : bool
        If True, the function processes labels to extract a substring starting from the first digit
        found in each label (assuming vintage data formatting). If False, it processes labels to
        extract a single character or digit before the first digit found.

    Returns
    -------
    numpy.ndarray
        An array of processed labels.

    Notes
    -----
    This function is similar to the `_process_labels` method within the `Classifier` class, but
    it operates on an external list of labels rather than an instance attribute.
    """
    processed_labels = []

    # Iterate over each label in the provided list of labels
    for label in labels:
        # Search for the first digit in the label
        match = re.search(r'\d+', label)

        if vintage:
            # If processing vintage data, extract the substring starting from the first digit
            processed_labels.append(label[match.start():])
        else:
            # If not vintage, extract the character before the first digit
            if label[match.start() - 1] == '_':
                # If the character before the digit is an underscore, take the character before the underscore
                lb = label[match.start() - 2]
            else:
                # Otherwise, take the character directly before the first digit
                lb = label[match.start() - 1]
            processed_labels.append(lb)

    # Return the processed labels as a numpy array
    return np.array(processed_labels)


def assign_country_to_pinot_noir(original_keys):
    """
        Map wine sample keys to their corresponding country .

        This function takes a list of wine sample keys, where the first letter of each key represents
        the Chateau and returns a list of corresponding countries (Switzerland, US, or France).

        Parameters
        ----------
        original_keys : list of str
            A list of strings where each string is a wine sample key. The first letter of each key
            corresponds to a Chateau(e.g., 'C14', 'M08').

        Returns
        -------
        origine_keys : list of str
            A list of strings where each string is the corresponding country ('Switzerland',
            'US', 'France') of the wine sample based on the first letter of the key.

        Examples
        --------
        >>> original_keys = ['C14', 'M08', 'U08', 'D10', 'X13']
        >>> assign_country_to_pinot_noir(original_keys)
        ['France', 'Switzerland', 'US', 'France', 'US']

        Notes
        -----
        The first letter of the key is used to determine the country:
            - 'M', 'N', 'J', 'L', 'H' => Switzerland
            - 'U', 'X' => US
            - 'D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'K', 'W', 'Y' => France
        """
    # Dictionary to map letters to their origins
    letter_to_country = {
        # Switzerland
        'M': 'Switzerland',
        'N': 'Switzerland',
        'J': 'Switzerland',
        'L': 'Switzerland',
        'H': 'Switzerland',

        # US
        'U': 'US',
        'X': 'US',

        # France
        'D': 'France',
        'E': 'France',
        'Q': 'France',
        'P': 'France',
        'R': 'France',
        'Z': 'France',
        'C': 'France',
        'K': 'France',
        'W': 'France',
        'Y': 'France'
    }

    # Create a new list by mapping the first letter of each key to its "Origine"
    country_keys = [letter_to_country[key[0]] for key in original_keys]

    return country_keys


def assign_origin_to_pinot_noir(original_keys):
    """
    Map wine sample keys to their corresponding region of origin (Origine).

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding regions ("Origine") for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin (e.g., 'C14', 'M08').

    Returns
    -------
    origine_keys : list of str
        A list of strings where each string is the corresponding region of origin based on the
        first letter of the key.

    Examples
    --------
    >>> original_keys = ['C14', 'M08', 'U08', 'D10', 'X13']
    >>> assign_origin_to_pinot_noir(original_keys)
    ['Alsace', 'Neuchatel', 'Californie', 'Beaune', 'Oregon']

    Notes
    -----
    The first letter of the key is used to determine the specific region of origin:
        - 'M', 'N' => Neuchatel (Switzerland)
        - 'J', 'L' => Genève (Switzerland)
        - 'H' => Valais (Switzerland)
        - 'U' => Californie (US)
        - 'X' => Oregon (US)
        - 'D', 'E', 'Q', 'P', 'R', 'Z' => Beaune (France)
        - 'C', 'K', 'W', 'Y' => Alsace (France)
    """
    # Dictionary to map letters to their specific regions (Origine)
    letter_to_origine = {
        # Switzerland
        'M': 'Neuchatel',
        'N': 'Neuchatel',
        'J': 'Genève',
        'L': 'Genève',
        'H': 'Valais',

        # US
        'U': 'Californie',
        'X': 'Oregon',

        # France
        'D': 'Beaune',
        'E': 'Beaune',
        'Q': 'Beaune',
        'P': 'Beaune',
        'R': 'Beaune',
        'Z': 'Beaune',
        'C': 'Alsace',
        'K': 'Alsace',
        'W': 'Alsace',
        'Y': 'Alsace'
    }

    # Create a new list by mapping the first letter of each key to its specific "Origine"
    origin_keys = [letter_to_origine[key[0]] for key in original_keys]

    return origin_keys


def assign_continent_to_pinot_noir(original_keys):
    """
    Map wine sample keys to their corresponding continent.

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding continents for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin.

    Returns
    -------
    continent_keys : list of str
        A list of strings where each string is the corresponding continent based on the
        first letter of the key.

    Examples
    --------
    >>> original_keys = ['C14', 'M08', 'U08', 'D10', 'X13']
    >>> assign_continent_to_pinot_noir(original_keys)
    ['Europe', 'Europe', 'North America', 'Europe', 'North America']

    Notes
    -----
    The first letter of the key is used to determine the continent:
        - 'M', 'N', 'J', 'L', 'H' => Europe (Switzerland)
        - 'U', 'X' => North America (US)
        - 'D', 'E', 'Q', 'P', 'R', 'Z', 'C', 'K', 'W', 'Y' => Europe (France)
    """
    # Dictionary to map letters to their continents
    letter_to_continent = {
        # Switzerland (Europe)
        'M': 'Europe',
        'N': 'Europe',
        'J': 'Europe',
        'L': 'Europe',
        'H': 'Europe',

        # US (North America)
        'U': 'North America',
        'X': 'North America',

        # France (Europe)
        'D': 'Europe',
        'E': 'Europe',
        'Q': 'Europe',
        'P': 'Europe',
        'R': 'Europe',
        'Z': 'Europe',
        'C': 'Europe',
        'K': 'Europe',
        'W': 'Europe',
        'Y': 'Europe'
    }

    # Create a new list by mapping the first letter of each key to its continent
    continent_keys = [letter_to_continent[key[0]] for key in original_keys]

    return continent_keys


def assign_north_south_to_beaune(original_keys):
    """
    Map wine sample keys to either 'North Beaune (NB)' or 'South Beaune (SB)'.

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding regions ('North Beaune' or 'South Beaune') for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin.

    Returns
    -------
    beaune_region_keys : list of str
        A list of strings where each string is either 'North Beaune' or 'South Beaune' based on the
        first letter of the key.

    """
    if len(original_keys) != 61:
        raise ValueError(f"Incorrect wines passed. Input should be Beaume wines only")

    # Dictionary to map letters to North or South Beaune
    letter_to_beaune_region = {
        # North Beaune (NB) or Côte de Nuits
        'Q': 'NB',
        'R': 'NB',
        'Z': 'NB',

        # South Beaune (SB) or Côte de Beaune
        'D': 'SB',
        'E': 'SB',
        'P': 'SB',
    }

    # Create a new list by mapping the first letter of each key to North or South Beaune
    beaune_region_keys = [letter_to_beaune_region[key[0]] for key in original_keys]

    return beaune_region_keys


def assign_winery_to_pinot_noir(labels):
    """
    Assign the first letter of each label, which corresponds to the winery (Chateau)

    Parameters
    ----------
    labels : list of str
        A list of label strings.

    Returns
    -------
    first_letters : list of str
        A list of the first letters of each label.
    """
    # Create a list of the first letters of each label
    first_letters = [label[0] for label in labels]

    return first_letters


def assign_year_to_pinot_noir(labels):
    """
    Assign the last two letters  of each label, which corresponds to the year.

    Parameters
    ----------
    labels : list of str
        A list of label strings.

    Returns
    -------
    year : list of str
        A list of the years from each label.
    """
    # Create a list of the first letters of each label
    first_letters = [label[-2:] for label in labels]

    return first_letters

def assign_category_to_press_wine(labels):
    """
    Assigns categories (A, B, or C) to each wine label based on whether the letters
    'A', 'B', or 'C' appear immediately before a number in the label.

    Args:
        labels (dict_keys or list of str):
            A list of wine sample labels (or dictionary keys).

    Returns:
        list of str:
            A list of categories ('A', 'B', or 'C') corresponding to each label.

    Example:
        labels = ['Est22CSA1-1', 'Est22CSB1-1', 'Est22CSC1-1']
        assign_category_to_press_wine(labels)
        >>> ['A', 'B', 'C']
    """
    # Regex pattern to find 'A', 'B', or 'C' followed by a number
    pattern = re.compile(r'(A|B|C)(?=\d)')

    # Loop through each label, extract category, and store in the list
    categories = []
    for label in labels:
        match = pattern.search(label)
        if match:
            categories.append(match.group())  # Append 'A', 'B', or 'C'
        else:
            categories.append(None)  # If no match, append None or custom label

    return categories

def assign_composite_label_to_press_wine(labels):
    """
    Assigns composite labels (e.g., 'A1', 'B9', 'C2') to each wine label based on the
    letter 'A', 'B', or 'C' followed by a number in the label.

    Args:
        labels (dict_keys or list of str):
            A list of wine sample labels (or dictionary keys).

    Returns:
        list of str:
            A list of composite labels (e.g., 'A1', 'B9', 'C2') corresponding to each label.

    Example:
        labels = ['Est22CSA1-1', 'Est22CSB9-2', 'Est22CSC3-1']
        assign_composite_label_to_press_wine(labels)
        >>> ['A1', 'B9', 'C3']
    """
    # Regex pattern to find 'A', 'B', or 'C' followed by a number (e.g., 'A1', 'B9', 'C3')
    pattern = re.compile(r'([A-C])(\d+)')

    # Loop through each label, extract composite label, and store in the list
    composite_labels = []
    for label in labels:
        match = pattern.search(label)
        if match:
            # Combine category and number (e.g., 'A' + '1' = 'A1')
            composite_labels.append(match.group(1) + match.group(2))
        else:
            composite_labels.append(None)  # If no match, append None or custom label

    return composite_labels


def extract_year_from_samples(sample_names):
    """
    Extracts the year from a list of sample names.
    Assumes the first two-digit number after 'Est' represents the year (e.g., '23' → 2023).

    Parameters:
        sample_names (list of str): List of sample names.

    Returns:
        list of int or None: A list of extracted years (e.g., [2023, 2022, None]) corresponding to each sample.
    """
    import re
    pattern = re.compile(r'Est(\d{2})')

    years = []
    for sample in sample_names:
        match = pattern.search(sample)
        if match:
            year = int(match.group(1))
            years.append(str(2000 + year if year >= 20 else 1900 + year))  # Adjust century if needed
        else:
            years.append(None)  # Append None if no match is found

    return years


def greedy_channel_selection(
        data, labels, alpha=1.0, test_size=0.2, max_channels=None, num_splits=5, tolerated_no_improvement_steps=2,
        normalize=True, scaler_type='standard', random_seed=None, parallel=True, n_jobs=-1, corr_threshold=0.85):
    """
    Greedy Forward Selection with correlation-based filtering to remove highly correlated channels.
    """
    num_channels = data.shape[2]
    max_channels = max_channels or num_channels

    selected_channels = []
    remaining_channels = list(range(num_channels))
    accuracies_on_test = []

    print("Starting Greedy Channel Selection with Correlation Filtering...")

    model = RidgeClassifier(alpha=alpha)
    best_accuracy_overall = 0.0
    non_improving_steps = 0

    if random_seed is None:
        random_seed = np.random.randint(0, 1e6)

    rng = np.random.default_rng(random_seed)

    predefined_splits = []
    for _ in range(num_splits):
        train_idx, temp_idx = train_test_split(
            np.arange(len(labels)), test_size=test_size + test_size, stratify=labels, random_state=rng.integers(0, 1e6)
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=rng.integers(0, 1e6)
        )
        predefined_splits.append((train_idx, val_idx, test_idx))

    def evaluate_channel(ch):
        candidate_channels = selected_channels + [ch]
        split_accuracies_on_val = []
        split_accuracies_on_test = []

        for train_idx, val_idx, test_idx in predefined_splits:
            X_train = data[train_idx][:, :, candidate_channels].reshape(len(train_idx), -1)
            X_val = data[val_idx][:, :, candidate_channels].reshape(len(val_idx), -1)
            X_test = data[test_idx][:, :, candidate_channels].reshape(len(test_idx), -1)
            y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_val = scaler.transform(X_val)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred_val = model.predict(X_val)
            accuracy_val = balanced_accuracy_score(y_val, y_pred_val)
            split_accuracies_on_val.append(accuracy_val)

            y_pred_test = model.predict(X_test)
            accuracy_test = balanced_accuracy_score(y_test, y_pred_test)
            split_accuracies_on_test.append(accuracy_test)

        avg_accuracy_on_val = np.mean(split_accuracies_on_val)
        avg_accuracy_on_test = np.mean(split_accuracies_on_test)
        return ch, avg_accuracy_on_val, avg_accuracy_on_test

    # def remove_correlated_channels(new_channel):
    #     new_channel_data = data[:, :, new_channel].flatten()
    #     to_remove = []
    #     for ch in remaining_channels:
    #         if ch == new_channel:
    #             continue
    #         ch_data = data[:, :, ch].flatten()
    #         corr = np.corrcoef(new_channel_data, ch_data)[0, 1]
    #         if abs(corr) > corr_threshold:
    #             to_remove.append(ch)
    #     return to_remove

    def remove_correlated_channels(new_channel):
        """
        Optimized function to remove highly correlated channels using NumPy.
        Computes the correlation matrix once for all channels.
        """
        # Reshape data for correlation computation
        reshaped_data = data.reshape(data.shape[0] * data.shape[1], data.shape[2])
        correlation_matrix = np.corrcoef(reshaped_data, rowvar=False)

        # Find highly correlated channels to the newly added one
        correlated_indices = np.where(np.abs(correlation_matrix[new_channel]) > corr_threshold)[0]

        # Remove the selected channel itself from the correlated list
        correlated_indices = correlated_indices[correlated_indices != new_channel]

        return list(correlated_indices)


    for step in range(min(max_channels, num_channels)):
        if not remaining_channels:
            print("No channels left to evaluate.")
            break
        if parallel:
            results = Parallel(n_jobs=n_jobs)(delayed(evaluate_channel)(ch) for ch in remaining_channels)
        else:
            results = [evaluate_channel(ch) for ch in remaining_channels]

        if results:
            best_channel, best_step_accuracy_val, best_step_accuracy_test = max(results, key=lambda x: x[1])
        else:
            print("No valid channels found for evaluation. Stopping early.")
            break

        best_channel, best_step_accuracy_val, best_step_accuracy_test = max(results, key=lambda x: x[1])

        if best_channel is not None:
            selected_channels.append(best_channel)
            remaining_channels.remove(best_channel)
            accuracies_on_test.append(best_step_accuracy_test)

            print(
                f"Step {step + 1}: Added Channel {best_channel} - Validation Accuracy: {best_step_accuracy_val:.4f}, Test Accuracy: {best_step_accuracy_test:.4f}")

            if best_step_accuracy_test > best_accuracy_overall:
                best_accuracy_overall = best_step_accuracy_test
                non_improving_steps = 0
            else:
                non_improving_steps += 1

            # if corr_threshold >= 1:
            #     correlated_channels = []
            # else:
            #     correlated_channels = remove_correlated_channels(best_channel)
            # for ch in correlated_channels:
            #     remaining_channels.remove(ch)
            #     print(f"Removed correlated channel {ch} (correlation > {corr_threshold})")

            if corr_threshold < 1:
                correlated_channels = remove_correlated_channels(best_channel)
                for ch in correlated_channels:
                    if ch in remaining_channels:
                        remaining_channels.remove(ch)
                        print(f"Removed correlated channel {ch} (correlation > {corr_threshold})")

        if non_improving_steps >= tolerated_no_improvement_steps:
            print(f"No improvement in the last {tolerated_no_improvement_steps} steps. Stopping selection.")
            break

    print("Greedy Channel Selection Completed.")

    # plt.plot(range(1, len(accuracies_on_test) + 1), accuracies_on_test, marker='o')
    # plt.xlabel('Number of Channels')
    # plt.ylabel('Test Accuracy')
    # plt.title('Test Accuracy During Channel Selection')
    # plt.grid(True)
    # plt.show()

    return selected_channels, accuracies_on_test

def split_by_split_greedy_channel_selection(
    data, labels, alpha=1.0, num_splits=5, test_size=0.2, corr_threshold=0.85,
    max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
    parallel=True, n_jobs=-1):
    """
    Perform split-by-split greedy channel selection, then evaluate a fixed ordered channel set across all splits.
    """

    if random_seed is None:
        random_seed = np.random.randint(0, 1e6)
    rng = np.random.default_rng(random_seed)

    split_selected_channels = []
    predefined_splits = []
    channel_selection_orders = defaultdict(list)
    validation_accuracy_per_step = []

    def evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val):
        """ Train RidgeClassifier with the given channel and evaluate accuracy. """
        candidate_channels = selected_channels + [ch]
        X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
        X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)

        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred_val = model.predict(X_val_subset)
        return ch, balanced_accuracy_score(y_val, y_pred_val)

    def compute_accuracy(X_train, y_train, X_test, y_test, selected_channels):
        """ Train RidgeClassifier on fixed channels and evaluate on test data. """
        X_train_subset = X_train[:, :, selected_channels].reshape(len(X_train), -1)
        X_test_subset = X_test[:, :, selected_channels].reshape(len(X_test), -1)

        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_test_subset)
        return balanced_accuracy_score(y_test, y_pred)

    # ---- STEP 1: CREATE TRAIN-VALIDATION-TEST SPLITS ---- #
    for split in range(num_splits):
        train_idx, temp_idx = train_test_split(
            np.arange(len(labels)), test_size=test_size + test_size, stratify=labels, random_state=rng.integers(0, 1e6)
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, stratify=labels[temp_idx], random_state=rng.integers(0, 1e6)
        )
        predefined_splits.append((train_idx, val_idx, test_idx))

    # ---- STEP 2: GREEDY CHANNEL SELECTION FOR EACH SPLIT ---- #
    for split, (train_idx, val_idx, test_idx) in enumerate(predefined_splits):
        print(f'Split: {split + 1}')
        X_train, X_val, X_test = data[train_idx], data[val_idx], data[test_idx]
        y_train, y_val, y_test = labels[train_idx], labels[val_idx], labels[test_idx]

        if normalize:
            X_train, scaler = normalize_data(X_train, scaler=scaler_type)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

        remaining_channels = list(range(data.shape[2]))
        selected_channels = []
        validation_accuracies = []

        for step in range(max_channels):
            if not remaining_channels:
                break

            if parallel:
                results = Parallel(n_jobs=n_jobs, backend='threading')(
                    delayed(evaluate_channel)(ch, selected_channels, X_train, y_train, X_val, y_val)
                    for ch in remaining_channels
                )
            else:
                results = [
                    evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val)
                    for ch in remaining_channels
                ]

            best_channel, best_accuracy = max(results, key=lambda x: x[1])

            selected_channels.append(best_channel)
            remaining_channels.remove(best_channel)

            # Record selection order of each channel
            channel_selection_orders[best_channel].append(step + 1)

            validation_accuracies.append(best_accuracy)

        validation_accuracy_per_step.append(validation_accuracies)
        split_selected_channels.append(selected_channels)

    # ---- STEP 3: DETERMINE FIXED SET OF CHANNELS ---- #
    channel_counter = Counter()
    for split in split_selected_channels:
        for ch in split:
            channel_counter[ch] += 1

    most_common_channels = [ch for ch, _ in channel_counter.most_common(max_channels)]

    # ---- STEP 4: ORDER CHANNELS BASED ON MEDIAN SELECTION ORDER ---- #
    ordered_channels = sorted(
        most_common_channels,
        key=lambda ch: np.median(channel_selection_orders[ch])
    )

    print(f"Fixed ordered set of channels: {ordered_channels}")

    # ---- STEP 5: RE-EVALUATE FIXED ORDERED SET ACROSS SPLITS ---- #
    test_accuracies_per_step = []

    for split, (train_idx, val_idx, test_idx) in enumerate(predefined_splits):
        X_train, X_test = data[train_idx], data[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if normalize:
            X_train, scaler = normalize_data(X_train, scaler=scaler_type)
            X_test = scaler.transform(X_test)

        test_accuracies = []
        for i in range(1, len(ordered_channels) + 1):
            test_accuracy = compute_accuracy(X_train, y_train, X_test, y_test, ordered_channels[:i])
            test_accuracies.append(test_accuracy)

        test_accuracies_per_step.append(test_accuracies)

    avg_validation_accuracies = np.nanmean(np.array([np.pad(acc, (0, max_channels - len(acc)), 'constant', constant_values=np.nan) for acc in validation_accuracy_per_step]), axis=0)
    avg_test_accuracies = np.nanmean(np.array([np.pad(acc, (0, max_channels - len(acc)), 'constant', constant_values=np.nan) for acc in test_accuracies_per_step]), axis=0)

    # ---- STEP 6: PLOT ACCURACIES AS NUMBER OF CHANNELS INCREASES ---- #
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_channels + 1), avg_validation_accuracies, marker='o', linestyle='-', label="Validation Accuracy")
    plt.plot(range(1, max_channels + 1), avg_test_accuracies, marker='s', linestyle='--', label="Test Accuracy")
    plt.xlabel("Number of Selected Channels")
    plt.ylabel("Balanced Accuracy")
    plt.title("Validation and Test Accuracy as Number of Channels Increases")
    plt.legend()
    plt.grid()
    plt.show()

    return ordered_channels, avg_validation_accuracies, avg_test_accuracies


def greedy_nested_cv_channel_selection(
        data, labels, alpha=1.0, num_outer_splits=5, num_outer_repeats=3, inner_cv_folds=3,
        max_channels=40, normalize=True, scaler_type='standard', random_seed=None, parallel=True, n_jobs=-1, method='concatenation'):
    """
    Perform nested cross-validation (CV) for greedy channel selection.
    Tracks validation and test accuracy for each added channel.

    Returns:
        - selected_channels (list): Ordered list of selected channels
        - avg_validation_accuracies (array): Mean validation accuracy per step
        - avg_test_accuracies (array): Mean test accuracy per step
    """
    import numpy as np
    from collections import Counter
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold, BaseCrossValidator, StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    import matplotlib
    # matplotlib.use("TkAgg")

    if random_seed is None:
        random_seed = np.random.randint(0, int(1e6))
    rng = np.random.default_rng(random_seed)

    all_selected_channels = []
    all_test_accuracy_per_step = []
    all_validation_accuracy_per_step = []

    class LeaveOneFromEachClassCV(BaseCrossValidator):
        """
        Custom cross-validator that, in each fold, leaves one sample per class as the test set.
        The number of folds is determined by the minimum number of samples among all classes.
        """
        def __init__(self, shuffle=True, random_state=None):
            self.shuffle = shuffle
            self.random_state = random_state

        def get_n_splits(self, X, y, groups=None):
            _, counts = np.unique(y, return_counts=True)
            return int(np.min(counts))

        def split(self, X, y, groups=None):
            indices_by_class = {}
            for idx, label in enumerate(y):
                indices_by_class.setdefault(label, []).append(idx)
            rng_local = np.random.default_rng(self.random_state)
            for label in indices_by_class:
                if self.shuffle:
                    rng_local.shuffle(indices_by_class[label])
            n_splits = self.get_n_splits(X, y)
            for split in range(n_splits):
                test_indices = []
                for label, indices in indices_by_class.items():
                    test_indices.append(indices[split])
                test_indices = np.array(test_indices)
                train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                yield train_indices, test_indices

    class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
        """
        Custom cross-validator that randomly selects one sample per class as the test set,
        and repeats the process a specified number of times.
        """

        def __init__(self, n_repeats=50, shuffle=True, random_state=None):
            self.n_repeats = n_repeats
            self.shuffle = shuffle
            self.random_state = random_state

        def get_n_splits(self, X, y, groups=None):
            return self.n_repeats

        def split(self, X, y, groups=None):
            indices_by_class = {}
            for idx, label in enumerate(y):
                indices_by_class.setdefault(label, []).append(idx)

            rng = np.random.default_rng(self.random_state)
            for _ in range(self.n_repeats):
                test_indices = []
                for label, indices in indices_by_class.items():
                    if self.shuffle:
                        chosen = rng.choice(indices, size=1, replace=False)
                    else:
                        chosen = [indices[0]]
                    test_indices.extend(chosen)
                test_indices = np.array(test_indices)
                train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                yield train_indices, test_indices

    # def evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val):
    #     candidate_channels = selected_channels + [ch]
    #     X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
    #     X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
    #
    #     model = RidgeClassifier(alpha=alpha)
    #     model.fit(X_train_subset, y_train)
    #     y_pred_val = model.predict(X_val_subset)
    #     return ch, balanced_accuracy_score(y_val, y_pred_val)

    def evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val, method='concatenation'):
        candidate_channels = selected_channels + [ch]
        if method == "concatenation":
            X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
            X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
        elif method == "average":
            #  Average into a single TIC per sample
            X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
            X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)

        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred_val = model.predict(X_val_subset)
        return ch, balanced_accuracy_score(y_val, y_pred_val)


    def scale_data(X_train, X_test, scaler_type='standard'):
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
        return X_train, X_test

    model = RidgeClassifier(alpha=alpha)
    # model = LogisticRegression(solver='liblinear', random_state=42)

    # Loop over outer repetitions
    for repeat in range(num_outer_repeats):
        print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
        # outer_cv = StratifiedKFold(n_splits=num_outer_splits, shuffle=True, random_state=rng.integers(0, int(1e6)))
        outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rng.integers(0, int(1e6)))

        # Temporary lists to hold metrics for this repetition
        rep_selected_channels = []
        rep_validation_accuracy_per_step = []
        rep_test_accuracy_per_step = []

        for split, (train_val_idx, test_idx) in enumerate(outer_cv.split(data, labels)):
            print(f"▶️ Split {split + 1}/{num_outer_splits}")
            X_train_val, X_test = data[train_val_idx], data[test_idx]
            if normalize:
                X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
            y_train_val, y_test = labels[train_val_idx], labels[test_idx]

            # Use the custom inner CV (for example, LeaveOneFromEachClassCV)
            # inner_cv = LeaveOneFromEachClassCV(shuffle=True, random_state=rng.integers(0, int(1e6)))
            inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True, random_state=rng.integers(0, int(1e6)))

            step_selected_channels = []
            split_validation_accuracies = []
            split_test_accuracies = []

            for step in range(max_channels):
                if (step + 1) % 1 == 0 or (step + 1) == max_channels:
                    print(f"\u23E9 m/z channel {step + 1}/{max_channels}")
                remaining_channels = [ch for ch in range(data.shape[2]) if ch not in step_selected_channels]
                if not remaining_channels:
                    break

                best_channel_per_fold = []
                val_accuracies_per_fold = []

                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val):
                    X_inner_train, X_inner_val = X_train_val[inner_train_idx], X_train_val[inner_val_idx]
                    y_inner_train, y_inner_val = y_train_val[inner_train_idx], y_train_val[inner_val_idx]

                    if parallel:
                        results = Parallel(n_jobs=n_jobs, backend='loky')(
                            delayed(evaluate_channel)(ch, step_selected_channels, X_inner_train, y_inner_train,
                                                      X_inner_val, y_inner_val)
                            for ch in remaining_channels
                        )
                    else:
                        results = [
                            evaluate_channel(ch, step_selected_channels, X_inner_train, y_inner_train, X_inner_val,
                                             y_inner_val)
                            for ch in remaining_channels
                        ]

                    best_channel, best_accuracy = max(results, key=lambda x: x[1])
                    best_channel_per_fold.append(best_channel)
                    val_accuracies_per_fold.append(best_accuracy)

                channel_counts = Counter(best_channel_per_fold)
                channel_frequencies = channel_counts.keys()

                channel_avg_accuracy = {
                    ch: np.mean([acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
                                if ch_fold == ch])
                    for ch in channel_frequencies
                }

                best_channel = max(channel_frequencies, key=lambda ch: (channel_counts[ch], channel_avg_accuracy[ch]))
                step_selected_channels.append(best_channel)

                accuracies_for_best_channel = [acc for ch, acc in zip(best_channel_per_fold, val_accuracies_per_fold) if
                                               ch == best_channel]
                # split_validation_accuracies.append(np.mean(val_accuracies_per_fold))
                split_validation_accuracies.append(np.mean(accuracies_for_best_channel))

                X_train_subset = X_train_val[:, :, step_selected_channels].reshape(len(X_train_val), -1)
                X_test_subset = X_test[:, :, step_selected_channels].reshape(len(X_test), -1)

                model.fit(X_train_subset, y_train_val)
                y_pred = model.predict(X_test_subset)
                split_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))

        rep_selected_channels.append(step_selected_channels)
        rep_validation_accuracy_per_step.append(split_validation_accuracies)
        rep_test_accuracy_per_step.append(split_test_accuracies)

        # Compute averages for this repetition over all splits
        rep_avg_validation_accuracies = np.mean(rep_validation_accuracy_per_step, axis=0)
        rep_std_validation_accuracies = np.std(rep_validation_accuracy_per_step, axis=0)
        rep_avg_test_accuracies = np.mean(rep_test_accuracy_per_step, axis=0)
        rep_std_test_accuracies = np.std(rep_test_accuracy_per_step, axis=0)

        # Append repetition-level results to the global lists
        all_selected_channels.extend(rep_selected_channels)
        all_validation_accuracy_per_step.extend(rep_validation_accuracy_per_step)
        all_test_accuracy_per_step.extend(rep_test_accuracy_per_step)

        # # Plot for this repetition (non-blocking or save to file)
        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.plot(range(1, len(rep_avg_validation_accuracies) + 1), rep_avg_validation_accuracies,
        #         marker='s', linestyle='-', label='Validation Accuracy')
        # ax.fill_between(range(1, len(rep_avg_validation_accuracies) + 1),
        #                 rep_avg_validation_accuracies - rep_std_validation_accuracies,
        #                 rep_avg_validation_accuracies + rep_std_validation_accuracies,
        #                 alpha=0.2)
        # ax.plot(range(1, len(rep_avg_test_accuracies) + 1), rep_avg_test_accuracies,
        #         marker='o', linestyle='--', label='Test Accuracy')
        # ax.fill_between(range(1, len(rep_avg_test_accuracies) + 1),
        #                 rep_avg_test_accuracies - rep_std_test_accuracies,
        #                 rep_avg_test_accuracies + rep_std_test_accuracies,
        #                 alpha=0.2)
        # # Annotate each test accuracy point with its corresponding channel
        # for i, txt in enumerate(step_selected_channels):
        #     ax.annotate(str(txt), (i + 1, split_test_accuracies[i]), textcoords="offset points", xytext=(0, 5), ha='center')
        #
        # ax.set_xlabel("Number of Selected Channels")
        # ax.set_ylabel("Balanced Accuracy")
        # ax.set_title(f"Validation and Test Accuracy Progression (Repetition {repeat + 1})")
        # ax.legend()
        # ax.grid()
        # # Option 1: Non-blocking display
        # plt.show(block=False)
        # plt.pause(0.5)
        # # Option 2: Save to file and close the figure
        # # plt.savefig(f"rep_{repeat + 1}_plot.png")
        # # plt.close(fig)

    # After all repetitions, compute global averages if needed
    avg_test_accuracies = np.mean(all_test_accuracy_per_step, axis=0)
    avg_validation_accuracies = np.mean(all_validation_accuracy_per_step, axis=0)

    # Global plot if desired
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(avg_validation_accuracies) + 1), avg_validation_accuracies, marker='s', linestyle='-',
            label='Validation Accuracy')
    ax.fill_between(range(1, len(avg_validation_accuracies) + 1),
                    np.array(avg_validation_accuracies) - np.array(np.std(all_validation_accuracy_per_step, axis=0)),
                    np.array(avg_validation_accuracies) + np.array(np.std(all_validation_accuracy_per_step, axis=0)),
                    alpha=0.2)
    ax.plot(range(1, len(avg_test_accuracies) + 1), avg_test_accuracies, marker='o', linestyle='--', label='Test Accuracy')
    ax.fill_between(range(1, len(avg_test_accuracies) + 1),
                    np.array(avg_test_accuracies) - np.array(np.std(all_test_accuracy_per_step, axis=0)),
                    np.array(avg_test_accuracies) + np.array(np.std(all_test_accuracy_per_step, axis=0)),
                    alpha=0.2)

    # Final selection of channels (same as before)
    final_selected_channels = sorted(
        Counter([ch for sublist in all_selected_channels for ch in sublist]).keys(),
        key=lambda ch: (
            -Counter([ch for sublist in all_selected_channels for ch in sublist])[ch],
            -np.mean([acc for step_channels, accs in zip(all_selected_channels, all_test_accuracy_per_step)
                      if ch in step_channels for acc in accs]),
            np.std([acc for step_channels, accs in zip(all_selected_channels, all_test_accuracy_per_step)
                    if ch in step_channels for acc in accs])
        )
    )[:max_channels]

    # Annotate each test accuracy point with its corresponding channel
    for i, txt in enumerate(final_selected_channels):
        ax.annotate(txt, (i + 1, avg_test_accuracies[i]), textcoords="offset points", xytext=(0, 5), ha='center')
    ax.set_xlabel("Number of Selected Channels")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title("Global Validation and Test Accuracy Progression")
    ax.legend()
    ax.grid()
    plt.show()

    return final_selected_channels, avg_test_accuracies, avg_validation_accuracies


# def greedy_nested_cv_channel_selection_snr(
# #         data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
# #         max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
# #         parallel=True, n_jobs=-1, method='concatenation'):
# #     """
# #     Perform nested CV for greedy channel selection using SNR-based candidate selection.
# #     Uses a single outer split per repetition.
# #
# #     After each outer repetition is completed, the global average performance (over all repetitions so far)
# #     is recalculated and the plot is updated showing the performance for all greedy elimination steps.
# #     In the global aggregation, for each greedy step, candidate channels that appear in fewer than min_freq
# #     repetitions are discarded. Among the remaining, the candidate with the highest average SNR (average
# #     validation accuracy divided by standard deviation) is chosen, and only the accuracies from repetitions
# #     selecting that channel are aggregated.
# #
# #     Returns:
# #         - all_selected_channels (list): List of removal sequences (one per outer repetition)
# #         - all_test_accuracy_per_step (list): List of test accuracy lists (one per outer repetition)
# #         - all_validation_accuracy_per_step (list): List of validation accuracy lists (one per outer repetition)
# #     """
# #     import numpy as np
# #     from sklearn.linear_model import RidgeClassifier
# #     from sklearn.metrics import balanced_accuracy_score
# #     from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
# #     from sklearn.preprocessing import StandardScaler, MinMaxScaler
# #     from joblib import Parallel, delayed
# #     import matplotlib.pyplot as plt
# #
# #     # Set random seed if not provided.
# #     if random_seed is None:
# #         random_seed = np.random.randint(0, int(1e6))
# #     rng = np.random.default_rng(random_seed)
# #
# #     # Global accumulators across outer repetitions.
# #     all_selected_channels = []  # Each element is a list (greedy elimination sequence) from one outer repetition.
# #     all_test_accuracy_per_step = []  # Each element is a list of test accuracies (per step) from one rep.
# #     all_validation_accuracy_per_step = []  # Each element is a list of validation accuracies (per step) from one rep.
# #
# #     # ----------------------------------
# #     # Custom CV classes.
# #     # ----------------------------------
# #     class LeaveOneFromEachClassCV(BaseCrossValidator):
# #         def __init__(self, shuffle=True, random_state=None):
# #             self.shuffle = shuffle
# #             self.random_state = random_state
# #
# #         def get_n_splits(self, X, y, groups=None):
# #             _, counts = np.unique(y, return_counts=True)
# #             return int(np.min(counts))
# #
# #         def split(self, X, y, groups=None):
# #             indices_by_class = {}
# #             for idx, label in enumerate(y):
# #                 indices_by_class.setdefault(label, []).append(idx)
# #             rng_local = np.random.default_rng(self.random_state)
# #             for label in indices_by_class:
# #                 if self.shuffle:
# #                     rng_local.shuffle(indices_by_class[label])
# #             n_splits = self.get_n_splits(X, y)
# #             for split in range(n_splits):
# #                 test_indices = []
# #                 for label, indices in indices_by_class.items():
# #                     test_indices.append(indices[split])
# #                 test_indices = np.array(test_indices)
# #                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
# #                 yield train_indices, test_indices
# #
# #     class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
# #         def __init__(self, n_repeats=50, shuffle=True, random_state=None):
# #             self.n_repeats = n_repeats
# #             self.shuffle = shuffle
# #             self.random_state = random_state
# #
# #         def get_n_splits(self, X, y, groups=None):
# #             return self.n_repeats
# #
# #         def split(self, X, y, groups=None):
# #             indices_by_class = {}
# #             for idx, label in enumerate(y):
# #                 indices_by_class.setdefault(label, []).append(idx)
# #             rng_local = np.random.default_rng(self.random_state)
# #             for _ in range(self.n_repeats):
# #                 test_indices = []
# #                 for label, indices in indices_by_class.items():
# #                     if self.shuffle:
# #                         chosen = rng_local.choice(indices, size=1, replace=False)
# #                     else:
# #                         chosen = [indices[0]]
# #                     test_indices.extend(chosen)
# #                 test_indices = np.array(test_indices)
# #                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
# #                 yield train_indices, test_indices
# #
# #     # ----------------------------------
# #     # Helper functions.
# #     # ----------------------------------
# #
# #     def evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val, method=method):
# #         candidate_channels = selected_channels + [ch]
# #         if method == "concatenation":
# #             X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
# #             X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
# #         elif method == "average":
# #             X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
# #             X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)
# #         model = RidgeClassifier(alpha=alpha)
# #         model.fit(X_train_subset, y_train)
# #         y_pred_val = model.predict(X_val_subset)
# #         return ch, balanced_accuracy_score(y_val, y_pred_val)
# #
# #     def evaluate_candidate_incremental(ch, cum_sum_train, cum_sum_val, count,
# #                                        X_train, y_train, X_val, y_val, alpha):
# #         """
# #         Evaluate candidate channel 'ch' using precomputed cumulative sums.
# #
# #         Instead of re‑summing the selected channels every time, we use the cached
# #         cumulative sums (cum_sum_train and cum_sum_val) and simply add the candidate's data.
# #
# #         Parameters:
# #           - ch: Candidate channel index.
# #           - cum_sum_train (np.ndarray): Precomputed sum over selected channels for training data.
# #           - cum_sum_val (np.ndarray): Precomputed sum over selected channels for validation data.
# #           - count (int): Number of channels in the cumulative sum.
# #           - X_train, X_val: Data arrays of shape (n_samples, n_time, n_channels).
# #           - y_train, y_val: Corresponding labels.
# #           - alpha: Regularization parameter for the RidgeClassifier.
# #
# #         Returns:
# #           - (ch, balanced_validation_accuracy)
# #         """
# #         # Compute the average for training data by adding the candidate channel
# #         # and dividing by count+1 (i.e. the new total number of channels)
# #         X_train_candidate = (cum_sum_train + X_train[:, :, ch]) / (count + 1)
# #         X_val_candidate = (cum_sum_val + X_val[:, :, ch]) / (count + 1)
# #
# #         # Train and evaluate the classifier
# #         model = RidgeClassifier(alpha=alpha)
# #         model.fit(X_train_candidate, y_train)
# #         y_pred_val = model.predict(X_val_candidate)
# #         return ch, balanced_accuracy_score(y_val, y_pred_val)
# #
# #     def evaluate_candidate_average_single(i, candidate_channels, avg_train, avg_val, y_train, y_val, alpha):
# #         """
# #         Evaluate a single candidate channel (used in batch evaluation for the average method)
# #         and intended to be called in parallel.
# #         """
# #         # Extract candidate's averaged features.
# #         X_train_candidate = avg_train[:, :, i]
# #         X_val_candidate = avg_val[:, :, i]
# #         model = RidgeClassifier(alpha=alpha)
# #         model.fit(X_train_candidate, y_train)
# #         y_pred_val = model.predict(X_val_candidate)
# #         score = balanced_accuracy_score(y_val, y_pred_val)
# #         return candidate_channels[i], score
# #
# #     def evaluate_candidates_average_batch(cum_sum_train, cum_sum_val, count,
# #                                           candidate_channels, X_train, y_train, X_val, y_val, alpha, parallel_flag=True, n_jobs_flag=-1):
# #         """
# #         Batch-evaluate multiple candidate channels for the "average" method.
# #         Uses broadcasting to compute the averaged features for all candidates,
# #         then evaluates each candidate in parallel.
# #         """
# #         n_train = X_train.shape[0]
# #         n_val = X_val.shape[0]
# #         # Extract candidate channel data.
# #         candidate_train = X_train[:, :, candidate_channels]  # shape: (n_train, n_time, n_candidates)
# #         candidate_val = X_val[:, :, candidate_channels]        # shape: (n_val, n_time, n_candidates)
# #         # Compute averaged features with broadcasting.
# #         avg_train = (cum_sum_train[..., None] + candidate_train) / (count + 1)
# #         avg_val = (cum_sum_val[..., None] + candidate_val) / (count + 1)
# #         n_candidates = avg_train.shape[2]
# #         # Evaluate each candidate in parallel.
# #         if parallel_flag:
# #             results = Parallel(n_jobs=n_jobs_flag, backend='loky')(
# #                 delayed(evaluate_candidate_average_single)(i, candidate_channels, avg_train, avg_val, y_train, y_val, alpha)
# #                 for i in range(n_candidates)
# #             )
# #         else:
# #             results = [evaluate_candidate_average_single(i, candidate_channels, avg_train, avg_val, y_train, y_val, alpha)
# #                        for i in range(n_candidates)]
# #         return results
# #
# #
# #     def scale_data(X_train, X_test, scaler_type='standard'):
# #         if scaler_type == 'standard':
# #             scaler = StandardScaler()
# #         elif scaler_type == 'minmax':
# #             scaler = MinMaxScaler()
# #         else:
# #             raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
# #         X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
# #         X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
# #         return X_train, X_test
# #
# #     model = RidgeClassifier(alpha=alpha)
# #
# #     # ----------------------------------
# #     # Set up interactive plotting (update after every outer repetition).
# #     # ----------------------------------
# #     plt.ion()
# #     fig, ax = plt.subplots(figsize=(10, 6))
# #
# #     # ----------------------------------
# #     # Outer repetition loop.
# #     # ----------------------------------
# #     for repeat in range(num_outer_repeats):
# #         print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
# #         # Use a single outer split per repetition.
# #         outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
# #                                           random_state=rng.integers(0, int(1e6)))
# #         train_val_idx, test_idx = next(outer_cv.split(data, labels))
# #         X_train_val, X_test = data[train_val_idx], data[test_idx]
# #         if normalize:
# #             X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
# #         y_train_val, y_test = labels[train_val_idx], labels[test_idx]
# #
# #         # Set up inner CV.
# #         inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
# #                                                    random_state=rng.integers(0, int(1e6)))
# #
# #         step_selected_channels = []  # Greedy channel sequence for this repetition.
# #         rep_validation_accuracies = []  # Validation accuracy per greedy step.
# #         rep_test_accuracies = []  # Test accuracy per greedy step.
# #
# #         # Greedy selection loop.
# #         for step in range(max_channels):
# #             print(f"Selecting channel {step + 1}/{max_channels}")
# #             remaining_channels = [ch for ch in range(data.shape[2]) if ch not in step_selected_channels]
# #             if not remaining_channels:
# #                 break
# #
# #             best_channel_per_fold = []
# #             val_accuracies_per_fold = []
# #
# #             # For each inner CV fold, evaluate all remaining candidates.
# #             for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val):
# #                 X_inner_train, X_inner_val = X_train_val[inner_train_idx], X_train_val[inner_val_idx]
# #                 y_inner_train, y_inner_val = y_train_val[inner_train_idx], y_train_val[inner_val_idx]
# #                 if method == "concatenation":
# #                     # Existing branch for "concatenation" remains unchanged:
# #                     if parallel:
# #                         results = Parallel(n_jobs=n_jobs, backend='loky')(
# #                             delayed(evaluate_channel)(
# #                                 ch, step_selected_channels,
# #                                 X_inner_train, y_inner_train,
# #                                 X_inner_val, y_inner_val,
# #                                 method=method
# #                             ) for ch in remaining_channels
# #                         )
# #                     else:
# #                         results = [evaluate_channel(
# #                             ch, step_selected_channels,
# #                             X_inner_train, y_inner_train,
# #                             X_inner_val, y_inner_val,
# #                             method=method
# #                         ) for ch in remaining_channels]
# #
# #                 elif method == "average":
# #                     # Precompute the cumulative sum for the already selected channels once:
# #                     if step_selected_channels:
# #                         cum_sum_train = np.sum(X_inner_train[:, :, step_selected_channels], axis=2)
# #                         cum_sum_val = np.sum(X_inner_val[:, :, step_selected_channels], axis=2)
# #                         count = len(step_selected_channels)
# #                     else:
# #                         # No channels selected yet: use zeros so that adding a candidate gives its own values.
# #                         cum_sum_train = np.zeros_like(X_inner_train[:, :, 0])
# #                         cum_sum_val = np.zeros_like(X_inner_val[:, :, 0])
# #                         count = 0
# #
# #                     # Now, for each candidate channel, we only need one vector addition and one division.
# #                     results = evaluate_candidates_average_batch(
# #                         cum_sum_train, cum_sum_val, count,
# #                         remaining_channels,
# #                         X_inner_train, y_inner_train,
# #                         X_inner_val, y_inner_val,
# #                         alpha,
# #                         parallel_flag=parallel,
# #                         n_jobs_flag=n_jobs
# #                     )
# #                     # if parallel:
# #                     #     results = Parallel(n_jobs=n_jobs, backend='loky')(
# #                     #         delayed(evaluate_candidate_incremental)(
# #                     #             ch, cum_sum_train, cum_sum_val, count,
# #                     #             X_inner_train, y_inner_train,
# #                     #             X_inner_val, y_inner_val, alpha
# #                     #         ) for ch in remaining_channels
# #                     #     )
# #                     # else:
# #                     #     results = [evaluate_candidate_incremental(
# #                     #         ch, cum_sum_train, cum_sum_val, count,
# #                     #         X_inner_train, y_inner_train,
# #                     #         X_inner_val, y_inner_val, alpha
# #                     #     ) for ch in remaining_channels]
# #
# #                 # Select the best candidate from this fold.
# #                 best_channel, best_accuracy = max(results, key=lambda x: x[1])
# #                 best_channel_per_fold.append(best_channel)
# #                 val_accuracies_per_fold.append(best_accuracy)
# #
# #             # Compute SNR for candidate channels across inner folds.
# #             epsilon = 1e-8
# #             candidate_channels = list(set(best_channel_per_fold))
# #             channel_avg_accuracy = {
# #                 ch: np.mean([acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
# #                              if ch_fold == ch])
# #                 for ch in candidate_channels
# #             }
# #             channel_std_accuracy = {
# #                 ch: np.std([acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
# #                             if ch_fold == ch])
# #                 for ch in candidate_channels
# #             }
# #             channel_snr = {
# #                 ch: channel_avg_accuracy[ch] / (channel_std_accuracy[ch] + epsilon)
# #                 for ch in candidate_channels
# #             }
# #             best_channel = max(candidate_channels, key=lambda ch: channel_snr[ch])
# #             step_selected_channels.append(best_channel)
# #             best_channel_snr = channel_snr[best_channel]
# #             # Record the SNR along with the validation accuracy.
# #             accuracies_for_best_channel = [acc for ch, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
# #                                            if ch == best_channel]
# #             rep_validation_accuracies.append(np.mean(accuracies_for_best_channel))
# #             # Evaluate test accuracy using channels selected so far.
# #             X_train_subset = X_train_val[:, :, step_selected_channels].reshape(len(X_train_val), -1)
# #             X_test_subset = X_test[:, :, step_selected_channels].reshape(len(X_test), -1)
# #             model.fit(X_train_subset, y_train_val)
# #             y_pred = model.predict(X_test_subset)
# #             rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
# #
# #         # End of greedy loop for this repetition.
# #         all_selected_channels.append(step_selected_channels.copy())
# #         all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
# #         all_test_accuracy_per_step.append(rep_test_accuracies.copy())
# #
# #         # ----------------------------------
# #         # Global aggregation after this repetition:
# #         # For each greedy step, consider only those repetitions where the selected channel appears,
# #         # and discard channels that appear in fewer than min_freq repetitions.
# #         # ----------------------------------
# #         min_freq = 3  # Set minimum frequency threshold.
# #         num_reps = len(all_validation_accuracy_per_step)
# #         min_steps = min(len(rep) for rep in all_validation_accuracy_per_step)
# #         final_selected_channels = []
# #         global_val = []
# #         global_val_std = []
# #         global_test = []
# #         global_test_std = []
# #         global_snr = []
# #         global_freq = []  # store frequency of the aggregated best candidate at each step
# #
# #
# #         for step in range(min_steps):
# #             candidate_info = []
# #             for rep in range(num_reps):
# #                 candidate_info.append((
# #                     all_selected_channels[rep][step],
# #                     all_validation_accuracy_per_step[rep][step],
# #                     all_test_accuracy_per_step[rep][step]
# #                 ))
# #             epsilon = 1e-8
# #             # Calculate frequency for each candidate channel.
# #             freq_dict = {}
# #             for ch, _, _ in candidate_info:
# #                 freq_dict[ch] = freq_dict.get(ch, 0) + 1
# #             # Discard candidates that appear less than min_freq times.
# #             candidate_channels = [ch for ch in freq_dict if freq_dict[ch] >= min_freq]
# #             # If no candidate meets the threshold, fall back to all candidates.
# #             if not candidate_channels:
# #                 candidate_channels = list(freq_dict.keys())
# #             # Compute average and standard deviation of validation accuracy for each candidate.
# #             channel_avg = {ch: np.mean([score for c, score, _ in candidate_info if c == ch])
# #                            for ch in candidate_channels}
# #             channel_std = {ch: np.std([score for c, score, _ in candidate_info if c == ch])
# #                            for ch in candidate_channels}
# #             # Compute SNR.
# #             channel_snr = {ch: channel_avg[ch] / (channel_std[ch] + epsilon)
# #                            for ch in candidate_channels}
# #             best_candidate = max(candidate_channels, key=lambda ch: channel_snr[ch])
# #             final_selected_channels.append(best_candidate)
# #             global_snr.append(channel_snr[best_candidate])
# #             # Record the frequency.
# #             freq = freq_dict.get(best_candidate, 0)
# #             global_freq.append(freq)
# #             # Now, take only those repetitions where the candidate equals the best.
# #             vals = [val for c, val, _ in candidate_info if c == best_candidate]
# #             tests = [test for c, _, test in candidate_info if c == best_candidate]
# #             global_val.append(np.mean(vals))
# #             global_val_std.append(np.std(vals))
# #             global_test.append(np.mean(tests))
# #             global_test_std.append(np.std(tests))
# #
# #         # ----------------------------------
# #         # Update the global plot after this repetition.
# #         # ----------------------------------
# #         ax.clear()
# #         steps = np.arange(1, len(global_val) + 1)
# #         ax.plot(steps, global_val, marker='s', linestyle='-', label='Validation Accuracy')
# #         ax.fill_between(steps, np.array(global_val) - np.array(global_val_std),
# #                         np.array(global_val) + np.array(global_val_std), alpha=0.2)
# #         ax.plot(steps, global_test, marker='o', linestyle='--', label='Test Accuracy')
# #         ax.fill_between(steps, np.array(global_test) - np.array(global_test_std),
# #                         np.array(global_test) + np.array(global_test_std), alpha=0.2)
# #         ax.set_xlabel("Number of Selected Channels")
# #         ax.set_ylabel("Balanced Accuracy")
# #         ax.set_title(f"Global Accuracy Progression (After {num_reps} Repetitions)")
# #         ax.legend()
# #         ax.grid()
# #         # Annotate each step with the aggregated best channel and its average SNR.
# #         for i, ch in enumerate(final_selected_channels):
# #             annotation = f"{ch}" # (SNR={global_snr[i]:.2f})"
# #             ax.annotate(annotation, (i + 1, global_test[i]), textcoords="offset points", xytext=(0, 5), ha='center'
# #                         , fontsize=6)
# #             # Annotate the frequency below in a smaller font.
# #             ax.annotate(f"({global_freq[i]})", (i + 1, global_test[i]),
# #                     textcoords="offset points", xytext=(0, -10), ha='center', fontsize=6)
# #         plt.draw()
# #         plt.pause(1.0)
# #
# #     plt.ioff()
# #     plt.show()
# #
# #     return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step

# THIS VERSION PARALLELISES THE FOLDS, NOT THE CHANNELS
def greedy_nested_cv_channel_selection_snr(
        data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
        max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
        parallel=True, n_jobs=-1, method='concatenation', min_frequency=3):
    """
    Perform nested CV for greedy channel selection using SNR-based candidate selection.
    Uses a single outer split per repetition.

    After each outer repetition is completed, the global average performance (over all repetitions so far)
    is recalculated and the plot is updated showing the performance for all greedy elimination steps.
    In the global aggregation, for each greedy step, candidate channels that appear at least min_frequency times
    are given priority over those that do not (i.e. only candidates meeting this frequency threshold are considered;
    if none meet the threshold, all are considered). Among these, the candidate with the highest average accuracy
    (or SNR, computed as average accuracy divided by standard deviation) is chosen, and the number of outer repetitions
    that selected that channel is annotated.

    Returns:
        - all_selected_channels (list): List of elimination sequences (one per outer repetition)
        - all_test_accuracy_per_step (list): List of test accuracy lists (one per outer repetition)
        - all_validation_accuracy_per_step (list): List of validation accuracy lists (one per outer repetition)
    """
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt

    # Set random seed if not provided.
    if random_seed is None:
        random_seed = np.random.randint(0, int(1e6))
    rng = np.random.default_rng(random_seed)

    # Global accumulators.
    all_selected_channels = []  # One elimination sequence per outer repetition.
    all_test_accuracy_per_step = []  # Test accuracies per elimination step.
    all_validation_accuracy_per_step = []  # Validation accuracies per elimination step.

    # ----------------------------------
    # Custom CV classes.
    # ----------------------------------
    class LeaveOneFromEachClassCV(BaseCrossValidator):
        def __init__(self, shuffle=True, random_state=None):
            self.shuffle = shuffle
            self.random_state = random_state

        def get_n_splits(self, X, y, groups=None):
            _, counts = np.unique(y, return_counts=True)
            return int(np.min(counts))

        def split(self, X, y, groups=None):
            indices_by_class = {}
            for idx, label in enumerate(y):
                indices_by_class.setdefault(label, []).append(idx)
            rng_local = np.random.default_rng(self.random_state)
            for label in indices_by_class:
                if self.shuffle:
                    rng_local.shuffle(indices_by_class[label])
            n_splits = self.get_n_splits(X, y)
            for split in range(n_splits):
                test_indices = []
                for label, indices in indices_by_class.items():
                    test_indices.append(indices[split])
                test_indices = np.array(test_indices)
                train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                yield train_indices, test_indices

    class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
        def __init__(self, n_repeats=50, shuffle=True, random_state=None):
            self.n_repeats = n_repeats
            self.shuffle = shuffle
            self.random_state = random_state

        def get_n_splits(self, X, y, groups=None):
            return self.n_repeats

        def split(self, X, y, groups=None):
            indices_by_class = {}
            for idx, label in enumerate(y):
                indices_by_class.setdefault(label, []).append(idx)
            rng_local = np.random.default_rng(self.random_state)
            for _ in range(self.n_repeats):
                test_indices = []
                for label, indices in indices_by_class.items():
                    if self.shuffle:
                        chosen = rng_local.choice(indices, size=1, replace=False)
                    else:
                        chosen = [indices[0]]
                    test_indices.extend(chosen)
                test_indices = np.array(test_indices)
                train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                yield train_indices, test_indices

    # ----------------------------------
    # Helper functions.
    # ----------------------------------
    def evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val, method=method):
        """
        Per-candidate evaluation (used for the concatenation method).
        """
        candidate_channels = selected_channels + [ch]
        if method == "concatenation":
            X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
            X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
        elif method == "average":
            X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
            X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred_val = model.predict(X_val_subset)
        return ch, balanced_accuracy_score(y_val, y_pred_val)

    def evaluate_candidates_average_batch(cum_sum_train, cum_sum_val, count,
                                          candidate_channels, X_train, y_train, X_val, y_val, alpha):
        """
        Batch-evaluate candidate channels for the "average" method.
        This function computes the averaged features for all candidates at once using broadcasting.
        Candidate evaluation here is done sequentially within the fold.
        """
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
        candidate_train = X_train[:, :, candidate_channels]  # shape: (n_train, n_time, n_candidates)
        candidate_val = X_val[:, :, candidate_channels]
        avg_train = (cum_sum_train[..., None] + candidate_train) / (count + 1)
        avg_val = (cum_sum_val[..., None] + candidate_val) / (count + 1)
        n_candidates = avg_train.shape[2]
        results = []
        for i in range(n_candidates):
            X_train_candidate = avg_train[:, :, i]
            X_val_candidate = avg_val[:, :, i]
            model = RidgeClassifier(alpha=alpha)
            model.fit(X_train_candidate, y_train)
            y_pred_val = model.predict(X_val_candidate)
            score = balanced_accuracy_score(y_val, y_pred_val)
            results.append((candidate_channels[i], score))
        return results

    def evaluate_inner_fold(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                            selected_channels, remaining_channels, alpha, method):
        """
        Evaluate one inner CV fold.
        For the average method, precompute the cumulative sum for the selected channels
        and then evaluate all candidate channels (sequentially within that fold).
        For the concatenation method, evaluate candidates one by one.
        Returns a tuple: (best_candidate, best_accuracy) for this fold.
        """
        X_inner_train = X_train_val[inner_train_idx]
        X_inner_val = X_train_val[inner_val_idx]
        y_inner_train = y_train_val[inner_train_idx]
        y_inner_val = y_train_val[inner_val_idx]

        if method == "concatenation":
            results = [evaluate_channel(ch, selected_channels, X_inner_train, y_inner_train,
                                        X_inner_val, y_inner_val, method=method)
                       for ch in remaining_channels]
        elif method == "average":
            if selected_channels:
                cum_sum_train = np.sum(X_inner_train[:, :, selected_channels], axis=2)
                cum_sum_val = np.sum(X_inner_val[:, :, selected_channels], axis=2)
                count = len(selected_channels)
            else:
                cum_sum_train = np.zeros_like(X_inner_train[:, :, 0])
                cum_sum_val = np.zeros_like(X_inner_val[:, :, 0])
                count = 0
            results = evaluate_candidates_average_batch(cum_sum_train, cum_sum_val, count,
                                                        remaining_channels, X_inner_train, y_inner_train,
                                                        X_inner_val, y_inner_val, alpha)
        else:
            raise ValueError("Unsupported method. Choose 'concatenation' or 'average'.")

        best_candidate, best_accuracy = max(results, key=lambda x: x[1])
        return best_candidate, best_accuracy

    def scale_data(X_train, X_test, scaler_type='standard'):
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
        return X_train, X_test

    model = RidgeClassifier(alpha=alpha)

    # ----------------------------------
    # Set up interactive plotting.
    # ----------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # ----------------------------------
    # Outer repetition loop.
    # ----------------------------------
    for repeat in range(num_outer_repeats):
        print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
        outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                          random_state=rng.integers(0, int(1e6)))
        train_val_idx, test_idx = next(outer_cv.split(data, labels))
        X_train_val, X_test = data[train_val_idx], data[test_idx]
        if normalize:
            X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
                                                   random_state=rng.integers(0, int(1e6)))

        step_selected_channels = []  # Elimination sequence for this outer repetition.
        rep_validation_accuracies = []  # Validation accuracy per elimination step.
        rep_test_accuracies = []  # Test accuracy per elimination step.

        # Greedy elimination loop.
        for step in range(max_channels):
            print(f"Selecting channel {step + 1}/{max_channels}")
            remaining_channels = [ch for ch in range(data.shape[2]) if ch not in step_selected_channels]
            if not remaining_channels:
                break

            # Parallelize across inner CV folds (not candidate search within each fold).
            fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_inner_fold)(
                    inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                    step_selected_channels, remaining_channels, alpha, method
                )
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )

            # Aggregate the results from inner folds.
            best_channel_per_fold = [result[0] for result in fold_results]
            val_accuracies_per_fold = [result[1] for result in fold_results]

            # Compute frequency (across inner folds) for each candidate.
            candidate_channels = list(set(best_channel_per_fold))
            freq_dict = {ch: best_channel_per_fold.count(ch) for ch in candidate_channels}
            # Filter: if any candidate reaches min_frequency, consider only those.
            candidates_meeting_freq = [ch for ch in candidate_channels if freq_dict[ch] >= min_frequency]
            if candidates_meeting_freq:
                candidate_candidates = candidates_meeting_freq
            else:
                candidate_candidates = candidate_channels

            # Compute average and standard deviation of accuracies for the candidates.
            epsilon = 1e-8
            channel_avg_accuracy = {
                ch: np.mean(
                    [acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold) if ch_fold == ch])
                for ch in candidate_candidates
            }
            channel_std_accuracy = {
                ch: np.std(
                    [acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold) if ch_fold == ch])
                for ch in candidate_candidates
            }
            # Compute SNR.
            channel_snr = {
                ch: channel_avg_accuracy[ch] / (channel_std_accuracy[ch] + epsilon)
                for ch in candidate_candidates
            }
            # Select the best candidate: priority is given to frequency (by filtering), then highest SNR
            best_candidate = max(candidate_candidates, key=lambda ch: (channel_snr[ch], channel_avg_accuracy[ch]))
            step_selected_channels.append(best_candidate)
            accuracies_for_best_channel = [acc for ch, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
                                           if ch == best_candidate]
            rep_validation_accuracies.append(np.mean(accuracies_for_best_channel))

            # Evaluate test accuracy using the selected channels so far.
            X_train_subset = X_train_val[:, :, step_selected_channels].reshape(len(X_train_val), -1)
            X_test_subset = X_test[:, :, step_selected_channels].reshape(len(X_test), -1)
            model.fit(X_train_subset, y_train_val)
            y_pred = model.predict(X_test_subset)
            rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))

        # End of greedy loop for this outer repetition.
        all_selected_channels.append(step_selected_channels.copy())
        all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
        all_test_accuracy_per_step.append(rep_test_accuracies.copy())

        # ----------------------------------
        # Global aggregation and plotting (after current outer repetition)
        # ----------------------------------
        num_reps = len(all_test_accuracy_per_step)
        # Use the minimum number of elimination steps across outer repetitions.
        min_steps_global = min(len(rep) for rep in all_test_accuracy_per_step)

        global_val = []
        global_val_std = []
        global_test = []
        global_test_std = []
        for step in range(min_steps_global):
            val_accs = [all_validation_accuracy_per_step[r][step] for r in range(num_reps)]
            test_accs = [all_test_accuracy_per_step[r][step] for r in range(num_reps)]
            global_val.append(np.mean(val_accs))
            global_val_std.append(np.std(val_accs))
            global_test.append(np.mean(test_accs))
            global_test_std.append(np.std(test_accs))

        # Annotation: For each addition step, choose the candidate channel most frequently selected
        # across outer repetitions. Apply the min_frequency rule here as well.
        final_selected_channels = []
        global_freq = []
        for step in range(min_steps_global):
            candidate_info = [(all_selected_channels[r][step], all_test_accuracy_per_step[r][step])
                              for r in range(num_reps)]
            freq_dict = {}
            candidate_acc = {}
            for ch, test_acc in candidate_info:
                freq_dict[ch] = freq_dict.get(ch, 0) + 1
                candidate_acc.setdefault(ch, []).append(test_acc)
            # Filter by min_frequency.
            candidates_meeting = [ch for ch in freq_dict if freq_dict[ch] >= min_frequency]
            if candidates_meeting:
                candidate_candidates = candidates_meeting
            else:
                candidate_candidates = list(freq_dict.keys())
            best_candidate = max(candidate_candidates, key=lambda ch: (freq_dict[ch], np.mean(candidate_acc[ch])))
            final_selected_channels.append(best_candidate)
            global_freq.append(freq_dict[best_candidate])

        # Plot the aggregated average accuracies.
        ax.clear()
        steps_axis = np.arange(1, min_steps_global + 1)
        ax.plot(steps_axis, global_val, marker='s', linestyle='-', label='Validation Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_val) - np.array(global_val_std),
                        np.array(global_val) + np.array(global_val_std), alpha=0.2)
        ax.plot(steps_axis, global_test, marker='o', linestyle='--', label='Test Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_test) - np.array(global_test_std),
                        np.array(global_test) + np.array(global_test_std), alpha=0.2)
        ax.set_xlabel("Number of Selected Channels")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
        ax.legend()
        ax.grid()
        # Annotate each elimination step.
        for i, ch in enumerate(final_selected_channels):
            ax.annotate(f"{ch}", (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10)
            ax.annotate(f"({global_freq[i]})", (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, -10), ha="center", fontsize=6)
        plt.draw()
        plt.pause(1.0)

    plt.ioff()
    plt.show()

    return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step


# #THIS VERSION PARALLELISES HE FOLDS AND IMPLEMENT MANY SPEEDUP TRICKS
# def greedy_nested_cv_channel_selection_snr(
#         data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
#         max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
#         parallel=True, n_jobs=-1, method='concatenation'):
#     """
#     Perform nested CV for greedy channel selection using SNR-based candidate selection.
#     Uses a single outer split per repetition.
#
#     After each outer repetition is completed, the global average performance (over all repetitions so far)
#     is recalculated and the plot is updated showing the performance for all greedy elimination steps.
#     In the global aggregation, for each greedy step, candidate channels that appear in fewer than min_freq
#     repetitions are discarded. Among the remaining, the candidate with the highest average SNR (average
#     validation accuracy divided by standard deviation) is chosen, and only the accuracies from repetitions
#     selecting that channel are aggregated.
#
#     Returns:
#         - all_selected_channels (list): List of removal sequences (one per outer repetition)
#         - all_test_accuracy_per_step (list): List of test accuracy lists (one per outer repetition)
#         - all_validation_accuracy_per_step (list): List of validation accuracy lists (one per outer repetition)
#     """
#     import numpy as np
#     from sklearn.linear_model import RidgeClassifier
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     from joblib import Parallel, delayed
#     import matplotlib.pyplot as plt
#
#     # Set random seed if not provided.
#     if random_seed is None:
#         random_seed = np.random.randint(0, int(1e6))
#     rng = np.random.default_rng(random_seed)
#
#     # Global accumulators across outer repetitions.
#     all_selected_channels = []  # One elimination sequence per outer repetition.
#     all_test_accuracy_per_step = []  # Test accuracies per elimination step.
#     all_validation_accuracy_per_step = []  # Validation accuracies per elimination step.
#
#     # ----------------------------------
#     # Custom CV classes.
#     # ----------------------------------
#     class LeaveOneFromEachClassCV(BaseCrossValidator):
#         def __init__(self, shuffle=True, random_state=None):
#             self.shuffle = shuffle
#             self.random_state = random_state
#
#         def get_n_splits(self, X, y, groups=None):
#             _, counts = np.unique(y, return_counts=True)
#             return int(np.min(counts))
#
#         def split(self, X, y, groups=None):
#             indices_by_class = {}
#             for idx, label in enumerate(y):
#                 indices_by_class.setdefault(label, []).append(idx)
#             rng_local = np.random.default_rng(self.random_state)
#             for label in indices_by_class:
#                 if self.shuffle:
#                     rng_local.shuffle(indices_by_class[label])
#             n_splits = self.get_n_splits(X, y)
#             for split in range(n_splits):
#                 test_indices = []
#                 for label, indices in indices_by_class.items():
#                     test_indices.append(indices[split])
#                 test_indices = np.array(test_indices)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield train_indices, test_indices
#
#     class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
#         def __init__(self, n_repeats=50, shuffle=True, random_state=None):
#             self.n_repeats = n_repeats
#             self.shuffle = shuffle
#             self.random_state = random_state
#
#         def get_n_splits(self, X, y, groups=None):
#             return self.n_repeats
#
#         def split(self, X, y, groups=None):
#             indices_by_class = {}
#             for idx, label in enumerate(y):
#                 indices_by_class.setdefault(label, []).append(idx)
#             rng_local = np.random.default_rng(self.random_state)
#             for _ in range(self.n_repeats):
#                 test_indices = []
#                 for label, indices in indices_by_class.items():
#                     if self.shuffle:
#                         chosen = rng_local.choice(indices, size=1, replace=False)
#                     else:
#                         chosen = [indices[0]]
#                     test_indices.extend(chosen)
#                 test_indices = np.array(test_indices)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield train_indices, test_indices
#
#     # ----------------------------------
#     # Helper functions.
#     # ----------------------------------
#     def evaluate_channel(ch, selected_channels, X_train, y_train, X_val, y_val, method=method):
#         """
#         Legacy per-candidate evaluation (used for the concatenation method).
#         """
#         candidate_channels = selected_channels + [ch]
#         if method == "concatenation":
#             X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
#             X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
#         elif method == "average":
#             X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
#             X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(X_train_subset, y_train)
#         y_pred_val = model.predict(X_val_subset)
#         return ch, balanced_accuracy_score(y_val, y_pred_val)
#
#     def evaluate_candidates_average_batch(cum_sum_train, cum_sum_val, count,
#                                           candidate_channels, X_train, y_train, X_val, y_val, alpha,
#                                           parallel_flag=False, n_jobs_flag=1):
#         """
#         Batch-evaluate candidate channels for the "average" method.
#         This function computes the averaged features for all candidates at once using broadcasting.
#         The internal candidate evaluations can be run sequentially (parallel_flag=False) since
#         we now parallelize the inner CV folds.
#         """
#         n_train = X_train.shape[0]
#         n_val = X_val.shape[0]
#         # Extract candidate channel data (shape: [n_train, n_time, n_candidates])
#         candidate_train = X_train[:, :, candidate_channels]
#         candidate_val = X_val[:, :, candidate_channels]
#         # Compute averaged features with broadcasting.
#         avg_train = (cum_sum_train[..., None] + candidate_train) / (count + 1)
#         avg_val = (cum_sum_val[..., None] + candidate_val) / (count + 1)
#         n_candidates = avg_train.shape[2]
#
#         # Define a simple inner loop over candidate indices.
#         results = []
#         for i in range(n_candidates):
#             X_train_candidate = avg_train[:, :, i]
#             X_val_candidate = avg_val[:, :, i]
#             model = RidgeClassifier(alpha=alpha)
#             model.fit(X_train_candidate, y_train)
#             y_pred_val = model.predict(X_val_candidate)
#             score = balanced_accuracy_score(y_val, y_pred_val)
#             results.append((candidate_channels[i], score))
#         return results
#
#     def evaluate_inner_fold(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                             selected_channels, remaining_channels, alpha, method):
#         """
#         Evaluate one inner CV fold:
#          - Slice the training and validation sets for this fold.
#          - For the average method: precompute the cumulative sum for the selected channels,
#            then evaluate all candidate channels in batch (without further parallelism).
#          - For the concatenation method: use the legacy per-candidate evaluation.
#          - Return the best candidate (and its score) for this fold.
#         """
#         X_inner_train = X_train_val[inner_train_idx]
#         X_inner_val = X_train_val[inner_val_idx]
#         y_inner_train = y_train_val[inner_train_idx]
#         y_inner_val = y_train_val[inner_val_idx]
#
#         if method == "concatenation":
#             # Evaluate candidates one by one.
#             results = [evaluate_channel(ch, selected_channels, X_inner_train, y_inner_train, X_inner_val, y_inner_val,
#                                         method=method)
#                        for ch in remaining_channels]
#         elif method == "average":
#             if selected_channels:
#                 cum_sum_train = np.sum(X_inner_train[:, :, selected_channels], axis=2)
#                 cum_sum_val = np.sum(X_inner_val[:, :, selected_channels], axis=2)
#                 count = len(selected_channels)
#             else:
#                 cum_sum_train = np.zeros_like(X_inner_train[:, :, 0])
#                 cum_sum_val = np.zeros_like(X_inner_val[:, :, 0])
#                 count = 0
#             # Evaluate all candidates in batch.
#             results = evaluate_candidates_average_batch(cum_sum_train, cum_sum_val, count,
#                                                         remaining_channels, X_inner_train, y_inner_train,
#                                                         X_inner_val, y_inner_val, alpha,
#                                                         parallel_flag=False, n_jobs_flag=10)
#         else:
#             raise ValueError("Unsupported method. Choose 'concatenation' or 'average'.")
#
#         # Choose the best candidate for this fold.
#         best_candidate, best_accuracy = max(results, key=lambda x: x[1])
#         return best_candidate, best_accuracy
#
#     def scale_data(X_train, X_test, scaler_type='standard'):
#         if scaler_type == 'standard':
#             scaler = StandardScaler()
#         elif scaler_type == 'minmax':
#             scaler = MinMaxScaler()
#         else:
#             raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
#         X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
#         X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
#         return X_train, X_test
#
#     model = RidgeClassifier(alpha=alpha)
#
#     # ----------------------------------
#     # Set up interactive plotting.
#     # ----------------------------------
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # ----------------------------------
#     # Outer repetition loop.
#     # ----------------------------------
#     for repeat in range(num_outer_repeats):
#         print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
#         outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
#                                           random_state=rng.integers(0, int(1e6)))
#         train_val_idx, test_idx = next(outer_cv.split(data, labels))
#         X_train_val, X_test = data[train_val_idx], data[test_idx]
#         if normalize:
#             X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
#         y_train_val, y_test = labels[train_val_idx], labels[test_idx]
#
#         inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
#                                                    random_state=rng.integers(0, int(1e6)))
#
#         step_selected_channels = []  # Greedy channel sequence for this repetition.
#         rep_validation_accuracies = []  # Validation accuracy per elimination step.
#         rep_test_accuracies = []  # Test accuracy per elimination step.
#
#         # Greedy elimination loop.
#         for step in range(max_channels):
#             print(f"Selecting channel {step + 1}/{max_channels}")
#             remaining_channels = [ch for ch in range(data.shape[2]) if ch not in step_selected_channels]
#             if not remaining_channels:
#                 break
#
#             # Parallelize across inner CV folds.
#             fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_inner_fold)(
#                     inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                     step_selected_channels, remaining_channels, alpha, method
#                 )
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#
#             # Collect best candidate and scores from each inner fold.
#             best_channel_per_fold = [result[0] for result in fold_results]
#             val_accuracies_per_fold = [result[1] for result in fold_results]
#
#             # Compute SNR for candidate channels across inner folds.
#             epsilon = 1e-8
#             candidate_channels = list(set(best_channel_per_fold))
#             channel_avg_accuracy = {
#                 ch: np.mean([acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
#                              if ch_fold == ch])
#                 for ch in candidate_channels
#             }
#             channel_std_accuracy = {
#                 ch: np.std([acc for ch_fold, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
#                             if ch_fold == ch])
#                 for ch in candidate_channels
#             }
#             channel_snr = {
#                 ch: channel_avg_accuracy[ch] / (channel_std_accuracy[ch] + epsilon)
#                 for ch in candidate_channels
#             }
#             best_candidate = max(candidate_channels, key=lambda ch: channel_snr[ch])
#             step_selected_channels.append(best_candidate)
#             accuracies_for_best_channel = [acc for ch, acc in zip(best_channel_per_fold, val_accuracies_per_fold)
#                                            if ch == best_candidate]
#             rep_validation_accuracies.append(np.mean(accuracies_for_best_channel))
#
#             # Evaluate test accuracy using the channels selected so far.
#             X_train_subset = X_train_val[:, :, step_selected_channels].reshape(len(X_train_val), -1)
#             X_test_subset = X_test[:, :, step_selected_channels].reshape(len(X_test), -1)
#             model.fit(X_train_subset, y_train_val)
#             y_pred = model.predict(X_test_subset)
#             rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#
#         # End of greedy loop for this repetition.
#         all_selected_channels.append(step_selected_channels.copy())
#         all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
#         all_test_accuracy_per_step.append(rep_test_accuracies.copy())
#
#         # ----------------------------------
#         # Global aggregation and plotting.
#         # ----------------------------------
#         # min_freq = 3
#         # num_reps = len(all_validation_accuracy_per_step)
#         # min_steps = min(len(rep) for rep in all_validation_accuracy_per_step)
#         # final_selected_channels = []
#         # global_val = []
#         # global_val_std = []
#         # global_test = []
#         # global_test_std = []
#         # global_snr = []
#         # global_freq = []
#         #
#         # for step in range(min_steps):
#         #     candidate_info = []
#         #     for rep in range(num_reps):
#         #         candidate_info.append((
#         #             all_selected_channels[rep][step],
#         #             all_validation_accuracy_per_step[rep][step],
#         #             all_test_accuracy_per_step[rep][step]
#         #         ))
#         #     epsilon = 1e-8
#         #     freq_dict = {}
#         #     for ch, _, _ in candidate_info:
#         #         freq_dict[ch] = freq_dict.get(ch, 0) + 1
#         #     candidate_channels = [ch for ch in freq_dict if freq_dict[ch] >= min_freq]
#         #     if not candidate_channels:
#         #         candidate_channels = list(freq_dict.keys())
#         #     channel_avg = {ch: np.mean([score for c, score, _ in candidate_info if c == ch])
#         #                    for ch in candidate_channels}
#         #     channel_std = {ch: np.std([score for c, score, _ in candidate_info if c == ch])
#         #                    for ch in candidate_channels}
#         #     channel_snr = {ch: channel_avg[ch] / (channel_std[ch] + epsilon)
#         #                    for ch in candidate_channels}
#         #     best_candidate = max(candidate_channels, key=lambda ch: channel_snr[ch])
#         #     final_selected_channels.append(best_candidate)
#         #     global_snr.append(channel_snr[best_candidate])
#         #     freq = freq_dict.get(best_candidate, 0)
#         #     global_freq.append(freq)
#         #     vals = [val for c, val, _ in candidate_info if c == best_candidate]
#         #     tests = [test for c, _, test in candidate_info if c == best_candidate]
#         #     global_val.append(np.mean(vals))
#         #     global_val_std.append(np.std(vals))
#         #     global_test.append(np.mean(tests))
#         #     global_test_std.append(np.std(tests))
#         #
#         # ax.clear()
#         # steps = np.arange(1, len(global_val) + 1)
#         # ax.plot(steps, global_val, marker='s', linestyle='-', label='Validation Accuracy')
#         # ax.fill_between(steps, np.array(global_val) - np.array(global_val_std),
#         #                 np.array(global_val) + np.array(global_val_std), alpha=0.2)
#         # ax.plot(steps, global_test, marker='o', linestyle='--', label='Test Accuracy')
#         # ax.fill_between(steps, np.array(global_test) - np.array(global_test_std),
#         #                 np.array(global_test) + np.array(global_test_std), alpha=0.2)
#         # ax.set_xlabel("Number of Selected Channels")
#         # ax.set_ylabel("Balanced Accuracy")
#         # ax.set_title(f"Global Accuracy Progression (After {num_reps} Repetitions)")
#         # ax.legend()
#         # ax.grid()
#         # for i, ch in enumerate(final_selected_channels):
#         #     annotation = f"{ch}"
#         #     ax.annotate(annotation, (i + 1, global_test[i]), textcoords="offset points",
#         #                 xytext=(0, 5), ha='center', fontsize=10)
#         #     ax.annotate(f"({global_freq[i]})", (i + 1, global_test[i]),
#         #                 textcoords="offset points", xytext=(0, -10), ha='center', fontsize=6)
#         # plt.draw()
#         # plt.pause(1.0)
#
#         # ----------------------------------
#         # Global aggregation and plotting (Average across outer repetitions)
#         # ----------------------------------
#         num_reps = len(all_test_accuracy_per_step)
#         # Use the minimum number of steps present in all repetitions.
#         min_steps = min(len(rep) for rep in all_test_accuracy_per_step)
#
#         # Aggregate the test and validation accuracies across outer repetitions.
#         global_val = []
#         global_val_std = []
#         global_test = []
#         global_test_std = []
#         for step in range(min_steps):
#             # Extract the accuracies for this elimination step across all outer repetitions.
#             val_accs = [all_validation_accuracy_per_step[rep][step] for rep in range(num_reps)]
#             test_accs = [all_test_accuracy_per_step[rep][step] for rep in range(num_reps)]
#             global_val.append(np.mean(val_accs))
#             global_val_std.append(np.std(val_accs))
#             global_test.append(np.mean(test_accs))
#             global_test_std.append(np.std(test_accs))
#
#         # For annotation: For each elimination step, determine the candidate channel
#         # that was selected most frequently across outer repetitions.
#         # In case of a tie, choose the one with the highest average test accuracy.
#         final_selected_channels = []
#         global_freq = []
#         for step in range(min_steps):
#             # Get candidate info from each repetition: (candidate, test_accuracy)
#             candidate_info = [
#                 (all_selected_channels[rep][step], all_test_accuracy_per_step[rep][step])
#                 for rep in range(num_reps)
#             ]
#             # Count frequencies and accumulate test accuracies.
#             freq_dict = {}
#             candidate_acc = {}
#             for ch, test_acc in candidate_info:
#                 freq_dict[ch] = freq_dict.get(ch, 0) + 1
#                 if ch in candidate_acc:
#                     candidate_acc[ch].append(test_acc)
#                 else:
#                     candidate_acc[ch] = [test_acc]
#             # Get the list of candidate channels.
#             unique_candidates = list(freq_dict.keys())
#             # Choose the candidate maximizing (frequency, average test accuracy).
#             best_candidate = max(unique_candidates,
#                                  key=lambda ch: (freq_dict[ch], np.mean(candidate_acc[ch])))
#             final_selected_channels.append(best_candidate)
#             global_freq.append(freq_dict[best_candidate])
#
#         # Plot the aggregated average accuracies.
#         ax.clear()
#         steps = np.arange(1, min_steps + 1)
#         ax.plot(steps, global_val, marker='s', linestyle='-', label='Validation Accuracy')
#         ax.fill_between(steps,
#                         np.array(global_val) - np.array(global_val_std),
#                         np.array(global_val) + np.array(global_val_std), alpha=0.2)
#         ax.plot(steps, global_test, marker='o', linestyle='--', label='Test Accuracy')
#         ax.fill_between(steps,
#                         np.array(global_test) - np.array(global_test_std),
#                         np.array(global_test) + np.array(global_test_std), alpha=0.2)
#         ax.set_xlabel("Number of Selected Channels")
#         ax.set_ylabel("Balanced Accuracy")
#         # Updated title: now includes the number of outer repetitions completed.
#         ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
#         ax.legend()
#         ax.grid()
#
#         # Annotate each step on the test accuracy graph:
#         # The annotation shows the candidate channel and, below, the number of outer repetitions that selected it.
#         for i, ch in enumerate(final_selected_channels):
#             annotation = f"{ch}"
#             ax.annotate(annotation,
#                         (steps[i], global_test[i]),
#                         textcoords="offset points",
#                         xytext=(0, 5),
#                         ha="center",
#                         fontsize=10)
#             ax.annotate(f"({global_freq[i]})",
#                         (steps[i], global_test[i]),
#                         textcoords="offset points",
#                         xytext=(0, -10),
#                         ha="center",
#                         fontsize=6)
#         plt.draw()
#         plt.pause(1.0)
#
#     plt.ioff()
#     plt.show()
#
#     return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step


# THIS VERSION IS ALL IN ONE (ADDING OR REMOVING CHANNELS)


# THIS VERSION PARALLELISES THE CHANNELS
# def greedy_nested_cv_channel_elimination(
#         data, labels, alpha=1.0, num_outer_splits=5, num_outer_repeats=3, inner_cv_folds=3,
#         max_channels=40, normalize=True, scaler_type='standard', random_seed=None, parallel=True, n_jobs=-1):
#     """
#     Perform nested cross-validation (CV) for greedy channel elimination from the mean TIC or TIS.
#     Tracks validation and test accuracy for each added channel.
#
#     Returns:
#         - selected_channels (list): Ordered list of selected channels
#         - avg_validation_accuracies (array): Mean validation accuracy per step
#         - avg_test_accuracies (array): Mean test accuracy per step
#     """
#     import numpy as np
#     from collections import Counter
#     from sklearn.linear_model import RidgeClassifier
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.model_selection import StratifiedKFold, BaseCrossValidator, StratifiedShuffleSplit
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     from joblib import Parallel, delayed
#     import matplotlib.pyplot as plt
#     import matplotlib
#     # matplotlib.use("TkAgg")
#
#     if random_seed is None:
#         random_seed = np.random.randint(0, int(1e6))
#     rng = np.random.default_rng(random_seed)
#
#     all_selected_channels = []
#     all_test_accuracy_per_step = []
#     all_validation_accuracy_per_step = []
#
#     class LeaveOneFromEachClassCV(BaseCrossValidator):
#         """
#         Custom cross-validator that, in each fold, leaves one sample per class as the test set.
#         The number of folds is determined by the minimum number of samples among all classes.
#         """
#         def __init__(self, shuffle=True, random_state=None):
#             self.shuffle = shuffle
#             self.random_state = random_state
#
#         def get_n_splits(self, X, y, groups=None):
#             _, counts = np.unique(y, return_counts=True)
#             return int(np.min(counts))
#
#         def split(self, X, y, groups=None):
#             indices_by_class = {}
#             for idx, label in enumerate(y):
#                 indices_by_class.setdefault(label, []).append(idx)
#             rng_local = np.random.default_rng(self.random_state)
#             for label in indices_by_class:
#                 if self.shuffle:
#                     rng_local.shuffle(indices_by_class[label])
#             n_splits = self.get_n_splits(X, y)
#             for split in range(n_splits):
#                 test_indices = []
#                 for label, indices in indices_by_class.items():
#                     test_indices.append(indices[split])
#                 test_indices = np.array(test_indices)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield train_indices, test_indices
#
#     class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
#         """
#         Custom cross-validator that randomly selects one sample per class as the test set,
#         and repeats the process a specified number of times.
#         """
#
#         def __init__(self, n_repeats=50, shuffle=True, random_state=None):
#             self.n_repeats = n_repeats
#             self.shuffle = shuffle
#             self.random_state = random_state
#
#         def get_n_splits(self, X, y, groups=None):
#             return self.n_repeats
#
#         def split(self, X, y, groups=None):
#             indices_by_class = {}
#             for idx, label in enumerate(y):
#                 indices_by_class.setdefault(label, []).append(idx)
#
#             rng = np.random.default_rng(self.random_state)
#             for _ in range(self.n_repeats):
#                 test_indices = []
#                 for label, indices in indices_by_class.items():
#                     if self.shuffle:
#                         chosen = rng.choice(indices, size=1, replace=False)
#                     else:
#                         chosen = [indices[0]]
#                     test_indices.extend(chosen)
#                 test_indices = np.array(test_indices)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield train_indices, test_indices
#
#     def evaluate_channel(selected_channels, X_train, y_train, X_val, y_val, method='concatenation'):
#         candidate_channels = selected_channels
#         if method == "concatenation":
#             X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
#             X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
#         elif method == "average":
#             #  Average into a single TIC per sample
#             X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
#             X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)
#
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(X_train_subset, y_train)
#         y_pred_val = model.predict(X_val_subset)
#         return balanced_accuracy_score(y_val, y_pred_val)
#
#     # def candidate_score(candidate, current_channels, X_train_val, y_train_val, inner_cv):
#     #     """
#     #     Evaluate the performance (inner CV score) of the candidate set obtained by removing
#     #     the specified candidate channel from current_channels.
#     #
#     #     Parameters
#     #     ----------
#     #     candidate : int
#     #         The candidate channel to remove.
#     #     current_channels : list
#     #         The current list of channel indices.
#     #     X_train_val : array-like
#     #         The training data used for inner CV (expected to be 3D: samples x ... x channels).
#     #     y_train_val : array-like
#     #         The corresponding labels for X_train_val.
#     #     inner_cv : BaseCrossValidator
#     #         The cross-validation splitter for inner CV.
#     #
#     #     Returns
#     #     -------
#     #     tuple
#     #         A tuple (candidate, average_score) where average_score is the mean balanced accuracy
#     #         obtained over the inner folds when candidate is removed.
#     #     """
#     #     # Form candidate set by removing the candidate channel.
#     #     candidate_set = [ch for ch in current_channels if ch != candidate]
#     #     fold_scores = []
#     #
#     #     # # Ensure the training data is a NumPy array.
#     #     # X_train_val = np.array(X_train_val)
#     #     # y_train_val = np.array(y_train_val)
#     #
#     #     # Evaluate candidate_set on each inner fold.
#     #     for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val):
#     #         X_inner_train = X_train_val[inner_train_idx]
#     #         X_inner_val = X_train_val[inner_val_idx]
#     #         score = evaluate_channel(candidate_set, X_inner_train, y_train_val[inner_train_idx],
#     #                                  X_inner_val, y_train_val[inner_val_idx], method="average")
#     #         fold_scores.append(score)
#     #
#     #     candidate_avg_score = np.mean(fold_scores)
#     #     return candidate, candidate_avg_score
#
#     def candidate_score(candidate, current_channels, X_train_val, y_train_val, inner_cv, n_jobs_inner=1):
#         candidate_set = [ch for ch in current_channels if ch != candidate]
#
#         def evaluate_fold(inner_train_idx, inner_val_idx):
#             X_inner_train = X_train_val[inner_train_idx]
#             X_inner_val = X_train_val[inner_val_idx]
#             return evaluate_channel(candidate_set, X_inner_train, y_train_val[inner_train_idx],
#                                     X_inner_val, y_train_val[inner_val_idx], method="average")
#
#         # Parallelize over folds
#         fold_scores = Parallel(n_jobs=n_jobs_inner, backend='loky')(
#             delayed(evaluate_fold)(inner_train_idx, inner_val_idx)
#             for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#         )
#         candidate_avg_score = np.mean(fold_scores)
#         return candidate, candidate_avg_score
#
#     def scale_data(X_train, X_test, scaler_type='standard'):
#         if scaler_type == 'standard':
#             scaler = StandardScaler()
#         elif scaler_type == 'minmax':
#             scaler = MinMaxScaler()
#         else:
#             raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
#         X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
#         X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
#         return X_train, X_test
#
#     model = RidgeClassifier(alpha=alpha)
#     # model = LogisticRegression(solver='liblinear', random_state=42)
#     # model = SGDClassifier()
#
#     # Loop over outer repetitions
#     for repeat in range(num_outer_repeats):
#         print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
#         # outer_cv = StratifiedKFold(n_splits=num_outer_splits, shuffle=True, random_state=rng.integers(0, int(1e6)))
#         outer_cv = StratifiedShuffleSplit(n_splits=num_outer_splits, test_size=0.2,
#                                           random_state=rng.integers(0, int(1e6))
#                                           )
#
#         # Temporary lists to hold metrics for this repetition
#         rep_selected_channels = []
#         rep_validation_accuracy_per_step = []
#         rep_test_accuracy_per_step = []
#
#         for split, (train_val_idx, test_idx) in enumerate(outer_cv.split(data, labels)):
#             print(f"▶️ Split {split + 1}/{num_outer_splits}")
#             X_train_val, X_test = data[train_val_idx], data[test_idx]
#             if normalize:
#                 X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
#             y_train_val, y_test = labels[train_val_idx], labels[test_idx]
#
#             # Use the custom inner CV (for example, LeaveOneFromEachClassCV)
#             # inner_cv = LeaveOneFromEachClassCV(shuffle=True, random_state=rng.integers(0, int(1e6)))
#             inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True, random_state=rng.integers(0, int(1e6)))
#
#             # Start with ALL channels
#             current_channels = list(range(data.shape[2]))
#             split_validation_accuracies = []
#             split_test_accuracies = []
#             removal_sequence = []  # Record the channel removed at each step
#
#             for step in range(max_channels):
#                 # Stop if only one channel remains
#                 if len(current_channels) <= 1:
#                     break
#
#                 if (step + 1) % 1 == 0 or (step + 1) == max_channels:
#                     print(f"\u23E9 m/z channel removal step: {step + 1}/{max_channels}")
#
#                 candidate_scores = []
#                 # For each channel in the current set, evaluate performance if that channel is removed.
#                 if parallel:
#                     candidate_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                         delayed(candidate_score)(candidate, current_channels, X_train_val, y_train_val, inner_cv)
#                         for candidate in current_channels
#                     )
#
#                 else:
#                     candidate_scores = [candidate_score(candidate, current_channels, X_train_val, y_train_val, inner_cv,
#                                                         n_jobs_inner=20)
#                                         for candidate in current_channels]
#
#                 # Select the candidate whose removal yields the highest average score.
#                 best_candidate, best_candidate_score = max(candidate_scores, key=lambda x: x[1])
#                 removal_sequence.append(best_candidate)
#                 current_channels.remove(best_candidate)
#                 split_validation_accuracies.append(best_candidate_score)
#
#                 # Evaluate outer test accuracy using the remaining channels.
#                 X_train_subset = np.mean(X_train_val[:, :, current_channels], axis=2)
#                 X_test_subset = np.mean(X_test[:, :, current_channels], axis=2)
#                 model.fit(X_train_subset, y_train_val)
#                 y_pred = model.predict(X_test_subset)
#                 split_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#
#             rep_selected_channels.append(current_channels)
#             rep_validation_accuracy_per_step.append(split_validation_accuracies)
#             rep_test_accuracy_per_step.append(split_test_accuracies)
#
#         # Compute averages for this repetition over all splits
#         rep_avg_validation_accuracies = np.mean(rep_validation_accuracy_per_step, axis=0)
#         rep_std_validation_accuracies = np.std(rep_validation_accuracy_per_step, axis=0)
#         rep_avg_test_accuracies = np.mean(rep_test_accuracy_per_step, axis=0)
#         rep_std_test_accuracies = np.std(rep_test_accuracy_per_step, axis=0)
#
#         # Append repetition-level results to the global lists
#         all_selected_channels.extend(rep_selected_channels)
#         all_validation_accuracy_per_step.extend(rep_validation_accuracy_per_step)
#         all_test_accuracy_per_step.extend(rep_test_accuracy_per_step)
#
#         # Plot for this repetition (non-blocking or save to file)
#         fig, ax = plt.subplots(figsize=(10, 6))
#         ax.plot(range(1, len(rep_avg_validation_accuracies) + 1), rep_avg_validation_accuracies,
#                 marker='s', linestyle='-', label='Validation Accuracy')
#         ax.fill_between(range(1, len(rep_avg_validation_accuracies) + 1),
#                         rep_avg_validation_accuracies - rep_std_validation_accuracies,
#                         rep_avg_validation_accuracies + rep_std_validation_accuracies,
#                         alpha=0.2)
#         ax.plot(range(1, len(rep_avg_test_accuracies) + 1), rep_avg_test_accuracies,
#                 marker='o', linestyle='--', label='Test Accuracy')
#         ax.fill_between(range(1, len(rep_avg_test_accuracies) + 1),
#                         rep_avg_test_accuracies - rep_std_test_accuracies,
#                         rep_avg_test_accuracies + rep_std_test_accuracies,
#                         alpha=0.2)
#
#         # Annotate each test accuracy point with its corresponding removed channel (from the last outer split)
#         for i, txt in enumerate(removal_sequence):
#             ax.annotate(str(txt), (i + 1, rep_avg_test_accuracies[i]), textcoords="offset points", xytext=(0, 5), ha='center')
#
#         ax.set_xlabel("Number of Removed Channels")
#         ax.set_ylabel("Balanced Accuracy")
#         ax.set_title(f"Validation and Test Accuracy Progression (Repetition {repeat + 1})")
#         ax.legend()
#         ax.grid()
#         # Option 1: Non-blocking display
#         plt.show(block=False)
#         plt.pause(0.5)
#         # Option 2: Save to file and close the figure
#         # plt.savefig(f"rep_{repeat + 1}_plot.png")
#         # plt.close(fig)
#
#     # After all repetitions, compute global averages if needed
#     avg_test_accuracies = np.mean(all_test_accuracy_per_step, axis=0)
#     avg_validation_accuracies = np.mean(all_validation_accuracy_per_step, axis=0)
#
#     # Global plot if desired
#     fig, ax = plt.subplots(figsize=(10, 6))
#     ax.plot(range(1, len(avg_validation_accuracies) + 1), avg_validation_accuracies, marker='s', linestyle='-',
#             label='Validation Accuracy')
#     ax.fill_between(range(1, len(avg_validation_accuracies) + 1),
#                     np.array(avg_validation_accuracies) - np.array(np.std(all_validation_accuracy_per_step, axis=0)),
#                     np.array(avg_validation_accuracies) + np.array(np.std(all_validation_accuracy_per_step, axis=0)),
#                     alpha=0.2)
#     ax.plot(range(1, len(avg_test_accuracies) + 1), avg_test_accuracies, marker='o', linestyle='--', label='Test Accuracy')
#     ax.fill_between(range(1, len(avg_test_accuracies) + 1),
#                     np.array(avg_test_accuracies) - np.array(np.std(all_test_accuracy_per_step, axis=0)),
#                     np.array(avg_test_accuracies) + np.array(np.std(all_test_accuracy_per_step, axis=0)),
#                     alpha=0.2)
#
#     ax.set_xlabel("Number of Removed Channels")
#     ax.set_ylabel("Balanced Accuracy")
#     ax.set_title("Global Validation and Test Accuracy Progression")
#     ax.legend()
#     ax.grid()
#
#     # Final selection per step:
#     # For each removal step, select the channel that appears most frequently as the best removed channel.
#     # If tied, use the one with the highest average validation accuracy.
#     final_selected_channels = []
#     max_steps = max(len(seq) for seq in all_selected_channels)
#     for step in range(max_steps):
#         candidate_info = []  # Will hold tuples (channel, candidate_score) for this step.
#         for rep in range(len(all_selected_channels)):
#             if step < len(all_selected_channels[rep]) and step < len(all_validation_accuracy_per_step[rep]):
#                 ch = all_selected_channels[rep][step]
#                 score = all_validation_accuracy_per_step[rep][step]
#                 candidate_info.append((ch, score))
#         if candidate_info:  # Only proceed if we got data for this step.
#             freq = {}
#             avg_score = {}
#             for ch, score in candidate_info:
#                 freq[ch] = freq.get(ch, 0) + 1
#                 avg_score[ch] = avg_score.get(ch, 0) + score
#             for ch in avg_score:
#                 avg_score[ch] /= freq[ch]
#             best_candidate = None
#             best_freq = -1
#             best_avg = -np.inf
#             for ch in freq:
#                 if freq[ch] > best_freq or (freq[ch] == best_freq and avg_score[ch] > best_avg):
#                     best_candidate = ch
#                     best_freq = freq[ch]
#                     best_avg = avg_score[ch]
#             final_selected_channels.append(best_candidate)
#
#     # Annotate the global plot with final_selected_channels:
#     for i, channel in enumerate(final_selected_channels):
#         ax.annotate(str(channel), (i + 1, avg_test_accuracies[i]),
#                     textcoords="offset points", xytext=(0, 5), ha="center")
#
#     plt.show()
#
#     return final_selected_channels, avg_test_accuracies, avg_validation_accuracies


# def greedy_nested_cv_channel_elimination(
#         data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
#         max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
#         parallel=True, n_jobs=-1, method='concatenation', min_frequency=3):
#     """
#     Perform nested CV for greedy channel elimination using SNR-based candidate evaluation.
#     (Backward elimination: start with all channels and, at each step, remove the channel whose removal
#     yields the best (or least harmed) validation accuracy.)
#
#     After each outer repetition (using a single outer split), the global average performance (over all repetitions so far)
#     is recalculated and the plot is updated showing the progression of performance across elimination steps.
#     In the global aggregation, for each step candidate channels that appear in fewer than min_frequency inner folds
#     are discarded; among those remaining, the candidate with the highest average SNR (average validation accuracy divided
#     by standard deviation) is chosen, and the number of outer repetitions that selected that channel is annotated.
#
#     Returns:
#       - all_selected_channels (list): List (per outer repetition) of removal sequences (i.e. channels removed in order).
#       - all_test_accuracy_per_step (list): List (per outer repetition) of test accuracy lists (one per elimination step).
#       - all_validation_accuracy_per_step (list): List (per outer repetition) of validation accuracy lists (one per elimination step).
#     """
#     import numpy as np
#     from sklearn.linear_model import RidgeClassifier
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     from joblib import Parallel, delayed
#     import matplotlib.pyplot as plt
#
#     # Set random seed if not provided.
#     if random_seed is None:
#         random_seed = np.random.randint(0, int(1e6))
#     rng = np.random.default_rng(random_seed)
#
#     # Global accumulators.
#     all_selected_channels = []  # One elimination sequence per outer repetition.
#     all_test_accuracy_per_step = []  # Outer test accuracies per elimination step.
#     all_validation_accuracy_per_step = []  # Inner CV (validation) accuracies per elimination step.
#
#     # ----------------------------------
#     # Custom CV classes.
#     # (These are identical to your previous version.)
#     # ----------------------------------
#     class LeaveOneFromEachClassCV(BaseCrossValidator):
#         def __init__(self, shuffle=True, random_state=None):
#             self.shuffle = shuffle
#             self.random_state = random_state
#
#         def get_n_splits(self, X, y, groups=None):
#             _, counts = np.unique(y, return_counts=True)
#             return int(np.min(counts))
#
#         def split(self, X, y, groups=None):
#             indices_by_class = {}
#             for idx, label in enumerate(y):
#                 indices_by_class.setdefault(label, []).append(idx)
#             rng_local = np.random.default_rng(self.random_state)
#             for label in indices_by_class:
#                 if self.shuffle:
#                     rng_local.shuffle(indices_by_class[label])
#             n_splits = self.get_n_splits(X, y)
#             for split in range(n_splits):
#                 test_indices = []
#                 for label, indices in indices_by_class.items():
#                     test_indices.append(indices[split])
#                 test_indices = np.array(test_indices)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield train_indices, test_indices
#
#     class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
#         def __init__(self, n_repeats=50, shuffle=True, random_state=None):
#             self.n_repeats = n_repeats
#             self.shuffle = shuffle
#             self.random_state = random_state
#
#         def get_n_splits(self, X, y, groups=None):
#             return self.n_repeats
#
#         def split(self, X, y, groups=None):
#             indices_by_class = {}
#             for idx, label in enumerate(y):
#                 indices_by_class.setdefault(label, []).append(idx)
#             rng_local = np.random.default_rng(self.random_state)
#             for _ in range(self.n_repeats):
#                 test_indices = []
#                 for label, indices in indices_by_class.items():
#                     if self.shuffle:
#                         chosen = rng_local.choice(indices, size=1, replace=False)
#                     else:
#                         chosen = [indices[0]]
#                     test_indices.extend(chosen)
#                 test_indices = np.array(test_indices)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield train_indices, test_indices
#
#     # ----------------------------------
#     # Helper functions.
#     # ----------------------------------
#
#     def evaluate_channel_elimination(ch, selected_channels, X_train, y_train, X_val, y_val, method=method):
#         """
#         Evaluate performance when channel 'ch' is removed from the current selection.
#         For "concatenation", this is implemented as:
#             candidate_set = [x for x in selected_channels if x != ch]
#         For "average", it simply computes the mean over candidate_set.
#         """
#         candidate_channels = [x for x in selected_channels if x != ch]
#         if method == "concatenation":
#             X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
#             X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
#         elif method == "average":
#             X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
#             X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(X_train_subset, y_train)
#         y_pred = model.predict(X_val_subset)
#         return ch, balanced_accuracy_score(y_val, y_pred)
#
#     def evaluate_candidates_average_batch_elimination(cum_sum_train, cum_sum_val, count,
#                                                       candidate_channels, X_train, y_train, X_val, y_val, alpha):
#         """
#         Batch-evaluate candidate channels for the "average" method in elimination mode.
#         In elimination, the new average (if channel c is removed) is computed as:
#              (cum_sum - X[:,:,c]) / (count - 1)
#         """
#         n_train = X_train.shape[0]
#         n_val = X_val.shape[0]
#         candidate_train = X_train[:, :, candidate_channels]
#         candidate_val = X_val[:, :, candidate_channels]
#         avg_train = (cum_sum_train[..., None] - candidate_train) / (count - 1)
#         avg_val = (cum_sum_val[..., None] - candidate_val) / (count - 1)
#         n_candidates = avg_train.shape[2]
#         results = []
#         for i in range(n_candidates):
#             X_train_candidate = avg_train[:, :, i]
#             X_val_candidate = avg_val[:, :, i]
#             model = RidgeClassifier(alpha=alpha)
#             model.fit(X_train_candidate, y_train)
#             y_pred = model.predict(X_val_candidate)
#             score = balanced_accuracy_score(y_val, y_pred)
#             results.append((candidate_channels[i], score))
#         return results
#
#     def evaluate_inner_fold_elimination(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                                         selected_channels, candidate_pool, alpha, method):
#         """
#         Evaluate one inner CV fold for elimination.
#         For the "average" method, if selected_channels is nonempty, precompute the cumulative sum
#         over the currently selected channels and then, for each candidate in candidate_pool (which is a subset
#         of selected_channels), compute the new average as (cum_sum - candidate_data) / (count - 1).
#         For the concatenation method, it calls evaluate_channel_elimination.
#         Returns (best_candidate, best_accuracy) for this fold.
#         """
#         X_inner_train = X_train_val[inner_train_idx]
#         X_inner_val = X_train_val[inner_val_idx]
#         y_inner_train = y_train_val[inner_train_idx]
#         y_inner_val = y_train_val[inner_val_idx]
#
#         if method == "concatenation":
#             results = [evaluate_channel_elimination(ch, selected_channels, X_inner_train, y_inner_train,
#                                                     X_inner_val, y_inner_val, method=method)
#                        for ch in candidate_pool]
#         elif method == "average":
#             if selected_channels:
#                 cum_sum_train = np.sum(X_inner_train[:, :, selected_channels], axis=2)
#                 cum_sum_val = np.sum(X_inner_val[:, :, selected_channels], axis=2)
#                 count = len(selected_channels)
#             else:
#                 # Should not happen in removal mode.
#                 cum_sum_train = np.zeros_like(X_inner_train[:, :, 0])
#                 cum_sum_val = np.zeros_like(X_inner_val[:, :, 0])
#                 count = 0
#             results = evaluate_candidates_average_batch_elimination(cum_sum_train, cum_sum_val, count,
#                                                                     candidate_pool, X_inner_train, y_inner_train,
#                                                                     X_inner_val, y_inner_val, alpha)
#         else:
#             raise ValueError("Unsupported method. Choose 'concatenation' or 'average'.")
#         best_candidate, best_accuracy = max(results, key=lambda x: x[1])
#         return best_candidate, best_accuracy
#
#     def scale_data(X_train, X_test, scaler_type='standard'):
#         if scaler_type == 'standard':
#             scaler = StandardScaler()
#         elif scaler_type == 'minmax':
#             scaler = MinMaxScaler()
#         else:
#             raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
#         X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
#         X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
#         return X_train, X_test
#
#     def evaluate_candidate_baseline(current_selection, X_train, y_train, X_val, y_val, method='concatenation'):
#         """
#         Evaluate the performance (baseline accuracy) using the current selection of channels.
#         If current_selection is empty and method=='average', it returns 0.
#         """
#         from sklearn.linear_model import RidgeClassifier
#         from sklearn.metrics import balanced_accuracy_score
#         if method == "concatenation":
#             # Use all selected channels.
#             X_train_subset = X_train[:, :, current_selection].reshape(len(X_train), -1)
#             X_val_subset = X_val[:, :, current_selection].reshape(len(X_val), -1)
#         elif method == "average":
#             count = len(current_selection)
#             if count > 0:
#                 X_train_subset = np.sum(X_train[:, :, current_selection], axis=2) / count
#                 X_val_subset = np.sum(X_val[:, :, current_selection], axis=2) / count
#             else:
#                 return 0
#         else:
#             raise ValueError("Unsupported method. Choose 'concatenation' or 'average'.")
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(X_train_subset, y_train)
#         y_pred = model.predict(X_val_subset)
#         return balanced_accuracy_score(y_val, y_pred)
#
#     model = RidgeClassifier(alpha=alpha)
#
#     # ----------------------------------
#     # Set up interactive plotting.
#     # ----------------------------------
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # ----------------------------------
#     # Outer repetition loop.
#     # ----------------------------------
#     for repeat in range(num_outer_repeats):
#         print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
#         outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
#                                           random_state=rng.integers(0, int(1e6)))
#         train_val_idx, test_idx = next(outer_cv.split(data, labels))
#         X_train_val, X_test = data[train_val_idx], data[test_idx]
#         if normalize:
#             X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
#         y_train_val, y_test = labels[train_val_idx], labels[test_idx]
#
#         inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
#                                                    random_state=rng.integers(0, int(1e6)))
#
#         # In elimination mode, start with all channels.
#         current_selection = list(range(data.shape[2]))
#         selection_sequence = []  # Record the channel removed at each step.
#         rep_validation_accuracies = []  # Inner CV accuracy after each elimination step.
#         rep_test_accuracies = []  # Outer test accuracy after each elimination step.
#
#         # Compute baseline performance using the current selection.
#         baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#             delayed(evaluate_candidate_baseline)(current_selection,
#                                                  X_train_val[inner_train_idx],
#                                                  y_train_val[inner_train_idx],
#                                                  X_train_val[inner_val_idx],
#                                                  y_train_val[inner_val_idx],
#                                                  method=method)
#             for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#         )
#         baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0
#
#         for step in range(max_channels):
#             print(f"Selecting channel {step + 1}/{max_channels}")
#             # In removal mode, candidate pool = current selection.
#             candidate_pool = current_selection.copy()
#             if not candidate_pool:
#                 break
#
#             # Parallelize across inner CV folds.
#             fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_inner_fold_elimination)(
#                     inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                     current_selection, candidate_pool, alpha, method
#                 )
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#             best_candidate_per_fold = [result[0] for result in fold_results]
#             val_accuracies_per_fold = [result[1] for result in fold_results]
#
#             # Compute frequency for each candidate across inner folds.
#             freq_dict = {cand: best_candidate_per_fold.count(cand) for cand in candidate_pool}
#             candidates_meeting_freq = [cand for cand in candidate_pool if freq_dict[cand] >= min_frequency]
#             if candidates_meeting_freq:
#                 eligible = candidates_meeting_freq
#             else:
#                 eligible = candidate_pool
#
#             # Re-run candidate evaluation over inner folds to get average accuracy.
#             inner_folds = list(inner_cv.split(X_train_val, y_train_val))
#             candidate_avg_dict = {}
#             for cand in candidate_pool:
#                 fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                     delayed(evaluate_channel_elimination)(cand, current_selection,
#                                                           X_train_val[inner_train_idx],
#                                                           y_train_val[inner_train_idx],
#                                                           X_train_val[inner_val_idx],
#                                                           y_train_val[inner_val_idx],
#                                                           method=method)
#                     for inner_train_idx, inner_val_idx in inner_folds
#                 )
#                 candidate_avg_dict[cand] = np.mean([score for _, score in fold_scores])
#             best_candidate = max(eligible, key=lambda cand: candidate_avg_dict[cand] - baseline)
#             best_avg_score = candidate_avg_dict[best_candidate]
#             selection_sequence.append(best_candidate)
#             # Remove the best candidate.
#             current_selection.remove(best_candidate)
#             rep_validation_accuracies.append(best_avg_score)
#
#             # Evaluate outer test accuracy using the current selection.
#             if method == "concatenation":
#                 X_train_subset = X_train_val[:, :, current_selection].reshape(len(X_train_val), -1)
#                 X_test_subset = X_test[:, :, current_selection].reshape(len(X_test), -1)
#             elif method == "average":
#                 count = len(current_selection)
#                 X_train_subset = np.sum(X_train_val[:, :, current_selection], axis=2) / count
#                 X_test_subset = np.sum(X_test[:, :, current_selection], axis=2) / count
#             model.fit(X_train_subset, y_train_val)
#             y_pred = model.predict(X_test_subset)
#             rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#
#             # Update baseline.
#             baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_candidate_baseline)(current_selection,
#                                                      X_train_val[inner_train_idx],
#                                                      y_train_val[inner_train_idx],
#                                                      X_train_val[inner_val_idx],
#                                                      y_train_val[inner_val_idx],
#                                                      method=method)
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#             baseline = np.mean(baseline_fold_scores)
#
#         # End of elimination loop for this outer repetition.
#         all_selected_channels.append(selection_sequence.copy())
#         all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
#         all_test_accuracy_per_step.append(rep_test_accuracies.copy())
#
#         # Plot repetition-level results.
#         num_steps = len(rep_validation_accuracies)
#         rep_avg_val = np.atleast_1d(np.mean(np.array(rep_validation_accuracies), axis=0))
#         rep_std_val = np.atleast_1d(np.std(np.array(rep_validation_accuracies), axis=0))
#         rep_avg_test = np.atleast_1d(np.mean(np.array(rep_test_accuracies), axis=0))
#         rep_std_test = np.atleast_1d(np.std(np.array(rep_test_accuracies), axis=0))
#         fig, ax = plt.subplots(figsize=(10, 6))
#         steps_range = range(1, num_steps + 1)
#         xlabel = "Number of Removed Channels"
#         ax.plot(steps_range, rep_avg_val, marker='s', linestyle='-', label='Validation Accuracy')
#         ax.fill_between(steps_range, rep_avg_val - rep_std_val, rep_avg_val + rep_std_val, alpha=0.2)
#         ax.plot(steps_range, rep_avg_test, marker='o', linestyle='--', label='Test Accuracy')
#         ax.fill_between(steps_range, rep_avg_test - rep_std_test, rep_avg_test + rep_std_test, alpha=0.2)
#         ax.set_xlabel(xlabel)
#         ax.set_ylabel("Balanced Accuracy")
#         ax.set_title(f"Validation & Test Accuracy Progression (Repetition {repeat + 1})")
#         ax.legend()
#         ax.grid()
#         if selection_sequence:
#             for i, ch in enumerate(selection_sequence):
#                 ax.annotate(str(ch), (i + 1, rep_avg_test[i]),
#                             textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10)
#         plt.show(block=False)
#         plt.pause(0.5)
#         plt.close(fig)
#     # End of outer repetitions.
#
#     # Global aggregation and plotting (Average across outer repetitions)
#     num_reps = len(all_test_accuracy_per_step)
#     min_steps = min(len(rep) for rep in all_test_accuracy_per_step)
#     global_val = []
#     global_val_std = []
#     global_test = []
#     global_test_std = []
#     for step in range(min_steps):
#         val_accs = [all_validation_accuracy_per_step[r][step] for r in range(num_reps)]
#         test_accs = [all_test_accuracy_per_step[r][step] for r in range(num_reps)]
#         global_val.append(np.mean(val_accs))
#         global_val_std.append(np.std(val_accs))
#         global_test.append(np.mean(test_accs))
#         global_test_std.append(np.std(test_accs))
#     final_selected_channels = []
#     max_steps = max(len(seq) for seq in all_selected_channels)
#     for step in range(max_steps):
#         candidate_info = []
#         for rep in range(len(all_selected_channels)):
#             if step < len(all_selected_channels[rep]) and step < len(all_validation_accuracy_per_step[rep]):
#                 ch = all_selected_channels[rep][step]
#                 score = all_validation_accuracy_per_step[rep][step]
#                 candidate_info.append((ch, score))
#         if candidate_info:
#             freq = {}
#             avg_score = {}
#             for ch, score in candidate_info:
#                 freq[ch] = freq.get(ch, 0) + 1
#                 avg_score[ch] = avg_score.get(ch, 0) + score
#             for ch in avg_score:
#                 avg_score[ch] /= freq[ch]
#             best_candidate = None
#             best_freq = -1
#             best_avg = -np.inf
#             for ch in freq:
#                 if freq[ch] > best_freq or (freq[ch] == best_freq and avg_score[ch] > best_avg):
#                     best_candidate = ch
#                     best_freq = freq[ch]
#                     best_avg = avg_score[ch]
#             final_selected_channels.append(best_candidate)
#     fig, ax = plt.subplots(figsize=(10, 6))
#     steps_axis = np.arange(1, min_steps + 1)
#     ax.plot(steps_axis, global_val, marker='s', linestyle='-', label='Validation Accuracy')
#     ax.fill_between(steps_axis, np.array(global_val) - np.array(global_val_std),
#                     np.array(global_val) + np.array(global_val_std), alpha=0.2)
#     ax.plot(steps_axis, global_test, marker='o', linestyle='--', label='Test Accuracy')
#     ax.fill_between(steps_axis, np.array(global_test) - np.array(global_test_std),
#                     np.array(global_test) + np.array(global_test_std), alpha=0.2)
#     xlabel = "Number of Selected Channels"  # in elimination mode, these are the channels that remain.
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel("Balanced Accuracy")
#     ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
#     ax.legend()
#     ax.grid()
#     for i, ch in enumerate(final_selected_channels):
#         ax.annotate(str(ch), (steps_axis[i], global_test[i]),
#                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10)
#     plt.draw()
#     plt.pause(1.0)
#
#     plt.ioff()
#     plt.show()
#
#     return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step

# THIS VERSION PLOTS DYNAMICALLY AFTER EACH REPEAT
def greedy_nested_cv_channel_elimination(
        data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
        max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
        parallel=True, n_jobs=-1, method='concatenation', min_frequency=3):
    """
    Perform nested CV for greedy channel elimination using SNR-based candidate evaluation.
    (Backward elimination: start with all channels and, at each step, remove the channel whose removal
    yields the best (or least harmed) validation accuracy.)

    After each outer repetition (using a single outer split), the global average performance (over all outer
    repetitions so far) is recalculated and a global plot is updated showing the progression of performance
    across elimination steps. In the global aggregation, for each elimination step candidate channels that appear
    in fewer than min_frequency inner folds are discarded. Among those remaining, the candidate with the highest
    average SNR (average validation accuracy divided by standard deviation) is chosen and annotated with the
    number of outer repetitions that selected that channel.

    Returns:
      - all_selected_channels (list): List (per outer repetition) of removal sequences (i.e. channels removed in order).
      - all_test_accuracy_per_step (list): List (per outer repetition) of test accuracy lists (one per elimination step).
      - all_validation_accuracy_per_step (list): List (per outer repetition) of validation accuracy lists (one per elimination step).
    """
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt

    # Set random seed if not provided.
    if random_seed is None:
        random_seed = np.random.randint(0, int(1e6))
    rng = np.random.default_rng(random_seed)

    # Global accumulators.
    all_selected_channels = []          # One elimination sequence per outer repetition.
    all_test_accuracy_per_step = []       # Outer test accuracies per elimination step.
    all_validation_accuracy_per_step = [] # Inner CV (validation) accuracies per elimination step.

    # ----------------------------------
    # Custom CV classes.
    # ----------------------------------
    class LeaveOneFromEachClassCV(BaseCrossValidator):
        def __init__(self, shuffle=True, random_state=None):
            self.shuffle = shuffle
            self.random_state = random_state
        def get_n_splits(self, X, y, groups=None):
            _, counts = np.unique(y, return_counts=True)
            return int(np.min(counts))
        def split(self, X, y, groups=None):
            indices_by_class = {}
            for idx, label in enumerate(y):
                indices_by_class.setdefault(label, []).append(idx)
            rng_local = np.random.default_rng(self.random_state)
            for label in indices_by_class:
                if self.shuffle:
                    rng_local.shuffle(indices_by_class[label])
            n_splits = self.get_n_splits(X, y)
            for split in range(n_splits):
                test_indices = []
                for label, indices in indices_by_class.items():
                    test_indices.append(indices[split])
                test_indices = np.array(test_indices)
                train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                yield train_indices, test_indices

    class RepeatedLeaveOneFromEachClassCV(BaseCrossValidator):
        def __init__(self, n_repeats=50, shuffle=True, random_state=None):
            self.n_repeats = n_repeats
            self.shuffle = shuffle
            self.random_state = random_state
        def get_n_splits(self, X, y, groups=None):
            return self.n_repeats
        def split(self, X, y, groups=None):
            indices_by_class = {}
            for idx, label in enumerate(y):
                indices_by_class.setdefault(label, []).append(idx)
            rng_local = np.random.default_rng(self.random_state)
            for _ in range(self.n_repeats):
                test_indices = []
                for label, indices in indices_by_class.items():
                    if self.shuffle:
                        chosen = rng_local.choice(indices, size=1, replace=False)
                    else:
                        chosen = [indices[0]]
                    test_indices.extend(chosen)
                test_indices = np.array(test_indices)
                train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
                yield train_indices, test_indices

    # ----------------------------------
    # Helper functions.
    # ----------------------------------
    def evaluate_channel_elimination(ch, selected_channels, X_train, y_train, X_val, y_val, method=method):
        """
        Evaluate performance when channel ch is removed from the current selection.
        For "concatenation", candidate set = [x for x in selected_channels if x != ch].
        For "average", it computes the mean over that candidate set.
        """
        candidate_channels = [x for x in selected_channels if x != ch]
        if method == "concatenation":
            X_train_subset = X_train[:, :, candidate_channels].reshape(len(X_train), -1)
            X_val_subset = X_val[:, :, candidate_channels].reshape(len(X_val), -1)
        elif method == "average":
            X_train_subset = np.mean(X_train[:, :, candidate_channels], axis=2)
            X_val_subset = np.mean(X_val[:, :, candidate_channels], axis=2)
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_val_subset)
        return ch, balanced_accuracy_score(y_val, y_pred)

    def evaluate_candidates_average_batch_elimination(cum_sum_train, cum_sum_val, count,
                                                      candidate_channels, X_train, y_train, X_val, y_val, alpha):
        """
        Batch-evaluate candidate channels for the "average" method in elimination mode.
        New average if channel c is removed is computed as:
            (cum_sum - X[:,:,c]) / (count - 1)
        """
        n_train = X_train.shape[0]
        n_val = X_val.shape[0]
        candidate_train = X_train[:, :, candidate_channels]
        candidate_val = X_val[:, :, candidate_channels]
        avg_train = (cum_sum_train[..., None] - candidate_train) / (count - 1)
        avg_val = (cum_sum_val[..., None] - candidate_val) / (count - 1)
        n_candidates = avg_train.shape[2]
        results = []
        for i in range(n_candidates):
            X_train_candidate = avg_train[:, :, i]
            X_val_candidate = avg_val[:, :, i]
            model = RidgeClassifier(alpha=alpha)
            model.fit(X_train_candidate, y_train)
            y_pred = model.predict(X_val_candidate)
            score = balanced_accuracy_score(y_val, y_pred)
            results.append((candidate_channels[i], score))
        return results

    def evaluate_inner_fold_elimination(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                                        selected_channels, candidate_pool, alpha, method):
        """
        Evaluate one inner CV fold in elimination mode.
        For the "average" method, if selected_channels is nonempty, precompute the cumulative sum
        over the currently selected channels and then, for each candidate in candidate_pool (a subset
        of selected_channels), compute the new average as (cum_sum - candidate data) / (count - 1).
        For "concatenation", it calls evaluate_channel_elimination.
        Returns (best_candidate, best_accuracy) for this fold.
        """
        X_inner_train = X_train_val[inner_train_idx]
        X_inner_val = X_train_val[inner_val_idx]
        y_inner_train = y_train_val[inner_train_idx]
        y_inner_val = y_train_val[inner_val_idx]
        if method == "concatenation":
            results = [evaluate_channel_elimination(ch, selected_channels, X_inner_train, y_inner_train,
                                                    X_inner_val, y_inner_val, method=method)
                       for ch in candidate_pool]
        elif method == "average":
            if selected_channels:
                cum_sum_train = np.sum(X_inner_train[:, :, selected_channels], axis=2)
                cum_sum_val = np.sum(X_inner_val[:, :, selected_channels], axis=2)
                count = len(selected_channels)
            else:
                cum_sum_train = np.zeros_like(X_inner_train[:, :, 0])
                cum_sum_val = np.zeros_like(X_inner_val[:, :, 0])
                count = 0
            results = evaluate_candidates_average_batch_elimination(cum_sum_train, cum_sum_val, count,
                                                                    candidate_pool, X_inner_train, y_inner_train,
                                                                    X_inner_val, y_inner_val, alpha)
        else:
            raise ValueError("Unsupported method. Choose 'concatenation' or 'average'.")
        best_candidate, best_accuracy = max(results, key=lambda x: x[1])
        return best_candidate, best_accuracy

    def scale_data(X_train, X_test, scaler_type='standard'):
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
        X_train = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
        X_test = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)
        return X_train, X_test

    def evaluate_candidate_baseline(current_selection, X_train, y_train, X_val, y_val, method='concatenation'):
        """
        Evaluate the baseline performance (accuracy) using the current selection.
        If the current selection is empty (for average), return 0.
        """
        if method == "concatenation":
            X_train_subset = X_train[:, :, current_selection].reshape(len(X_train), -1)
            X_val_subset = X_val[:, :, current_selection].reshape(len(X_val), -1)
        elif method == "average":
            count = len(current_selection)
            if count > 0:
                X_train_subset = np.sum(X_train[:, :, current_selection], axis=2) / count
                X_val_subset = np.sum(X_val[:, :, current_selection], axis=2) / count
            else:
                return 0
        else:
            raise ValueError("Unsupported method. Choose 'concatenation' or 'average'.")
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_val_subset)
        return balanced_accuracy_score(y_val, y_pred)

    model = RidgeClassifier(alpha=alpha)

    # ----------------------------------
    # Set up interactive plotting.
    # ----------------------------------
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # ----------------------------------
    # Outer repetition loop.
    # ----------------------------------
    for repeat in range(num_outer_repeats):
        print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
        outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                          random_state=rng.integers(0, int(1e6)))
        train_val_idx, test_idx = next(outer_cv.split(data, labels))
        X_train_val, X_test = data[train_val_idx], data[test_idx]
        if normalize:
            X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
                                                   random_state=rng.integers(0, int(1e6)))

        # In elimination mode, start with all channels.
        current_selection = list(range(data.shape[2]))
        selection_sequence = []  # Record the channel removed at each step.
        rep_validation_accuracies = []  # Inner CV accuracy per elimination step.
        rep_test_accuracies = []  # Outer test accuracy per elimination step.

        # Compute baseline performance using the current selection.
        baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_candidate_baseline)(current_selection,
                                                 X_train_val[inner_train_idx],
                                                 y_train_val[inner_train_idx],
                                                 X_train_val[inner_val_idx],
                                                 y_train_val[inner_val_idx],
                                                 method=method)
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
        )
        baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0

        for step in range(max_channels):
            print(f"Selecting channel {step + 1}/{max_channels}")
            # In removal mode, candidate pool = current selection.
            candidate_pool = current_selection.copy()
            if not candidate_pool:
                break

            # Parallelize across inner CV folds.
            fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_inner_fold_elimination)(
                    inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                    current_selection, candidate_pool, alpha, method
                )
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )
            best_candidate_per_fold = [result[0] for result in fold_results]
            val_accuracies_per_fold = [result[1] for result in fold_results]

            # Compute frequency for each candidate across inner folds.
            freq_dict = {cand: best_candidate_per_fold.count(cand) for cand in candidate_pool}
            candidates_meeting_freq = [cand for cand in candidate_pool if freq_dict[cand] >= min_frequency]
            if candidates_meeting_freq:
                eligible = candidates_meeting_freq
            else:
                eligible = candidate_pool

            # Re-run candidate evaluation over inner folds to get average accuracy.
            inner_folds = list(inner_cv.split(X_train_val, y_train_val))
            candidate_avg_dict = {}
            for cand in candidate_pool:
                fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(evaluate_channel_elimination)(cand, current_selection,
                                                          X_train_val[inner_train_idx],
                                                          y_train_val[inner_train_idx],
                                                          X_train_val[inner_val_idx],
                                                          y_train_val[inner_val_idx],
                                                          method=method)
                    for inner_train_idx, inner_val_idx in inner_folds
                )
                candidate_avg_dict[cand] = np.mean([score for _, score in fold_scores])
            best_candidate = max(eligible, key=lambda cand: candidate_avg_dict[cand] - baseline)
            best_avg_score = candidate_avg_dict[best_candidate]
            selection_sequence.append(best_candidate)
            # Remove the best candidate.
            current_selection.remove(best_candidate)
            rep_validation_accuracies.append(best_avg_score)

            # Evaluate outer test accuracy using the current selection.
            if method == "concatenation":
                X_train_subset = X_train_val[:, :, current_selection].reshape(len(X_train_val), -1)
                X_test_subset = X_test[:, :, current_selection].reshape(len(X_test), -1)
            elif method == "average":
                count = len(current_selection)
                X_train_subset = np.sum(X_train_val[:, :, current_selection], axis=2) / count
                X_test_subset = np.sum(X_test[:, :, current_selection], axis=2) / count
            model.fit(X_train_subset, y_train_val)
            y_pred = model.predict(X_test_subset)
            rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))

            # Update baseline.
            baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_candidate_baseline)(current_selection,
                                                     X_train_val[inner_train_idx],
                                                     y_train_val[inner_train_idx],
                                                     X_train_val[inner_val_idx],
                                                     y_train_val[inner_val_idx],
                                                     method=method)
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )
            baseline = np.mean(baseline_fold_scores)

        # End of elimination loop for this outer repetition.
        all_selected_channels.append(selection_sequence.copy())
        all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
        all_test_accuracy_per_step.append(rep_test_accuracies.copy())

        # ----- Dynamic Global Aggregation and Plotting After Each Repetition -----
        num_reps = len(all_test_accuracy_per_step)
        min_steps = min(len(rep) for rep in all_test_accuracy_per_step)
        global_val = []
        global_val_std = []
        global_test = []
        global_test_std = []
        for step in range(min_steps):
            val_accs = [all_validation_accuracy_per_step[r][step] for r in range(num_reps)]
            test_accs = [all_test_accuracy_per_step[r][step] for r in range(num_reps)]
            global_val.append(np.mean(val_accs))
            global_val_std.append(np.std(val_accs))
            global_test.append(np.mean(test_accs))
            global_test_std.append(np.std(test_accs))
        final_selected_channels = []
        max_steps = max(len(seq) for seq in all_selected_channels)
        for step in range(max_steps):
            candidate_info = []
            for rep in range(len(all_selected_channels)):
                if step < len(all_selected_channels[rep]) and step < len(all_validation_accuracy_per_step[rep]):
                    ch = all_selected_channels[rep][step]
                    score = all_validation_accuracy_per_step[rep][step]
                    candidate_info.append((ch, score))
            if candidate_info:
                freq = {}
                avg_score = {}
                for ch, score in candidate_info:
                    freq[ch] = freq.get(ch, 0) + 1
                    avg_score[ch] = avg_score.get(ch, 0) + score
                for ch in avg_score:
                    avg_score[ch] /= freq[ch]
                best_candidate = None
                best_freq = -1
                best_avg = -np.inf
                for ch in freq:
                    if freq[ch] > best_freq or (freq[ch] == best_freq and avg_score[ch] > best_avg):
                        best_candidate = ch
                        best_freq = freq[ch]
                        best_avg = avg_score[ch]
                final_selected_channels.append(best_candidate)
        ax.clear()
        steps_axis = np.arange(1, min_steps + 1)
        ax.plot(steps_axis, global_val, marker='s', linestyle='-', label='Validation Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_val) - np.array(global_val_std),
                        np.array(global_val) + np.array(global_val_std), alpha=0.2)
        ax.plot(steps_axis, global_test, marker='o', linestyle='--', label='Test Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_test) - np.array(global_test_std),
                        np.array(global_test) + np.array(global_test_std), alpha=0.2)
        ax.set_xlabel("Number of Selected Channels")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
        ax.legend()
        ax.grid()
        for i, ch in enumerate(final_selected_channels):
            ax.annotate(str(ch), (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, 5), ha="center", fontsize=10)
        plt.draw()
        plt.pause(1.0)
        # End dynamic global plotting for this repetition.

    plt.ioff()
    plt.show()

    return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step



def classify_all_channels(data, labels, alpha=1.0, test_size=0.2, num_splits=5, use_pca=False, n_components=None):
    """
    Perform classification on all concatenated m/z channels with optional PCA for dimensionality reduction.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array with shape (samples, features, channels).
    labels : np.ndarray
        Array of labels for the samples.
    alpha : float
        Regularization strength for RidgeClassifier.
    test_size : float
        Proportion of data to use for testing.
    num_splits : int
        Number of train-test splits for averaging accuracy.
    use_pca : bool
        Whether to apply PCA for dimensionality reduction.
    n_components : int or None
        Number of PCA components to retain. If None, PCA will not be applied.

    Returns
    -------
    dict
        A dictionary containing mean accuracy, balanced accuracy, and explained variance (if PCA is used).
    """
    # Flatten channels into a single feature vector per sample
    flattened_data = data.transpose(0, 2, 1).reshape(data.shape[0], -1)

    # Initialize metrics
    accuracies = []
    balanced_accuracies = []
    explained_variance_ratios = [] if use_pca else None

    print("Starting classification with all concatenated channels...")
    model = RidgeClassifier(alpha=alpha)

    # Cross-validation loop
    for split in range(num_splits):
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            flattened_data, labels, test_size=test_size, stratify=labels, random_state=split
        )

        # Apply PCA if enabled
        if use_pca:
            pca = PCA(n_components=n_components)
            X_train = pca.fit_transform(X_train)
            X_test = pca.transform(X_test)

            # Track explained variance
            explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))

        # Train and evaluate the Ridge Classifier
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        accuracies.append(model.score(X_test, y_test))
        balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))

        print(f"Split {split + 1}/{num_splits} completed.")

    # Calculate mean metrics
    mean_accuracy = np.mean(accuracies)
    mean_balanced_accuracy = np.mean(balanced_accuracies)
    mean_explained_variance = np.mean(explained_variance_ratios) if use_pca else None

    # Print summary
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean Balanced Accuracy: {mean_balanced_accuracy:.4f}")
    if use_pca:
        print(f"Mean Explained Variance (PCA): {mean_explained_variance:.4f}")

    # Return metrics
    results = {
        "mean_accuracy": mean_accuracy,
        "mean_balanced_accuracy": mean_balanced_accuracy,
    }
    if use_pca:
        results["mean_explained_variance"] = mean_explained_variance

    return results


# def classify_all_channels(data, labels, alpha=1.0, test_size=0.2, num_splits=5, use_pca=False, n_components=None, use_groups=True):
#     from sklearn.model_selection import train_test_split
#     from sklearn.linear_model import RidgeClassifier
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.decomposition import PCA
#     from sklearn.utils import shuffle
#     import numpy as np
#
#     def extract_category_labels(composite_labels):
#         """
#         Convert composite labels (e.g., 'A1', 'B9', 'C2') to category labels ('A', 'B', 'C').
#
#         Args:
#             composite_labels (array-like): List or array of composite labels.
#
#         Returns:
#             list: List of category labels.
#         """
#         return [re.match(r'([A-C])', label).group(1) if re.match(r'([A-C])', label) else label
#                 for label in composite_labels]
#
#     class RepeatedLeaveOneFromEachClassCV:
#         def __init__(self, n_repeats=5, shuffle=True, random_state=None, use_groups=True):
#             self.n_repeats = n_repeats
#             self.shuffle = shuffle
#             self.random_state = random_state
#             self.use_groups = use_groups
#
#         def split(self, X, y):
#             unique_labels = np.unique(y)
#             rng = np.random.default_rng(self.random_state)
#
#             for _ in range(self.n_repeats):
#                 test_indices = []
#                 for label in unique_labels:
#                     indices = np.where(np.char.equal(y, label))[0]
#                     if len(indices) == 0:
#                         continue
#                     if self.use_groups:
#                         chosen_index = rng.choice(indices, size=1, replace=False).astype(int)
#                         test_indices.extend(chosen_index)
#                     else:
#                         chosen_index = rng.choice(indices, size=1, replace=False).astype(int)
#                         test_indices.append(int(chosen_index[0]))
#
#                 test_indices = np.array(test_indices, dtype=int)
#                 train_indices = np.setdiff1d(np.arange(len(y)), test_indices)
#                 yield np.array(train_indices, dtype=int), np.array(test_indices, dtype=int)
#
#     flattened_data = data.transpose(0, 2, 1).reshape(data.shape[0], -1)
#
#     accuracies = []
#     balanced_accuracies = []
#     explained_variance_ratios = [] if use_pca else None
#
#     print("Starting classification with balanced cross-validation...")
#     model = RidgeClassifier(alpha=alpha)
#
#     cv = RepeatedLeaveOneFromEachClassCV(n_repeats=num_splits, use_groups=use_groups)
#
#     for split_idx, (train_idx, test_idx) in enumerate(cv.split(flattened_data, labels)):
#         X_train, X_test = flattened_data[train_idx.astype(int)], flattened_data[test_idx.astype(int)]
#         y_train, y_test = np.array(labels)[train_idx.astype(int)], np.array(labels)[test_idx.astype(int)]
#
#         # Apply extract_category_labels before training
#         y_train = extract_category_labels(y_train)
#         y_test = extract_category_labels(y_test)
#
#         if use_pca:
#             pca = PCA(n_components=n_components)
#             X_train = pca.fit_transform(X_train)
#             X_test = pca.transform(X_test)
#             explained_variance_ratios.append(np.sum(pca.explained_variance_ratio_))
#
#         model.fit(X_train, y_train)
#         y_pred = model.predict(X_test)
#
#         accuracies.append(model.score(X_test, y_test))
#         balanced_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#
#         print(f"Split {split_idx + 1}/{num_splits} completed.")
#
#     mean_accuracy = np.mean(accuracies)
#     mean_balanced_accuracy = np.mean(balanced_accuracies)
#     mean_explained_variance = np.mean(explained_variance_ratios) if use_pca else None
#
#     print(f"Mean Accuracy: {mean_accuracy:.4f}")
#     print(f"Mean Balanced Accuracy: {mean_balanced_accuracy:.4f}")
#     if use_pca:
#         print(f"Mean Explained Variance (PCA): {mean_explained_variance:.4f}")
#
#     results = {
#         "mean_accuracy": mean_accuracy,
#         "mean_balanced_accuracy": mean_balanced_accuracy,
#     }
#     if use_pca:
#         results["mean_explained_variance"] = mean_explained_variance
#
#     return results


def remove_highly_correlated_channels(data, correlation_threshold=0.9):
    """
    Removes highly correlated channels from the data based on the correlation of their full feature profiles.

    Parameters
    ----------
    data : np.ndarray
        Input 3D array with shape (samples, features, channels).
    correlation_threshold : float
        Threshold for considering two channels as highly correlated.

    Returns
    -------
    np.ndarray
        Reduced data with highly correlated channels removed.
    list
        Indices of retained channels.
    """
    # Reshape data to merge samples and features into one dimension for correlation
    reshaped_data = data.transpose(2, 0, 1).reshape(data.shape[2], -1)  # Shape: (channels, samples * features)

    # Compute the correlation matrix across channels
    correlation_matrix = np.corrcoef(reshaped_data)

    # Keep track of selected channels
    num_channels = correlation_matrix.shape[0]
    selected_channels = []

    # Track channels to exclude
    excluded_channels = set()

    for i in range(num_channels):
        if i not in excluded_channels:
            selected_channels.append(i)
            # Mark all highly correlated channels for exclusion
            for j in range(i + 1, num_channels):
                if abs(correlation_matrix[i, j]) > correlation_threshold:
                    excluded_channels.add(j)

    # Select only the retained channels
    reduced_data = data[:, :, selected_channels]

    return reduced_data, selected_channels