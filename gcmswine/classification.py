import numpy as np
np.set_printoptions(linewidth=200)  # or higher if needed
from pynndescent.optimal_transport import total_cost
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, BaseCrossValidator
from sklearn.utils import resample
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_selection import mutual_info_classif

from gcmswine import utils
from gcmswine.dimensionality_reduction import DimensionalityReducer
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

from gcmswine.utils import find_first_and_last_position, normalize_dict, normalize_data
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

def leave_one_sample_per_class_split(X, y, random_state=None, is_composite_labels=True):
    """
    Split dataset by selecting one sample per class (plus its duplicates if composite labels) for the test set.

    Parameters
    ----------
    X : array-like
        Feature matrix (not used, kept for compatibility).
    y : array-like
        Labels (composite labels or simple class labels).
    random_state : int, optional
        Random seed for reproducibility.
    is_composite_labels : bool, optional
        Whether y are composite labels like 'A1', 'B2'. If False, assumes simple class labels.

    Returns
    -------
    train_indices : np.ndarray
        Indices for training set.
    test_indices : np.ndarray
        Indices for test set.
    """
    import numpy as np
    from collections import defaultdict
    import re

    rng = np.random.default_rng(random_state)

    if is_composite_labels:
        # Group indices by full sample labels (for composite cases)
        sample_to_indices = defaultdict(list)
        for idx, label in enumerate(y):
            sample_to_indices[label].append(idx)

        # Group sample labels by class (A, B, C)
        class_to_samples = defaultdict(list)

        def get_class_from_label(label):
            """Extract the class (first letter) from composite label like 'A1'."""
            match = re.match(r'([A-Z])', label)
            return match.group(1) if match else label

        for label in sample_to_indices.keys():
            class_label = get_class_from_label(label)
            class_to_samples[class_label].append(label)

        # Choose one sample label per class
        test_sample_labels = []
        for class_label, samples in class_to_samples.items():
            chosen_sample = rng.choice(samples)
            test_sample_labels.append(chosen_sample)

        # Expand into indices
        test_indices = []
        for label in test_sample_labels:
            test_indices.extend(sample_to_indices[label])

    else:
        # Simple class labels (e.g., 'Beaune', 'Alsace', 'C', 'D')
        class_to_indices = defaultdict(list)
        for idx, label in enumerate(y):
            class_to_indices[label].append(idx)

        # Pick one random index per class
        test_indices = []
        for class_label, indices in class_to_indices.items():
            chosen_idx = rng.choice(indices)
            test_indices.append(chosen_idx)

    test_indices = np.array(test_indices)
    all_indices = np.arange(len(y))
    train_indices = np.setdiff1d(all_indices, test_indices)

    return train_indices, test_indices

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


    def train_and_evaluate_balanced(self, n_inner_repeats=50, random_seed=42,
                                    test_size=0.2, normalize=False, scaler_type='standard',
                                    use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
                                    test_on_discarded=False, LOOPC=True):
        """
        Train and evaluate the classifier using a train/test split followed by cross-validation on the training set.
        Evaluation metrics are averaged across multiple inner cross-validation repetitions.

        Depending on the value of `test_on_discarded`, either the inner validation set or the held-out test set
        is used to compute final metrics. Supports normalization, PCA, and custom confusion matrix ordering.

        Parameters
        ----------
        n_inner_repeats : int, optional (default=50)
            Number of inner cross-validation folds or repetitions.
        random_seed : int, optional (default=42)
            Seed for reproducibility.
        test_size : float, optional (default=0.2)
            Fraction of data to hold out as a test set (only used when test_on_discarded=True).
        normalize : bool, optional (default=False)
            Whether to apply feature normalization (e.g., standard scaling).
        scaler_type : str, optional (default='standard')
            Type of scaler to use if normalization is enabled ('standard', 'minmax', etc.).
        use_pca : bool, optional (default=False)
            Whether to apply PCA for dimensionality reduction.
        vthresh : float, optional (default=0.97)
            Proportion of variance to retain when applying PCA.
        region : str or None, optional (default=None)
            Custom region ordering for confusion matrix labels.
        print_results : bool, optional (default=True)
            Whether to print performance metrics to the console.
        n_jobs : int, optional (default=-1)
            Number of parallel jobs to use for cross-validation.
        test_on_discarded : bool, optional (default=False)
            If True, final evaluation is done on the held-out test set (outside CV).
            If False, metrics are computed on the inner cross-validation folds.
        LOOPC : bool, optional (default=True)
            If True, use Leave-One-Out-Per-Class strategy for cross-validation (i.e., select one full sample per class).
            If False, use standard cross-validation with grouped or stratified shuffle splits.

        Returns
        -------
        dict
            Dictionary containing average accuracy, precision, recall, F1-score, and normalized confusion matrix
            across all evaluation folds or test splits.
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

        def leave_one_sample_per_class_split(X, y, random_state=None, is_composite_labels=True):
            """
            Split dataset by selecting one sample per class (plus its duplicates if composite labels) for the test set.

            Parameters
            ----------
            X : array-like
                Feature matrix (not used, kept for compatibility).
            y : array-like
                Labels (composite labels or simple class labels).
            random_state : int, optional
                Random seed for reproducibility.
            is_composite_labels : bool, optional
                Whether y are composite labels like 'A1', 'B2'. If False, assumes simple class labels.

            Returns
            -------
            train_indices : np.ndarray
                Indices for training set.
            test_indices : np.ndarray
                Indices for test set.
            """
            import numpy as np
            from collections import defaultdict
            import re

            rng = np.random.default_rng(random_state)

            if is_composite_labels:
                # Group indices by full sample labels (for composite cases)
                sample_to_indices = defaultdict(list)
                for idx, label in enumerate(y):
                    sample_to_indices[label].append(idx)

                # Group sample labels by class (A, B, C)
                class_to_samples = defaultdict(list)

                def get_class_from_label(label):
                    """Extract the class (first letter) from composite label like 'A1'."""
                    match = re.match(r'([A-Z])', label)
                    return match.group(1) if match else label

                for label in sample_to_indices.keys():
                    class_label = get_class_from_label(label)
                    class_to_samples[class_label].append(label)

                # Choose one sample label per class
                test_sample_labels = []
                for class_label, samples in class_to_samples.items():
                    chosen_sample = rng.choice(samples)
                    test_sample_labels.append(chosen_sample)

                # Expand into indices
                test_indices = []
                for label in test_sample_labels:
                    test_indices.extend(sample_to_indices[label])

            else:
                # Simple class labels (e.g., 'Beaune', 'Alsace', 'C', 'D')
                class_to_indices = defaultdict(list)
                for idx, label in enumerate(y):
                    class_to_indices[label].append(idx)

                # Pick one random index per class
                test_indices = []
                for class_label, indices in class_to_indices.items():
                    chosen_idx = rng.choice(indices)
                    test_indices.append(chosen_idx)

            test_indices = np.array(test_indices)
            all_indices = np.arange(len(y))
            train_indices = np.setdiff1d(all_indices, test_indices)

            return train_indices, test_indices


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


        if region == "winery":
            category_labels = extract_category_labels(self.labels)
        else:
            category_labels = self.labels

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
        custom_order = utils.get_custom_order_for_pinot_noir_region(region)

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

        if LOOPC:
            train_idx, test_idx = leave_one_sample_per_class_split(self.data, self.labels, random_state=random_seed,
                                                                   is_composite_labels=False)
        else:
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

            # if self.year_labels.size > 0 and np.any(self.year_labels != None):
            #     self.classifier.fit(X_train_full, y_train_full)
            # else:
            #     self.classifier.fit(X_train_full, np.array(extract_category_labels(y_train_full)))
            #     y_test = extract_category_labels(y_test)

            try:
                if self.year_labels.size > 0 and np.any(self.year_labels != None):
                    self.classifier.fit(X_train_full, y_train_full)
                else:
                    if region == "winery":
                        self.classifier.fit(X_train_full, np.array(extract_category_labels(y_train_full)))
                    else:
                        self.classifier.fit(X_train_full, np.array(y_train_full))

            except np.linalg.LinAlgError:
                print(
                    "⚠️ Skipping evaluation due to SVD convergence error (likely caused by LDA on low-variance or singular data).")
                return {
                    'overall_accuracy': np.nan,
                    'overall_balanced_accuracy': np.nan,
                    'overall_weighted_accuracy': np.nan,
                    'overall_precision': np.nan,
                    'overall_recall': np.nan,
                    'overall_f1_score': np.nan,
                    'confusion_matrix': None,
                }
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
            eval_cm.append(confusion_matrix(y_test, y_pred, labels=custom_order))

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
        with np.errstate(invalid='ignore', divide='ignore'):
            overall_cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1, keepdims=True)
            overall_cm_normalized[np.isnan(overall_cm_normalized)] = 0

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


    def train_and_evaluate_all_channels(
            self, num_repeats=10, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
            feature_type="concatenated", classifier_type="RGC", LOOPC=True
    ):
        """
        Trains and evaluates a classifier using all available channels in the dataset,
        repeating the evaluation multiple times to assess performance stability.

        Feature extraction is flexible, allowing for different representations (e.g., concatenated raw data, TIC, TIS, or both).
        At each repeat, the data is randomly split into training and testing sets.

        Parameters:
        ----------
        num_repeats : int, optional (default=10)
            Number of times to repeat the training and evaluation process (with different random seeds).
        num_outer_repeats : int, optional (default=1)
            Currently unused. Reserved for compatibility with outer loop evaluations.
        random_seed : int, optional (default=42)
            Base random seed for reproducibility. A different seed is used at each repeat.
        test_size : float, optional (default=0.2)
            Fraction of the data used for testing in each train/test split.
        normalize : bool, optional (default=False)
            Whether to apply feature normalization (e.g., standard scaling) before training.
        scaler_type : str, optional (default='standard')
            Type of scaler to use if normalization is enabled. Options: 'standard', 'minmax', etc.
        use_pca : bool, optional (default=False)
            Whether to apply PCA for dimensionality reduction before training.
        vthresh : float, optional (default=0.97)
            Variance threshold to retain during PCA (if use_pca=True).
        region : str or None, optional (default=None)
            If specified, restricts training/testing to samples from a given region.
        print_results : bool, optional (default=True)
            Whether to print detailed results after evaluation.
        n_jobs : int, optional (default=-1)
            Number of CPU cores to use for training (if supported by the classifier).
        feature_type : str, optional (default='concatenated')
            Feature extraction mode: 'concatenated', 'tic', 'tis', or 'tic_tis'.
        classifier_type : str, optional (default='RGC')
            Type of classifier to use.

        Returns:
        -------
        mean_test_accuracy : float
            Average balanced accuracy over all repeats.
        std_test_accuracy : float
            Standard deviation of balanced accuracy over all repeats.

        Notes:
        -----
        - The method internally computes features according to the selected `feature_type`.
        - Results are averaged across all repeats to provide a robust estimate of performance.
        - Normalized confusion matrices are averaged across repeats if dimensions match.
        """
        cls_data = self.data.copy()
        labels = self.labels
        num_samples, num_timepoints, num_channels = cls_data.shape
        # Initialize lists
        balanced_accuracies = []
        confusion_matrices = []  # Store confusion matrices

        # --- Feature extraction helper ---
        def compute_features(channels):
            """Compute features based on the chosen feature type."""
            print(f"Computing features for channels: {channels}")
            if feature_type == "concatenated":
                # Flatten each selected channel across time and concatenate them
                return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            elif feature_type == "tic":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                return tic
            elif feature_type == "tis":
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return  tis
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated' or 'tic_tis'.")

         # --- Use all channels by default ---
        feature_matrix = compute_features(list(range(num_channels)))

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")
            # classifiers = ["DTC", "GNB", "KNN", "LDA", "LR", "PAC", "PER", "RFC", "RGC", "SGD", "SVM"]
            try:
                cls = Classifier(feature_matrix, labels, classifier_type=classifier_type, wine_kind=self.wine_kind,
                                 year_labels=self.year_labels)
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
                    test_on_discarded=True,
                    LOOPC=LOOPC
                )
                if 'overall_balanced_accuracy' in results and not np.isnan(results['overall_balanced_accuracy']):
                    balanced_accuracies.append(results['overall_balanced_accuracy'])
                else:
                    print(f"⚠️ No valid accuracy returned in repeat {repeat_idx + 1}")

                if 'overall_confusion_matrix_normalized' in results:
                    if confusion_matrices and results['overall_confusion_matrix_normalized'].shape != confusion_matrices[
                        0].shape:
                        print("⚠️ Skipping confusion matrix with different shape.")
                    else:
                        confusion_matrices.append(results['overall_confusion_matrix_normalized'])
                else:
                    print("⚠️ Skipping confusion matrix due to missing key.")

            except Exception as e:
                print(f"⚠️ Skipping repeat {repeat_idx + 1} due to error: {e}")

        # Compute average performance across repeats
        mean_test_accuracy = np.mean(balanced_accuracies, axis=0)
        std_test_accuracy = np.std(balanced_accuracies, axis=0)
        mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)
        # mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
        custom_order = utils.get_custom_order_for_pinot_noir_region(region)

        print("\n##################################")
        print(f"Mean Balanced Accuracy: {mean_test_accuracy:.3f} ± {std_test_accuracy:.3f}")
        # Print label order used in confusion matrix
        if custom_order is not None:
            print("\nLabel order (custom):")
            print(", ".join(custom_order))

        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        return mean_test_accuracy, std_test_accuracy


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


