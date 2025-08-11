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

from gcmswine.wine_kind_strategy import WineKindStrategy


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
        # Simple class labels (e.g., 'Burgundy', 'Alsace', 'C', 'D')
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
    def __init__(self, data, labels, classifier_type='LDA', strategy: WineKindStrategy = None,
                 wine_kind='bordeaux', class_by_year=False, window_size=5000, stride=2500,
                 alpha=1, year_labels= None, dataset_origins=None, sample_labels=None, **kwargs
                 ):
        self.data = data
        self.labels = labels
        self.sample_labels = sample_labels
        self.labels_raw = kwargs.get("labels_raw", labels)
        self.strategy = strategy if strategy else WineKindStrategy()
        self.window_size = window_size
        self.stride = stride
        self.wine_kind = wine_kind
        self.class_by_year = class_by_year
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
            return PassiveAggressiveClassifier(random_state=0, n_jobs=-1)
        elif classifier_type == 'PER':
            return Perceptron(random_state=0)
        elif classifier_type == 'RGC':
            return RidgeClassifier(alpha=self.alpha)
        elif classifier_type == 'SGD':
            return SGDClassifier(random_state=0)
        elif classifier_type == 'SVM':
            return SVC(kernel='rbf', random_state=0)
        elif classifier_type == 'KNN':
            return KNeighborsClassifier(n_neighbors=3)
        elif classifier_type == 'DTC':
            return DecisionTreeClassifier(random_state=0)
        elif classifier_type == 'GNB':
            return GaussianNB()
        elif classifier_type == 'GBC':
            return GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=0)
        elif classifier_type == 'HGBC':
            return HistGradientBoostingClassifier(
                max_leaf_nodes=31, learning_rate=0.2, max_iter=50, max_bins=128, random_state=0)


    # def extract_category_labels(self, composite_labels):
    #     return [re.match(r'([A-C])', label).group(1) if re.match(r'([A-C])', label) else label
    #             for label in composite_labels]

    def extract_category_labels(self, composite_labels):
        return [label[0] if label else None for label in composite_labels]



    def preprocess_data(self, X_train, X_val, normalize, scaler_type, use_pca, vthresh):
        scaler, pca = None, None

        if normalize:
            from sklearn.preprocessing import StandardScaler, MinMaxScaler
            scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)

        if use_pca:
            pca = PCA(n_components=None, svd_solver='randomized')
            pca.fit(X_train)
            cum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.searchsorted(cum_var, vthresh) + 1
            pca = PCA(n_components=n_components, svd_solver='randomized')
            X_train = pca.fit_transform(X_train)
            X_val = pca.transform(X_val)

        return X_train, X_val, scaler, pca

    def evaluate_fold(self, clf, X_train, y_train, X_val, y_val, custom_order):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = clf.score(X_val, y_val)
        bal_acc = balanced_accuracy_score(y_val, y_pred)
        sw = compute_sample_weight(class_weight='balanced', y=y_val)
        w_acc = np.average(y_pred == y_val, weights=sw)
        prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        cm = confusion_matrix(y_val, y_pred, labels=custom_order if custom_order else None)

        return acc, bal_acc, w_acc, prec, rec, f1, cm

    def run_cross_validation(self, cv, X, y, clf, normalize, scaler_type, use_pca, vthresh, custom_order, n_jobs):
        def process(inner_train_idx, inner_val_idx):
            X_train, X_val = X[inner_train_idx], X[inner_val_idx]
            y_train, y_val = y[inner_train_idx], y[inner_val_idx]

            X_train, X_val, _, _ = self.preprocess_data(X_train, X_val, normalize, scaler_type, use_pca, vthresh)
            return self.evaluate_fold(clf, X_train, y_train, X_val, y_val, custom_order)

        results = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(process)(train_idx, val_idx)
            for train_idx, val_idx in cv.split(X, y)
        )

        return results

    def summarize_cv_results(self, results):
        accs, bals, waccs, precs, recs, f1s, cms = zip(*results)

        summary = {
            'overall_accuracy': np.mean(accs),
            'overall_balanced_accuracy': np.mean(bals),
            'overall_weighted_accuracy': np.mean(waccs),
            'overall_precision': np.mean(precs),
            'overall_recall': np.mean(recs),
            'overall_f1_score': np.mean(f1s),
            'overall_confusion_matrix': np.mean(cms, axis=0),
        }

        with np.errstate(invalid='ignore', divide='ignore'):
            norm_cm = summary['overall_confusion_matrix'].astype('float')
            norm_cm /= norm_cm.sum(axis=1, keepdims=True)
            norm_cm[np.isnan(norm_cm)] = 0
            summary['overall_confusion_matrix_normalized'] = norm_cm

        return summary

    def train_and_evaluate_leave_one_out(
            self,
            left_out_index,
            normalize=False,
            scaler_type='standard',
            projection_source=False,
    ):
        # Step 1: label preprocessing
        processed_labels = np.array(self.strategy.extract_labels(self.labels))  # ← same labels used for y_train/y_test
        use_composites = self.strategy.use_composite_labels(self.labels)
        custom_order = self.strategy.get_custom_order(processed_labels, self.year_labels)
        # labels_used = self.strategy.extract_labels(self.labels)
        # use_composites = self.strategy.use_composite_labels(self.labels)
        # custom_order = self.strategy.get_custom_order(labels_used, self.year_labels)

        # Step 2: leave out one composite group
        split_labels = self.strategy.get_split_labels(self.labels_raw, self.class_by_year)

        n_samples = len(self.data)
        if not (0 <= left_out_index < n_samples):
            raise ValueError(f"Invalid left_out_index: {left_out_index} (must be in [0, {n_samples - 1}])")

        if use_composites:
            left_out_label = split_labels[left_out_index]
            test_idx = np.where(split_labels == left_out_label)[0]
        else:
            test_idx = np.array([left_out_index])

        all_indices = np.arange(len(self.data))
        train_idx = np.setdiff1d(all_indices, test_idx)


        # Step 3: data preparation
        X_train, X_test = self.data[train_idx], self.data[test_idx]
        X_train_proc, X_test_proc, _, _ = self.preprocess_data(
            X_train, X_test, normalize, scaler_type, use_pca=None, vthresh=None
        )

        if self.class_by_year:
            y_train = self.year_labels[train_idx]
            y_test = self.year_labels[test_idx]
        else:
            processed_labels = np.array(self.strategy.extract_labels(self.labels))
            y_train = processed_labels[train_idx]
            y_test = processed_labels[test_idx]

        # Step 4: model training
        self.classifier.fit(X_train_proc, y_train)
        y_pred = self.classifier.predict(X_test_proc)

        # === Handle decision_function or fallback ===
        if hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X_test_proc)
        elif hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X_test_proc)
            # Convert probabilities to "score-like" format by using log-odds or raw probabilities
            scores = proba  # shape (n_samples, n_classes)
        else:
            # Fallback: create a dummy 2-class score (useful for single sample LOO eval)
            scores = np.zeros((len(y_pred), len(np.unique(y_train))))
            for i, pred in enumerate(y_pred):
                scores[i, np.where(np.unique(y_train) == pred)[0][0]] = 1.0

        # scores = self.classifier.decision_function(X_test_proc)

        # create compatible format for binary classification
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)  # Shape (1, 2)

        # Step 5: evaluation (not meaningful with 1 test point, but for completeness)
        result = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=custom_order)
        }

        if projection_source == "scores":
            raw_test_labels = self.sample_labels[test_idx]
            return result, scores, y_test, raw_test_labels
        else:
            return result, None, None, None


    def train_and_evaluate_balanced(
            self,
            normalize=False,
            scaler_type='standard',
            region=None,
            random_seed=42,
            test_size=0.2, LOOPC=True,
            projection_source=False
    ):
        # Step 1: label preprocessing
        labels_used = self.strategy.extract_labels(self.labels)
        use_composites = self.strategy.use_composite_labels(self.labels)
        custom_order = self.strategy.get_custom_order(labels_used, self.year_labels)

        # Step 2: split
        if LOOPC:
            # Use composite-aware grouping for duplicate-safe splitting
            # composite_labels = assign_bordeaux_label(self.labels_raw, vintage=False)
            composite_labels = self.strategy.get_split_labels(self.labels_raw, self.class_by_year)
            train_idx, test_idx = leave_one_sample_per_class_split(
                X=self.data,
                y=composite_labels,
                random_state=random_seed,
                is_composite_labels=use_composites
            )
        else:
            train_idx, test_idx = self.shuffle_split_without_splitting_duplicates(
                X=self.data,
                y=self.labels,
                test_size=test_size,
                random_state=random_seed,
                group_duplicates=use_composites,
                dataset_origins=self.dataset_origins)

        # Step 3: data preparation
        X_train, X_test = self.data[train_idx], self.data[test_idx]
        X_train_proc, X_test_proc, _, _ = self.preprocess_data(
            X_train, X_test, normalize, scaler_type, use_pca=None, vthresh=None
        )

        if self.class_by_year:
            y_train = self.year_labels[train_idx]
            y_test = self.year_labels[test_idx]
        else:
            processed_labels = np.array(self.strategy.extract_labels(self.labels))
            y_train = processed_labels[train_idx]
            y_test = processed_labels[test_idx]

        # Step 4: model training
        self.classifier.fit(X_train_proc, y_train)
        y_pred = self.classifier.predict(X_test_proc)

        # === Handle decision_function or fallback ===
        if hasattr(self.classifier, "decision_function"):
            scores = self.classifier.decision_function(X_test_proc)
        elif hasattr(self.classifier, "predict_proba"):
            proba = self.classifier.predict_proba(X_test_proc)
            # Convert probabilities to "score-like" format by using log-odds or raw probabilities
            scores = proba  # shape (n_samples, n_classes)
        else:
            # Fallback: create a dummy 2-class score (useful for single sample LOO eval)
            scores = np.zeros((len(y_pred), len(np.unique(y_train))))
            for i, pred in enumerate(y_pred):
                scores[i, np.where(np.unique(y_train) == pred)[0][0]] = 1.0
        # scores = self.classifier.decision_function(X_test_proc)

        # create compatible format for binary classification
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)  # Shape (n_samples, 2)

        # Step 5: evaluation
        result = {
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, y_pred, labels=custom_order)
        }

        if projection_source == "scores":
            raw_test_labels = self.sample_labels[test_idx]
            return result, scores, y_test, raw_test_labels
        else:
            return result, None, None, None
        # return result


    def train_and_evaluate_balanced_old(self, n_inner_repeats=50, random_seed=42,
                                    test_size=0.2, normalize=False, scaler_type='standard',
                                    use_pca=False, vthresh=0.97, region=None, print_results=True,
                                    n_jobs=-1, test_on_discarded=False, LOOPC=True, return_umap_data=False):
        from sklearn.model_selection import StratifiedShuffleSplit
        from collections import defaultdict

        # 1. Determine label type and grouping
        if region == "winery" or self.wine_kind == "press":
            labels_used = self.extract_category_labels(self.labels)
        else:
            labels_used = self.labels

        # 2. Compute class distribution for chance accuracy
        class_counts = Counter(
            self.year_labels if self.year_labels.size > 0 and np.any(self.year_labels != None) else labels_used)
        total_samples = sum(class_counts.values())
        class_probabilities = np.array([count / total_samples for count in class_counts.values()])
        chance_accuracy = np.sum(class_probabilities ** 2)

        # 3. Define label order for confusion matrix
        if self.wine_kind == 'bordeaux':
            custom_order = None
        elif self.wine_kind == "press":
            if self.year_labels.size > 0 and np.any(self.year_labels != None):
                custom_order = list(Counter(self.year_labels).keys())
            else:
                custom_order = ["A", "B", "C"]
        else:
            custom_order = utils.get_custom_order_for_pinot_noir_region(region)

        # 4. Outer train/test split
        use_groups = True if self.wine_kind == "press" else False

        if LOOPC:
            # icl = True if self.wine_kind == "press" and len(self.labels[0]) > 1 else False
            icl = True if self.wine_kind in ("press", "bordeaux") and len(self.labels[0]) > 1 else False
            train_idx, test_idx = leave_one_sample_per_class_split(
                self.data, self.labels, random_state=random_seed, is_composite_labels=icl)
        else:
            train_idx, test_idx = self.shuffle_split_without_splitting_duplicates(
                self.data, self.labels, test_size=test_size, random_state=random_seed, group_duplicates=use_groups)

        X_train, X_test = self.data[train_idx], self.data[test_idx]
        y_train_raw = self.year_labels if self.year_labels.size > 0 and np.any(
            self.year_labels != None) else self.labels
        y_train_full, y_test = y_train_raw[train_idx], y_train_raw[test_idx]

        # if region == "winery" or self.wine_kind == "press":
        if region == "winery" or self.wine_kind == "press" or  self.wine_kind == "bordeaux":
            y_train_full = self.extract_category_labels(y_train_full)
            y_test = self.extract_category_labels(y_test)

        # 5. Choose evaluation mode
        if test_on_discarded:
            try:
                X_train_proc, X_test_proc, _, _ = self.preprocess_data(
                    X_train, X_test, normalize, scaler_type, use_pca, vthresh)
                self.classifier.fit(X_train_proc, y_train_full)
                y_pred = self.classifier.predict(X_test_proc)
                scores = self.classifier.decision_function(X_test_proc)

                # create compatible format for binary classification
                if scores.ndim == 1:
                    scores = np.stack([-scores, scores], axis=1)  # Shape (n_samples, 2)

                # Evaluate
                acc = accuracy_score(y_test, y_pred)
                bal = balanced_accuracy_score(y_test, y_pred)
                w_acc = np.average(y_pred == y_test, weights=compute_sample_weight('balanced', y_test))
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                cm = confusion_matrix(y_test, y_pred, labels=custom_order)

                with np.errstate(invalid='ignore', divide='ignore'):
                    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
                    cm_norm[np.isnan(cm_norm)] = 0

                results = {
                    'chance_accuracy': chance_accuracy,
                    'overall_accuracy': acc,
                    'overall_balanced_accuracy': bal,
                    'overall_weighted_accuracy': w_acc,
                    'overall_precision': prec,
                    'overall_recall': rec,
                    'overall_f1_score': f1,
                    'overall_confusion_matrix': cm,
                    'overall_confusion_matrix_normalized': cm_norm,
                }

                if print_results:
                    print(f"  Test on held-out data:")
                    print(f"    Accuracy: {acc:.3f}")
                    print(f"    Balanced Accuracy: {bal:.3f}")
                    print(f"    Weighted Accuracy: {w_acc:.3f}")
                    print(f"    Precision: {prec:.3f}")
                    print(f"    Recall: {rec:.3f}")
                    print(f"    F1 Score: {f1:.3f}")

                if return_umap_data:
                    return results, scores, y_test
                else:
                    return results, None, None

            except np.linalg.LinAlgError:
                print("⚠️ Skipping due to SVD convergence error.")
                return {k: np.nan for k in [
                    'overall_accuracy', 'overall_balanced_accuracy', 'overall_weighted_accuracy',
                    'overall_precision', 'overall_recall', 'overall_f1_score', 'confusion_matrix']}

        # 6. Cross-validation mode
        else:
            # Build custom CV
            cv = self.RepeatedLeaveOneSamplePerClassCV(
                n_repeats=n_inner_repeats, shuffle=True,
                random_state=random_seed, use_groups=use_groups)

            results = self.run_cross_validation(
                cv=cv,
                X=X_train,
                y=np.array(y_train_full),
                clf=self.classifier,
                normalize=normalize,
                scaler_type=scaler_type,
                use_pca=use_pca,
                vthresh=vthresh,
                custom_order=custom_order,
                n_jobs=n_jobs
            )

            summary = self.summarize_cv_results(results)
            summary['chance_accuracy'] = chance_accuracy

            if print_results:
                print("\nCross-validation Results:")
                print(f"  Accuracy: {summary['overall_accuracy']:.3f}")
                print(f"  Balanced Accuracy: {summary['overall_balanced_accuracy']:.3f}")
                print(f"  Weighted Accuracy: {summary['overall_weighted_accuracy']:.3f}")
                print(f"  Precision: {summary['overall_precision']:.3f}")
                print(f"  Recall: {summary['overall_recall']:.3f}")
                print(f"  F1 Score: {summary['overall_f1_score']:.3f}")

            return summary


    # def train_and_evaluate_balanced(self, n_inner_repeats=50, random_seed=42,
    #                                 test_size=0.2, normalize=False, scaler_type='standard',
    #                                 use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
    #                                 test_on_discarded=False, LOOPC=True):
    #     """
    #     Train and evaluate the classifier using a train/test split followed by cross-validation on the training set.
    #     Evaluation metrics are averaged across multiple inner cross-validation repetitions.
    #
    #     Depending on the value of `test_on_discarded`, either the inner validation set or the held-out test set
    #     is used to compute final metrics. Supports normalization, PCA, and custom confusion matrix ordering.
    #
    #     Parameters
    #     ----------
    #     n_inner_repeats : int, optional (default=50)
    #         Number of inner cross-validation folds or repetitions.
    #     random_seed : int, optional (default=42)
    #         Seed for reproducibility.
    #     test_size : float, optional (default=0.2)
    #         Fraction of data to hold out as a test set (only used when test_on_discarded=True).
    #     normalize : bool, optional (default=False)
    #         Whether to apply feature normalization (e.g., standard scaling).
    #     scaler_type : str, optional (default='standard')
    #         Type of scaler to use if normalization is enabled ('standard', 'minmax', etc.).
    #     use_pca : bool, optional (default=False)
    #         Whether to apply PCA for dimensionality reduction.
    #     vthresh : float, optional (default=0.97)
    #         Proportion of variance to retain when applying PCA.
    #     region : str or None, optional (default=None)
    #         Custom region ordering for confusion matrix labels.
    #     print_results : bool, optional (default=True)
    #         Whether to print performance metrics to the console.
    #     n_jobs : int, optional (default=-1)
    #         Number of parallel jobs to use for cross-validation.
    #     test_on_discarded : bool, optional (default=False)
    #         If True, final evaluation is done on the held-out test set (outside CV).
    #         If False, metrics are computed on the inner cross-validation folds.
    #     LOOPC : bool, optional (default=True)
    #         If True, use Leave-One-Out-Per-Class strategy for cross-validation (i.e., select one full sample per class).
    #         If False, use standard cross-validation with grouped or stratified shuffle splits.
    #
    #     Returns
    #     -------
    #     dict
    #         Dictionary containing average accuracy, precision, recall, F1-score, and normalized confusion matrix
    #         across all evaluation folds or test splits.
    #     """
    #     from sklearn.model_selection import StratifiedShuffleSplit
    #     from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    #     from sklearn.decomposition import PCA
    #     from sklearn.utils.class_weight import compute_sample_weight
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
    #     def leave_one_sample_per_class_split(X, y, random_state=None, is_composite_labels=True):
    #         """
    #         Split dataset by selecting one sample per class (plus its duplicates if composite labels) for the test set.
    #
    #         Parameters
    #         ----------
    #         X : array-like
    #             Feature matrix (not used, kept for compatibility).
    #         y : array-like
    #             Labels (composite labels or simple class labels).
    #         random_state : int, optional
    #             Random seed for reproducibility.
    #         is_composite_labels : bool, optional
    #             Whether y are composite labels like 'A1', 'B2'. If False, assumes simple class labels.
    #
    #         Returns
    #         -------
    #         train_indices : np.ndarray
    #             Indices for training set.
    #         test_indices : np.ndarray
    #             Indices for test set.
    #         """
    #         import numpy as np
    #         from collections import defaultdict
    #         import re
    #
    #         rng = np.random.default_rng(random_state)
    #
    #         if is_composite_labels:
    #             # Group indices by full sample labels (for composite cases)
    #             sample_to_indices = defaultdict(list)
    #             for idx, label in enumerate(y):
    #                 sample_to_indices[label].append(idx)
    #
    #             # Group sample labels by class (A, B, C)
    #             class_to_samples = defaultdict(list)
    #
    #             def get_class_from_label(label):
    #                 """Extract the class (first letter) from composite label like 'A1'."""
    #                 match = re.match(r'([A-Z])', label)
    #                 return match.group(1) if match else label
    #
    #             for label in sample_to_indices.keys():
    #                 class_label = get_class_from_label(label)
    #                 class_to_samples[class_label].append(label)
    #
    #             # Choose one sample label per class
    #             test_sample_labels = []
    #             for class_label, samples in class_to_samples.items():
    #                 chosen_sample = rng.choice(samples)
    #                 test_sample_labels.append(chosen_sample)
    #
    #             # Expand into indices
    #             test_indices = []
    #             for label in test_sample_labels:
    #                 test_indices.extend(sample_to_indices[label])
    #
    #         else:
    #             # Simple class labels (e.g., 'Beaune', 'Alsace', 'C', 'D')
    #             class_to_indices = defaultdict(list)
    #             for idx, label in enumerate(y):
    #                 class_to_indices[label].append(idx)
    #
    #             # Pick one random index per class
    #             test_indices = []
    #             for class_label, indices in class_to_indices.items():
    #                 chosen_idx = rng.choice(indices)
    #                 test_indices.append(chosen_idx)
    #
    #         test_indices = np.array(test_indices)
    #         all_indices = np.arange(len(y))
    #         train_indices = np.setdiff1d(all_indices, test_indices)
    #
    #         return train_indices, test_indices
    #
    #     def shuffle_split_without_splitting_duplicates(X, y, test_size=0.2, random_state=None, group_duplicates=True):
    #         """
    #             Perform ShuffleSplit on samples while ensuring that:
    #             - Duplicates of the same sample are kept together (if enabled).
    #             - Each class is represented in the test set.
    #
    #         Args:
    #             X (array-like): Feature matrix or sample labels.
    #             y (array-like): Composite labels (e.g., ['A1', 'A1', 'B2', 'B2']).
    #             test_size (float): Fraction of samples to include in the test set.
    #             random_state (int): Random seed for reproducibility.
    #             group_duplicates (bool): If True, duplicates of the same sample are kept together.
    #                                      If False, duplicates are treated independently.
    #
    #         Returns:
    #             tuple: train_indices, test_indices (numpy arrays)
    #         """
    #         import numpy as np
    #
    #         rng = np.random.default_rng(random_state)
    #
    #         if group_duplicates:
    #             unique_samples = {}  # {sample_label: [indices]}
    #             class_samples = defaultdict(list)  # {class_label: [sample_label1, sample_label2, ...]}
    #
    #             for idx, label in enumerate(y):
    #                 unique_samples.setdefault(label, []).append(idx)
    #                 class_samples[label[0]].append(label)  # Assuming class is the first character (e.g., 'A1' -> 'A')
    #
    #             sample_labels = list(unique_samples.keys())
    #             rng.shuffle(sample_labels)
    #
    #             # Step 1: Randomly select test samples
    #             num_test_samples = int(len(sample_labels) * test_size)
    #             test_sample_labels = set(sample_labels[:num_test_samples])
    #
    #             # Step 2: Ensure each class is represented in the test set
    #             test_classes = {label[0] for label in test_sample_labels}
    #             missing_classes = [c for c in class_samples if c not in test_classes]
    #
    #             # Step 3: Force at least one sample from missing classes
    #             for class_label in missing_classes:
    #                 additional_sample = rng.choice(class_samples[class_label])
    #                 test_sample_labels.add(additional_sample)
    #
    #             # Step 4: Convert sample labels to index lists
    #             test_indices = [idx for label in test_sample_labels for idx in unique_samples[label]]
    #             train_indices = [idx for label in sample_labels if label not in test_sample_labels for idx in
    #                              unique_samples[label]]
    #
    #         else:
    #             # Treat each instance independently
    #             indices = np.arange(len(y))
    #             rng.shuffle(indices)
    #
    #             # Calculate the number of test samples
    #             num_test_samples = int(len(indices) * test_size)
    #
    #             # Split into train and test sets
    #             test_indices = indices[:num_test_samples]
    #             train_indices = indices[num_test_samples:]
    #
    #         return np.array(train_indices), np.array(test_indices)
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
    #     def process_fold(inner_train_idx, inner_val_idx, X_train_full, y_train_full, normalize, scaler_type, use_pca,
    #                      vthresh, custom_order):
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
    #     if region == "winery" or self.wine_kind == "press":
    #         category_labels = extract_category_labels(self.labels)
    #     else:
    #         category_labels = self.labels
    #
    #     # Compute class distribution
    #     if self.year_labels.size > 0 and np.any(self.year_labels != None):
    #         class_counts = Counter(self.year_labels)
    #     else:
    #         class_counts = Counter(category_labels)
    #     # class_counts = Counter(category_labels)
    #
    #     total_samples = sum(class_counts.values())
    #
    #     # Compute Correct Chance Accuracy (Taking Class Distribution into Account)
    #     class_probabilities = np.array([count / total_samples for count in class_counts.values()])
    #     chance_accuracy = np.sum(class_probabilities ** 2)  # Sum of squared class probabilities
    #
    #
    #     # Set up a custom order for the confusion matrix if a region is specified.
    #     if self.wine_kind == "press":
    #         if self.year_labels.size > 0 and np.any(self.year_labels != None):
    #             year_count = Counter(self.year_labels)
    #             custom_order = list(year_count.keys())
    #         else:
    #             custom_order = ["A", "B", "C"]
    #     else:
    #         custom_order = utils.get_custom_order_for_pinot_noir_region(region)
    #
    #     # Initialize accumulators for outer-repetition averaged metrics.
    #     eval_accuracy = []
    #     eval_balanced_accuracy = []
    #     eval_weighted_accuracy = []
    #     eval_precision = []
    #     eval_recall = []
    #     eval_f1 = []
    #     eval_cm = []
    #
    #     # Use a reproducible RNG.
    #     if random_seed is None:
    #         random_seed = np.random.randint(0, int(1e6))
    #     rng = np.random.default_rng(random_seed)
    #
    #
    #     # Choose whether to use groups based on wine type.
    #     use_groups = True if self.wine_kind == "press" else False
    #     # use_groups = False
    #
    #     # Outer split: use StratifiedShuffleSplit to split the data.
    #     # # sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.integers(0, int(1e6)))
    #     # sss = ShuffleSplit(n_splits=1, test_size=test_size, random_state=rng.integers(0, int(1e6)))
    #     # train_idx, _ = next(sss.split(self.data, self.labels))
    #
    #     # train_idx, test_idx = shuffle_split_without_splitting_duplicates(
    #     #     self.data, self.labels, test_size=test_size, random_state=rng.integers(0, int(1e6)),
    #     #     group_duplicates=use_groups
    #     # )
    #     if LOOPC:
    #         icl = True if self.wine_kind == "press" and len(self.labels[0]) > 1 else False
    #         train_idx, test_idx = leave_one_sample_per_class_split(self.data, self.labels, random_state=random_seed,
    #                                                                is_composite_labels=icl)
    #     else:
    #         train_idx, test_idx = shuffle_split_without_splitting_duplicates(
    #             self.data, self.labels, test_size=test_size, random_state=random_seed,
    #             group_duplicates=use_groups
    #         )
    #
    #     X_train_full, X_test = self.data[train_idx], self.data[test_idx]
    #     if self.year_labels.size > 0 and np.any(self.year_labels != None):
    #         y_train_full, y_test = self.year_labels[train_idx], self.year_labels[test_idx]
    #     else:
    #         y_train_full, y_test = self.labels[train_idx], self.labels[test_idx]
    #
    #     # y_train_full, y_test = self.labels[train_idx], self.labels[test_idx]
    #
    #     if test_on_discarded:
    #         if normalize:
    #             X_train_full, scaler = normalize_data(X_train_full, scaler=scaler_type)
    #             X_test = scaler.transform(X_test)
    #
    #         if use_pca:
    #             pca = PCA(n_components=None, svd_solver='randomized')
    #             pca.fit(X_train_full)
    #             X_train_full = pca.transform(X_train_full)
    #             X_test = pca.transform(X_test)
    #
    #         # if self.year_labels.size > 0 and np.any(self.year_labels != None):
    #         #     self.classifier.fit(X_train_full, y_train_full)
    #         # else:
    #         #     self.classifier.fit(X_train_full, np.array(extract_category_labels(y_train_full)))
    #         #     y_test = extract_category_labels(y_test)
    #
    #         try:
    #             if self.year_labels.size > 0 and np.any(self.year_labels != None):
    #                 self.classifier.fit(X_train_full, y_train_full)
    #             else:
    #                 if region == "winery" or self.wine_kind == "press":
    #                     self.classifier.fit(X_train_full, np.array(extract_category_labels(y_train_full)))
    #                     y_test = extract_category_labels(y_test)
    #                 else:
    #                     self.classifier.fit(X_train_full, np.array(y_train_full))
    #
    #
    #         except np.linalg.LinAlgError:
    #             print(
    #                 "⚠️ Skipping evaluation due to SVD convergence error (likely caused by LDA on low-variance or singular data).")
    #             return {
    #                 'overall_accuracy': np.nan,
    #                 'overall_balanced_accuracy': np.nan,
    #                 'overall_weighted_accuracy': np.nan,
    #                 'overall_precision': np.nan,
    #                 'overall_recall': np.nan,
    #                 'overall_f1_score': np.nan,
    #                 'confusion_matrix': None,
    #             }
    #         # self.classifier.fit(X_train_full, extract_category_labels(y_train_full))
    #         # y_test = extract_category_labels(y_test)
    #
    #         y_pred = self.classifier.predict(X_test)
    #
    #
    #         eval_accuracy.append(accuracy_score(y_test, y_pred))
    #         eval_balanced_accuracy.append(balanced_accuracy_score(y_test, y_pred))
    #         eval_weighted_accuracy.append(
    #             np.average(y_pred == y_test, weights=compute_sample_weight('balanced', y_test)))
    #         eval_precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
    #         eval_recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
    #         eval_f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
    #         eval_cm.append(confusion_matrix(y_test, y_pred, labels=custom_order))
    #
    #         if print_results:
    #             print(f"  Test on discarded data metrics:")
    #             print(f"    Accuracy: {eval_accuracy[-1]:.3f}")
    #             print(f"    Balanced Accuracy: {eval_balanced_accuracy[-1]:.3f}")
    #             print(f"    Weighted Accuracy: {eval_weighted_accuracy[-1]:.3f}")
    #             print(f"    Precision: {eval_precision[-1]:.3f}")
    #             print(f"    Recall: {eval_recall[-1]:.3f}")
    #             print(f"    F1 Score: {eval_f1[-1]:.3f}")
    #
    #     else:
    #         # cv = RepeatedLeaveOneFromEachClassCV(n_repeats=n_inner_repeats, shuffle=True, random_state=random_seed)
    #         cv = RepeatedLeaveOneSamplePerClassCV(
    #             n_repeats=n_inner_repeats,
    #             shuffle=True,
    #             random_state=random_seed,
    #             use_groups=use_groups
    #         )
    #
    #         results = Parallel(n_jobs=n_jobs, backend='loky')(
    #             delayed(process_fold)(inner_train_idx, inner_val_idx, X_train_full, y_train_full, normalize,
    #                                   scaler_type, use_pca, vthresh, custom_order)
    #             for inner_train_idx, inner_val_idx in cv.split(X_train_full, y_train_full)
    #         )
    #
    #         inner_acc, inner_bal_acc, inner_w_acc, inner_prec, inner_rec, inner_f1, inner_cm = zip(*results)
    #
    #         eval_accuracy.append(np.mean(inner_acc))
    #         eval_balanced_accuracy.append(np.mean(inner_bal_acc))
    #         eval_weighted_accuracy.append(np.mean(inner_w_acc))
    #         eval_precision.append(np.mean(inner_prec))
    #         eval_recall.append(np.mean(inner_rec))
    #         eval_f1.append(np.mean(inner_f1))
    #         eval_cm.append(np.mean(inner_cm, axis=0))
    #
    #         if print_results:
    #             print(f"  Inner CV Averages:")
    #             print(f"    Accuracy: {eval_accuracy[-1]:.3f}")
    #             print(f"    Balanced Accuracy: {eval_balanced_accuracy[-1]:.3f}")
    #             print(f"    Weighted Accuracy: {eval_weighted_accuracy[-1]:.3f}")
    #             print(f"    Precision: {eval_precision[-1]:.3f}")
    #             print(f"    Recall: {eval_recall[-1]:.3f}")
    #             print(f"    F1 Score: {eval_f1[-1]:.3f}")
    #
    #     # Compute the averaged confusion matrix across all repetitions
    #     overall_cm = np.mean(eval_cm, axis=0)
    #
    #     # Normalize confusion matrix row-wise (true label-wise)
    #     with np.errstate(invalid='ignore', divide='ignore'):
    #         overall_cm_normalized = overall_cm.astype('float') / overall_cm.sum(axis=1, keepdims=True)
    #         overall_cm_normalized[np.isnan(overall_cm_normalized)] = 0
    #
    #     overall_results = {
    #         'chance_accuracy': chance_accuracy,
    #         'overall_accuracy': np.mean(eval_accuracy),
    #         'overall_balanced_accuracy': np.mean(eval_balanced_accuracy),
    #         'overall_weighted_accuracy': np.mean(eval_weighted_accuracy),
    #         'overall_precision': np.mean(eval_precision),
    #         'overall_recall': np.mean(eval_recall),
    #         'overall_f1_score': np.mean(eval_f1),
    #         'overall_confusion_matrix': overall_cm,
    #         'overall_confusion_matrix_normalized': overall_cm_normalized,
    #     }
    #
    #     if print_results:
    #         print("\nFinal Results:")
    #         print(f"  Overall Accuracy: {overall_results['overall_accuracy']:.3f}")
    #         print(f"  Overall Balanced Accuracy: {overall_results['overall_balanced_accuracy']:.3f}")
    #         print(f"  Overall Weighted Accuracy: {overall_results['overall_weighted_accuracy']:.3f}")
    #         print(f"  Overall Precision: {overall_results['overall_precision']:.3f}")
    #         print(f"  Overall Recall: {overall_results['overall_recall']:.3f}")
    #         print(f"  Overall F1 Score: {overall_results['overall_f1_score']:.3f}")
    #         # print("Overall Mean Confusion Matrix:")
    #         # print(overall_results['overall_confusion_matrix'])
    #         # print(overall_results['overall_confusion_matrix_normalized'])
    #
    #     return overall_results



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

    def evaluate_feature_matrix_LOO(feature_matrix, labels, year_labels, strategy,
                                    classifier_type, normalize, scaler_type,
                                    projection_source, show_confusion_matrix=False,
                                    sample_labels=None, dataset_origins=None,
                                    labels_raw=None, class_by_year=False):
        """
        Perform Leave-One-Out evaluation for a given feature matrix.

        Returns:
            mean_acc, std_acc, all_scores, all_labels, test_sample_names, mean_confusion_matrix
        """
        from collections import Counter
        import numpy as np
        from gcmswine import utils
        from sklearn.metrics import ConfusionMatrixDisplay

        num_samples = feature_matrix.shape[0]
        all_scores, all_labels, test_sample_names = [], [], []
        confusion_matrices, accuracies = [], []

        for idx in range(num_samples):
            try:
                cls = Classifier(
                    data=feature_matrix,
                    labels=labels,
                    classifier_type=classifier_type,
                    year_labels=year_labels,
                    dataset_origins=dataset_origins,
                    strategy=strategy,
                    class_by_year=class_by_year,
                    labels_raw=labels_raw,
                    sample_labels=sample_labels
                )

                results, scores, y_test, raw_test_labels = cls.train_and_evaluate_leave_one_out(
                    left_out_index=idx,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    projection_source=projection_source,
                )

                if 'accuracy' in results:
                    accuracies.append(results['accuracy'])
                else:
                    print(f"⚠️ Accuracy missing for sample {idx}")

                if scores is not None:
                    all_scores.append(scores[0])
                    all_labels.append(y_test[0])
                    test_sample_names.append(raw_test_labels[0])

                if 'confusion_matrix' in results:
                    confusion_matrices.append(results['confusion_matrix'])

            except Exception as e:
                print(f"⚠️ Skipping sample {idx} due to error: {e}")

        accuracies = np.array(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        if projection_source == "scores" and all_scores:
            all_scores = np.vstack(all_scores)
            all_labels = np.array(all_labels)
            test_sample_names = np.array(test_sample_names)
        else:
            all_scores, all_labels, test_sample_names = None, None, np.array(
                list(sample_labels) if sample_labels is not None else [])

        mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)

        # Optional visualization
        if show_confusion_matrix:
            labels_used = year_labels if class_by_year else strategy.extract_labels(labels)
            custom_order = strategy.get_custom_order(labels_used, year_labels)
            fig, ax = plt.subplots(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix,
                                          display_labels=custom_order)
            disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
            plt.tight_layout()
            plt.show()

        return mean_acc, std_acc, all_scores, all_labels, test_sample_names, mean_confusion_matrix


    def train_and_evaluate_leave_one_out_all_samples(
            self,
            normalize=False,
            scaler_type='standard',
            region=None,
            projection_source=False,
            classifier_type="RGC",
            feature_type="concatenated",
            show_confusion_matrix=False,
    ):
        from collections import Counter
        from gcmswine import utils
        import numpy as np
        from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
        from sklearn.metrics import confusion_matrix

        # Setup data and strategy
        cls_data = self.data.copy()
        labels = np.array(self.labels)
        year_labels = np.array(self.year_labels) if hasattr(self, 'year_labels') else np.array([])
        num_samples, num_timepoints, num_channels = cls_data.shape

        def compute_features(channels):
            if feature_type == "concat_channels":
                return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            elif feature_type == "tic":
                return np.sum(cls_data[:, :, channels], axis=2)
            elif feature_type == "tis":
                return np.sum(cls_data[:, :, channels], axis=1)
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            elif feature_type in ["best_channel", "greedy_add"]:
                # Placeholder: handled later
                return None
            else:
                raise ValueError(
                    "Invalid feature_type. Use 'concatenated', 'tic', 'tis', 'tic_tis', or 'best_channel'.")

        def evaluate_feature_matrix_LOO(feature_matrix, labels, year_labels, strategy,
                                        classifier_type, normalize, scaler_type,
                                        projection_source, show_confusion_matrix=False,
                                        sample_labels=None, dataset_origins=None,
                                        labels_raw=None, class_by_year=False):
            """
            Perform Leave-One-Out evaluation for a given feature matrix.

            Returns:
                mean_acc, std_acc, all_scores, all_labels, test_sample_names, mean_confusion_matrix
            """
            from collections import Counter
            import numpy as np
            from gcmswine import utils
            from sklearn.metrics import ConfusionMatrixDisplay

            num_samples = feature_matrix.shape[0]
            all_scores, all_labels, test_sample_names = [], [], []
            confusion_matrices, accuracies = [], []

            for idx in range(num_samples):
                try:
                    cls = Classifier(
                        data=feature_matrix,
                        labels=labels,
                        classifier_type=classifier_type,
                        year_labels=year_labels,
                        dataset_origins=dataset_origins,
                        strategy=strategy,
                        class_by_year=class_by_year,
                        labels_raw=labels_raw,
                        sample_labels=sample_labels
                    )

                    results, scores, y_test, raw_test_labels = cls.train_and_evaluate_leave_one_out(
                        left_out_index=idx,
                        normalize=normalize,
                        scaler_type=scaler_type,
                        projection_source=projection_source,
                    )

                    if 'accuracy' in results:
                        accuracies.append(results['accuracy'])
                    else:
                        print(f"⚠️ Accuracy missing for sample {idx}")

                    if scores is not None:
                        all_scores.append(scores[0])
                        all_labels.append(y_test[0])
                        test_sample_names.append(raw_test_labels[0])

                    if 'confusion_matrix' in results:
                        confusion_matrices.append(results['confusion_matrix'])

                except Exception as e:
                    print(f"⚠️ Skipping sample {idx} due to error: {e}")

            accuracies = np.array(accuracies)
            mean_acc = np.mean(accuracies)
            std_acc = np.std(accuracies)

            if projection_source == "scores" and all_scores:
                all_scores = np.vstack(all_scores)
                all_labels = np.array(all_labels)
                test_sample_names = np.array(test_sample_names)
            else:
                all_scores, all_labels, test_sample_names = None, None, np.array(
                    list(sample_labels) if sample_labels is not None else [])

            mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)

            # Optional visualization
            if show_confusion_matrix:
                labels_used = year_labels if class_by_year else strategy.extract_labels(labels)
                custom_order = strategy.get_custom_order(labels_used, year_labels)
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix,
                                              display_labels=custom_order)
                disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
                plt.tight_layout()
                plt.show()

            return mean_acc, std_acc, all_scores, all_labels, test_sample_names, mean_confusion_matrix


        # === Handle normal multi-channel feature computation ===
        if feature_type != "best_channel":
            feature_matrix = compute_features(list(range(num_channels)))
        else:
            feature_matrix = None  # Computed per-channel later

        strategy = get_strategy_by_wine_kind(
            self.wine_kind, region,
            utils.get_custom_order_for_pinot_noir_region,
            class_by_year=self.class_by_year
        )

        # === BEST CHANNEL MODE ===
        if feature_type == "best_channel":
            best_channel_results = []

            for ch in range(num_channels):
                print(f"\n=== Evaluating channel {ch} / {num_channels - 1} ===")
                feature_matrix = cls_data[:, :, ch]  # Single-channel TIC trace
                feature_matrix = feature_matrix.reshape(num_samples, -1)

                mean_acc, std_acc, scores, lbls, names, _ = evaluate_feature_matrix_LOO(
                    feature_matrix, labels, year_labels, strategy,
                    classifier_type, normalize, scaler_type,
                    projection_source, show_confusion_matrix=False,
                    sample_labels=self.sample_labels,
                    dataset_origins=getattr(self, 'dataset_origins', None),
                    labels_raw=self.labels_raw,
                    class_by_year=self.class_by_year
                )
                best_channel_results.append((ch, mean_acc, std_acc, scores, lbls, names))

                # all_scores, all_labels, test_sample_names = [], [], []
                # confusion_matrices, accuracies = [], []
                #
                # # LOO evaluation for this channel
                # for idx in range(num_samples):
                #     try:
                #         cls = Classifier(
                #             data=feature_matrix,
                #             labels=labels,
                #             classifier_type=classifier_type,
                #             year_labels=year_labels,
                #             dataset_origins=self.dataset_origins if hasattr(self, 'dataset_origins') else None,
                #             strategy=strategy,
                #             class_by_year=self.class_by_year,
                #             labels_raw=self.labels_raw,
                #             sample_labels=self.sample_labels
                #         )
                #
                #         results, scores, y_test, raw_test_labels = cls.train_and_evaluate_leave_one_out(
                #             left_out_index=idx,
                #             normalize=normalize,
                #             scaler_type=scaler_type,
                #             projection_source=projection_source,
                #         )
                #
                #         if 'accuracy' in results:
                #             accuracies.append(results['accuracy'])
                #         else:
                #             print(f"⚠️ Accuracy missing for sample {idx}")
                #
                #         if scores is not None:
                #             all_scores.append(scores[0])
                #             all_labels.append(y_test[0])
                #             test_sample_names.append(raw_test_labels[0])
                #
                #         if 'confusion_matrix' in results:
                #             confusion_matrices.append(results['confusion_matrix'])
                #
                #     except Exception as e:
                #         print(f"⚠️ Skipping sample {idx} due to error: {e}")
                #
                # accuracies = np.array(accuracies)
                # mean_acc = np.mean(accuracies)
                # std_acc = np.std(accuracies)
                #
                # if projection_source == "scores" and all_scores:
                #     all_scores = np.vstack(all_scores)
                #     all_labels = np.array(all_labels)
                #     test_sample_names = np.array(test_sample_names)
                # else:
                #     all_scores, all_labels, test_sample_names = None, None, np.array(list(self.sample_labels))
                #
                # mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)
                #
                # # === Print summary for this channel ===
                # print("##################################")
                # print(f"Channel {ch}: Leave-One-Out Evaluation Finished")
                # print("##################################")
                # print(f"Mean Accuracy (LOO): {mean_acc:.3f} ± {std_acc:.3f}")
                #
                # if self.class_by_year:
                #     labels_used = self.year_labels
                # else:
                #     labels_used = strategy.extract_labels(labels)
                #
                # custom_order = strategy.get_custom_order(labels_used, year_labels)
                # counts = Counter(labels_used)
                #
                # print("\nLabel order:")
                # if custom_order is not None:
                #     for label in custom_order:
                #         print(f"{label} ({counts.get(label, 0)})")
                # else:
                #     for label in sorted(counts.keys()):
                #         print(f"{label} ({counts[label]})")
                #
                # print("\nFinal Averaged Normalized Confusion Matrix:")
                # print(mean_confusion_matrix)
                # print("##################################")
                #
                # if show_confusion_matrix:
                #     import matplotlib.pyplot as plt
                #     from sklearn.metrics import ConfusionMatrixDisplay
                #
                #     fig, ax = plt.subplots(figsize=(8, 6))
                #     ax.set_xlabel('Predicted Label', fontsize=14)
                #     ax.set_ylabel('True Label', fontsize=14)
                #     ax.set_title(f'Confusion matrix (Channel {ch})')
                #     disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=custom_order)
                #     disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)
                #     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
                #     plt.tight_layout()
                #     plt.show()
                #
                # best_channel_results.append((ch, mean_acc, std_acc, all_scores, all_labels, test_sample_names))

            # Pick best channel
            best_channel_results.sort(key=lambda x: x[1], reverse=True)
            best_channel_idx, best_mean_acc, best_std_acc, best_scores, best_labels, best_names = best_channel_results[0]

            print("\n########## BEST CHANNEL ##########")
            print(f"Best Channel: {best_channel_idx} | Accuracy: {best_mean_acc:.3f} ± {best_std_acc:.3f}")

            return best_mean_acc, best_std_acc, best_scores, best_labels, best_names
            # return best_channel_results

        if feature_type == "greedy_add":
            print("\n=== Ranking channels individually ===")
            channel_accs = []
            for ch in range(num_channels):
                fm = cls_data[:, :, ch].reshape(num_samples, -1)
                mean_acc, *_ = evaluate_feature_matrix_LOO(
                    fm, labels, year_labels, strategy,
                    classifier_type, normalize, scaler_type,
                    projection_source, show_confusion_matrix=False,
                    sample_labels=self.sample_labels,
                    dataset_origins=getattr(self, 'dataset_origins', None),
                    labels_raw=self.labels_raw,
                    class_by_year=self.class_by_year
                )
                channel_accs.append((ch, mean_acc))

            # Rank by accuracy
            ranked_channels = [ch for ch, _ in sorted(channel_accs, key=lambda x: x[1], reverse=True)]

            print("\n=== Greedy channel addition ===")
            cumulative_results = []
            selected_channels = []
            for k, ch in enumerate(ranked_channels, 1):
                selected_channels.append(ch)
                print(f"\nEvaluating with top {k} channels: {selected_channels}")
                # fm = np.hstack([cls_data[:, :, c].reshape(num_samples, -1) for c in selected_channels])
                fm = np.sum(cls_data[:, :, selected_channels], axis=2)  # Summation across selected channels

                mean_acc, std_acc, *_ = evaluate_feature_matrix_LOO(
                    fm, labels, year_labels, strategy,
                    classifier_type, normalize, scaler_type,
                    projection_source, show_confusion_matrix=False,
                    sample_labels=self.sample_labels,
                    dataset_origins=getattr(self, 'dataset_origins', None),
                    labels_raw=self.labels_raw,
                    class_by_year=self.class_by_year
                )
                cumulative_results.append((k, selected_channels.copy(), mean_acc, std_acc))
                print(f"Top {k} channels: Accuracy {mean_acc:.3f} ± {std_acc:.3f}")
                # === Plot accuracy vs. number of channels ===

            import matplotlib.pyplot as plt
            ks = [r[0] for r in cumulative_results]
            accs = [r[2] for r in cumulative_results]
            stds = [r[3] for r in cumulative_results]

            plt.figure(figsize=(8, 5))
            plt.plot(ks, accs, '-o', label='Accuracy')
            plt.xlabel("Number of Channels Added", fontsize=14)
            plt.ylabel("Accuracy (LOO)", fontsize=14)
            plt.title("Greedy Channel Addition: Accuracy vs Channels", fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

            best_idx = np.argmax(accs)
            best_mean_acc, best_std_acc = accs[best_idx], stds[best_idx]
            print(f"\nBest accuracy: {best_mean_acc:.3f} ± {best_std_acc:.3f} using top {ks[best_idx]} channels.")

            return best_mean_acc, best_std_acc, None, None, np.array(list(self.sample_labels)), cumulative_results

        # === STANDARD MULTI-CHANNEL FLOW (unchanged) ===
        all_scores, all_labels, test_sample_names = [], [], []
        confusion_matrices, accuracies = [], []

        for idx in range(num_samples):
            print(f'{idx} ', end="")
            try:
                cls = Classifier(
                    data=feature_matrix,
                    labels=labels,
                    classifier_type=classifier_type,
                    year_labels=year_labels,
                    dataset_origins=self.dataset_origins if hasattr(self, 'dataset_origins') else None,
                    strategy=strategy,
                    class_by_year=self.class_by_year,
                    labels_raw=self.labels_raw,
                    sample_labels=self.sample_labels
                )

                results, scores, y_test, raw_test_labels = cls.train_and_evaluate_leave_one_out(
                    left_out_index=idx,
                    normalize=normalize,
                    scaler_type=scaler_type,
                    projection_source=projection_source,
                )

                if 'accuracy' in results:
                    accuracies.append(results['accuracy'])
                else:
                    print(f"⚠️ Accuracy missing for sample {idx}")

                if scores is not None:
                    all_scores.append(scores[0])
                    all_labels.append(y_test[0])
                    test_sample_names.append(raw_test_labels[0])

                if 'confusion_matrix' in results:
                    confusion_matrices.append(results['confusion_matrix'])

            except Exception as e:
                print(f"⚠️ Skipping sample {idx} due to error: {e}")

        accuracies = np.array(accuracies)
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)

        if projection_source == "scores" and all_scores:
            all_scores = np.vstack(all_scores)
            all_labels = np.array(all_labels)
            test_sample_names = np.array(test_sample_names)
        else:
            all_scores, all_labels, test_sample_names = None, None, np.array(list(self.sample_labels))

        mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)

        # === Standard summary printing ===
        print("##################################")
        print("Leave-One-Out Evaluation Finished")
        print("##################################")
        print(f"Mean Accuracy (LOO): {mean_acc:.3f} ± {std_acc:.3f}")

        if self.class_by_year:
            labels_used = self.year_labels
        else:
            labels_used = strategy.extract_labels(labels)

        custom_order = strategy.get_custom_order(labels_used, year_labels)
        counts = Counter(labels_used)

        print("\nLabel order:")
        if custom_order is not None:
            for label in custom_order:
                print(f"{label} ({counts.get(label, 0)})")
        else:
            for label in sorted(counts.keys()):
                print(f"{label} ({counts[label]})")

        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        if show_confusion_matrix:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel('Predicted Label', fontsize=14)
            ax.set_ylabel('True Label', fontsize=14)
            ax.set_title(f'Confusion matrix by region')
            disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=custom_order)
            disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
            plt.tight_layout()
            plt.show()

        return mean_acc, std_acc, all_scores, all_labels, test_sample_names

    # def train_and_evaluate_leave_one_out_all_samples(
    #         self,
    #         normalize=False,
    #         scaler_type='standard',
    #         region=None,
    #         projection_source=False,
    #         classifier_type="RGC",
    #         feature_type="concatenated",
    #         show_confusion_matrix=False,
    # ):
    #     from collections import Counter
    #     from gcmswine import utils
    #     import numpy as np
    #     from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
    #     from sklearn.metrics import confusion_matrix
    #
    #     # Setup data and strategy
    #     cls_data = self.data.copy()
    #     labels = np.array(self.labels)
    #     year_labels = np.array(self.year_labels) if hasattr(self, 'year_labels') else np.array([])
    #     num_samples, num_timepoints, num_channels = cls_data.shape
    #
    #     def compute_features(channels):
    #         if feature_type == "concat_channels":
    #             return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
    #         elif feature_type == "tic":
    #             return np.sum(cls_data[:, :, channels], axis=2)
    #         elif feature_type == "tis":
    #             return np.sum(cls_data[:, :, channels], axis=1)
    #         elif feature_type == "tic_tis":
    #             tic = np.sum(cls_data[:, :, channels], axis=2)
    #             tis = np.sum(cls_data[:, :, channels], axis=1)
    #             return np.hstack([tic, tis])
    #         else:
    #             raise ValueError("Invalid feature_type. Use 'concatenated', 'tic', 'tis', or 'tic_tis'.")
    #
    #     feature_matrix = compute_features(list(range(num_channels)))
    #
    #     all_scores = []
    #     all_labels = []
    #     test_sample_names = []
    #     confusion_matrices = []
    #     accuracies = []
    #
    #     strategy = get_strategy_by_wine_kind(
    #         self.wine_kind, region,
    #         utils.get_custom_order_for_pinot_noir_region,
    #         class_by_year=self.class_by_year
    #     )
    #
    #     for idx in range(num_samples):
    #         try:
    #             cls = Classifier(
    #                 data=feature_matrix,
    #                 labels=labels,
    #                 classifier_type=classifier_type,
    #                 year_labels=year_labels,
    #                 dataset_origins=self.dataset_origins if hasattr(self, 'dataset_origins') else None,
    #                 strategy=strategy,
    #                 class_by_year=self.class_by_year,
    #                 labels_raw=self.labels_raw,
    #                 sample_labels=self.sample_labels
    #             )
    #
    #             results, scores, y_test, raw_test_labels = cls.train_and_evaluate_leave_one_out(
    #                 left_out_index=idx,
    #                 normalize=normalize,
    #                 scaler_type=scaler_type,
    #                 projection_source=projection_source,
    #             )
    #
    #             if 'accuracy' in results:
    #                 accuracies.append(results['accuracy'])
    #             else:
    #                 print(f"⚠️ Accuracy missing for sample {idx}")
    #
    #
    #             if scores is not None:
    #                 all_scores.append(scores[0])  # scores is (1, C) for one sample
    #                 all_labels.append(y_test[0])
    #                 test_sample_names.append(raw_test_labels[0])
    #
    #             if 'confusion_matrix' in results:
    #                 confusion_matrices.append(results['confusion_matrix'])
    #
    #         except Exception as e:
    #             print(f"⚠️ Skipping sample {idx} due to error: {e}")
    #
    #     accuracies = np.array(accuracies)
    #     mean_acc = np.mean(accuracies)
    #     std_acc = np.std(accuracies)
    #     if projection_source == "scores" and all_scores:
    #         all_scores = np.vstack(all_scores)
    #         all_labels = np.array(all_labels)
    #         test_sample_names = np.array(test_sample_names)
    #     else:
    #         all_scores, all_labels, test_sample_names = None, None, np.array(list(self.sample_labels))
    #     mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)
    #
    #     # Summary
    #     print("##################################")
    #     print("Leave-One-Out Evaluation Finished")
    #     print("##################################")
    #     print("\n##################################")
    #     print(f"Mean Accuracy (LOO): {mean_acc:.3f} ± {std_acc:.3f}")
    #
    #     if self.class_by_year:
    #         labels_used = self.year_labels
    #     else:
    #         labels_used = strategy.extract_labels(labels)
    #
    #     custom_order = strategy.get_custom_order(labels_used, year_labels)
    #     counts = Counter(labels_used)
    #
    #     print("\nLabel order:")
    #     if custom_order is not None:
    #         for label in custom_order:
    #             print(f"{label} ({counts.get(label, 0)})")
    #     else:
    #         for label in sorted(counts.keys()):
    #             print(f"{label} ({counts[label]})")
    #
    #     print("\nFinal Averaged Normalized Confusion Matrix:")
    #     print(mean_confusion_matrix)
    #     print("##################################")
    #     labels_used = year_labels if self.class_by_year else strategy.extract_labels(labels)
    #     custom_order = strategy.get_custom_order(labels_used, year_labels)
    #     if region == "winery":
    #         long_labels = [
    #         "Clos Des Mouches. Drouhin (FR): D",
    #         "Les Petits Monts. Drouhin (FR): R",
    #         "Vigne de l’Enfant Jésus. Bouchard (FR): E",
    #         "Les Cailles. Bouchard (FR): Q",
    #         "Bressandes. Jadot (FR): P",
    #         "Les Boudots. Jadot (FR): Z",
    #         "Domaine Schlumberger (FR): C",
    #         "Domaine Jean Sipp (FR): W",
    #         "Domaine Weinbach (FR): Y",
    #         "Domaine Brunner (CH): M",
    #         "Vin des Croisés (CH): N",
    #         "Domaine Villard et Fils (CH): J",
    #         "Domaine de la République (CH): L",
    #         "Les Maladaires (CH): H",
    #         "Marimar Estate (US): U",
    #         "Domaine Drouhin (US): X",
    #     ]
    #     counts = Counter(labels_used)
    #
    #     if show_confusion_matrix:
    #         import matplotlib.pyplot as plt
    #         from sklearn.metrics import ConfusionMatrixDisplay
    #
    #         fig, ax = plt.subplots(figsize=(8, 6))
    #         ax.set_xlabel('Predicted Label', fontsize=14)
    #         ax.set_ylabel('True Label', fontsize=14)
    #         ax.set_title(f'Confusion matrix by region')
    #         # ax.set_title(f'LOO Confusion Matrix by {region}')
    #         disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=custom_order)
    #         disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)
    #         plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
    #         # Conditional y-axis labels:
    #         if region == "winery":
    #             ax.set_yticks(range(len(long_labels)))  # Ensure ticks align
    #             ax.set_yticklabels(long_labels, fontsize=12)
    #         else:
    #             # Default: use the same custom_order for y-axis
    #             ax.set_yticks(range(len(custom_order)))
    #             ax.set_yticklabels(custom_order, fontsize=12)
    #         # plt.setp(ax.get_yticklabels(), fontsize=12)
    #         plt.tight_layout()
    #         plt.show()
    #
    #     return mean_acc, std_acc, all_scores, all_labels, test_sample_names



    def train_and_evaluate_all_channels(
            self, num_repeats=10, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None,
            print_results=True, n_jobs=-1, feature_type="concatenated",
            classifier_type="RGC", LOOPC=True,
            show_confusion_matrix=False,
            projection_source=False
    ):
        import re
        from collections import Counter
        from gcmswine import utils
        import numpy as np
        from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind

        cls_data = self.data.copy()
        labels = np.array(self.labels)
        year_labels = np.array(self.year_labels) if hasattr(self, 'year_labels') else np.array([])
        num_samples, num_timepoints, num_channels = cls_data.shape

        balanced_accuracies = []
        confusion_matrices = []
        all_scores = []
        all_labels = []
        test_samples_names = []

        def compute_features(channels):
            print(f"Computing features for channels: {channels}")
            if feature_type == "concat_channels":
                return np.hstack([cls_data[:, :, ch].reshape(num_samples, -1) for ch in channels])
            elif feature_type == "tic":
                return np.sum(cls_data[:, :, channels], axis=2)
            elif feature_type == "tis":
                return np.sum(cls_data[:, :, channels], axis=1)
            elif feature_type == "tic_tis":
                tic = np.sum(cls_data[:, :, channels], axis=2)
                tis = np.sum(cls_data[:, :, channels], axis=1)
                return np.hstack([tic, tis])
            else:
                raise ValueError("Invalid feature_type. Use 'concatenated', 'tic', 'tis', or 'tic_tis'.")

        feature_matrix = compute_features(list(range(num_channels)))

        dataset_origins_list = [self.dataset_origins[name] for name in self.sample_labels]

        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")
            try:
                strategy = get_strategy_by_wine_kind(
                    self.wine_kind, region,
                    utils.get_custom_order_for_pinot_noir_region,
                    class_by_year=self.class_by_year
                )

                cls = Classifier(
                    data=feature_matrix,
                    labels=labels,
                    classifier_type=classifier_type,
                    year_labels=year_labels,
                    dataset_origins=dataset_origins_list if hasattr(self, 'dataset_origins') else None,
                    strategy=strategy,
                    class_by_year=self.class_by_year,
                    labels_raw=self.labels_raw,
                    sample_labels=self.sample_labels,
                )

                results, scores, projection_labels, raw_test_labels = cls.train_and_evaluate_balanced(
                    normalize=normalize,
                    scaler_type=scaler_type,
                    region=region,
                    random_seed=random_seed + repeat_idx,
                    test_size=test_size,
                    LOOPC=LOOPC,
                    projection_source=projection_source,
                )
                if scores is not None:
                    all_scores.append(scores)
                    all_labels.append(projection_labels)
                    test_samples_names.append(raw_test_labels)

                if 'balanced_accuracy' in results and not np.isnan(results['balanced_accuracy']):
                    balanced_accuracies.append(results['balanced_accuracy'])
                else:
                    print(f"⚠️ No valid accuracy returned in repeat {repeat_idx + 1}")

                if 'confusion_matrix' in results:
                    if confusion_matrices and results['confusion_matrix'].shape != confusion_matrices[0].shape:
                        print("⚠️ Skipping confusion matrix with different shape.")
                    else:
                        confusion_matrices.append(results['confusion_matrix'])
                else:
                    print("⚠️ Skipping confusion matrix due to missing key.")

            except Exception as e:
                print(f"⚠️ Skipping repeat {repeat_idx + 1} due to error: {e}")

        if projection_source and all_scores:
            all_scores = np.vstack(all_scores)
            all_labels = np.concatenate(all_labels)
            test_samples_names = np.concatenate(test_samples_names)
        else:
            all_scores, all_labels = None, None

        mean_test_accuracy = np.mean(balanced_accuracies, axis=0)
        std_test_accuracy = np.std(balanced_accuracies, axis=0)
        mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)

        print("\n##################################")
        print(f"Mean Balanced Accuracy: {mean_test_accuracy:.3f} ± {std_test_accuracy:.3f}")

        if self.class_by_year:
            labels_used = self.year_labels
        else:
            labels_used = strategy.extract_labels(labels)
            # labels_used = self.strategy.extract_labels(labels)
        custom_order = strategy.get_custom_order(labels_used, self.year_labels)
        # custom_order = self.strategy.get_custom_order(labels_used, self.year_labels)
        counts = Counter(labels_used)

        print("\nLabel order:")
        if custom_order is not None:
            for label in custom_order:
                print(f"{label} ({counts.get(label, 0)})")
        else:
            for label in sorted(counts.keys()):
                print(f"{label} ({counts[label]})")

        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        if show_confusion_matrix:
            import matplotlib.pyplot as plt
            from sklearn.metrics import ConfusionMatrixDisplay

            headers = custom_order  # Can be overwritten manually below if needed
            # Example manual header sets (uncomment one as needed)
            # headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
            #             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
            #             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
            # headers = ["Burgundy", "Alsace", "Neuchatel", "Genève", "Valais", "Californie", "Oregon"]
            # headers = ["France", "Switzerland", "US"]
            # headers = ["Côte de Nuits", "Côte de Beaune"]

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.set_xlabel('Predicted Label', fontsize=14)
            ax.set_ylabel('True Label', fontsize=14)
            if LOOPC:
                cv_strategy = "LOOPC"
            else:
                cv_strategy = "Stratified"
            ax.set_title(f'{cv_strategy} Confusion Matrix by {region}')
            disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=headers)
            disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
            plt.setp(ax.get_yticklabels(), fontsize=12)
            plt.tight_layout()
            plt.show()

        if projection_source == "scores":
            return mean_test_accuracy, std_test_accuracy, all_scores, all_labels, test_samples_names
        else:
            return mean_test_accuracy, std_test_accuracy, None, None, None


    def train_and_evaluate_all_channels_old(
            self, num_repeats=10, random_seed=42, test_size=0.2, normalize=False,
            scaler_type='standard', use_pca=False, vthresh=0.97, region=None, print_results=True, n_jobs=-1,
            feature_type="concatenated", classifier_type="RGC", LOOPC=True, return_umap_data=False
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
            if feature_type == "concat_channels":
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


        all_umap_scores = []
        all_umap_labels = []
        for repeat_idx in range(num_repeats):
            print(f"\nRepeat {repeat_idx + 1}/{num_repeats}")
            # classifiers = ["DTC", "GNB", "KNN", "LDA", "LR", "PAC", "PER", "RFC", "RGC", "SGD", "SVM"]
            try:
                cls = Classifier(feature_matrix, labels, classifier_type=classifier_type, wine_kind=self.wine_kind,
                                 year_labels=self.year_labels)
                results, scores, umap_labels = cls.train_and_evaluate_balanced_old(
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
                    LOOPC=LOOPC,
                    return_umap_data=return_umap_data
                )
                if scores is not None:
                    all_umap_scores.append(scores)
                    all_umap_labels.append(umap_labels)

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

        if return_umap_data and all_umap_scores:
            all_umap_scores = np.vstack(all_umap_scores)
            all_umap_labels = np.concatenate(all_umap_labels)
        else:
            all_umap_scores, all_umap_labels = None, None

        # Compute average performance across repeats
        mean_test_accuracy = np.mean(balanced_accuracies, axis=0)
        std_test_accuracy = np.std(balanced_accuracies, axis=0)
        mean_confusion_matrix = utils.average_confusion_matrices_ignore_empty_rows(confusion_matrices)
        # mean_confusion_matrix = np.mean(confusion_matrices, axis=0)
        if self.wine_kind == "bordeaux":
            custom_order = None
        elif self.wine_kind == "press":
            custom_order = ["A", "B", "C"]
        else:
            custom_order = utils.get_custom_order_for_pinot_noir_region(region)

        print("\n##################################")
        print(f"Mean Balanced Accuracy: {mean_test_accuracy:.3f} ± {std_test_accuracy:.3f}")
        # Print label order used in confusion matrix

        if custom_order is not None:
            if region == "winery" or self.wine_kind == "press":
                # category_labels = extract_category_labels(self.labels)
                category_labels = self.extract_category_labels(self.labels)
            else:
                category_labels = self.labels
            # Compute counts
            counts = Counter(category_labels)

            # Print label order with counts, respecting custom_order
            print("\nLabel order (custom):")
            if self.year_labels.size > 0 and np.any(self.year_labels != None):
                year_count = Counter(self.year_labels)
                for year in year_count:
                    print(f"{year} ({year_count.get(year, 0)})")
            else:
                for label in custom_order:
                    print(f"{label} ({counts.get(label, 0)})")
        else:
            print("\nLabel order (default):")
            category_labels = self.extract_category_labels(self.labels)
            counts = Counter(category_labels)
            if self.year_labels.size > 0 and np.any(self.year_labels != None):
                year_count = Counter(self.year_labels)
                for year in year_count:
                    print(f"{year} ({year_count.get(year, 0)})")
            else:
                for label in np.unique(category_labels):
                    print(f"{label} ({counts.get(label, 0)})")


        print("\nFinal Averaged Normalized Confusion Matrix:")
        print(mean_confusion_matrix)
        print("##################################")

        # Display confusion matrix
        # headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
        #             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
        #             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
        # headers = ["Burgundy", "Alsace", "Neuchatel", "Genève", "Valais", "Californie", "Oregon"]
        # headers = ["France", "Switzerland", "US"]
        # headers = ["Côte de Nuits", "Côte de Beaune"]
        # from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
        # fig, ax = plt.subplots(figsize=(8, 6))
        # ax.set_xlabel('Predicted Label', fontsize=14)
        # ax.set_ylabel('True Label', fontsize=14)
        # # ax.set_title(f'Confusion Matrix by {region}')
        # ax.set_title(f'Confusion Matrix by Burgundy region')
        # disp = ConfusionMatrixDisplay(confusion_matrix=mean_confusion_matrix, display_labels=headers)
        # disp.plot(cmap="Blues", values_format=".0%", ax=ax, colorbar=False)  # you can change the colormap
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=12)
        # plt.setp(ax.get_yticklabels(), fontsize=12)
        # plt.tight_layout()
        # plt.show()

        if return_umap_data:
            return mean_test_accuracy, std_test_accuracy, all_umap_scores, all_umap_labels
        else:
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


def assign_origin_to_pinot_noir(original_keys, split_burgundy_ns=False):
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
        - 'D', 'E', 'Q', 'P', 'R', 'Z' => Burgundy (France)
        - 'C', 'K', 'W', 'Y' => Alsace (France)
    """
    # Dictionary to map letters to their specific regions (Origine)
    letter_to_origine = {
        # Switzerland
        'M': 'Neuchâtel',
        'N': 'Neuchâtel',
        'J': 'Geneva',
        'L': 'Geneva',
        'H': 'Valais',

        # US
        'U': 'California',
        'X': 'Oregon',

        # France
        # 'D': 'Burgundy',
        # 'E': 'Burgundy',
        # 'Q': 'Burgundy',
        # 'P': 'Burgundy',
        # 'R': 'Burgundy',
        # 'Z': 'Burgundy',
        'C': 'Alsace',
        'K': 'Alsace',
        'W': 'Alsace',
        'Y': 'Alsace'
    }

    burgundy_north = {'D', 'E', 'Q'}
    burgundy_south = {'P', 'R', 'Z'}

    origin_keys = []
    for key in original_keys:
        first_letter = key[0]
        if split_burgundy_ns:
            if first_letter in burgundy_north:
                origin_keys.append('Burgundy_North')
            elif first_letter in burgundy_south:
                origin_keys.append('Burgundy_South')
            elif first_letter in letter_to_origine:
                origin_keys.append(letter_to_origine[first_letter])
            else:
                origin_keys.append('Unknown')
        else:
            if first_letter in burgundy_north or first_letter in burgundy_south:
                origin_keys.append('Burgundy')
            elif first_letter in letter_to_origine:
                origin_keys.append(letter_to_origine[first_letter])
            else:
                origin_keys.append('Unknown')
    # # Create a new list by mapping the first letter of each key to its specific "Origine"
    # origin_keys = [letter_to_origine[key[0]] for key in original_keys]

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


def assign_north_south_to_burgundy(original_keys):
    """
    Map wine sample keys to either 'North Burgundy (NB)' or 'South Burgundy (SB)'.

    This function takes a list of wine sample keys, where the first letter of each key represents
    a region of origin, and returns a list of corresponding regions ('North Burgundy' or 'South Burgundy') for each key.

    Parameters
    ----------
    original_keys : list of str
        A list of strings where each string is a wine sample key. The first letter of each key
        corresponds to a specific region of origin.

    Returns
    -------
    burgundy_region_keys : list of str
        A list of strings where each string is either 'North Burgundy' or 'South Burgundy' based on the
        first letter of the key.

    """
    # if len(original_keys) != 61:
    #     raise ValueError(f"Incorrect wines passed. Input should be Burgundy wines only")

    # Dictionary to map letters to North or South Beaune
    letter_to_burgundy_region = {
        # North Burgundy (NB) or Côte de Nuits
        'Q': 'NB',
        'R': 'NB',
        'Z': 'NB',

        # South Burgundy (SB) or Côte de Beaune
        'D': 'SB',
        'E': 'SB',
        'P': 'SB',
    }

    # Create a new list by mapping the first letter of each key to North or South Beaune
    burgundy_region_keys = [letter_to_burgundy_region[key[0]] for key in original_keys]

    return burgundy_region_keys


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

# def assign_bordeaux_label(labels, vintage=False):
#     """
#     Assigns labels for Bordeaux wines, optionally grouping by vintage or by composite label.
#
#     Args:
#         labels (list of str): A list of wine sample labels (e.g., 'A2022', 'B2021B').
#         vintage (bool): If True, extract only the year (e.g., '2022').
#                         If False, extract composite label like 'A2022' but remove trailing 'B' duplicates.
#
#     Returns:
#         np.ndarray: Processed labels as per selected mode.
#     """
#     processed_labels = []
#
#     for label in labels:
#         match = re.search(r'(\d{4})', label)
#         if not match:
#             processed_labels.append(None)
#             continue
#
#         if vintage:
#             year = match.group(1)
#             processed_labels.append(year)
#         else:
#             # Extract leading letter and year, ignore trailing B
#             letter = label[match.start() - 1]
#             year = match.group(1)
#             processed_labels.append(f"{letter}{year}")
#
#     return np.array(processed_labels)


# def assign_composite_label_to_bordeaux_wine(labels):
#     """
#     Assigns composite labels (e.g., 'A2022') by grouping duplicates like 'A2022B'
#     under the same base label.
#
#     Args:
#         labels (list of str): A list of wine sample labels.
#
#     Returns:
#         list of str: A list of composite labels (e.g., 'A2022') with duplicates normalized.
#
#     Example:
#         labels = ['A2022', 'B2021B', 'C2023']
#         assign_composite_label_to_bordeaux_wine(labels)
#         >>> ['A2022', 'B2021', 'C2023']
#     """
#     pattern = re.compile(r'([A-Z]\d{4})B?')  # capture just the base part, ignore B
#
#     composite_labels = []
#     for label in labels:
#         match = pattern.search(label)
#         if match:
#             composite_labels.append(match.group(1))  # just 'A2022', no trailing 'B'
#         else:
#             composite_labels.append(None)
#
#     return composite_labels


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


