from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from utils import find_first_and_last_position, normalize_dict, normalize_data
import numpy as np
import re

class Classifier:
    def __init__(self, data, labels, classifier_type='LDA'):
        self.data = data
        self.labels = labels
        self.classifier = self._get_classifier(classifier_type)

    def _get_classifier(self, classifier_type):
        if classifier_type == 'LDA':
            return LinearDiscriminantAnalysis()
        elif classifier_type == 'LR':
            return LogisticRegression(C=1.0, random_state=0, n_jobs=-1, max_iter=1000)
        elif classifier_type == 'RFC':
            return RandomForestClassifier(n_estimators=100)
        elif classifier_type == 'PAC':
            return PassiveAggressiveClassifier()
        elif classifier_type == 'PER':
            return Perceptron()
        elif classifier_type == 'RGC':
            return RidgeClassifier()
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
            return GradientBoostingClassifier(n_estimators=100)

    def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None, normalize=False, scaler_type='standard'):
        """
        Train and evaluate the classifier using cross-validation.

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

        Returns
        -------
        float
            The mean accuracy score from cross-validation.

        Notes
        -----
        This function performs cross-validation on the classifier and prints the accuracy and its standard deviation.
        """
        np.random.seed(random_seed)
        scores = []
        print('Split', end=' ', flush=True)
        for i in range(n_splits):
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                vintage=vintage, test_size=test_size
            )

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)  # normalize test data using the scaler fit on train data only

            self.classifier.fit(X_train, y_train)
            print(i, end=' ', flush=True) if i % 5 == 0 else None
            scores.append(self.classifier.score(X_test, y_test))
        print()
        scores = np.asarray(scores)
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")
        return scores.mean()


    def train_and_evaluate_separate_datasets(self, X_train, y_train, X_test, y_test, n_splits=50, vintage=False,
                                             random_seed=42, normalize=True, scaler_type='standard'):
        """
        Train and evaluate the classifier on separate training and testing datasets using cross-validation.

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
        random_seed : int, optional
            The random seed for reproducibility. Default is 42.
        normalize : bool, optional
            Whether to normalize the data. Default is False.
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
        np.random.seed(random_seed)
        if normalize:
                X_train, _ = normalize_data(X_train, scaler=scaler_type)
        self.classifier.fit(X_train, y_train)

        test_cls = Classifier(X_test, y_test, classifier_type='LDA')
        scores = []
        print('Split', end=' ', flush=True)
        for i in range(n_splits):
            in_indices, out_indices, X_in, X_out, y_in, y_out = test_cls.split_data(vintage=vintage, test_size=None)
            if normalize:
                X_in, scaler = normalize_data(X_in, scaler=scaler_type)
                X_out = scaler.transform(X_out)
            scores.append(self.classifier.score(X_out, y_out))
            print(i, end=' ', flush=True) if i % 5 == 0 else None

        scores = np.asarray(scores)
        print()
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")

        return scores.mean()


    # def train_and_evaluate_separate_datasets(self, X_train, y_train, X_test, y_test, random_seed=42, normalize=False, scaler_type='standard'):
    #     """
    #     Train and evaluate the classifier on separate training and testing datasets using cross-validation.
    #
    #     Parameters
    #     ----------
    #     X_train : numpy.ndarray
    #         Training data.
    #     y_train : numpy.ndarray
    #         Training labels.
    #     X_test : numpy.ndarray
    #         Testing data.
    #     y_test : numpy.ndarray
    #         Testing labels.
    #     random_seed : int, optional
    #         The random seed for reproducibility. Default is 42.
    #     normalize : bool, optional
    #         Whether to normalize the data. Default is False.
    #     scaler_type : str, optional
    #         The type of scaler to use for normalization if `normalize` is True. Default is 'standard'.
    #
    #     Returns
    #     -------
    #     float
    #         The mean accuracy score from cross-validation.
    #
    #     Notes
    #     -----
    #     This function trains the classifier on the training data and evaluates it on the testing data.
    #     """
    #     np.random.seed(random_seed)
    #     # Normalize the data if requested
    #     if normalize:
    #         X_train, scaler = normalize_data(X_train, scaler=scaler_type)
    #         X_test, _ = normalize_data(X_test, scaler=scaler_type)  # normalize test data using the scaler fit on train data
    #     # Train the classifier
    #     self.classifier.fit(X_train, y_train)
    #     # Evaluate the classifier
    #     score = self.classifier.score(X_test, y_test)
    #     print("\033[96m" + "Accuracy: %0.3f " % (score) + "\033[0m")
    #     return score



    def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None, normalize=False, scaler_type='standard'):
        """
        Train and evaluate the classifier using cross-validation.

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

        Returns
        -------
        float
            The mean accuracy score from cross-validation.

        Notes
        -----
        This function performs cross-validation on the classifier and prints the accuracy and its standard deviation.
        """
        np.random.seed(random_seed)
        scores = []
        print('Split', end=' ', flush=True)
        for i in range(n_splits):
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                vintage=vintage, test_size=test_size
            )

            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)
                X_test = scaler.transform(X_test)  # normalize test data using the scaler fit on train data only

            self.classifier.fit(X_train, y_train)
            print(i, end=' ', flush=True) if i % 5 == 0 else None
            scores.append(self.classifier.score(X_test, y_test))
        print()
        scores = np.asarray(scores)
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")
        return scores.mean()


    # def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None):
    #     """
    #     Train and evaluate the classifier using cross-validation.
    #
    #     Parameters
    #     ----------
    #     n_splits : int, optional
    #         The number of splits for cross-validation. Default is 50.
    #     vintage : bool, optional
    #         Whether to process labels for vintage data. Default is False.
    #     random_seed : int, optional
    #         The random seed for reproducibility. Default is 42.
    #     test_size : float, optional
    #         The proportion of the dataset to include in the test split. If None, only one sample is used for testing. Default is None.
    #
    #     Returns
    #     -------
    #     float
    #         The mean accuracy score from cross-validation.
    #
    #     Notes
    #     -----
    #     This function performs cross-validation on the classifier and prints the accuracy and its standard deviation.
    #
    #     """
    #     np.random.seed(random_seed)
    #     scores = []
    #     num_samples = len(self.labels)
    #     processed_labels = self._process_labels(vintage)
    #     print('Split', end=' ', flush=True)
    #     for i in range(n_splits):
    #         test_indices = []
    #         train_indices = []
    #
    #         for label in np.unique(processed_labels):
    #             label_indices = np.where(np.array(processed_labels) == label)[0]
    #             np.random.shuffle(label_indices)
    #             if test_size is None:
    #                 # If test_size is not specified, use only one sample for testing
    #                 test_indices.extend(label_indices[:1])
    #                 train_indices.extend(label_indices[1:])
    #             else:
    #                 # Otherwise, use the specified percentage for test set
    #                 split_point = int(len(label_indices) * test_size)
    #                 test_indices.extend(label_indices[:split_point])
    #                 train_indices.extend(label_indices[split_point:])
    #
    #         test_indices = np.array(test_indices)
    #         train_indices = np.array(train_indices)
    #
    #         X_train, X_test = self.data[train_indices], self.data[test_indices]
    #         y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]
    #
    #         self.classifier.fit(X_train, y_train)
    #         print(i, end=' ', flush=True) if i % 5 == 0 else None
    #         scores.append(self.classifier.score(X_test, y_test))
    #     print()
    #     scores = np.asarray(scores)
    #     print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")
    #     return scores.mean()

    def _process_labels(self, vintage):
        processed_labels = []
        for label in self.labels:
            match = re.search(r'\d+', label) # search for first digit
            if vintage:
                # processed_labels.append(label[-5])
                processed_labels.append(label[match.start():])
            else:
                if label[match.start() - 1] == '_':
                    lb = label[match.start() - 2]
                else:
                    lb = label[match.start() - 1]
                processed_labels.append(lb)
        return np.array(processed_labels)


    def split_data(self, vintage=False, test_size=None):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, only one sample per label is used for
            testing. Default is None.

        Returns
        -------
        tuple
            A tuple containing the training indices, testing indices, training data, testing data, training labels, and testing labels.
        """
        processed_labels = self._process_labels(vintage)
        test_indices = []
        train_indices = []

        for label in np.unique(processed_labels):
            label_indices = np.where(np.array(processed_labels) == label)[0]
            np.random.shuffle(label_indices)
            if test_size is None:
                # If test_size is not specified, use only one sample for testing
                test_indices.extend(label_indices[:1])
                train_indices.extend(label_indices[1:])
            else:
                # Otherwise, use the specified percentage for test set
                split_point = int(len(label_indices) * test_size)
                test_indices.extend(label_indices[:split_point])
                train_indices.extend(label_indices[split_point:])

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        X_train, X_test = self.data[train_indices], self.data[test_indices]
        y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]

        return train_indices, test_indices, X_train, X_test, y_train, y_test

def process_labels(labels, vintage):
    processed_labels = []
    for label in labels:
        match = re.search(r'\d+', label) # search for first digit
        if vintage:
            # processed_labels.append(label[-5])
            processed_labels.append(label[match.start():])
        else:
            if label[match.start() - 1] == '_':
                lb = label[match.start() - 2]
            else:
                lb = label[match.start() - 1]
            processed_labels.append(lb)

    return np.array(processed_labels)
