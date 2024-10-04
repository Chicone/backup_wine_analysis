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
    def __init__(self, data, labels, classifier_type='LDA', wine_kind='bordeaux'):
        self.data = data
        self.labels = labels
        self.classifier = self._get_classifier(classifier_type)
        self.wine_kind = wine_kind

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

    from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
    import numpy as np

    def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None, normalize=False,
                           scaler_type='standard'):
        """
        Train and evaluate the classifier using cross-validation, and calculate the mean confusion matrix.

        Parameters
        ----------
        n_splits : int, optional
            The number of splits for cross-validation. Default is 50.
        vintage : bool, optional
            Whether to process labels for vintage data. Default is False.
        random_seed : int, optional
            The random seed for reproducibility. Default is 42.
        test_size : float, optional
            The proportion of the dataset to include in the test split. If None, only one sample
            is used for testing. Default is None.
        normalize : bool, optional
            Whether to normalize the data. Default is False.
        scaler_type : str, optional
            The type of scaler to use for normalization if `normalize` is True. Default is 'standard'.

        Returns
        -------
        dict
            A dictionary containing mean accuracy, precision, recall, F1-score, and the mean confusion matrix.

        Notes
        -----
        This function performs cross-validation on the classifier and prints accuracy, precision, recall, F1-score,
        and the mean confusion matrix over all splits.
        """
        from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

        # Set the random seed for reproducibility of the results
        np.random.seed(random_seed)

        # Initialize lists to store scores for different metrics
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []

        # Initialize a confusion matrix accumulator
        confusion_matrix_sum = None

        print('Split', end=' ', flush=True)

        # Perform cross-validation over the specified number of splits
        for i in range(n_splits):
            # Split the data into training and testing sets
            train_indices, test_indices, X_train, X_test, y_train, y_test = self.split_data(
                vintage=vintage, test_size=test_size
            )

            # Normalize the data if normalization is enabled
            if normalize:
                X_train, scaler = normalize_data(X_train, scaler=scaler_type)  # Fit scaler on training data
                X_test = scaler.transform(X_test)  # Transform test data using the train scaler

            # Train the classifier on the training data
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
        print(y_test)

        # Return the mean scores and the mean confusion matrix
        return {
            'mean_accuracy': accuracy_scores.mean(),
            'mean_precision': precision_scores.mean(),
            'mean_recall': recall_scores.mean(),
            'mean_f1_score': f1_scores.mean(),
            'mean_confusion_matrix': mean_confusion_matrix
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

    def _process_labels(self, vintage):
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
        if self.wine_kind == 'pinot':
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

    def split_data(self, vintage=False, test_size=None):
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

        if self.wine_kind == 'bordeaux':
            # Process the labels according to whether they are vintage or not
            processed_labels = self._process_labels(vintage)
        else:
            processed_labels = self.labels

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
                test_indices.extend(label_indices[:1])  # Take the first shuffled index for testing
                train_indices.extend(label_indices[1:])  # The rest is for training
            else:
                # If test_size is specified, calculate the split point based on the test_size proportion
                split_point = int(len(label_indices) * test_size)
                test_indices.extend(label_indices[:split_point])  # The first part goes into testing
                train_indices.extend(label_indices[split_point:])  # The remaining is for training

        test_indices = np.array(test_indices)
        train_indices = np.array(train_indices)

        # Split the data and labels into training and testing sets based on the calculated indices
        X_train, X_test = self.data[train_indices], self.data[test_indices]
        y_train, y_test = np.array(processed_labels)[train_indices], np.array(processed_labels)[test_indices]

        # Return the indices, data, and labels for both training and testing sets
        return train_indices, test_indices, X_train, X_test, y_train, y_test


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



