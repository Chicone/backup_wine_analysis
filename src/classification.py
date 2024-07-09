from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier, PassiveAggressiveClassifier, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from utils import find_first_and_last_position
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

    def train_and_evaluate(self, n_splits=50, vintage=False, random_seed=42, test_size=None):
        np.random.seed(random_seed)
        scores = []
        num_samples = len(self.labels)
        processed_labels = self._process_labels(vintage)
        print('Split', end=' ', flush=True)
        for i in range(n_splits):
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

            self.classifier.fit(X_train, y_train)
            print(i, end=' ', flush=True) if i % 5 == 0 else None
            scores.append(self.classifier.score(X_test, y_test))
        print()
        scores = np.asarray(scores)
        print("\033[96m" + "Accuracy: %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2) + "\033[0m")
        return scores.mean()

    def _process_labels(self, vintage):
        processed_labels = []
        for label in self.labels:
            match = re.search(r'\d+', label)
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