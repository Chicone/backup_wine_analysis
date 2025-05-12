import numpy as np
import timeit
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from joblib import Parallel, delayed

# Create synthetic data.
# Let's assume:
#   - Training set: 50 samples, 500 features.
#   - Validation set: 20 samples, 500 features.
#   - Binary labels.
np.random.seed(42)
X_train = np.random.rand(50, 500)
X_val   = np.random.rand(20, 500)
y_train = np.random.randint(0, 2, size=50)
y_val   = np.random.randint(0, 2, size=20)

# Define a candidate evaluation function that simulates selecting a subset of features.
# For example, for a given candidate index, we select features [candidate:candidate+50].
def evaluate_ridge(candidate):
    # Ensure that candidate + 50 does not exceed number of features.
    # (For this simulation, assume candidate is chosen so that candidate+50 <= 500.)
    X_train_subset = X_train[:, candidate:candidate+50]
    X_val_subset   = X_val[:, candidate:candidate+50]
    model = RidgeClassifier(alpha=1.0)
    model.fit(X_train_subset, y_train)
    y_pred = model.predict(X_val_subset)
    return balanced_accuracy_score(y_val, y_pred)

# Define a list of candidate indices. For example, evaluate 100 candidates.
tasks = list(range(100))

# Serial version: iterate over the candidate tasks in a simple loop.
def serial_version():
    results = [evaluate_ridge(candidate) for candidate in tasks]
    return results

# Parallel version: use joblib.Parallel to process tasks in parallel.
def parallel_version():
    results = Parallel(n_jobs=15, backend='loky')(
        delayed(evaluate_ridge)(candidate) for candidate in tasks
    )
    return results

# Profile the execution times. We'll run each version multiple times.
reps = 100

serial_time = timeit.timeit(serial_version, number=reps)
parallel_time = timeit.timeit(parallel_version, number=reps)

print("Serial version time: {:.4f} seconds".format(serial_time))
print("Parallel version time: {:.4f} seconds".format(parallel_time))
print("Overhead factor: {:.2f}x".format(parallel_time / serial_time))
