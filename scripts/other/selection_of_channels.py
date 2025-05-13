from config import (DATA_DIRECTORY, N_DECIMATION, REGION, VINTAGE, WINE_KIND)
from classification import (
    Classifier, process_labels, assign_country_to_pinot_noir, assign_origin_to_pinot_noir,
    assign_continent_to_pinot_noir, assign_winery_to_pinot_noir, assign_year_to_pinot_noir,
    assign_north_south_to_beaune
)
from wine_analysis import GCMSDataProcessor
import utils
import numpy as np

def greedy_nested_cv_channel_selection_tis(
        data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
        max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
        parallel=True, n_jobs=-1, min_frequency=3, selection_direction="backward"):
    """
    Perform nested CV for greedy channel selection using SNR‐based candidate evaluation.

    Depending on the parameter 'selection_direction', the algorithm can either:

      - Do backward elimination (selection_direction="backward"): start with all channels and, at each step,
        remove the channel whose removal yields the best (or least harmed) validation accuracy.

      - Do forward selection (selection_direction="forward"): start with an empty set and, at each step,
        add the channel that improves validation accuracy the most.

    The algorithm uses an outer CV loop (each repetition is a new stratified split) and, within each repetition,
    an inner CV loop to evaluate the performance when each candidate channel (from the candidate pool) is added
    (or removed). A candidate’s performance is aggregated over inner folds, and candidates that appear in fewer
    than min_frequency folds are discarded (for robustness). The candidate with the highest improvement over the
    baseline (current selection performance) is selected. After each step the outer test accuracy is evaluated.
    Finally, the function dynamically plots the global (averaged across outer repetitions) validation and test
    accuracy progression after each outer repetition. The candidate channel selection uses its own validation SNR,
    and the final plot is annotated with the channel (aggregated across repetitions) that produced the best SNR,
    along with the number of repetitions that channel appeared.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_channels)
        The data matrix.
    labels : array-like, shape (n_samples,)
        The corresponding labels.
    alpha : float, default=1.0
        Regularization parameter for RidgeClassifier.
    num_outer_repeats : int, default=3
        Number of outer repetitions (each using a new stratified train/test split).
    inner_cv_folds : int, default=3
        Number of inner CV repeats (using a custom leave-one-per-class splitter).
    max_channels : int, default=40
        Maximum number of selection steps.
    normalize : bool, default=True
        Whether to normalize the data.
    scaler_type : str, default='standard'
        Either 'standard' or 'minmax' to specify the scaler.
    random_seed : int, default=None
        Random seed for reproducibility.
    parallel : bool, default=True
        Whether to use parallel processing for inner folds.
    n_jobs : int, default=-1
        Number of jobs for parallel processing.
    min_frequency : int, default=3
        Minimum frequency required for a candidate (across inner folds) to be eligible.
    selection_direction : str, default="backward"
        Either "backward" for backward elimination or "forward" for forward selection.

    Returns
    -------
    all_selected_channels : list
        A list (one element per outer repetition) of selection sequences (i.e. the channels removed or added in order).
    all_test_accuracy_per_step : list
        A list (one element per outer repetition) of lists of outer test accuracies (one per selection step).
    all_validation_accuracy_per_step : list
        A list (one element per outer repetition) of lists of inner CV validation accuracies (one per selection step).
    """
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    from collections import Counter

    # Set random seed if not provided.
    if random_seed is None:
        random_seed = np.random.randint(0, int(1e6))
    rng = np.random.default_rng(random_seed)

    # Global accumulators.
    all_selected_channels = []  # One selection sequence per outer repetition.
    all_test_accuracy_per_step = []  # Outer test accuracies per selection step.
    all_validation_accuracy_per_step = []  # Inner CV validation accuracies per selection step.

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
    # Candidate evaluation function (works for both directions).
    # ----------------------------------
    def evaluate_channel_candidate(ch, current_selection, X_train, y_train, X_val, y_val, direction):
        """
        Evaluate performance when a candidate channel is either added (forward) or removed (backward)
        from the current selection.
        """
        if direction == "backward":
            candidate_channels = [x for x in current_selection if x != ch]
        elif direction == "forward":
            candidate_channels = current_selection + [ch]
        else:
            raise ValueError("Direction must be either 'backward' or 'forward'")
        X_train_subset = X_train[:, candidate_channels]
        X_val_subset = X_val[:, candidate_channels]
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_val_subset)
        return ch, balanced_accuracy_score(y_val, y_pred)

    def evaluate_inner_fold_selection(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                                      current_selection, candidate_pool, alpha, direction,
                                      normalize=False, scaler_type='standard'):
        """
        Evaluate one inner CV fold for selection.
        Returns (best_candidate, best_accuracy) for that fold.
        """
        X_inner_train = X_train_val[inner_train_idx]
        X_inner_val = X_train_val[inner_val_idx]
        y_inner_train = y_train_val[inner_train_idx]
        y_inner_val = y_train_val[inner_val_idx]

        if normalize:
            X_inner_train, X_inner_val = scale_data(X_inner_train, X_inner_val, scaler_type)

        results = [evaluate_channel_candidate(ch, current_selection, X_inner_train, y_inner_train,
                                              X_inner_val, y_inner_val, direction)
                   for ch in candidate_pool]
        # Add a tiny random jitter to each score.
        jitter = np.random.rand(len(results)) * 1e-10  # Adjust the factor as needed.
        results_jittered = [(cand, score + j) for ((cand, score), j) in zip(results, jitter)]

        best_candidate, best_accuracy = max(results_jittered, key=lambda x: x[1])
        return best_candidate, best_accuracy

    def scale_data(X_train, X_test, scaler_type='standard'):
        if scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def evaluate_candidate_baseline(current_selection, X_train, y_train, X_val, y_val):
        """
        Evaluate the baseline performance using the current selection of channels.
        If current_selection is empty, returns 0.
        """
        if len(current_selection) == 0:
            return 0
        X_train_subset = X_train[:, current_selection]
        X_val_subset = X_val[:, current_selection]
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
        outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=rng.integers(0, int(1e6)))
        train_val_idx, test_idx = next(outer_cv.split(data, labels))
        X_train_val, X_test = data[train_val_idx], data[test_idx]

        if normalize:
            X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)

        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
                                                   random_state=rng.integers(0, int(1e6)))
        # inner_cv = LeaveOneFromEachClassCV(shuffle=True, random_state=rng.integers(0, int(1e6)))

        # Initialize the current selection based on the direction.
        if selection_direction == "backward":
            current_selection = list(range(data.shape[1]))
        elif selection_direction == "forward":
            current_selection = []  # start empty
        else:
            raise ValueError("selection_direction must be either 'backward' or 'forward'")

        selection_sequence = []  # Record the channel selected at each step.
        rep_validation_accuracies = []  # Inner CV (validation) accuracy per selection step.
        rep_test_accuracies = []  # Outer test accuracy per selection step.

        # Compute average baseline performance for all folds using the current selection.
        baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_candidate_baseline)(current_selection,
                                                 X_train_val[inner_train_idx],
                                                 y_train_val[inner_train_idx],
                                                 X_train_val[inner_val_idx],
                                                 y_train_val[inner_val_idx])
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
        )
        baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0

        # Compute accuracy when all channels are used (before removing any channel)
        X_train_full = X_train_val[:, current_selection]
        X_test_full = X_test[:, current_selection]

        # Fit and evaluate model
        model.fit(X_train_full, y_train_val)
        y_pred = model.predict(X_test_full)
        initial_test_accuracy = balanced_accuracy_score(y_test, y_pred)

        # Store this as the first point for plotting
        rep_test_accuracies.append(initial_test_accuracy)

        # Compute baseline validation accuracy for 0 channels removed
        baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_candidate_baseline)(current_selection,
                                                 X_train_val[inner_train_idx],
                                                 y_train_val[inner_train_idx],
                                                 X_train_val[inner_val_idx],
                                                 y_train_val[inner_val_idx])
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
        )
        initial_validation_accuracy = np.mean(baseline_fold_scores)
        rep_validation_accuracies.append(initial_validation_accuracy)

        for step in range(max_channels):
            print(f"Selecting channel {step + 1}/{max_channels}")
            # Determine the candidate pool based on the direction.
            if selection_direction == "backward":
                candidate_pool = current_selection.copy()
            else:  # forward
                candidate_pool = list(set(range(data.shape[1])) - set(current_selection))

            if not candidate_pool:
                break

            # Evaluate each candidate on the inner folds in parallel.
            fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_inner_fold_selection)(
                    inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                    current_selection, candidate_pool, alpha, selection_direction,
                    normalize, scaler_type
                )
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )
            best_candidate_per_fold = [result[0] for result in fold_results]
            # Compute frequency for each candidate across inner folds.
            freq_dict = {cand: best_candidate_per_fold.count(cand) for cand in candidate_pool}
            candidates_meeting_freq = [cand for cand in candidate_pool if freq_dict[cand] >= min_frequency]

            # ---- Validation SNR Computation for Candidate Selection ----
            inner_folds = list(inner_cv.split(X_train_val, y_train_val))
            candidate_scores = {}
            for cand in candidate_pool:
                fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(evaluate_channel_candidate)(cand, current_selection,
                                                        X_train_val[inner_train_idx],
                                                        y_train_val[inner_train_idx],
                                                        X_train_val[inner_val_idx],
                                                        y_train_val[inner_val_idx],
                                                        selection_direction)
                    for inner_train_idx, inner_val_idx in inner_folds
                )
                scores = np.array([score for _, score in fold_scores])
                candidate_scores[cand] = scores

            epsilon = 1e-8  # avoid division by zero.
            candidate_snr = {}
            for cand, scores in candidate_scores.items():
                mean_diff = np.mean(scores) - baseline
                std_diff = np.std(scores)
                candidate_snr[cand] = mean_diff / (std_diff + epsilon)

            eligible_candidates = [cand for cand in candidate_pool if freq_dict.get(cand, 0) >= min_frequency]
            if not eligible_candidates:
                eligible_candidates = candidate_pool
            best_candidate = max(eligible_candidates, key=lambda cand: candidate_snr[cand])
            best_avg_score = np.mean(candidate_scores[best_candidate])
            selection_sequence.append(best_candidate)
            # Update current_selection based on the direction.
            if selection_direction == "backward":
                current_selection.remove(best_candidate)
            else:  # forward
                current_selection.append(best_candidate)
            rep_validation_accuracies.append(best_avg_score)
            # -------------------------------------------------------------

            # Evaluate outer test accuracy using the current selection.
            X_train_subset = X_train_val[:, current_selection]
            X_test_subset = X_test[:, current_selection]
            model.fit(X_train_subset, y_train_val)
            y_pred = model.predict(X_test_subset)
            rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))

            # Update baseline.
            baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_candidate_baseline)(current_selection,
                                                     X_train_val[inner_train_idx],
                                                     y_train_val[inner_train_idx],
                                                     X_train_val[inner_val_idx],
                                                     y_train_val[inner_val_idx])
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )
            baseline = np.mean(baseline_fold_scores)

        # End of selection loop for this outer repetition.
        all_selected_channels.append(selection_sequence.copy())
        all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
        all_test_accuracy_per_step.append(rep_test_accuracies.copy())

        # ----- Global Aggregation and Plotting After Each Outer Repetition -----
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

        # ---- Aggregate the Selected Channels Across Outer Repetitions ----
        final_selected_channels = []
        final_selected_channels_counts = []
        max_steps = max(len(seq) for seq in all_selected_channels)
        for step in range(max_steps):
            candidate_info = []
            for rep in range(len(all_selected_channels)):
                if step < len(all_selected_channels[rep]):
                    candidate_info.append(all_selected_channels[rep][step])
            if candidate_info:
                best_channel, count = Counter(candidate_info).most_common(1)[0]
                final_selected_channels.append(best_channel)
                final_selected_channels_counts.append(count)
        # ---------------------------------------------------------------------

        ax.clear()
        # steps_axis = np.arange(0, min_steps + 1)
        steps_axis = np.arange(0, len(final_selected_channels) + 1)

        ax.plot(steps_axis, np.array(global_val), marker='s', linestyle='-', label='Validation Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_val) - np.array(global_val_std),
                        np.array(global_val) + np.array(global_val_std), alpha=0.2)
        ax.plot(steps_axis, np.array(global_test), marker='o', linestyle='--', label='Test Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_test) - np.array(global_test_std),
                        np.array(global_test) + np.array(global_test_std), alpha=0.2)
        if selection_direction == "backward":
            ax.set_xlabel("Number of Removed Channels")
        else:
            ax.set_xlabel("Number of Added Channels")

        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
        ax.legend()
        ax.grid()

        # Annotate each selection step with the channel and its frequency (number of repetitions).
        for i, ch in enumerate(final_selected_channels):
            count = final_selected_channels_counts[i]
            if i == 0:
                continue  # Skip annotation at step 0
            # Annotate the channel number with a larger font.
            ax.annotate(f"{ch}", (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=8, color='black')
            # Annotate the count (in parentheses) with a smaller font, slightly lower.
            ax.annotate(f"({count})", (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, -30),
                        ha="center", fontsize=6, color='black')
        plt.draw()
        plt.pause(5.0)
        # End dynamic global plotting for this outer repetition.

    plt.ioff()
    plt.show()

    return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step


# def greedy_nested_cv_channel_selection_3d(
#         data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
#         max_channels=40, normalize=False, scaler_type='standard', random_seed=42,
#         parallel=True, n_jobs=-1, min_frequency=3, selection_direction="backward"):
#     """
#     Perform nested CV for greedy channel selection using SNR‐based candidate evaluation,
#     adapted for 3D GCMS data where each sample is of shape
#     (n_samples, n_timepoints, n_channels) (channels in the last dimension).
#
#     Classification is performed on the mean chromatogram computed over the selected channels,
#     i.e. np.mean(X[:, :, selected_channels], axis=2).
#
#     Depending on 'selection_direction':
#       - "backward": Start with all channels and remove one at a time.
#       - "forward":  Start with no channels and add one at a time.
#
#     To improve efficiency, the function caches the cumulative sum (and count) over the currently selected
#     channels so that when evaluating a candidate channel, the new mean can be computed by simply updating the sum.
#
#     Parameters
#     ----------
#     data : array-like, shape (n_samples, n_timepoints, n_channels)
#         The input data.
#     labels : array-like, shape (n_samples,)
#         The corresponding labels.
#     alpha : float, default=1.0
#         Regularization parameter for RidgeClassifier.
#     num_outer_repeats : int, default=3
#         Number of outer CV repetitions.
#     inner_cv_folds : int, default=3
#         Number of inner CV folds.
#     max_channels : int, default=40
#         Maximum number of selection steps.
#     normalize : bool, default=False
#         Whether to apply scaling (not used for 3D data in this version).
#     scaler_type : str, default='standard'
#         Either 'standard' or 'minmax' (not used for 3D data).
#     random_seed : int, default=42
#         Random seed for reproducibility.
#     parallel : bool, default=True
#         Whether to use parallel processing.
#     n_jobs : int, default=-1
#         Number of jobs for parallel processing.
#     min_frequency : int, default=3
#         Minimum frequency required for a candidate to be eligible.
#     selection_direction : str, default="backward"
#         Either "backward" for elimination or "forward" for selection.
#
#     Returns
#     -------
#     all_selected_channels : list
#         A list (one per outer repetition) of selection sequences (channels removed or added in order).
#     all_test_accuracy_per_step : list
#         A list (one per outer repetition) of outer test accuracies (one per selection step).
#     all_validation_accuracy_per_step : list
#         A list (one per outer repetition) of inner CV validation accuracies (one per selection step).
#     """
#     import numpy as np
#     from sklearn.linear_model import RidgeClassifier
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
#     from joblib import Parallel, delayed
#     import matplotlib.pyplot as plt
#     from collections import Counter
#
#     def scale_data(X_train, X_test, scaler_type='standard'):
#         if scaler_type == 'standard':
#             from sklearn.preprocessing import StandardScaler
#             scaler = StandardScaler()
#         elif scaler_type == 'minmax':
#             from sklearn.preprocessing import MinMaxScaler
#             scaler = MinMaxScaler()
#         else:
#             raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
#         X_train = scaler.fit_transform(X_train)
#         X_test = scaler.transform(X_test)
#         return X_train, X_test
#
#     # --- Helper function: Compute mean chromatogram over selected channels ---
#     def get_mean_chromatogram(X, channels):
#         """
#         Given X of shape (n_samples, n_timepoints, n_channels), compute the mean
#         over the given channels along the last axis, returning an array of shape (n_samples, n_timepoints).
#         """
#         return np.mean(X[:, :, channels], axis=2)
#
#     # --- Custom CV classes (unchanged) ---
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
#     # --- Optimized Candidate Evaluation Using Cumulative Sums ---
#     def evaluate_channel_candidate(cand, current_selection, X_train, y_train, X_val, y_val,
#                                    direction, current_sum_train, current_count_train,
#                                    current_sum_val, current_count_val, alpha=1.0):
#         """
#         Evaluate candidate 'cand' by updating the cumulative sum over selected channels.
#
#         For data of shape (n_samples, n_timepoints, n_channels), the current mean is computed as:
#              current_mean = current_sum / current_count
#
#         For forward selection:
#              new_mean = (current_sum + X[:, :, cand]) / (current_count + 1)
#         For backward elimination:
#              new_mean = (current_sum - X[:, :, cand]) / (current_count - 1)
#
#         If any of the caching arguments (current_sum_train, etc.) are None, they are computed from current_selection.
#         If current_count is zero (i.e. no channels selected), then the candidate’s data is used directly.
#         """
#         # If caching values are not provided, compute them from current_selection.
#         if current_sum_train is None or current_count_train is None or current_sum_val is None or current_count_val is None:
#             if len(current_selection) > 0:
#                 current_sum_train = np.sum(X_train[:, :, current_selection], axis=2)
#                 current_count_train = len(current_selection)
#                 current_sum_val = np.sum(X_val[:, :, current_selection], axis=2)
#                 current_count_val = len(current_selection)
#             else:
#                 # When the current selection is empty, there is no cumulative sum.
#                 current_sum_train = np.zeros((X_train.shape[0], X_train.shape[1]))
#                 current_count_train = 0
#                 current_sum_val = np.zeros((X_val.shape[0], X_val.shape[1]))
#                 current_count_val = 0
#
#         # If current_count is zero, use the candidate channel's data directly.
#         if current_count_train == 0 or current_count_val == 0:
#             new_mean_train = X_train[:, :, cand]
#             new_mean_val = X_val[:, :, cand]
#         else:
#             if direction == "backward":
#                 new_sum_train = current_sum_train - X_train[:, :, cand]
#                 new_count_train = current_count_train - 1
#                 new_sum_val = current_sum_val - X_val[:, :, cand]
#                 new_count_val = current_count_val - 1
#             elif direction == "forward":
#                 new_sum_train = current_sum_train + X_train[:, :, cand]
#                 new_count_train = current_count_train + 1
#                 new_sum_val = current_sum_val + X_val[:, :, cand]
#                 new_count_val = current_count_val + 1
#             else:
#                 raise ValueError("Direction must be 'backward' or 'forward'")
#             new_mean_train = new_sum_train / new_count_train
#             new_mean_val = new_sum_val / new_count_val
#
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(new_mean_train, y_train)
#         y_pred = model.predict(new_mean_val)
#         return cand, balanced_accuracy_score(y_val, y_pred)
#
#     def evaluate_inner_fold_selection(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                                       current_selection, candidate_pool, alpha, direction):
#         """
#         Evaluate one inner CV fold for selection.
#         Returns (best_candidate, best_accuracy) for that fold.
#         For 3D data, compute the cumulative sum and count over the currently selected channels once,
#         and update these for each candidate.
#         """
#         X_inner_train = X_train_val[inner_train_idx]
#         X_inner_val = X_train_val[inner_val_idx]
#         y_inner_train = y_train_val[inner_train_idx]
#         y_inner_val = y_train_val[inner_val_idx]
#         if X_inner_train.ndim == 3:
#             # Compute cumulative sum and count for the current selection.
#             if len(current_selection) > 0:
#                 current_sum_train = np.sum(X_inner_train[:, :, current_selection],
#                                            axis=2)  # shape: (n_samples, n_timepoints)
#                 current_sum_val = np.sum(X_inner_val[:, :, current_selection], axis=2)
#                 current_count_train = len(current_selection)
#                 current_count_val = len(current_selection)
#             else:
#                 # For forward selection with an empty selection, initialize zeros.
#                 current_sum_train = np.zeros((X_inner_train.shape[0], X_inner_train.shape[1]))
#                 current_sum_val = np.zeros((X_inner_val.shape[0], X_inner_val.shape[1]))
#                 current_count_train = 0
#                 current_count_val = 0
#         else:
#             current_sum_train = None
#             current_sum_val = None
#             current_count_train = len(current_selection)
#             current_count_val = len(current_selection)
#         results = []
#         for cand in candidate_pool:
#             # Special case: forward selection with empty current_selection.
#             if direction == "forward" and current_count_train == 0:
#                 new_mean_train = X_inner_train[:, :, cand].copy()  # shape: (n_samples, n_timepoints)
#                 new_mean_val = X_inner_val[:, :, cand].copy()
#                 model = RidgeClassifier(alpha=alpha)
#                 model.fit(new_mean_train, y_inner_train)
#                 y_pred = model.predict(new_mean_val)
#                 score = balanced_accuracy_score(y_inner_val, y_pred)
#                 results.append((cand, score))
#             else:
#                 res = evaluate_channel_candidate(cand, current_selection, X_inner_train, y_inner_train,
#                                                  X_inner_val, y_inner_val, direction,
#                                                  current_sum_train, current_count_train,
#                                                  current_sum_val, current_count_val, alpha=alpha)
#                 results.append(res)
#
#         # Add a tiny random jitter to each score.
#         jitter = np.random.rand(len(results)) * 1e-10  # Adjust the factor as needed.
#         results_jittered = [(cand, score + j) for ((cand, score), j) in zip(results, jitter)]
#
#         best_candidate, best_accuracy = max(results_jittered, key=lambda x: x[1])
#         return best_candidate, best_accuracy
#
#     def evaluate_candidate_baseline(current_selection, X_train, y_train, X_val, y_val,
#                                     normalize):
#         if len(current_selection) == 0:
#             return 0
#         X_train_subset = get_mean_chromatogram(X_train, current_selection)
#         X_val_subset = get_mean_chromatogram(X_val, current_selection)
#         if normalize:
#             X_train_subset, X_val_subset = scale_data(X_train_subset, X_val_subset)
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(X_train_subset, y_train)
#         y_pred = model.predict(X_val_subset)
#         return balanced_accuracy_score(y_val, y_pred)
#
#     # --- Set random seed and prepare accumulators ---
#     rng = np.random.default_rng(random_seed)
#     all_selected_channels = []
#     all_test_accuracy_per_step = []
#     all_validation_accuracy_per_step = []
#
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(10, 6))
#
#     # --- Outer CV Loop ---
#     for repeat in range(num_outer_repeats):
#         print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
#         outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
#                                           random_state=rng.integers(0, int(1e6)))
#         train_val_idx, test_idx = next(outer_cv.split(data, labels))
#         X_train_val, X_test = data[train_val_idx], data[test_idx]
#         # For 3D data, we skip scaling.
#         y_train_val, y_test = labels[train_val_idx], labels[test_idx]
#
#         inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
#                                                    random_state=rng.integers(0, int(1e6)))
#         if selection_direction == "backward":
#             current_selection = list(range(data.shape[2]))  # channels are in the last dimension
#         elif selection_direction == "forward":
#             current_selection = []
#         else:
#             raise ValueError("selection_direction must be either 'backward' or 'forward'")
#         selection_sequence = []
#         rep_validation_accuracies = []
#         rep_test_accuracies = []
#
#         baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#             delayed(evaluate_candidate_baseline)(current_selection,
#                                                  X_train_val[inner_train_idx],
#                                                  y_train_val[inner_train_idx],
#                                                  X_train_val[inner_val_idx],
#                                                  y_train_val[inner_val_idx],
#                                                  normalize)
#             for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#         )
#         baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0
#
#         for step in range(max_channels):
#             print(f"Selecting channel {step + 1}/{max_channels}")
#             if selection_direction == "backward":
#                 candidate_pool = current_selection.copy()
#             else:
#                 candidate_pool = list(set(range(data.shape[2])) - set(current_selection))
#             if not candidate_pool:
#                 break
#
#             fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_inner_fold_selection)(
#                     inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                     current_selection, candidate_pool, alpha, selection_direction
#                 )
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#             best_candidate_per_fold = [result[0] for result in fold_results]
#             freq_dict = {cand: best_candidate_per_fold.count(cand) for cand in candidate_pool}
#             eligible_candidates = [cand for cand in candidate_pool if freq_dict.get(cand, 0) >= min_frequency]
#             if not eligible_candidates:
#                 eligible_candidates = candidate_pool
#
#             inner_folds = list(inner_cv.split(X_train_val, y_train_val))
#             candidate_scores = {}
#             for cand in candidate_pool:
#                 fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                     delayed(evaluate_channel_candidate)(cand, current_selection,
#                                                         X_train_val[inner_train_idx],
#                                                         y_train_val[inner_train_idx],
#                                                         X_train_val[inner_val_idx],
#                                                         y_train_val[inner_val_idx],
#                                                         selection_direction,
#                                                         None, None, None, None)
#                     for inner_train_idx, inner_val_idx in inner_folds
#                 )
#                 scores = np.array([score for _, score in fold_scores])
#                 candidate_scores[cand] = scores
#
#             epsilon = 1e-8
#             candidate_snr = {}
#             for cand, scores in candidate_scores.items():
#                 mean_diff = np.mean(scores) - baseline
#                 std_diff = np.std(scores)
#                 candidate_snr[cand] = mean_diff / (std_diff + epsilon)
#
#             best_candidate = max(eligible_candidates, key=lambda cand: candidate_snr[cand])
#             best_avg_score = np.mean(candidate_scores[best_candidate])
#             selection_sequence.append(best_candidate)
#             if selection_direction == "backward":
#                 current_selection.remove(best_candidate)
#             else:
#                 current_selection.append(best_candidate)
#             rep_validation_accuracies.append(best_avg_score)
#
#             X_train_subset = get_mean_chromatogram(X_train_val, current_selection)
#             X_test_subset = get_mean_chromatogram(X_test, current_selection)
#             if normalize:
#                 X_train_subset,  X_test_subset = scale_data(X_train_subset, X_test_subset)
#             model = RidgeClassifier(alpha=alpha)
#             model.fit(X_train_subset, y_train_val)
#             y_pred = model.predict(X_test_subset)
#             rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#
#             baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_candidate_baseline)(current_selection,
#                                                      X_train_val[inner_train_idx],
#                                                      y_train_val[inner_train_idx],
#                                                      X_train_val[inner_val_idx],
#                                                      y_train_val[inner_val_idx],
#                                                      normalize)
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#             baseline = np.mean(baseline_fold_scores)
#
#         all_selected_channels.append(selection_sequence.copy())
#         all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
#         all_test_accuracy_per_step.append(rep_test_accuracies.copy())
#
#         num_reps = len(all_test_accuracy_per_step)
#         min_steps = min(len(rep) for rep in all_test_accuracy_per_step)
#         global_val = []
#         global_val_std = []
#         global_test = []
#         global_test_std = []
#         for step in range(min_steps):
#             val_accs = [all_validation_accuracy_per_step[r][step] for r in range(num_reps)]
#             test_accs = [all_test_accuracy_per_step[r][step] for r in range(num_reps)]
#             global_val.append(np.mean(val_accs))
#             global_val_std.append(np.std(val_accs))
#             global_test.append(np.mean(test_accs))
#             global_test_std.append(np.std(test_accs))
#
#         final_selected_channels = []
#         final_selected_channels_counts = []
#         max_steps = max(len(seq) for seq in all_selected_channels)
#         for step in range(max_steps):
#             candidate_info = []
#             for rep in range(len(all_selected_channels)):
#                 if step < len(all_selected_channels[rep]):
#                     candidate_info.append(all_selected_channels[rep][step])
#             if candidate_info:
#                 best_channel, count = Counter(candidate_info).most_common(1)[0]
#                 final_selected_channels.append(best_channel)
#                 final_selected_channels_counts.append(count)
#
#         ax.clear()
#         steps_axis = np.arange(1, min_steps + 1)
#         ax.plot(steps_axis, np.array(global_val), marker='s', linestyle='-', label='Validation Accuracy')
#         ax.fill_between(steps_axis,
#                         np.array(global_val) - np.array(global_val_std),
#                         np.array(global_val) + np.array(global_val_std), alpha=0.2)
#         ax.plot(steps_axis, np.array(global_test), marker='o', linestyle='--', label='Test Accuracy')
#         ax.fill_between(steps_axis,
#                         np.array(global_test) - np.array(global_test_std),
#                         np.array(global_test) + np.array(global_test_std), alpha=0.2)
#         if selection_direction == "backward":
#             ax.set_xlabel("Number of Removed Channels")
#         else:
#             ax.set_xlabel("Number of Added Channels")
#         ax.set_ylabel("Balanced Accuracy")
#         ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
#         ax.legend()
#         ax.grid()
#
#         for i, ch in enumerate(final_selected_channels):
#             count = final_selected_channels_counts[i]
#             # Annotate the channel number with a larger font.
#             ax.annotate(f"{ch}", (steps_axis[i], global_test[i]),
#                         textcoords="offset points", xytext=(0, -15),
#                         ha="center", fontsize=8, color='black')
#             # Annotate the count (in parentheses) with a smaller font, slightly lower.
#             ax.annotate(f"({count})", (steps_axis[i], global_test[i]),
#                         textcoords="offset points", xytext=(0, -30),
#                         ha="center", fontsize=6, color='black')
#         plt.draw()
#         plt.pause(5.0)
#
#     plt.ioff()
#     plt.show()
#
#     return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step


def greedy_nested_cv_channel_selection_3d(
        data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
        max_channels=40, normalize=False, scaler_type='standard', random_seed=42,
        parallel=True, n_jobs=-1, min_frequency=3, selection_direction="backward",
        aggregation_method="average"):
    """
    Perform nested CV for greedy channel selection using SNR‐based candidate evaluation,
    adapted for 3D GCMS data where each sample is of shape
    (n_samples, n_timepoints, n_channels) (channels in the last dimension).

    Classification is performed on an aggregated version of the data computed over the selected channels.
    The aggregation can be performed by either averaging (default) or concatenating the channels.

    Parameters
    ----------
    data : array-like, shape (n_samples, n_timepoints, n_channels)
        The input data.
    labels : array-like, shape (n_samples,)
        The corresponding labels.
    alpha : float, default=1.0
        Regularization parameter for RidgeClassifier.
    num_outer_repeats : int, default=3
        Number of outer CV repetitions.
    inner_cv_folds : int, default=3
        Number of inner CV folds.
    max_channels : int, default=40
        Maximum number of selection steps.
    normalize : bool, default=False
        Whether to apply scaling (not used for 3D data in this version).
    scaler_type : str, default='standard'
        Either 'standard' or 'minmax' (not used for 3D data).
    random_seed : int, default=42
        Random seed for reproducibility.
    parallel : bool, default=True
        Whether to use parallel processing.
    n_jobs : int, default=-1
        Number of jobs for parallel processing.
    min_frequency : int, default=3
        Minimum frequency required for a candidate to be eligible.
    selection_direction : str, default="backward"
        Either "backward" for elimination or "forward" for selection.
    aggregation_method : str, default="average"
        Method for aggregating selected channels. Either "average" (compute the mean chromatogram)
        or "concatenate" (concatenate channels along the time axis).

    Returns
    -------
    all_selected_channels : list
        A list (one per outer repetition) of selection sequences (channels removed or added in order).
    all_test_accuracy_per_step : list
        A list (one per outer repetition) of outer test accuracies (one per selection step).
    all_validation_accuracy_per_step : list
        A list (one per outer repetition) of inner CV validation accuracies (one per selection step).
    """
    import numpy as np
    from sklearn.linear_model import RidgeClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    from collections import Counter

    def scale_data(X_train, X_test, scaler_type='standard'):
        if scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaler type. Choose 'standard' or 'minmax'.")
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    # --- Helper function: Aggregate channels ---
    def aggregate_channels(X, channels, method):
        """
        Given X of shape (n_samples, n_timepoints, n_channels), aggregate the data
        from the given channels according to the specified method.

        For "average", compute the mean over the selected channels (resulting in shape (n_samples, n_timepoints)).
        For "concatenate", flatten the selected channels along the time axis
           (resulting in shape (n_samples, n_timepoints * len(channels))).
        """
        if len(channels) == 0:
            return None  # Caller should handle the case of an empty selection.
        if method == "average":
            return np.mean(X[:, :, channels], axis=2)
        elif method == "concatenate":
            # X[:, :, channels] has shape (n_samples, n_timepoints, len(channels))
            # Reshape to (n_samples, n_timepoints * len(channels))
            return X[:, :, channels].reshape(X.shape[0], -1)
        else:
            raise ValueError("Invalid aggregation_method. Choose 'average' or 'concatenate'.")

    # --- Custom CV classes (unchanged) ---
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

    # --- Optimized Candidate Evaluation ---
    def evaluate_channel_candidate(cand, current_selection, X_train, y_train, X_val, y_val,
                                   direction, current_sum_train, current_count_train,
                                   current_sum_val, current_count_val, alpha=1.0, aggregation_method="average"):
        """
        Evaluate candidate 'cand' by updating the current selection.

        For aggregation_method "average", a cumulative sum trick is used.
        For "concatenate", the aggregated feature matrix is recomputed directly.
        """
        if aggregation_method == "average":
            # Use cumulative sums if available; if not, compute from scratch.
            if current_sum_train is None or current_count_train is None or current_sum_val is None or current_count_val is None:
                if len(current_selection) > 0:
                    current_sum_train = np.sum(X_train[:, :, current_selection], axis=2)
                    current_count_train = len(current_selection)
                    current_sum_val = np.sum(X_val[:, :, current_selection], axis=2)
                    current_count_val = len(current_selection)
                else:
                    current_sum_train = np.zeros((X_train.shape[0], X_train.shape[1]))
                    current_count_train = 0
                    current_sum_val = np.zeros((X_val.shape[0], X_val.shape[1]))
                    current_count_val = 0

            # If no channels have been selected yet, use the candidate channel's data directly.
            if current_count_train == 0 or current_count_val == 0:
                new_mean_train = X_train[:, :, cand]
                new_mean_val = X_val[:, :, cand]
            else:
                if direction == "backward":
                    new_sum_train = current_sum_train - X_train[:, :, cand]
                    new_count_train = current_count_train - 1
                    new_sum_val = current_sum_val - X_val[:, :, cand]
                    new_count_val = current_count_val - 1
                elif direction == "forward":
                    new_sum_train = current_sum_train + X_train[:, :, cand]
                    new_count_train = current_count_train + 1
                    new_sum_val = current_sum_val + X_val[:, :, cand]
                    new_count_val = current_count_val + 1
                else:
                    raise ValueError("Direction must be 'backward' or 'forward'")
                new_mean_train = new_sum_train / new_count_train
                new_mean_val = new_sum_val / new_count_val
        elif aggregation_method == "concatenate":
            # For concatenation, recompute the aggregated features directly.
            if direction == "backward":
                new_selection = [ch for ch in current_selection if ch != cand]
            elif direction == "forward":
                new_selection = current_selection + [cand]
            else:
                raise ValueError("Direction must be 'backward' or 'forward'")
            new_mean_train = aggregate_channels(X_train, new_selection, aggregation_method)
            new_mean_val = aggregate_channels(X_val, new_selection, aggregation_method)
        else:
            raise ValueError("Invalid aggregation_method. Choose 'average' or 'concatenate'.")

        model = RidgeClassifier(alpha=alpha)
        model.fit(new_mean_train, y_train)
        y_pred = model.predict(new_mean_val)
        return cand, balanced_accuracy_score(y_val, y_pred)

    def evaluate_inner_fold_selection(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                                      current_selection, candidate_pool, alpha, direction,
                                      aggregation_method="average"):
        """
        Evaluate one inner CV fold for channel selection.
        Returns (best_candidate, best_accuracy) for that fold.
        """
        X_inner_train = X_train_val[inner_train_idx]
        X_inner_val = X_train_val[inner_val_idx]
        y_inner_train = y_train_val[inner_train_idx]
        y_inner_val = y_train_val[inner_val_idx]
        if X_inner_train.ndim == 3:
            if aggregation_method == "average" and len(current_selection) > 0:
                current_sum_train = np.sum(X_inner_train[:, :, current_selection], axis=2)
                current_sum_val = np.sum(X_inner_val[:, :, current_selection], axis=2)
                current_count_train = len(current_selection)
                current_count_val = len(current_selection)
            else:
                current_sum_train = None
                current_sum_val = None
                current_count_train = len(current_selection)
                current_count_val = len(current_selection)
        else:
            current_sum_train = None
            current_sum_val = None
            current_count_train = len(current_selection)
            current_count_val = len(current_selection)
        results = []
        for cand in candidate_pool:
            # Special case: forward selection when no channel has been selected yet.
            if direction == "forward" and current_count_train == 0:
                new_mean_train = aggregate_channels(X_inner_train, [cand], aggregation_method)
                new_mean_val = aggregate_channels(X_inner_val, [cand], aggregation_method)
                model = RidgeClassifier(alpha=alpha)
                model.fit(new_mean_train, y_inner_train)
                y_pred = model.predict(new_mean_val)
                score = balanced_accuracy_score(y_inner_val, y_pred)
                results.append((cand, score))
            else:
                res = evaluate_channel_candidate(cand, current_selection, X_inner_train, y_inner_train,
                                                 X_inner_val, y_inner_val, direction,
                                                 current_sum_train, current_count_train,
                                                 current_sum_val, current_count_val, alpha=alpha,
                                                 aggregation_method=aggregation_method)
                results.append(res)

        # Add a tiny random jitter to each score.
        jitter = np.random.rand(len(results)) * 1e-10
        results_jittered = [(cand, score + j) for ((cand, score), j) in zip(results, jitter)]

        best_candidate, best_accuracy = max(results_jittered, key=lambda x: x[1])
        return best_candidate, best_accuracy

    def evaluate_candidate_baseline(current_selection, X_train, y_train, X_val, y_val,
                                    normalize, aggregation_method="average"):
        if len(current_selection) == 0:
            return 0
        X_train_subset = aggregate_channels(X_train, current_selection, aggregation_method)
        X_val_subset = aggregate_channels(X_val, current_selection, aggregation_method)
        if normalize:
            X_train_subset, X_val_subset = scale_data(X_train_subset, X_val_subset)
        model = RidgeClassifier(alpha=alpha)
        model.fit(X_train_subset, y_train)
        y_pred = model.predict(X_val_subset)
        return balanced_accuracy_score(y_val, y_pred)

    # --- Set random seed and prepare accumulators ---
    rng = np.random.default_rng(random_seed)
    all_selected_channels = []
    all_test_accuracy_per_step = []
    all_validation_accuracy_per_step = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))

    # --- Outer CV Loop ---
    for repeat in range(num_outer_repeats):
        print(f"\n🔁 Outer CV Repetition {repeat + 1}/{num_outer_repeats}")
        outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                          random_state=rng.integers(0, int(1e6)))
        train_val_idx, test_idx = next(outer_cv.split(data, labels))
        X_train_val, X_test = data[train_val_idx], data[test_idx]
        # For 3D data, we skip scaling.
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
                                                   random_state=rng.integers(0, int(1e6)))
        if selection_direction == "backward":
            current_selection = list(range(data.shape[2]))  # channels are in the last dimension
        elif selection_direction == "forward":
            current_selection = []
        else:
            raise ValueError("selection_direction must be either 'backward' or 'forward'")
        selection_sequence = []
        rep_validation_accuracies = []
        rep_test_accuracies = []

        baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_candidate_baseline)(current_selection,
                                                 X_train_val[inner_train_idx],
                                                 y_train_val[inner_train_idx],
                                                 X_train_val[inner_val_idx],
                                                 y_train_val[inner_val_idx],
                                                 normalize,
                                                 aggregation_method)
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
        )
        baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0

        for step in range(max_channels):
            print(f"Selecting channel {step + 1}/{max_channels}")
            if selection_direction == "backward":
                candidate_pool = current_selection.copy()
            else:
                candidate_pool = list(set(range(data.shape[2])) - set(current_selection))
            if not candidate_pool:
                break

            fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_inner_fold_selection)(
                    inner_train_idx, inner_val_idx, X_train_val, y_train_val,
                    current_selection, candidate_pool, alpha, selection_direction,
                    aggregation_method=aggregation_method
                )
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )
            best_candidate_per_fold = [result[0] for result in fold_results]
            freq_dict = {cand: best_candidate_per_fold.count(cand) for cand in candidate_pool}
            eligible_candidates = [cand for cand in candidate_pool if freq_dict.get(cand, 0) >= min_frequency]
            if not eligible_candidates:
                eligible_candidates = candidate_pool

            inner_folds = list(inner_cv.split(X_train_val, y_train_val))
            candidate_scores = {}
            for cand in candidate_pool:
                fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(evaluate_channel_candidate)(cand, current_selection,
                                                        X_train_val[inner_train_idx],
                                                        y_train_val[inner_train_idx],
                                                        X_train_val[inner_val_idx],
                                                        y_train_val[inner_val_idx],
                                                        selection_direction,
                                                        None, None, None, None,
                                                        aggregation_method=aggregation_method)
                    for inner_train_idx, inner_val_idx in inner_folds
                )
                scores = np.array([score for _, score in fold_scores])
                candidate_scores[cand] = scores

            epsilon = 1e-8
            candidate_snr = {}
            for cand, scores in candidate_scores.items():
                mean_diff = np.mean(scores) - baseline
                std_diff = np.std(scores)
                candidate_snr[cand] = mean_diff / (std_diff + epsilon)

            best_candidate = max(eligible_candidates, key=lambda cand: candidate_snr[cand])
            best_avg_score = np.mean(candidate_scores[best_candidate])
            selection_sequence.append(best_candidate)
            if selection_direction == "backward":
                current_selection.remove(best_candidate)
            else:
                current_selection.append(best_candidate)
            rep_validation_accuracies.append(best_avg_score)

            X_train_subset = aggregate_channels(X_train_val, current_selection, aggregation_method)
            X_test_subset = aggregate_channels(X_test, current_selection, aggregation_method)
            if normalize:
                X_train_subset, X_test_subset = scale_data(X_train_subset, X_test_subset)
            model = RidgeClassifier(alpha=alpha)
            model.fit(X_train_subset, y_train_val)
            y_pred = model.predict(X_test_subset)
            rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))

            baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
                delayed(evaluate_candidate_baseline)(current_selection,
                                                     X_train_val[inner_train_idx],
                                                     y_train_val[inner_train_idx],
                                                     X_train_val[inner_val_idx],
                                                     y_train_val[inner_val_idx],
                                                     normalize,
                                                     aggregation_method)
                for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
            )
            baseline = np.mean(baseline_fold_scores)

        all_selected_channels.append(selection_sequence.copy())
        all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
        all_test_accuracy_per_step.append(rep_test_accuracies.copy())

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
        final_selected_channels_counts = []
        max_steps = max(len(seq) for seq in all_selected_channels)
        for step in range(max_steps):
            candidate_info = []
            for rep in range(len(all_selected_channels)):
                if step < len(all_selected_channels[rep]):
                    candidate_info.append(all_selected_channels[rep][step])
            if candidate_info:
                best_channel, count = Counter(candidate_info).most_common(1)[0]
                final_selected_channels.append(best_channel)
                final_selected_channels_counts.append(count)

        ax.clear()
        steps_axis = np.arange(1, min_steps + 1)
        ax.plot(steps_axis, np.array(global_val), marker='s', linestyle='-', label='Validation Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_val) - np.array(global_val_std),
                        np.array(global_val) + np.array(global_val_std), alpha=0.2)
        ax.plot(steps_axis, np.array(global_test), marker='o', linestyle='--', label='Test Accuracy')
        ax.fill_between(steps_axis,
                        np.array(global_test) - np.array(global_test_std),
                        np.array(global_test) + np.array(global_test_std), alpha=0.2)
        if selection_direction == "backward":
            ax.set_xlabel("Number of Removed Channels")
        else:
            ax.set_xlabel("Number of Added Channels")
        ax.set_ylabel("Balanced Accuracy")
        ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
        ax.legend()
        ax.grid()

        for i, ch in enumerate(final_selected_channels):
            count = final_selected_channels_counts[i]
            ax.annotate(f"{ch}", (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=8, color='black')
            ax.annotate(f"({count})", (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, -30),
                        ha="center", fontsize=6, color='black')
        plt.draw()
        plt.pause(5.0)

    plt.ioff()
    plt.show()

    return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step


def bin_time_dimension_max(data, n_bins, time_axis=1):
    """
    Bin the time dimension of a 3D array into n_bins by taking the maximum value over groups
    of consecutive time points along the specified time axis.

    Parameters
    ----------
    data : np.ndarray
        A 3D array with shape (n_samples, n_timepoints, n_channels).
    n_bins : int
        The desired number of bins along the time axis.
    time_axis : int, default=1
        The axis that corresponds to time.

    Returns
    -------
    binned_data : np.ndarray
        The binned data with shape (n_samples, n_bins, n_channels).
        (The time axis is reduced to n_bins.)
    """
    # First, bring the time axis to a fixed position (say axis=1) if it's not already there.
    if time_axis != 1:
        # Permute axes so that time is axis=1.
        axes = list(range(data.ndim))
        axes.remove(time_axis)
        axes = [time_axis] + axes
        data = np.transpose(data, axes)

    n_samples, n_timepoints, n_channels = data.shape
    bin_size = n_timepoints // n_bins
    truncated_timepoints = n_bins * bin_size
    data_truncated = data[:, :truncated_timepoints, :]
    # Now reshape the time dimension into (n_bins, bin_size) and take the max over bin_size.
    binned_data = data_truncated.reshape(n_samples, n_bins, bin_size, n_channels).max(axis=2)
    return binned_data


def reduce_time_dimension_middle(data, n_bins=500):
    """
    Reduce the time dimension of 3D GC-MS data by selecting the middle value in each bin.

    Parameters:
    ----------
    data : np.ndarray, shape (n_samples, n_timepoints, n_channels)
        The input GC-MS data.
    n_bins : int
        The desired number of bins along the time axis.

    Returns:
    -------
    np.ndarray
        Data with reduced time dimension (n_samples, n_bins, n_channels).
    """
    n_samples, n_timepoints, n_channels = data.shape

    if n_timepoints < n_bins:
        print(f"⚠️ Warning: n_timepoints ({n_timepoints}) < n_bins ({n_bins}). Returning original data.")
        return data

    bin_size = n_timepoints // n_bins
    truncated_timepoints = bin_size * n_bins
    data_truncated = data[:, :truncated_timepoints, :]

    # Select the middle index in each bin
    mid_idx = bin_size // 2
    return data_truncated.reshape(n_samples, n_bins, bin_size, n_channels)[:, :, mid_idx, :]

N_DECIMATION = 5
DATA_TYPE = "GCMS"

# Set rows in columns to read
row_start = 1
row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(DATA_DIRECTORY)
column_indices = list(range(fc_idx, lc_idx + 1))
data_dict = utils.load_ms_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end)
min_length = min(array.shape[0] for array in data_dict.values())
data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}

# import scipy.signal as signal
# data_dict = {key: signal.decimate(matrix, 20, axis=0, zero_phase=True) for key, matrix in data_dict.items()}

if DATA_TYPE == "TIS":
    data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}
    gcms = GCMSDataProcessor(data_dict)
    chromatograms = gcms.compute_tiss()
else:
    chromatograms = data_dict

data = chromatograms.values()
labels = chromatograms.keys()

if WINE_KIND == "bordeaux":
    region = 'bordeaux_chateaux'
    labels = process_labels(labels, vintage=VINTAGE)
elif WINE_KIND == "pinot_noir":
    region = REGION
    # region = 'origin'
    if region == 'continent':
        labels = assign_continent_to_pinot_noir(labels)
    elif region == 'country':
        labels = assign_country_to_pinot_noir(labels)
    elif region == 'origin':
        labels = assign_origin_to_pinot_noir(labels)
    elif region == 'winery':
        labels = assign_winery_to_pinot_noir(labels)
    elif region == 'year':
        labels = assign_year_to_pinot_noir(labels)
    elif region == 'beaume':
        labels = assign_north_south_to_beaune(labels)
    else:
        raise ValueError("Invalid region. Options are 'continent', 'country', 'origin', 'winery', or 'year'")

# Convert dictionary to array
data = np.array(list(data))
labels = np.array(labels)

if DATA_TYPE == "TIS":
    greedy_nested_cv_channel_selection_tis(
            data, labels, alpha=1.0, num_outer_repeats=100, inner_cv_folds=100,
            max_channels=180, normalize=True, scaler_type='standard', random_seed=42,
            parallel=True, n_jobs=100, min_frequency=3, selection_direction='backward')
elif DATA_TYPE == "GCMS":
    # data = bin_time_dimension_max(data, 100)
    data = reduce_time_dimension_middle(data, 1000)
    # from scipy.signal import decimate
    # data = decimate(data, q=10, axis=1, zero_phase=True)
    greedy_nested_cv_channel_selection_3d(
            data, labels, alpha=1.0, num_outer_repeats=50, inner_cv_folds=20,
            max_channels=180, normalize=True, scaler_type='standard', random_seed=42,
            parallel=True, n_jobs=20, min_frequency=3, selection_direction='forward', aggregation_method='concatenate')


