from config import (DATA_DIRECTORY, N_DECIMATION, REGION, VINTAGE, WINE_KIND)
from classification import (
    Classifier, process_labels, assign_country_to_pinot_noir, assign_origin_to_pinot_noir,
    assign_continent_to_pinot_noir, assign_winery_to_pinot_noir, assign_year_to_pinot_noir,
    assign_north_south_to_beaune
)
from wine_analysis import GCMSDataProcessor
import utils
import numpy as np


# def greedy_nested_cv_channel_elimination(
#         data, labels, alpha=1.0, num_outer_repeats=3, inner_cv_folds=3,
#         max_channels=40, normalize=True, scaler_type='standard', random_seed=None,
#         parallel=True, n_jobs=-1, min_frequency=3):
#     """
#     Perform nested CV for greedy channel elimination using SNR‐based candidate evaluation.
#     (Backward elimination: start with all channels and, at each step, remove the channel whose removal
#     yields the best (or least harmed) validation accuracy.)
#
#     The algorithm uses an outer CV loop (each repetition is a new stratified split) and, within each repetition,
#     an inner CV loop to evaluate the performance when each candidate channel (from the current selection)
#     is removed. A candidate’s performance is aggregated over inner folds, and candidates that appear in fewer
#     than min_frequency folds are discarded (for robustness). The candidate with the highest improvement over
#     the baseline (current selection) is removed. After each elimination step the outer test accuracy is evaluated.
#     Finally, the function dynamically plots the global (averaged across outer repetitions) validation and test
#     accuracy progression after each outer repetition. The candidate channel selection uses its own validation SNR,
#     and the final plot is annotated with the channel (aggregated across repetitions) that produced the best SNR,
#     along with the number of repetitions that channel appeared.
#
#     Parameters
#     ----------
#     data : array-like, shape (n_samples, n_channels)
#         The data matrix.
#     labels : array-like, shape (n_samples,)
#         The corresponding labels.
#     alpha : float, default=1.0
#         Regularization parame for RidgeClassifier.
#     num_outer_repeats : int, default=3
#         Number of outer repetitions (each using a new stratified train/test split).
#     inner_cv_folds : int, default=3
#         Number of inner CV repeats (using a custom leave-one-per-class splitter).
#     max_channels : int, default=40
#         Maximum number of elimination steps.
#     normalize : bool, default=True
#         Whether to normalize the data.
#     scaler_type : str, default='standard'
#         Either 'standard' or 'minmax' to specify the scaler.
#     random_seed : int, default=None
#         Random seed for reproducibility.
#     parallel : bool, default=True
#         Whether to use parallel processing for inner folds.
#     n_jobs : int, default=-1
#         Number of jobs for parallel processing.
#     min_frequency : int, default=3
#         Minimum frequency required for a candidate (across inner folds) to be eligible.
#
#     Returns
#     -------
#     all_selected_channels : list
#         A list (one element per outer repetition) of removal sequences (i.e. the channels removed in order).
#     all_test_accuracy_per_step : list
#         A list (one element per outer repetition) of lists of outer test accuracies (one per elimination step).
#     all_validation_accuracy_per_step : list
#         A list (one element per outer repetition) of lists of inner CV validation accuracies (one per elimination step).
#     """
#     import numpy as np
#     from sklearn.linear_model import RidgeClassifier
#     from sklearn.metrics import balanced_accuracy_score
#     from sklearn.model_selection import StratifiedShuffleSplit, BaseCrossValidator
#     from sklearn.preprocessing import StandardScaler, MinMaxScaler
#     from joblib import Parallel, delayed
#     import matplotlib.pyplot as plt
#     from collections import Counter
#
#     # Set random seed if not provided.
#     if random_seed is None:
#         random_seed = np.random.randint(0, int(1e6))
#     rng = np.random.default_rng(random_seed)
#
#     # Global accumulators.
#     all_selected_channels = []          # One elimination sequence per outer repetition.
#     all_test_accuracy_per_step = []       # Outer test accuracies per elimination step.
#     all_validation_accuracy_per_step = [] # Inner CV validation accuracies per elimination step.
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
#     # Helper functions for elimination.
#     # ----------------------------------
#     def evaluate_channel_elimination(ch, selected_channels, X_train, y_train, X_val, y_val):
#         """
#         Evaluate performance when channel 'ch' is removed from the current selection.
#         For 2D data, simply remove the column.
#         """
#         candidate_channels = [x for x in selected_channels if x != ch]
#         X_train_subset = X_train[:, candidate_channels]
#         X_val_subset = X_val[:, candidate_channels]
#         model = RidgeClassifier(alpha=alpha)
#         model.fit(X_train_subset, y_train)
#         y_pred = model.predict(X_val_subset)
#         return ch, balanced_accuracy_score(y_val, y_pred)
#
#     def evaluate_inner_fold_elimination(inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                                         selected_channels, candidate_pool, alpha):
#         """
#         Evaluate one inner CV fold for elimination.
#         For 2D data, simply select the candidate subset.
#         Returns (best_candidate, best_accuracy) for that fold.
#         """
#         X_inner_train = X_train_val[inner_train_idx]
#         X_inner_val = X_train_val[inner_val_idx]
#         y_inner_train = y_train_val[inner_train_idx]
#         y_inner_val = y_train_val[inner_val_idx]
#
#         results = [evaluate_channel_elimination(ch, selected_channels, X_inner_train, y_inner_train,
#                                                 X_inner_val, y_inner_val)
#                    for ch in candidate_pool]
#         best_candidate, best_accuracy = max(results, key=lambda x: x[1])
#         return best_candidate, best_accuracy
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
#     def evaluate_candidate_baseline(current_selection, X_train, y_train, X_val, y_val):
#         """
#         Evaluate the baseline performance (accuracy) using the current selection of channels.
#         If current_selection is empty, returns 0.
#         """
#         if len(current_selection) == 0:
#             return 0
#         X_train_subset = X_train[:, current_selection]
#         X_val_subset = X_val[:, current_selection]
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
#         current_selection = list(range(data.shape[1]))
#         selection_sequence = []          # Record the channel removed at each step.
#         rep_validation_accuracies = []   # Inner CV (validation) accuracy per elimination step.
#         rep_test_accuracies = []         # Outer test accuracy per elimination step.
#
#         # Compute baseline performance using the current selection.
#         baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#             delayed(evaluate_candidate_baseline)(current_selection,
#                                                  X_train_val[inner_train_idx],
#                                                  y_train_val[inner_train_idx],
#                                                  X_train_val[inner_val_idx],
#                                                  y_train_val[inner_val_idx])
#             for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#         )
#         baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0
#
#         for step in range(max_channels):
#             print(f"Selecting channel {step + 1}/{max_channels}")
#             candidate_pool = current_selection.copy()
#             if not candidate_pool:
#                 break
#
#             # Evaluate each candidate on the inner folds in parallel.
#             fold_results = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_inner_fold_elimination)(
#                     inner_train_idx, inner_val_idx, X_train_val, y_train_val,
#                     current_selection, candidate_pool, alpha
#                 )
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#             best_candidate_per_fold = [result[0] for result in fold_results]
#             # Compute frequency for each candidate across inner folds.
#             freq_dict = {cand: best_candidate_per_fold.count(cand) for cand in candidate_pool}
#             candidates_meeting_freq = [cand for cand in candidate_pool if freq_dict[cand] >= min_frequency]
#             eligible = candidates_meeting_freq if candidates_meeting_freq else candidate_pool
#
#             # ---- Validation SNR Computation for Candidate Selection ----
#             # For each candidate, re-run the evaluation over inner folds to get an array of validation accuracies.
#             inner_folds = list(inner_cv.split(X_train_val, y_train_val))
#             candidate_scores = {}  # Will hold an array of scores for each candidate.
#             for cand in candidate_pool:
#                 fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                     delayed(evaluate_channel_elimination)(cand, current_selection,
#                                                           X_train_val[inner_train_idx],
#                                                           y_train_val[inner_train_idx],
#                                                           X_train_val[inner_val_idx],
#                                                           y_train_val[inner_val_idx])
#                     for inner_train_idx, inner_val_idx in inner_folds
#                 )
#                 # Use the scores from each fold.
#                 scores = np.array([score for _, score in fold_scores])
#                 candidate_scores[cand] = scores
#
#             # Compute the validation SNR for each candidate.
#             # Here, SNR = (mean(candidate score) - baseline) / (std(candidate score) + epsilon)
#             epsilon = 1e-8  # To avoid division by zero.
#             candidate_snr = {}
#             for cand, scores in candidate_scores.items():
#                 mean_diff = np.mean(scores) - baseline
#                 std_diff = np.std(scores)
#                 candidate_snr[cand] = mean_diff / (std_diff + epsilon)
#
#             # Among the eligible candidates (based on min_frequency), select the one with highest validation SNR.
#             eligible_candidates = [cand for cand in candidate_pool if freq_dict.get(cand, 0) >= min_frequency]
#             if not eligible_candidates:
#                 eligible_candidates = candidate_pool
#             best_candidate = max(eligible_candidates, key=lambda cand: candidate_snr[cand])
#             best_avg_score = np.mean(candidate_scores[best_candidate])
#             selection_sequence.append(best_candidate)
#             current_selection.remove(best_candidate)
#             rep_validation_accuracies.append(best_avg_score)
#             # -------------------------------------------------------------
#
#             # Evaluate outer test accuracy using the remaining channels.
#             X_train_subset = X_train_val[:, current_selection]
#             X_test_subset = X_test[:, current_selection]
#             model.fit(X_train_subset, y_train_val)
#             y_pred = model.predict(X_test_subset)
#             rep_test_accuracies.append(balanced_accuracy_score(y_test, y_pred))
#
#             # Update baseline based on the current selection.
#             baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
#                 delayed(evaluate_candidate_baseline)(current_selection,
#                                                      X_train_val[inner_train_idx],
#                                                      y_train_val[inner_train_idx],
#                                                      X_train_val[inner_val_idx],
#                                                      y_train_val[inner_val_idx])
#                 for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
#             )
#             baseline = np.mean(baseline_fold_scores)
#
#         # End of elimination loop for this outer repetition.
#         all_selected_channels.append(selection_sequence.copy())
#         all_validation_accuracy_per_step.append(rep_validation_accuracies.copy())
#         all_test_accuracy_per_step.append(rep_test_accuracies.copy())
#
#         # ----- Dynamic Global Aggregation and Plotting After Each Outer Repetition -----
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
#         # ---- Aggregate the Eliminated Channels Across Outer Repetitions ----
#         # For each elimination step, we choose the channel that appears most frequently across outer repetitions
#         # and record how many repetitions selected that channel.
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
#         # ---------------------------------------------------------------------
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
#         ax.set_xlabel("Number of Selected Channels")
#         ax.set_ylabel("Balanced Accuracy")
#         ax.set_title(f"Average Accuracy Progression Across Outer Repetitions (After {num_reps} Repetitions)")
#         ax.legend()
#         ax.grid()
#
#         # Annotate each elimination step with the channel that produced the best validation SNR
#         # and, on the line below, the frequency that channel appeared across repeats.
#         for i, ch in enumerate(final_selected_channels):
#             count = final_selected_channels_counts[i]
#             annotation_text = f"Ch: {ch}\n({count})"
#             ax.annotate(annotation_text, (steps_axis[i], global_test[i]),
#                         textcoords="offset points", xytext=(0, -15),
#                         ha="center", fontsize=6, color='red')
#         plt.draw()
#         plt.pause(1.0)
#         # End dynamic global plotting for this repetition.
#
#     plt.ioff()
#     plt.show()
#
#     return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step

def greedy_nested_cv_channel_selection(
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
                                      current_selection, candidate_pool, alpha, direction):
        """
        Evaluate one inner CV fold for selection.
        Returns (best_candidate, best_accuracy) for that fold.
        """
        X_inner_train = X_train_val[inner_train_idx]
        X_inner_val = X_train_val[inner_val_idx]
        y_inner_train = y_train_val[inner_train_idx]
        y_inner_val = y_train_val[inner_val_idx]
        results = [evaluate_channel_candidate(ch, current_selection, X_inner_train, y_inner_train,
                                              X_inner_val, y_inner_val, direction)
                   for ch in candidate_pool]
        best_candidate, best_accuracy = max(results, key=lambda x: x[1])
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
        outer_cv = StratifiedShuffleSplit(n_splits=1, test_size=0.2,
                                          random_state=rng.integers(0, int(1e6)))
        train_val_idx, test_idx = next(outer_cv.split(data, labels))
        X_train_val, X_test = data[train_val_idx], data[test_idx]
        if normalize:
            X_train_val, X_test = scale_data(X_train_val, X_test, scaler_type)
        y_train_val, y_test = labels[train_val_idx], labels[test_idx]

        inner_cv = RepeatedLeaveOneFromEachClassCV(n_repeats=inner_cv_folds, shuffle=True,
                                                   random_state=rng.integers(0, int(1e6)))

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

        # Compute baseline performance using the current selection.
        baseline_fold_scores = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(evaluate_candidate_baseline)(current_selection,
                                                 X_train_val[inner_train_idx],
                                                 y_train_val[inner_train_idx],
                                                 X_train_val[inner_val_idx],
                                                 y_train_val[inner_val_idx])
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train_val, y_train_val)
        )
        baseline = np.mean(baseline_fold_scores) if baseline_fold_scores else 0

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
                    current_selection, candidate_pool, alpha, selection_direction
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

        # Annotate each selection step with the channel and its frequency (number of repetitions).
        for i, ch in enumerate(final_selected_channels):
            count = final_selected_channels_counts[i]
            annotation_text = f"{ch}\n({count} )"
            ax.annotate(annotation_text, (steps_axis[i], global_test[i]),
                        textcoords="offset points", xytext=(0, -15),
                        ha="center", fontsize=7, color='black')
        plt.draw()
        plt.pause(1.0)
        # End dynamic global plotting for this outer repetition.

    plt.ioff()
    plt.show()

    return all_selected_channels, all_test_accuracy_per_step, all_validation_accuracy_per_step


N_DECIMATION = 5
DATA_TYPE = "TIS"

# Set rows in columns to read
row_start = 1
row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(DATA_DIRECTORY)
column_indices = list(range(fc_idx, lc_idx + 1))
data_dict = utils.load_ms_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end)
min_length = min(array.shape[0] for array in data_dict.values())
data_dict = {key: array[:min_length, :] for key, array in data_dict.items()}
data_dict = {key: matrix[::N_DECIMATION, :] for key, matrix in data_dict.items()}
gcms = GCMSDataProcessor(data_dict)

chromatograms = gcms.compute_tiss()
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

# greedy_nested_cv_channel_elimination(
#         data, labels, alpha=1.0, num_outer_repeats=50, inner_cv_folds=50,
#         max_channels=180, normalize=True, scaler_type='standard', random_seed=None,
#         parallel=True, n_jobs=50, min_frequency=3)

greedy_nested_cv_channel_selection(
        data, labels, alpha=1.0, num_outer_repeats=50, inner_cv_folds=50,
        max_channels=3, normalize=True, scaler_type='standard', random_seed=None,
        parallel=True, n_jobs=50, min_frequency=3, selection_direction='forward')
