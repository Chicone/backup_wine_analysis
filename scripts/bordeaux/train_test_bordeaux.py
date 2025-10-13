"""
To train and test classification of Bordeaux wines, we use the script **train_test_bordeaux.py**.
The goal is to classify Bordeaux wine samples based on their GC-MS chemical fingerprint, using either
sample-level identifiers (e.g., A2022) or vintage year labels (e.g., 2022) depending on the configuration.

The script implements a complete machine learning pipeline including data loading, label parsing,
feature extraction, classification, and repeated evaluation using replicate-safe splitting.
"""

import numpy as np
import os
import sys
import yaml
import matplotlib.pyplot as plt
from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
from gcmswine.logger_setup import logger, logger_raw
from sklearn.preprocessing import normalize
from gcmswine.dimensionality_reduction import DimensionalityReducer
from scripts.bordeaux.plotting_bordeaux import plot_bordeaux


# === Utility Functions ===
def split_into_bins(data, n_bins):
    total_points = data.shape[1]
    bin_edges = np.linspace(0, total_points, n_bins + 1, dtype=int)
    return [(bin_edges[i], bin_edges[i+1]) for i in range(n_bins)]

def remove_bins(data, bins_to_remove, bin_ranges):
    """Remove (delete) specified bins in TIC by index."""
    mask = np.ones(data.shape[1], dtype=bool)
    for b in bins_to_remove:
        start, end = bin_ranges[b]
        mask[start:end] = False
    return data[:, mask]


# === Mode A: Normal classification (no SOTF) ===
def run_classification(
    data,
    labels,
    raw_sample_labels,
    year_labels,
    classifier,
    wine_kind,
    class_by_year,
    strategy,
    dataset_origins,
    cv_type,
    num_repeats,
    normalize_flag,
    region,
    feature_type,
    projection_source,
    show_confusion_matrix,
    # projection plotting parameters
    plot_projection,
    projection_method,
    projection_dim,
    n_neighbors,
    random_state,
    color_by_country,
    invert_x,
    invert_y,
):
    """Single evaluation without SOTF masking; preserves original plotting behavior."""

    cls = Classifier(
        data,
        labels,
        classifier_type=classifier,
        wine_kind=wine_kind,
        class_by_year=class_by_year,
        year_labels=np.array(year_labels),
        strategy=strategy,
        sample_labels=raw_sample_labels,
        dataset_origins=dataset_origins,
    )

    if cv_type in ["LOOPC", "stratified"]:
        loopc = (cv_type == "LOOPC")
        mean_acc, std_acc, scores, all_labels, test_samples_names = cls.train_and_evaluate_all_channels(
            num_repeats=num_repeats,
            random_seed=42,
            test_size=0.2,
            normalize=normalize_flag,
            scaler_type='standard',
            use_pca=False,
            vthresh=0.97,
            region=region,
            print_results=False,
            n_jobs=20,
            feature_type=feature_type,
            classifier_type=classifier,
            LOOPC=loopc,
            projection_source=projection_source,
            show_confusion_matrix=show_confusion_matrix,
        )
    elif cv_type == "LOO":
        mean_acc, std_acc, scores, all_labels, test_samples_names, all_preds = cls.train_and_evaluate_leave_one_out_all_samples(
            normalize=normalize_flag,
            scaler_type='standard',
            region=region,
            feature_type=feature_type,
            classifier_type=classifier,
            projection_source=projection_source,
            show_confusion_matrix=show_confusion_matrix,
        )
    else:
        raise ValueError(f"Invalid CV type: {cv_type}")

    logger.info(f"Final Accuracy (no survival): {mean_acc:.3f}")

    # === Projection Plotting (unchanged from original) ===
    if plot_projection:
        if projection_source == "scores":
            data_for_projection = normalize(scores)
            projection_labels = all_labels
        elif projection_source in {"tic", "tis", "tic_tis"}:
            data_for_projection = utils.compute_features(data, feature_type=projection_source)
            data_for_projection = normalize(data_for_projection)
            projection_labels = year_labels if year_labels is not None else labels
        else:
            raise ValueError(f"Unknown projection source: {projection_source}")

        pretty_source = {
            "scores": "Classification Scores",
            "tic": "TIC",
            "tis": "TIS",
            "tic_tis": "TIC + TIS"
        }.get(projection_source, projection_source)

        pretty_method = {
            "UMAP": "UMAP",
            "T-SNE": "t-SNE",
            "PCA": "PCA"
        }.get(projection_method, projection_method)

        plot_title = f"{pretty_method} of {pretty_source}"
        sys.stdout.flush()

        if data_for_projection is not None:
            reducer = DimensionalityReducer(data_for_projection)
            if projection_method == "UMAP":
                plot_bordeaux(
                    reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state),
                    plot_title, projection_labels, labels, color_by_country, invert_x=invert_x, invert_y=invert_y
                )
            elif projection_method == "PCA":
                plot_bordeaux(
                    reducer.pca(components=projection_dim),
                    plot_title, projection_labels, labels, color_by_country, invert_x=invert_x, invert_y=invert_y
                )
            elif projection_method == "T-SNE":
                plot_bordeaux(
                    reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state),
                    plot_title, projection_labels, labels, color_by_country, invert_x=invert_x, invert_y=invert_y
                )
            else:
                raise ValueError(f"Unsupported projection method: {projection_method}")
        plt.show()

from collections import defaultdict
import re
def loopc_splits(
    class_labels,
    num_repeats=1,
    random_state=None,
    *,
    class_by_year=False,
    raw_labels=None,
    prefix_regex=r'^[A-Za-z]+'
):
    """
    Generate LOOPC (Leave-One-Sample-Per-Class) splits.

    Modes
    -----
    - class_by_year = True:
        class_labels = array-like of years (one per row/sample)
        raw_labels   = replicate-aware IDs (e.g. '00ML-B-11-1'), required
        -> For each year (class), select ONE *base sample* per repeat and
           put ALL its replicates in the test set.

    - class_by_year = False (default):
        class_labels = flat class labels (e.g. varieties/regions)
        -> For each class, select ONE *index* per repeat as test.

    Parameters
    ----------
    class_labels : array-like, shape (n_samples,)
        Class label for each row/sample.
    num_repeats : int, default=1
        Number of LOOPC splits to generate.
    random_state : int or None
        Seed for reproducibility.
    class_by_year : bool, default=False
        Enable year-based, replicate-aware split (requires raw_labels).
    raw_labels : array-like or None
        Replicate-aware sample IDs (e.g. '00ML-B-11-1'). Required if class_by_year=True.

    Returns
    -------
    splits : list of (train_idx, test_idx)
        Each element is a tuple of np.ndarray indices.
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(class_labels)

    def collapse_alpha_prefix(arr, regex):
        out = []
        all_had_prefix = True
        pat = re.compile(regex)
        for s in map(str, arr):
            m = pat.match(s)
            if m:
                out.append(m.group(0))
            else:
                out.append(s)
                all_had_prefix = False
        # Even if some had no prefix, we still return the mixed result
        return np.asarray(out, dtype=object)

    splits = []

    if class_by_year:
        if raw_labels is None:
            raise ValueError("loopc_splits: raw_labels must be provided when class_by_year=True")

        raw_labels = np.asarray(raw_labels)

        # Map base sample -> list of replicate indices
        base_to_indices = defaultdict(list)
        for i, sid in enumerate(raw_labels):
            s = str(sid)
            base = s.rsplit("-", 1)[0] if "-" in s else s   # strip final '-rep' if present
            base_to_indices[base].append(i)

        # Map class (year) -> list of BASES (not indices)
        class_to_bases = defaultdict(list)
        for base, idxs in base_to_indices.items():
            # assume all replicates share the same class label
            cls = y[idxs[0]]
            class_to_bases[cls].append(base)

        # Validate: need at least 2 base samples per class (so train keeps that class)
        bad = {cls: len(bases) for cls, bases in class_to_bases.items() if len(bases) < 2}
        if bad:
            detail = ", ".join(f"{cls}: {n}" for cls, n in bad.items())
            raise ValueError(f"LOOPC requires ≥2 base samples per class; found {{{detail}}}")

        # Shuffle base lists deterministically, then round-robin across repeats
        for cls, bases in class_to_bases.items():
            bases_arr = np.array(bases, dtype=object)
            rng.shuffle(bases_arr)
            class_to_bases[cls] = bases_arr

        for r in range(max(1, num_repeats)):
            # pick one BASE per class (cycle through bases per class)
            chosen_bases = []
            for cls, bases_arr in class_to_bases.items():
                chosen_bases.append(bases_arr[r % len(bases_arr)])

            # expand chosen bases into ALL their replicate indices
            test_idx = []
            for base in chosen_bases:
                test_idx.extend(base_to_indices[base])

            test_idx = np.array(test_idx, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            splits.append((train_idx, test_idx))

    else:
        y_eff = collapse_alpha_prefix(y, prefix_regex)

        # Group by transformed y (not raw labels)
        class_to_indices = defaultdict(list)
        for i, cls in enumerate(y_eff):
            class_to_indices[cls].append(i)

        bad = {cls: len(ix) for cls, ix in class_to_indices.items() if len(ix) < 2}
        if bad:
            detail = ", ".join(f"{cls}: {n}" for cls, n in bad.items())
            raise ValueError(f"LOOPC requires ≥2 samples per class; found {{{detail}}}")

        for cls, ix in class_to_indices.items():
            arr = np.array(ix, dtype=int)
            rng.shuffle(arr)
            class_to_indices[cls] = arr

        for r in range(max(1, num_repeats)):
            test_idx = [arr[r % len(arr)] for arr in class_to_indices.values()]
            test_idx = np.array(test_idx, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            splits.append((train_idx, test_idx))

    return splits

def run_sotf_ret_time(
    data,
    labels,
    raw_sample_labels,
    year_labels,
    classifier,
    wine_kind,
    class_by_year,
    strategy,
    dataset_origins,
    cv_type,
    num_repeats,
    normalize_flag,
    region,
    feature_type,
    projection_source,
    show_confusion_matrix,
    n_bins: int = 50,
    min_bins: int = 1,
    # projection plotting parameters
    plot_projection=False,
    projection_method="UMAP",
    projection_dim=2,
    n_neighbors=30,
    random_state=42,
    color_by_country=False,
    show_sample_names=False,
    invert_x=False,
    invert_y=False,
):
    """Greedy removal over m/z retention-time bins with nested CV:
    - Outer CV provides unbiased accuracy estimate.
    - Inner greedy loop removes bins using all folds (global decision).
    """

    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    import numpy as np
    import matplotlib.pyplot as plt

    from gcmswine.utils import compute_features

    # === Preprocess features ===
    X_proc = compute_features(data, feature_type=feature_type)
    bin_ranges = split_into_bins(X_proc, n_bins)
    active_bins = list(range(n_bins))
    n_iterations = n_bins - min_bins + 1

    # Baseline TIC size
    baseline_nonzero = np.count_nonzero(X_proc)

    # === Outer CV splitter ===
    if cv_type == "LOO":
        outer_splits = list(LeaveOneOut().split(X_proc, labels))
        cv_label = "LOO"
    elif cv_type == "LOOPC":
        outer_splits = loopc_splits(year_labels if class_by_year else labels,
                                    num_repeats=num_repeats,
                                    random_state=random_state)
        cv_label = "LOOPC"
    elif cv_type == "stratified":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        y_for_split = year_labels if class_by_year else labels
        outer_splits = list(cv.split(X_proc, y_for_split))
        cv_label = "Stratified"
    else:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    # === Plot setup ===
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], marker='o')
    ax.set_xlabel("Percentage of TIC Data Remaining (%)")
    ax.set_ylabel("Accuracy")
    # ax.set_title(f"Greedy Survival: Accuracy vs TIC Data Remaining ({cv_label} CV)")
    ax.grid(True)
    ax.set_xlim(100, 0)

    accuracies = []
    percent_remaining = []

    # === Baseline: accuracy with all bins active ===
    baseline_fold_accs = []
    for train_idx, test_idx in outer_splits:
        X_train, X_test = X_proc[train_idx], X_proc[test_idx]
        if class_by_year:
            y_train, y_test = year_labels[train_idx], year_labels[test_idx]
        else:
            y_train, y_test = labels[train_idx], labels[test_idx]

        final_wrap = Classifier(
            X_train, y_train,
            classifier_type=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            year_labels=(np.array(year_labels)[train_idx]
                         if (year_labels is not None and class_by_year) else None),
            strategy=strategy,
            sample_labels=np.array(raw_sample_labels)[train_idx],
            dataset_origins=dataset_origins,
        )

        res, _, _, _ = final_wrap.train_and_evaluate_balanced(
            normalize=normalize_flag,
            scaler_type='standard',
            region=region,
            random_seed=random_state,
            test_size=0.2,
            LOOPC=False,
            projection_source=False,
            X_test=X_test,
            y_test=y_test
        )
        baseline_fold_accs.append(res["balanced_accuracy"])

    baseline_acc = np.mean(baseline_fold_accs)
    accuracies.append(baseline_acc)
    percent_remaining.append(100.0)
    line.set_data(percent_remaining, accuracies)
    plt.draw()
    plt.pause(0.2)

    logger.info(f"Baseline (all bins): mean outer acc = {baseline_acc:.3f}")

    n_classes = len(np.unique(labels))

    # === Main loop ===
    for step in range(n_iterations):
        if len(active_bins) <= min_bins:
            break
        logger.info(f"=== Iteration {step+1}/{n_iterations} | Active bins: {len(active_bins)} ===")

        # ---- Try removing each candidate bin globally ----
        candidate_scores = []
        already_removed = [x for x in range(n_bins) if x not in active_bins]
        for b in active_bins:
            fold_accs = []
            bins_to_remove = already_removed + [b]
            for train_idx, test_idx in outer_splits:
                X_train, X_test = X_proc[train_idx], X_proc[test_idx]
                if class_by_year:
                    y_train, y_test = year_labels[train_idx], year_labels[test_idx]
                else:
                    y_train, y_test = labels[train_idx], labels[test_idx]

                # remove this candidate bin
                temp_train = remove_bins(X_train, bins_to_remove=bins_to_remove,bin_ranges=bin_ranges)
                temp_test = remove_bins(X_test, bins_to_remove=bins_to_remove,  bin_ranges=bin_ranges)

                # train wrapper
                temp_cls = Classifier(
                    temp_train, y_train,
                    classifier_type=classifier,
                    wine_kind=wine_kind,
                    class_by_year=class_by_year,
                    year_labels=(np.array(year_labels)[train_idx]
                                 if (year_labels is not None and class_by_year) else None),
                    strategy=strategy,
                    sample_labels=np.array(raw_sample_labels)[train_idx],
                    dataset_origins=dataset_origins,
                )

                res, _, _, _ = temp_cls.train_and_evaluate_balanced(
                    normalize=normalize_flag,
                    scaler_type='standard',
                    region=region,
                    random_seed=random_state,
                    test_size=0.2,
                    LOOPC=False,
                    projection_source=False,
                    X_test=temp_test,
                    y_test=y_test
                )
                fold_accs.append(res["balanced_accuracy"])

            # mean CV accuracy if bin b were removed
            candidate_scores.append((b, np.mean(fold_accs)))

        # ---- Pick the best bin to remove ----
        best_bin, best_score = max(candidate_scores, key=lambda x: x[1])
        active_bins.remove(best_bin)

        # ---- Record performance ----
        # Now use TIC fraction instead of bin count
        masked_data = remove_bins(X_proc, bins_to_remove=[b for b in range(n_bins) if b not in active_bins],
                                  bin_ranges=bin_ranges)

        # pct_data = (np.sum(masked_data) / np.sum(X_proc)) * 100
        pct_data = (np.count_nonzero(masked_data) / baseline_nonzero) * 100

        accuracies.append(best_score)
        percent_remaining.append(pct_data)

        line.set_data(percent_remaining, accuracies)
        ax.set_xlim(100, min(percent_remaining) - 5)
        ax.set_ylim(0, 1)
        plt.draw()
        plt.pause(0.2)

        logger.info(f"Iteration {step+1}: removed bin {best_bin}, mean outer acc = {best_score:.3f}, % data = {pct_data:.1f}")

    print(X_train, X_test)
    plt.ioff()
    plt.show()


def filter_singletons(X, labels, raw_labels=None, class_by_year=False):
    """
    Remove samples belonging to classes (or bases, if class_by_year)
    that occur only once.
    """
    y = np.asarray(labels)

    if class_by_year:
        if raw_labels is None:
            raise ValueError("raw_labels required when class_by_year=True")
        raw_labels = np.asarray(raw_labels)

        # base -> indices
        base_to_indices = {}
        for i, sid in enumerate(raw_labels):
            base = str(sid).rsplit("-", 1)[0] if "-" in str(sid) else str(sid)
            base_to_indices.setdefault(base, []).append(i)

        # class -> bases
        class_to_bases = {}
        for base, idxs in base_to_indices.items():
            cls = y[idxs[0]]
            class_to_bases.setdefault(cls, []).append(base)

        # keep only classes with ≥2 bases
        keep_idx = []
        for cls, bases in class_to_bases.items():
            if len(bases) >= 2:
                for base in bases:
                    keep_idx.extend(base_to_indices[base])

    else:
        # Group by the prefix (e.g. "A" from "A1990")
        prefixes = np.array([str(lbl)[0] for lbl in y])  # take first letter
        unique_prefixes, counts = np.unique(prefixes, return_counts=True)

        # keep only prefixes with ≥2 samples
        keep_prefixes = {p for p, c in zip(unique_prefixes, counts) if c >= 2}
        keep_idx = [i for i, p in enumerate(prefixes) if p in keep_prefixes]

    keep_idx = np.array(sorted(keep_idx), dtype=int)
    return X[keep_idx], y[keep_idx], (raw_labels[keep_idx] if raw_labels is not None else None)


def mask_cubes(data, cubes, rt_edges, mz_edges, mode="remove"):
    masked = np.zeros_like(data) if mode=="add" else data.copy()
    for (i, j) in cubes:
        rt_start, rt_end = rt_edges[i], rt_edges[i+1]
        mz_start, mz_end = mz_edges[j], mz_edges[j+1]
        if mode == "remove":
            masked[:, rt_start:rt_end, mz_start:mz_end] = 0
        else:  # add mode
            masked[:, rt_start:rt_end, mz_start:mz_end] = data[:, rt_start:rt_end, mz_start:mz_end]
    return masked


def run_sotf_remove_2d(
    data3d,
    labels,
    raw_sample_labels,
    year_labels,
    classifier,
    wine_kind,
    class_by_year,
    strategy,
    dataset_origins,
    cv_type,
    num_repeats,
    normalize_flag,
    region,
    feature_type,
    n_rt_bins=10,
    n_mz_bins=10,
    min_cubes=1,
    n_jobs=-1,
    random_state=42,
    sample_frac=1.0,
    rt_min=None, rt_max=None, mz_min=None, mz_max=None,
    cube_repr="tic"
):
    """
    2D Survival-of-the-fittest (Greedy Remove) over RT × m/z cubes.
    Iteratively removes cubes, masking them to zero, and evaluates via nested CV.
    Produces both survival curve and heatmap (removal order).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from joblib import Parallel, delayed

    # === Singleton filtering ===
    if cv_type in ("LOO", "LOOPC"):
        data3d, labels, raw_sample_labels = filter_singletons(
            data3d,
            year_labels if class_by_year else labels,
            raw_labels=raw_sample_labels,
            class_by_year=class_by_year,
        )
        if class_by_year:
            year_labels = labels

    n_samples, n_time, n_channels = data3d.shape

    # === Crop data ===
    rt_min = 0 if rt_min is None else max(0, rt_min)
    rt_max = n_time if rt_max is None else min(n_time, rt_max)
    mz_min = 0 if mz_min is None else max(0, mz_min)
    mz_max = n_channels if mz_max is None else min(n_channels, mz_max)
    data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]

    n_samples, n_time, n_channels = data3d.shape

    # === Define cube grid (force last edge to cover full range) ===
    rt_edges = np.linspace(rt_min, rt_max, n_rt_bins + 1, dtype=int)
    mz_edges = np.linspace(mz_min, mz_max, n_mz_bins + 1, dtype=int)
    rt_edges[-1] = rt_max
    mz_edges[-1] = mz_max

    all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]
    active_cubes, cubes_to_remove = all_cubes.copy(), []

    n_iterations = len(all_cubes) - min_cubes + 1
    n_classes = len(np.unique(year_labels if class_by_year else labels))

    # === Precompute features if concatenated ===
    precomputed_features, subcubes = None, None
    if feature_type == "concat_channels":
        precomputed_features = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i+1] - rt_min
            mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j+1] - mz_min
            cube = data3d[:, rt_start:rt_end, mz_start:mz_end]

            if cube_repr == "flat":
                features = cube.reshape(n_samples, -1)
            elif cube_repr == "tic":
                features = np.sum(cube, axis=2)  # sum across m/z
            elif cube_repr == "tis":
                features = np.sum(cube, axis=1)  # sum across RT
            elif cube_repr == "tic_tis":
                tic = np.sum(cube, axis=2)
                tis = np.sum(cube, axis=1)
                features = np.hstack([tic, tis])
            else:
                raise ValueError(f"Unsupported cube_repr: {cube_repr}")

            precomputed_features[(i, j)] = features
    else:
        subcubes = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i+1] - rt_min
            mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j+1] - mz_min
            subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)

    # === Outer CV splitter ===
    full_X = np.zeros((n_samples, 1))
    if cv_type == "LOO":
        outer_splits, cv_label = list(LeaveOneOut().split(full_X, labels)), "LOO"
    elif cv_type == "LOOPC":
        outer_splits, cv_label = loopc_splits(
            year_labels if class_by_year else labels,
            num_repeats=num_repeats, random_state=random_state,
            class_by_year=class_by_year,
            raw_labels=raw_sample_labels if class_by_year else None,
        ), "LOOPC"
    elif cv_type == "stratified":
        y_for_split = year_labels if class_by_year else labels
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        outer_splits, cv_label = list(cv.split(full_X, y_for_split)), "Stratified"
    else:
        raise ValueError(f"Unsupported cv_type: {cv_type}")

    # === Plot setup ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))

    line, = ax_curve.plot([], [], marker="o")
    ax_curve.set_xlabel("Percentage of cubes remaining (%)")
    ax_curve.set_ylabel("Accuracy")
    ax_curve.set_title(f"2D SOTF Greedy Remove ({cv_label} CV)")
    ax_curve.grid(True)
    ax_curve.set_xlim(100, 0)
    ax_curve.set_ylim(0, 1)

    removal_order = np.full((n_mz_bins, n_rt_bins), np.nan)
    annotations = np.full((n_mz_bins, n_rt_bins), "", dtype=object)
    sns.heatmap(
        removal_order, vmin=0, vmax=n_iterations, cmap="viridis_r",
        ax=ax_heat, cbar=False,
        xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
        yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)],
        annot=annotations, fmt="",
        annot_kws={"size": 8, "color": "black"}
    )
    ax_heat.set_title("Cube removal order (hot = early)")
    ax_heat.invert_yaxis()
    plt.pause(0.1)

    accuracies, percent_remaining = [], []

    # === Safe train/eval ===
    def safe_train_eval(Xtr, ytr, Xte, yte, train_idx):
        if Xtr.shape[1] == 0 or Xte.shape[1] == 0:
            return {"balanced_accuracy": 1.0 / n_classes}
        try:
            cls_wrap = Classifier(
                Xtr, ytr, classifier_type=classifier,
                wine_kind=wine_kind, class_by_year=class_by_year,
                year_labels=(np.array(year_labels)[train_idx]
                             if (year_labels is not None and class_by_year) else None),
                strategy=strategy, sample_labels=np.array(raw_sample_labels)[train_idx],
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag, scaler_type="standard",
                region=region, random_seed=random_state,
                test_size=0.2, LOOPC=False, projection_source=False,
                X_test=Xte, y_test=yte,
            )
            return res
        except np.linalg.LinAlgError:
            return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}

    # === Baseline accuracy (all cubes) ===
    baseline_accs = []
    for train_idx, test_idx in outer_splits:
        if feature_type == "concat_channels":
            X_train = np.hstack([precomputed_features[c][train_idx] for c in all_cubes])
            X_test  = np.hstack([precomputed_features[c][test_idx] for c in all_cubes])
        else:
            X_train = utils.compute_features(data3d[train_idx], feature_type=feature_type)
            X_test  = utils.compute_features(data3d[test_idx], feature_type=feature_type)
        y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
        y_test  = year_labels[test_idx] if class_by_year else labels[test_idx]
        res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
        baseline_accs.append(res["balanced_accuracy"])
    baseline_acc = np.mean(baseline_accs)

    accuracies.append(baseline_acc)
    percent_remaining.append(100.0)
    line.set_data(percent_remaining, accuracies)
    plt.pause(0.2)

    rng = np.random.default_rng(random_state)
    best_global_score, best_global_cube = baseline_acc, None

    # === Greedy loop (remove) ===
    for step in range(n_iterations):
        if len(active_cubes) <= min_cubes: break

        if sample_frac < 1.0:
            n_sample = max(1, int(len(active_cubes) * sample_frac))
            candidates = rng.choice(active_cubes, size=n_sample, replace=False)
            candidates = [tuple(c) for c in candidates]
        else:
            candidates = active_cubes

        def eval_candidate(cube):
            candidate_remove = cubes_to_remove + [cube]
            fold_accs = []
            for train_idx, test_idx in outer_splits:
                if feature_type == "concat_channels":
                    remaining = [c for c in all_cubes if c not in candidate_remove]
                    X_train = np.hstack([precomputed_features[c][train_idx] for c in remaining])
                    X_test  = np.hstack([precomputed_features[c][test_idx] for c in remaining])
                else:
                    masked_train = mask_cubes(data3d[train_idx], candidate_remove, rt_edges, mz_edges)
                    masked_test  = mask_cubes(data3d[test_idx], candidate_remove, rt_edges, mz_edges)
                    X_train = utils.compute_features(masked_train, feature_type=feature_type)
                    X_test  = utils.compute_features(masked_test,  feature_type=feature_type)
                y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
                y_test  = year_labels[test_idx] if class_by_year else labels[test_idx]
                res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
                fold_accs.append(res["balanced_accuracy"])
            return cube, np.mean(fold_accs)

        candidate_scores = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(cube) for cube in candidates
        )
        best_cube, best_score = max(candidate_scores, key=lambda x: x[1])
        cubes_to_remove.append(best_cube); active_cubes.remove(best_cube)

        pct_data = (len(active_cubes) / len(all_cubes)) * 100
        accuracies.append(best_score); percent_remaining.append(pct_data)

        if best_score > best_global_score:
            best_global_score, best_global_cube = best_score, best_cube

        # Update plots
        line.set_data(percent_remaining, accuracies)
        ax_curve.set_xlim(100, min(percent_remaining) - 5); plt.pause(0.2)

        (i, j) = best_cube
        removal_order[j, i] = step + 1
        for x in range(n_rt_bins):
            for y in range(n_mz_bins):
                if not np.isnan(removal_order[y, x]):
                    annotations[y, x] = f"{int(removal_order[y, x])}"
        if best_global_cube is not None:
            bi, bj = best_global_cube
            annotations[bj, bi] = f"{int(removal_order[bj, bi])}\n{best_global_score:.2f}"

        ax_heat.clear()
        sns.heatmap(
            removal_order, vmin=0, vmax=len(all_cubes),
            cmap="viridis_r", ax=ax_heat, cbar=False,
            xticklabels=[f"{rt_edges[x]}–{rt_edges[x+1]}" for x in range(n_rt_bins)],
            yticklabels=[f"{mz_edges[y]}–{mz_edges[y+1]}" for y in range(n_mz_bins)],
            annot=annotations, fmt="",
            annot_kws={"size": 8, "color": "black"}
        )
        for text in ax_heat.texts:
            x, y = text.get_position()
            i, j = int(round(x - 0.5)), int(round(y - 0.5))
            val = removal_order[j, i]
            if not np.isnan(val):
                norm_val = (val - 0) / (len(all_cubes) - 0)
                rgba = plt.cm.viridis_r(norm_val)
                luminance = 0.299*rgba[0] + 0.587*rgba[1] + 0.114*rgba[2]
                text.set_color("white" if luminance < 0.5 else "black")

        ax_heat.set_title("Cube removal order (hot = early)")
        ax_heat.invert_yaxis(); plt.pause(0.1)

    plt.ioff(); plt.show()
    return accuracies, percent_remaining, removal_order


# def run_sotf_2d(
#     data3d,
#     labels,
#     raw_sample_labels,
#     year_labels,
#     classifier,
#     wine_kind,
#     class_by_year,
#     strategy,
#     dataset_origins,
#     cv_type,
#     num_repeats,
#     normalize_flag,
#     region,
#     feature_type,
#     projection_source,
#     show_confusion_matrix,
#     n_rt_bins=10,
#     n_mz_bins=10,
#     min_cubes=1,
#     n_jobs=20,
#     random_state=42,
#     sample_frac=1.0,   # fraction of cubes to evaluate at each step (0–1)
# ):
#     """
#     2D Survival-of-the-fittest over RT × m/z cubes.
#     Iteratively removes cubes, masking them to zero, and evaluates via nested CV.
#     Produces both survival curve and heatmap (removal order).
#     """
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#     from joblib import Parallel, delayed
#
#     # === Singleton filtering ===
#     if cv_type in ("LOO", "LOOPC"):
#         data3d, labels, raw_sample_labels = filter_singletons(
#             data3d,
#             year_labels if class_by_year else labels,
#             raw_labels=raw_sample_labels,
#             class_by_year=class_by_year,
#         )
#         if class_by_year:
#             year_labels = labels
#
#     # data3d = data3d[:, :, 0:91]
#
#     n_samples, n_time, n_channels = data3d.shape
#
#     # === Define cube grid (force last edge to cover full range) ===
#     rt_edges = np.linspace(0, n_time, n_rt_bins + 1, dtype=int)
#     mz_edges = np.linspace(0, n_channels, n_mz_bins + 1, dtype=int)
#
#     # make sure last edges reach the end
#     rt_edges[-1] = n_time
#     mz_edges[-1] = n_channels
#
#     all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]
#     active_cubes = all_cubes.copy()
#     cubes_to_remove = []
#
#     n_iterations = len(all_cubes) - min_cubes + 1
#     n_classes = len(np.unique(year_labels if class_by_year else labels))
#
#     # === Outer CV splitter ===
#     full_X = utils.compute_features(data3d, feature_type=feature_type)
#     if cv_type == "LOO":
#         outer_splits = list(LeaveOneOut().split(full_X, labels))
#         cv_label = "LOO"
#     elif cv_type == "LOOPC":
#         outer_splits = loopc_splits(
#             year_labels if class_by_year else labels,
#             num_repeats=num_repeats,
#             random_state=random_state,
#             class_by_year=class_by_year,
#             raw_labels=raw_sample_labels if class_by_year else None,
#         )
#         cv_label = "LOOPC"
#     elif cv_type == "stratified":
#         y_for_split = year_labels if class_by_year else labels
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#         outer_splits = list(cv.split(full_X, y_for_split))
#         cv_label = "Stratified"
#     else:
#         raise ValueError(f"Unsupported cv_type: {cv_type}")
#
#     # === Plot setup ===
#     plt.ion()
#     fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
#
#     # Survival curve
#     line, = ax_curve.plot([], [], marker="o")
#     ax_curve.set_xlabel("Percentage of cubes remaining (%)")
#     ax_curve.set_ylabel("Accuracy")
#     ax_curve.set_title(f"2D SOTF Curve ({cv_label} CV)")
#     ax_curve.grid(True)
#     ax_curve.set_xlim(100, 0)
#     ax_curve.set_ylim(0, 1)
#
#     # Heatmap (removal order)
#     removal_order = np.full((n_mz_bins, n_rt_bins), np.nan)
#     annotations = np.full((n_mz_bins, n_rt_bins), "", dtype=object)
#     sns.heatmap(
#         removal_order, vmin=0, vmax=n_iterations, cmap="viridis_r",
#         ax=ax_heat, cbar=False,
#         xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#         yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)],
#         annot=annotations, fmt="",
#         annot_kws={"size": 8, "color": "black"}
#     )
#     ax_heat.set_xlabel("RT bins")
#     ax_heat.set_ylabel("m/z bins")
#     ax_heat.set_title("Cube removal order (hot = early)")
#     ax_heat.invert_yaxis()
#     ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")
#     ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
#     plt.pause(0.1)
#
#     accuracies, percent_remaining = [], []
#
#     # === Safe train/eval helper ===
#     def safe_train_eval(Xtr, ytr, Xte, yte, train_idx):
#         if Xtr.shape[1] == 0 or Xte.shape[1] == 0:
#             return {"balanced_accuracy": 1.0 / n_classes}
#         try:
#             cls_wrap = Classifier(
#                 Xtr, ytr,
#                 classifier_type=classifier,
#                 wine_kind=wine_kind,
#                 class_by_year=class_by_year,
#                 year_labels=(np.array(year_labels)[train_idx]
#                              if (year_labels is not None and class_by_year) else None),
#                 strategy=strategy,
#                 sample_labels=np.array(raw_sample_labels)[train_idx],
#                 dataset_origins=dataset_origins,
#             )
#             res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
#                 normalize=normalize_flag,
#                 scaler_type="standard",
#                 region=region,
#                 random_seed=random_state,
#                 test_size=0.2,
#                 LOOPC=False,
#                 projection_source=False,
#                 X_test=Xte,
#                 y_test=yte,
#             )
#             return res
#         except np.linalg.LinAlgError:
#             return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}
#
#     # === Baseline accuracy ===
#     baseline_accs = []
#     for train_idx, test_idx in outer_splits:
#         X_train = utils.compute_features(data3d[train_idx], feature_type=feature_type)
#         X_test  = utils.compute_features(data3d[test_idx],  feature_type=feature_type)
#         y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
#         y_test  = year_labels[test_idx]  if class_by_year else labels[test_idx]
#         res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
#         baseline_accs.append(res["balanced_accuracy"])
#     baseline_acc = np.mean(baseline_accs)
#
#     accuracies.append(baseline_acc)
#     percent_remaining.append(100.0)
#     line.set_data(percent_remaining, accuracies)
#     plt.pause(0.2)
#     logger.info(f"Baseline (all cubes): mean acc = {baseline_acc:.3f}")
#
#     # === Greedy loop ===
#     rng = np.random.default_rng(random_state)
#     best_global_score = baseline_acc
#     best_global_cube = None
#
#     for step in range(n_iterations):
#         if len(active_cubes) <= min_cubes:
#             break
#
#         # Choose candidate cubes (possibly subsampled)
#         if sample_frac < 1.0:
#             n_sample = max(1, int(len(active_cubes) * sample_frac))
#             candidates = rng.choice(active_cubes, size=n_sample, replace=False)
#             candidates = [tuple(c) for c in candidates]
#         else:
#             candidates = active_cubes
#
#         # Evaluate candidates
#         def eval_candidate(cube):
#             candidate_remove = cubes_to_remove + [cube]
#             fold_accs = []
#             for train_idx, test_idx in outer_splits:
#                 masked_train = mask_cubes(data3d[train_idx], candidate_remove, rt_edges, mz_edges)
#                 masked_test  = mask_cubes(data3d[test_idx],  candidate_remove, rt_edges, mz_edges)
#                 X_train = utils.compute_features(masked_train, feature_type=feature_type)
#                 X_test  = utils.compute_features(masked_test,  feature_type=feature_type)
#                 y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
#                 y_test  = year_labels[test_idx]  if class_by_year else labels[test_idx]
#                 res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
#                 fold_accs.append(res["balanced_accuracy"])
#             return cube, np.mean(fold_accs)
#
#         candidate_scores = Parallel(n_jobs=n_jobs)(
#             delayed(eval_candidate)(cube) for cube in candidates
#         )
#
#         # Pick best
#         best_cube, best_score = max(candidate_scores, key=lambda x: x[1])
#         best_cube = tuple(best_cube)
#         cubes_to_remove.append(best_cube)
#         active_cubes.remove(best_cube)
#
#         # Record progress
#         pct_data = (len(active_cubes) / len(all_cubes)) * 100
#         accuracies.append(best_score)
#         percent_remaining.append(pct_data)
#
#         # Track best global
#         if best_score > best_global_score:
#             best_global_score = best_score
#             best_global_cube = best_cube
#
#         # Update curve
#         line.set_data(percent_remaining, accuracies)
#         ax_curve.set_xlim(100, min(percent_remaining) - 5)
#         plt.pause(0.2)
#
#         # Update heatmap with annotation
#         (i, j) = best_cube
#         removal_order[j, i] = step + 1
#         annotations[j, i] = f"{step+1}"
#
#         for x in range(n_rt_bins):
#             for y in range(n_mz_bins):
#                 if not np.isnan(removal_order[y, x]):
#                     annotations[y, x] = f"{int(removal_order[y, x])}"
#
#         if best_global_cube is not None:
#             bi, bj = best_global_cube
#             annotations[bj, bi] = f"{int(removal_order[bj, bi])}\n{best_global_score:.2f}"
#
#         ax_heat.clear()
#         sns.heatmap(
#             removal_order,
#             vmin=0, vmax=len(all_cubes),
#             cmap="viridis_r", ax=ax_heat, cbar=False,
#             xticklabels=[f"{rt_edges[x]}–{rt_edges[x+1]}" for x in range(n_rt_bins)],
#             yticklabels=[f"{mz_edges[y]}–{mz_edges[y+1]}" for y in range(n_mz_bins)],
#             annot=annotations, fmt="",
#             annot_kws={"size": 8, "color": "black"}
#         )
#         for text in ax_heat.texts:
#             x, y = text.get_position()
#             i, j = int(round(x - 0.5)), int(round(y - 0.5))
#             val = removal_order[j, i]
#             if not np.isnan(val):
#                 norm_val = (val - 0) / (len(all_cubes) - 0)
#                 rgba = plt.cm.viridis_r(norm_val)
#                 luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
#                 text.set_color("white" if luminance < 0.5 else "black")
#
#         ax_heat.set_xlabel("RT bins")
#         ax_heat.set_ylabel("m/z bins")
#         ax_heat.set_title("Cube removal order (hot = early)")
#         ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")
#         ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
#         ax_heat.invert_yaxis()
#         plt.pause(0.1)
#
#         logger.info(f"Iteration {step+1}: removed cube {best_cube}, acc={best_score:.3f}, % cubes={pct_data:.1f}")
#
#     plt.ioff()
#     plt.show()
#
#     return accuracies, percent_remaining, removal_order


def run_sotf_add_leaky(
    data3d,
    labels,
    raw_sample_labels,
    year_labels,
    classifier,
    wine_kind,
    class_by_year,
    strategy,
    dataset_origins,
    cv_type,
    num_repeats,
    normalize_flag,
    region,
    feature_type,
    n_rt_bins=10,
    n_mz_bins=10,
    max_cubes=None,
    n_jobs=-1,
    random_state=42,
    sample_frac=1.0,
    rt_min=None, rt_max=None, mz_min=None, mz_max=None,
    cube_repr="concatenate"
):
    """
    2D Greedy Add (SOTF):
      - Iteratively adds cubes, evaluates accuracy.
      - Shows survival curve and cube addition order heatmap.
      - For feature_type='concatenated', supports different cube_repr: flat, tic, tis, tic_tis.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from joblib import Parallel, delayed

    # === Singleton filtering ===
    if cv_type in ("LOO", "LOOPC"):
        data3d, labels, raw_sample_labels = filter_singletons(
            data3d,
            year_labels if class_by_year else labels,
            raw_labels=raw_sample_labels,
            class_by_year=class_by_year,
        )
        if class_by_year:
            year_labels = labels

    n_samples, n_time, n_channels = data3d.shape

    # === Crop data ===
    rt_min = 0 if rt_min is None else max(0, rt_min)
    rt_max = n_time if rt_max is None else min(n_time, rt_max)
    mz_min = 0 if mz_min is None else max(0, mz_min)
    mz_max = n_channels if mz_max is None else min(n_channels, mz_max)

    data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]
    n_samples, n_time, n_channels = data3d.shape

    # === Define cube grid ===
    rt_edges = np.linspace(rt_min, rt_max, n_rt_bins + 1, dtype=int)
    mz_edges = np.linspace(mz_min, mz_max, n_mz_bins + 1, dtype=int)
    rt_edges[-1] = rt_max
    mz_edges[-1] = mz_max

    all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]
    added_cubes, not_added = [], all_cubes.copy()

    if max_cubes is None:
        max_cubes = len(all_cubes)
    n_iterations = max_cubes
    n_classes = len(np.unique(year_labels if class_by_year else labels))

    # === Precompute features if concatenated ===
    precomputed_features, subcubes = None, None
    if feature_type == "concat_channels":
        precomputed_features = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i + 1] - rt_min
            mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j + 1] - mz_min
            cube = data3d[:, rt_start:rt_end, mz_start:mz_end]

            if cube_repr == "flat":
                features = cube.reshape(n_samples, -1)
            elif cube_repr == "tic":
                features = np.sum(cube, axis=2)  # TIC inside cube
            elif cube_repr == "tis":
                features = np.sum(cube, axis=1)  # TIS inside cube
            elif cube_repr == "tic_tis":
                tic = np.sum(cube, axis=2)
                tis = np.sum(cube, axis=1)
                features = np.hstack([tic, tis])
            else:
                raise ValueError(f"Unsupported cube_repr: {cube_repr}")

            precomputed_features[(i, j)] = features
    else:
        subcubes = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i + 1] - rt_min
            mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j + 1] - mz_min
            subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)

    # === Outer CV splitter ===
    dummy_X = np.zeros((n_samples, 1))
    if cv_type == "LOO":
        outer_splits, cv_label = list(LeaveOneOut().split(dummy_X, labels)), "LOO"
    elif cv_type == "LOOPC":
        outer_splits, cv_label = loopc_splits(
            year_labels if class_by_year else labels,
            num_repeats=num_repeats, random_state=random_state,
            class_by_year=class_by_year,
            raw_labels=raw_sample_labels if class_by_year else None,
        ), "LOOPC"
    elif cv_type == "stratified":
        y_for_split = year_labels if class_by_year else labels
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        outer_splits, cv_label = list(cv.split(dummy_X, y_for_split)), "Stratified"
    else:
        raise ValueError(f"Unsupported cv_type: {cv_type}")

    # === Plot setup ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    line, = ax_curve.plot([], [], marker="o")
    ax_curve.set_xlabel("Percentage of cubes added (%)")
    ax_curve.set_ylabel("Accuracy")
    ax_curve.set_title(f"2D SOTF Greedy Add ({cv_label} CV)")
    ax_curve.grid(True)
    ax_curve.set_xlim(0, 100)
    ax_curve.set_ylim(0, 1)

    addition_order = np.full((n_mz_bins, n_rt_bins), np.nan)
    annotations = np.full((n_mz_bins, n_rt_bins), "", dtype=object)
    sns.heatmap(
        addition_order, vmin=0, vmax=n_iterations, cmap="viridis",
        ax=ax_heat, cbar=False,
        xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
        yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)],
        annot=annotations, fmt="",
        annot_kws={"size": 8}
    )
    ax_heat.set_title("Cube addition order (cold = early)")
    ax_heat.invert_yaxis()
    plt.pause(0.1)

    accuracies, percent_added = [], []

    # === Safe train/eval ===
    def safe_train_eval(Xtr, ytr, Xte, yte, train_idx):
        if Xtr.shape[1] == 0 or Xte.shape[1] == 0:
            return {"balanced_accuracy": 1.0 / n_classes}
        try:
            cls_wrap = Classifier(
                Xtr, ytr, classifier_type=classifier,
                wine_kind=wine_kind, class_by_year=class_by_year,
                year_labels=(np.array(year_labels)[train_idx]
                             if (year_labels is not None and class_by_year) else None),
                strategy=strategy, sample_labels=np.array(raw_sample_labels)[train_idx],
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag, scaler_type="standard",
                region=region, random_seed=random_state,
                test_size=0.2, LOOPC=False, projection_source=False,
                X_test=Xte, y_test=yte,
            )
            return res
        except np.linalg.LinAlgError:
            return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}

    # === Baseline (no cubes) ===
    baseline_acc = 1.0 / n_classes
    accuracies.append(baseline_acc)
    percent_added.append(0.0)
    line.set_data(percent_added, accuracies)
    plt.pause(0.2)

    rng = np.random.default_rng(random_state)
    best_global_score, best_global_cube = baseline_acc, None

    # === Greedy loop ===
    for step in range(n_iterations):
        if len(added_cubes) >= max_cubes: break

        # Candidate subset
        if sample_frac < 1.0:
            n_sample = max(1, int(len(not_added) * sample_frac))
            candidates = rng.choice(not_added, size=n_sample, replace=False)
            candidates = [tuple(c) for c in candidates]
        else:
            candidates = not_added

        def eval_candidate(cube):
            candidate_add = added_cubes + [cube]
            fold_accs = []
            for train_idx, test_idx in outer_splits:
                if feature_type == "concat_channels":
                    X_train = np.hstack([precomputed_features[c][train_idx] for c in candidate_add])
                    X_test  = np.hstack([precomputed_features[c][test_idx] for c in candidate_add])
                else:
                    masked = np.zeros_like(data3d)
                    for (i, j) in candidate_add:
                        rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
                        masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
                    X_train = utils.compute_features(masked[train_idx], feature_type=feature_type)
                    X_test  = utils.compute_features(masked[test_idx], feature_type=feature_type)
                y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
                y_test  = year_labels[test_idx] if class_by_year else labels[test_idx]
                res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
                fold_accs.append(res["balanced_accuracy"])
            return cube, np.mean(fold_accs)

        candidate_scores = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(cube) for cube in candidates
        )
        best_cube, best_score = max(candidate_scores, key=lambda x: x[1])
        added_cubes.append(best_cube); not_added.remove(best_cube)

        pct_data = (len(added_cubes) / len(all_cubes)) * 100
        accuracies.append(best_score); percent_added.append(pct_data)

        if best_score > best_global_score:
            best_global_score, best_global_cube = best_score, best_cube

        # Update curve
        line.set_data(percent_added, accuracies)
        ax_curve.set_xlim(0, max(percent_added) + 5); plt.pause(0.2)

        # Update heatmap annotations
        (i, j) = best_cube
        addition_order[j, i] = step + 1
        for x in range(n_rt_bins):
            for y in range(n_mz_bins):
                if not np.isnan(addition_order[y, x]):
                    annotations[y, x] = f"{int(addition_order[y, x])}"
        if best_global_cube is not None:
            bi, bj = best_global_cube
            annotations[bj, bi] = f"{int(addition_order[bj, bi])}\n{best_global_score:.2f}"

        ax_heat.clear()
        sns.heatmap(
            addition_order, vmin=0, vmax=len(all_cubes),
            cmap="viridis", ax=ax_heat, cbar=False,
            xticklabels=[f"{rt_edges[x]}–{rt_edges[x+1]}" for x in range(n_rt_bins)],
            yticklabels=[f"{mz_edges[y]}–{mz_edges[y+1]}" for y in range(n_mz_bins)],
            annot=annotations, fmt="",
            annot_kws={"size": 8}
        )
        for text in ax_heat.texts:
            x, y = text.get_position()
            i, j = int(round(x - 0.5)), int(round(y - 0.5))
            val = addition_order[j, i]
            if not np.isnan(val):
                norm_val = (val - 0) / (len(all_cubes) - 0)
                rgba = plt.cm.viridis(norm_val)
                luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                text.set_color("white" if luminance < 0.5 else "black")

        ax_heat.set_title("Cube addition order (cold = early)")
        ax_heat.invert_yaxis(); plt.pause(0.1)

    plt.ioff(); plt.show()
    return accuracies, percent_added, addition_order


def run_sotf_remove_noleak(
    data3d,
    labels,
    raw_sample_labels,
    year_labels,
    classifier,
    wine_kind,
    class_by_year,
    strategy,
    dataset_origins,
    cv_type,
    num_repeats,
    normalize_flag,
    region,
    feature_type,
    n_rt_bins=10,
    n_mz_bins=10,
    min_cubes=1,
    max_cubes=None,
    n_jobs=-1,
    random_state=42,
    sample_frac=1.0,                 # not used for remove (kept for API parity)
    rt_min=None, rt_max=None, mz_min=None, mz_max=None,
    cube_repr="tic",
):
    """
    2D Greedy Remove (SOTF), leak-free, with:
      - Parallel outer CV folds (joblib)
      - Sequential inner candidate scoring (no oversubscription)
      - Efficient column dropping for feature_type='concat_channels' (all cube_repr modes)
      - Live plot updated once per finished outer fold:
          * Balanced accuracy vs % cubes removed (start at 0% removed = all cubes)
          * Heatmap of most-common removal order with % occurrence
          * Figure window title shows Fold x/total
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter, defaultdict
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
    from joblib import Parallel, delayed
    from tqdm import trange, tqdm

    rng = np.random.default_rng(random_state)

    # === Crop data (identical semantics) ===
    n_samples, n_time, n_channels = data3d.shape
    rt_min = 0 if rt_min is None else max(0, rt_min)
    rt_max = n_time if rt_max is None else min(n_time, rt_max)
    mz_min = 0 if mz_min is None else max(0, mz_min)
    mz_max = n_channels if mz_max is None else min(n_channels, mz_max)
    data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]
    n_samples, n_time, n_channels = data3d.shape

    # === Cube grid ===
    rt_edges = np.linspace(0, n_time, n_rt_bins + 1, dtype=int)
    mz_edges = np.linspace(0, n_channels, n_mz_bins + 1, dtype=int)
    rt_edges[-1] = n_time
    mz_edges[-1] = n_channels
    all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]

    if max_cubes is None:
        max_cubes = len(all_cubes)  # number of removals we'll perform
    n_iterations = max_cubes

    y_all = (year_labels if class_by_year else labels)
    classes = np.unique(y_all)
    n_classes = len(classes)

    # === Precompute per-cube features for concat_channels; honor cube_repr ===
    precomputed_features = None
    cube_dims = None  # per-cube feature dimension (for column slicing)
    if feature_type == "concat_channels":
        precomputed_features = {}
        cube_dims = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            cube = data3d[:, rt_start:rt_end, mz_start:mz_end]  # (N, rt, mz)
            cr = "flat" if cube_repr == "concatenate" else cube_repr
            if cr == "flat":
                feats = cube.reshape(n_samples, -1)
            elif cr == "tic":
                feats = np.sum(cube, axis=2)    # (N, rt)
            elif cr == "tis":
                feats = np.sum(cube, axis=1)    # (N, mz)
            elif cr == "tic_tis":
                tic = np.sum(cube, axis=2)      # (N, rt)
                tis = np.sum(cube, axis=1)      # (N, mz)
                feats = np.hstack([tic, tis])
            else:
                raise ValueError(f"Unsupported cube_repr: {cube_repr}")
            feats = feats.astype(np.float32, copy=False)
            precomputed_features[(i, j)] = feats
            cube_dims[(i, j)] = feats.shape[1]
    else:
        # masked-path coords
        subcubes = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)

    # === Prepare classification labels ===
    if class_by_year:
            y_for_split = np.asarray(year_labels)
    else:
        y_for_split = np.asarray([str(lbl)[0] for lbl in labels])

    # === Outer CV ===
    dummy_X = np.zeros((n_samples, 1))
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_repeats, random_state=random_state)
    outer_splits = list(outer_cv.split(dummy_X, y_for_split))
    cv_label = f"RepeatedStratified (5x{num_repeats})"

    # === Safe train/eval wrapper (with your exception fallback) ===
    def safe_train_eval(Xtr, ytr, Xte, yte, train_idx):
        if Xtr.shape[1] == 0 or Xte.shape[1] == 0:
            return {"balanced_accuracy": 1.0 / n_classes}
        try:
            cls_wrap = Classifier(
                Xtr, ytr,
                classifier_type=classifier,
                wine_kind=wine_kind,
                class_by_year=class_by_year,
                year_labels=(np.array(year_labels)[train_idx]
                             if (year_labels is not None and class_by_year) else None),
                strategy=strategy,
                sample_labels=np.array(raw_sample_labels)[train_idx] if raw_sample_labels is not None else None,
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type="standard",
                region=region,
                random_seed=random_state,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte,
                y_test=yte,
            )
            return res
        except ValueError as e:
            if "At least one label specified must be in y_true" in str(e):
                from sklearn.metrics import (
                    accuracy_score, balanced_accuracy_score,
                    precision_score, recall_score, f1_score, confusion_matrix
                )
                y_pred = cls_wrap.classifier.predict(Xte)
                observed_labels = np.unique(np.concatenate([yte, y_pred]))
                cm = confusion_matrix(yte, y_pred, labels=observed_labels)
                return {
                    "accuracy": accuracy_score(yte, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(yte, y_pred),
                    "precision": precision_score(yte, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(yte, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(yte, y_pred, average="weighted", zero_division=0),
                    "confusion_matrix": cm,
                }
            else:
                raise

    # === Inner CV factory (kept identical to your add version) ===
    def make_inner_cv(y_outer_train, raw_outer):
        if cv_type == "LOO":
            return LeaveOneOut().split(np.zeros((len(y_outer_train), 1)), y_outer_train)
        elif cv_type == "stratified":
            return StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)\
                   .split(np.zeros((len(y_outer_train), 1)), y_outer_train)
        elif cv_type == "LOOPC":
            # NOTE: follows your provided code that used a fixed 50 repeats inside;
            # if you want this tied to num_repeats, swap 50 -> num_repeats.
            return loopc_splits(
                y_outer_train,
                num_repeats=num_repeats, random_state=random_state,
                class_by_year=class_by_year,
                raw_labels=raw_outer if class_by_year else None,
            )
        else:
            raise ValueError(f"Unsupported inner cv_type: {cv_type}")

    # === Utility for concat_channels: build initial full matrices and per-cube column slices ===
    def build_full_design_and_slices(idx):
        """Return X_full (n_idx, D) and dict cube->slice for columns."""
        # Concatenate in a fixed order = all_cubes order
        parts = []
        slices = {}
        col = 0
        for c in all_cubes:
            F = precomputed_features[c][idx]  # (n_idx, d_c)
            d = cube_dims[c]
            parts.append(F)
            slices[c] = slice(col, col + d)
            col += d
        X_full = np.concatenate(parts, axis=1).astype(np.float32, copy=False) if parts else np.zeros((len(idx), 0), dtype=np.float32)
        return X_full, slices

    # === One outer fold worker ===
    def run_one_outer_fold(fold_id, outer_train_idx, outer_test_idx, fold_seed):
        rng_local = np.random.default_rng(fold_seed)

        y_train_outer = y_all[outer_train_idx]
        y_test_outer  = y_all[outer_test_idx]
        raw_outer = np.array(raw_sample_labels)[outer_train_idx] if raw_sample_labels is not None else None

        # Singleton filtering only for inner CV (like your design)
        if cv_type in ("LOO", "LOOPC"):
            _, y_train_outer, raw_outer = filter_singletons(
                data3d[outer_train_idx],
                y_train_outer,
                raw_labels=raw_outer,
                class_by_year=class_by_year,
            )

        inner_cv = list(make_inner_cv(y_train_outer, raw_outer))

        # Start with ALL cubes present; we will remove one per step
        remaining = set(all_cubes)
        removed_order = []
        outer_curve = []

        if feature_type == "concat_channels":
            # Build full design matrices ONCE; then drop columns per removal
            X_train_full, col_slices_tr = build_full_design_and_slices(outer_train_idx)
            X_test_full,  col_slices_te = build_full_design_and_slices(outer_test_idx)

            # masks of active columns
            active_cols_tr = np.ones(X_train_full.shape[1], dtype=bool)
            active_cols_te = np.ones(X_test_full.shape[1], dtype=bool)

        # Greedy REMOVE loop – do up to n_iterations removals
        for step in trange(n_iterations, desc=f"Fold {fold_id}", leave=False):
            if len(remaining) <= min_cubes:
                break

            # Candidate subset to remove = either all remaining or a sample (rare for remove; we keep all)
            candidates = list(remaining)

            best_cube, best_score = None, -np.inf

            for cube in candidates:
                accs = []

                if feature_type == "concat_channels":
                    # Simulate removal: drop cube's columns
                    s_tr = col_slices_tr[cube]
                    s_te = col_slices_te[cube]
                    # Use temporary masks (avoid copying big arrays)
                    tmp_mask_tr = active_cols_tr.copy()
                    tmp_mask_te = active_cols_te.copy()
                    tmp_mask_tr[s_tr] = False
                    tmp_mask_te[s_te] = False
                    X_tr_all = X_train_full[:, tmp_mask_tr]
                    X_val_all = None  # per-fold below
                # else: masked path rebuilt below

                for tr_rel, val_rel in inner_cv:
                    tr_idx = outer_train_idx[tr_rel]
                    val_idx = outer_train_idx[val_rel]

                    if feature_type == "concat_channels":
                        # Slice rows for the inner split
                        X_tr = X_tr_all[tr_rel]                      # (n_tr, d_active_minus_cube)
                        X_val = X_train_full[:, tmp_mask_tr][val_rel]
                    else:
                        # Build masked data with ALL remaining except 'cube'
                        masked = np.zeros_like(data3d)
                        for (ii, jj) in remaining:
                            if (ii, jj) == cube:
                                continue
                            rts, rte, mzs, mze = subcubes[(ii, jj)]
                            masked[:, rts:rte, mzs:mze] = data3d[:, rts:rte, mzs:mze]
                        X_tr = utils.compute_features(masked[tr_idx], feature_type=feature_type)
                        X_val = utils.compute_features(masked[val_idx], feature_type=feature_type)

                    y_tr = y_all[tr_idx]
                    y_val = y_all[val_idx]
                    res = safe_train_eval(X_tr, y_tr, X_val, y_val, tr_idx)
                    accs.append(res["balanced_accuracy"])

                mean_acc = float(np.mean(accs))
                if mean_acc > best_score:
                    best_score, best_cube = mean_acc, cube

            # Commit the removal
            remaining.remove(best_cube)
            removed_order.append(best_cube)

            # Compute outer accuracy with CURRENT remaining set
            if feature_type == "concat_channels":
                # Permanently drop that cube's columns from active masks
                active_cols_tr[col_slices_tr[best_cube]] = False
                active_cols_te[col_slices_te[best_cube]] = False
                X_train = X_train_full[:, active_cols_tr]
                X_test  = X_test_full[:,  active_cols_te]
            else:
                masked = np.zeros_like(data3d)
                for (ii, jj) in remaining:
                    rts, rte, mzs, mze = subcubes[(ii, jj)]
                    masked[:, rts:rte, mzs:mze] = data3d[:, rts:rte, mzs:mze]
                X_train = utils.compute_features(masked[outer_train_idx], feature_type=feature_type)
                X_test  = utils.compute_features(masked[outer_test_idx],  feature_type=feature_type)

            res_outer = safe_train_eval(X_train, y_all[outer_train_idx], X_test, y_all[outer_test_idx], outer_train_idx)
            outer_curve.append(res_outer["balanced_accuracy"])

        # # === Add terminal point: ALL cubes removed ===
        chance_acc = 1.0 / n_classes
        outer_curve.append(chance_acc)

        return outer_curve, removed_order

    # === Live plotting (curve + heatmap), updated once per finished fold ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))

    line, = ax_curve.plot([], [], marker="o")
    ax_curve.set_xlabel("Cubes removed (%)")
    ax_curve.set_ylabel("Balanced accuracy (outer test)")
    ax_curve.grid(True)
    ax_curve.set_title(f"2D SOTF Greedy Remove ({cv_label})\nClassifier: {classifier}, Feature: {feature_type}")

    # No artificial chance point here; starts at 0% removed (all cubes), then grows
    order_positions = defaultdict(Counter)
    mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
    sns.heatmap(mode_matrix, vmin=1, vmax=n_iterations,
                cmap="viridis", ax=ax_heat, cbar=True,
                xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
                yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)])
    ax_heat.set_title("Most common removal order")
    ax_heat.invert_yaxis()

    all_outer_curves, all_removed_orders = [], []
    seeds = [random_state + k for k in range(len(outer_splits))]

    for fold_id, (fold_curve, removed_order) in enumerate(
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run_one_outer_fold)(fid, tr, te, seed)
            for fid, ((tr, te), seed) in enumerate(zip(outer_splits, seeds), start=1)
        ), start=1
    ):
        all_outer_curves.append(fold_curve)
        all_removed_orders.append(removed_order)

        # Update accuracy curve: x = % removed
        max_len = max(len(c) for c in all_outer_curves)
        avg_curve = np.array([
            np.mean([c[k] for c in all_outer_curves if len(c) > k])
            for k in range(max_len)
        ])
        # Percent removed from 0 to 100 across steps
        x = np.linspace(0, 100 * max_len / len(all_cubes), len(avg_curve))
        y = avg_curve
        line.set_data(x, y)
        ax_curve.relim(); ax_curve.autoscale_view()

        # Update heatmap with removal order stats
        for step, (i, j) in enumerate(removed_order):
            order_positions[(i, j)][step + 1] += 1

        mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
        annotations = np.full(mode_matrix.shape, "", dtype=object)
        for (i, j), counter in order_positions.items():
            most_common_step, freq = counter.most_common(1)[0]
            mode_matrix[j, i] = most_common_step
            percentage = (freq / len(all_outer_curves)) * 100
            annotations[j, i] = f"{most_common_step}|{percentage:.0f}%"

        ax_heat.clear()
        sns.heatmap(
            mode_matrix, vmin=1, vmax=n_iterations,
            cmap="viridis", ax=ax_heat, cbar=False,
            xticklabels=[f"{rt_edges[x]}–{rt_edges[x+1]}" for x in range(n_rt_bins)],
            yticklabels=[f"{mz_edges[y]}–{mz_edges[y+1]}" for y in range(n_mz_bins)],
            annot=annotations, fmt="", annot_kws={"size": 6}
        )
        ax_heat.set_title(f"Most common removal order (after {len(all_outer_curves)} folds)")
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
        ax_heat.invert_yaxis()

        # Window title with progress
        try:
            fig.canvas.manager.set_window_title(f"Greedy Remove Progress: Fold {fold_id}/{len(outer_splits)}")
        except Exception:
            pass

        plt.pause(0.3)

    plt.ioff()
    plt.show()

    return all_outer_curves, all_removed_orders


def run_sotf_add_noleak(
    data3d,
    labels,
    raw_sample_labels,
    year_labels,
    classifier,
    wine_kind,
    class_by_year,
    strategy,
    dataset_origins,
    cv_type,
    num_repeats,
    normalize_flag,
    region,
    feature_type,
    n_rt_bins=10,
    n_mz_bins=10,
    max_cubes=None,
    n_jobs=-1,
    random_state=42,
    sample_frac=1.0,
    rt_min=None, rt_max=None, mz_min=None, mz_max=None,
    cube_repr="tic",
):
    """
    2D Greedy Add (SOTF), leak-free, with:
      - Parallel outer CV folds (joblib)
      - Sequential inner candidate scoring (no oversubscription)
      - Append-only concat for feature_type='concat_channels' (all cube_repr modes)
      - Live plot updated once per finished outer fold:
          * Accuracy curve with first point at 0% = chance (1/C)
          * Heatmap of most-common cube order with % occurrence
          * Figure window title shows Fold x/total
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter, defaultdict
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
    from joblib import Parallel, delayed

    rng = np.random.default_rng(random_state)

    n_samples, n_time, n_channels = data3d.shape

    # === Crop data (same semantics) ===
    rt_min = 0 if rt_min is None else max(0, rt_min)
    rt_max = n_time if rt_max is None else min(n_time, rt_max)
    mz_min = 0 if mz_min is None else max(0, mz_min)
    mz_max = n_channels if mz_max is None else min(n_channels, mz_max)
    data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]
    n_samples, n_time, n_channels = data3d.shape

    # === Cube grid ===
    rt_edges = np.linspace(0, n_time, n_rt_bins + 1, dtype=int)
    mz_edges = np.linspace(0, n_channels, n_mz_bins + 1, dtype=int)
    rt_edges[-1] = n_time
    mz_edges[-1] = n_channels
    all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]

    if max_cubes is None:
        max_cubes = len(all_cubes)
    n_iterations = max_cubes

    y_all = (year_labels if class_by_year else labels)
    classes = np.unique(y_all)
    n_classes = len(classes)

    # === Precompute per-cube features ONLY for concat_channels; honor cube_repr exactly ===
    precomputed_features = None
    if feature_type == "concat_channels":
        precomputed_features = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            cube = data3d[:, rt_start:rt_end, mz_start:mz_end]  # (N, rt, mz)

            cr = "flat" if cube_repr == "concatenate" else cube_repr
            if cr == "flat":
                feats = cube.reshape(n_samples, -1)
            elif cr == "tic":
                feats = np.sum(cube, axis=2)  # sum over m/z -> (N, rt)
            elif cr == "tis":
                feats = np.sum(cube, axis=1)  # sum over rt  -> (N, mz)
            elif cr == "tic_tis":
                tic = np.sum(cube, axis=2)    # (N, rt)
                tis = np.sum(cube, axis=1)    # (N, mz)
                feats = np.hstack([tic, tis])
            else:
                raise ValueError(f"Unsupported cube_repr: {cube_repr}")

            precomputed_features[(i, j)] = feats.astype(np.float32, copy=False)
    else:
        # Keep original masked-path behavior for other feature types
        subcubes = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)

    # === Prepare classification labels ===
    if class_by_year:
            y_for_split = np.asarray(year_labels)
    else:
        y_for_split = np.asarray([str(lbl)[0] for lbl in labels])

    # === Outer CV (same as your repeated stratified) ===
    dummy_X = np.zeros((n_samples, 1))
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_repeats, random_state=random_state)
    outer_splits = list(outer_cv.split(dummy_X, y_for_split))
    cv_label = f"RepeatedStratified (5x{num_repeats})"

    # === Safe train/eval wrapper (with your exception fallback) ===
    def safe_train_eval(Xtr, ytr, Xte, yte, train_idx):
        if Xtr.shape[1] == 0 or Xte.shape[1] == 0:
            return {"balanced_accuracy": 1.0 / n_classes}
        try:
            cls_wrap = Classifier(
                Xtr, ytr,
                classifier_type=classifier,
                wine_kind=wine_kind,
                class_by_year=class_by_year,
                year_labels=(np.array(year_labels)[train_idx]
                             if (year_labels is not None and class_by_year) else None),
                strategy=strategy,
                sample_labels=np.array(raw_sample_labels)[train_idx] if raw_sample_labels is not None else None,
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type="standard",
                region=region,
                random_seed=random_state,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte,
                y_test=yte,
            )
            return res
        except ValueError as e:
            if "At least one label specified must be in y_true" in str(e):
                from sklearn.metrics import (
                    accuracy_score, balanced_accuracy_score,
                    precision_score, recall_score, f1_score, confusion_matrix
                )
                # NOTE: cls_wrap is defined above; model was trained; use it to predict
                y_pred = cls_wrap.classifier.predict(Xte)
                observed_labels = np.unique(np.concatenate([yte, y_pred]))
                cm = confusion_matrix(yte, y_pred, labels=observed_labels)
                return {
                    "accuracy": accuracy_score(yte, y_pred),
                    "balanced_accuracy": balanced_accuracy_score(yte, y_pred),
                    "precision": precision_score(yte, y_pred, average="weighted", zero_division=0),
                    "recall": recall_score(yte, y_pred, average="weighted", zero_division=0),
                    "f1": f1_score(yte, y_pred, average="weighted", zero_division=0),
                    "confusion_matrix": cm,
                }
            else:
                raise

    # === Inner CV factory (same as your options) ===
    def make_inner_cv(y_outer_train, raw_outer):
        if cv_type == "LOO":
            return LeaveOneOut().split(np.zeros((len(y_outer_train), 1)), y_outer_train)
        elif cv_type == "stratified":
            return StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)\
                   .split(np.zeros((len(y_outer_train), 1)), y_outer_train)
        elif cv_type == "LOOPC":
            return loopc_splits(
                y_outer_train,
                num_repeats=num_repeats, random_state=random_state,
                class_by_year=class_by_year,
                raw_labels=raw_outer if class_by_year else None,
            )
        else:
            raise ValueError(f"Unsupported inner cv_type: {cv_type}")

    # === One outer fold worker ===
    def run_one_outer_fold(fold_id, outer_train_idx, outer_test_idx, fold_seed):
        rng_local = np.random.default_rng(fold_seed)

        y_train_outer = y_all[outer_train_idx]
        y_test_outer  = y_all[outer_test_idx]
        raw_outer = np.array(raw_sample_labels)[outer_train_idx] if raw_sample_labels is not None else None

        if cv_type in ("LOO", "LOOPC"):
            _, y_train_outer, raw_outer = filter_singletons(
                data3d[outer_train_idx],
                y_train_outer,
                raw_labels=raw_outer,
                class_by_year=class_by_year,
            )

        # (Optional) inner singleton filtering like your original (done inside scoring)
        inner_cv = list(make_inner_cv(y_train_outer, raw_outer))

        added_cubes, not_added = [], all_cubes.copy()
        cube_sequence, outer_curve = [], []

        # Running design matrices ALIGNED TO OUTER TRAIN/TEST ORDER
        X_train_running = None  # shape: (len(outer_train_idx), d)
        X_test_running  = None  # shape: (len(outer_test_idx),  d)

        from tqdm import trange
        for step in trange(
                n_iterations,
                desc=f"Fold {fold_id}",  # label with fold ID/seed
                leave=False,  # so only the outer bar stays
        ):
            if len(added_cubes) >= max_cubes:
                break

            # Candidate subset (sequential eval)
            if sample_frac < 1.0:
                n_sample = max(1, int(len(not_added) * sample_frac))
                cand = rng_local.choice(not_added, size=n_sample, replace=False)
                candidates = [tuple(c) for c in cand]
            else:
                candidates = not_added

            best_cube, best_score = None, -np.inf

            for cube in candidates:
                candidate_add = added_cubes + [cube]
                accs = []

                # Pre-slice per-candidate feats for speed (concat_channels path)
                if feature_type == "concat_channels":
                    feats_c = precomputed_features[cube]
                    feats_train_c = feats_c[outer_train_idx]  # (n_train, d_c)
                # else: masked path handled per fold below

                for tr_rel, val_rel in inner_cv:
                    tr_idx = outer_train_idx[tr_rel]
                    val_idx = outer_train_idx[val_rel]

                    if feature_type == "concat_channels":
                        if X_train_running is None:
                            X_tr = feats_train_c[tr_rel]
                            X_val = feats_train_c[val_rel]
                        else:
                            X_tr = np.concatenate((X_train_running[tr_rel], feats_train_c[tr_rel]), axis=1)
                            X_val = np.concatenate((X_train_running[val_rel], feats_train_c[val_rel]), axis=1)
                    else:
                        # Build masked data for candidate_add, then compute_features (original semantics)
                        masked = np.zeros_like(data3d)
                        for (ii, jj) in candidate_add:
                            rts, rte, mzs, mze = subcubes[(ii, jj)]
                            masked[:, rts:rte, mzs:mze] = data3d[:, rts:rte, mzs:mze]
                        X_tr = compute_features(masked[tr_idx], feature_type=feature_type)
                        X_val = compute_features(masked[val_idx], feature_type=feature_type)

                    y_tr = y_all[tr_idx]
                    y_val = y_all[val_idx]
                    res = safe_train_eval(X_tr, y_tr, X_val, y_val, tr_idx)
                    accs.append(res["balanced_accuracy"])

                mean_acc = float(np.mean(accs))
                if mean_acc > best_score:
                    best_score, best_cube = mean_acc, cube

            # Commit best cube
            added_cubes.append(best_cube)
            not_added.remove(best_cube)
            cube_sequence.append(best_cube)

            # Update running outer TRAIN/TEST matrices once for chosen cube
            if feature_type == "concat_channels":
                feats_best = precomputed_features[best_cube]
                feats_train_best = feats_best[outer_train_idx]
                feats_test_best  = feats_best[outer_test_idx]
                if X_train_running is None:
                    X_train_running = feats_train_best
                    X_test_running  = feats_test_best
                else:
                    X_train_running = np.concatenate((X_train_running, feats_train_best), axis=1)
                    X_test_running  = np.concatenate((X_test_running,  feats_test_best),  axis=1)
                X_train, X_test = X_train_running, X_test_running
            else:
                masked = np.zeros_like(data3d)
                for (ii, jj) in added_cubes:
                    rts, rte, mzs, mze = subcubes[(ii, jj)]
                    masked[:, rts:rte, mzs:mze] = data3d[:, rts:rte, mzs:mze]
                X_train = compute_features(masked[outer_train_idx], feature_type=feature_type)
                X_test  = compute_features(masked[outer_test_idx],  feature_type=feature_type)

            res_outer = safe_train_eval(X_train, y_train_outer, X_test, y_test_outer, outer_train_idx)
            outer_curve.append(res_outer["balanced_accuracy"])

        return outer_curve, cube_sequence

    # === Live plotting (curve + heatmap), updated once per finished fold ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))

    line, = ax_curve.plot([], [], marker="o")
    ax_curve.set_xlabel("Cubes added (%)")
    ax_curve.set_ylabel("Balanced accuracy (outer test)")
    ax_curve.grid(True)
    ax_curve.set_title(f"2D SOTF Greedy Add ({cv_label})\nClassifier: {classifier}, Feature: {feature_type}")

    baseline_acc = 1.0 / n_classes
    line.set_data([0], [baseline_acc])  # 0% point = chance; no horizontal chance line

    order_positions = defaultdict(Counter)
    mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
    sns.heatmap(mode_matrix, vmin=1, vmax=n_iterations,
                cmap="viridis", ax=ax_heat, cbar=True,
                xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
                yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)])
    ax_heat.set_title("Most common cube order")
    ax_heat.invert_yaxis()

    all_outer_curves, all_selected_cubes = [], []
    seeds = [random_state + k for k in range(len(outer_splits))]

    for fold_id, (fold_curve, fold_cubes) in enumerate(
        Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(run_one_outer_fold)(fid, tr, te, seed)
            for fid, ((tr, te), seed) in enumerate(zip(outer_splits, seeds), start=1)
        ), start=1
    ):
        all_outer_curves.append(fold_curve)
        all_selected_cubes.append(fold_cubes)

        # Update accuracy curve (prepend 0% chance point)
        max_len = max(len(c) for c in all_outer_curves)
        avg_curve = np.array([
            np.mean([c[k] for c in all_outer_curves if len(c) > k])
            for k in range(max_len)
        ])
        x = np.linspace(0, 100, len(avg_curve) + 1)
        y = np.concatenate(([baseline_acc], avg_curve))
        line.set_data(x, y)
        ax_curve.relim(); ax_curve.autoscale_view()

        # Update heatmap counts and redraw with annotations
        for step, (i, j) in enumerate(fold_cubes):
            order_positions[(i, j)][step + 1] += 1

        mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
        annotations = np.full(mode_matrix.shape, "", dtype=object)
        for (i, j), counter in order_positions.items():
            most_common_step, freq = counter.most_common(1)[0]
            mode_matrix[j, i] = most_common_step
            percentage = (freq / len(all_outer_curves)) * 100
            annotations[j, i] = f"{most_common_step}|{percentage:.0f}%"

        ax_heat.clear()
        sns.heatmap(
            mode_matrix, vmin=1, vmax=n_iterations,
            cmap="viridis", ax=ax_heat, cbar=False,
            xticklabels=[f"{rt_edges[x]}–{rt_edges[x+1]}" for x in range(n_rt_bins)],
            yticklabels=[f"{mz_edges[y]}–{mz_edges[y+1]}" for y in range(n_mz_bins)],
            annot=annotations, fmt="", annot_kws={"size": 6}
        )
        ax_heat.set_title(f"Most common cube order (after {len(all_outer_curves)} folds)")
        ax_heat.invert_yaxis()

        # Update figure window title with fold progress (kept off the axis title)
        try:
            fig.canvas.manager.set_window_title(f"Greedy Add Progress: Fold {fold_id}/{len(outer_splits)}")
        except Exception:
            pass  # some backends may not support set_window_title

        plt.pause(0.3)

    plt.ioff()
    plt.show()

    return all_outer_curves, all_selected_cubes


# def run_sotf_add_2d(
#     data3d,
#     labels,
#     raw_sample_labels,
#     year_labels,
#     classifier,
#     wine_kind,
#     class_by_year,
#     strategy,
#     dataset_origins,
#     cv_type,
#     num_repeats,
#     normalize_flag,
#     region,
#     feature_type,
#     n_rt_bins=10,
#     n_mz_bins=10,
#     max_cubes=None,
#     n_jobs=-1,
#     random_state=42,
#     sample_frac=1.0,
#     rt_min=None, rt_max=None, mz_min=None, mz_max=None,
#     cube_repr="concatenate"
# ):
#     """
#     2D Greedy Add (SOTF):
#       - Iteratively adds cubes, evaluates accuracy.
#       - Shows survival curve and cube addition order heatmap.
#       - Optimised for 'concatenated' via PCA compression per cube.
#       - TIC/TIS/TIC+TIS recomputed with masking to preserve semantics.
#     """
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#     from sklearn.decomposition import PCA
#     from joblib import Parallel, delayed
#
#
#
#     # === Singleton filtering ===
#     if cv_type in ("LOO", "LOOPC"):
#         data3d, labels, raw_sample_labels = filter_singletons(
#             data3d,
#             year_labels if class_by_year else labels,
#             raw_labels=raw_sample_labels,
#             class_by_year=class_by_year,
#         )
#         if class_by_year:
#             year_labels = labels
#
#     n_samples, n_time, n_channels = data3d.shape
#
#     # === Crop data ===
#     rt_min = 0 if rt_min is None else max(0, rt_min)
#     rt_max = n_time if rt_max is None else min(n_time, rt_max)
#     mz_min = 0 if mz_min is None else max(0, mz_min)
#     mz_max = n_channels if mz_max is None else min(n_channels, mz_max)
#
#     data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]
#     n_samples, n_time, n_channels = data3d.shape
#
#     # === Define cube grid ===
#     rt_edges = np.linspace(0, n_time, n_rt_bins + 1, dtype=int)
#     mz_edges = np.linspace(0, n_channels, n_mz_bins + 1, dtype=int)
#     rt_edges[-1] = n_time
#     mz_edges[-1] = n_channels
#     # rt_edges = np.linspace(rt_min, rt_max, n_rt_bins + 1, dtype=int)
#     # mz_edges = np.linspace(mz_min, mz_max, n_mz_bins + 1, dtype=int)
#     # rt_edges[-1] = rt_max
#     # mz_edges[-1] = mz_max
#
#     all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]
#     added_cubes, not_added = [], all_cubes.copy()
#
#     if max_cubes is None:
#         max_cubes = len(all_cubes)
#
#     n_iterations = max_cubes
#     n_classes = len(np.unique(year_labels if class_by_year else labels))
#
#     # === Precompute features ===
#     precomputed_features, subcubes = None, None
#     if feature_type == "concatenated":
#         precomputed_features = {}
#         for (i, j) in all_cubes:
#             rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
#             mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
#             cube = data3d[:, rt_start:rt_end, mz_start:mz_end]
#
#             if cube_repr == "flat":
#                 features = cube.reshape(n_samples, -1)
#
#             elif cube_repr == "tic":
#                 # sum across m/z (axis=2), keep RT structure
#                 features = np.sum(cube, axis=2)
#
#             elif cube_repr == "tis":
#                 # sum across RT (axis=1), keep spectrum
#                 features = np.sum(cube, axis=1)
#
#             elif cube_repr == "tic_tis":
#                 tic = np.sum(cube, axis=2)
#                 tis = np.sum(cube, axis=1)
#                 features = np.hstack([tic, tis])
#
#             else:
#                 raise ValueError(f"Unsupported cube_repr: {cube_repr}")
#
#             precomputed_features[(i, j)] = features
#     else:
#         subcubes = {}
#         for (i, j) in all_cubes:
#             rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i + 1] - rt_min
#             mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j + 1] - mz_min
#             subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)
#
#     # === Outer CV splitter ===
#     dummy_X = np.zeros((n_samples, 1))
#     if cv_type == "LOO":
#         outer_splits, cv_label = list(LeaveOneOut().split(dummy_X, labels)), "LOO"
#     elif cv_type == "LOOPC":
#         outer_splits, cv_label = loopc_splits(
#             year_labels if class_by_year else labels,
#             num_repeats=num_repeats, random_state=random_state,
#             class_by_year=class_by_year,
#             raw_labels=raw_sample_labels if class_by_year else None,
#         ), "LOOPC"
#     elif cv_type == "stratified":
#         y_for_split = year_labels if class_by_year else labels
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#         outer_splits, cv_label = list(cv.split(dummy_X, y_for_split)), "Stratified"
#     else:
#         raise ValueError(f"Unsupported cv_type: {cv_type}")
#
#     # === Plot setup ===
#     plt.ion()
#     fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
#
#     line, = ax_curve.plot([], [], marker="o")
#     ax_curve.set_xlabel("Percentage of cubes added (%)")
#     ax_curve.set_ylabel("Accuracy")
#     ax_curve.set_title(f"2D SOTF Greedy Add ({cv_label} CV)")
#     ax_curve.grid(True)
#     ax_curve.set_xlim(0, 100)
#     ax_curve.set_ylim(0, 1)
#
#     addition_order = np.full((n_mz_bins, n_rt_bins), np.nan)
#     annotations = np.full((n_mz_bins, n_rt_bins), "", dtype=object)
#     sns.heatmap(
#         addition_order, vmin=0, vmax=n_iterations, cmap="viridis",
#         ax=ax_heat, cbar=False,
#         xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#         yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)],
#         annot=annotations, fmt="",
#         annot_kws={"size": 8}
#     )
#     ax_heat.set_title("Cube addition order (cold = early)")
#     ax_heat.invert_yaxis()
#     plt.pause(0.1)
#
#     accuracies, percent_added = [], []
#
#     # === Safe train/eval ===
#     def safe_train_eval(Xtr, ytr, Xte, yte, train_idx):
#         if Xtr.shape[1] == 0 or Xte.shape[1] == 0:
#             return {"balanced_accuracy": 1.0 / n_classes}
#         try:
#             cls_wrap = Classifier(
#                 Xtr, ytr, classifier_type=classifier,
#                 wine_kind=wine_kind, class_by_year=class_by_year,
#                 year_labels=(np.array(year_labels)[train_idx]
#                              if (year_labels is not None and class_by_year) else None),
#                 strategy=strategy, sample_labels=np.array(raw_sample_labels)[train_idx],
#                 dataset_origins=dataset_origins,
#             )
#             res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
#                 normalize=normalize_flag, scaler_type="standard",
#                 region=region, random_seed=random_state,
#                 test_size=0.2, LOOPC=False, projection_source=False,
#                 X_test=Xte, y_test=yte,
#             )
#             return res
#         except np.linalg.LinAlgError:
#             return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}
#
#     # === Baseline (no cubes) ===
#     baseline_acc = 1.0 / n_classes
#     accuracies.append(baseline_acc)
#     percent_added.append(0.0)
#     line.set_data(percent_added, accuracies)
#     plt.pause(0.2)
#
#     rng = np.random.default_rng(random_state)
#     best_global_score, best_global_cube = baseline_acc, None
#
#     # === Greedy loop ===
#     for step in range(n_iterations):
#         if len(added_cubes) >= max_cubes: break
#
#         # Candidate subset
#         if sample_frac < 1.0:
#             n_sample = max(1, int(len(not_added) * sample_frac))
#             candidates = rng.choice(not_added, size=n_sample, replace=False)
#             candidates = [tuple(c) for c in candidates]
#         else:
#             candidates = not_added
#
#         # Evaluate candidates
#         def eval_candidate(cube):
#             candidate_add = added_cubes + [cube]
#             fold_accs = []
#             for train_idx, test_idx in outer_splits:
#                 if feature_type == "concatenated":
#                     X_train = np.hstack([precomputed_features[c][train_idx] for c in candidate_add])
#                     X_test  = np.hstack([precomputed_features[c][test_idx] for c in candidate_add])
#                 else:
#                     masked = np.zeros_like(data3d)
#                     for (i, j) in candidate_add:
#                         rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
#                         masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
#                     X_train = utils.compute_features(masked[train_idx], feature_type=feature_type)
#                     X_test  = utils.compute_features(masked[test_idx], feature_type=feature_type)
#                 y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
#                 y_test  = year_labels[test_idx] if class_by_year else labels[test_idx]
#                 res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
#                 fold_accs.append(res["balanced_accuracy"])
#             return cube, np.mean(fold_accs)
#
#         candidate_scores = Parallel(n_jobs=n_jobs)(
#             delayed(eval_candidate)(cube) for cube in candidates
#         )
#
#         best_cube, best_score = max(candidate_scores, key=lambda x: x[1])
#         added_cubes.append(best_cube); not_added.remove(best_cube)
#
#         pct_data = (len(added_cubes) / len(all_cubes)) * 100
#         accuracies.append(best_score); percent_added.append(pct_data)
#
#         if best_score > best_global_score:
#             best_global_score, best_global_cube = best_score, best_cube
#
#         # Update curve
#         line.set_data(percent_added, accuracies)
#         ax_curve.set_xlim(0, max(percent_added) + 5); plt.pause(0.2)
#
#         # Update heatmap annotations
#         (i, j) = best_cube
#         addition_order[j, i] = step + 1
#         for x in range(n_rt_bins):
#             for y in range(n_mz_bins):
#                 if not np.isnan(addition_order[y, x]):
#                     annotations[y, x] = f"{int(addition_order[y, x])}"
#         if best_global_cube is not None:
#             bi, bj = best_global_cube
#             annotations[bj, bi] = f"{int(addition_order[bj, bi])}\n{best_global_score:.2f}"
#
#         ax_heat.clear()
#         sns.heatmap(
#             addition_order, vmin=0, vmax=len(all_cubes),
#             cmap="viridis", ax=ax_heat, cbar=False,
#             xticklabels=[f"{rt_edges[x]}–{rt_edges[x+1]}" for x in range(n_rt_bins)],
#             yticklabels=[f"{mz_edges[y]}–{mz_edges[y+1]}" for y in range(n_mz_bins)],
#             annot=annotations, fmt="",
#             annot_kws={"size": 8}
#         )
#         for text in ax_heat.texts:
#             x, y = text.get_position()
#             i, j = int(round(x - 0.5)), int(round(y - 0.5))
#             val = addition_order[j, i]
#             if not np.isnan(val):
#                 norm_val = (val - 0) / (len(all_cubes) - 0)
#                 rgba = plt.cm.viridis(norm_val)
#                 luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
#                 text.set_color("white" if luminance < 0.5 else "black")
#
#         ax_heat.set_title("Cube addition order (cold = early)")
#         ax_heat.invert_yaxis(); plt.pause(0.1)
#
#     plt.ioff(); plt.show()
#     return accuracies, percent_added, addition_order


# === Entrypoint (loads config & routes to the selected mode) ===
if __name__ == "__main__":
    # === Load Config ===
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parameters from config file
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]
    selected_paths = [dataset_directories[name] for name in selected_datasets]

    if not all("bordeaux" in path.lower() for path in selected_paths):
        raise ValueError("Please use this script for Bordeaux datasets.")

    wine_kind = utils.infer_wine_kind(selected_datasets, dataset_directories)

    # Run Parameters
    sotf_ret_time_flag = config.get("sotf_ret_time")
    sotf_remove_2d_flag = config.get("sotf_remove_2d")
    sotf_add_2d_flag = config.get("sotf_add_2d")
    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_repeats = config["num_repeats"]
    normalize_flag = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    class_by_year = config["class_by_year"]
    region = "winery"
    show_confusion_matrix = config["show_confusion_matrix"]
    retention_time_range = config["rt_range"]
    cv_type = config["cv_type"]
    rt_bins = config["n_rt_bins"]
    mz_bins = config["n_mz_bins"]

    # Survival parameters
    n_bins = 50
    min_bins = 1

    # Projection plotting parameters
    plot_projection = config.get("plot_projection", False)
    projection_method = config.get("projection_method", "UMAP").upper()
    projection_source = config.get("projection_source", False) if plot_projection else False
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)
    color_by_country = config.get("color_by_country", False)
    show_sample_names = config.get("show_sample_names", False)  # kept for parity
    invert_x = config.get("invert_x", False)
    invert_y = config.get("invert_y", False)

    # === Load and preprocess data ===
    cl = ChromatogramAnalysis(ndec=n_decimation)
    data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
    chrom_length = len(list(data_dict.values())[0])

    if retention_time_range:
        min_rt = retention_time_range['min'] // n_decimation
        raw_max_rt = retention_time_range['max'] // n_decimation
        max_rt = min(raw_max_rt, chrom_length)
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}

    data_dict, _ = utils.remove_zero_variance_channels(data_dict)
    gcms = GCMSDataProcessor(data_dict)
    if sync_state:
        _, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
        gcms = GCMSDataProcessor(data_dict)

    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    raw_sample_labels = labels.copy()
    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)
    strategy = get_strategy_by_wine_kind(wine_kind, class_by_year=class_by_year)

    # === Define binning (needed if SOTF) ===
    bin_ranges = split_into_bins(data, n_bins)

    # === Route by mode ===
    if sotf_ret_time_flag:
        run_sotf_ret_time(
            data=data,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cv_type,
            num_repeats=num_repeats,
            normalize_flag=normalize_flag,
            region=region,
            feature_type=feature_type,
            projection_source=projection_source,
            show_confusion_matrix=show_confusion_matrix,
            n_bins=n_bins,
            min_bins=min_bins,
            # projection params:
            plot_projection=plot_projection,
            projection_method=projection_method,
            projection_dim=projection_dim,
            n_neighbors=n_neighbors,
            random_state=random_state,
            color_by_country=color_by_country,
            show_sample_names=show_sample_names,
            invert_x=invert_x,
            invert_y=invert_y,
        )
    elif sotf_remove_2d_flag:
        results = run_sotf_remove_noleak(
            data3d=data,  # shape (samples, RT, m/z)
            labels=labels,  # class labels
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cv_type,  # "LOO", "LOOPC", or "stratified"
            num_repeats=num_repeats,
            normalize_flag=normalize_flag,
            region=region,
            feature_type=feature_type,
            n_rt_bins=rt_bins,  # number of retention time bins
            n_mz_bins=mz_bins,  # number of m/z bins
            random_state=42,
        )
    elif sotf_add_2d_flag:
        # results = run_sotf_add_leaky(
        results = run_sotf_add_noleak(
            data3d=data,  # shape (samples, RT, m/z)
            labels=labels,  # class labels
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cv_type,  # "LOO", "LOOPC", or "stratified"
            num_repeats=num_repeats,
            normalize_flag=normalize_flag,
            region=region,
            feature_type=feature_type,
            n_rt_bins=rt_bins,  # number of retention time bins
            n_mz_bins=mz_bins,  # number of m/z bins
            random_state=42,
            n_jobs=-1,
            mz_min=0, mz_max=181,
            cube_repr = "tic"
        )
    else:
        run_classification(
            data=data,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cv_type,
            num_repeats=num_repeats,
            normalize_flag=normalize_flag,
            region=region,
            feature_type=feature_type,
            projection_source=projection_source,
            show_confusion_matrix=show_confusion_matrix,
            # projection params:
            plot_projection=plot_projection,
            projection_method=projection_method,
            projection_dim=projection_dim,
            n_neighbors=n_neighbors,
            random_state=random_state,
            color_by_country=color_by_country,
            invert_x=invert_x,
            invert_y=invert_y,
        )


