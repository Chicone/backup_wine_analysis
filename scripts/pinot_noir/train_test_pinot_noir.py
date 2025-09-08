"""
Refactored version of train_test_pinot_noir.py
- Cleanly separates: (A) Normal classification, (B) SOTF over retention time, (C) SOTF over m/z channels
- Preserves all existing functionality, plotting, and printing behaviors
- Keeps LOO / LOOPC / Stratified exactly as in original

"""
import os
import time

import yaml
import numpy as np
import matplotlib.pyplot as plt

from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import (
    GCMSDataProcessor,
    ChromatogramAnalysis,
    process_labels_by_wine_kind,
)
from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
from gcmswine.utils import (
    string_to_latex_confusion_matrix,
    string_to_latex_confusion_matrix_modified,
)
from gcmswine.logger_setup import logger, logger_raw
from sklearn.preprocessing import normalize
from gcmswine.dimensionality_reduction import DimensionalityReducer
from scripts.bordeaux.train_test_bordeaux import run_sotf_ret_time
from scripts.pinot_noir.plotting_pinot_noir import plot_pinot_noir

# -----------------------------
# Utility helpers (unchanged logic)
# -----------------------------

def split_into_bins(data: np.ndarray, n_bins: int):
    """Split TIC into uniform bins (segments). Returns list of (start, end)."""
    total_points = data.shape[1]
    bin_edges = np.linspace(0, total_points, n_bins + 1, dtype=int)
    return [(bin_edges[i], bin_edges[i + 1]) for i in range(n_bins)]



def remove_bins(data, bins_to_remove, bin_ranges):
    """Remove (drop) specified bins by slicing out their columns."""
    import numpy as np
    mask = np.ones(data.shape[1], dtype=bool)
    for b in bins_to_remove:
        start, end = bin_ranges[b]  # end assumed exclusive
        mask[start:end] = False
    return data[:, mask]

# def remove_bins(data: np.ndarray, bins_to_remove, bin_ranges):
#     """Zero out specified bins by index, returning a copy."""
#     data_copy = data.copy()
#     for b in bins_to_remove:
#         start, end = bin_ranges[b]
#         data_copy[:, start:end] = 0
#     return data_copy

def remove_channels(data3d, channels_to_remove):
    """
    Keep only the selected m/z channels from a 3D dataset. """
    keep = [c for c in range(data3d.shape[1]) if c not in channels_to_remove]
    return data3d[:, keep, :]


def dict_to_array3d(d):
    """Stack per-sample 2D chromatograms into (N, T, C)."""
    arrs = []
    for v in d.values():
        v = np.asarray(v)
        if v.ndim == 1:
            raise ValueError("sotf_mz requires per-channel data (time × channels) per sample.")
        arrs.append(v)
    return np.stack(arrs, axis=0)


def run_cv(
    cls: Classifier,
    cv_type: str,
    num_repeats: int,
    normalize_flag: bool,
    region: str,
    feature_type: str,
    classifier: str,
    projection_source,
    show_confusion_matrix: bool,
):
    """Wrapper to apply selected CV method."""
    if cv_type in ["LOOPC", "stratified"]:
        loopc = (cv_type == "LOOPC")
        return cls.train_and_evaluate_all_channels(
            num_repeats=num_repeats,
            random_seed=42,
            test_size=0.2,
            normalize=normalize_flag,
            scaler_type="standard",
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
        mean_acc, std_acc, all_scores, all_labels, test_sample_names, all_preds = \
            cls.train_and_evaluate_leave_one_out_all_samples(
                normalize=normalize_flag,
                scaler_type="standard",
                region=region,
                feature_type=feature_type,
                classifier_type=classifier,
                projection_source=projection_source,
                show_confusion_matrix=show_confusion_matrix,
            )

        import matplotlib.pyplot as plt
        import numpy as np
        from gcmswine.classification import assign_origin_to_pinot_noir

        # Convert lists to arrays
        y_true = np.array(all_labels, dtype=int)
        y_pred = np.array(all_preds, dtype=int)

        # Color by region if available
        if region is not None:
            # Map test sample names (e.g. 'C14', 'M08') to their regions
            regions = assign_origin_to_pinot_noir(test_sample_names, split_burgundy_ns=False)
            regions = np.array(regions)

            unique_regions = np.unique(regions)
            colors = {r: plt.cm.tab10(i % 10) for i, r in enumerate(unique_regions)}
            markers = ["o", "s", "D", "^", "v", "P", "X", "*", "<", ">"]
            markers_map = {r: markers[i % len(markers)] for i, r in enumerate(unique_regions)}

            plt.figure(figsize=(7, 7))
            for r in unique_regions:
                mask = regions == r
                plt.scatter(
                    y_true[mask], y_pred[mask],
                    c=[colors[r]], marker=markers_map[r],
                    alpha=0.7, label=r
                )
        else:
            plt.figure(figsize=(7, 7))
            plt.scatter(y_true, y_pred, c="blue", marker="o", alpha=0.7,  label="All")

        # perfect diagonal
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", label="Perfect prediction")

        plt.xlabel("True Year")
        plt.ylabel("Predicted Year")
        plt.title("True vs Predicted (LOO)")
        plt.legend(title="Regions", loc="best")
        plt.grid(True)
        plt.show()
        # return cls.train_and_evaluate_leave_one_out_all_samples(
        #     normalize=normalize_flag,
        #     scaler_type="standard",
        #     region=region,
        #     feature_type=feature_type,
        #     classifier_type=classifier,
        #     projection_source=projection_source,
        #     show_confusion_matrix=show_confusion_matrix,
        # )
    else:
        raise ValueError(f"Invalid cross-validation type: '{cv_type}'.")


# -----------------------------
# Data / config loading
# -----------------------------

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_prepare_data(config):
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]

    selected_paths = [dataset_directories[name] for name in selected_datasets]
    if not all("pinot_noir" in path.lower() for path in selected_paths):
        raise ValueError("Please use this script for Pinot Noir datasets.")

    wine_kind = utils.infer_wine_kind(selected_datasets, dataset_directories)

    # Alignment / decimation setup
    n_decimation = config["n_decimation"]
    retention_time_range = config["rt_range"]

    cl = ChromatogramAnalysis(ndec=n_decimation)

    # Join & remove zero-variance channels
    data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
    chrom_length = len(list(data_dict.values())[0])

    if retention_time_range:
        min_rt = retention_time_range["min"] // n_decimation
        raw_max_rt = retention_time_range["max"] // n_decimation
        max_rt = min(raw_max_rt, chrom_length)
        logger.info(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}

    data_dict, _ = utils.remove_zero_variance_channels(data_dict)
    gcms = GCMSDataProcessor(data_dict)

    return wine_kind, cl, gcms, data_dict, dataset_origins


# -----------------------------
# Modes
# -----------------------------

def run_normal_classification(
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
):
    """Single evaluation without SOTF masking."""
    cls = Classifier(
        data,
        labels,
        classifier_type=classifier,
        wine_kind=wine_kind,
        class_by_year=class_by_year,
        year_labels=np.array(year_labels),
        strategy=strategy,
        sample_labels=np.array(raw_sample_labels),
        dataset_origins=dataset_origins,
    )

    result = run_cv(
        cls,
        cv_type=cv_type,
        num_repeats=num_repeats,
        normalize_flag=normalize_flag,
        region=region,
        feature_type=feature_type,
        classifier=classifier,
        projection_source=projection_source,
        show_confusion_matrix=show_confusion_matrix,
    )

    if cv_type == "LOO":
        mean_acc, std_acc, scores, all_labels, test_samples_names = result
    else:
        mean_acc, std_acc, *rest = result

    logger.info(f"Final Accuracy (no survival): {mean_acc:.3f}")
    return result


from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import cross_val_score

import re
from collections import defaultdict

def loopc_splits(
    class_labels,
    num_repeats=1,
    random_state=None,
    *,
    class_by_year=False,
    raw_labels=None,
    prefix_regex=r'^[A-Za-z]+'  # when class_by_year=False, 'A1990' -> 'A'
):
    """
    Generate LOOPC (Leave-One-Sample-Per-Class) splits.

    Modes
    -----
    - class_by_year = True:
        class_labels = array-like of years (one per sample)
        raw_labels   = replicate-aware IDs (e.g. '00ML-B-11-1'), required
        -> For each class (year), select ONE base sample per repeat,
           putting ALL its replicates in test.

    - class_by_year = False:
        class_labels = class labels (can be like 'A1990'); we collapse to
        alphabetic prefix by default so 'A1990' -> 'A'. Override with prefix_regex.

    Returns
    -------
    splits : list of (train_idx, test_idx)
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(class_labels)

    def collapse_alpha_prefix(arr, regex):
        pat = re.compile(regex)
        out = []
        for s in map(str, arr):
            m = pat.match(s)
            out.append(m.group(0) if m else s)
        return np.asarray(out, dtype=object)

    splits = []

    if class_by_year:
        if raw_labels is None:
            raise ValueError("loopc_splits: raw_labels must be provided when class_by_year=True")
        raw_labels = np.asarray(raw_labels)

        # base sample -> indices of replicates
        base_to_indices = defaultdict(list)
        for i, sid in enumerate(raw_labels):
            s = str(sid)
            base = s.rsplit("-", 1)[0] if "-" in s else s  # strip final '-rep'
            base_to_indices[base].append(i)

        # class -> list of BASES (not indices)
        class_to_bases = defaultdict(list)
        for base, idxs in base_to_indices.items():
            cls = y[idxs[0]]  # assume replicates share class
            class_to_bases[cls].append(base)

        # need ≥2 bases per class
        bad = {cls: len(bases) for cls, bases in class_to_bases.items() if len(bases) < 2}
        if bad:
            detail = ", ".join(f"{cls}: {n}" for cls, n in bad.items())
            raise ValueError(f"LOOPC requires ≥2 base samples per class; found {{{detail}}}")

        # shuffle bases deterministically
        for cls, bases in class_to_bases.items():
            arr = np.array(bases, dtype=object)
            rng.shuffle(arr)
            class_to_bases[cls] = arr

        for r in range(max(1, num_repeats)):
            # pick one BASE per class (cycle)
            chosen_bases = [arr[r % len(arr)] for arr in class_to_bases.values()]

            # expand to replicate indices
            test_idx = []
            for base in chosen_bases:
                test_idx.extend(base_to_indices[base])

            test_idx = np.array(test_idx, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            splits.append((train_idx, test_idx))

    else:
        # collapse labels like 'A1990' -> 'A' for grouping
        y_eff = collapse_alpha_prefix(y, prefix_regex)

        # class -> indices
        class_to_indices = defaultdict(list)
        for i, cls in enumerate(y_eff):
            class_to_indices[cls].append(i)

        # need ≥2 samples per class
        bad = {cls: len(ix) for cls, ix in class_to_indices.items() if len(ix) < 2}
        if bad:
            detail = ", ".join(f"{cls}: {n}" for cls, n in bad.items())
            raise ValueError(f"LOOPC requires ≥2 samples per class; found {{{detail}}}")

        # shuffle within each class
        for cls, ix in class_to_indices.items():
            arr = np.array(ix, dtype=int)
            rng.shuffle(arr)
            class_to_indices[cls] = arr

        for r in range(max(1, num_repeats)):
            # pick one index per class (cycle)
            test_idx = [arr[r % len(arr)] for arr in class_to_indices.values()]
            test_idx = np.array(test_idx, dtype=int)
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.nonzero(train_mask)[0]
            splits.append((train_idx, test_idx))

    return splits

# def loopc_splits(y, num_repeats=1, random_state=None):
#     """
#     Generate LOOPC (Leave-One-Out Per Class) splits.
#
#     Each split leaves exactly one sample per class in the test set,
#     with the rest in the training set.
#
#     Parameters
#     ----------
#     y : array-like of shape (n_samples,)
#         Class labels for the samples.
#     num_repeats : int, default=1
#         Number of random repeats.
#     random_state : int or None
#         Seed for reproducibility.
#
#     Returns
#     -------
#     splits : list of (train_idx, test_idx) tuples
#         Each tuple contains arrays of train and test indices.
#     """
#     rng = np.random.default_rng(random_state)
#     y = np.array(y)
#     classes = np.unique(y)
#     n_samples = len(y)
#
#     splits = []
#     for _ in range(num_repeats):
#         test_idx = []
#         for c in classes:
#             class_indices = np.where(y == c)[0]
#             chosen = rng.choice(class_indices)
#             test_idx.append(chosen)
#         test_idx = np.array(test_idx)
#         train_idx = np.setdiff1d(np.arange(n_samples), test_idx)
#         splits.append((train_idx, test_idx))
#
#     return splits

from gcmswine.utils import compute_features
def estimate_bin_score(data, labels, bins, bin_ranges,
                       classifier, wine_kind,
                       class_by_year, year_labels,
                       strategy, raw_sample_labels,
                       dataset_origins, cv_type, num_repeats,
                       normalize_flag, region,
                       feature_type, projection_source,
                       show_confusion_matrix):
    """Estimate accuracy after removing bins using inner CV (corrected bin handling)."""

    # --- choose labels explicitly ---
    if class_by_year:
        y = np.array(year_labels)
    elif region == "winery":
        y = np.array(raw_sample_labels)
    else:  # default: origin
        y = np.array(labels)

    # --- build keep_indices from bin ranges, then mask ---
    keep_indices = []
    for i, (start, end) in enumerate(bin_ranges):
        if i in bins:  # keep this bin
            keep_indices.extend(range(start, end))
    keep_indices = np.array(keep_indices)
    masked = data[:, keep_indices, :]  # keep times + all channels

    X = compute_features(masked, feature_type=feature_type)

    base_model = RidgeClassifier()
    if normalize_flag:
        model = make_pipeline(StandardScaler(with_mean=False), base_model)
    else:
        model = base_model

    rng = np.random.RandomState(42)
    scores = []

    if cv_type == "Stratified":
        # run num_repeats independent stratified splits (different seeds)
        n_splits = 5  # keep light; adjust if you need 5
        for rep in range(max(1, num_repeats)):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(rng.randint(0,1e9)))
            rep_scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
            scores.extend(rep_scores)

    elif cv_type == "LOO":
        # single full LOO pass; num_repeats is ignored
        cv = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)

    elif cv_type == "LOOPC":
        # exactly num_repeats folds; each has 1 sample per class in test
        cv = loopc_splits(y, num_repeats=max(1, num_repeats), random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)

    else:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    return np.mean(scores)


def estimate_channel_score(data3d, labels, channels,
                           classifier, wine_kind,
                           class_by_year, year_labels,
                           strategy, raw_sample_labels,
                           dataset_origins, cv_type, num_repeats,
                           normalize_flag, region,
                           feature_type, projection_source,
                           show_confusion_matrix):
    """Estimate accuracy after removing channels using inner CV (optimized)."""

    # --- choose labels explicitly ---
    if class_by_year:
        y = np.array(year_labels)
    elif region == "winery":
        y = np.array(raw_sample_labels)
    else:  # default: origin
        y = np.array(labels)

    # --- mask channels using keep indices ---
    keep_channels = [c for c in range(data3d.shape[1]) if c in channels]
    masked = data3d[:, keep_channels, :].reshape(data3d.shape[0], -1)

    if masked.ndim > 2:
        masked = masked.reshape(masked.shape[0], -1)

    # --- classifier ---
    if classifier == "RGC":
        base_model = RidgeClassifier()
    else:
        raise ValueError(f"Unsupported classifier {classifier}")

    if normalize_flag:
        model = make_pipeline(StandardScaler(with_mean=False), base_model)
    else:
        model = base_model

    # --- CV strategy ---
    if cv_type == "Stratified":
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    elif cv_type == "LOO":
        cv = LeaveOneOut()
    else:
        raise ValueError(f"Unsupported CV type: {cv_type}")

    # --- scoring ---
    scores = []
    for _ in range(max(1, num_repeats)):
        cv_scores = cross_val_score(model, masked, y, cv=cv, n_jobs=-1)
        scores.extend(cv_scores)

    return np.mean(scores)


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


def ensure_2d(X):
    """Flatten to 2D if input has more than 2 dimensions."""
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


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
):
    """
    Leakage-free greedy removal over m/z retention-time bins with nested CV:
      - Outer CV provides unbiased accuracy.
      - Inner greedy loop removes bins using all outer folds (global decision).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold

    # 1) Precompute features ONCE (so bin ranges index the original feature columns)
    X_proc = compute_features(data, feature_type=feature_type)
    bin_ranges = split_into_bins(X_proc, n_bins)  # list of (start, end) over ORIGINAL columns
    active_bins = list(range(n_bins))
    n_iterations = n_bins - min_bins + 1

    baseline_sum = np.sum(data)
    n_classes = len(np.unique(year_labels if class_by_year else labels))
    baseline_nonzero = np.count_nonzero(X_proc)
    n_classes = len(np.unique(year_labels if class_by_year else labels))

    # remove single samples in classes
    if cv_type in ("LOO", "LOOPC"):
        X_proc, labels, raw_sample_labels = filter_singletons(
            X_proc,
            year_labels if class_by_year else labels,
            raw_labels=raw_sample_labels,
            class_by_year=class_by_year
        )
        if class_by_year:
            year_labels = labels  # already filtered

    # 2) Build outer CV splits on the FULL (unmasked) X_proc and labels
    if cv_type == "LOO":
        outer_splits = list(LeaveOneOut().split(X_proc, labels))
        cv_label = "LOO"
    elif cv_type == "LOOPC":
        # Uses your loopc_splits() (replicate-aware if class_by_year=True)
        outer_splits = loopc_splits(
            year_labels if class_by_year else labels,
            num_repeats=num_repeats,
            random_state=42,
            class_by_year=class_by_year,
            raw_labels=raw_sample_labels if class_by_year else None,
        )
        cv_label = "LOOPC"
    elif cv_type == "stratified":
        y_for_split = year_labels if class_by_year else labels
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_splits = list(cv.split(X_proc, y_for_split))
        cv_label = "Stratified"
    else:
        raise ValueError(f"Unsupported cv_type: {cv_type}")

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

    # --- helper: safe train/eval that returns chance-level if features are empty
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
                sample_labels=np.array(raw_sample_labels)[train_idx],
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                random_seed=42,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte,
                y_test=yte
            )
            return res

        except np.linalg.LinAlgError as e:
            logger.warning(f"Fold failed due to {e}; returning chance-level acc.")
            return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}

    # 3) Baseline (all bins active) evaluated with outer CV
    baseline_fold_accs = []
    for train_idx, test_idx in outer_splits:
        X_train, X_test = X_proc[train_idx], X_proc[test_idx]
        y_train = (year_labels[train_idx] if class_by_year else labels[train_idx])
        y_test  = (year_labels[test_idx]  if class_by_year else labels[test_idx])
        res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
        baseline_fold_accs.append(res["balanced_accuracy"])

    baseline_acc = float(np.mean(baseline_fold_accs))
    accuracies.append(baseline_acc)
    percent_remaining.append(100.0)
    line.set_data(percent_remaining, accuracies)
    plt.draw(); plt.pause(0.2)
    logger.info(f"Baseline (all bins): mean outer acc = {baseline_acc:.3f}")

    # 4) Greedy loop (global decision using all outer folds)
    for step in range(n_iterations):
        if len(active_bins) <= min_bins:
            break

        logger.info(f"=== Iteration {step+1}/{n_iterations} | Active bins: {len(active_bins)} ===")
        already_removed = [b for b in range(n_bins) if b not in active_bins]
        candidate_scores = []

        for b in active_bins:
            bins_to_remove = already_removed + [b]
            fold_accs = []

            for train_idx, test_idx in outer_splits:
                X_train_full, X_test_full = X_proc[train_idx], X_proc[test_idx]
                y_train = (year_labels[train_idx] if class_by_year else labels[train_idx])
                y_test  = (year_labels[test_idx]  if class_by_year else labels[test_idx])

                # Drop columns using ORIGINAL bin_ranges
                X_train = remove_bins(X_train_full, bins_to_remove=bins_to_remove, bin_ranges=bin_ranges)
                X_test  = remove_bins(X_test_full,  bins_to_remove=bins_to_remove, bin_ranges=bin_ranges)

                res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
                fold_accs.append(res["balanced_accuracy"])

            candidate_scores.append((b, float(np.mean(fold_accs))))

        # choose best bin to remove
        best_bin, best_score = max(candidate_scores, key=lambda x: x[1])
        active_bins.remove(best_bin)

        # removed = [b for b in range(n_bins) if b not in active_bins]
        # masked_raw = remove_bins(data, bins_to_remove=removed, bin_ranges=bin_ranges_raw)
        # pct_data = (np.sum(masked_raw) / baseline_sum) * 100

        # progress point (apply all removed so far to FULL matrix)
        masked_full = remove_bins(
            X_proc,
            bins_to_remove=[b for b in range(n_bins) if b not in active_bins],
            bin_ranges=bin_ranges
        )
        # pct_data = (np.sum(masked_full) / baseline_sum) * 100 if masked_full.shape[1] > 0 else 0.0
        # pct_data = (np.count_nonzero(masked_full) / baseline_nonzero) * 100
        pct_data = (len(active_bins) / n_bins) * 100


        accuracies.append(best_score)
        percent_remaining.append(pct_data)

        # dynamic tick density so labels don’t clutter
        span = 100 - (min(percent_remaining) - 5)
        if span > 50:
            step_size = 10
        elif span > 20:
            step_size = 5
        else:
            step_size = 2
        ax.xaxis.set_major_locator(ticker.MultipleLocator(step_size))

        # update plot
        line.set_data(percent_remaining, accuracies)
        ax.set_xlim(100, min(percent_remaining) - 5)
        ax.set_ylim(0, 1)
        plt.draw(); plt.pause(0.2)

        logger.info(f"Iteration {step+1}: removed bin {best_bin}, "
                    f"mean outer acc = {best_score:.3f}, % data = {pct_data:.1f}")

        # if we actually hit zero features (min_bins=0), append chance-level once and stop
        if len(active_bins) == 0:
            if percent_remaining[-1] != 0.0:
                accuracies.append(1.0 / n_classes)
                percent_remaining.append(0.0)
                line.set_data(percent_remaining, accuracies)
                plt.draw();
                plt.pause(0.2)
            logger.info("All bins removed → terminating at chance-level.")
            break
        # if masked_full.shape[1] == 0:
        #     # ensure last point recorded (0% at chance)
        #     if percent_remaining[-1] != 0.0:
        #         accuracies.append(1.0 / n_classes)
        #         percent_remaining.append(0.0)
        #         line.set_data(percent_remaining, accuracies)
        #         plt.draw(); plt.pause(0.2)
        #     logger.info("All bins removed → terminating at chance-level.")
        #     break

    plt.ioff()
    plt.show()
    return accuracies, percent_remaining


from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut, StratifiedKFold

def run_sotf_mz(
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
    projection_source,
    show_confusion_matrix,
    min_channels: int = 1,
    group_size: int = 1,
    n_jobs: int = -1,
):
    """
    Greedy survival-of-the-fittest over m/z channels.
    Removes entire channels (all retention times), nested CV to avoid leakage.
    Parallelized candidate evaluation with joblib.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if cv_type in ("LOO", "LOOPC"):
        data3d, labels, raw_sample_labels = filter_singletons(
            data3d,
            year_labels if class_by_year else labels,
            raw_labels=raw_sample_labels,
            class_by_year=class_by_year
        )
        if class_by_year:
            year_labels = labels  # already filtered

    n_samples, n_timepoints, n_channels = data3d.shape
    active_channels = list(range(n_channels))
    n_iterations = (len(active_channels) - min_channels) // group_size + 1
    # n_iterations = len(active_channels) - min_channels + 1
    n_classes = len(np.unique(year_labels if class_by_year else labels))

    # === Outer CV splitter (like sotf_ret_time) ===
    X_full = compute_features(data3d, feature_type=feature_type)
    if cv_type == "LOO":
        outer_splits = list(LeaveOneOut().split(X_full, labels))
        cv_label = "LOO"
    elif cv_type == "LOOPC":
        outer_splits = loopc_splits(
            year_labels if class_by_year else labels,
            num_repeats=num_repeats,
            random_state=42,
            class_by_year=class_by_year,
            raw_labels=raw_sample_labels if class_by_year else None,
        )
        cv_label = "LOOPC"
    elif cv_type == "stratified":
        y_for_split = year_labels if class_by_year else labels
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_splits = list(cv.split(X_full, y_for_split))
        cv_label = "Stratified"
    else:
        raise ValueError(f"Unsupported cv_type: {cv_type}")

    # === Plot setup ===
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], marker='o')
    ax.set_xlabel("Percentage of m/z channels remaining (%)")
    ax.set_ylabel("Balanced Accuracy")
    ax.set_title(f"Greedy Survival: Accuracy vs m/z channels ({cv_label} CV)")
    ax.grid(True)
    ax.set_xlim(100, 0)

    accuracies, percent_remaining = [], []

    # --- Helper: train/eval safely
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
                sample_labels=np.array(raw_sample_labels)[train_idx],
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                random_seed=42,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte,
                y_test=yte
            )
            return res
        except np.linalg.LinAlgError:
            return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}

    # --- Baseline (all channels active)
    baseline_accs = []
    for train_idx, test_idx in outer_splits:
        X_train = compute_features(data3d[train_idx][:, :, active_channels], feature_type=feature_type)
        X_test  = compute_features(data3d[test_idx][:, :, active_channels],  feature_type=feature_type)
        y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
        y_test  = year_labels[test_idx]  if class_by_year else labels[test_idx]
        res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
        baseline_accs.append(res["balanced_accuracy"])
    baseline_acc = np.mean(baseline_accs)

    accuracies.append(baseline_acc)
    percent_remaining.append(100.0)
    line.set_data(percent_remaining, accuracies)
    plt.draw(); plt.pause(0.2)

    logger.info(f"Baseline (all channels): mean outer acc = {baseline_acc:.3f}")

    # === Greedy loop ===
    for step in range(n_iterations):
        if len(active_channels) <= min_channels:
            break

        logger.info(f"=== Iteration {step + 1}/{n_iterations} | Active channels: {len(active_channels)} ===")

        # window size this iteration (avoid dropping below min_channels)
        g = min(group_size, len(active_channels) - min_channels)
        if g <= 0:
            break

        # freeze the snapshot of current channels
        active_snapshot = active_channels[:]

        # candidate starts are windows over the ACTIVE list
        starts = range(0, len(active_snapshot) - g + 1)

        def eval_candidate(start):
            remove_group = active_snapshot[start:start + g]  # contiguous in active list
            new_channels = [c for c in active_snapshot if c not in remove_group]
            fold_accs = []
            for train_idx, test_idx in outer_splits:
                X_train = compute_features(data3d[train_idx][:, :, new_channels], feature_type=feature_type)
                X_test = compute_features(data3d[test_idx][:, :, new_channels], feature_type=feature_type)
                y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
                y_test = year_labels[test_idx] if class_by_year else labels[test_idx]
                res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
                fold_accs.append(res["balanced_accuracy"])
            return remove_group, np.mean(fold_accs)

        # parallel eval over candidate windows
        candidate_scores = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(s) for s in starts
        )

        # pick best contiguous group and remove them
        best_group, best_score = max(candidate_scores, key=lambda x: x[1])
        active_channels = [c for c in active_channels if c not in best_group]

        # record
        pct_data = (len(active_channels) / n_channels) * 100
        accuracies.append(best_score)
        percent_remaining.append(pct_data)

        line.set_data(percent_remaining, accuracies)
        ax.set_xlim(100, min(percent_remaining) - 5)
        ax.set_ylim(0, 1)
        plt.draw();
        plt.pause(0.2)

        removed_str = f"{best_group[0]}–{best_group[-1]}" if len(best_group) > 1 else str(best_group[0])
        logger.info(
            f"Iteration {step + 1}: removed channels {removed_str}, "
            f"mean outer acc = {best_score:.3f}, % channels = {pct_data:.1f}"
        )

    plt.ioff(); plt.show()
    return accuracies, percent_remaining


################ Code for creating SOTF 2D ####################

def select_rt_bins_greedy(X, y, rt_bins, min_bins, cv_type, strategy_args):
    """
    Greedy selection of retention-time bins (given all channels).
    Returns indices of bins kept.
    """
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold

    classifier, wine_kind, class_by_year, strategy, dataset_origins, \
        num_repeats, normalize_flag, region, feature_type, show_confusion_matrix = strategy_args

    active_bins = list(range(len(rt_bins)))
    n_classes = len(np.unique(y))

    # outer CV
    if cv_type == "LOO":
        outer_splits = list(LeaveOneOut().split(X, y))
    elif cv_type == "stratified":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_splits = list(cv.split(X, y))
    else:
        raise ValueError(f"Unsupported cv_type for RT selection: {cv_type}")

    def safe_eval(Xtr, ytr, Xte, yte):
        if Xtr.shape[1] == 0: return 1.0 / n_classes
        try:
            cls = Classifier(Xtr, ytr,
                             classifier_type=classifier,
                             wine_kind=wine_kind,
                             class_by_year=class_by_year,
                             strategy=strategy,
                             sample_labels=None,
                             dataset_origins=dataset_origins)
            res, _, _, _ = cls.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                random_seed=42,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte, y_test=yte
            )
            return res["balanced_accuracy"]
        except Exception:
            return 1.0 / n_classes

    while len(active_bins) > min_bins:
        candidate_scores = []
        for b in active_bins:
            new_bins = [x for x in active_bins if x != b]
            fold_accs = []
            for tr, te in outer_splits:
                Xtr = X[tr][:, new_bins, :]  # keep only RT bins new_bins
                Xte = X[te][:, new_bins, :]
                fold_accs.append(safe_eval(Xtr.reshape(len(tr), -1), y[tr],
                                           Xte.reshape(len(te), -1), y[te]))
            candidate_scores.append((b, np.mean(fold_accs)))
        # remove the "worst" bin
        best_bin, best_score = max(candidate_scores, key=lambda x: x[1])
        active_bins.remove(best_bin)

    return active_bins


def select_mz_channels_greedy(X, y, rt_indices, channels, min_channels, cv_type, strategy_args):
    """
    Greedy selection of m/z channels with RT bins fixed.
    Returns indices of channels kept.
    """
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold

    classifier, wine_kind, class_by_year, strategy, dataset_origins, \
        num_repeats, normalize_flag, region, feature_type, show_confusion_matrix = strategy_args

    active_channels = list(channels)
    n_classes = len(np.unique(y))

    # outer CV
    if cv_type == "LOO":
        outer_splits = list(LeaveOneOut().split(X, y))
    elif cv_type == "stratified":
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        outer_splits = list(cv.split(X, y))
    else:
        raise ValueError(f"Unsupported cv_type for channel selection: {cv_type}")

    def safe_eval(Xtr, ytr, Xte, yte):
        if Xtr.shape[1] == 0: return 1.0 / n_classes
        try:
            cls = Classifier(Xtr, ytr,
                             classifier_type=classifier,
                             wine_kind=wine_kind,
                             class_by_year=class_by_year,
                             strategy=strategy,
                             sample_labels=None,
                             dataset_origins=dataset_origins)
            res, _, _, _ = cls.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                random_seed=42,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte, y_test=yte
            )
            return res["balanced_accuracy"]
        except Exception:
            return 1.0 / n_classes

    while len(active_channels) > min_channels:
        candidate_scores = []
        for c in active_channels:
            new_ch = [x for x in active_channels if x != c]
            fold_accs = []
            for tr, te in outer_splits:
                Xtr = X[tr][:, rt_indices, :][:, :, new_ch]
                Xte = X[te][:, rt_indices, :][:, :, new_ch]
                fold_accs.append(safe_eval(Xtr.reshape(len(tr), -1), y[tr],
                                           Xte.reshape(len(te), -1), y[te]))
            candidate_scores.append((c, np.mean(fold_accs)))
        best_ch, best_score = max(candidate_scores, key=lambda x: x[1])
        active_channels.remove(best_ch)

    return active_channels



def run_sotf_cube_heatmap(
    data3d,              # shape (samples, rt_points, mz_channels)
    labels,              # e.g. ["Bordeaux", "Bordeaux", "Pinot", ...]
    raw_sample_labels,   # replicate-aware IDs e.g. ["00ML-B-11-1", ...]
    year_labels,         # e.g. [2011, 2012, ...] (only used if class_by_year=True)
    classifier,          # sklearn classifier instance (e.g. RidgeClassifier())
    wine_kind,           # your tag like "pinot_noir"
    class_by_year,       # True → use year_labels for classification, False → use labels
    strategy,            # your strategy object (handles label extraction etc.)
    dataset_origins,     # origins if you track cross-dataset CV
    cv_type,             # "LOO", "LOOPC", or "stratified"
    num_repeats,         # used in LOOPC
    normalize_flag,      # True/False
    region,              # region string
    feature_type,        # "TIC", "profiles", etc.
    projection_source,   # unused here (pass None or False)
    show_confusion_matrix, # False (not needed here)
    rt_bin_counts,       # list of RT bin counts to evaluate, e.g. [10, 20, 30]
    channel_counts       # list of channel counts to evaluate, e.g. [20, 50, 100]
):
    """
    Bias-free two-stage greedy search:
    Stage 1 → RT bins on split A
    Stage 2 → Channels on split B
    Heatmap of accuracies across (RT bins × channels).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import train_test_split, LeaveOneOut

    # Pick labels
    y_all = np.array(year_labels if class_by_year else labels)

    # filter singletons
    if cv_type in ("LOO", "LOOPC"):
        data3d, y_all, raw_sample_labels = filter_singletons(
            data3d, y_all, raw_labels=raw_sample_labels, class_by_year=class_by_year
        )

    # split A/B
    idx = np.arange(len(y_all))
    train_idx, test_idx = train_test_split(idx, test_size=0.5, stratify=y_all, random_state=42)
    A, B = data3d[train_idx], data3d[test_idx]
    yA, yB = y_all[train_idx], y_all[test_idx]

    strategy_args = (
        classifier, wine_kind, class_by_year, strategy, dataset_origins,
        num_repeats, normalize_flag, region, feature_type, show_confusion_matrix
    )

    heatmap_data = np.zeros((len(rt_bin_counts), len(channel_counts)))

    for i, rt_keep in enumerate(rt_bin_counts):
        rt_bins_sel = select_rt_bins_greedy(A, yA, split_into_bins(A, rt_keep),
                                            min_bins=rt_keep, cv_type=cv_type,
                                            strategy_args=strategy_args)

        for j, ch_keep in enumerate(channel_counts):
            ch_sel = select_mz_channels_greedy(B, yB,
                                               rt_indices=rt_bins_sel,
                                               channels=list(range(data3d.shape[2])),
                                               min_channels=ch_keep,
                                               cv_type=cv_type,
                                               strategy_args=strategy_args)

            # final evaluation on B
            X_eval = B[:, rt_bins_sel, :][:, :, ch_sel].reshape(len(B), -1)
            accs = []
            for tr, te in LeaveOneOut().split(X_eval, yB):
                cls = Classifier(X_eval[tr], yB[tr],
                                 classifier_type=classifier,
                                 wine_kind=wine_kind,
                                 class_by_year=class_by_year,
                                 strategy=strategy,
                                 sample_labels=None,
                                 dataset_origins=dataset_origins)
                res, _, _, _ = cls.train_and_evaluate_balanced(
                    normalize=normalize_flag,
                    scaler_type="standard",
                    region=region,
                    random_seed=42,
                    test_size=0.2,
                    LOOPC=False,
                    projection_source=False,
                    X_test=X_eval[te], y_test=yB[te]
                )
                accs.append(res["balanced_accuracy"])
            mean_acc = np.mean(accs)

            heatmap_data[i, j] = mean_acc
            logger.info(f"RT={rt_keep}, CH={ch_keep} → {mean_acc:.3f}")

    plt.figure(figsize=(8,6))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f",
                xticklabels=channel_counts,
                yticklabels=rt_bin_counts, cmap="viridis")
    plt.xlabel("Channels kept")
    plt.ylabel("RT bins kept")
    plt.title("Balanced accuracy across RT–m/z subsets")
    plt.show()

    return heatmap_data

from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
#     full_X = compute_features(data3d, feature_type=feature_type)
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
#         X_train = compute_features(data3d[train_idx], feature_type=feature_type)
#         X_test  = compute_features(data3d[test_idx],  feature_type=feature_type)
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
#                 X_train = compute_features(masked_train, feature_type=feature_type)
#                 X_test  = compute_features(masked_test,  feature_type=feature_type)
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

# #### SEMI_LEAKY ###
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
#     2D Greedy Add (SOTF) with GLOBAL decision across folds:
#       - Outer CV folds provide unbiased accuracy
#       - At each iteration, candidate cubes are evaluated by average outer accuracy
#       - One canonical cube order is produced (same for all folds)
#       - Faster than fully nested version
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#     from joblib import Parallel, delayed
#
#     rng = np.random.default_rng(random_state)
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
#     # === Crop ===
#     rt_min = 0 if rt_min is None else max(0, rt_min)
#     rt_max = n_time if rt_max is None else min(n_time, rt_max)
#     mz_min = 0 if mz_min is None else max(0, mz_min)
#     mz_max = n_channels if mz_max is None else min(n_channels, mz_max)
#     data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]
#     n_samples, n_time, n_channels = data3d.shape
#
#     # === Define grid ===
#     rt_edges = np.linspace(rt_min, rt_max, n_rt_bins + 1, dtype=int)
#     mz_edges = np.linspace(mz_min, mz_max, n_mz_bins + 1, dtype=int)
#     rt_edges[-1] = rt_max
#     mz_edges[-1] = mz_max
#     all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]
#     if max_cubes is None:
#         max_cubes = len(all_cubes)
#     n_iterations = max_cubes
#     n_classes = len(np.unique(year_labels if class_by_year else labels))
#
#     # === Precompute features ===
#     precomputed_features, subcubes = None, None
#     if feature_type == "concat_channels":
#         precomputed_features = {}
#         for (i, j) in all_cubes:
#             rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i + 1] - rt_min
#             mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j + 1] - mz_min
#             cube = data3d[:, rt_start:rt_end, mz_start:mz_end]
#
#             if cube_repr == "flat":
#                 features = cube.reshape(n_samples, -1)
#             elif cube_repr == "tic":
#                 features = np.sum(cube, axis=2)
#             elif cube_repr == "tis":
#                 features = np.sum(cube, axis=1)
#             elif cube_repr == "tic_tis":
#                 tic = np.sum(cube, axis=2)
#                 tis = np.sum(cube, axis=1)
#                 features = np.hstack([tic, tis])
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
#     # === Outer CV splits ===
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
#     line, = ax_curve.plot([], [], marker="o")
#     ax_curve.set_xlabel("% of cubes added")
#     ax_curve.set_ylabel("Outer Accuracy")
#     ax_curve.set_title(f"2D SOTF Greedy Add ({cv_label} CV)\nClassifier={classifier}, Feature={feature_type}")
#     ax_curve.grid(True)
#
#     addition_order = np.full((n_mz_bins, n_rt_bins), np.nan)
#
#     accuracies = []
#     percent_added = []
#
#     # === Safe eval ===
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
#     baseline_accs = []
#     for train_idx, test_idx in outer_splits:
#         y_train = (year_labels if class_by_year else labels)[train_idx]
#         y_test  = (year_labels if class_by_year else labels)[test_idx]
#         baseline_accs.append(1.0 / n_classes)
#     baseline_acc = np.mean(baseline_accs)
#     accuracies.append(baseline_acc)
#     percent_added.append(0)
#
#     # === Greedy loop ===
#     added_cubes, not_added = [], all_cubes.copy()
#     for step in range(n_iterations):
#         if len(added_cubes) >= max_cubes:
#             break
#
#         # Candidate subset
#         if sample_frac < 1.0:
#             n_sample = max(1, int(len(not_added) * sample_frac))
#             candidates = rng.choice(not_added, size=n_sample, replace=False)
#             candidates = [tuple(c) for c in candidates]
#         else:
#             candidates = not_added
#
#         candidate_scores = []
#         for cube in candidates:
#             fold_accs = []
#             for train_idx, test_idx in outer_splits:
#                 if feature_type == "concat_channels":
#                     X_train = np.hstack([precomputed_features[c][train_idx] for c in added_cubes + [cube]])
#                     X_test  = np.hstack([precomputed_features[c][test_idx]  for c in added_cubes + [cube]])
#                 else:
#                     masked = np.zeros_like(data3d)
#                     for (i, j) in added_cubes + [cube]:
#                         rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
#                         masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
#                     X_train = compute_features(masked[train_idx], feature_type=feature_type)
#                     X_test  = compute_features(masked[test_idx], feature_type=feature_type)
#
#                 y_train = (year_labels if class_by_year else labels)[train_idx]
#                 y_test  = (year_labels if class_by_year else labels)[test_idx]
#                 res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
#                 fold_accs.append(res["balanced_accuracy"])
#             candidate_scores.append((cube, np.mean(fold_accs)))
#
#         # Select best cube globally
#         best_cube, best_score = max(candidate_scores, key=lambda x: x[1])
#         added_cubes.append(best_cube)
#         not_added.remove(best_cube)
#
#         # Record
#         accuracies.append(best_score)
#         pct = (len(added_cubes) / len(all_cubes)) * 100
#         percent_added.append(pct)
#         addition_order[best_cube[1], best_cube[0]] = step + 1
#
#         # Update plots
#         line.set_data(percent_added, accuracies)
#         ax_curve.relim(); ax_curve.autoscale_view()
#
#         ax_heat.clear()
#         sns.heatmap(
#             addition_order,
#             vmin=1, vmax=n_iterations,
#             cmap="viridis", ax=ax_heat, cbar=False,
#             xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#             yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)],
#             annot=True, fmt=".0f", annot_kws={"size": 7}
#         )
#         ax_heat.set_title("Cube addition order (global)")
#         ax_heat.invert_yaxis()
#         plt.pause(0.3)
#
#     plt.ioff()
#     plt.show()
#
#     return accuracies, percent_added, addition_order


# ##### LEAK_FREE SWAPPED CV######
def run_sotf_add_2d_fast(
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

    # === Outer CV (same as your repeated stratified) ===
    dummy_X = np.zeros((n_samples, 1))
    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_repeats, random_state=random_state)
    outer_splits = list(outer_cv.split(dummy_X, y_all))
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
                num_repeats=50, random_state=random_state,
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





def run_sotf_add_2d(
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
    2D Greedy Add (SOTF) with nested CV:
      - Outer CV: unbiased test accuracy
      - Inner CV: cube selection
      - Outer loop uses RepeatedStratifiedKFold for smoother estimates
      - Tracks per-fold curves + cubes
      - Updates running average accuracy + cube order live
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
    from joblib import Parallel, delayed
    from collections import Counter, defaultdict

    rng = np.random.default_rng(random_state)

    # === Singleton filtering (only if inner CV type requires it) ===
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
    if max_cubes is None:
        max_cubes = len(all_cubes)
    n_iterations = max_cubes
    n_classes = len(np.unique(year_labels if class_by_year else labels))

    # === Precompute features ===
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
                features = np.sum(cube, axis=2)
            elif cube_repr == "tis":
                features = np.sum(cube, axis=1)
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
    y_for_split = year_labels if class_by_year else labels

    # Instead of plain StratifiedKFold, use repeated stratified
    outer_cv = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=num_repeats, random_state=random_state
    )
    outer_splits = list(outer_cv.split(dummy_X, y_for_split))
    cv_label = f"RepeatedStratified (5x{num_repeats})"

    # === Storage ===
    all_outer_curves = []
    all_selected_cubes = []
    order_positions = defaultdict(Counter)

    # === Plot setup ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))

    line, = ax_curve.plot([], [], marker="o")
    ax_curve.set_xlabel("Percentage of cubes added (%)")
    ax_curve.set_ylabel("Outer Accuracy")
    ax_curve.set_title(
        f"2D SOTF Greedy Add ({cv_label})\nClassifier: {classifier}, Feature: {feature_type}"
    )
    ax_curve.grid(True)

    baseline_acc = 1.0 / n_classes
    line.set_data([0], [baseline_acc])

    heatmap = sns.heatmap(np.zeros((n_mz_bins, n_rt_bins)), vmin=1, vmax=n_iterations,
                          cmap="viridis", ax=ax_heat, cbar=True,
                          xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
                          yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)])
    ax_heat.set_title("Most common cube order")
    ax_heat.invert_yaxis()
    plt.pause(0.1)

    def safe_train_eval(Xtr, ytr, Xte, yte, train_idx, n_classes):
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
                sample_labels=np.array(raw_sample_labels)[train_idx],
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

    progress_text = ax_curve.text(
        0.95, 0.05, "Fold 0/0",
        transform=ax_curve.transAxes,
        ha="right", va="bottom",
        fontsize=10, color="black",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
    )

    # === Outer CV loop ===
    for fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
        added_cubes, not_added = [], all_cubes.copy()
        outer_curve = []
        cube_sequence = []

        # === Greedy loop ===
        for step in range(n_iterations):
            if len(added_cubes) >= max_cubes:
                break

            # Candidate subset
            if sample_frac < 1.0:
                n_sample = max(1, int(len(not_added) * sample_frac))
                candidates = rng.choice(not_added, size=n_sample, replace=False)
                candidates = [tuple(c) for c in candidates]
            else:
                candidates = not_added

            # === Inner CV eval (uses cv_type: LOO, LOOPC, stratified etc.) ===
            def eval_candidate(cube):
                candidate_add = added_cubes + [cube]
                fold_accs = []

                # Extract outer training subset
                y_outer_train = (year_labels if class_by_year else labels)[outer_train_idx]
                raw_outer = np.array(raw_sample_labels)[outer_train_idx] if raw_sample_labels is not None else None

                # Apply singleton filtering if needed
                if cv_type in ("LOO", "LOOPC"):
                    _, y_outer_train, raw_outer = filter_singletons(
                        data3d[outer_train_idx],  # not used later, so just discard X
                        y_outer_train,
                        raw_labels=raw_outer,
                        class_by_year=class_by_year,
                    )

                # Build inner CV on filtered labels
                if cv_type == "LOO":
                    inner_cv = LeaveOneOut().split(np.zeros((len(y_outer_train), 1)), y_outer_train)
                elif cv_type == "stratified":
                    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state).split(
                        np.zeros((len(y_outer_train), 1)), y_outer_train
                    )
                elif cv_type == "LOOPC":
                    inner_cv = loopc_splits(
                        y_outer_train,
                        num_repeats=num_repeats, random_state=random_state,
                        class_by_year=class_by_year,
                        raw_labels=raw_outer if class_by_year else None,
                    )
                else:
                    raise ValueError(f"Unsupported inner cv_type: {cv_type}")

                # Evaluate each inner fold
                for tr_rel, val_rel in inner_cv:
                    tr_idx = outer_train_idx[tr_rel]
                    val_idx = outer_train_idx[val_rel]

                    if feature_type == "concat_channels":
                        X_tr = np.hstack([precomputed_features[c][tr_idx] for c in candidate_add])
                        X_val = np.hstack([precomputed_features[c][val_idx] for c in candidate_add])
                    else:
                        masked = np.zeros_like(data3d)
                        for (i, j) in candidate_add:
                            rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
                            masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
                        X_tr = compute_features(masked[tr_idx], feature_type=feature_type)
                        X_val = compute_features(masked[val_idx], feature_type=feature_type)

                    y_tr = (year_labels if class_by_year else labels)[tr_idx]
                    y_val = (year_labels if class_by_year else labels)[val_idx]

                    res = safe_train_eval(X_tr, y_tr, X_val, y_val, tr_idx, n_classes)
                    fold_accs.append(res["balanced_accuracy"])

                return cube, np.mean(fold_accs)

            candidate_scores = Parallel(n_jobs=n_jobs)(
                delayed(eval_candidate)(cube) for cube in candidates
            )
            best_cube, _ = max(candidate_scores, key=lambda x: x[1])
            added_cubes.append(best_cube); not_added.remove(best_cube)
            cube_sequence.append(best_cube)

            # === Outer accuracy at this step ===
            if feature_type == "concat_channels":
                X_train = np.hstack([precomputed_features[c][outer_train_idx] for c in added_cubes])
                X_test  = np.hstack([precomputed_features[c][outer_test_idx] for c in added_cubes])
            else:
                masked = np.zeros_like(data3d)
                for (i, j) in added_cubes:
                    rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
                    masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
                X_train = compute_features(masked[outer_train_idx], feature_type=feature_type)
                X_test  = compute_features(masked[outer_test_idx], feature_type=feature_type)

            y_train = (year_labels if class_by_year else labels)[outer_train_idx]
            y_test  = (year_labels if class_by_year else labels)[outer_test_idx]

            res = safe_train_eval(X_train, y_train, X_test, y_test, outer_train_idx, n_classes)
            outer_curve.append(res["balanced_accuracy"])

        # === Store results for this fold ===
        all_outer_curves.append(outer_curve)
        all_selected_cubes.append(cube_sequence)

        # === Update running average accuracy ===
        max_len = max(len(c) for c in all_outer_curves)
        avg_curve = np.array([
            np.mean([c[k] for c in all_outer_curves if len(c) > k])
            for k in range(max_len)
        ])
        baseline_acc = 1.0 / n_classes
        x_vals = np.linspace(0, 100, len(avg_curve) + 1)
        y_vals = np.concatenate(([baseline_acc], avg_curve))
        line.set_data(x_vals, y_vals)
        ax_curve.relim(); ax_curve.autoscale_view()

        # === Progress text annotation ===
        progress_text.set_text(f"Fold {fold_id + 1}/{len(outer_splits)}")

        # === Update order positions ===
        for step, (i, j) in enumerate(cube_sequence):
            order_positions[(i, j)][step + 1] += 1

        # Build mode+percentage annotations
        mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
        annotations = np.full(mode_matrix.shape, "", dtype=object)
        for (i, j), counter in order_positions.items():
            most_common_step, freq = counter.most_common(1)[0]
            mode_matrix[j, i] = most_common_step
            percentage = (freq / len(all_outer_curves)) * 100
            annotations[j, i] = f"{most_common_step}|{percentage:.0f}%"

        # === Update heatmap dynamically ===
        ax_heat.clear()
        sns.heatmap(
            mode_matrix,
            vmin=1, vmax=n_iterations,
            cmap="viridis", ax=ax_heat, cbar=False,
            xticklabels=[f"{rt_edges[x]}–{rt_edges[x + 1]}" for x in range(n_rt_bins)],
            yticklabels=[f"{mz_edges[y]}–{mz_edges[y + 1]}" for y in range(n_mz_bins)],
            annot=annotations, fmt="",
            annot_kws={"size": 6}
        )
        ax_heat.set_title(f"Most common cube order (after {len(all_outer_curves)} folds)")
        ax_heat.invert_yaxis()
        plt.pause(0.3)

    plt.ioff()
    plt.show()

    return all_outer_curves, all_selected_cubes




# ##### LEAK_FREE ######
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
#     2D Greedy Add (SOTF) with nested CV:
#       - Outer CV for unbiased test accuracy
#       - Inner CV for cube selection
#       - Tracks per-fold curves + cubes
#       - Updates running average accuracy + cube order live
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#     from joblib import Parallel, delayed
#     from collections import Counter, defaultdict
#
#     rng = np.random.default_rng(random_state)
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
#     data3d = data3d[:, rt_min:rt_max, mz_min:mz_max]
#     n_samples, n_time, n_channels = data3d.shape
#
#     # === Define cube grid ===
#     rt_edges = np.linspace(rt_min, rt_max, n_rt_bins + 1, dtype=int)
#     mz_edges = np.linspace(mz_min, mz_max, n_mz_bins + 1, dtype=int)
#     rt_edges[-1] = rt_max
#     mz_edges[-1] = mz_max
#     all_cubes = [(i, j) for i in range(n_rt_bins) for j in range(n_mz_bins)]
#     if max_cubes is None:
#         max_cubes = len(all_cubes)
#     n_iterations = max_cubes
#     n_classes = len(np.unique(year_labels if class_by_year else labels))
#
#     # === Precompute features ===
#     precomputed_features, subcubes = None, None
#     if feature_type == "concat_channels":
#         precomputed_features = {}
#         for (i, j) in all_cubes:
#             rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i + 1] - rt_min
#             mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j + 1] - mz_min
#             cube = data3d[:, rt_start:rt_end, mz_start:mz_end]
#             if cube_repr == "flat":
#                 features = cube.reshape(n_samples, -1)
#             elif cube_repr == "tic":
#                 features = np.sum(cube, axis=2)
#             elif cube_repr == "tis":
#                 features = np.sum(cube, axis=1)
#             elif cube_repr == "tic_tis":
#                 tic = np.sum(cube, axis=2)
#                 tis = np.sum(cube, axis=1)
#                 features = np.hstack([tic, tis])
#             else:
#                 raise ValueError(f"Unsupported cube_repr: {cube_repr}")
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
#     # === Storage ===
#     all_outer_curves = []     # outer test curves per fold
#     all_selected_cubes = []   # cube sequence per fold
#     order_positions = defaultdict(Counter)  # (i,j) -> Counter of step positions
#
#     # === Plot setup ===
#     plt.ion()
#     fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
#
#     line, = ax_curve.plot([], [], marker="o")
#     ax_curve.set_xlabel("Percentage of cubes added (%)")
#     ax_curve.set_ylabel("Outer Accuracy")
#     ax_curve.set_title(
#         f"2D SOTF Greedy Add ({cv_label} CV)\nClassifier: {classifier}, Feature: {feature_type}"
#     )
#     ax_curve.grid(True)
#     # Baseline (chance level) as first point
#     baseline_acc = 1.0 / n_classes
#     line.set_data([0], [baseline_acc])
#
#     # Baseline (chance level)
#     line.set_data([0], [1.0 / n_classes])
#
#     heatmap = sns.heatmap(np.zeros((n_mz_bins, n_rt_bins)), vmin=1, vmax=n_iterations,
#                           cmap="viridis", ax=ax_heat, cbar=True,
#                           xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#                           yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)])
#     ax_heat.set_title("Most common cube order")
#     ax_heat.invert_yaxis()
#     plt.pause(0.1)
#
#     def safe_train_eval(Xtr, ytr, Xte, yte, train_idx, n_classes):
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
#     progress_text = ax_curve.text(
#         0.95, 0.05, "Fold 0/0",
#         transform=ax_curve.transAxes,
#         ha="right", va="bottom",
#         fontsize=10, color="black",
#         bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
#     )
#
#     # === Outer CV loop ===
#     for fold_id, (outer_train_idx, outer_test_idx) in enumerate(outer_splits):
#         added_cubes, not_added = [], all_cubes.copy()
#         outer_curve = []
#         cube_sequence = []
#
#         # === Greedy loop ===
#         for step in range(n_iterations):
#             if len(added_cubes) >= max_cubes:
#                 break
#
#             # Candidate subset
#             if sample_frac < 1.0:
#                 n_sample = max(1, int(len(not_added) * sample_frac))
#                 candidates = rng.choice(not_added, size=n_sample, replace=False)
#                 candidates = [tuple(c) for c in candidates]
#             else:
#                 candidates = not_added
#
#             # === Inner CV eval ===
#             def eval_candidate(cube):
#                 candidate_add = added_cubes + [cube]
#                 fold_accs = []
#                 inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
#                 y_outer_train = (year_labels if class_by_year else labels)[outer_train_idx]
#                 for tr_rel, val_rel in inner_cv.split(np.zeros((len(outer_train_idx), 1)), y_outer_train):
#                     tr_idx = outer_train_idx[tr_rel]
#                     val_idx = outer_train_idx[val_rel]
#                     if feature_type == "concat_channels":
#                         X_tr = np.hstack([precomputed_features[c][tr_idx] for c in candidate_add])
#                         X_val = np.hstack([precomputed_features[c][val_idx] for c in candidate_add])
#                     else:
#                         masked = np.zeros_like(data3d)
#                         for (i, j) in candidate_add:
#                             rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
#                             masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
#                         X_tr = compute_features(masked[tr_idx], feature_type=feature_type)
#                         X_val = compute_features(masked[val_idx], feature_type=feature_type)
#                     y_tr = (year_labels if class_by_year else labels)[tr_idx]
#                     y_val = (year_labels if class_by_year else labels)[val_idx]
#                     res = safe_train_eval(X_tr, y_tr, X_val, y_val, tr_idx, n_classes)
#                     fold_accs.append(res["balanced_accuracy"])
#                 return cube, np.mean(fold_accs)
#
#             candidate_scores = Parallel(n_jobs=n_jobs)(
#                 delayed(eval_candidate)(cube) for cube in candidates
#             )
#             best_cube, _ = max(candidate_scores, key=lambda x: x[1])
#             added_cubes.append(best_cube); not_added.remove(best_cube)
#             cube_sequence.append(best_cube)
#
#             # === Outer accuracy at this step ===
#             if feature_type == "concat_channels":
#                 X_train = np.hstack([precomputed_features[c][outer_train_idx] for c in added_cubes])
#                 X_test  = np.hstack([precomputed_features[c][outer_test_idx] for c in added_cubes])
#             else:
#                 masked = np.zeros_like(data3d)
#                 for (i, j) in added_cubes:
#                     rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
#                     masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
#                 X_train = compute_features(masked[outer_train_idx], feature_type=feature_type)
#                 X_test  = compute_features(masked[outer_test_idx], feature_type=feature_type)
#
#             y_train = (year_labels if class_by_year else labels)[outer_train_idx]
#             y_test  = (year_labels if class_by_year else labels)[outer_test_idx]
#
#             res = safe_train_eval(X_train, y_train, X_test, y_test, outer_train_idx, n_classes)
#             outer_curve.append(res["balanced_accuracy"])
#
#         # === Store results for this fold ===
#         all_outer_curves.append(outer_curve)
#         all_selected_cubes.append(cube_sequence)
#
#         # === Update running average accuracy ===
#         max_len = max(len(c) for c in all_outer_curves)
#         avg_curve = np.array([
#             np.mean([c[k] for c in all_outer_curves if len(c) > k])
#             for k in range(max_len)
#         ])
#         baseline_acc = 1.0 / n_classes
#         x_vals = np.linspace(0, 100, len(avg_curve) + 1)
#         y_vals = np.concatenate(([baseline_acc], avg_curve))
#         line.set_data(x_vals, y_vals)
#         ax_curve.relim(); ax_curve.autoscale_view()
#
#         # === Progress text annotation ===
#         progress_text.set_text(f"Fold {fold_id + 1}/{len(outer_splits)}")
#
#         # === Update order positions ===
#         for step, (i, j) in enumerate(cube_sequence):
#             order_positions[(i, j)][step + 1] += 1
#
#         # Build mode matrix (most frequent selection step)
#         mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
#         for (i, j), counter in order_positions.items():
#             most_common_step, _ = counter.most_common(1)[0]
#             mode_matrix[j, i] = most_common_step
#
#         # === Update heatmap dynamically ===
#         ax_heat.clear()
#         annotations = np.full(mode_matrix.shape, "", dtype=object)
#
#         for (i, j), counter in order_positions.items():
#             most_common_step, freq = counter.most_common(1)[0]
#             percentage = (freq / len(all_outer_curves)) * 100
#             annotations[j, i] = f"{most_common_step} \n {percentage:.0f}"
#
#         sns.heatmap(
#             mode_matrix,
#             vmin=1, vmax=n_iterations,
#             cmap="viridis", ax=ax_heat, cbar=False,
#             xticklabels=[f"{rt_edges[x]}–{rt_edges[x + 1]}" for x in range(n_rt_bins)],
#             yticklabels=[f"{mz_edges[y]}–{mz_edges[y + 1]}" for y in range(n_mz_bins)],
#             annot=annotations, fmt="",
#             annot_kws={"size": 5, "va": "center"}  # smaller font, inline
#         )
#         ax_heat.set_title(f"Most common cube order (after {len(all_outer_curves)} folds)")
#         ax_heat.invert_yaxis()
#         plt.pause(0.3)
#
#     plt.ioff()
#     plt.show()
#
#     return all_outer_curves, all_selected_cubes


##### LEAKY ######
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
#     rt_edges = np.linspace(rt_min, rt_max, n_rt_bins + 1, dtype=int)
#     mz_edges = np.linspace(mz_min, mz_max, n_mz_bins + 1, dtype=int)
#     rt_edges[-1] = rt_max
#     mz_edges[-1] = mz_max
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
#     if feature_type == "concat_channels":
#         precomputed_features = {}
#         for (i, j) in all_cubes:
#             rt_start, rt_end = rt_edges[i] - rt_min, rt_edges[i + 1] - rt_min
#             mz_start, mz_end = mz_edges[j] - mz_min, mz_edges[j + 1] - mz_min
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
#                 if feature_type == "concat_channels":
#                     X_train = np.hstack([precomputed_features[c][train_idx] for c in candidate_add])
#                     X_test  = np.hstack([precomputed_features[c][test_idx] for c in candidate_add])
#                 else:
#                     masked = np.zeros_like(data3d)
#                     for (i, j) in candidate_add:
#                         rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
#                         masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
#                     X_train = compute_features(masked[train_idx], feature_type=feature_type)
#                     X_test  = compute_features(masked[test_idx], feature_type=feature_type)
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



def run_sotf_add_lookahead_2d(
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
    cube_repr="concatenate",
    lambda_lookahead=0.5   # weight for lookahead (0=plain greedy add, 1=full 2-step lookahead)
):
    """
    2D Hybrid Greedy Add + Lookahead (SOTF):
      - Iteratively adds cubes, but scores each candidate as:
            immediate_gain + lambda * best_next_gain
      - Shows survival curve and cube addition order heatmap.
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
                features = np.sum(cube, axis=2)
            elif cube_repr == "tis":
                features = np.sum(cube, axis=1)
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
    ax_curve.set_title(f"2D Hybrid Greedy Add + Lookahead ({cv_label} CV)")
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

    # === Helper to build features from selected cubes ===
    def build_features(cubes, train_idx, test_idx):
        if feature_type == "concat_channels":
            X_train = np.hstack([precomputed_features[c][train_idx] for c in cubes])
            X_test  = np.hstack([precomputed_features[c][test_idx]  for c in cubes])
        else:
            masked = np.zeros_like(data3d)
            for (i, j) in cubes:
                rt_start, rt_end, mz_start, mz_end = subcubes[(i, j)]
                masked[:, rt_start:rt_end, mz_start:mz_end] = data3d[:, rt_start:rt_end, mz_start:mz_end]
            X_train = compute_features(masked[train_idx], feature_type=feature_type)
            X_test  = compute_features(masked[test_idx],  feature_type=feature_type)
        return X_train, X_test

    # === Baseline (no cubes) ===
    baseline_acc = 1.0 / n_classes
    accuracies.append(baseline_acc)
    percent_added.append(0.0)
    line.set_data(percent_added, accuracies)
    plt.pause(0.2)

    rng = np.random.default_rng(random_state)
    best_global_score, best_global_cube = baseline_acc, None

    # === Hybrid Greedy loop ===
    for step in range(n_iterations):
        if len(added_cubes) >= max_cubes:
            break

        # Candidate subset
        if sample_frac < 1.0:
            n_sample = max(1, int(len(not_added) * sample_frac))
            candidates = rng.choice(not_added, size=n_sample, replace=False)
            candidates = [tuple(c) for c in candidates]
        else:
            candidates = not_added

        def eval_candidate(cube):
            # --- immediate accuracy ---
            candidate_add = added_cubes + [cube]
            fold_accs = []
            for train_idx, test_idx in outer_splits:
                X_train, X_test = build_features(candidate_add, train_idx, test_idx)
                y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
                y_test  = year_labels[test_idx] if class_by_year else labels[test_idx]
                res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
                fold_accs.append(res["balanced_accuracy"])
            acc1 = np.mean(fold_accs)

            # --- lookahead (best possible second cube) ---
            best_acc2 = acc1
            for cube2 in not_added:
                if cube2 == cube:
                    continue
                candidate_add2 = candidate_add + [cube2]
                fold_accs2 = []
                for train_idx, test_idx in outer_splits:
                    X_train, X_test = build_features(candidate_add2, train_idx, test_idx)
                    y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
                    y_test  = year_labels[test_idx] if class_by_year else labels[test_idx]
                    res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
                    fold_accs2.append(res["balanced_accuracy"])
                acc2 = np.mean(fold_accs2)
                if acc2 > best_acc2:
                    best_acc2 = acc2

            score = acc1 + lambda_lookahead * (best_acc2 - acc1)
            return cube, acc1, score

        candidate_scores = Parallel(n_jobs=n_jobs)(
            delayed(eval_candidate)(cube) for cube in candidates
        )

        # Pick best by hybrid score
        best_cube, best_acc, best_score = max(candidate_scores, key=lambda x: x[2])
        added_cubes.append(best_cube)
        not_added.remove(best_cube)

        # Record
        pct_data = (len(added_cubes) / len(all_cubes)) * 100
        accuracies.append(best_acc)
        percent_added.append(pct_data)

        if best_acc > best_global_score:
            best_global_score, best_global_cube = best_acc, best_cube

        # Update plots
        line.set_data(percent_added, accuracies)
        ax_curve.set_xlim(0, max(percent_added) + 5)
        plt.pause(0.2)

        (i, j) = best_cube
        addition_order[j, i] = step + 1
        annotations[j, i] = f"{step+1}"
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
        ax_heat.set_title("Cube addition order (cold = early)")
        ax_heat.invert_yaxis()
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    return accuracies, percent_added, addition_order






################ Code for creating SOTF 2D ####################

def run_region_accuracy_heatmap(
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
    projection_source,
    show_confusion_matrix,
    n_rt_bins=5,
    n_mz_bins=5,
    random_state=42,
    n_jobs=-1,
):
    """
    RT × m/z discriminative map (parallelised, true dynamic updating).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold
    from joblib import Parallel, delayed

    acc_annot = False if n_rt_bins * n_mz_bins >= 400 else True

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

    # Bin edges
    rt_edges = np.linspace(0, n_time, n_rt_bins + 1, dtype=int)
    mz_edges = np.linspace(0, n_channels, n_mz_bins + 1, dtype=int)

    heatmap = np.full((n_mz_bins, n_rt_bins), np.nan)

    # CV splitter
    full_X = compute_features(data3d, feature_type=feature_type)
    if cv_type == "LOO":
        outer_splits = list(LeaveOneOut().split(full_X, labels))
    elif cv_type == "LOOPC":
        outer_splits = loopc_splits(
            year_labels if class_by_year else labels,
            num_repeats=num_repeats,
            random_state=random_state,
            class_by_year=class_by_year,
            raw_labels=raw_sample_labels if class_by_year else None,
        )
    elif cv_type == "stratified":
        y_for_split = year_labels if class_by_year else labels
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        outer_splits = list(cv.split(full_X, y_for_split))
    else:
        raise ValueError(f"Unsupported cv_type: {cv_type}")

    n_classes = len(np.unique(year_labels if class_by_year else labels))

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
                sample_labels=np.array(raw_sample_labels)[train_idx],
                dataset_origins=dataset_origins,
            )
            res, _, _, _ = cls_wrap.train_and_evaluate_balanced(
                normalize=normalize_flag,
                scaler_type='standard',
                region=region,
                random_seed=random_state,
                test_size=0.2,
                LOOPC=False,
                projection_source=False,
                X_test=Xte,
                y_test=yte
            )
            return res
        except np.linalg.LinAlgError:
            return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}

    def evaluate_cell(i, j):
        rt_start, rt_end = rt_edges[i], rt_edges[i+1]
        mz_start, mz_end = mz_edges[j], mz_edges[j+1]

        subcube = data3d[:, rt_start:rt_end, mz_start:mz_end]
        X = compute_features(subcube, feature_type=feature_type)

        fold_accs = []
        for train_idx, test_idx in outer_splits:
            y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
            y_test  = year_labels[test_idx]  if class_by_year else labels[test_idx]
            res = safe_train_eval(X[train_idx], y_train, X[test_idx], y_test, train_idx)
            fold_accs.append(res["balanced_accuracy"])

        return np.mean(fold_accs)

    # === Live plotting setup ===
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    mesh = sns.heatmap(
        heatmap, vmin=0, vmax=1, cmap="viridis", ax=ax, cbar=True,
        xticklabels=[f"{rt_edges[col]}–{rt_edges[col+1]}" for col in range(n_rt_bins)],
        yticklabels=[f"{mz_edges[row]}–{mz_edges[row+1]}" for row in range(n_mz_bins)],
        annot=False, mask=np.isnan(heatmap)
    )
    ax.set_xlabel("RT bins")
    ax.set_ylabel("m/z bins")
    ax.set_title("Balanced accuracy per RT×m/z subcube")
    ax.invert_yaxis()

    # === Parallelised loop (row by row) ===
    for i in range(n_rt_bins):
        row_results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate_cell)(i, j) for j in range(n_mz_bins)
        )
        for j, acc in enumerate(row_results):
            heatmap[j, i] = acc

        # update only the data array of the heatmap
        mesh.collections[0].set_array(heatmap.ravel())
        plt.pause(0.1)

    plt.ioff()
    plt.show()
    return heatmap


# def run_region_accuracy_heatmap(
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
#     n_rt_bins=5,
#     n_mz_bins=5,
#     random_state=42,
# ):
#     """
#     RT × m/z discriminative map:
#     For each cell (rt_bin_i, mz_bin_j), trains a model using ONLY that subcube,
#     evaluates it with outer CV, and stores balanced accuracy in heatmap.
#     Heatmap is updated dynamically during the loop.
#
#     Returns
#     -------
#     heatmap : ndarray of shape (n_rt_bins, n_mz_bins)
#         Balanced accuracy for each (RT bin, m/z bin) cell.
#     """
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#
#     acc_annot = False if n_rt_bins *  n_mz_bins >= 400 else True
#
#     # === Step 0: Singleton filtering (to avoid CV crashes) ===
#     if cv_type in ("LOO", "LOOPC"):
#         data3d, labels, raw_sample_labels = filter_singletons(
#             data3d,
#             year_labels if class_by_year else labels,
#             raw_labels=raw_sample_labels,
#             class_by_year=class_by_year,
#         )
#         if class_by_year:
#             year_labels = labels  # already filtered
#
#     n_samples, n_time, n_channels = data3d.shape
#
#     # === Step 1: Define RT and m/z bin edges ===
#     rt_edges = np.linspace(0, n_time, n_rt_bins + 1, dtype=int)
#     mz_edges = np.linspace(0, n_channels, n_mz_bins + 1, dtype=int)
#
#     # Initialize empty heatmap
#     heatmap = np.full((n_mz_bins, n_rt_bins), np.nan)
#
#
#     # === Step 2: Setup live heatmap plot ===
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     # Create a dedicated colorbar axis once
#     cbar_ax = fig.add_axes([0.92, 0.3, 0.02, 0.4])
#
#     sns.heatmap(
#         heatmap, vmin=0, vmax=1, cmap="viridis", ax=ax, cbar=True, cbar_ax=cbar_ax,
#         mask=np.isnan(heatmap),
#         xticklabels=[f"{rt_edges[i]}–{rt_edges[i + 1]}" for i in range(n_rt_bins)],
#         yticklabels=[f"{mz_edges[j]}–{mz_edges[j + 1]}" for j in range(n_mz_bins)],
#         annot=False,
#     )
#     ax.set_xlabel("RT bins")
#     ax.set_ylabel("m/z channels (bins)")
#     ax.set_title("Balanced accuracy per RT×m/z subcube")
#     ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#     ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#     plt.pause(0.1)
#
#     # === Step 3: Outer CV splitter (same for all cells) ===
#     # Just build once using full X for consistency in stratification
#     full_X = compute_features(data3d, feature_type=feature_type)
#     if cv_type == "LOO":
#         outer_splits = list(LeaveOneOut().split(full_X, labels))
#     elif cv_type == "LOOPC":
#         outer_splits = loopc_splits(
#             year_labels if class_by_year else labels,
#             num_repeats=num_repeats,
#             random_state=random_state,
#             class_by_year=class_by_year,
#             raw_labels=raw_sample_labels if class_by_year else None,
#         )
#     elif cv_type == "stratified":
#         y_for_split = year_labels if class_by_year else labels
#         cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
#         outer_splits = list(cv.split(full_X, y_for_split))
#     else:
#         raise ValueError(f"Unsupported cv_type: {cv_type}")
#
#     n_classes = len(np.unique(year_labels if class_by_year else labels))
#
#     # === Step 4: Helper for training safely ===
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
#                 scaler_type='standard',
#                 region=region,
#                 random_seed=random_state,
#                 test_size=0.2,
#                 LOOPC=False,
#                 projection_source=False,
#                 X_test=Xte,
#                 y_test=yte
#             )
#             return res
#         except np.linalg.LinAlgError:
#             return {"balanced_accuracy": 1.0 / len(np.unique(ytr))}
#
#     # === Step 5: Iterate over all cells (RT × m/z) ===
#     for i in range(n_rt_bins):
#         for j in range(n_mz_bins):
#             rt_start, rt_end = rt_edges[i], rt_edges[i+1]
#             mz_start, mz_end = mz_edges[j], mz_edges[j+1]
#
#             subcube = data3d[:, rt_start:rt_end, mz_start:mz_end]
#             X = compute_features(subcube, feature_type=feature_type)
#
#             # Evaluate this cube using outer CV
#             fold_accs = []
#             for train_idx, test_idx in outer_splits:
#                 y_train = year_labels[train_idx] if class_by_year else labels[train_idx]
#                 y_test = year_labels[test_idx] if class_by_year else labels[test_idx]
#
#                 res = safe_train_eval(X[train_idx], y_train, X[test_idx], y_test, train_idx)
#                 fold_accs.append(res["balanced_accuracy"])
#
#             mean_acc = float(np.mean(fold_accs))
#             heatmap[j, i] = mean_acc
#
#             # Live update of the heatmap
#             ax.clear()
#             sns.heatmap(
#                 heatmap, vmin=0, vmax=1, cmap="viridis", ax=ax, cbar=False,
#                 mask=np.isnan(heatmap),  # keep uncomputed cells blank
#                 xticklabels=[f"{rt_edges[col]}–{rt_edges[col + 1]}" for col in range(n_rt_bins)],
#                 yticklabels=[f"{mz_edges[row]}–{mz_edges[row + 1]}" for row in range(n_mz_bins)],
#                 annot=acc_annot, fmt=".2f",  # <-- write accuracy inside each cell
#                 annot_kws={"size": 8, "color": "white", "ha": "center", "va": "center"},
#             )
#             ax.invert_yaxis()  # <--- flip vertical axis
#             ax.set_xlabel("RT bins")
#             ax.set_ylabel("m/z channel bins")
#             ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
#             ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
#             plt.pause(0.05)
#
#             logger.info(f"RT {i}/{n_rt_bins}, m/z {j}/{n_mz_bins} → {mean_acc:.3f}")
#
#     plt.ioff()
#     plt.show()
#
#     return heatmap





# -----------------------------
# Projection plotting (unchanged behavior)
# -----------------------------

def do_projection_plot(
    plot_projection,
    projection_source,
    projection_method,
    projection_dim,
    n_neighbors,
    random_state,
    color_by_country,
    invert_x,
    invert_y,
    rot_axes,
    sample_display_mode,
    color_by_winery,
    color_by_origin,
    exclude_us,
    density_plot,
    region,
    labels,
    raw_sample_labels,
    scores=None,
    all_labels=None,
    data=None,
):
    if not plot_projection:
        return

    show_year = True if sample_display_mode == "years" else False
    show_sample_names = True if sample_display_mode == "names" else False

    # Legend labels
    if region == "winery":
        legend_labels = {
            "D": "D = Clos Des Mouches. Drouhin (FR)",
            "R": "R = Les Petits Monts. Drouhin (FR)",
            "X": "X = Domaine Drouhin (US)",
            "E": "E = Vigne de l’Enfant Jésus. Bouchard (FR)",
            "Q": "Q = Les Cailles. Bouchard (FR)",
            "P": "P = Bressandes. Jadot (FR)",
            "Z": "Z = Les Boudots. Jadot (FR)",
            "C": "C = Domaine Schlumberger (FR)",
            "W": "W = Domaine Jean Sipp (FR)",
            "Y": "Y = Domaine Weinbach (FR)",
            "M": "M = Domaine Brunner (CH)",
            "N": "N = Vin des Croisés (CH)",
            "J": "J = Domaine Villard et Fils (CH)",
            "L": "L = Domaine de la République (CH)",
            "H": "H = Les Maladaires (CH)",
            "U": "U = Marimar Estate (US)",
        }
    elif region == "origin":
        legend_labels = {
            "A": "Alsace",
            "B": "Burgundy",
            "N": "Neuchâtel",
            "G": "Geneva",
            "V": "Valais",
            "C": "California",
            "O": "Oregon",
        }
    elif region == "country":
        legend_labels = {"F": "France", "S": "Switzerland", "U": "US"}
    elif region == "continent":
        legend_labels = {"E": "Europe", "N": "America"}
    elif region == "burgundy":
        legend_labels = {"N": "Côte de Nuits (north)", "S": "Côte de Beaune (south)"}
    else:
        legend_labels = utils.get_custom_order_for_pinot_noir_region(region)

    # data_for_umap and labels for projection
    if projection_source == "scores":
        data_for_umap = normalize(scores)
        projection_labels = all_labels
    elif projection_source in {"tic", "tis", "tic_tis"}:
        data_for_umap = utils.compute_features(data, feature_type=projection_source)
        data_for_umap = normalize(data_for_umap)
        projection_labels = labels
    else:
        raise ValueError(f"Unknown projection source: {projection_source}")

    pretty_source = {
        "scores": "Classification Scores",
        "tic": "TIC",
        "tis": "TIS",
        "tic_tis": "TIC + TIS",
    }.get(projection_source, projection_source)

    pretty_method = {"UMAP": "UMAP", "T-SNE": "t-SNE", "PCA": "PCA"}.get(
        projection_method, projection_method
    )

    pretty_region = {
        "winery": "Winery",
        "origin": "Origin",
        "country": "Country",
        "continent": "Continent",
        "burgundy": "N/S Burgundy",
    }.get(region, region)

    plot_title = (
        f"{pretty_method} of {pretty_source} ({pretty_region})" if region else f"{pretty_method} of {pretty_source}"
    )

    test_samples_names = None if not show_sample_names else None  # keep behavior: disable unless explicitly names

    reducer = DimensionalityReducer(data_for_umap)
    if projection_method == "UMAP":
        plot_pinot_noir(
            reducer.umap(components=2, n_neighbors=n_neighbors, random_state=random_state),
            plot_title,
            projection_labels,
            legend_labels,
            color_by_country,
            test_sample_names=test_samples_names,
            unique_samples_only=False,
            n_neighbors=n_neighbors,
            random_state=random_state,
            invert_x=invert_x,
            invert_y=invert_y,
            rot_axes=rot_axes,
            raw_sample_labels=raw_sample_labels,
            show_year=show_year,
            color_by_origin=color_by_origin,
            color_by_winery=color_by_winery,
            highlight_burgundy_ns=False,
            exclude_us=exclude_us,
            density_plot=density_plot,
            region=region,
        )
    elif projection_method == "PCA":
        plot_pinot_noir(
            reducer.pca(components=2),
            plot_title,
            projection_labels,
            legend_labels,
            color_by_country,
            test_sample_names=test_samples_names,
        )
    elif projection_method == "T-SNE":
        plot_pinot_noir(
            reducer.tsne(components=2, perplexity=5, random_state=random_state),
            plot_title,
            projection_labels,
            legend_labels,
            color_by_country,
            test_sample_names=test_samples_names,
        )
    else:
        raise ValueError(f"Unsupported projection method: {projection_method}")

    plt.show()


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Load config
    config = load_config()

    # Plot parameters
    plot_projection = config.get("plot_projection", False)
    projection_method = config.get("projection_method", "UMAP").upper()
    projection_source = config.get("projection_source", False) if plot_projection else False
    projection_dim = config.get("projection_dim", 2)
    n_neighbors = config.get("n_neighbors", 30)
    random_state = config.get("random_state", 42)
    color_by_country = config["color_by_country"]
    show_sample_names = config["show_sample_names"]
    invert_x = config["invert_x"]
    invert_y = config["invert_y"]
    rot_axes = config["rot_axes"]
    sample_display_mode = config["sample_display_mode"]
    color_by_winery = config.get("color_by_winery", False)
    color_by_origin = config.get("color_by_origin", False)
    exclude_us = config.get("exclude_us", False)
    density_plot = config.get("density_plot", False)

    # Run parameters
    sotf_ret_time_flag = config.get("sotf_ret_time")
    sotf_mz_flag = config.get("sotf_mz")
    if sotf_ret_time_flag and sotf_mz_flag:
        raise ValueError("Only one of 'sotf_ret_time' or 'sotf_mz' can be True.")
    sotf_remove_2d_flag = config.get("sotf_remove_2d")
    sotf_add_2d_flag = config.get("sotf_add_2d")
    reg_acc_map_flag = config.get("reg_acc_map")


    feature_type = config["feature_type"]
    classifier = config["classifier"]
    num_repeats = config["num_repeats"]
    normalize_flag = config["normalize"]
    n_decimation = config["n_decimation"]
    sync_state = config["sync_state"]
    class_by_year = config["class_by_year"]
    region = config["region"]
    rt_bins = config["n_rt_bins"]
    mz_bins = config["n_mz_bins"]

    # Default color-by behavior based on region (preserve logic)
    if not color_by_origin and not color_by_winery:
        if region == "origin":
            color_by_origin = True
        elif region == "winery":
            color_by_winery = True

    show_confusion_matrix = config["show_confusion_matrix"]
    cv_type = config["cv_type"]

    # Summary header
    wine_kind, cl, gcms, data_dict, dataset_origins = load_and_prepare_data(config)

    task = "classification"
    summary = {
        "Task": task,
        "Wine kind": wine_kind,
        "Datasets": ", ".join(config["selected_datasets"]),
        "Feature type": feature_type,
        "Classifier": classifier,
        "Repeats": num_repeats,
        "Normalize": normalize_flag,
        "Decimation": n_decimation,
        "Sync": sync_state,
        "Year Classification": class_by_year,
        "Region": region,
        "CV type": cv_type,
        "RT range": config["rt_range"],
        "Confusion matrix": show_confusion_matrix,
    }

    logger_raw("\n")
    logger.info("------------------------ RUN SCRIPT -------------------------")
    logger.info("Configuration Parameters")
    for k, v in summary.items():
        logger_raw(f"{k:>20s}: {v}")

    # Strategy
    strategy = get_strategy_by_wine_kind(
        wine_kind=wine_kind,
        region=region,
        get_custom_order_func=utils.get_custom_order_for_pinot_noir_region,
    )

    # Optional alignment
    if sync_state:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
        gcms = GCMSDataProcessor(data_dict)

    # Prepare arrays
    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    raw_sample_labels = labels.copy()

    # Restrict to Burgundy if requested
    if wine_kind == "pinot_noir" and region == "burgundy":
        burgundy_prefixes = ("D", "P", "R", "Q", "Z", "E")
        mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
        data = data[mask]
        labels = labels[mask]
        raw_sample_labels = raw_sample_labels[mask]

    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)

    # Prepare 3D for m/z mode (if needed)
    data3d = None
    # if sotf_mz_flag:
    data3d = dict_to_array3d(data_dict)  # (N, T, C)
    if wine_kind == "pinot_noir" and region == "burgundy":
        data3d = data3d[mask]

    # -----------------------------
    # Branch by mode
    # -----------------------------
    results = None
    if sotf_ret_time_flag:
        accuracies, percent_remaining = run_sotf_ret_time(
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
            n_bins=50,
            min_bins=1,
        )
    elif sotf_mz_flag:
        accuracies, percent_remaining = run_sotf_mz(
            data3d=data,
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
            min_channels=25,
            group_size = 25,
        )
    elif sotf_remove_2d_flag:
        results = run_sotf_remove_2d(
            data3d=data3d,  # shape (samples, RT, m/z)
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
            min_cubes=1,  # minimum cubes to leave
            random_state=42,
        )
    elif sotf_add_2d_flag:
        # results = run_sotf_add_2d(
        results = run_sotf_add_2d_fast(
            data3d=data3d,  # shape (samples, RT, m/z)
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
    elif reg_acc_map_flag:
        heatmap = run_region_accuracy_heatmap(
            data3d=data3d,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=classifier,
            wine_kind=wine_kind,
            class_by_year=class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cv_type,  # e.g. "LOO"
            num_repeats=num_repeats,
            normalize_flag=normalize_flag,
            region=region,
            feature_type=feature_type,
            projection_source=projection_source,
            show_confusion_matrix=show_confusion_matrix,
            n_rt_bins=rt_bins,  # number of RT slices
            n_mz_bins=mz_bins,  # number of m/z slices
        )
    else:
        results = run_normal_classification(
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
        )

    # -----------------------------
    # Optional projection plot (preserve behavior)
    # -----------------------------
    scores = None
    all_labels = None
    test_samples_names = None
    if results and cv_type == "LOO":
        mean_acc, std_acc, scores, all_labels, test_samples_names = results
    elif results and cv_type in ["LOOPC", "stratified"]:
        mean_acc, std_acc, *_ = results

    if plot_projection:
        do_projection_plot(
            plot_projection=plot_projection,
            projection_source=projection_source,
            projection_method=projection_method,
            projection_dim=projection_dim,
            n_neighbors=n_neighbors,
            random_state=random_state,
            color_by_country=color_by_country,
            invert_x=invert_x,
            invert_y=invert_y,
            rot_axes=rot_axes,
            sample_display_mode=sample_display_mode,
            color_by_winery=color_by_winery,
            color_by_origin=color_by_origin,
            exclude_us=exclude_us,
            density_plot=density_plot,
            region=region,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            scores=scores,
            all_labels=all_labels,
            data=data,
        )





# import numpy as np
# import os
# import yaml
# import matplotlib.pyplot as plt
# from gcmswine.classification import Classifier
# from gcmswine import utils
# from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
# from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
# from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified
# from gcmswine.logger_setup import logger, logger_raw
# from sklearn.preprocessing import normalize
# from gcmswine.dimensionality_reduction import DimensionalityReducer
# from scripts.pinot_noir.plotting_pinot_noir import plot_pinot_noir
# from distinctipy import distinctipy
# from sklearn.model_selection import StratifiedShuffleSplit
#
#
#
# # ------------------------------
# # Helper: projection plotting
# # ------------------------------
# def do_plot_projection(
#     projection_source,
#     projection_method,
#     projection_dim,
#     n_neighbors,
#     random_state,
#     invert_x,
#     invert_y,
#     rot_axes,
#     show_sample_names,
#     color_by_country,
#     color_by_origin,
#     color_by_winery,
#     exclude_us,
#     density_plot,
#     region,
#     data,
#     labels,
#     raw_sample_labels,
#     scores=None,
#     all_labels=None,
#     test_samples_names=None,
#     show_year=False,
# ):
#     """Wrapper for dimensionality reduction + Pinot Noir plotting (keeps your original behavior)."""
#     # Legend labels (exact logic preserved)
#     if region == "winery":
#         legend_labels = {
#             "D": "D = Clos Des Mouches. Drouhin (FR)",
#             "R": "R = Les Petits Monts. Drouhin (FR)",
#             "X": "X = Domaine Drouhin (US)",
#             "E": "E = Vigne de l’Enfant Jésus. Bouchard (FR)",
#             "Q": "Q = Les Cailles. Bouchard (FR)",
#             "P": "P = Bressandes. Jadot (FR)",
#             "Z": "Z = Les Boudots. Jadot (FR)",
#             "C": "C = Domaine Schlumberger (FR)",
#             "W": "W = Domaine Jean Sipp (FR)",
#             "Y": "Y = Domaine Weinbach (FR)",
#             "M": "M = Domaine Brunner (CH)",
#             "N": "N = Vin des Croisés (CH)",
#             "J": "J = Domaine Villard et Fils (CH)",
#             "L": "L = Domaine de la République (CH)",
#             "H": "H = Les Maladaires (CH)",
#             "U": "U = Marimar Estate (US)",
#         }
#     elif region == "origin":
#         legend_labels = {
#             "A": "Alsace",
#             "B": "Burgundy",
#             "N": "Neuchâtel",
#             "G": "Geneva",
#             "V": "Valais",
#             "C": "California",
#             "O": "Oregon",
#         }
#     elif region == "country":
#         legend_labels = {"F": "France", "S": "Switzerland", "U": "US"}
#     elif region == "continent":
#         legend_labels = {"E": "Europe", "N": "America"}
#     elif region == "burgundy":
#         legend_labels = {"N": "Côte de Nuits (north)", "S": "Côte de Beaune (south)"}
#     else:
#         legend_labels = utils.get_custom_order_for_pinot_noir_region(region)
#
#     # Data for reduction (identical branching)
#     if projection_source == "scores":
#         if scores is None or all_labels is None:
#             raise ValueError("Scores projection requires 'scores' and 'all_labels' from evaluation (usually LOO).")
#         data_for_umap = normalize(scores)
#         projection_labels = all_labels
#     elif projection_source in {"tic", "tis", "tic_tis"}:
#         data_for_umap = utils.compute_features(data, feature_type=projection_source)
#         data_for_umap = normalize(data_for_umap)
#         projection_labels = labels
#     else:
#         raise ValueError(f"Unknown projection source: {projection_source}")
#
#     # Pretty titles (unchanged)
#     pretty_source = {
#         "scores": "Classification Scores",
#         "tic": "TIC",
#         "tis": "TIS",
#         "tic_tis": "TIC + TIS",
#     }.get(projection_source, projection_source)
#
#     pretty_method = {"UMAP": "UMAP", "T-SNE": "t-SNE", "PCA": "PCA"}.get(projection_method, projection_method)
#
#     pretty_region = {
#         "winery": "Winery",
#         "origin": "Origin",
#         "country": "Country",
#         "continent": "Continent",
#         "burgundy": "N/S Burgundy",
#     }.get(region, region)
#
#     if region:
#         plot_title = f"{pretty_method} of {pretty_source} ({pretty_region})"
#     else:
#         plot_title = f"{pretty_method} of {pretty_source}"
#
#     if not show_sample_names:
#         test_samples_names = None
#
#     # Reduce + plot (unchanged options to plot_pinot_noir)
#     reducer = DimensionalityReducer(data_for_umap)
#     if projection_method == "UMAP":
#         emb = reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state)
#         plot_pinot_noir(
#             emb,
#             plot_title,
#             projection_labels,
#             legend_labels,
#             color_by_country,
#             test_sample_names=test_samples_names,
#             unique_samples_only=False,
#             n_neighbors=n_neighbors,
#             random_state=random_state,
#             invert_x=invert_x,
#             invert_y=invert_y,
#             rot_axes=rot_axes,
#             raw_sample_labels=raw_sample_labels,
#             show_year=show_year,
#             color_by_origin=color_by_origin,
#             color_by_winery=color_by_winery,
#             highlight_burgundy_ns=False,
#             exclude_us=exclude_us,
#             density_plot=density_plot,
#             region=region,
#         )
#     elif projection_method == "PCA":
#         emb = reducer.pca(components=projection_dim)
#         plot_pinot_noir(
#             emb, plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names
#         )
#     elif projection_method == "T-SNE":
#         emb = reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state)
#         plot_pinot_noir(
#             emb, plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names
#         )
#     else:
#         raise ValueError(f"Unsupported projection method: {projection_method}")
#
#     plt.show()
#
#
# # ------------------------------
# # Helper: small utilities
# # ------------------------------
# def split_into_bins(data, n_bins):
#     """Split TIC (N x T) into uniform bins (segments). Return list of (start_idx, end_idx)."""
#     total_points = data.shape[1]
#     bin_edges = np.linspace(0, total_points, n_bins + 1, dtype=int)
#     return [(bin_edges[i], bin_edges[i + 1]) for i in range(n_bins)]
#
#
# def remove_bins(data, bins_to_remove, bin_ranges):
#     """Zero out specified bins in TIC by index."""
#     data_copy = data.copy()
#     for b in bins_to_remove:
#         start, end = bin_ranges[b]
#         data_copy[:, start:end] = 0
#     return data_copy
#
#
# def dict_to_array3d(d):
#     """Stack each sample's 2D chromatogram (time × channels) into (N, T, C)."""
#     arrs = []
#     for v in d.values():
#         v = np.asarray(v)
#         if v.ndim == 1:
#             raise ValueError("sotf_mz requires per-channel data (time × channels) per sample.")
#         arrs.append(v)
#     return np.stack(arrs, axis=0)  # (N, T, C)
#
#
# def train_and_eval(
#     cls,
#     cv_type,
#     num_repeats,
#     normalize_flag,
#     region,
#     feature_type,
#     classifier,
#     projection_source,
#     show_confusion_matrix,
# ):
#     """
#     Run training/eval, returning
#     (mean_acc, std_acc, scores, all_labels, test_samples_names) with Nones if not available.
#     """
#     if cv_type in ["LOOPC", "stratified"]:
#         loopc = (cv_type == "LOOPC")
#         mean_acc, std_acc, *_ = cls.train_and_evaluate_all_channels(
#             num_repeats=num_repeats,
#             random_seed=42,
#             test_size=0.2,
#             normalize=normalize_flag,
#             scaler_type="standard",
#             use_pca=False,
#             vthresh=0.97,
#             region=region,
#             print_results=False,
#             n_jobs=20,
#             feature_type=feature_type,
#             classifier_type=classifier,
#             LOOPC=loopc,
#             projection_source=projection_source,
#             show_confusion_matrix=show_confusion_matrix,
#         )
#         return mean_acc, std_acc, None, None, None
#     elif cv_type == "LOO":
#         mean_acc, std_acc, scores, all_labels, test_samples_names = cls.train_and_evaluate_leave_one_out_all_samples(
#             normalize=normalize_flag,
#             scaler_type="standard",
#             region=region,
#             feature_type=feature_type,
#             classifier_type=classifier,
#             projection_source=projection_source,
#             show_confusion_matrix=show_confusion_matrix,
#         )
#         return mean_acc, std_acc, scores, all_labels, test_samples_names
#     else:
#         raise ValueError(f"Invalid cross-validation type: '{cv_type}'.")
#
#
# # ------------------------------
# # Task runners
# # ------------------------------
# def run_classification(
#     data,
#     labels,
#     raw_sample_labels,
#     dataset_origins,
#     year_labels,
#     strategy,
#     # training config
#     classifier,
#     class_by_year,
#     cv_type,
#     num_repeats,
#     normalize_flag,
#     region,
#     feature_type,
#     projection_source,
#     show_confusion_matrix,
#     wine_kind,
# ):
#     """Single evaluation (original non-survival case)."""
#     cls = Classifier(
#         data,
#         labels,
#         classifier_type=classifier,
#         wine_kind=wine_kind,
#         class_by_year=class_by_year,
#         year_labels=np.array(year_labels),
#         strategy=strategy,
#         sample_labels=np.array(raw_sample_labels),
#         dataset_origins=dataset_origins,
#     )
#     mean_acc, std_acc, scores, all_labels, test_samples_names = train_and_eval(
#         cls,
#         cv_type=cv_type,
#         num_repeats=num_repeats,
#         normalize_flag=normalize_flag,
#         region=region,
#         feature_type=feature_type,
#         classifier=classifier,
#         projection_source=projection_source,
#         show_confusion_matrix=show_confusion_matrix,
#     )
#     logger.info(f"Final Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
#     return {
#         "mean_acc": mean_acc,
#         "std_acc": std_acc,
#         "scores": scores,
#         "all_labels": all_labels,
#         "test_samples_names": test_samples_names,
#     }
#
#
# from sklearn.model_selection import LeaveOneOut, StratifiedKFold
#
# def run_sotf_ret_time(
#     data, labels, raw_sample_labels, dataset_origins, year_labels, strategy,
#     classifier, class_by_year, cv_type, num_repeats, normalize_flag,
#     region, feature_type, projection_source, show_confusion_matrix,
#     wine_kind, n_bins=50, min_bins=1
# ):
#     """Leakage-free greedy removal across retention time bins."""
#     labels = np.array(labels)
#     raw_sample_labels = np.array(raw_sample_labels)
#     year_labels = np.array(year_labels)
#
#     bin_ranges = split_into_bins(data, n_bins)
#     baseline_nonzero = np.count_nonzero(data)
#
#     # Outer CV splitter
#     if cv_type == "LOO":
#         cv_outer = LeaveOneOut()
#     elif cv_type == "LOOPC":
#         cv_outer = StratifiedKFold(n_splits=len(np.unique(labels)), shuffle=True, random_state=42)
#     else:
#         raise ValueError(f"Unsupported CV type: {cv_type}")
#
#     outer_accuracies = []
#
#     for fold, (sel_train_idx, sel_test_idx) in enumerate(cv_outer.split(data, labels)):
#         logger.info(f"\n=== Outer fold {fold+1} ===")
#
#         data_train, data_test = data[sel_train_idx], data[sel_test_idx]
#         labels_train, labels_test = labels[sel_train_idx], labels[sel_test_idx]
#
#         active_bins = list(range(n_bins))
#         n_iterations = n_bins - min_bins + 1
#
#         # Track fold results
#         accuracies, percent_remaining = [], []
#
#         for step in range(n_iterations):
#             masked_train = remove_bins(
#                 data_train,
#                 bins_to_remove=[b for b in range(n_bins) if b not in active_bins],
#                 bin_ranges=bin_ranges,
#             )
#             pct_data = (np.count_nonzero(masked_train) / baseline_nonzero) * 100.0
#             percent_remaining.append(pct_data)
#
#             cls = Classifier(
#                 masked_train, labels_train,
#                 classifier_type=classifier, wine_kind=wine_kind,
#                 class_by_year=class_by_year, year_labels=np.array(year_labels)[sel_train_idx],
#                 strategy=strategy, sample_labels=np.array(raw_sample_labels)[sel_train_idx],
#                 dataset_origins=dataset_origins,
#             )
#
#             mean_acc, std_acc, *_ = train_and_eval(
#                 cls, cv_type="stratified",  # inner loop only on train
#                 num_repeats=num_repeats, normalize_flag=normalize_flag,
#                 region=region, feature_type=feature_type,
#                 classifier=classifier, projection_source=projection_source,
#                 show_confusion_matrix=False,
#             )
#             accuracies.append(mean_acc)
#
#             # Greedy removal
#             if len(active_bins) > min_bins:
#                 candidate_accuracies = []
#                 for b in active_bins:
#                     temp_bins = [x for x in active_bins if x != b]
#                     temp_masked = remove_bins(
#                         data_train,
#                         bins_to_remove=[x for x in range(n_bins) if x not in temp_bins],
#                         bin_ranges=bin_ranges,
#                     )
#                     temp_cls = Classifier(
#                         temp_masked, labels_train,
#                         classifier_type=classifier, wine_kind=wine_kind,
#                         class_by_year=class_by_year, year_labels=np.array(year_labels)[sel_train_idx],
#                         strategy=strategy, sample_labels=np.array(raw_sample_labels)[sel_train_idx],
#                         dataset_origins=dataset_origins,
#                     )
#                     temp_acc, _, *_ = temp_cls.train_and_evaluate_all_channels(
#                         num_repeats=5, test_size=0.2, random_seed=42,
#                         normalize=normalize_flag, scaler_type="standard",
#                         feature_type=feature_type, classifier_type=classifier,
#                         projection_source=projection_source,
#                         LOOPC=False, show_confusion_matrix=False,
#                     )
#                     candidate_accuracies.append((b, temp_acc))
#
#                 best_bin, _ = max(candidate_accuracies, key=lambda x: x[1])
#                 active_bins.remove(best_bin)
#
#         # === Final outer test ===
#         masked_train = remove_bins(data_train, bins_to_remove=[b for b in range(n_bins) if b not in active_bins], bin_ranges=bin_ranges)
#         masked_test  = remove_bins(data_test,  bins_to_remove=[b for b in range(n_bins) if b not in active_bins], bin_ranges=bin_ranges)
#
#         final_cls = Classifier(
#             masked_train, labels_train,
#             classifier_type=classifier, wine_kind=wine_kind,
#             class_by_year=class_by_year, year_labels=np.array(year_labels)[sel_train_idx],
#             strategy=strategy, sample_labels=np.array(raw_sample_labels)[sel_train_idx],
#             dataset_origins=dataset_origins,
#         )
#         final_cls.fit()
#         preds = final_cls.predict(masked_test)
#         outer_acc = np.mean(preds == labels_test)
#         outer_accuracies.append(outer_acc)
#         logger.info(f"[Fold {fold+1}] Outer holdout accuracy: {outer_acc:.3f}")
#
#     return {"outer_mean_acc": np.mean(outer_accuracies), "outer_std_acc": np.std(outer_accuracies)}
#
#
# def run_sotf_mz(
#     data3d,  # (N, T, C)
#     labels,
#     raw_sample_labels,
#     dataset_origins,
#     year_labels,
#     strategy,
#     # training config
#     classifier,
#     class_by_year,
#     cv_type,
#     num_repeats,
#     normalize_flag,
#     region,
#     feature_type,
#     projection_source,
#     show_confusion_matrix,
#     wine_kind,
#     # survival config
#     min_channels=1,
#     selector_test_size=0.2,   # <- new
#     selector_random_state=42, # <- new
# ):
#     """
#     Greedy removal across m/z channels.
#     LEAKAGE-FREE: channel selection scored ONLY on a stratified selector-train
#     subset. The held-out selector set is never used during selection.
#     """
#
#     labels = np.asarray(labels)
#     year_labels = np.asarray(year_labels)
#     raw_sample_labels = np.asarray(raw_sample_labels)
#
#     n_channels = data3d.shape[2]
#     active_channels = list(range(n_channels))
#     n_iterations = n_channels - min_channels + 1
#
#     # --- split once for selection (leakage-free) ---
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=selector_test_size,
#                                  random_state=selector_random_state)
#     (sel_train_idx, sel_holdout_idx) = next(sss.split(np.zeros(len(labels)), labels))
#
#     labels_sel = labels[sel_train_idx]
#     years_sel  = np.array(year_labels)[sel_train_idx]
#     raw_sel    = np.array(raw_sample_labels)[sel_train_idx]
#     data3d_sel = data3d[sel_train_idx]
#
#     # Progressive plot (selector-CV accuracy, leakage-free)
#     plt.ion()
#     fig, ax = plt.subplots(figsize=(8, 5))
#     line, = ax.plot([], [], marker="o")
#     ax.set_xlabel("Percentage of m/z channels remaining (%)")
#     ax.set_ylabel("Accuracy (selector CV)")
#     ax.grid(True)
#     ax.set_xlim(100, 0)
#
#     accuracies = []
#     percent_remaining = []
#     last_scores = last_all_labels = last_test_names = None
#
#     for step in range(n_iterations):
#         logger.info(f"=== Iteration {step + 1}/{n_iterations} | Active channels: {len(active_channels)} ===")
#
#         # Mask on selection subset only (for scoring)
#         masked_sel = data3d_sel.copy()
#         keep = np.zeros(n_channels, dtype=bool)
#         keep[active_channels] = True
#         masked_sel[:, :, ~keep] = 0
#
#         # Evaluate current set on selector-train subset only
#         cls_sel = Classifier(
#             masked_sel,
#             labels_sel,
#             classifier_type=classifier,
#             wine_kind=wine_kind,
#             class_by_year=class_by_year,
#             year_labels=years_sel,
#             strategy=strategy,
#             sample_labels=raw_sel,
#             dataset_origins=dataset_origins,
#         )
#         mean_acc, std_acc, scores, all_labels, test_samples_names = train_and_eval(
#             cls_sel,
#             cv_type=cv_type,
#             num_repeats=num_repeats,
#             normalize_flag=normalize_flag,
#             region=region,
#             feature_type=feature_type,
#             classifier=classifier,
#             projection_source=projection_source,
#             show_confusion_matrix=False,
#         )
#         accuracies.append(mean_acc)
#         logger.info(f"[SOTF-m/z | step {step+1}] selector-CV accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
#
#         last_scores, last_all_labels, last_test_names = scores, all_labels, test_samples_names
#
#         # % remaining by channels
#         pct_data = 100.0 * len(active_channels) / n_channels
#         percent_remaining.append(pct_data)
#
#         # live plot
#         line.set_data(percent_remaining, accuracies)
#         ax.set_xlim(100, min(percent_remaining) - 5)
#         ax.set_ylim(0, 1)
#         plt.draw()
#         plt.pause(0.2)
#
#         # Greedy channel removal (scored on selector-train subset only)
#         if len(active_channels) > min_channels:
#             candidate_accuracies = []
#             for ch in active_channels:
#                 temp_channels = [x for x in active_channels if x != ch]
#                 temp_sel = data3d_sel.copy()
#                 keep_tmp = np.zeros(n_channels, dtype=bool)
#                 keep_tmp[temp_channels] = True
#                 temp_sel[:, :, ~keep_tmp] = 0
#
#                 temp_cls = Classifier(
#                     temp_sel,
#                     labels_sel,
#                     classifier_type=classifier,
#                     wine_kind=wine_kind,
#                     class_by_year=class_by_year,
#                     year_labels=years_sel,
#                     strategy=strategy,
#                     sample_labels=raw_sel,
#                     dataset_origins=dataset_origins,
#                 )
#                 temp_acc, _, *_ = temp_cls.train_and_evaluate_all_channels(
#                     num_repeats=10,
#                     random_seed=42,
#                     test_size=0.2,
#                     normalize=normalize_flag,
#                     scaler_type="standard",
#                     use_pca=False,
#                     vthresh=0.97,
#                     region=region,
#                     print_results=False,
#                     n_jobs=10,
#                     feature_type=feature_type,
#                     classifier_type=classifier,
#                     LOOPC=(cv_type == "LOOPC"),
#                     projection_source=projection_source,
#                     show_confusion_matrix=False,
#                 )
#                 candidate_accuracies.append((ch, temp_acc))
#
#             best_ch, best_candidate_acc = max(candidate_accuracies, key=lambda x: x[1])
#             logger.info(f"[SOTF-m/z] Removing channel {best_ch} (selector-CV best={best_candidate_acc:.3f})")
#             active_channels.remove(best_ch)
#
#     plt.ioff()
#     plt.show()
#
#     return {
#         "accuracies": accuracies,  # selector-CV
#         "percent_remaining": percent_remaining,
#         "scores": last_scores,
#         "all_labels": last_all_labels,
#         "test_samples_names": last_test_names,
#         "final_active_channels": active_channels,
#     }
#
#
#
#
# # =====================================================================
# #                                MAIN
# # =====================================================================
# if __name__ == "__main__":
#
#     # # Use this function to convert the printed confusion matrix to a latex confusion matrix
#     # # Copy the matrix into data_str using """ """ and create the list of headers, then call the function
#     # headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
#     #             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
#     #             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
#     # headers = ["Beaune", "Alsace", "Neuchatel", "Genève", "Valais", "Californie", "Oregon"]
#     # headers = ["France", "Switzerland", "US"]
#     # string_to_latex_confusion_matrix_modified(data_str, headers)
#
#     # Load dataset paths from config.yaml
#     config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
#     config_path = os.path.abspath(config_path)
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#
#     # Parameters from config file
#     dataset_directories = config["datasets"]
#     selected_datasets = config["selected_datasets"]
#
#     # Get the paths corresponding to the selected datasets
#     selected_paths = [config["datasets"][name] for name in selected_datasets]
#
#     # Check if all selected dataset contains "pinot"
#     if not all("pinot_noir" in path.lower() for path in selected_paths):
#         raise ValueError("Please use this script for Pinot Noir datasets.")
#
#     # Infer wine_kind from selected dataset paths
#     wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])
#
#     # Plot parameters
#     plot_projection = config.get("plot_projection", False)
#     projection_method = config.get("projection_method", "UMAP").upper()
#     projection_source = config.get("projection_source", False) if plot_projection else False
#     projection_dim = config.get("projection_dim", 2)
#     n_neighbors = config.get("n_neighbors", 30)
#     random_state = config.get("random_state", 42)
#     color_by_country = config["color_by_country"]
#     show_sample_names = config["show_sample_names"]
#     invert_x = config["invert_x"]
#     invert_y = config["invert_y"]
#     rot_axes = config["rot_axes"]
#     sample_display_mode = config["sample_display_mode"]
#     show_year = True if sample_display_mode == "years" else False
#     show_sample_names = True if sample_display_mode == "names" else False
#     color_by_winery = config.get("color_by_winery", False)
#     color_by_origin = config.get("color_by_origin", False)
#     exclude_us = config.get("exclude_us", False)
#     density_plot = config.get("density_plot", False)
#
#     # Run Parameters
#     sotf_ret_time = config.get("sotf_ret_time")
#     sotf_mz = config.get("sotf_mz")
#     if sotf_ret_time and sotf_mz:
#         raise ValueError("Only one of 'sotf_ret_time' or 'sotf_mz' can be True.")
#     feature_type = config["feature_type"]
#     classifier = config["classifier"]
#     num_repeats = config["num_repeats"]
#     normalize_flag = config["normalize"]
#     n_decimation = config["n_decimation"]
#     sync_state = config["sync_state"]
#     class_by_year = config["class_by_year"]
#     region = config["region"]
#
#     # Enforce exclusivity logic (preserved)
#     if not color_by_origin and not color_by_winery:
#         if region == "origin":
#             color_by_origin = True
#         elif region == "winery":
#             color_by_winery = True
#
#     show_confusion_matrix = config["show_confusion_matrix"]
#     retention_time_range = config["rt_range"]
#     cv_type = config["cv_type"]
#     task = "classification" if (not sotf_ret_time and not sotf_mz) else ("sotf_ret_time" if sotf_ret_time else "sotf_mz")
#     split_burgundy_ns = True
#     burg_by_year = True
#
#     summary = {
#         "Task": task,
#         "Wine kind": wine_kind,
#         "Datasets": ", ".join(selected_datasets),
#         "Feature type": config["feature_type"],
#         "Classifier": config["classifier"],
#         "Repeats": config["num_repeats"],
#         "Normalize": config["normalize"],
#         "Decimation": config["n_decimation"],
#         "Sync": config["sync_state"],
#         "Year Classification": config["class_by_year"],
#         "Region": config["region"],
#         "CV type": config["cv_type"],
#         "RT range": config["rt_range"],
#         "Confusion matrix": config["show_confusion_matrix"],
#     }
#
#     logger_raw("\n")
#     logger.info("------------------------ RUN SCRIPT -------------------------")
#     logger.info("Configuration Parameters")
#     for k, v in summary.items():
#         logger_raw(f"{k:>20s}: {v}")
#
#     # Strategy
#     strategy = get_strategy_by_wine_kind(
#         wine_kind=wine_kind,
#         region=region,
#         get_custom_order_func=utils.get_custom_order_for_pinot_noir_region,
#     )
#
#     # Optional alignment
#     cl = ChromatogramAnalysis(ndec=n_decimation)
#
#     # Load dataset, removing zero-variance channels
#     selected_paths_map = {name: dataset_directories[name] for name in selected_datasets}
#     data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
#     chrom_length = len(list(data_dict.values())[0])
#
#     if retention_time_range:
#         min_rt = retention_time_range["min"] // n_decimation
#         raw_max_rt = retention_time_range["max"] // n_decimation
#         max_rt = min(raw_max_rt, chrom_length)
#         logger.info(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
#         data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}
#
#     data_dict, _ = utils.remove_zero_variance_channels(data_dict)
#
#     gcms = GCMSDataProcessor(data_dict)
#     if sync_state:
#         tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
#         gcms = GCMSDataProcessor(data_dict)
#
#     # Prepare 3D array for m/z survival mode if requested
#     data3d = None
#     if sotf_mz:
#         data3d = dict_to_array3d(data_dict)  # (N, T, C)
#
#     # Extract data matrix and labels
#     data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
#     raw_sample_labels = labels.copy()
#
#     # Extract only Burgundy if region=="burgundy" (preserved)
#     if wine_kind == "pinot_noir" and region == "burgundy":
#         burgundy_prefixes = ("D", "P", "R", "Q", "Z", "E")
#         mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
#         data = data[mask]
#         labels = labels[mask]
#         raw_sample_labels = raw_sample_labels[mask]
#         if sotf_mz:
#             data3d = data3d[mask]
#
#     labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)
#
#     # ----------------- DISPATCH BY TASK -----------------
#     ctx = {"scores": None, "all_labels": None, "test_samples_names": None}
#
#     if not sotf_ret_time and not sotf_mz:
#         # Classification
#         ctx = run_classification(
#             data=data,
#             labels=labels,
#             raw_sample_labels=raw_sample_labels,
#             dataset_origins=dataset_origins,
#             year_labels=year_labels,
#             strategy=strategy,
#             classifier=classifier,
#             class_by_year=class_by_year,
#             cv_type=cv_type,
#             num_repeats=num_repeats,
#             normalize_flag=normalize_flag,
#             region=region,
#             feature_type=feature_type,
#             projection_source=projection_source,
#             show_confusion_matrix=show_confusion_matrix,
#             wine_kind=wine_kind,
#         )
#
#     elif sotf_ret_time:
#         # SOTF over retention time
#         ctx = run_sotf_ret_time(
#             data=data,
#             labels=labels,
#             raw_sample_labels=raw_sample_labels,
#             dataset_origins=dataset_origins,
#             year_labels=year_labels,
#             strategy=strategy,
#             classifier=classifier,
#             class_by_year=class_by_year,
#             cv_type=cv_type,
#             num_repeats=num_repeats,
#             normalize_flag=normalize_flag,
#             region=region,
#             feature_type=feature_type,
#             projection_source=projection_source,
#             show_confusion_matrix=show_confusion_matrix,
#             wine_kind=wine_kind,
#             n_bins=50,
#             min_bins=1,
#         )
#
#     elif sotf_mz:
#         # SOTF over m/z channels
#         ctx = run_sotf_mz(
#             data3d=data3d,
#             labels=labels,
#             raw_sample_labels=raw_sample_labels,
#             dataset_origins=dataset_origins,
#             year_labels=year_labels,
#             strategy=strategy,
#             classifier=classifier,
#             class_by_year=class_by_year,
#             cv_type=cv_type,
#             num_repeats=num_repeats,
#             normalize_flag=normalize_flag,
#             region=region,
#             feature_type=feature_type,
#             projection_source=projection_source,
#             show_confusion_matrix=show_confusion_matrix,
#             wine_kind=wine_kind,
#             min_channels=1,
#         )
#
#     # ----------------- Optional projection plot -----------------
#     if plot_projection:
#         do_plot_projection(
#             projection_source=projection_source,
#             projection_method=projection_method,
#             projection_dim=projection_dim,
#             n_neighbors=n_neighbors,
#             random_state=random_state,
#             invert_x=invert_x,
#             invert_y=invert_y,
#             rot_axes=rot_axes,
#             show_sample_names=show_sample_names,
#             color_by_country=color_by_country,
#             color_by_origin=color_by_origin,
#             color_by_winery=color_by_winery,
#             exclude_us=exclude_us,
#             density_plot=density_plot,
#             region=region,
#             data=data,
#             labels=labels,
#             raw_sample_labels=raw_sample_labels,
#             scores=ctx.get("scores"),
#             all_labels=ctx.get("all_labels"),
#             test_samples_names=ctx.get("test_samples_names"),
#             show_year=show_year,
#         )



# """
# To train and test classification of Pinot Noir wines, we use the script **train_test_pinot_noir.py**.
# The goal is to classify wine samples based on their GC-MS chemical fingerprint, using geographic labels
# at different levels of granularity (e.g., winery, region, country, north-south of Burgundy, or continent).
#
# The script implements a complete machine learning pipeline including data loading, preprocessing,
# region-based label extraction, feature computation, and repeated classifier evaluation.
#
# Configuration Parameters
# ------------------------
#
# The script reads configuration parameters from a file (`config.yaml`) located at the root of the repository.
# Below is a description of the key parameters:
#
# - **datasets**: Dictionary mapping dataset names to local paths. Each path must contain `.D` folders for each chromatogram.
#
# - **selected_datasets**: The list of datasets to use for the analysis. Must be compatible in terms of m/z channels.
#
# - **feature_type**: Defines how chromatograms are converted into features for classification:
#
#   - ``tic``: Use the Total Ion Chromatogram only.
#   - ``tis``: Use individual Total Ion Spectrum channels.
#   - ``tic_tis``: Concatenate TIC and TIS.
#   - ``concatenated``: Flatten raw chromatograms across all channels.
#
# - **classifier**: Classification model to apply. Available options:
#
#   - ``DTC``: Decision Tree Classifier
#   - ``GNB``: Gaussian Naive Bayes
#   - ``KNN``: K-Nearest Neighbors
#   - ``LDA``: Linear Discriminant Analysis
#   - ``LR``: Logistic Regression
#   - ``PAC``: Passive-Aggressive Classifier
#   - ``PER``: Perceptron
#   - ``RFC``: Random Forest Classifier
#   - ``RGC``: Ridge Classifier
#   - ``SGD``: Stochastic Gradient Descent
#   - ``SVM``: Support Vector Machine
#
# - **num_splits**: Number of repeated train/test splits to run.
#
# - **normalize**: Whether to apply standard scaling before classification. Normalization is fit on training data only.
#
# - **n_decimation**: Downsampling factor along the retention time axis to reduce dimensionality.
#
# - **sync_state**: Whether to align chromatograms using retention time synchronization (useful for Pinot Noir samples with retention drift).
#
# - **region**: Defines the classification target. Available options:
#
#   - ``winery``: Classify by individual wine producer
#   - ``origin``: Group samples by geographic region (e.g., Beaune, Alsace)
#   - ``country``: Group by country (e.g., France, Switzerland, USA)
#   - ``continent``: Group by continent
#   - ``north_south_burgundy``: Binary classification of northern vs southern Burgundy subregions
#
# - **wine_kind**: Internally inferred from dataset paths. Should not be set manually.
#
# Script Overview
# ---------------
#
# This script performs classification of **Pinot Noir wine samples** using GC-MS data and a configurable
# classification pipeline. It allows for flexible region-based classification using a strategy abstraction.
#
# The main workflow is:
#
# 1. **Configuration Loading**:
#
#    - Loads classifier, region, feature type, and dataset settings from `config.yaml`.
#    - Confirms that all dataset paths are compatible (must contain `'pinot'`).
#
# 2. **Data Loading and Preprocessing**:
#
#    - Chromatograms are loaded and decimated.
#    - Channels with zero variance are removed.
#    - If `sync_state=True`, samples are aligned by retention time.
#
# 3. **Label Processing**:
#
#    - Region-based labels are extracted using `process_labels_by_wine_kind()` and the `WineKindStrategy` abstraction.
#    - Granularity is determined by the `region` parameter (e.g., `"winery"` or `"country"`).
#
# 4. **Classification**:
#
#    - Initializes a `Classifier` instance with the chosen feature representation and classifier model.
#    - Runs repeated evaluation via `train_and_evaluate_all_channels()` using the selected splitting strategy.
#
# 5. **Cross-Validation and Replicate Handling**:
#
#    - If `LOOPC=True`, one sample is randomly selected per class along with all of its replicates, then used as the test set. This ensures that each test fold contains exactly one unique wine per class, and no sample is split across train and test. The rest of the data is used for training.
#    - If `LOOPC=False`, stratified shuffling is used while still preventing replicate leakage.
#
# 6. **Evaluation**:
#
#    - Prints average and standard deviation of balanced accuracy across splits.
#    - Displays label ordering and sample distribution.
#    - Set `show_confusion_matrix=True` to visualize the averaged confusion matrix with matplotlib.
#
# Requirements
# ------------
#
# - Properly structured Pinot Noir GC-MS dataset folders
# - All dependencies installed (see `README.md`)
# - Valid paths and regions configured in `config.yaml`
#
# Usage
# -----
#
# From the root of the repository, run:
#
# .. code-block:: bash
#
#    python scripts/pinot_noir/train_test_pinot_noir.py
# """
# import numpy as np
# import os
# import yaml
# import matplotlib.pyplot as plt
# from gcmswine.classification import Classifier
# from gcmswine import utils
# from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
# from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
# from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified
# from gcmswine.logger_setup import logger, logger_raw
# from sklearn.preprocessing import normalize
# from gcmswine.dimensionality_reduction import DimensionalityReducer
# from scripts.pinot_noir.plotting_pinot_noir import plot_pinot_noir
# from distinctipy import distinctipy
#
# if __name__ == "__main__":
#
#     # # Use this function to convert the printed confusion matrix to a latex confusion matrix
#     # # Copy the matrix into data_str using """ """ and create the list of headers, then call the function
#     # headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
#     #             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
#     #             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
#     # headers = ["Beaune", "Alsace", "Neuchatel", "Genève", "Valais", "Californie", "Oregon"]
#     # headers = ["France", "Switzerland", "US"]
#     # string_to_latex_confusion_matrix_modified(data_str, headers)
#
#     # Load dataset paths from config.yaml
#     config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
#     config_path = os.path.abspath(config_path)
#     with open(config_path, "r") as f:
#         config = yaml.safe_load(f)
#
#     # Parameters from config file
#     dataset_directories = config["datasets"]
#     selected_datasets = config["selected_datasets"]
#
#     # Get the paths corresponding to the selected datasets
#     selected_paths = [config["datasets"][name] for name in selected_datasets]
#
#     # Check if all selected dataset contains "pinot"
#     if not all("pinot_noir" in path.lower() for path in selected_paths):
#         raise ValueError("Please use this script for Pinot Noir datasets.")
#
#     # Infer wine_kind from selected dataset paths
#     wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])
#
#     # Plot parameters
#     plot_projection = config.get("plot_projection", False)
#     projection_method = config.get("projection_method", "UMAP").upper()
#     # projection_source = config.get("projection_source", False)
#     projection_source = config.get("projection_source", False) if plot_projection else False
#     projection_dim = config.get("projection_dim", 2)
#     n_neighbors = config.get("n_neighbors", 30)
#     random_state = config.get("random_state", 42)
#     color_by_country = config["color_by_country"]
#     show_sample_names = config["show_sample_names"]
#     invert_x =  config["invert_x"]
#     invert_y =  config["invert_y"]
#     rot_axes =  config["rot_axes"]
#     sample_display_mode = config["sample_display_mode"]
#     show_year = True if sample_display_mode == "years" else False
#     show_sample_names = True if sample_display_mode == "names" else False
#     color_by_winery = config.get("color_by_winery", False)
#     color_by_origin = config.get("color_by_origin", False)
#     exclude_us = config.get("exclude_us", False)
#     density_plot = config.get("density_plot", False)
#
#     # Run Parameters
#     sotf_ret_time = config.get("sotf_ret_time")
#     sotf_mz = config.get("sotf_mz")
#     if sotf_ret_time and sotf_mz:
#         raise ValueError("Only one of 'sotf_ret_time' or 'sotf_mz' can be True.")
#     feature_type = config["feature_type"]
#     classifier = config["classifier"]
#     num_repeats = config["num_repeats"]
#     normalize_flag = config["normalize"]
#     n_decimation = config["n_decimation"]
#     sync_state = config["sync_state"]
#     class_by_year = config['class_by_year']
#     region = config["region"]
#     # Enforce exclusivity logic
#     if not color_by_origin and not color_by_winery:
#         if region == "origin":
#             color_by_origin = True
#         elif region == "winery":
#             color_by_winery = True
#
#     # wine_kind = config["wine_kind"]
#     show_confusion_matrix = config['show_confusion_matrix']
#     retention_time_range = config['rt_range']
#     cv_type = config['cv_type']
#     task="classification"  # hard-coded for now
#     split_burgundy_ns = True  # config.get("split_burgundy_north_south", False)
#     burg_by_year = True
#     summary = {
#         "Task": task,
#         "Wine kind": wine_kind,
#         "Datasets": ", ".join(selected_datasets),
#         "Feature type": config["feature_type"],
#         "Classifier": config["classifier"],
#         "Repeats": config["num_repeats"],
#         "Normalize": config["normalize"],
#         "Decimation": config["n_decimation"],
#         "Sync": config["sync_state"],
#         "Year Classification": config["class_by_year"],
#         "Region": config["region"],
#         "CV type": config["cv_type"],
#         "RT range": config["rt_range"],
#         "Confusion matrix": config["show_confusion_matrix"]
#     }
#
#     logger_raw("\n")# Blank line with no timestamp
#     logger.info('------------------------ RUN SCRIPT -------------------------')
#     logger.info("Configuration Parameters")
#     for k, v in summary.items():
#         logger_raw(f"{k:>20s}: {v}")
#
#     # strategy = get_strategy_by_wine_kind(wine_kind, get_custom_order_func=utils.get_custom_order_for_pinot_noir_region())
#     strategy = get_strategy_by_wine_kind(
#         wine_kind=wine_kind,
#         region=region,
#         get_custom_order_func=utils.get_custom_order_for_pinot_noir_region,
#     )
#
#     # Create ChromatogramAnalysis instance for optional alignment
#     cl = ChromatogramAnalysis(ndec=n_decimation)
#
#     # Load dataset, removing zero-variance channels
#     selected_paths = {name: dataset_directories[name] for name in selected_datasets}
#     data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
#     chrom_length = len(list(data_dict.values())[0])
#     # print(f'Chromatogram length: {chrom_length}')
#
#     if retention_time_range:
#         min_rt = retention_time_range['min'] // n_decimation
#         raw_max_rt = retention_time_range['max'] // n_decimation
#         max_rt = min(raw_max_rt, chrom_length)
#         logger.info(f"Applying RT range: {min_rt} to {max_rt} (capped at {chrom_length})")
#         data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}
#
#     data_dict, _ = utils.remove_zero_variance_channels(data_dict)
#
#     gcms = GCMSDataProcessor(data_dict)
#
#     def split_into_bins(data, n_bins):
#         """
#         Split TIC into uniform bins (segments).
#         Returns a list of (start_idx, end_idx) for each bin.
#         """
#         total_points = data.shape[1]
#         bin_edges = np.linspace(0, total_points, n_bins + 1, dtype=int)
#         return [(bin_edges[i], bin_edges[i+1]) for i in range(n_bins)]
#
#     def remove_bins(data, bins_to_remove, bin_ranges):
#         """
#         Remove (zero out) specified bins in TIC by index.
#         """
#         data_copy = data.copy()
#         for b in bins_to_remove:
#             start, end = bin_ranges[b]
#             data_copy[:, start:end] = 0  # Mask out that bin segment
#         return data_copy
#
#     def dict_to_array3d(d):
#         """
#         Stack each sample's 2D chromatogram (time × channels) into (N, T, C).
#         Requires each value in data_dict to be a 2D array.
#         """
#         arrs = []
#         for v in d.values():
#             v = np.asarray(v)
#             if v.ndim == 1:
#                 raise ValueError("sotf_mz requires per-channel data (time × channels) per sample.")
#             arrs.append(v)
#         return np.stack(arrs, axis=0)  # (N, T, C)
#
#     if sync_state:
#         tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
#         gcms = GCMSDataProcessor(data_dict)
#
#     # Prepare 3D array for m/z survival mode if requested
#     data3d = None
#     if sotf_mz:
#         data3d = dict_to_array3d(data_dict)  # shape: (N, T, C)
#
#     # === Extract data matrix (samples × channels) and associated labels ===
#     data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
#     raw_sample_labels = labels.copy()  # Save raw labels for annotation
#
#     # Extract only Burgundy if region is "burgundy"
#     if wine_kind == "pinot_noir" and region == "burgundy":
#         burgundy_prefixes = ('D', 'P', 'R', 'Q', 'Z', 'E')
#         mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
#         data = data[mask]
#         labels = labels[mask]
#         raw_sample_labels = raw_sample_labels[mask]
#         if sotf_mz:
#             data3d = data3d[mask]
#
#     labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, class_by_year, None)
#
#     # === Define binning ===
#     n_bins = 50
#     min_bins = 1
#     bin_ranges = split_into_bins(data, n_bins)
#     active_bins = list(range(n_bins))
#
#     # === Define channels (m/z) for m/z survival mode ===
#     if sotf_mz:
#         n_channels = data3d.shape[2]
#         min_channels = 1
#         active_channels = list(range(n_channels))
#
#     # === Iteration logic ===
#     if sotf_ret_time:
#         n_iterations = n_bins - min_bins + 1
#     elif sotf_mz:
#         n_iterations = n_channels - min_channels + 1
#     else:
#         n_iterations = 1
#
#     # === Progressive plot setup (only for survival mode) ===
#     if sotf_ret_time or sotf_mz:
#         cv_label = "LOO" if cv_type == "LOO" else "LOOPC" if cv_type == "LOOPC" else "Stratified"
#         plt.ion()
#         fig, ax = plt.subplots(figsize=(8, 5))
#         line, = ax.plot([], [], marker='o')
#         if sotf_ret_time:
#             ax.set_xlabel("Percentage of TIC data remaining (%)")
#         elif sotf_mz:
#             ax.set_xlabel("Percentage of m/z channels remaining (%)")
#
#         ax.set_ylabel("Accuracy")
#         # ax.set_title(f"Survival of the fittest: Accuracy vs TIC data Remaining ({cv_label} CV)")
#         ax.grid(True)
#         ax.set_xlim(100, 0)  # start at 100% and decrease
#
#     accuracies = []
#     percent_remaining = []
#     baseline_nonzero = np.count_nonzero(data) if sotf_ret_time else None
#
#     # === Main loop ===
#     for step in range(n_iterations):
#         if sotf_mz:
#             logger.info(f"=== Iteration {step + 1}/{n_iterations} | Active channels: {len(active_channels)} ===")
#             # Zero-out removed channels
#             masked3d = data3d.copy()
#             keep = np.zeros(masked3d.shape[2], dtype=bool)
#             keep[active_channels] = True
#             masked3d[:, :, ~keep] = 0
#
#             # Use TIC after channel masking as features (shape: N × T)
#             masked_data = masked3d
#
#             # % remaining by channels
#             pct_data = 100.0 * len(active_channels) / n_channels
#             percent_remaining.append(pct_data)
#
#         else:
#             logger.info(f"=== Iteration {step + 1}/{n_iterations} | Active bins: {len(active_bins)} ===")
#             masked_data = remove_bins(
#                 data,
#                 bins_to_remove=[b for b in range(n_bins) if b not in active_bins],
#                 bin_ranges=bin_ranges
#             )
#             if sotf_ret_time:
#                 pct_data = (np.count_nonzero(masked_data) / baseline_nonzero) * 100
#                 percent_remaining.append(pct_data)
#
#         # Instantiate classifier
#         cls = Classifier(
#             masked_data,
#             labels,
#             classifier_type=classifier,
#             wine_kind=wine_kind,
#             class_by_year=class_by_year,
#             year_labels=np.array(year_labels),
#             strategy=strategy,
#             sample_labels=np.array(raw_sample_labels),
#             dataset_origins=dataset_origins,
#         )
#
#         # Train & evaluate
#         if cv_type in ["LOOPC", "stratified"]:
#             loopc = (cv_type == "LOOPC")
#             mean_acc, std_acc, *_ = cls.train_and_evaluate_all_channels(
#                 num_repeats=num_repeats,
#                 random_seed=42,
#                 test_size=0.2,
#                 normalize=normalize_flag,
#                 scaler_type='standard',
#                 use_pca=False,
#                 vthresh=0.97,
#                 region=region,
#                 print_results=False,
#                 n_jobs=20,
#                 feature_type=feature_type,
#                 classifier_type=classifier,
#                 LOOPC=loopc,
#                 projection_source=projection_source,
#                 show_confusion_matrix=show_confusion_matrix,
#             )
#         elif cv_type == "LOO":
#             mean_acc, std_acc, scores, all_labels, test_samples_names  = cls.train_and_evaluate_leave_one_out_all_samples(
#                 normalize=normalize_flag,
#                 scaler_type='standard',
#                 region=region,
#                 feature_type=feature_type,
#                 classifier_type=classifier,
#                 projection_source=projection_source,
#                 show_confusion_matrix=show_confusion_matrix,
#             )
#         else:
#             raise ValueError(f"Invalid cross-validation type: '{cv_type}'.")
#
#         accuracies.append(mean_acc)
#         logger.info(f"Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
#
#         # === Update live plot (only survival mode) ===
#         if sotf_ret_time or sotf_mz:
#             line.set_data(percent_remaining, accuracies)
#             ax.set_xlim(100, min(percent_remaining) - 5)
#             ax.set_ylim(0, 1)
#             plt.draw()
#             plt.pause(0.2)
#
#         # === Greedy bin removal ===
#         if sotf_ret_time and len(active_bins) > min_bins:
#             candidate_accuracies = []
#             for b in active_bins:
#                 temp_bins = [x for x in active_bins if x != b]
#                 temp_masked_data = remove_bins(
#                     data,
#                     bins_to_remove=[x for x in range(n_bins) if x not in temp_bins],
#                     bin_ranges=bin_ranges
#                 )
#
#                 temp_cls = Classifier(
#                     temp_masked_data,
#                     labels,
#                     classifier_type=classifier,
#                     wine_kind=wine_kind,
#                     class_by_year=class_by_year,
#                     year_labels=np.array(year_labels),
#                     strategy=strategy,
#                     sample_labels=np.array(raw_sample_labels),
#                     dataset_origins=dataset_origins,
#                 )
#
#                 temp_acc, _, *_ = temp_cls.train_and_evaluate_all_channels(
#                     num_repeats=10,  # fewer repeats for speed
#                     random_seed=42,
#                     test_size=0.2,
#                     normalize=normalize_flag,
#                     scaler_type='standard',
#                     use_pca=False,
#                     vthresh=0.97,
#                     region=region,
#                     print_results=False,
#                     n_jobs=10,
#                     feature_type=feature_type,
#                     classifier_type=classifier,
#                     LOOPC=(cv_type == "LOOPC"),
#                     projection_source=projection_source,
#                     show_confusion_matrix=False,
#                 )
#                 candidate_accuracies.append((b, temp_acc))
#
#             # Pick bin whose removal gives best accuracy
#             best_bin, best_candidate_acc = max(candidate_accuracies, key=lambda x: x[1])
#             logger.info(f"Removing bin {best_bin}: next accuracy would be {best_candidate_acc:.3f}")
#             active_bins.remove(best_bin)
#         elif sotf_mz and len(active_channels) > min_channels:
#             candidate_accuracies = []
#             for ch in active_channels:
#                 temp_channels = [x for x in active_channels if x != ch]
#
#                 temp3d = data3d.copy()
#                 keep = np.zeros(n_channels, dtype=bool)
#                 keep[temp_channels] = True
#                 temp3d[:, :, ~keep] = 0
#
#                 temp_cls = Classifier(
#                     temp3d,  # keep 3D
#                     labels,
#                     classifier_type=classifier,
#                     wine_kind=wine_kind,
#                     class_by_year=class_by_year,
#                     year_labels=np.array(year_labels),
#                     strategy=strategy,
#                     sample_labels=np.array(raw_sample_labels),
#                     dataset_origins=dataset_origins,
#                 )
#
#                 temp_acc, _, *_ = temp_cls.train_and_evaluate_all_channels(
#                     num_repeats=10,  # speed
#                     random_seed=42,
#                     test_size=0.2,
#                     normalize=normalize_flag,
#                     scaler_type='standard',
#                     use_pca=False,
#                     vthresh=0.97,
#                     region=region,
#                     print_results=False,
#                     n_jobs=10,
#                     feature_type=feature_type,  # unchanged; let your pipeline decide
#                     classifier_type=classifier,
#                     LOOPC=(cv_type == "LOOPC"),
#                     projection_source=projection_source,
#                     show_confusion_matrix=False,
#                 )
#                 candidate_accuracies.append((ch, temp_acc))
#
#             best_ch, best_candidate_acc = max(candidate_accuracies, key=lambda x: x[1])
#             logger.info(f"Removing m/z channel {best_ch}: next accuracy would be {best_candidate_acc:.3f}")
#             active_channels.remove(best_ch)
#
#
#     # === Finalize plot ===
#     if sotf_ret_time or sotf_mz:
#         plt.ioff()
#         plt.show()
#     else:
#         logger.info(f"Final Accuracy (no survival): {accuracies[0]:.3f}")
#
#
#     if plot_projection:
#         ordered_labels = [
#             "D", "E", "Q", "P", "R", "Z", "C", "W", "Y", "M", "N", "J", "L", "H", "U", "X"
#         ]
#         if region == "winery":
#             legend_labels = {
#                 "D": "D = Clos Des Mouches. Drouhin (FR)",
#                 "R": "R = Les Petits Monts. Drouhin (FR)",
#                 "X": "X = Domaine Drouhin (US)",
#                 "E": "E = Vigne de l’Enfant Jésus. Bouchard (FR)",
#                 "Q": "Q = Les Cailles. Bouchard (FR)",
#                 "P": "P = Bressandes. Jadot (FR)",
#                 "Z": "Z = Les Boudots. Jadot (FR)",
#                 "C": "C = Domaine Schlumberger (FR)",
#                 "W": "W = Domaine Jean Sipp (FR)",
#                 "Y": "Y = Domaine Weinbach (FR)",
#                 "M": "M = Domaine Brunner (CH)",
#                 "N": "N = Vin des Croisés (CH)",
#                 "J": "J = Domaine Villard et Fils (CH)",
#                 "L": "L = Domaine de la République (CH)",
#                 "H": "H = Les Maladaires (CH)",
#                 "U": "U = Marimar Estate (US)",
#         }
#         elif region == "origin":
#             legend_labels = {
#                 "A": "Alsace",
#                 "B": "Burgundy",
#                 "N": "Neuchâtel",
#                 "G": "Geneva",
#                 "V": "Valais",
#                 "C": "California",
#                 "O": "Oregon",
#             }
#         elif region == "country":
#             legend_labels = {
#                 "F": "France",
#                 "S": "Switzerland",
#                 "U": "US"
#             }
#         elif region == "continent":
#             legend_labels = {
#                 "E": "Europe",
#                 "N": "America"
#             }
#         elif region == "burgundy":
#             legend_labels = {
#                 "N": "Côte de Nuits (north)",
#                 "S": "Côte de Beaune (south)"
#             }
#         else:
#             legend_labels = utils.get_custom_order_for_pinot_noir_region(region)
#
#         # Manage plot titles
#         if projection_source == "scores":
#             data_for_umap = normalize(scores)
#             projection_labels = all_labels
#         elif projection_source in {"tic", "tis", "tic_tis"}:
#             channels = list(range(data.shape[2]))  # use all channels
#             data_for_umap = utils.compute_features(data, feature_type=projection_source)
#             data_for_umap = normalize(data_for_umap)
#             projection_labels = labels  # use raw labels from data
#         else:
#             raise ValueError(f"Unknown projection source: {projection_source}")
#
#         # Generate title dynamically
#         pretty_source = {
#             "scores": "Classification Scores",
#             "tic": "TIC",
#             "tis": "TIS",
#             "tic_tis": "TIC + TIS"
#         }.get(projection_source, projection_source)
#
#         pretty_method = {
#             "UMAP": "UMAP",
#             "T-SNE": "t-SNE",
#             "PCA": "PCA"
#         }.get(projection_method, projection_method)
#
#         pretty_region = {
#             "winery": "Winery",
#             "origin": "Origin",
#             "country": "Country",
#             "continent": "Continent",
#             "burgundy": "N/S Burgundy"
#         }.get(region, region)
#
#         if region:
#             plot_title = f"{pretty_method} of {pretty_source} ({pretty_region})"
#         else:
#             plot_title = f"{pretty_method} of {pretty_source}"
#
#         # Disable showing sample names
#         if not show_sample_names:
#             test_samples_names = None
#
#
#         legend_labels = {
#             "D": "D = Clos Des Mouches. Drouhin (FR)",
#             "R": "R = Les Petits Monts. Drouhin (FR)",
#             "X": "X = Domaine Drouhin (US)",
#             "E": "E = Vigne de l’Enfant Jésus. Bouchard (FR)",
#             "Q": "Q = Les Cailles. Bouchard (FR)",
#             "P": "P = Bressandes. Jadot (FR)",
#             "Z": "Z = Les Boudots. Jadot (FR)",
#             "C": "C = Domaine Schlumberger (FR)",
#             "W": "W = Domaine Jean Sipp (FR)",
#             "Y": "Y = Domaine Weinbach (FR)",
#             "M": "M = Domaine Brunner (CH)",
#             "N": "N = Vin des Croisés (CH)",
#             "J": "J = Domaine Villard et Fils (CH)",
#             "L": "L = Domaine de la République (CH)",
#             "H": "H = Les Maladaires (CH)",
#             "U": "U = Marimar Estate (US)",
#         }
#
#         if data_for_umap is not None:
#             reducer = DimensionalityReducer(data_for_umap)
#             if projection_method == "UMAP":
#                 plot_pinot_noir(
#                     reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state),
#                     plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names,
#                     unique_samples_only=False, n_neighbors=n_neighbors, random_state=random_state,
#                     invert_x=invert_x, invert_y=invert_y, rot_axes=rot_axes,
#                     raw_sample_labels=raw_sample_labels, show_year=show_year,
#                     color_by_origin=color_by_origin, color_by_winery=color_by_winery, highlight_burgundy_ns=False,
#                     exclude_us=exclude_us, density_plot=density_plot,
#                     region=region
#                 )
#             elif projection_method == "PCA":
#                 plot_pinot_noir(reducer.pca(components=projection_dim),plot_title, projection_labels, legend_labels,
#                                 color_by_country, test_sample_names=test_samples_names)
#             elif projection_method == "T-SNE":
#                 plot_pinot_noir(reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state),
#                         plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names
#                         )
#             # if projection_method == "UMAP":
#             #     plot_pinot_noir(
#             #         reducer.umap(components=projection_dim, n_neighbors=n_neighbors, random_state=random_state),
#             #         plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names,
#             #         unique_samples_only=False, n_neighbors=n_neighbors, random_state=random_state,
#             #         invert_x=invert_x, invert_y=invert_y,
#             #         only_europe=False, split_burgundy_north_south=False, raw_sample_labels=raw_sample_labels, region=region
#             #     )
#             # elif projection_method == "PCA":
#             #     plot_pinot_noir(reducer.pca(components=projection_dim),plot_title, projection_labels, legend_labels,
#             #                     color_by_country, test_sample_names=test_samples_names,
#             #             unique_samples_only = False, n_neighbors = n_neighbors, random_state = random_state,
#             #             invert_x = invert_x, invert_y = invert_y
#             #             )
#             # elif projection_method == "T-SNE":
#             #     plot_pinot_noir(reducer.tsne(components=projection_dim, perplexity=5, random_state=random_state),
#             #             plot_title, projection_labels, legend_labels, color_by_country, test_sample_names=test_samples_names,
#             #             unique_samples_only = False, n_neighbors = n_neighbors, random_state = random_state,
#             #             invert_x = invert_x, invert_y = invert_y
#             #             )
#             else:
#                 raise ValueError(f"Unsupported projection method: {projection_method}")
#
#
#         plt.show()
#
#
