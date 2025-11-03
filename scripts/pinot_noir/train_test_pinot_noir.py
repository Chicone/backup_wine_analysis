"""
Refactored version of train_test_pinot_noir.py
- Cleanly separates: (A) Normal classification, (B) SOTF over retention time, (C) SOTF over m/z channels
- Preserves all existing functionality, plotting, and printing behaviors
- Keeps LOO / LOOPC / Stratified exactly as in original

"""

# -----------------------------
# Imports
# -----------------------------
import os
import yaml
from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from gcmswine.wine_kind_strategy import get_strategy_by_wine_kind
from gcmswine.logger_setup import logger
from sklearn.preprocessing import normalize
from gcmswine.dimensionality_reduction import DimensionalityReducer
from scripts.pinot_noir.plotting_pinot_noir import plot_pinot_noir
from gcmswine.utils import compute_features
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_val_score
import re
from collections import defaultdict
from joblib import Parallel, delayed
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
import matplotlib.pyplot as plt
from gcmswine.config import build_run_config
from gcmswine.logger_setup import  logger_raw
import numpy as np
import pandas as pd


# -----------------------------
# Config & Data Loading
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
# Utility helpers
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


# -----------------------------
# Scoring/Estimation
# -----------------------------
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

# -----------------------------
# Cross-validation Wrappers
# -----------------------------
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
    origins: np.ndarray,
    show_pred_plot=False,
    pred_plot_region="all",
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
                # feature_type="greedy_add",
                feature_type=feature_type,
                classifier_type=classifier,
                projection_source=projection_source,
                show_confusion_matrix=show_confusion_matrix,
            )

        # from sklearn.linear_model import LinearRegression
        # from sklearn.metrics import r2_score
        #
        # # Reshape predictions for regression
        # y_true = np.array(all_labels, dtype=float)  # vintages (true)
        # X_pred = np.array(all_preds, dtype=float).reshape(-1, 1)  # LDA predictions
        #
        # reg = LinearRegression()
        # reg.fit(X_pred, y_true)
        #
        # y_fit = reg.predict(X_pred)
        #
        # r = np.corrcoef(y_true, X_pred.ravel())[0, 1]
        # r2 = r2_score(y_true, y_fit)
        #
        # print(f"LDA post-hoc regression: R = {r:.3f}, R² = {r2:.3f}")

        # Predicted vs. True scatter plot, colored by origin ---
        if show_pred_plot:

            try:
                 plot_result = plot_true_vs_pred(all_labels, all_preds, origins, cfg.pred_plot_mode,
                                  cls.year_labels, cls.data, feature_type, pred_plot_region)

                 # If coefficients (or other metrics) are returned, store them
                 if isinstance(plot_result, tuple) and len(plot_result) >= 6:
                     *_, all_coefs = plot_result

            except Exception as e:
                print(f"⚠️ Skipping predicted vs true year plot due to error: {e}")
        else:
            all_coefs = None

        return mean_acc, std_acc, all_scores, all_labels, test_sample_names, all_preds, all_coefs
        # return mean_acc, std_acc, all_scores, all_labels, test_sample_names, all_preds

    else:
        raise ValueError(f"Invalid cross-validation type: '{cv_type}'.")

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


# -----------------------------
# Feature Selection / SOTF
# -----------------------------
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

def run_sotf_ret_time_leaky(
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
       Greedy removal over retention-time bins (LEAKY VERSION).

       ⚠️ Important:
           This implementation is NOT leakage-free.
           Bin removal decisions are based on accuracies averaged across all
           outer CV folds, including test folds. This causes data leakage and
           leads to optimistically biased accuracy estimates.

       Behavior:
           - Precompute features and split into retention-time bins.
           - Baseline accuracy computed with outer CV.
           - Greedy loop: at each step, remove the bin whose removal maximizes
             mean outer accuracy (using all folds, test data included).
           - Produces an accuracy curve and % of bins remaining.

       Use cases:
           - Exploratory analysis of which retention-time bins are influential.
           - Visualization and hypothesis generation.

       Not suitable for:
           - Unbiased model evaluation or reporting true generalization accuracy.
           - Publication-quality results (use a nested CV "noleak" version instead).
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
        y_test = (year_labels[test_idx] if class_by_year else labels[test_idx])
        res = safe_train_eval(X_train, y_train, X_test, y_test, train_idx)
        baseline_fold_accs.append(res["balanced_accuracy"])

    baseline_acc = float(np.mean(baseline_fold_accs))
    accuracies.append(baseline_acc)
    percent_remaining.append(100.0)
    line.set_data(percent_remaining, accuracies)
    plt.draw();
    plt.pause(0.2)
    logger.info(f"Baseline (all bins): mean outer acc = {baseline_acc:.3f}")

    # 4) Greedy loop (global decision using all outer folds)
    for step in range(n_iterations):
        if len(active_bins) <= min_bins:
            break

        logger.info(f"=== Iteration {step + 1}/{n_iterations} | Active bins: {len(active_bins)} ===")
        already_removed = [b for b in range(n_bins) if b not in active_bins]
        candidate_scores = []

        for b in active_bins:
            bins_to_remove = already_removed + [b]
            fold_accs = []

            for train_idx, test_idx in outer_splits:
                X_train_full, X_test_full = X_proc[train_idx], X_proc[test_idx]
                y_train = (year_labels[train_idx] if class_by_year else labels[train_idx])
                y_test = (year_labels[test_idx] if class_by_year else labels[test_idx])

                # Drop columns using ORIGINAL bin_ranges
                X_train = remove_bins(X_train_full, bins_to_remove=bins_to_remove, bin_ranges=bin_ranges)
                X_test = remove_bins(X_test_full, bins_to_remove=bins_to_remove, bin_ranges=bin_ranges)

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
        plt.draw();
        plt.pause(0.2)

        logger.info(f"Iteration {step + 1}: removed bin {best_bin}, "
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

def run_sotf_mz_leaky(
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
    Greedy removal over m/z channels (LEAKY VERSION).

    ⚠️ Important:
        This implementation is NOT leakage-free.
        Channel removal decisions are made using accuracies averaged across all
        outer CV folds, including test folds. This introduces data leakage and
        results in overly optimistic accuracy estimates.

    Behavior:
        - Baseline accuracy computed with outer CV.
        - Greedy loop: at each step, evaluates removal of candidate channels
          (or contiguous groups of channels) using all folds.
        - The best group is removed globally, and the process repeats.
        - Accuracy curve vs. % channels remaining is plotted.

    Use cases:
        - Exploratory inspection of which m/z channels may carry information.
        - Visualization and hypothesis generation.

    Not suitable for:
        - Unbiased accuracy estimation.
        - Proper nested CV experiments (use a "noleak" variant instead).
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

def run_sotf_remove_2d_leaky(
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
    2D Greedy Remove over RT × m/z cubes (LEAKY VERSION).

    ⚠️ Warning:
        This implementation suffers from data leakage.
        At each step, candidate cube removals are chosen by averaging
        accuracy across ALL outer CV folds, including their test folds.
        This means test data influences feature selection, leading to
        overly optimistic accuracy estimates.

    Use only for:
        - Exploratory visualization of informative cubes
        - Hypothesis generation

    Not suitable for:
        - Unbiased evaluation
        - Publication-quality results
        (use run_sotf_remove_noleak instead)
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
    if feature_type == "concatenated":
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
        if feature_type == "concatenated":
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
                if feature_type == "concatenated":
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

def run_sotf_remove_2D_noleak(
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
    sample_frac=1.0,   # not used for remove (kept for API parity)
    rt_min=None, rt_max=None, mz_min=None, mz_max=None,
    cube_repr="tic",
    cv_design="A",     # NEW: "A" = original (outer=RSKFold, inner=cv_type), "B" = outer=LOO, inner=Stratified
):
    """
        2D Survival-of-the-Fittest (Greedy Remove), leak-free.

        This function performs feature ablation over RT × m/z "cubes"
        using a nested cross-validation design to avoid data leakage:

        - Outer CV:
            Provides unbiased test-set accuracy at each step of the removal process.
            Two design options:
                * "A": RepeatedStratifiedKFold outer loop; inner loop = cv_type (LOO, Stratified, LOOPC).
                * "B": Leave-One-Out outer loop; inner loop = StratifiedKFold.

        - Inner CV:
            Within each outer training set, cubes are greedily removed one at a time.
            At each step, the cube whose removal yields the best inner CV score
            is selected. The resulting removal order is applied to the outer test set.

        - Outputs:
            * Accuracy curve vs % data remaining (leak-free, averaged over folds).
            * Heatmap showing the most common removal order across folds.

        Parameters
        ----------
        data3d : ndarray (N, T, C)
            Chromatographic data (samples × retention time × m/z channels).
        labels : array-like
            Class labels (per sample).
        raw_sample_labels : array-like
            Sample IDs or raw labels, used for stratification / bookkeeping.
        year_labels : array-like
            Vintage/year labels (optional, used if class_by_year=True).
        classifier : str
            Classifier name (passed into Classifier wrapper).
        wine_kind : str
            Type of wine dataset ("pinot_noir", "bordeaux", etc.).
        class_by_year : bool
            If True, classify vintages instead of classes.
        strategy : object
            Label-handling strategy (ordering, grouping, etc.).
        dataset_origins : array-like
            Tracks dataset provenance for each sample.
        cv_type : str
            Inner CV type if cv_design="A" ("LOO", "stratified", or "LOOPC").
        num_repeats : int
            Number of repetitions for repeated CV.
        normalize_flag : bool
            Whether to normalize features before classification.
        region : str
            Region label (used in reporting/plots).
        feature_type : str
            Feature representation ("concatenated", "tic", "tis", etc.).
        n_rt_bins, n_mz_bins : int
            Resolution of cube grid along retention time and m/z axes.
        min_cubes : int
            Minimum number of cubes to retain before stopping.
        max_cubes : int or None
            Maximum number of removal steps (default = all cubes).
        cube_repr : str
            Cube feature aggregation mode ("tic", "tis", "tic_tis", "concatenate").
        cv_design : {"A", "B"}
            CV design option (see above).
        n_jobs : int
            Parallel jobs for outer folds.
        random_state : int
            RNG seed.

        Returns
        -------
        all_outer_curves : list of lists
            Accuracy curves for each outer fold.
        all_removed_orders : list of lists
            Removal order (sequence of cubes) per outer fold.

        Notes
        -----
        - This is the correct *leak-free* implementation.
        - Use this for publication-quality results.
        - For quick exploratory runs, the corresponding `*_leaky` versions may
          be faster but will overestimate accuracy.
        """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter, defaultdict
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
    from joblib import Parallel, delayed
    from tqdm import trange

    # time.sleep(5000)

    rng = np.random.default_rng(random_state)

    # === Crop data ===
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
        max_cubes = len(all_cubes)
    n_iterations = max_cubes

    y_all = np.array((year_labels if class_by_year else labels))
    classes = np.unique(y_all)
    n_classes = len(classes)

    # === Precompute features if concatenated ===
    precomputed_features, cube_dims = None, None
    if feature_type == "concatenated":
        precomputed_features, cube_dims = {}, {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            cube = data3d[:, rt_start:rt_end, mz_start:mz_end]  # (N, rt, mz)
            cr = "flat" if cube_repr == "concatenate" else cube_repr
            if cr == "flat":
                feats = cube.reshape(n_samples, -1)
            elif cr == "tic":
                feats = np.sum(cube, axis=2)
            elif cr == "tis":
                feats = np.sum(cube, axis=1)
            elif cr == "tic_tis":
                tic = np.sum(cube, axis=2)
                tis = np.sum(cube, axis=1)
                feats = np.hstack([tic, tis])
            else:
                raise ValueError(f"Unsupported cube_repr: {cube_repr}")
            feats = feats.astype(np.float32, copy=False)
            precomputed_features[(i, j)] = feats
            cube_dims[(i, j)] = feats.shape[1]
    else:
        subcubes = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)

    # === Prepare y_for_split ===
    if class_by_year:
        y_for_split = np.asarray(year_labels)
    else:
        y_for_split = np.asarray([str(lbl)[0] for lbl in labels])

    # === Outer CV selection ===
    dummy_X = np.zeros((n_samples, 1))
    if cv_design == "A":
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_repeats, random_state=random_state)
        outer_splits = list(outer_cv.split(dummy_X, y_for_split))
        cv_label = f"RepeatedStratified (5x{num_repeats})"
    elif cv_design == "B":
        outer_cv = LeaveOneOut()
        outer_splits = list(outer_cv.split(dummy_X, y_all))
        cv_label = "LOO outer / Stratified inner"
    else:
        raise ValueError("cv_design must be 'A' or 'B'")

    # === Safe train/eval wrapper ===
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
                from sklearn.metrics import accuracy_score, balanced_accuracy_score
                y_pred = cls_wrap.classifier.predict(Xte)
                return {"balanced_accuracy": balanced_accuracy_score(yte, y_pred)}
            else:
                raise

    # === Inner CV factory ===
    def make_inner_cv(y_outer_train, raw_outer):
        if cv_design == "A":
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
        elif cv_design == "B":
            return RepeatedStratifiedKFold(n_splits=3, n_repeats=num_repeats, random_state=random_state)\
                   .split(np.zeros((len(y_outer_train), 1)), y_outer_train)
            # return StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)\
            #        .split(np.zeros((len(y_outer_train), 1)), y_outer_train)

    # === Utility to build full design ===
    def build_full_design_and_slices(idx):
        parts, slices, col = [], {}, 0
        for c in all_cubes:
            F = precomputed_features[c][idx]
            d = cube_dims[c]
            parts.append(F)
            slices[c] = slice(col, col + d)
            col += d
        X_full = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        return X_full, slices

    # === One outer fold worker ===
    def run_one_outer_fold(fold_id, outer_train_idx, outer_test_idx, fold_seed):
        y_train_outer = y_all[outer_train_idx]
        y_test_outer  = y_all[outer_test_idx]

        raw_outer = np.array(raw_sample_labels)[outer_train_idx] if raw_sample_labels is not None else None

        # --- Singleton filtering for inner CV ---
        if (cv_design == "A" and cv_type in ("LOO", "LOOPC")) or cv_design == "B":
            _, y_train_outer, raw_outer = filter_singletons(
                data3d[outer_train_idx],
                y_train_outer,
                raw_labels=raw_outer,
                class_by_year=class_by_year,
            )
        # raw_outer = np.array(raw_sample_labels)[outer_train_idx] if raw_sample_labels is not None else None
        inner_cv = list(make_inner_cv(y_train_outer, raw_outer))

        remaining = set(all_cubes)
        removed_order, outer_curve = [], []

        if feature_type == "concatenated":
            X_train_full, col_slices_tr = build_full_design_and_slices(outer_train_idx)
            X_test_full,  col_slices_te = build_full_design_and_slices(outer_test_idx)
            active_cols_tr = np.ones(X_train_full.shape[1], dtype=bool)
            active_cols_te = np.ones(X_test_full.shape[1], dtype=bool)

            # --- Baseline accuracy with ALL cubes ---
            res_outer = safe_train_eval(
                X_train_full, y_all[outer_train_idx],
                X_test_full, y_all[outer_test_idx],
                outer_train_idx
            )
            outer_curve.append(res_outer["balanced_accuracy"])

        for step in trange(n_iterations, desc=f"Fold {fold_id}", leave=False):
            if len(remaining) <= min_cubes:
                break
            candidates = list(remaining)
            best_cube, best_score = None, -np.inf

            for cube in candidates:
                accs = []
                if feature_type == "concatenated":
                    s_tr, s_te = col_slices_tr[cube], col_slices_te[cube]
                    tmp_mask_tr, tmp_mask_te = active_cols_tr.copy(), active_cols_te.copy()
                    tmp_mask_tr[s_tr] = False
                    tmp_mask_te[s_te] = False
                    X_tr_all = X_train_full[:, tmp_mask_tr]
                for tr_rel, val_rel in inner_cv:
                    tr_idx = outer_train_idx[tr_rel]
                    val_idx = outer_train_idx[val_rel]
                    if feature_type == "concatenated":
                        X_tr = X_tr_all[tr_rel]
                        X_val = X_train_full[:, tmp_mask_tr][val_rel]
                    else:
                        masked = np.zeros_like(data3d)
                        for (ii, jj) in remaining:
                            if (ii, jj) == cube: continue
                            rts, rte, mzs, mze = subcubes[(ii, jj)]
                            masked[:, rts:rte, mzs:mze] = data3d[:, rts:rte, mzs:mze]
                        X_tr = utils.compute_features(masked[tr_idx], feature_type=feature_type)
                        X_val = utils.compute_features(masked[val_idx], feature_type=feature_type)
                    y_tr, y_val = y_all[tr_idx], y_all[val_idx]
                    res = safe_train_eval(X_tr, y_tr, X_val, y_val, tr_idx)
                    accs.append(res["balanced_accuracy"])
                mean_acc = float(np.mean(accs))
                if mean_acc > best_score:
                    best_score, best_cube = mean_acc, cube

            remaining.remove(best_cube)
            removed_order.append(best_cube)
            if feature_type == "concatenated":
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

        chance_acc = 1.0 / n_classes
        outer_curve.append(chance_acc)

        # === Ensure the final cube is logged so the heatmap shows something ===
        if len(remaining) == 1 and min_cubes >= 1:
            last_cube = next(iter(remaining))
            if not removed_order or removed_order[-1] != last_cube:
                removed_order.append(last_cube)

        return outer_curve, removed_order

    # === Live plotting ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    line, = ax_curve.plot([], [], marker="o")
    ax_curve.set_xlabel("Percentage of data remaining")
    ax_curve.set_ylabel("Accuracy")
    ax_curve.grid(True)
    ax_curve.set_title(f"2D SOTF Greedy Remove ({cv_label})\nClassifier: {classifier}, Feature: {feature_type}")
    order_positions = defaultdict(Counter)

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

        # --- Update accuracy curve ---
        max_len = max(len(c) for c in all_outer_curves)
        avg_curve = np.array([
            np.mean([c[k] for c in all_outer_curves if len(c) > k])
            for k in range(max_len)
        ])

        # Correct x-axis: baseline = 0%, last point = 100%
        steps = np.arange(max_len)  # 0..N-1 (baseline included at 0)
        x = 100 * (1 - steps / len(all_cubes))  # fraction of cubes removed
        y = avg_curve

        line.set_data(x, y)
        ax_curve.relim()
        ax_curve.autoscale_view()
        ax_curve.set_xlim(100, 0)  # <-- flip axis so it shows 100 → 0


        # --- Update heatmap ---
        for step, (i, j) in enumerate(removed_order):
            order_positions[(i, j)][step + 1] += 1  # step+1 (since baseline is step=0)

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
            xticklabels=[f"{rt_edges[x]}–{rt_edges[x + 1]}" for x in range(n_rt_bins)],
            yticklabels=[f"{mz_edges[y]}–{mz_edges[y + 1]}" for y in range(n_mz_bins)],
            annot=annotations, fmt="s", annot_kws={"size": 6}
        )

        ax_heat.set_title(f"Most common removal order (after {len(all_outer_curves)} folds)")
        ax_heat.set_xticklabels(ax_heat.get_xticklabels(), rotation=45, ha="right")
        ax_heat.set_yticklabels(ax_heat.get_yticklabels(), rotation=0)
        ax_heat.invert_yaxis()

        # Update window title with progress
        try:
            fig.canvas.manager.set_window_title(
                f"Greedy Remove Progress: Fold {fold_id}/{len(outer_splits)}"
            )
        except Exception:
            pass

        plt.pause(0.3)

    plt.ioff()
    plt.show()

    return all_outer_curves, all_removed_orders

def run_sotf_add_2d_leaky(
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
    2D Greedy Add (SOTF), **leaky version**.

    Procedure:
        - Outer CV is used both for unbiased test evaluation and for selecting
          which cubes to add, causing information leakage from the test folds.
        - At each step, candidate cubes are evaluated across outer folds and the
          best one is added to the feature set.
        - Produces per-fold accuracy curves and a heatmap of most common cube
          selection order.

    Note:
        This version overestimates performance because cube selection is
        influenced by test folds. Use the *_noleak variant for correct estimates.
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
    if feature_type == "concatenated":
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

                    if feature_type == "concatenated":
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
            if feature_type == "concatenated":
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

def run_sotf_add_2d_noleak(
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
    cv_design="A",  # "A" = outer=RSKFold, inner=cv_type; "B" = outer=LOO, inner=Stratified
):
    """
    2D Survival-of-the-Fittest (Greedy Add) over RT × m/z cubes (LEAK-FREE).

    cv_design:
        "A" = Outer = RepeatedStratifiedKFold, inner = cv_type ("LOO", "stratified", "LOOPC").
        "B" = Outer = LeaveOneOut, inner = StratifiedKFold.

    Behavior:
        - Start from chance-level accuracy (0% cubes).
        - Iteratively add cubes one by one, using nested CV.
        - Outer folds are used only for final scoring (no leakage).
        - Produces accuracy curves and a heatmap of addition order.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter, defaultdict
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold, RepeatedStratifiedKFold
    from joblib import Parallel, delayed
    from tqdm import trange

    rng = np.random.default_rng(random_state)

    # === Crop data ===
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
        max_cubes = len(all_cubes)
    n_iterations = max_cubes

    y_all = np.asarray(year_labels if class_by_year else labels)
    raw_sample_labels = np.asarray(raw_sample_labels) if raw_sample_labels is not None else None
    year_labels = np.asarray(year_labels) if year_labels is not None else None
    labels = np.asarray(labels)

    classes = np.unique(y_all)
    n_classes = len(classes)

    # === Precompute features for concatenated ===
    precomputed_features, subcubes = None, None
    if feature_type == "concatenated":
        precomputed_features = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            cube = data3d[:, rt_start:rt_end, mz_start:mz_end]

            # this is for the representation of each cube
            cr = "flat" if cube_repr == "concatenate" else cube_repr
            if cr == "flat":
                feats = cube.reshape(n_samples, -1)
            elif cr == "tic":
                feats = np.sum(cube, axis=2)
            elif cr == "tis":
                feats = np.sum(cube, axis=1)
            elif cr == "tic_tis":
                tic = np.sum(cube, axis=2)
                tis = np.sum(cube, axis=1)
                feats = np.hstack([tic, tis])
            else:
                raise ValueError(f"Unsupported cube_repr: {cube_repr}")

            precomputed_features[(i, j)] = feats.astype(np.float32, copy=False)
    else:
        subcubes = {}
        for (i, j) in all_cubes:
            rt_start, rt_end = rt_edges[i], rt_edges[i + 1]
            mz_start, mz_end = mz_edges[j], mz_edges[j + 1]
            subcubes[(i, j)] = (rt_start, rt_end, mz_start, mz_end)

    # === Outer CV selection ===
    dummy_X = np.zeros((n_samples, 1))
    if cv_design == "A":
        outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=num_repeats, random_state=random_state)
        outer_splits = list(outer_cv.split(dummy_X, y_all))
        cv_label = f"RepeatedStratified (5x{num_repeats})"
    elif cv_design == "B":
        outer_cv = LeaveOneOut()
        outer_splits = list(outer_cv.split(dummy_X, y_all))
        cv_label = "LOO outer / Stratified inner"
    else:
        raise ValueError("cv_design must be 'A' or 'B'")

    # === Safe train/eval ===
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
        except Exception:
            return {"balanced_accuracy": 1.0 / n_classes}

    # === Inner CV factory ===
    def make_inner_cv(y_outer_train, raw_outer):
        # create a slightly different seed each time the function is called
        local_seed = np.random.default_rng(random_state).integers(1e6)

        if cv_design == "A":
            if cv_type == "LOO":
                return LeaveOneOut().split(np.zeros((len(y_outer_train), 1)), y_outer_train)
            elif cv_type == "stratified":
                rskf = RepeatedStratifiedKFold(
                    n_splits=5,
                    n_repeats=5,
                    random_state=local_seed
                )
                return rskf.split(np.zeros((len(y_outer_train), 1)), y_outer_train)

                # return StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)\
                #        .split(np.zeros((len(y_outer_train), 1)), y_outer_train)
            elif cv_type == "LOOPC":
                return loopc_splits(
                    y_outer_train,
                    num_repeats=num_repeats, random_state=random_state,
                    class_by_year=class_by_year,
                    raw_labels=raw_outer if class_by_year else None,
                )
            else:
                raise ValueError(f"Unsupported inner cv_type: {cv_type}")
        elif cv_design == "B":
            return StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)\
                   .split(np.zeros((len(y_outer_train), 1)), y_outer_train)

    # === One outer fold worker ===
    def run_one_outer_fold(fold_id, outer_train_idx, outer_test_idx, fold_seed):
        y_train_outer = y_all[outer_train_idx]
        y_test_outer = y_all[outer_test_idx]
        raw_outer = np.array(raw_sample_labels)[outer_train_idx] if raw_sample_labels is not None else None

        # filter singletons if needed
        if (cv_design == "A" and cv_type in ("LOO", "LOOPC")) or cv_design == "B":
            _, y_train_outer, raw_outer = filter_singletons(
                data3d[outer_train_idx],
                y_train_outer,
                raw_labels=raw_outer,
                class_by_year=class_by_year,
            )

        inner_cv = list(make_inner_cv(y_train_outer, raw_outer))

        added_cubes, not_added = [], all_cubes.copy()
        cube_sequence, outer_curve = [], []
        X_train_running, X_test_running = None, None

        for step in trange(n_iterations, desc=f"Fold {fold_id}", leave=False):
            if len(added_cubes) >= max_cubes:
                break

            candidates = not_added
            best_cube, best_score = None, -np.inf

            for cube in candidates:
                candidate_add = added_cubes + [cube]
                accs = []

                if feature_type == "concatenated":
                    feats_c = precomputed_features[cube]
                    feats_train_c = feats_c[outer_train_idx]

                for tr_rel, val_rel in inner_cv:
                    tr_idx = outer_train_idx[tr_rel]
                    val_idx = outer_train_idx[val_rel]
                    if feature_type == "concatenated":
                        if X_train_running is None:
                            X_tr = feats_train_c[tr_rel]
                            X_val = feats_train_c[val_rel]
                        else:
                            X_tr = np.concatenate((X_train_running[tr_rel], feats_train_c[tr_rel]), axis=1)
                            X_val = np.concatenate((X_train_running[val_rel], feats_train_c[val_rel]), axis=1)
                    else:
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

                mean_acc = np.mean(accs)
                if mean_acc > best_score:
                    best_score, best_cube = mean_acc, cube

            # commit best cube
            added_cubes.append(best_cube)
            not_added.remove(best_cube)
            cube_sequence.append(best_cube)

            # build outer train/test once for chosen cube
            if feature_type == "concatenated":
                feats_best = precomputed_features[best_cube]
                feats_train_best = feats_best[outer_train_idx]
                feats_test_best = feats_best[outer_test_idx]
                if X_train_running is None:
                    X_train_running = feats_train_best
                    X_test_running = feats_test_best
                else:
                    X_train_running = np.concatenate((X_train_running, feats_train_best), axis=1)
                    X_test_running = np.concatenate((X_test_running, feats_test_best), axis=1)
                X_train, X_test = X_train_running, X_test_running
            else:
                masked = np.zeros_like(data3d)
                for (ii, jj) in added_cubes:
                    rts, rte, mzs, mze = subcubes[(ii, jj)]
                    masked[:, rts:rte, mzs:mze] = data3d[:, rts:rte, mzs:mze]
                X_train = compute_features(masked[outer_train_idx], feature_type=feature_type)
                X_test = compute_features(masked[outer_test_idx], feature_type=feature_type)

            res_outer = safe_train_eval(X_train, y_train_outer, X_test, y_test_outer, outer_train_idx)
            outer_curve.append(res_outer["balanced_accuracy"])

        return outer_curve, cube_sequence

    # === Live plotting ===
    plt.ion()
    fig, (ax_curve, ax_heat) = plt.subplots(1, 2, figsize=(14, 6))
    line, = ax_curve.plot([], [], marker="o")
    baseline_acc = 1.0 / n_classes
    line.set_data([0], [baseline_acc])  # 0% added = chance
    ax_curve.set_xlabel("Percentage of data added")
    ax_curve.set_ylabel("Accuracy")
    ax_curve.grid(True)
    ax_curve.set_title(f"2D SOTF Greedy Add ({cv_label})\nClassifier: {classifier}, Feature: {feature_type}")

    baseline_acc = 1.0 / n_classes
    line.set_data([0], [baseline_acc])

    order_positions = defaultdict(Counter)
    mode_matrix = np.full((n_mz_bins, n_rt_bins), np.nan)
    sns.heatmap(mode_matrix, vmin=1, vmax=n_iterations,
                cmap="viridis", ax=ax_heat, cbar=True,
                xticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
                yticklabels=[f"{mz_edges[j]}–{mz_edges[j+1]}" for j in range(n_mz_bins)])
    ax_heat.set_title("Most common cube order")
    ax_heat.invert_yaxis()
    ax_heat.tick_params(axis="x", labelrotation=45)
    for t in ax_heat.get_xticklabels():
        t.set_horizontalalignment("right")

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

        max_len = max(len(c) for c in all_outer_curves)
        avg_curve = np.array([
            np.mean([c[k] for c in all_outer_curves if len(c) > k])
            for k in range(max_len)
        ])
        x = np.linspace(0, 100, len(avg_curve) + 1)
        y = np.concatenate(([baseline_acc], avg_curve))
        line.set_data(x, y)
        ax_curve.relim(); ax_curve.autoscale_view()

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

        try:
            fig.canvas.manager.set_window_title(f"Greedy Add Progress: Fold {fold_id}/{len(outer_splits)}")
        except Exception:
            pass
        plt.pause(0.3)

    plt.ioff()
    plt.show()

    # === NEW: Save RT-bin frequency distribution for every addition step ===
    from collections import Counter
    import os
    os.makedirs("results", exist_ok=True)

    max_len = max(len(c) for c in all_selected_cubes)
    n_steps_total = max_len
    n_folds = len(all_selected_cubes)

    # Per-step RT-bin counts, but each fold contributes at most once per RT bin overall
    rt_freq_by_step = np.zeros((n_steps_total, n_rt_bins), dtype=int)

    for fold_cubes in all_selected_cubes:
        seen_rt = set()  # RT bins already credited by this fold
        for step, (i, j) in enumerate(fold_cubes, start=1):
            if i in seen_rt:
                continue  # don't double-count this RT bin for this fold
            rt_freq_by_step[step - 1, i] += 1
            seen_rt.add(i)

    # Cumulative (now bounded by n_folds)
    rt_freq_cumulative = np.cumsum(rt_freq_by_step, axis=0)

    save_path_full = (
        f"results/rt_distribution_allsteps_{region}_{feature_type}_"
        f"{n_rt_bins}rt_{n_mz_bins}mz_{num_repeats}rep.npz"
    )
    np.savez(
        save_path_full,
        rt_edges=np.array(rt_edges),
        rt_freq_by_step=rt_freq_by_step,
        rt_freq_cumulative=rt_freq_cumulative,
        n_steps=n_steps_total,
        n_folds=np.int32(n_folds),  # <-- save folds for proper normalization
    )
    print(f"[INFO] Saved full RT-bin distribution across all {n_steps_total} additions to {save_path_full}")

    # --- Build per-step selection frequency (first N steps only) ---
    cube_distribution_by_step = [Counter() for _ in range(n_steps_total)]
    for fold_cubes in all_selected_cubes:
        for step, cube in enumerate(fold_cubes[:n_steps_total], start=1):
            cube_distribution_by_step[step - 1][cube] += 1

    # --- Aggregate cube selections across the first N steps ---
    combined_first_bins = Counter()
    for s in range(n_steps_total):
        combined_first_bins += cube_distribution_by_step[s]

    # --- Sum counts across m/z bins to get per-RT-bin frequency ---
    rt_freq_first = np.zeros(n_rt_bins, dtype=float)
    for (i, j), count in combined_first_bins.items():
        rt_freq_first[i] += count

    return all_outer_curves, all_selected_cubes


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
    if feature_type == "concatenated":
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
        if feature_type == "concatenated":
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


# -----------------------------
# Plotting
# -----------------------------
import seaborn as sns
def plot_rtbin_evolution_full(
    npz_path,
    all_coefs,
    ref_chrom,
    rt_edges=None,
    normalize="counts",   # "counts", "global", "column", or "row"
    cumulative=True,
    cmap="inferno",
    max_xticks=20,
    figsize=(12, 10),
    title=None,
    step_pct=None,        # percentage of data added for snapshot
    region_label="all",
    color_raw="#4C72B0",
    color_chrom="#55A868",
    color_weights="#E08214",
    color_bins="gray",
):
    """
    Combined visualization:
      1. Chromatogram (raw + z-normalized) and average regression weights
      2. RT-bin frequency vs mean |weights| per RT-bin (snapshot)
      3. Evolution heatmap of RT-bin selection (SOTF)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.interpolate import interp1d

    def resolve_path(npz_path):
        if os.path.isabs(npz_path):
            return npz_path

        # Start from this file’s directory
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Prefer a local 'results' folder if it exists
        local_results = os.path.join(base_dir, "results")
        if os.path.isdir(local_results):
            return os.path.join(local_results, os.path.basename(npz_path))

        # Otherwise, fall back to the main project-level results
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(base_dir)))
        return os.path.join(project_root, "results", os.path.basename(npz_path))

    npz_path = resolve_path(npz_path)

    # --- Load RT-bin evolution data ---
    data = np.load(npz_path)
    if rt_edges is None:
        rt_edges = data["rt_edges"]
    rt_freq_plot = data["rt_freq_cumulative"] if cumulative else data["rt_freq_by_step"]
    n_steps, n_rt_bins = rt_freq_plot.shape

    # --- Normalization ---
    if normalize == "fraction":
        vmin, vmax = 0.0, 1.0
        rt_freq_plot = rt_freq_plot / np.nanmax(rt_freq_plot)
    elif normalize == "counts":
        vmin, vmax = 0, np.nanmax(rt_freq_plot)
    elif normalize == "global":
        vmin, vmax = 0.0, 1.0
        rt_freq_plot = rt_freq_plot / np.nanmax(rt_freq_plot)
    elif normalize == "column":
        rt_freq_plot = rt_freq_plot / (rt_freq_plot.sum(axis=1, keepdims=True) + 1e-12)
        vmin, vmax = 0.0, 1.0
    elif normalize == "row":
        rt_freq_plot = rt_freq_plot / (rt_freq_plot.max(axis=0, keepdims=True) + 1e-12)
        vmin, vmax = 0.0, 1.0
    else:
        raise ValueError("normalize must be one of: 'counts', 'fraction', 'global', 'column', or 'row'")

    # --- Determine snapshot step ---
    step = None
    if step_pct is not None:
        step = int(round((step_pct / 100) * (n_steps - 1)))
        rt_freq_step = rt_freq_plot[step, :]

    # --- Compute averaged regression weights ---
    all_coefs = np.asarray(all_coefs)
    mean_weights = np.mean(all_coefs, axis=tuple(range(all_coefs.ndim - 1)))
    n_features = mean_weights.shape[0]

    # --- Medoid chromatogram (closest to mean) ---
    ref_chrom = np.asarray(ref_chrom)
    mean_chrom = np.mean(ref_chrom, axis=0)
    distances = np.linalg.norm(ref_chrom - mean_chrom, axis=1)
    medoid_chrom = ref_chrom[np.argmin(distances)]

    # --- Z-normalize chromatogram ---
    mu, sigma = np.mean(ref_chrom, axis=0), np.std(ref_chrom, axis=0)
    sigma[sigma == 0] = 1
    medoid_chrom_z = (medoid_chrom - mu) / sigma

    # --- Interpolate if needed ---
    if medoid_chrom_z.shape[0] != n_features:
        x_old = np.linspace(0, 1, medoid_chrom_z.shape[0])
        x_new = np.linspace(0, 1, n_features)
        medoid_chrom_z = interp1d(x_old, medoid_chrom_z)(x_new)
        medoid_chrom = interp1d(x_old, medoid_chrom)(x_new)

    # --- Normalize weights ---
    mean_weights_norm = mean_weights / np.max(np.abs(mean_weights))

    # --- Compute weights averaged per RT-bin ---
    n_rt_bins = len(rt_edges) - 1
    weights_per_bin = np.zeros(n_rt_bins)
    for i in range(n_rt_bins):
        start, end = int(rt_edges[i]), int(rt_edges[i + 1])
        weights_per_bin[i] = np.mean(np.abs(mean_weights_norm[start:end]))
    weights_per_bin /= np.max(weights_per_bin)

    # --- Figure layout: heatmap last ---
    fig, axes = plt.subplots(
        3, 1, figsize=figsize, gridspec_kw={"height_ratios": [1.6, 1.2, 2.4]}
    )
    ax_top, ax_mid, ax_heat = axes

    # === 1️⃣ Top: chromatogram + weights ===
    x = np.arange(n_features)
    ax_top.plot(x, medoid_chrom_z / np.max(np.abs(medoid_chrom_z)),
                color=color_chrom, lw=1.0, alpha=0.9,
                label="Z-normalized chromatogram")
    ax_top.plot(x, mean_weights_norm,
                color=color_weights, lw=1.0, alpha=0.9,
                label="Average regression weights (rescaled)")

    ax_top_right = ax_top.twinx()
    ax_top_right.plot(x, medoid_chrom, color=color_raw, lw=1.0, alpha=0.7,
                      label="Raw chromatogram")
    ax_top.set_ylabel("Scaled response / weights", fontsize=12)
    ax_top_right.set_ylabel("Raw response [a.u.]", fontsize=12, color=color_raw)
    ax_top.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_top.set_title(f"{region_label.capitalize()} — Regression weights and chromatogram")

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_top_right.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    # === 2️⃣ Middle: RT-bin frequency vs weights ===
    if step is not None:
        rt_freq_step_norm = rt_freq_step / np.max(rt_freq_step)
        bin_centers = [(rt_edges[i] + rt_edges[i + 1]) / 2 for i in range(n_rt_bins)]
        widths = np.diff(rt_edges)

        ax_mid.bar(
            bin_centers,
            rt_freq_step_norm,
            width=widths,
            color=color_bins,
            alpha=0.4,
            label="SOTF RT-bin frequency",
            align="center",
        )
        ax_mid.plot(
            bin_centers,
            weights_per_bin,
            "-o",
            color=color_weights,
            lw=1.5,
            label="Mean |weights| per RT-bin (normalized)",
        )

        ax_mid.set_xlabel("Retention time bin")
        ax_mid.set_ylabel("Normalized amplitude")
        ax_mid.grid(alpha=0.3, linestyle="--", linewidth=0.5)
        ax_mid.legend(loc="upper right", fontsize=9)
        ax_mid.set_title(
            f"RT-bin frequency vs regression weights at {step_pct:.0f}% "
            f"({step+1}/{n_steps})"
        )

    # === 3️⃣ Bottom: SOTF RT-bin evolution heatmap ===
    sns.heatmap(
        rt_freq_plot.T,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        cbar_kws={"label": "Selection count" if normalize == "counts" else "Normalized frequency"},
        xticklabels=max_xticks if n_steps > max_xticks else 1,
        yticklabels=[f"{rt_edges[i]}–{rt_edges[i + 1]}" for i in range(n_rt_bins)],
        ax=ax_heat,
    )
    # # Only show every 2nd y-tick label
    # for i, label in enumerate(ax_heat.get_yticklabels()):
    #     if i % 2 != 0:  # hide odd labels
    #         label.set_visible(False)

    ax_heat.tick_params(axis='y', labelsize=5)  # make y-axis (RT bins) smaller
    ax_heat.tick_params(axis='x', labelsize=8)
    ax_heat.set_xlabel("Addition step")
    ax_heat.set_ylabel("Retention time bin")
    ax_heat.set_title(title or "Cumulative evolution of RT-bin selection")

    if step is not None:
        ax_heat.axvline(
            x=step + 0.5,
            color="red",
            lw=1.5,
            linestyle="--",
            label=f"{step_pct:.1f}% ({step+1}/{n_steps})",
        )
        ax_heat.legend(loc="upper right", frameon=False)

    plt.tight_layout()
    plt.show()

# def plot_rt_bin_evolution_heatmap(
#     npz_path,
#     normalize="counts",   # "counts", "global", "column", or "row"
#     cumulative=True,      # usually True for SOTF visualization
#     cmap="inferno",
#     max_xticks=20,
#     figsize=(10, 8),
#     title=None,
#     step_pct=None,        # percentage of data added for snapshot
#
# ):
#     """
#     Plot the evolution of RT-bin selection frequency across SOTF addition steps.
#
#     Parameters
#     ----------
#     npz_path : str
#         Path to the saved .npz file (rt_distribution_allsteps_*.npz)
#     normalize : str, default="counts"
#         - "counts" : absolute selection counts (no normalization)
#         - "global" : divide by global max
#         - "column" : normalize per step
#         - "row"    : normalize per RT bin
#     cumulative : bool, default=True
#         If True, use cumulative frequencies (recommended)
#     cmap : str, default="inferno"
#         Colormap for the heatmap
#     max_xticks : int, default=20
#         Limit number of x-axis tick labels
#     figsize : tuple, default=(10, 8)
#         Figure size for both subplots combined
#     title : str or None
#         Optional custom title
#     step_pct : float or None
#         Percentage (0–100) of data added; if provided, show bar plot at that step
#     """
#
#     # --- Load data ---
#     data = np.load(npz_path)
#     rt_edges = data["rt_edges"]
#     n_folds = int(data.get("n_folds",  max(1, int(data["rt_freq_cumulative"].max()))))
#
#     # rt_freq_by_step = data["rt_freq_cumulative"] if cumulative else data["rt_freq_by_step"]
#     # rt_freq_by_step = data["rt_freq_by_step"]
#
#     if cumulative:
#         rt_freq_plot = data["rt_freq_cumulative"]  # already cumulative from file
#     else:
#         rt_freq_plot = data["rt_freq_by_step"]
#
#     n_steps, n_rt_bins = rt_freq_plot.shape
#
#     if normalize == "fraction":
#         rt_freq_plot = rt_freq_plot / max(1, n_folds)
#         vmin, vmax = 0.0, 1.0
#     elif normalize == "counts":
#         vmin, vmax = 0, n_folds
#     elif normalize == "global":
#         vmin, vmax = 0.0, 1.0
#     elif normalize == "column":
#         rt_freq_plot = rt_freq_plot / (rt_freq_plot.sum(axis=1, keepdims=True) + 1e-12)
#         vmin, vmax = 0.0, 1.0
#     elif normalize == "row":
#         rt_freq_plot = rt_freq_plot / (rt_freq_plot.max(axis=0, keepdims=True) + 1e-12)
#         vmin, vmax = 0.0, 1.0
#     else:
#         raise ValueError("normalize must be one of: 'counts', 'fraction', 'global', 'column', or 'row'")
#
#     # --- Determine snapshot step if requested ---
#     step = None
#     if step_pct is not None:
#         step = int(round((step_pct / 100) * (n_steps - 1)))
#         rt_freq_step = rt_freq_plot[step, :]
#
#     # --- Layout: vertical stacking ---
#     if step_pct is not None:
#         fig, (ax_heat, ax_bar) = plt.subplots(
#             2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
#         )
#     else:
#         fig, ax_heat = plt.subplots(figsize=(figsize[0], figsize[1] * 0.75))
#         ax_bar = None
#
#     # --- Determine color scale (avoid saturation) ---
#     # vmax = np.percentile(rt_freq_plot, 99) if normalize == "counts" else 1.0
#     # vmax = np.max(rt_freq_plot)
#
#     # --- Heatmap ---
#     label_map = {
#         "counts": "Cumulative selection count",
#         "global": "Cumulative frequency (scaled to max)",
#         "column": "Normalized frequency (per step)",
#         "row": "Relative activation per RT bin",
#     }
#
#     # --- Determine color scale based on counts ---
#     if normalize == "counts":
#         # Estimate number of folds from maximum count across all bins
#         n_folds_est = max(1, int(np.nanmax(rt_freq_plot)))
#         vmax = n_folds_est  # e.g., if 5 folds → yellow = selected by all 5
#     else:
#         vmax = 1.0  # for normalized plots (fraction, global, etc.)
#
#     # --- Plot ---
#     sns.heatmap(
#         rt_freq_plot.T,
#         cmap=cmap,
#         vmin=vmin, vmax=vmax,
#         cbar_kws={"label": "Fraction of folds" if normalize == "fraction" else "Count"},
#         xticklabels=max_xticks if n_steps > max_xticks else 1,
#         yticklabels=[f"{rt_edges[i]}–{rt_edges[i + 1]}" for i in range(n_rt_bins)],
#         ax=ax_heat,
#     )
#     ax_heat.set_xlabel("Addition step")
#     ax_heat.set_ylabel("Retention time bin")
#     if title is None:
#         title = "Cumulative evolution of RT-bin selection"
#     ax_heat.set_title(title)
#
#     # --- Mark snapshot column if requested ---
#     if step is not None:
#         ax_heat.axvline(
#             x=step + 0.5,
#             color="red",
#             lw=1.5,
#             linestyle="--",
#             label=f"{step_pct:.1f}% ({step+1}/{n_steps})",
#         )
#         ax_heat.legend(loc="upper right", frameon=False)
#
#     # --- RT-bin bar chart (bottom panel) ---
#     if ax_bar is not None:
#         ax_bar.bar(
#             range(n_rt_bins),
#             rt_freq_step,
#             color="skyblue",
#             edgecolor="black",
#             alpha=0.85,
#         )
#         ax_bar.set_xticks(range(n_rt_bins))
#         ax_bar.set_xticklabels(
#             [f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#             rotation=45,
#             ha="right",
#         )
#         ax_bar.set_xlabel("Retention time bin")
#         ax_bar.set_ylabel("Frequency" if normalize == "counts" else "Relative frequency")
#         ax_bar.set_title(
#             f"RT-bin distribution at {step_pct:.0f}% of data added (step {step+1}/{n_steps})"
#         )
#
#     plt.tight_layout()
#     plt.show()

# def plot_rt_bin_evolution_heatmap(
#     npz_path,
#     normalize=True,
#     cumulative=False,
#     cmap="viridis",
#     max_xticks=20,
#     figsize=(10, 8),
#     title=None,
#     step_pct=None,  # percentage of data added for snapshot
# ):
#     """
#     Plot a heatmap showing the evolution of RT-bin selection frequency
#     across SOTF addition steps, and (optionally) the RT-bin distribution
#     at a chosen step percentage as a bar plot below.
#
#     Parameters
#     ----------
#     npz_path : str
#         Path to the saved .npz file (rt_distribution_allsteps_*.npz)
#     normalize : bool, default=True
#         Normalize per-step frequencies to sum to 1 (relative importance)
#     cumulative : bool, default=False
#         If True, plot cumulative frequencies instead of per-step values
#     cmap : str, default="viridis"
#         Colormap for the heatmap
#     max_xticks : int, default=20
#         Maximum number of x-axis tick labels (to avoid overcrowding)
#     figsize : tuple, default=(10, 8)
#         Figure size for combined plot (height > width)
#     title : str or None
#         Optional title for the heatmap
#     step_pct : float or None
#         Percentage (0–100) of data added; if provided, a bar plot of RT-bin
#         frequencies at that step is shown beneath the heatmap.
#     """
#
#     # --- Load data ---
#     data = np.load(npz_path)
#     rt_edges = data["rt_edges"]
#     rt_freq_by_step = data["rt_freq_by_step"]
#     n_steps, n_rt_bins = rt_freq_by_step.shape
#
#     # --- Compute data for heatmap ---
#     if cumulative:
#         rt_freq_plot = np.cumsum(rt_freq_by_step, axis=0)
#     else:
#         rt_freq_plot = rt_freq_by_step.copy()
#
#     if normalize == "column":
#         # normalize each step to sum to 1 (relative frequencies)
#         rt_freq_plot = rt_freq_plot / (rt_freq_plot.sum(axis=1, keepdims=True) + 1e-12)
#     elif normalize == "global":
#         # normalize by the global maximum (absolute cumulative strength)
#         rt_freq_plot = rt_freq_plot / (rt_freq_plot.max() + 1e-12)
#
#     # --- Determine snapshot step if requested ---
#     step = None
#     if step_pct is not None:
#         step = int(round((step_pct / 100) * (n_steps - 1)))
#         rt_freq_step = rt_freq_plot[step, :]
#         if normalize and cumulative:
#             rt_freq_step = rt_freq_step / (rt_freq_step.max() + 1e-12)
#
#     # --- Layout: vertical stacking ---
#     if step_pct is not None:
#         fig, (ax_heat, ax_bar) = plt.subplots(
#             2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]}
#         )
#     else:
#         fig, ax_heat = plt.subplots(figsize=(figsize[0], figsize[1] * 0.75))
#         ax_bar = None
#
#     # --- Heatmap ---
#     sns.heatmap(
#         rt_freq_plot.T,
#         cmap=cmap,
#         cbar_kws={
#             "label": "Cumulative frequency" if cumulative else "Normalized frequency"
#         },
#         xticklabels=max_xticks if n_steps > max_xticks else 1,
#         yticklabels=[f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#         ax=ax_heat,
#     )
#     ax_heat.set_xlabel("Addition step")
#     ax_heat.set_ylabel("Retention time bin")
#     if title is None:
#         title = (
#             "Cumulative evolution of RT-bin selection"
#             if cumulative else
#             "Evolution of RT-bin selection frequency"
#         )
#     ax_heat.set_title(title)
#
#     # --- Mark snapshot column if requested ---
#     if step is not None:
#         ax_heat.axvline(x=step + 0.5, color="red", lw=1.5, linestyle="--",
#                         label=f"{step_pct:.1f}% ({step+1}/{n_steps})")
#         ax_heat.legend(loc="upper right", frameon=False)
#
#     # --- RT-bin bar chart (bottom panel) ---
#     if ax_bar is not None:
#         ax_bar.bar(
#             range(n_rt_bins),
#             rt_freq_step,
#             color="skyblue",
#             edgecolor="black",
#             alpha=0.85,
#         )
#         ax_bar.set_xticks(range(n_rt_bins))
#         ax_bar.set_xticklabels(
#             [f"{rt_edges[i]}–{rt_edges[i+1]}" for i in range(n_rt_bins)],
#             rotation=45,
#             ha="right",
#         )
#         ax_bar.set_xlabel("Retention time bin")
#         ax_bar.set_ylabel("Frequency" if not normalize else "Relative frequency")
#         ax_bar.set_title(f"RT-bin distribution at {step_pct:.0f}% of data added "
#                          f"(step {step+1}/{n_steps})")
#
#     plt.tight_layout()
#     plt.show()



from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
def compare_regression_weights(
    weights_all,
    weights_burgundy,
    savefig_path=None,
    title_prefix="Ridge regression"
):
    """
    Compare and visualize regression weights between All wines and Burgundy models.

    Parameters
    ----------
    weights_all : str or np.ndarray
        Path to .npy file or array of weights for All wines.
    weights_burgundy : str or np.ndarray
        Path to .npy file or array of weights for Burgundy wines.
    savefig_path : str, optional
        Path to save the output figure (PNG).
    title_prefix : str, optional
        Prefix for figure title.

    Returns
    -------
    r : float
        Pearson correlation coefficient.
    p : float
        p-value of the correlation (not displayed in plots).
    """

    # --- Load arrays if paths are provided ---
    w_all = np.load(weights_all) if isinstance(weights_all, str) else np.asarray(weights_all)
    w_burg = np.load(weights_burgundy) if isinstance(weights_burgundy, str) else np.asarray(weights_burgundy)

    if w_all.shape != w_burg.shape:
        raise ValueError(f"Shape mismatch: {w_all.shape} vs {w_burg.shape}")

    # --- Compute correlation ---
    r, p = pearsonr(w_all, w_burg)

    # --- Create figure ---
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

    # === 1️⃣ Scatter with y=x and fit ===
    ax = axes[0]
    ax.scatter(w_all, w_burg, s=25, alpha=0.7, edgecolor="k", linewidth=0.3)

    min_val = float(min(w_all.min(), w_burg.min()))
    max_val = float(max(w_all.max(), w_burg.max()))
    ax.plot([min_val, max_val], [min_val, max_val],
            color="grey", lw=1.2, ls="--", alpha=0.85, label="y = x")

    model = LinearRegression()
    X = w_all.reshape(-1, 1)
    model.fit(X, w_burg)
    slope = model.coef_[0]
    intercept = model.intercept_
    x_line = np.linspace(min_val, max_val, 200)
    ax.plot(x_line, intercept + slope * x_line,
            color="red", lw=1.2, label=f"Linear fit (r = {r:.2f})")

    ax.set_xlabel("Weights (All wines)")
    ax.set_ylabel("Weights (Burgundy)")
    ax.set_title(f"{title_prefix}: correlation of regression weights")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # === 2️⃣ Weight curves across RT ===
    ax = axes[1]
    ax.plot(w_all, color="orange", lw=1, label="All wines")
    ax.plot(w_burg, color="green", lw=1, label="Burgundy")
    ax.set_ylabel("Weight value")
    ax.set_title("Regression weights across retention-time index")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    # === 3️⃣ Differences ===
    ax = axes[2]
    diff = w_burg - w_all
    ax.plot(diff, color="purple", lw=1)
    ax.set_xlabel("Retention-time index (feature index)")
    ax.set_ylabel("Δ weight (Burgundy − All)")
    ax.set_title("Differences between regression weights")
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if savefig_path:
        os.makedirs(os.path.dirname(savefig_path), exist_ok=True)
        fig.savefig(savefig_path, dpi=300, bbox_inches="tight")

    plt.show()
    return r, p


def plot_avg_weights_vs_rtbin_frequency(
    all_coefs,
    ref_chrom,
    rt_edges,
    rt_freq,
    bin_order_mode=None,
    color_raw="#4C72B0",     # raw chromatogram
    color_chrom="#55A868",   # z-normalized chromatogram
    color_weights="#E08214", # regression weights
    color_bins="gray",       # RT-bin frequency bars
    region_label="all",

):
    """
    Plot average regression weights vs chromatogram, and compare
    regression weights averaged per RT-bin to SOTF RT-bin frequencies.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # --- Average coefficients ---
    all_coefs = np.asarray(all_coefs)
    if all_coefs.ndim == 3:
        mean_weights = np.mean(all_coefs, axis=(0, 1))
    elif all_coefs.ndim == 2:
        mean_weights = np.mean(all_coefs, axis=0)
    else:
        mean_weights = all_coefs
    n_features = mean_weights.shape[0]

    # --- Medoid chromatogram (closest to mean) ---
    ref_chrom = np.asarray(ref_chrom)
    mean_chrom = np.mean(ref_chrom, axis=0)
    distances = np.linalg.norm(ref_chrom - mean_chrom, axis=1)
    medoid_chrom = ref_chrom[np.argmin(distances)]

    # --- Z-normalize chromatogram ---
    mu = np.mean(ref_chrom, axis=0)
    sigma = np.std(ref_chrom, axis=0)
    sigma[sigma == 0] = 1
    medoid_chrom_z = (medoid_chrom - mu) / sigma

    # --- Interpolate if dimensions differ ---
    if medoid_chrom_z.shape[0] != n_features:
        x_old = np.linspace(0, 1, medoid_chrom_z.shape[0])
        x_new = np.linspace(0, 1, n_features)
        medoid_chrom_z = interp1d(x_old, medoid_chrom_z)(x_new)
        medoid_chrom = interp1d(x_old, medoid_chrom)(x_new)

    # --- Normalize for plotting ---
    medoid_chrom_z /= np.max(np.abs(medoid_chrom_z))
    mean_weights_norm = mean_weights / np.max(np.abs(mean_weights))

    # --- Compute weights averaged per RT-bin ---
    rt_edges = np.asarray(rt_edges, dtype=int)
    n_rt_bins = len(rt_edges) - 1
    weights_per_bin = np.zeros(n_rt_bins)
    for i in range(n_rt_bins):
        start, end = rt_edges[i], rt_edges[i + 1]
        weights_per_bin[i] = np.mean(np.abs(mean_weights_norm[start:end]))

    # Normalize both distributions for fair comparison
    weights_per_bin /= np.max(weights_per_bin)
    rt_freq_norm = np.asarray(rt_freq) / np.max(rt_freq)

    # --- Plot ---
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                            gridspec_kw={'height_ratios': [2.5, 1]})
    x = np.arange(n_features)

    # === Top subplot: chromatogram + regression weights ===
    ax_top.plot(x, medoid_chrom_z, color=color_chrom, lw=1.0, alpha=0.9,
                label="Z-normalized chromatogram (closest to mean)")
    ax_top.plot(x, mean_weights_norm, color=color_weights, lw=1.0, alpha=0.9,
                label="Average regression weights (rescaled)")

    ax_top_right = ax_top.twinx()
    ax_top_right.plot(x, medoid_chrom, color=color_raw, lw=1.0, alpha=0.7,
                      label="Raw chromatogram")

    ax_top.set_ylabel("Scaled response / weights", fontsize=13)
    ax_top_right.set_ylabel("Raw response [a.u.]", fontsize=13, color=color_raw)
    ax_top.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_top.set_title(f"{region_label.capitalize()} — Regression weights and chromatogram")

    lines1, labels1 = ax_top.get_legend_handles_labels()
    lines2, labels2 = ax_top_right.get_legend_handles_labels()
    ax_top.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=10)

    # === Bottom subplot: RT-bin frequency vs weights averaged per bin ===
    bin_centers = [(rt_edges[i] + rt_edges[i + 1]) / 2 for i in range(n_rt_bins)]
    widths = np.diff(rt_edges)

    bars = ax_bottom.bar(
        bin_centers,
        rt_freq_norm,
        width=widths,
        color=color_bins,
        alpha=0.4,
        label="SOTF RT-bin frequency",
        align="center",
    )
    ax_bottom.plot(
        bin_centers,
        weights_per_bin,
        "-o",
        color=color_weights,
        lw=1.5,
        label="Mean |weights| per RT-bin (normalized)",
    )

    # # --- Annotate each bar with its most common selection order ---
    # if "bin_order_mode" in locals() or "bin_order_mode" in globals():
    #     for i, bar in enumerate(bars):
    #         order_val = bin_order_mode[i] if i < len(bin_order_mode) else 0
    #         if rt_freq[i] > 0 and order_val > 0:
    #             ax_bottom.text(
    #                 bar.get_x() + bar.get_width() / 2,
    #                 bar.get_height() * 0.85,  # slightly below the top of the bar
    #                 f"{int(order_val)}",
    #                 ha="center",
    #                 va="bottom",
    #                 color="black",
    #                 fontsize=9,
    #                 fontweight="bold",
    #             )

    ax_bottom.set_xlabel("Retention time index", fontsize=13)
    ax_bottom.set_ylabel("Normalized amplitude", fontsize=13)
    ax_bottom.grid(alpha=0.3, linestyle="--", linewidth=0.5)
    ax_bottom.legend(loc="upper right", fontsize=10)
    ax_bottom.set_title("Regression weights averaged per RT-bin vs SOTF bin frequency")

    plt.tight_layout()
    plt.show()

def plot_avg_weights_vs_global_mean(
    all_coefs,
    ref_chrom,
    color_raw="#4C72B0",     # blue for raw chromatogram
    color_chrom="#55A868",   # green for normalized chromatogram
    color_weights="#E08214", # orange for regression weights
    alpha_band=0.25,
    region_label="all",
):
    """
    Plot the average regression weights (across folds/repeats)
    together with the chromatogram closest to the mean,
    for the specified dataset region (e.g., all, burgundy, europe).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d

    # --- Average coefficients ---
    all_coefs = np.asarray(all_coefs)
    if all_coefs.ndim == 3:
        mean_weights = np.mean(all_coefs, axis=(0, 1))
    elif all_coefs.ndim == 2:
        mean_weights = np.mean(all_coefs, axis=0)
    elif all_coefs.ndim == 1:
        mean_weights = all_coefs
    else:
        raise ValueError(f"Unexpected all_coefs shape {all_coefs.shape}")

    n_features = mean_weights.shape[0]

    # --- Compute medoid chromatogram ---
    ref_chrom = np.asarray(ref_chrom)
    if ref_chrom.ndim != 2:
        raise ValueError("ref_chrom must be 2D (samples × retention times).")

    mean_chrom = np.mean(ref_chrom, axis=0)
    distances = np.linalg.norm(ref_chrom - mean_chrom, axis=1)
    medoid_idx = np.argmin(distances)
    medoid_chrom = ref_chrom[medoid_idx]

    # --- Z-normalize across samples ---
    mu = np.mean(ref_chrom, axis=0)
    sigma = np.std(ref_chrom, axis=0)
    sigma[sigma == 0] = 1.0
    medoid_chrom_z = (medoid_chrom - mu) / sigma

    # --- Interpolate if needed ---
    if medoid_chrom_z.shape[0] != n_features:
        x_old = np.linspace(0, 1, medoid_chrom_z.shape[0])
        x_new = np.linspace(0, 1, n_features)
        medoid_chrom_z = interp1d(x_old, medoid_chrom_z, kind="linear")(x_new)
        medoid_chrom = interp1d(x_old, medoid_chrom, kind="linear")(x_new)

    # --- Normalize for plotting ---
    medoid_chrom_z /= np.max(np.abs(medoid_chrom_z))
    mean_weights_norm = mean_weights / np.max(np.abs(mean_weights))

    # --- Plot ---
    fig, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(n_features)

    # Left axis: z-normalized chromatogram + weights
    ax1.plot(x, medoid_chrom_z, color=color_chrom, lw=1.0, alpha=0.9,
             label="Z-normalized chromatogram (closest to mean)")
    ax1.plot(x, mean_weights_norm, color=color_weights, lw=1.0, alpha=0.9,
             label="Average regression weights (rescaled)")
    ax1.set_xlabel("Retention time index", fontsize=16)
    ax1.set_ylabel("Scaled response [a.u.]", color=color_chrom, fontsize=16)
    ax1.tick_params(axis="both", labelsize=14)
    ax1.tick_params(axis='y', labelcolor=color_chrom)
    ax1.grid(alpha=0.3, linestyle="--", linewidth=0.5)

    # Right axis: raw chromatogram
    ax2 = ax1.twinx()
    ax2.plot(x, medoid_chrom, color=color_raw, lw=1.0, alpha=0.8,
             label="Raw chromatogram")
    ax2.set_ylabel("Response [a.u.]", color=color_raw, fontsize=16)
    ax2.tick_params(axis="both", labelsize=14)
    ax2.tick_params(axis='y', labelcolor=color_raw)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    # ax1.legend(
    #     lines_1 + lines_2,
    #     labels_1 + labels_2,
    #     loc="upper left",
    #     bbox_to_anchor=(0.0, 0.999),
    #     frameon=True,
    #     framealpha=0.9,
    #     facecolor="none",
    #     fontsize=15,
    # )

    # # Title
    # ax1.set_title(
    #     f"Average regression weights vs reference chromatogram (closest to mean) — {region_label.capitalize()} dataset",
    #     fontsize=16,
    #     pad=14,
    # )

    plt.tight_layout()
    plt.show()


def plot_fold_weight_heatmap(all_coefs, ref_chrom=None, cmap="inferno", clim=None):
    """
    Visualize per-fold regression weights as a 2D heatmap and optionally overlay
    both:
      - the mean (or sum) of weights across folds (white line)
      - a reference chromatogram (cyan line)
    on the same retention-time axis.

    Parameters
    ----------
    all_coefs : np.ndarray
        Shape (n_folds, n_features). Each row corresponds to one CV model.
        Each column corresponds to a retention-time index (after decimation).

    ref_chrom : np.ndarray or None, optional
        Reference chromatogram to overlay.
        - If 1D: used directly.
        - If 2D: chromatogram closest to the mean is plotted.
        Must have the same number of features as all_coefs.

    cmap : str, optional
        Matplotlib colormap. Default is "inferno".

    clim : tuple or None, optional
        (vmin, vmax) for color scaling. If None, scales automatically.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_folds, n_features = all_coefs.shape

    # --- Handle reference chromatogram ---
    ref_to_plot = None
    if ref_chrom is not None:
        ref_chrom = np.asarray(ref_chrom)
        if ref_chrom.ndim == 2:
            mean_vec = ref_chrom.mean(axis=0)
            distances = np.linalg.norm(ref_chrom - mean_vec, axis=1)
            closest_idx = np.argmin(distances)
            ref_to_plot = ref_chrom[closest_idx]
            print(f"Using chromatogram #{closest_idx} (closest to mean) as reference.")
        elif ref_chrom.ndim == 1:
            ref_to_plot = ref_chrom
        else:
            raise ValueError("ref_chrom must be 1D or 2D array.")

        if ref_to_plot.shape[0] != n_features:
            raise ValueError(
                f"Reference chromatogram length ({ref_to_plot.shape[0]}) "
                f"does not match number of features ({n_features})."
            )

        ref_to_plot = ref_to_plot / np.max(np.abs(ref_to_plot))

    # --- Create figure ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Heatmap ---
    im = ax1.imshow(
        all_coefs,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=[0, n_features, 0, n_folds],
        clim=clim
    )
    plt.colorbar(im, ax=ax1, label="Regression weight")

    # --- Overlay mean across folds ---
    fold_mean = np.mean(all_coefs, axis=0)
    norm_line = (fold_mean - np.min(fold_mean)) / (np.max(fold_mean) - np.min(fold_mean))
    norm_line *= n_folds
    line1, = ax1.plot(
        np.arange(n_features),
        norm_line,
        color="white",
        lw=2,
        alpha=0.9,
        label="Mean across folds"
    )

    ax1.set_xlabel("Retention-time index")
    ax1.set_ylabel("CV fold index")
    ax1.set_title("Per-fold regression weight map")
    ax1.grid(alpha=0.2, linestyle="--", linewidth=0.5)

    # --- Overlay chromatogram (if provided) ---
    legend_handles = [line1]
    if ref_to_plot is not None:
        ax2 = ax1.twinx()
        line2, = ax2.plot(
            np.arange(n_features),
            ref_to_plot,
            color="cyan",
            lw=1.5,
            alpha=0.8,
            label="Reference chromatogram (normalized)"
        )
        ax2.set_ylabel("Normalized intensity", color="cyan")
        ax2.tick_params(axis="y", colors="cyan")
        legend_handles.append(line2)

    # --- Unified legend ---
    ax1.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.75,   # semi-transparent background
        facecolor="white",
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

def plot_true_vs_pred(y_true, y_pred, origins, pred_plot_mode,
                      year_labels, data, feature_type, pred_plot_region,
                      plot=True):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from sklearn.linear_model import Ridge, RidgeClassifier
    from sklearn.model_selection import LeaveOneOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from gcmswine import utils

    # === CHOOSE CLASSIFIER HERE ===
    CLASSIFIER_TYPE = "RGC"  # options: "RGC" or "LDA"
    # ==============================

    origin_style_map = {
        "Burgundy": ("tab:red", "o"),
        "Neuchâtel": ("tab:blue", "s"),
        "Geneva": ("tab:green", "^"),
        "Valais": ("tab:orange", "D"),
        "Alsace": ("tab:purple", "P"),
        "California": ("tab:brown", "X"),
        "Oregon": ("tab:pink", "*"),
        "Unknown": ("gray", "v"),
    }

    acc, r, r2 = None, None, None  # prepare return values

    # === Special split cases ===
    if pred_plot_region in ["burgundy_eu", "burgundy_us"]:
        train_mask = np.isin(origins, ["Burgundy"])
        if pred_plot_region == "burgundy_eu":
            test_mask = np.isin(origins, ["Neuchâtel", "Geneva", "Valais", "Alsace"])
        else:  # burgundy_vs_us
            test_mask = np.isin(origins, ["California", "Oregon"])

        # Safety check
        if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
            print(f"⚠️ Skipping {pred_plot_region}: no samples in train/test split.")
            return None, None

        X_train = utils.compute_features(data[train_mask], feature_type=feature_type)
        X_test = utils.compute_features(data[test_mask], feature_type=feature_type)

        if pred_plot_mode == "classification":
            y_train = np.array(year_labels)[train_mask].astype(int)
            y_test = np.array(year_labels)[test_mask].astype(int)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # === select classifier ===
            if CLASSIFIER_TYPE == "LDA":
                model = LinearDiscriminantAnalysis()
            else:
                model = RidgeClassifier()

            # model = RidgeClassifier()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        else:
            # regression Ridge
            y_train = np.array(year_labels)[train_mask].astype(float)
            y_test = np.array(year_labels)[test_mask].astype(float)

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            model = Ridge(alpha=1.0)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        y_true = y_test
        origins = np.array(origins)[test_mask]
        year_labels = np.array(year_labels)[test_mask]

    # === Styling ===
    unique_origins = sorted(set(origins))
    origin_to_style = {
        org: origin_style_map.get(org, ("black", "o"))
        for org in set(origins)
    }

    # if plot:
    #     plt.figure(figsize=(9, 7))

    # === Classification ===
    if pred_plot_mode == "classification":
        from sklearn.metrics import accuracy_score

        all_coefs = None

        # === CASE 1: Hold-out region (Burgundy→EU/US) ===
        if pred_plot_region in ["burgundy_eu", "burgundy_us"]:
            # Reuse results computed earlier
            # (X_test, y_test, y_pred already exist)
            y_true = y_test

        # === CASE 2: Generic case → run LOO CV ===
        else:
            X_cls = utils.compute_features(data, feature_type=feature_type)
            y_cls = np.array(year_labels).astype(int)

            loo = LeaveOneOut()
            y_true, y_pred = [], []
            for train_idx, test_idx in loo.split(X_cls):
                X_train, X_test = X_cls[train_idx], X_cls[test_idx]
                y_train, y_test = y_cls[train_idx], y_cls[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                if CLASSIFIER_TYPE == "LDA":
                    model = LinearDiscriminantAnalysis()
                else:
                    model = RidgeClassifier()

                model.fit(X_train, y_train)
                y_pred.append(model.predict(X_test)[0])
                y_true.append(y_test[0])

            y_true, y_pred = np.array(y_true), np.array(y_pred)

        # --- Metrics ---
        from sklearn.metrics import accuracy_score

        # Base metrics
        acc = accuracy_score(y_true, y_pred) * 100
        r = np.corrcoef(y_true, y_pred)[0, 1]
        r2 = r2_score(y_true, y_pred)
        residuals = y_pred - y_true
        res_std = np.std(residuals)
        res_mean = np.mean(residuals)

        # Tolerance-based accuracies (e.g., within ±1, ±2 years)
        year_tolerances = [1, 2, 3, 4, 5]
        acc_tolerance = {}
        for tol in year_tolerances:
            acc_tol = np.mean(np.abs(y_pred - y_true) <= tol) * 100
            acc_tolerance[tol] = acc_tol
            print(f"Accuracy within ±{tol} years: {acc_tol:.2f}%")

        print(f"{CLASSIFIER_TYPE}:Residuals (years): mean = {res_mean:.3f}, std = {res_std:.3f}")
        print(f"Overall Accuracy = {acc:.2f}%, R = {r:.3f}, R² = {r2:.3f}")

        # # --- Metrics ---
        # acc = accuracy_score(y_true, y_pred) * 100  # convert to %
        # # acc = accuracy_score(y_true, y_pred)
        # r = np.corrcoef(y_true, y_pred)[0, 1]
        # r2 = r2_score(y_true, y_pred)
        # residuals = y_pred - y_true
        # res_std = np.std(residuals)
        # res_mean = np.mean(residuals)
        # print(f"{CLASSIFIER_TYPE}:Residuals (years): mean = {res_mean:.3f}, std = {res_std:.3f}")

        # === Plotting for classification ===
        if plot:
            plt.figure(figsize=(9, 7))

            for yt, yp, org in zip(y_true, y_pred, origins):
                color, marker = origin_style_map.get(org, ("black", "o"))
                plt.scatter(yt, yp, c=[color], marker=marker,
                            s=70, edgecolor="k", alpha=0.85)

            # === Reference and fit lines ===
            min_val, max_val = y_true.min(), y_true.max()
            plt.plot([min_val, max_val], [min_val, max_val],
                     linestyle="--", color="black", linewidth=1, label="x = y")

            slope, intercept = np.polyfit(y_true, y_pred, 1)
            x_line = np.linspace(min_val, max_val, 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5, label="Fit")

            plt.xlabel("True Year", fontsize=12)
            plt.ylabel("Predicted Year", fontsize=12)
            mode_text = "Hold-out" if pred_plot_region in ["burgundy_eu", "burgundy_us"] else "LOO CV"

            plt.title(
                f"{CLASSIFIER_TYPE} Classification – Predicted vs. True Year ({mode_text}, Samples: {pred_plot_region})\n"
                f"Accuracy = {acc:.1f}%, ±1yr = {acc_tolerance[1]:.1f}%, Pearson R = {r:.3f}, R² = {r2:.3f}, Residual SD = {res_std:.3f} years"
            )

            legend_elements = [
                Line2D([0], [0], marker=origin_style_map[org][1], color='w',
                       markerfacecolor=origin_style_map[org][0], markeredgecolor='k',
                       markersize=8, label=org)
                for org in sorted(set(origins))
            ]
            legend_elements += [
                Line2D([0], [0], color="black", linestyle="--", label="x = y"),
                Line2D([0], [0], color="red", linestyle="-", label="Fit"),
            ]
            plt.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=10)
            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

    # === Regression ===
    else:
        if pred_plot_region not in ["burgundy_eu", "burgundy_us"]:
            # redo LOO only for the filtered set
            X_reg = utils.compute_features(data, feature_type=feature_type)
            y_reg = np.array(year_labels).astype(float)
            loo = LeaveOneOut()
            all_coefs = []
            y_true, y_pred = [], []
            for train_idx, test_idx in loo.split(X_reg):
                X_train, X_test = X_reg[train_idx], X_reg[test_idx]
                y_train, y_test = y_reg[train_idx], y_reg[test_idx]

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                model = Ridge(alpha=1.0)
                model.fit(X_train, y_train)
                y_pred.append(model.predict(X_test)[0])
                y_true.append(y_test[0])
                all_coefs.append(model.coef_)
            y_true, y_pred = np.array(y_true), np.array(y_pred)

            all_coefs = np.vstack(all_coefs)  # shape (n_folds, n_features)
        else:
            all_coefs = None


        # plot_fold_weight_heatmap(all_coefs, ref_chrom=X_reg)

        y_true = np.array(y_true, dtype=float)
        y_pred = np.array(y_pred, dtype=float)

        # === Compute metrics (no normalization) ===
        r2 = r2_score(y_true, y_pred)
        r = np.corrcoef(y_true, y_pred)[0, 1]

        # === Residual analysis in year units ===
        residuals = y_pred - y_true
        res_mean = np.mean(residuals)
        res_std = np.std(residuals)
        rmse = np.sqrt(np.mean(residuals ** 2))

        # === Classification-style metrics for regression ===
        y_true_rounded = np.round(y_true)
        y_pred_rounded = np.round(y_pred)

        acc = np.mean(y_true_rounded == y_pred_rounded) * 100
        acc_tolerance = {}
        year_tolerances = [1, 2, 3, 4, 5]
        for tol in year_tolerances:
            acc_tol = np.mean(np.abs(y_true_rounded - y_pred_rounded) <= tol) * 100
            acc_tolerance[tol] = acc_tol

        print(f"Residuals (years): mean = {res_mean:.3f}, std = {res_std:.3f}, RMSE = {rmse:.3f}")
        print(f"Accuracy (rounded): {acc:.2f}%")
        for tol, val in acc_tolerance.items():
            print(f"Accuracy within ±{tol} years: {val:.2f}%")

        # # === Compute metrics (no normalization) ===
        # r2 = r2_score(y_true, y_pred)
        # r = np.corrcoef(y_true, y_pred)[0, 1]
        #
        # # === Residual analysis in year units ===
        # residuals = y_pred - y_true
        # res_mean = np.mean(residuals)
        # res_std = np.std(residuals)
        # rmse = np.sqrt(np.mean(residuals ** 2))
        # print(f"Residuals (years): mean = {res_mean:.3f}, std = {res_std:.3f}, RMSE = {rmse:.3f}")

        if plot:
            # === Scatter points ===
            for yt, yp, org in zip(y_true, y_pred, origins):
                color, marker = origin_style_map.get(org, ("black", "o"))
                plt.scatter(yt, yp, c=[color], marker=marker,
                            s=70, edgecolor="k", alpha=0.85)

            # === Identity line ===
            min_val, max_val = y_true.min(), y_true.max()
            plt.plot([min_val, max_val], [min_val, max_val],
                     linestyle="--", color="black", linewidth=1, label="x = y")

            # === Linear fit (year units) ===
            slope, intercept = np.polyfit(y_true, y_pred, 1)
            x_line = np.linspace(min_val, max_val, 100)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5, label="Fit")

            plt.xlabel("True Year", fontsize=12)
            plt.ylabel("Predicted Year", fontsize=12)

            # === Accuracy text ===
            acc_text = ""
            if "mean_acc" in locals() and "std_acc" in locals():
                acc_text = f", Acc = {mean_acc:.3f} ± {std_acc:.3f}"
            elif "acc" in locals() and acc is not None:
                acc_text = f", Acc = {acc:.3f}"


            # === Compose title ===
            mode_text = "Hold-out" if pred_plot_region in ["burgundy_eu", "burgundy_us"] else "LOO CV"
            plt.title(
                f"Predicted vs. True Year (Regression, {mode_text}, Samples: {pred_plot_region})\n"
                f"R = {r:.3f}, R² = {r2:.3f}, Acc = {acc:.1f}%, ±1yr = {acc_tolerance[1]:.1f}%, Residual SD = {res_std:.3f} years"
            )

            # === Legend ===
            unique_origins = sorted(set(origins))
            legend_elements = [
                Line2D([0], [0], marker=origin_style_map[org][1], color='w',
                       markerfacecolor=origin_style_map[org][0], markeredgecolor='k',
                       markersize=8, label=org)
                for org in unique_origins
            ]
            legend_elements += [
                Line2D([0], [0], color="black", linestyle="--", label="x = y"),
                Line2D([0], [0], color="red", linestyle="-", label="Fit"),
            ]
            plt.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=10)

            plt.grid(True, linestyle="--", alpha=0.6)
            plt.tight_layout()
            plt.show()

            # plot_fold_weight_heatmap(all_coefs, ref_chrom=X_reg)

    return acc, acc_tolerance, r, r2, res_std, all_coefs
    # return acc, r, r2, res_std

    # # === Regression ===
    # else:
    #     if pred_plot_region not in ["burgundy_eu", "burgundy_us"]:
    #         # redo LOO only for the filtered set
    #         X_reg = utils.compute_features(data, feature_type="tic")
    #         y_reg = np.array(year_labels).astype(float)
    #         loo = LeaveOneOut()
    #         y_true, y_pred = [], []
    #         for train_idx, test_idx in loo.split(X_reg):
    #             X_train, X_test = X_reg[train_idx], X_reg[test_idx]
    #             y_train, y_test = y_reg[train_idx], y_reg[test_idx]
    #             scaler = StandardScaler()
    #             X_train = scaler.fit_transform(X_train)
    #             X_test = scaler.transform(X_test)
    #             model = Ridge(alpha=1.0)
    #             model.fit(X_train, y_train)
    #             y_pred.append(model.predict(X_test)[0])
    #             y_true.append(y_test[0])
    #         y_true, y_pred = np.array(y_true), np.array(y_pred)
    #
    #     y_true = np.array(y_true, dtype=float)
    #     y_pred = np.array(y_pred, dtype=float)
    #
    #     # === NORMALISATION ADDED ===
    #     mean_y, std_y = y_true.mean(), y_true.std()
    #     y_true_norm = (y_true - mean_y) / std_y
    #     y_pred_norm = (y_pred - mean_y) / std_y
    #
    #     r2 = r2_score(y_true_norm, y_pred_norm)
    #     r = np.corrcoef(y_true_norm, y_pred_norm)[0, 1]
    #
    #     # === Residual analysis ===
    #     residuals = y_pred_norm - y_true_norm
    #     res_mean = np.mean(residuals)
    #     res_std = np.std(residuals)
    #     rmse = np.sqrt(np.mean(residuals ** 2))
    #
    #     print(f"Residuals: mean = {res_mean:.3f}, std = {res_std:.3f}, RMSE = {rmse:.3f}")
    #
    #     if plot:
    #         for yt, yp, org in zip(y_true, y_pred, origins):
    #             color, marker = origin_to_style[org]
    #             plt.scatter(yt, yp, c=[color], marker=marker,
    #                         s=70, edgecolor="k", alpha=0.85)
    #
    #         min_val, max_val = y_true.min(), y_true.max()
    #         plt.plot([min_val, max_val], [min_val, max_val],
    #                  linestyle="--", color="black", linewidth=1, label="x=y")
    #
    #         slope, intercept = np.polyfit(y_true_norm, y_pred_norm, 1)
    #         x_line = np.linspace(min_val, max_val, 100)
    #         x_line_norm = (x_line - mean_y) / std_y
    #         y_line_norm = slope * x_line_norm + intercept
    #         y_line = y_line_norm * std_y + mean_y
    #         plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5,
    #                  label=f"Fit (norm): y={slope:.2f}x+{intercept:.2f}")
    #
    #         plt.xlabel("True Year", fontsize=12)
    #         plt.ylabel("Predicted Year", fontsize=12)
    #
    #         if pred_plot_region in ["burgundy_eu", "burgundy_us"]:
    #             plt.title(f"Predicted vs. True Year (Regression, Hold-out, Samples: { pred_plot_region})\n"
    #                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
    #         else:
    #             plt.title(f"Predicted vs. True Year (Regression, LOO CV, Samples: {pred_plot_region})\n"
    #                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")





# def plot_true_vs_pred(y_true, y_pred, origins, pred_plot_mode,
#                       year_labels, data, feature_type, pred_plot_region):
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#     import numpy as np
#     from sklearn.linear_model import Ridge
#     from sklearn.model_selection import LeaveOneOut
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.metrics import r2_score
#     from gcmswine import utils
#
#     origin_style_map = {
#         "Burgundy": ("tab:red", "o"),
#         "Neuchâtel": ("tab:blue", "s"),
#         "Geneva": ("tab:green", "^"),
#         "Valais": ("tab:orange", "D"),
#         "Alsace": ("tab:purple", "P"),
#         "California": ("tab:brown", "X"),
#         "Oregon": ("tab:pink", "*"),
#         "Unknown": ("gray", "v"),
#     }
#
#     # === Special split cases ===
#     if pred_plot_region in ["burgundy_vs_eu", "burgundy_vs_us"]:
#         train_mask = np.isin(origins, ["Burgundy"])
#         if pred_plot_region == "burgundy_vs_eu":
#             test_mask = np.isin(origins, ["Neuchâtel", "Geneva", "Valais", "Alsace"])
#         else:  # burgundy_vs_us
#             test_mask = np.isin(origins, ["California", "Oregon"])
#
#         # Safety check
#         if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
#             print(f"⚠️ Skipping {pred_plot_region}: no samples in train/test split.")
#             return
#
#         X_train = utils.compute_features(data[train_mask], feature_type="tic")
#         X_test = utils.compute_features(data[test_mask], feature_type="tic")
#
#         if pred_plot_mode == "classification":
#             y_train = np.array(year_labels)[train_mask].astype(int)
#             y_test = np.array(year_labels)[test_mask].astype(int)
#
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = RidgeClassifier()
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#         else:
#             # regression Ridge
#             y_train = np.array(year_labels)[train_mask].astype(float)
#             y_test = np.array(year_labels)[test_mask].astype(float)
#
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = Ridge(alpha=1.0)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#
#         y_true = y_test
#         origins = np.array(origins)[test_mask]
#         year_labels = np.array(year_labels)[test_mask]
#
#     # === Styling ===
#     unique_origins = sorted(set(origins))
#     origin_to_style = {
#         org: origin_style_map.get(org, ("black", "o"))  # fallback if new origin
#         for org in set(origins)
#     }
#
#     plt.figure(figsize=(9, 7))
#
#     # === Classification ===
#     if pred_plot_mode == "classification":
#         try:
#             y_pred =  np.array(y_pred)
#             y_true_int = y_true.astype(int)
#             y_pred_int = y_pred.astype(int)
#             r = np.corrcoef(y_true_int, y_pred_int)[0, 1]
#             numeric = True
#         except ValueError:
#             y_true_int, y_pred_int = y_true, y_pred
#             r = np.nan
#             numeric = False
#
#         for yt, yp, org in zip(y_true_int, y_pred_int, origins):
#             color, marker = origin_to_style[org]
#             plt.scatter(yt, yp, c=[color], marker=marker,
#                         s=70, edgecolor="k", alpha=0.85)
#
#         min_val, max_val = min(y_true_int.min(), y_pred_int.min()), max(y_true_int.max(), y_pred_int.max())
#         plt.plot([min_val, max_val], [min_val, max_val],
#                  linestyle="--", color="black", linewidth=1, label="x=y")
#
#         # regression line (classification scatter)
#         slope, intercept = np.polyfit(y_true_int, y_pred_int, 1)
#         x_line = np.linspace(min_val, max_val, 100)
#         y_line = slope * x_line + intercept
#         plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5,
#                  label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
#
#         plt.xlabel("True Year Class")
#         plt.ylabel("Predicted Year Class")
#         plt.title(f"Predicted vs. True Year (Classification)\nCorrelation R = {r:.3f}")
#
#         if numeric:
#             unique_years = sorted(set(y_true_int) | set(y_pred_int))
#             plt.xticks(unique_years, rotation=45)
#             plt.yticks(unique_years)
#
#     # === Regression ===
#     else:
#         if pred_plot_region not in ["burgundy_vs_eu", "burgundy_vs_us"]:
#             # redo LOO only for the filtered set
#             X_reg = utils.compute_features(data, feature_type="tic")
#             y_reg = np.array(year_labels).astype(float)
#             loo = LeaveOneOut()
#             y_true, y_pred = [], []
#             for train_idx, test_idx in loo.split(X_reg):
#                 X_train, X_test = X_reg[train_idx], X_reg[test_idx]
#                 y_train, y_test = y_reg[train_idx], y_reg[test_idx]
#                 scaler = StandardScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)
#                 model = Ridge(alpha=1.0)
#                 model.fit(X_train, y_train)
#                 y_pred.append(model.predict(X_test)[0])
#                 y_true.append(y_test[0])
#             y_true, y_pred = np.array(y_true), np.array(y_pred)
#
#         y_true = np.array(y_true, dtype=float)
#         y_pred = np.array(y_pred, dtype=float)
#
#         # === NORMALISATION ADDED ===
#         mean_y, std_y = y_true.mean(), y_true.std()
#         y_true_norm = (y_true - mean_y) / std_y
#         y_pred_norm = (y_pred - mean_y) / std_y
#
#         # metrics on normalised scale
#         r2 = r2_score(y_true_norm, y_pred_norm)
#         r = np.corrcoef(y_true_norm, y_pred_norm)[0, 1]
#
#         # scatter in original scale
#         for yt, yp, org in zip(y_true, y_pred, origins):
#             color, marker = origin_to_style[org]
#             plt.scatter(yt, yp, c=[color], marker=marker,
#                         s=70, edgecolor="k", alpha=0.85)
#
#         min_val, max_val = y_true.min(), y_true.max()
#         plt.plot([min_val, max_val], [min_val, max_val],
#                  linestyle="--", color="black", linewidth=1, label="x=y")
#
#         # regression line: fit in normalised, plot in original
#         slope, intercept = np.polyfit(y_true_norm, y_pred_norm, 1)
#         x_line = np.linspace(min_val, max_val, 100)
#         x_line_norm = (x_line - mean_y) / std_y
#         y_line_norm = slope * x_line_norm + intercept
#         y_line = y_line_norm * std_y + mean_y
#         plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5,
#                  label=f"Fit (norm): y={slope:.2f}x+{intercept:.2f}")
#
#         plt.xlabel("True Year")
#         plt.ylabel("Predicted Year")
#
#     if pred_plot_mode == "classification":
#         plt.title(f"Predicted vs. True Year (Classification, {pred_plot_region})\n"
#                   f"Correlation R = {r:.3f}")
#     else:
#         if pred_plot_region in ["burgundy_vs_eu", "burgundy_vs_us"]:
#             plt.title(f"Predicted vs. True Year (Regression, Hold-out, {pred_plot_region})\n"
#                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
#         else:
#             plt.title(f"Predicted vs. True Year (Regression, LOO CV, {pred_plot_region})\n"
#                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
#     # === Legend ===
#     legend_elements = [
#         Line2D([0], [0], marker=origin_to_style[org][1], color='w',
#                markerfacecolor=origin_to_style[org][0], markeredgecolor="k",
#                markersize=8, label=org)
#         for org in unique_origins
#     ]
#     plt.legend(handles=legend_elements + [
#         Line2D([0], [0], color="black", linestyle="--", label="x=y"),
#         Line2D([0], [0], color="red", linestyle="-", label="Regression fit")
#     ], bbox_to_anchor=(1.05, 1),
#                loc="upper left", title="Origin")
#
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()

# def plot_true_vs_pred(y_true, y_pred, origins, pred_plot_mode,
#                       year_labels, data, feature_type, pred_plot_region):
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#     import numpy as np
#     from sklearn.linear_model import Ridge
#     from sklearn.model_selection import LeaveOneOut
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.metrics import r2_score
#     from gcmswine import utils
#
#     origin_style_map = {
#         "Burgundy": ("tab:red", "o"),
#         "Neuchâtel": ("tab:blue", "s"),
#         "Geneva": ("tab:green", "^"),
#         "Valais": ("tab:orange", "D"),
#         "Alsace": ("tab:purple", "P"),
#         "California": ("tab:brown", "X"),
#         "Oregon": ("tab:pink", "*"),
#         "Unknown": ("gray", "v"),
#     }
#
#     # === Special split cases ===
#     if pred_plot_region in ["burgundy_vs_eu", "burgundy_vs_us"]:
#         train_mask = np.isin(origins, ["Burgundy"])
#         if pred_plot_region == "burgundy_vs_eu":
#             test_mask = np.isin(origins, ["Neuchâtel", "Geneva", "Valais", "Alsace"])
#         else:  # burgundy_vs_us
#             test_mask = np.isin(origins, ["California", "Oregon"])
#
#         # Safety check
#         if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
#             print(f"⚠️ Skipping {pred_plot_region}: no samples in train/test split.")
#             return
#
#         X_train = utils.compute_features(data[train_mask], feature_type="tic")
#         X_test = utils.compute_features(data[test_mask], feature_type="tic")
#
#         if pred_plot_mode == "classification":
#             y_train = np.array(year_labels)[train_mask].astype(int)
#             y_test = np.array(year_labels)[test_mask].astype(int)
#
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = RidgeClassifier()
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#         else:
#             # regression Ridge
#             y_train = np.array(year_labels)[train_mask].astype(float)
#             y_test = np.array(year_labels)[test_mask].astype(float)
#
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = Ridge(alpha=1.0)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#
#         y_true = y_test
#         origins = np.array(origins)[test_mask]
#         year_labels = np.array(year_labels)[test_mask]
#
#     # === Styling ===
#     unique_origins = sorted(set(origins))
#     palette = plt.cm.tab10.colors
#     markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
#     origin_to_style = {
#         org: origin_style_map.get(org, ("black", "o"))  # fallback if new origin
#         for org in set(origins)
#     }
#
#     plt.figure(figsize=(9, 7))
#
#     # === Classification ===
#     if pred_plot_mode == "classification":
#         try:
#             y_pred =  np.array(y_pred)
#             y_true_int = y_true.astype(int)
#             y_pred_int = y_pred.astype(int)
#             r = np.corrcoef(y_true_int, y_pred_int)[0, 1]
#             numeric = True
#         except ValueError:
#             y_true_int, y_pred_int = y_true, y_pred
#             r = np.nan
#             numeric = False
#
#         for yt, yp, org in zip(y_true_int, y_pred_int, origins):
#             color, marker = origin_to_style[org]
#             plt.scatter(yt, yp, c=[color], marker=marker,
#                         s=70, edgecolor="k", alpha=0.85)
#
#         min_val, max_val = min(y_true_int.min(), y_pred_int.min()), max(y_true_int.max(), y_pred_int.max())
#         plt.plot([min_val, max_val], [min_val, max_val],
#                  linestyle="--", color="black", linewidth=1, label="x=y")
#
#         # === ADDED: regression line for classification scatter ===
#         slope, intercept = np.polyfit(y_true_int, y_pred_int, 1)
#         x_line = np.linspace(min_val, max_val, 100)
#         y_line = slope * x_line + intercept
#         plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5,
#                  label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
#
#         plt.xlabel("True Year Class")
#         plt.ylabel("Predicted Year Class")
#         plt.title(f"Predicted vs. True Year (Classification)\nCorrelation R = {r:.3f}")
#
#         if numeric:
#             unique_years = sorted(set(y_true_int) | set(y_pred_int))
#             plt.xticks(unique_years, rotation=45)
#             plt.yticks(unique_years)
#
#     # === Regression ===
#     else:
#         if pred_plot_region not in ["burgundy_vs_eu", "burgundy_vs_us"]:
#             # redo LOO only for the filtered set
#             X_reg = utils.compute_features(data, feature_type="tic")
#             y_reg = np.array(year_labels).astype(float)
#             loo = LeaveOneOut()
#             y_true, y_pred = [], []
#             for train_idx, test_idx in loo.split(X_reg):
#                 X_train, X_test = X_reg[train_idx], X_reg[test_idx]
#                 y_train, y_test = y_reg[train_idx], y_reg[test_idx]
#                 scaler = StandardScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)
#                 model = Ridge(alpha=1.0)
#                 model.fit(X_train, y_train)
#                 y_pred.append(model.predict(X_test)[0])
#                 y_true.append(y_test[0])
#             y_true, y_pred = np.array(y_true), np.array(y_pred)
#
#         y_true = np.array(y_true, dtype=float)
#         y_pred = np.array(y_pred, dtype=float)
#
#         r2 = r2_score(y_true, y_pred)
#         r = np.corrcoef(y_true, y_pred)[0, 1]
#
#         for yt, yp, org in zip(y_true, y_pred, origins):
#             color, marker = origin_to_style[org]
#             plt.scatter(yt, yp, c=[color], marker=marker,
#                         s=70, edgecolor="k", alpha=0.85)
#
#         min_val, max_val = y_true.min(), y_true.max()
#         plt.plot([min_val, max_val], [min_val, max_val],
#                  linestyle="--", color="black", linewidth=1, label="x=y")
#
#         # === ADDED: regression line for regression scatter ===
#         slope, intercept = np.polyfit(y_true, y_pred, 1)
#         x_line = np.linspace(min_val, max_val, 100)
#         y_line = slope * x_line + intercept
#         plt.plot(x_line, y_line, color="red", linestyle="-", linewidth=1.5,
#                  label=f"Fit: y={slope:.2f}x+{intercept:.2f}")
#
#         plt.xlabel("True Year")
#         plt.ylabel("Predicted Year")
#
#     if pred_plot_mode == "classification":
#         plt.title(f"Predicted vs. True Year (Classification, {pred_plot_region})\n"
#                   f"Correlation R = {r:.3f}")
#     else:
#         if pred_plot_region in ["burgundy_vs_eu", "burgundy_vs_us"]:
#             plt.title(f"Predicted vs. True Year (Regression, Hold-out, {pred_plot_region})\n"
#                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
#         else:
#             plt.title(f"Predicted vs. True Year (Regression, LOO CV, {pred_plot_region})\n"
#                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
#     # === Legend ===
#     legend_elements = [
#         Line2D([0], [0], marker=origin_to_style[org][1], color='w',
#                markerfacecolor=origin_to_style[org][0], markeredgecolor="k",
#                markersize=8, label=org)
#         for org in unique_origins
#     ]
#     plt.legend(handles=legend_elements + [
#         Line2D([0], [0], color="black", linestyle="--", label="x=y"),
#         Line2D([0], [0], color="red", linestyle="-", label="Regression fit")
#     ], bbox_to_anchor=(1.05, 1),
#                loc="upper left", title="Origin")
#
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()


# def plot_true_vs_pred(y_true, y_pred, origins, pred_plot_mode,
#                       year_labels, data, feature_type, pred_plot_region):
#     import matplotlib.pyplot as plt
#     from matplotlib.lines import Line2D
#     import numpy as np
#     from sklearn.linear_model import Ridge
#     from sklearn.model_selection import LeaveOneOut
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.metrics import r2_score
#     from gcmswine import utils
#
#     origin_style_map = {
#         "Burgundy": ("tab:red", "o"),
#         "Neuchâtel": ("tab:blue", "s"),
#         "Geneva": ("tab:green", "^"),
#         "Valais": ("tab:orange", "D"),
#         "Alsace": ("tab:purple", "P"),
#         "California": ("tab:brown", "X"),
#         "Oregon": ("tab:pink", "*"),
#         "Unknown": ("gray", "v"),
#     }
#
#     # === Special split cases ===
#     if pred_plot_region in ["burgundy_vs_eu", "burgundy_vs_us"]:
#         train_mask = np.isin(origins, ["Burgundy"])
#         if pred_plot_region == "burgundy_vs_eu":
#             test_mask = np.isin(origins, ["Neuchâtel", "Geneva", "Valais", "Alsace"])
#         else:  # burgundy_vs_us
#             test_mask = np.isin(origins, ["California", "Oregon"])
#
#         # Safety check
#         if np.sum(train_mask) == 0 or np.sum(test_mask) == 0:
#             print(f"⚠️ Skipping {pred_plot_region}: no samples in train/test split.")
#             return
#
#         X_train = utils.compute_features(data[train_mask], feature_type="tic")
#         X_test = utils.compute_features(data[test_mask], feature_type="tic")
#
#         if pred_plot_mode == "classification":
#             y_train = np.array(year_labels)[train_mask].astype(int)
#             y_test = np.array(year_labels)[test_mask].astype(int)
#
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = RidgeClassifier()
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#         else:
#             # regression Ridge
#             y_train = np.array(year_labels)[train_mask].astype(float)
#             y_test = np.array(year_labels)[test_mask].astype(float)
#
#             scaler = StandardScaler()
#             X_train = scaler.fit_transform(X_train)
#             X_test = scaler.transform(X_test)
#
#             model = Ridge(alpha=1.0)
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
#
#         y_true = y_test
#         origins = np.array(origins)[test_mask]
#         year_labels = np.array(year_labels)[test_mask]
#
#     # === Styling ===
#     unique_origins = sorted(set(origins))
#     palette = plt.cm.tab10.colors
#     markers = ['o', 's', '^', 'D', 'P', 'X', '*', 'v']
#     origin_to_style = {
#         org: origin_style_map.get(org, ("black", "o"))  # fallback if new origin
#         for org in set(origins)
#     }
#
#     plt.figure(figsize=(9, 7))
#
#     # === Classification ===
#     if pred_plot_mode == "classification":
#         try:
#             y_pred =  np.array(y_pred)
#             y_true_int = y_true.astype(int)
#             y_pred_int = y_pred.astype(int)
#             r = np.corrcoef(y_true_int, y_pred_int)[0, 1]
#             numeric = True
#         except ValueError:
#             y_true_int, y_pred_int = y_true, y_pred
#             r = np.nan
#             numeric = False
#
#         for yt, yp, org in zip(y_true_int, y_pred_int, origins):
#             color, marker = origin_to_style[org]
#             plt.scatter(yt, yp, c=[color], marker=marker,
#                         s=70, edgecolor="k", alpha=0.85)
#
#         min_val, max_val = min(y_true_int.min(), y_pred_int.min()), max(y_true_int.max(), y_pred_int.max())
#         plt.plot([min_val, max_val], [min_val, max_val],
#                  linestyle="--", color="black", linewidth=1)
#
#         plt.xlabel("True Year Class")
#         plt.ylabel("Predicted Year Class")
#         plt.title(f"Predicted vs. True Year (Classification)\nCorrelation R = {r:.3f}")
#
#         if numeric:
#             unique_years = sorted(set(y_true_int) | set(y_pred_int))
#             plt.xticks(unique_years, rotation=45)
#             plt.yticks(unique_years)
#
#     # === Regression ===
#     else:
#         if pred_plot_region not in ["burgundy_vs_eu", "burgundy_vs_us"]:
#             # redo LOO only for the filtered set
#             X_reg = utils.compute_features(data, feature_type="tic")
#             y_reg = np.array(year_labels).astype(float)
#             loo = LeaveOneOut()
#             y_true, y_pred = [], []
#             for train_idx, test_idx in loo.split(X_reg):
#                 X_train, X_test = X_reg[train_idx], X_reg[test_idx]
#                 y_train, y_test = y_reg[train_idx], y_reg[test_idx]
#                 scaler = StandardScaler()
#                 X_train = scaler.fit_transform(X_train)
#                 X_test = scaler.transform(X_test)
#                 model = Ridge(alpha=1.0)
#                 model.fit(X_train, y_train)
#                 y_pred.append(model.predict(X_test)[0])
#                 y_true.append(y_test[0])
#             y_true, y_pred = np.array(y_true), np.array(y_pred)
#
#         y_true = np.array(y_true, dtype=float)
#         y_pred = np.array(y_pred, dtype=float)
#
#         r2 = r2_score(y_true, y_pred)
#         r = np.corrcoef(y_true, y_pred)[0, 1]
#
#         for yt, yp, org in zip(y_true, y_pred, origins):
#             color, marker = origin_to_style[org]
#             plt.scatter(yt, yp, c=[color], marker=marker,
#                         s=70, edgecolor="k", alpha=0.85)
#
#         min_val, max_val = y_true.min(), y_true.max()
#         plt.plot([min_val, max_val], [min_val, max_val],
#                  linestyle="--", color="black", linewidth=1)
#
#         plt.xlabel("True Year")
#         plt.ylabel("Predicted Year")
#
#     if pred_plot_mode == "classification":
#         plt.title(f"Predicted vs. True Year (Classification, {pred_plot_region})\n"
#                   f"Correlation R = {r:.3f}")
#     else:
#         if pred_plot_region in ["burgundy_vs_eu", "burgundy_vs_us"]:
#             plt.title(f"Predicted vs. True Year (Regression, Hold-out, {pred_plot_region})\n"
#                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
#         else:
#             plt.title(f"Predicted vs. True Year (Regression, LOO CV, {pred_plot_region})\n"
#                       f"Pearson R = {r:.3f}, R² = {r2:.3f}")
#     # === Legend ===
#     legend_elements = [
#         Line2D([0], [0], marker=origin_to_style[org][1], color='w',
#                markerfacecolor=origin_to_style[org][0], markeredgecolor="k",
#                markersize=8, label=org)
#         for org in unique_origins
#     ]
#     plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1),
#                loc="upper left", title="Origin")
#
#     plt.grid(True, linestyle="--", alpha=0.6)
#     plt.tight_layout()
#     plt.show()

import pandas as pd
import matplotlib.pyplot as plt

def plot_accuracy_vs_channels(csv_path_sum, csv_path_concat,
                                   label_sum="Summed channels",
                                   label_concat="Concatenated channels",
                                   title="Accuracy vs Number of Channels"):
    """
    Compare accuracy vs number of channels between two methods (sum vs concat).

    Parameters
    ----------
    csv_path_sum : str
        Path to CSV file for the 'summed channels' method.
    csv_path_concat : str
        Path to CSV file for the 'concatenated channels' method.
    label_sum : str, optional
        Label for the summed-channels curve.
    label_concat : str, optional
        Label for the concatenated-channels curve.
    title : str, optional
        Title for the plot.
    """

    def load_data(csv_path):
        df = pd.read_csv(csv_path)
        ch_col = next((c for c in df.columns if "channel" in c.lower() or c.lower().startswith("k")), None)
        acc_col = next((c for c in df.columns if "acc" in c.lower()), None)
        std_col = next((c for c in df.columns if "std" in c.lower()), None)
        if ch_col is None or acc_col is None:
            raise ValueError(f"Couldn't detect channel or accuracy columns in {csv_path}. Columns are: {list(df.columns)}")
        return df[ch_col], df[acc_col], (df[std_col] if std_col is not None else None)

    # Load both datasets
    x1, y1, y1_std = load_data(csv_path_sum)
    x2, y2, y2_std = load_data(csv_path_concat)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x1, y1, '-o', label=label_sum, color='tab:blue')
    plt.plot(x2, y2, '-s', label=label_concat, color='tab:orange')

    # # Optional shading for std
    # if y1_std is not None:
    #     plt.fill_between(x1, y1 - y1_std, y1 + y1_std, color='tab:blue', alpha=0.15)
    # if y2_std is not None:
    #     plt.fill_between(x2, y2 - y2_std, y2 + y2_std, color='tab:orange', alpha=0.15)

    plt.xlabel("Number of ranked channels added", fontsize=13)
    plt.ylabel("Accuracy (LOO)", fontsize=13)
    plt.title(title, fontsize=15)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print summaries
    best_idx1 = y1.idxmax()
    best_idx2 = y2.idxmax()
    print(f"[{label_sum}] Best accuracy: {y1[best_idx1]:.3f}"
          f"{f' ±{y1_std[best_idx1]:.3f}' if y1_std is not None else ''} "
          f"at {x1[best_idx1]} channels")
    print(f"[{label_concat}] Best accuracy: {y2[best_idx2]:.3f}"
          f"{f' ±{y2_std[best_idx2]:.3f}' if y2_std is not None else ''} "
          f"at {x2[best_idx2]} channels")


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
        print("SCORES shape:", None if scores is None else scores.shape)
        print("ALL_LABELS len:", None if all_labels is None else len(all_labels))
    elif projection_source in {"tic", "tis", "tic_tis"}:
        data_for_umap = utils.compute_features(data, feature_type=projection_source)
        data_for_umap = normalize(data_for_umap)
        projection_labels = labels
        print("FEATURE shape:", data_for_umap.shape, "LABELS:", len(labels))
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

    # test_samples_names = None if not show_sample_names else None  # keep behavior: disable unless explicitly names
    test_samples_names = raw_sample_labels if show_sample_names else None

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
            # color_by_state=False,
            # color_by_burgns=False,
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
# Region/Cube Analysis
# -----------------------------
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

def rt_windows_to_indices(rt_axis, compounds):
    """
    Convert compound RT windows [min] into array index windows.

    Parameters
    ----------
    rt_axis : np.ndarray
        Retention time axis (minutes), length T.
    compounds : list[dict]
        Each dict must have 'name', 'mz', 'rt_start', 'rt_stop'.

    Returns
    -------
    list[dict]
        Same as compounds but with 'start_idx' and 'stop_idx' added.
    """
    updated = []
    for cmpd in compounds:
        start_idx = np.searchsorted(rt_axis, cmpd["rt_start"])
        stop_idx = np.searchsorted(rt_axis, cmpd["rt_stop"])
        cmpd_copy = cmpd.copy()
        cmpd_copy["start_idx"] = start_idx
        cmpd_copy["stop_idx"] = stop_idx
        updated.append(cmpd_copy)
    return updated

def mask_compounds(data_dict, rt_axis, mz_values, compounds, drop_list):
    """
    Zero out specific compounds (by name) in chromatograms.
    Supports multiple m/z channels per compound.

    Parameters
    ----------
    data_dict : dict
        {sample_id -> (T × C) matrix}.
    rt_axis : np.ndarray
        Retention times (length T).
    mz_values : list[int]
        List of m/z values corresponding to matrix columns.
    compounds : list[dict]
        Compound definitions with "name", "mz" (list of ints),
        "rt_start", "rt_stop".
    drop_list : list[str]
        List of compound names to zero out.
    """
    data_dict_masked = {}
    mz_to_idx = {m: i for i, m in enumerate(mz_values)}

    for sample, matrix in data_dict.items():
        # Align lengths in case of 1-point mismatch
        T = min(matrix.shape[0], len(rt_axis))
        rt_axis_aligned = rt_axis[:T]
        mat_copy = matrix[:T, :].copy()

        for cmpd in compounds:
            if cmpd["name"] in drop_list:
                # Iterate over all m/z channels of this compound
                for m in cmpd["mz"]:
                    mz_idx = mz_to_idx.get(m)
                    if mz_idx is None:
                        continue
                    mask = (rt_axis_aligned >= cmpd["rt_start"]) & (rt_axis_aligned <= cmpd["rt_stop"])
                    if not np.any(mask):
                        continue
                    mat_copy[mask, mz_idx] = 0.0

        data_dict_masked[sample] = mat_copy

    return data_dict_masked

def get_rt_axis_and_mz_values(dataset_path):
    # Pick first .D folder
    first_d = [d for d in os.listdir(dataset_path) if d.endswith(".D")][0]
    first_csv = [f for f in os.listdir(os.path.join(dataset_path, first_d)) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(dataset_path, first_d, first_csv))

    rt_axis = df.iloc[:, 0].values / 60000.0   # ms → minutes
    mz_values = [int(c) for c in df.columns[1:]]  # assumes cols are mz channels
    return rt_axis, mz_values

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
        xticklabels=[f"{rt_edges[col]}–{rt_edges[col + 1]}" for col in range(n_rt_bins)],
        yticklabels=[f"{mz_edges[row]}–{mz_edges[row + 1]}" for row in range(n_mz_bins)],
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

        # update the heatmap colors
        mesh.collections[0].set_array(heatmap.ravel())

        # remove old annotations
        for txt in ax.texts:
            txt.remove()

        # redraw fresh annotations
        for (row, col), val in np.ndenumerate(heatmap):
            if not np.isnan(val):
                ax.text(
                    col + 0.5, row + 0.5, f"{val:.2f}",
                    ha="center", va="center",
                    color="white", fontsize=8
                )

        plt.pause(0.1)

    plt.ioff()
    plt.show()

    return heatmap


# -----------------------------
# Main Classification
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
    show_pred_plot=False,
    pred_plot_mode="regression",
    plot_regress_corr=False,
    plot_rt_bin_analysis=False,
    rt_analysis_filename=None
):
    # === Origin mapping (same as before) ===
    letter_to_origin = {
        'M': 'Neuchâtel', 'N': 'Neuchâtel',
        'J': 'Geneva', 'L': 'Geneva',
        'H': 'Valais',
        'U': 'California', 'X': 'Oregon',
        'D': 'Burgundy', 'E': 'Burgundy', 'Q': 'Burgundy',
        'P': 'Burgundy', 'R': 'Burgundy', 'Z': 'Burgundy',
        'C': 'Alsace', 'K': 'Alsace', 'W': 'Alsace', 'Y': 'Alsace'
    }

    origins = [letter_to_origin.get(lbl[0], "Unknown") for lbl in raw_sample_labels]

    # === Special case: domain-split experiments ===
    if cfg.pred_plot_region in ["burgundy_eu", "burgundy_us"]:
        if cfg.pred_plot_region == "burgundy_eu":
            test_origins = ["Neuchâtel", "Geneva", "Valais", "Alsace"]
        else:  # burgundy_vs_us
            test_origins = ["California", "Oregon"]

        train_mask = np.isin(origins, ["Burgundy"])
        test_mask = np.isin(origins, test_origins)

        # Year labels only (class_by_year=True)
        y_train = np.array(year_labels)[train_mask].astype(int)
        y_test = np.array(year_labels)[test_mask].astype(int)

        origins_test = np.array(origins)[test_mask]
        raw_labels_test = np.array(raw_sample_labels)[test_mask]
        year_labels_test = np.array(year_labels)[test_mask]

        # Extract TIC features
        X_train = utils.compute_features(data[train_mask], feature_type=feature_type)
        X_test = utils.compute_features(data[test_mask], feature_type=feature_type)

        # Scale
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Train classifier (on Burgundy) and test on EU/US
        # ===== Select classifier =====
        CLASSIFIER_TYPE = "RGC"  # or "LDA"

        if CLASSIFIER_TYPE == "RGC":
            from sklearn.linear_model import RidgeClassifier
            model = RidgeClassifier()
        elif CLASSIFIER_TYPE == "LDA":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            model = LinearDiscriminantAnalysis()
        else:
            raise ValueError(f"Unsupported classifier: {CLASSIFIER_TYPE}")

        # ===== Train classifier (on Burgundy) and test on EU/US =====
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Accuracy
        acc = np.mean(y_pred == y_test)
        logger.info(f"{CLASSIFIER_TYPE} Hold-out Accuracy ({cfg.pred_plot_region}): {acc:.3f}")

        # Plot classification results
        try:
            plot_true_vs_pred(
                y_true=y_test,
                y_pred=y_pred,
                origins=origins,  # per-sample test origins
                pred_plot_mode=pred_plot_mode,
                year_labels=year_labels,  # vintages of test wines
                data=data,
                feature_type=feature_type,
                pred_plot_region=cfg.pred_plot_region,
            )
        except Exception as e:
            print(f"⚠️ Skipping predicted vs true plot for {cfg.pred_plot_region} due to error: {e}")

        return acc, y_test, y_pred, origins_test, raw_labels_test, year_labels_test


    # === Normal regional filters (all, european, burgundy) ===
    if cfg.pred_plot_region == "european":
        keep_mask = np.isin(origins, ["Neuchâtel", "Geneva", "Valais", "Burgundy", "Alsace"])
    elif cfg.pred_plot_region == "burgundy":
        keep_mask = np.isin(origins, ["Burgundy"])
    elif cfg.pred_plot_region == "burgundy_random":
        burgundy_mask = np.isin(origins, ["Burgundy"])
        n_burgundy = np.sum(burgundy_mask)

        acc_vals, r_vals, r2_vals, res_std_vals = [], [], [], []
        acc_tolerance_vals = {tol: [] for tol in [1, 2, 3, 4, 5]}  # store ±tolerances
        plot_flag = False

        all_repeats_coefs = []

        for rep in range(num_repeats):
            rand_indices = np.random.choice(len(origins), n_burgundy, replace=False)
            if rep == num_repeats - 1:
                plot_flag = True

            # --- Robust call (supports both old and new return forms)
            result = plot_true_vs_pred(
                y_true=None,
                y_pred=None,
                origins=np.array(origins)[rand_indices],
                pred_plot_mode=pred_plot_mode,
                year_labels=np.array(year_labels)[rand_indices],
                data=data[rand_indices],
                feature_type=feature_type,
                pred_plot_region="burgundy_random",
                plot=plot_flag
            )

            *_, all_coefs = result
            if all_coefs is not None:
                all_repeats_coefs.append(all_coefs)

            # --- Safe unpacking ---
            if isinstance(result, tuple):
                if len(result) == 6:
                    acc, acc_tolerance, r, r2, res_std, all_coefs = result
                elif len(result) == 5:
                    acc, r, r2, res_std = result
                    acc_tolerance = None
                else:
                    continue
            else:
                continue

            # --- Collect results ---
            if acc is not None:
                acc_vals.append(acc)
            if acc_tolerance:
                for tol, val in acc_tolerance.items():
                    acc_tolerance_vals[tol].append(val)
            if r is not None:
                r_vals.append(r)
            if r2 is not None:
                r2_vals.append(r2)
            if res_std is not None:
                res_std_vals.append(res_std)

        # Convert to arrays
        res_std_vals = np.array(res_std_vals)

        if all_repeats_coefs:
            mean_coefs = np.mean(all_repeats_coefs, axis=0)
            plot_avg_weights_vs_global_mean(mean_coefs, ref_chrom=np.sum(data, axis=2))
            # plot_fold_weight_heatmap(mean_coefs, ref_chrom=np.sum(data, axis=2))

        # === Summaries ===
        logger.info(
            f"Residual SDs across {num_repeats} repeats:\n"
            f"  Min = {res_std_vals.min():.3f}, "
            f"Max = {res_std_vals.max():.3f}, "
            f"Median = {np.median(res_std_vals):.3f}, "
            f"IQR = {np.percentile(res_std_vals, 75) - np.percentile(res_std_vals, 25):.3f}"
        )

        # === Optional plot ===
        plt.figure(figsize=(6, 4))
        plt.hist(res_std_vals, bins=10, color="skyblue", edgecolor="black")
        plt.xlabel("Residual SD (years)")
        plt.ylabel("Frequency")
        plt.title(f"Distribution of residual SDs across {num_repeats} repeats")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)

        # === Tolerance means ===
        tol_means = {tol: np.mean(vals) for tol, vals in acc_tolerance_vals.items() if len(vals) > 0}
        tol_stds = {tol: np.std(vals) for tol, vals in acc_tolerance_vals.items() if len(vals) > 0}

        # === Report all ===
        logger.info(
            f"Burgundy vs Random ({num_repeats}x): "
            f"Acc = {np.mean(acc_vals):.2f} ± {np.std(acc_vals):.2f}, "
            f"R = {np.mean(r_vals):.3f} ± {np.std(r_vals):.3f}, "
            f"R² = {np.mean(r2_vals):.3f} ± {np.std(r2_vals):.3f}, "
            f"Avg Residual SD = {np.mean(res_std_vals):.3f} years"
        )

        # Print tolerance accuracies
        tol_str = "  |  ".join(
            [f"±{tol}yr = {tol_means[tol]:.1f} ± {tol_stds[tol]:.1f}%" for tol in tol_means]
        )
        logger.info(f"Tolerances: {tol_str}")

        return (
            np.mean(acc_vals), np.std(acc_vals),
            np.mean(r_vals), np.std(r_vals),
            np.mean(r2_vals), np.std(r2_vals),
            np.mean(res_std_vals),
            tol_means
        )

    else:  # "all" or default
        keep_mask = np.ones(len(origins), dtype=bool)

    # Apply mask
    data = data[keep_mask]
    labels = np.array(labels)[keep_mask]
    raw_sample_labels = np.array(raw_sample_labels)[keep_mask]
    year_labels = np.array(year_labels)[keep_mask] if year_labels is not None else None
    origins = np.array(origins)[keep_mask]

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

    print(classifier)
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
        show_pred_plot=show_pred_plot,
        pred_plot_region=cfg.pred_plot_region,
        origins=origins,
    )

    if cv_type == "LOO":
        mean_acc, std_acc, all_scores, all_labels, test_sample_names, all_preds, all_coefs = result
        # mean_acc, std_acc, scores, all_labels, test_samples_names, _ = result
    else:
        mean_acc, std_acc, all_scores, all_labels, test_sample_names, all_preds = result
        all_coefs = None
        # mean_acc, std_acc, *rest = result

    if all_coefs is not None:
        # Create a region-specific subfolder, e.g. results/weights/burgundy/
        region_name = cfg.pred_plot_region if hasattr(cfg, "pred_plot_region") else "unknown_region"
        save_dir = os.path.join("results", "weights", region_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save raw weights (all folds/repeats)
        np.save(os.path.join(save_dir, f"weights_{region_name}.npy"), all_coefs)

        # Compute mean across folds/repeats
        mean_coefs = np.mean(all_coefs, axis=0)
        np.save(os.path.join(save_dir, f"weights_mean_{cfg.pred_plot_region}.npy"), mean_coefs)

        # Get directory of the current script, no matter where it's called from
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # file_path = os.path.join(script_dir, "results",
        #                          "rt_distribution_best_acc_winery_concatenated_10rt_1mz_10rep.npz")
        #
        # dist = np.load(file_path, allow_pickle=True)

        # --- Plot comparison ---
        ref_chrom = np.sum(data, axis=2)

        if plot_rt_bin_analysis:
            # Plot comparison
            plot_rtbin_evolution_full(
                npz_path=rt_analysis_filename,
                all_coefs=all_coefs,
                ref_chrom=ref_chrom,
                normalize="fraction",
                cumulative=True,
                cmap="plasma",
                step_pct=53,
                # step_pct=10 / 116. * 100 - 1,
            )


        # plot_avg_weights_vs_rtbin_frequency(
        #     all_coefs=mean_coefs,
        #     ref_chrom=ref_chrom,
        #     rt_edges=dist["rt_edges"],
        #     rt_freq=dist["rt_freq"],
        #     bin_order_mode=dist["bin_order_mode"],
        #     region_label=cfg.pred_plot_region,
        # )

        # plot_avg_weights_vs_global_mean(
        #     mean_coefs,
        #     ref_chrom=ref_chrom,
        #     alpha_band=0.25,
        #     region_label=cfg.pred_plot_region,
        # )

        if plot_regress_corr:
            r, p = compare_regression_weights(
                "results/weights/all/weights_mean_all.npy",
                "results/weights/burgundy/weights_mean_burgundy.npy",
                savefig_path="results/weights/weights_comparison_all_vs_burgundy.png",
                title_prefix="Ridge regression"
            )

            print(f"Pearson correlation: r = {r:.3f}, p = {p:.1e}")

    logger.info(f"Final Accuracy (no survival): {mean_acc:.3f}")
    return result



if __name__ == "__main__":
    # plot_rt_bin_evolution_heatmap(
    #     "results/rt_distribution_allsteps_winery_concatenated_5rt_1mz_50rep.npz",
    #     normalize=False,
    #     cumulative=True,
    #     cmap="plasma",
    #     step_pct=80
    # )
    # # plot_accuracy_vs_channels("ranked_add_sum_results.csv", "ranked_add_concat_results.csv")
    raw_cfg = load_config()
    cfg = build_run_config(raw_cfg)

    # Summary header
    wine_kind, cl, gcms, data_dict, dataset_origins = load_and_prepare_data(raw_cfg)

    task = "classification"
    summary = {
        "Task": task,
        "Wine kind": wine_kind,
        "Datasets": ", ".join(cfg.selected_datasets or []),
        "Feature type": cfg.feature_type,
        "Classifier": cfg.classifier,
        "Repeats": cfg.num_repeats,
        "Normalize": cfg.normalize,
        "Decimation": cfg.n_decimation,
        "Sync": cfg.sync_state,
        "Year Classification": cfg.class_by_year,
        "Region": cfg.region,
        "CV type": cfg.cv_type,
        "RT range": cfg.rt_range,
        "Confusion matrix": cfg.show_confusion_matrix,
    }

    logger_raw("\n")
    logger.info("------------------------ RUN SCRIPT -------------------------")
    logger.info("Configuration Parameters")
    for k, v in summary.items():
        logger_raw(f"{k:>20s}: {v}")

    # Strategy
    strategy = get_strategy_by_wine_kind(
        wine_kind=wine_kind,
        region=cfg.region,
        get_custom_order_func=utils.get_custom_order_for_pinot_noir_region,
    )

    # Optional alignment
    if cfg.sync_state:
        tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)
        gcms = GCMSDataProcessor(data_dict)

    # Prepare arrays
    data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
    raw_sample_labels = labels.copy()

    # Restrict to Burgundy if requested
    if wine_kind == "pinot_noir" and cfg.region == "burgundy":
        burgundy_prefixes = ("D", "P", "R", "Q", "Z", "E")
        mask = np.array([label.startswith(burgundy_prefixes) for label in labels])
        data = data[mask]
        labels = labels[mask]
        raw_sample_labels = raw_sample_labels[mask]

    labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, cfg.region, cfg.class_by_year, None)

    # Prepare 3D for m/z mode (if needed)
    data3d = None
    # if sotf_mz_flag:
    data3d = dict_to_array3d(data_dict)  # (N, T, C)
    if wine_kind == "pinot_noir" and cfg.region == "burgundy":
        data3d = data3d[mask]

    # -----------------------------
    # Branch by mode
    # -----------------------------
    results = None
    if cfg.oak_analysis:
        # -------------------------
        # Compound-driven analysis
        # -------------------------
        COMPOUNDS = [
            {"name": "Furfural", "mz": [96, 95, 67], "rt_start": 11.06, "rt_stop": 11.41},
            {"name": "5-Methylfurfural", "mz": [110, 109, 53], "rt_start": 14.06, "rt_stop": 14.33},
            {"name": "5-Hydroxymethylfurfural", "mz": [97, 126, 69], "rt_start": 16.63, "rt_stop": 16.88},
            {"name": "cis-Oak lactone", "mz": [99], "rt_start": 22.44, "rt_stop": 22.73},
            {"name": "trans-Oak lactone", "mz": [99], "rt_start": 24.12, "rt_stop": 24.40},
            {"name": "Guaiacol", "mz": [124, 94], "rt_start": 21.73, "rt_stop": 22.00},
            {"name": "Eugenol", "mz": [164, 149], "rt_start": 29.24, "rt_stop": 29.55},
            {"name": "Vanillin", "mz": [151, 152, 123], "rt_start": 38.83, "rt_stop": 39.70},
            {"name": "Acetovanillone", "mz": [166, 151], "rt_start": 38.82, "rt_stop": 39.50},
            {"name": "Syringaldehyde", "mz": [182, 181, 167], "rt_start": 45.35, "rt_stop": 46.08},
            {"name": "Syringol", "mz": [154, 139, 111], "rt_start": 31.36, "rt_stop": 31.65},
        ]

        drop_list = [
            "Furfural",
            "5-Methylfurfural",
            "5-Hydroxymethylfurfural",
            "cis-Oak lactone",
            "trans-Oak lactone",
            "Guaiacol",
            "Eugenol",
            "Vanillin",
            "Acetovanillone",
            "Syringaldehyde",
            "Syringol",
        ]
        logger.info("Running compound-level analysis")

        dataset_path = cfg.datasets[cfg.selected_datasets[0]]
        rt_axis_full, mz_values = get_rt_axis_and_mz_values(dataset_path)
        rt_axis = rt_axis_full[::cfg.n_decimation]

        # Case 1: Integrated peak areas already pre-computed
        if cfg.oak_mode == "integrated":
            df_peaks = pd.read_csv(cfg.oak_peaks_path)

            df_wide = df_peaks.pivot_table(
                index="Sample",
                columns="Compound",
                values="Area",
                aggfunc="first"
            ).fillna(0.0)

            if "Isoeugenol" in df_wide.columns:
                logger.info("Dropping Isoeugenol from integrated peak table")
                df_wide = df_wide.drop(columns=["Isoeugenol"])

            data = df_wide.values[:, :, np.newaxis]
            labels = df_wide.index.to_numpy()
            raw_sample_labels = labels.copy()

            labels, year_labels = process_labels_by_wine_kind(
                labels, wine_kind, cfg.region, cfg.class_by_year, None
            )

            logger.info(f"Loaded integrated peak table with shape {data.shape}")

        # Case 2: Mask specific compounds directly from chromatograms
        elif cfg.oak_mode == "mask":
            logger.info(f"Masking compounds: {drop_list}")
            COMPOUNDS_INDEXED = rt_windows_to_indices(rt_axis, COMPOUNDS)

            first_matrix = next(iter(data_dict.values()))
            T = min(len(rt_axis), first_matrix.shape[0])
            rt_axis = rt_axis[:T]
            data_dict = {s: m[:T, :] for s, m in data_dict.items()}

            data_dict = mask_compounds(data_dict, rt_axis, mz_values, COMPOUNDS_INDEXED, drop_list)
            gcms = GCMSDataProcessor(data_dict)

            data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
            raw_sample_labels = labels.copy()
            labels, year_labels = process_labels_by_wine_kind(
                labels, wine_kind, cfg.region, cfg.class_by_year, None
            )
        else:
            raise ValueError(f"Invalid oak_mode: {cfg.oak_mode}. Must be 'integrated' or 'mask'.")

        results = run_normal_classification(
            data=data,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=cfg.classifier,
            wine_kind=wine_kind,
            class_by_year=cfg.class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cfg.cv_type,
            num_repeats=cfg.num_repeats,
            normalize_flag=cfg.normalize,
            region=cfg.region,
            feature_type=cfg.feature_type,
            projection_source=cfg.projection_source,
            show_confusion_matrix=cfg.show_confusion_matrix,
            pred_plot_mode=cfg.pred_plot_mode
        )
    elif cfg.sotf_remove_2d:
        results = run_sotf_remove_2D_noleak(
            data3d=data,  # shape (samples, RT, m/z)
            labels=labels,  # class labels
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=cfg.classifier,
            wine_kind=wine_kind,
            class_by_year=cfg.class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cfg.cv_type,  # "LOO", "LOOPC", or "stratified"
            num_repeats=cfg.num_repeats,
            normalize_flag=cfg.normalize,
            region=cfg.region,
            feature_type=cfg.feature_type,
            n_rt_bins=cfg.n_rt_bins,  # number of retention time bins
            n_mz_bins=cfg.n_mz_bins,  # number of m/z bins
            random_state=42,
            cv_design='B'
        )
    elif cfg.sotf_add_2d:
        # results = run_sotf_add_2d(
        results = run_sotf_add_2d_noleak(
            data3d=data3d,  # shape (samples, RT, m/z)
            labels=labels,  # class labels
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=cfg.classifier,
            wine_kind=wine_kind,
            class_by_year=cfg.class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cfg.cv_type,  # "LOO", "LOOPC", or "stratified"
            num_repeats=cfg.num_repeats,
            normalize_flag=cfg.normalize,
            region=cfg.region,
            feature_type=cfg.feature_type,
            n_rt_bins=cfg.n_rt_bins,  # number of retention time bins
            n_mz_bins=cfg.n_mz_bins,  # number of m/z bins
            random_state=42,
            n_jobs=20,
            mz_min=0, mz_max=181,
            cube_repr = "tic",
            cv_design="A"
        )
    elif cfg.reg_acc_map:
        heatmap = run_region_accuracy_heatmap(
            data3d=data3d,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=cfg.classifier,
            wine_kind=wine_kind,
            class_by_year=cfg.class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cfg.cv_type,  # e.g. "LOO"
            num_repeats=cfg.num_repeats,
            normalize_flag=cfg.normalize,
            region=cfg.region,
            feature_type=cfg.feature_type,
            projection_source=cfg.projection_source,
            show_confusion_matrix=cfg.show_confusion_matrix,
            n_rt_bins=cfg.n_rt_bins,  # number of RT slices
            n_mz_bins=cfg.n_mz_bins,  # number of m/z slices
        )
    else:
        results = run_normal_classification(
            data=data,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            year_labels=year_labels,
            classifier=cfg.classifier,
            wine_kind=wine_kind,
            class_by_year=cfg.class_by_year,
            strategy=strategy,
            dataset_origins=dataset_origins,
            cv_type=cfg.cv_type,
            num_repeats=cfg.num_repeats,
            normalize_flag=cfg.normalize,
            region=cfg.region,
            feature_type=cfg.feature_type,
            projection_source=cfg.projection_source,
            show_confusion_matrix=cfg.show_confusion_matrix,
            show_pred_plot=cfg.show_pred_plot,
            pred_plot_mode=cfg.pred_plot_mode,
            plot_regress_corr=cfg.plot_regress_corr,
            plot_rt_bin_analysis=cfg.plot_rt_bin_analysis,
            rt_analysis_filename=cfg.rt_analysis_filename
        )

    # -----------------------------
    # Optional projection plot (preserve behavior)
    # -----------------------------
    scores = None
    all_labels = None
    test_samples_names = None
    if results and cfg.cv_type == "LOO":
        if cfg.pred_plot_region == "burgundy_random":
            mean_acc, std_acc, mean_r, std_r, mean_r2, std_r2, mean_res_std, tol_means = results
            # mean_acc, std_acc, mean_r, std_r, mean_r2, std_r2, _ = results
            # Pretty-print tolerance accuracies
            tol_summary = "  |  ".join([f"±{tol}yr = {tol_means[tol]:.1f}%" for tol in sorted(tol_means)])
            print(
                f"Acc={mean_acc:.3f}, "
                f"Average accuracies by tolerance: {tol_summary}"
                f"R={mean_r:.3f}±{std_r:.3f}, "
                f"R²={mean_r2:.3f}±{std_r2:.3f}"
            )
        else:
            # --- Flexible unpacking: handle both 6- and 7-element results ---
            if len(results) == 7:
                mean_acc, std_acc, scores, all_labels, test_samples_names, all_preds, all_coefs = results
            else:
                mean_acc, std_acc, scores, all_labels, test_samples_names, all_preds = results
                all_coefs = None

            print(f"Final Accuracy (no survival): {mean_acc:.3f}")
            # mean_acc, std_acc, scores, all_labels, test_samples_names, _ = results
    elif results and cfg.cv_type in ["LOOPC", "stratified"]:
        mean_acc, std_acc, *_ = results

    if cfg.plot_projection:
        do_projection_plot(
            plot_projection=cfg.plot_projection,
            projection_source=cfg.projection_source,
            projection_method=cfg.projection_method,
            projection_dim=cfg.projection_dim,
            n_neighbors=cfg.n_neighbors,
            random_state=cfg.random_state,
            color_by_country=cfg.color_by_country,
            invert_x=cfg.invert_x,
            invert_y=cfg.invert_y,
            rot_axes=cfg.rot_axes,
            sample_display_mode=cfg.sample_display_mode,
            color_by_winery=cfg.color_by_winery,
            color_by_origin=cfg.color_by_origin,
            exclude_us=cfg.exclude_us,
            density_plot=cfg.density_plot,
            region=cfg.region,
            labels=labels,
            raw_sample_labels=raw_sample_labels,
            scores=scores,
            all_labels=all_labels,
            data=data,
        )


