"""
Ridge Regression for Predicting Champagne Sensory Profiles Using GC-MS Data
============================================================================

This script implements a regression pipeline to predict detailed sensory scores for Champagne wines
based on chemical information from GC-MS chromatograms and taster identity. It combines signal
processing, metadata handling, and machine learning in a reproducible way.

Use case:
---------
Designed to assess the predictive power of GC-MS data (flattened chromatograms or TICs) in modeling
human sensory perception, taking into account taster-specific biases via one-hot encoding.

Taster Modeling Approach:
-------------------------
A single Ridge regression model is trained across all tasters. Taster identity is provided as an input
feature using one-hot encoding, allowing the model to learn both general chemical–sensory relationships
and individual taster preferences. There is no separate model per taster.

Key Features:
-------------
1. **GC-MS Data Loading**:
   - Reads chromatographic signals from `.csv` files using a custom utility (`utils.py`).
   - Chromatograms are decimated (downsampled) along the retention time axis.

2. **Sensory Metadata Preprocessing**:
   - Loads a structured sensory dataset (`sensory_scores.csv`) with wine codes, tasters, and 100+ attributes.
   - Cleans and averages duplicate ratings by (wine, taster) pairs.

3. **Feature Construction**:
   - Each sample’s feature vector is the mean chromatogram + one-hot encoded taster ID.
   - Targets are sensory attribute vectors (e.g., fruity, toasted, acid, etc.).

4. **Model Training & Evaluation**:
   - Runs `N_REPEATS` Ridge regression fits using randomized train/test splits (80/20).
   - Evaluates with MAE and RMSE for each sensory attribute.
   - Aggregates performance statistics and prints per-taster error summaries.

5. **Interpretability**:
   - Separates the model weights into chromatogram vs. taster contributions.
   - Identifies which tasters introduce the most prediction error.

6. **Visualization**:
   - For each taster, plots the predicted vs. true sensory profiles across test wines.
   - Optional single-wine plot function available for inspection.

Parameters to Adjust:
---------------------
- `directory`: Path to chromatogram files.
- `N_DECIMATION`: Decimation factor (to reduce RT dimensionality).
- `N_REPEATS`: Number of random train/test splits.
- `TEST_SIZE`: Test proportion (e.g., 0.2 = 20%).

Dependencies:
-------------
- pandas, numpy, scikit-learn, matplotlib
- Custom `utils.py` for chromatogram loading

Example Output:
---------------
- Per-attribute MAE and RMSE scores
- Per-taster mean absolute errors
- Prediction plots showing model quality for each taster
"""
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from gcmswine import utils  # Your custom module
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import re
from sklearn.metrics import r2_score
import os
import yaml
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold
from gcmswine.helpers import (
    load_config, get_model, load_chromatograms_decimated, load_and_clean_metadata, build_feature_target_arrays,
)
try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold
from gcmswine.wine_analysis import ChromatogramAnalysis, GCMSDataProcessor
from gcmswine.logger_setup import logger, logger_raw
from collections import Counter

# Define helper functions
def build_feature_target_arrays(metadata, sensory_cols, data_dict):
    X_raw, y, taster_ids, sample_ids = [], [], [], []
    skipped_count = 0

    for _, row in metadata.iterrows():
        sample_id = row['code vin']
        taster_id = row['taster']
        try:
            attributes = row[sensory_cols].astype(float).values
        except:
            continue  # skip row if attribute values can't be converted

        replicate_keys = [k for k in data_dict if k.startswith(sample_id)]
        if not replicate_keys:
            skipped_count += 1
            print(f"Warning: No chromatograms found for sample {sample_id}")
            continue

        replicates = np.array([data_dict[k] for k in replicate_keys])
        chromatogram = np.mean(replicates, axis=0).flatten()
        chromatogram = np.nan_to_num(chromatogram, nan=0.0)

        X_raw.append(chromatogram)
        y.append(attributes)
        taster_ids.append(taster_id)
        sample_ids.append(sample_id)

    print(f"\nTotal samples skipped due to missing chromatograms: {skipped_count}")
    return np.array(X_raw), np.array(y), np.array(taster_ids), np.array(sample_ids)

def compare_self_vs_group_models_per_taster(
    X_input, y, sample_ids, taster_ids, model, sensory_cols,
    num_repeats=5, test_size=0.2, normalize=True, random_seed=42
):
    """
    For each taster:
      - Train a model on their own scores (self-model)
      - Train a model on the group average (excluding them) and test on their scores
    Returns
    -------
    results_df : pd.DataFrame with R² per attribute and overall mean R² for both models.
    """
    unique_tasters = np.unique(taster_ids)
    rows = []

    for target_taster in unique_tasters:
        # logger.info(f"--- Evaluating taster: {taster} ---")

        # === SELF MODEL ===
        mask_self = (taster_ids == target_taster)
        X_self = X_input[mask_self]
        y_self = y[mask_self]

        r2_self_all = []
        for repeat in range(num_repeats):
            train_idx, test_idx = train_test_split(
                np.arange(len(X_self)), test_size=test_size, random_state=random_seed + repeat
            )
            X_train, X_test = X_self[train_idx], X_self[test_idx]
            y_train, y_test = y_self[train_idx], y_self[test_idx]

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y.shape[1])]
            r2_self_all.append(r2_scores)

        r2_self_avg = np.nanmean(r2_self_all, axis=0)

        # === GROUP MODEL (Train on average of others, test on taster) ===
        mask = taster_ids != target_taster
        X_sub = X_input[mask]
        y_sub = y[mask]
        sample_ids_sub = sample_ids[mask]
        results = train_and_evaluate_average_scores_model(
            X_input=X_sub,
            y=y_sub,
            sample_ids=sample_ids_sub,
            model=model,
            num_repeats=num_repeats,
            test_size=test_size,
            normalize=normalize,
            random_seed=random_seed
        )
        all_mae, all_rmse, all_y_test, all_y_pred, all_sample_ids, all_taster_ids = results

        # Concatenate all predictions and ground truth
        y_test_all = np.vstack(all_y_test)
        y_pred_all = np.vstack(all_y_pred)

        r2_per_attr = []
        for i, col in enumerate(sensory_cols):
            try:
                r2 = r2_score(y_test_all[:, i], y_pred_all[:, i])
            except ValueError:
                r2 = float("nan")
            r2_per_attr.append(r2)

        # Store results
        row = {
            'taster': target_taster,
            'mean_r2_self': np.nanmean(r2_self_avg),
            'mean_r2_group': np.nanmean(r2_per_attr),
        }
        for i, attr in enumerate(sensory_cols):
            row[f'{attr}_r2_self'] = r2_self_avg[i]
            row[f'{attr}_r2_group'] = r2_per_attr[i]

        rows.append(row)

    df = pd.DataFrame(rows).set_index("taster")
    logger.info("\n=== Comparison of self vs group models ===")
    df_to_log = df[['mean_r2_self', 'mean_r2_group']].round(3).reset_index()
    logger_raw(df_to_log.to_string(index=False))
    means = df[['mean_r2_self', 'mean_r2_group']].mean()
    logger_raw(f"Mean r2_self: {means['mean_r2_self']:.3f}\nMean r2_group: {means['mean_r2_group']:.3f}")
    return df


def train_and_evaluate_average_scores_model(X_input, y, sample_ids, model, *,
                                             num_repeats=5, test_size=0.2, normalize=True,
                                             random_seed=42,
                                             taster_ids=None):
    """
    Train and evaluate a model on average sensory scores per wine sample.

    Parameters
    ----------
    X_input : ndarray
        Input features (e.g., chromatograms).
    y : ndarray
        Target sensory scores (one row per taster-wine pair).
    sample_ids : ndarray
        Wine identifiers (aligned with rows in X_input).
    model : object
        Scikit-learn regressor.
    num_repeats : int
        Number of repeats.
    test_size : float
        Proportion of test set.
    normalize : bool
        Whether to normalize input features.
    random_seed : int
        Seed for reproducibility.
    taster_ids : ndarray or None
        Optional taster identifiers aligned with rows in `y`.

    Returns
    -------
    all_mae : list
        List of MAE arrays across repeats.
    all_rmse : list
        List of RMSE arrays across repeats.
    last_y_test : ndarray
        True values of the last test split.
    last_y_pred : ndarray
        Predictions of the last test split.
    last_sample_ids : ndarray
        Wine codes of the last test split.
    last_taster_ids : ndarray or None
        Taster identifiers for the test set, if provided.
    """
    # Average feature vectors and targets by wine code
    unique_wines = np.unique(sample_ids)
    X_avg, y_avg, wine_ids = [], [], []

    for wine in unique_wines:
        indices = np.where(sample_ids == wine)[0]
        X_avg.append(np.mean(X_input[indices], axis=0))
        y_avg.append(np.mean(y[indices], axis=0))
        wine_ids.append(wine)

    X_input = np.array(X_avg)
    y = np.array(y_avg)
    sample_ids = np.array(wine_ids)
    taster_ids = None  # dropped for average model

    all_mae, all_rmse = [], []
    last_y_test = last_y_pred = last_sample_ids = last_taster_ids = None

    n_splits = 5
    all_mae, all_rmse = [], []
    all_y_test, all_y_pred = [], []
    all_sample_ids, all_taster_ids = [], []
    for repeat in range(num_repeats):
        splits = [train_test_split(np.arange(len(X_input)), test_size=test_size, random_state=random_seed + repeat)]
        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"[AVG] Repeat {repeat + 1}, Fold {fold_idx + 1}")

            X_train, X_test = X_input[train_idx], X_input[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            s_test = sample_ids[test_idx]
            t_test = taster_ids[test_idx] if taster_ids is not None else None

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Accumulate
            all_y_test.append(y_test)
            all_y_pred.append(y_pred)
            all_sample_ids.append(s_test)
            if t_test is not None:
                all_taster_ids.append(t_test)

            mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))
            all_mae.append(mae)
            all_rmse.append(rmse)

    return all_mae, all_rmse, all_y_test, all_y_pred, all_sample_ids, all_taster_ids


def train_and_evaluate_model(X_input, y, taster_ids, sample_ids, model, *,
                             num_repeats=5, test_size=0.2, normalize=True,
                             taster_scaling=False, group_wines=False, random_seed=42):
    """
    Train and evaluate the model with or without grouping by wines.

    Parameters
    ----------
    X_input : ndarray
        Input feature matrix (chromatogram + one-hot taster).
    y : ndarray
        Target matrix (sensory attributes).
    taster_ids : ndarray
        Array of taster identifiers (1D).
    sample_ids : ndarray
        Array of wine identifiers (1D).
    model : regressor object
        The regression model to train.
    num_repeats : int
        Number of repeats.
    test_size : float
        Proportion of data to use for test split.
    normalize : bool
        Whether to normalize features before training.
    taster_scaling : bool
        Whether to apply per-taster prediction scaling.
    group_wines : bool
        Whether to group folds by wine ID.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    all_mae : list of np.ndarray
    all_rmse : list of np.ndarray
    taster_mae_summary : dict
    last_y_test : np.ndarray
    last_y_pred : np.ndarray
    last_sample_ids : np.ndarray
    last_taster_ids : np.ndarray
    """
    all_mae, all_rmse = [], []
    taster_mae_summary = defaultdict(list)
    last_y_test = last_y_pred = last_sample_ids = last_taster_ids = None

    n_splits = 5

    for repeat in range(num_repeats):
        if group_wines:
            gkf = GroupKFold(n_splits=n_splits)
            splits = gkf.split(X_input, y, groups=sample_ids)
        else:
            splits = [train_test_split(
                np.arange(len(X_input)),
                test_size=test_size,
                random_state=random_seed + repeat
            )]

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            print(f"Repeat {repeat + 1}, Fold {fold_idx + 1}")

            X_train, X_test = X_input[train_idx], X_input[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            t_train, t_test = taster_ids[train_idx], taster_ids[test_idx]
            s_test = sample_ids[test_idx]

            if normalize:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            if taster_scaling:
                taster_scalers = {}
                if group_wines:
                    y_train_pred = model.predict(X_train)
                else:
                    y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)

                for t in np.unique(t_train):
                    mask = (t_train == t)
                    if np.sum(mask) < 2:
                        continue
                    scaler_model = MultiOutputRegressor(LinearRegression(fit_intercept=False))
                    scaler_model.fit(y_train_pred[mask], y_train[mask])
                    scale = np.array([est.coef_[0] for est in scaler_model.estimators_])
                    taster_scalers[t] = scale

                for i, t in enumerate(t_test):
                    if t in taster_scalers:
                        y_pred[i] *= taster_scalers[t]

            mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
            rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

            all_mae.append(mae)
            all_rmse.append(rmse)

            for i, t in enumerate(t_test):
                taster_mae_summary[t].append(np.abs(y_test[i] - y_pred[i]))

            if repeat == num_repeats - 1 and (not group_wines or fold_idx == n_splits - 1):
                last_y_test = y_test
                last_y_pred = y_pred
                last_sample_ids = s_test
                last_taster_ids = t_test

    return all_mae, all_rmse, taster_mae_summary, last_y_test, last_y_pred, last_sample_ids, last_taster_ids

def plot_wine_across_tasters(y_true, y_pred, sample_ids, taster_ids, wine_code, sensory_cols):
        import matplotlib.pyplot as plt
        import numpy as np

        indices = np.where(sample_ids == wine_code)[0]
        if len(indices) < 2:
            logger.info(f"Wine {wine_code} is not rated by multiple tasters in the test set.\n")
            return

        n = len(indices)
        fig, axes = plt.subplots(1, n, figsize=(n * 4, 4), sharey=True)

        if n == 1:
            axes = [axes]

        for i, idx in enumerate(indices):
            ax = axes[i]
            taster = taster_ids[idx]
            ax.plot(y_true[idx], label="True", color="black", linewidth=2)
            ax.plot(y_pred[idx], label="Pred", color="red", linestyle="--")
            ax.set_title(f"Taster {taster}\nMAE: {np.mean(np.abs(y_true[idx] - y_pred[idx])):.2f}")
            ax.set_xticks(range(len(sensory_cols)))
            ax.set_xticklabels(sensory_cols, rotation=90, fontsize=6)
            ax.set_ylim(0, 100)
            ax.grid(True)

        plt.suptitle(f"Wine {wine_code}: Sensory predictions across tasters", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.92])
        plt.legend()
        plt.show()


# Main logic
if __name__ == "__main__":
    # Load configuration parameters from YAML
    config = load_config(os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml"))

    # --- Parameters ---
    N_DECIMATION = 10
    TEST_SIZE = 0.2
    CHROM_CAP = None
    directory = "/home/luiscamara/Documents/datasets/Champagnes/HETEROCYC"
    # directory = "/home/luiscamara/Documents/datasets/Champagnes/DMS"
    metadata_path = "/home/luiscamara/Documents/datasets/Champagnes/sensory_scores.csv"
    random_seed = 42                   # For reproducibility
    column_indices = None  # or specify which columns to use
    row_start, row_end = 0, None  # if you want to trim rows

    # Load user-defined options
    wine_kind = "Champagne"
    task= "Model Global"  # hard-coded for now
    dataset = os.path.split(directory)[1]
    model_name = config.get("regressor", "ridge").lower()
    feature_type = config["feature_type"]
    cv_type = config['cv_type']
    num_repeats = config.get("num_repeats", 10)
    retention_time_range = config.get("rt_range", None)
    normalize = config.get("normalize", True)
    show_taster_predictions = config.get("show_predicted_profiles", False)
    group_wines = config.get("group_wines", False)
    taster_scaling = config.get("taster_scaling", False)
    shuffle_labels = config.get("shuffle_labels", False)
    test_average_scores = config.get("test_average_scores", False)
    taster_vs_mean = config.get("taster_vs_mean", False)

    summary = {
        "Wine kind": wine_kind,
        "Task": task,
        "Dataset": dataset,
        "Regressor": model_name,
        "Feature type": feature_type,
        "CV type": cv_type,
        "Repeats": num_repeats,
        "RT range": retention_time_range,
        "Normalize": normalize,
        "Group wines": group_wines,
        "Taster scaling": taster_scaling,
        "Shuffle labels": shuffle_labels,
        "Avg. scores": test_average_scores,
        "Taster vs mean": taster_vs_mean,
        "Show taster profiles": show_taster_predictions,
    }

    logger_raw("\n")  # Blank line without timestamp
    logger.info('------------------------ RUN SCRIPT -------------------------')
    logger.info("Configuration Parameters (Champagne - Model Global)")
    for k, v in summary.items():
        logger_raw(f"{k:>22s}: {v}")

    # Select the model based on user input
    model = get_model(model_name, random_seed=random_seed)
    # print(f'Regressor is {model_name}')
    #
    # if taster_scaling:
    #     logger.info("Taster Scaling selected")

    # --- Load chromatograms ---
    data_dict, chome_length = load_chromatograms_decimated(
        directory, row_start, row_end, N_DECIMATION, retention_time_range=retention_time_range
    )

    # --- Load metadata ---
    metadata = load_and_clean_metadata(metadata_path)

    # --- Average duplicates ---
    known_metadata = ['code vin', 'taster', 'prod area', 'variety', 'cave', 'age']
    sensory_cols = [col for col in metadata.columns if col not in known_metadata and pd.api.types.is_numeric_dtype(metadata[col])]
    metadata[sensory_cols] = metadata[sensory_cols].apply(pd.to_numeric, errors='coerce')
    metadata = metadata.groupby(['code vin', 'taster'], as_index=False)[sensory_cols].mean()

    # --- Build input (X) and output (y) ---
    X_raw, y, taster_ids, sample_ids = [], [], [], []
    skipped_count = 0

    # Extract input features and target attributes from metadata and matched chromatograms
    X_raw, y, taster_ids, sample_ids = build_feature_target_arrays(metadata, sensory_cols, data_dict)

    if shuffle_labels:
        logger.info("Shuffle Labels selected")
        np.random.seed(random_seed)
        y = y.copy()
        np.random.shuffle(y)
        print("⚠️ Sensory labels have been shuffled across samples (diagnostic run).")

    # One-hot encode taster identities
    encoder = OneHotEncoder(sparse_output=False)
    taster_onehot = encoder.fit_transform(np.array(taster_ids).reshape(-1, 1))
    X_input = np.concatenate([X_raw, taster_onehot], axis=1)

    # Create a mask to filter out any samples with NaNs in input or output
    mask = ~np.isnan(X_input).any(axis=1) & ~np.isnan(y).any(axis=1)
    X_input = X_input[mask]
    y = y[mask]
    taster_ids = np.array(taster_ids)[mask]
    sample_ids = np.array(sample_ids)[mask]
    print(f"Removed {np.sum(~mask)} samples with NaNs.")

    all_mae, all_rmse, all_r2 = [], [], []
    taster_mae_summary = defaultdict(list)
    last_y_test, last_y_pred, last_sample_ids, last_taster_ids = None, None, None, None
    saved_model_coefs = []

    if test_average_scores:
        logger.info("Average Scores selected")
        results = train_and_evaluate_average_scores_model(
            X_input=X_input,
            y=y,
            sample_ids=sample_ids,
            model=model,
            num_repeats=num_repeats,
            test_size=TEST_SIZE,
            normalize=normalize,
            random_seed=random_seed
        )
        all_mae, all_rmse, all_y_test, all_y_pred, all_sample_ids, all_taster_ids = results

        # Concatenate all predictions and ground truth
        y_test_all = np.vstack(all_y_test)
        y_pred_all = np.vstack(all_y_pred)

        r2_per_attr = []
        lines = ["\nPer-attribute R² for averaged model (robust across all splits):"]
        for i, col in enumerate(sensory_cols):
            try:
                r2 = r2_score(y_test_all[:, i], y_pred_all[:, i])
            except ValueError:
                r2 = float("nan")
            r2_per_attr.append(r2)
            # Pad the name to 25 characters and right-align the R² value
            lines.append(f"{col:<20}   R² = {r2:>6.2f}")

        logger.info("\n".join(lines))
        overall_avg_r2 = np.nanmean(r2_per_attr)
        logger_raw(f"Overall average R² across attributes: {overall_avg_r2:.3f}\n")
        sys.exit(0)

    if taster_vs_mean:
        logger.info("Taster vs Mean selected")
        unique_tasters = np.unique(taster_ids)
        results = []

        results_df = compare_self_vs_group_models_per_taster(
            X_input, y, sample_ids, taster_ids,
            model=Ridge(alpha=1.0),
            sensory_cols=sensory_cols
        )
        sys.exit(0)

    results = train_and_evaluate_model(
        X_input=X_input,
        y=y,
        taster_ids=taster_ids,
        sample_ids=sample_ids,
        model=model,
        num_repeats=num_repeats,
        test_size=TEST_SIZE,
        normalize=normalize,
        taster_scaling=taster_scaling,
        group_wines=group_wines,
        random_seed=random_seed
    )
    all_mae, all_rmse, taster_mae_summary, last_y_test, last_y_pred, last_sample_ids, last_taster_ids = results

    wine_counts = Counter(last_sample_ids)
    multi_taster_wines = [wine for wine, count in wine_counts.items() if count > 1]

    mean_mae = np.mean(all_mae, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    rmse_pct = mean_rmse

    # mean_coef = np.mean(saved_model_coefs, axis=0)  # shape (n_outputs, n_features)

    # Separate chromatogram and taster weights
    n_chromatogram_features = X_raw.shape[1]
    n_taster_features = taster_onehot.shape[1]

    # chromatogram_weights = mean_coef[:, :n_chromatogram_features]
    # taster_weights = mean_coef[:, n_chromatogram_features:]

    def natural_key(t):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', t)]
    unique_tasters = sorted(set(taster_ids), key=natural_key)

    logger.info("\nPer-attribute average errors over multiple splits:")
    for i, col in enumerate(sensory_cols):
        logger_raw(f"{col:>16s}: MAE = {mean_mae[i]:5.2f}, RMSE = {mean_rmse[i]:5.2f} ({rmse_pct[i]:.1f}% of scale)")

    logger_raw(f"Overall RMSE across repeats: {np.sqrt(np.mean(mean_rmse**2)):.4f}")

    if not test_average_scores:
        logger.info("\nPer-taster average MAE (across all attributes):")
        for taster in sorted(taster_mae_summary.keys(), key=natural_key):
            all_errors = np.array(taster_mae_summary[taster])
            avg_mae_per_attr = np.mean(all_errors, axis=0)
            overall_avg = np.mean(avg_mae_per_attr)
            logger_raw(f"Taster {taster}: MAE = {overall_avg:.2f}")

        # R² estimation
        taster_r2_summary = defaultdict(list)
        for i in range(len(last_y_test)):
            taster = last_taster_ids[i]
            y_true_i = last_y_test[i]
            y_pred_i = last_y_pred[i]
            try:
                r2 = r2_score(y_true_i, y_pred_i)
            except ValueError:
                r2 = float('nan')
            taster_r2_summary[taster].append(r2)

        logger.info("\nPer-taster average R² (across all attributes):")
        all_r2_values = []
        for taster in sorted(taster_r2_summary.keys(), key=natural_key):
            r2_list = np.array(taster_r2_summary[taster])
            r2_list = r2_list[~np.isnan(r2_list)]
            avg_r2 = np.mean(r2_list) if len(r2_list) > 0 else float('nan')
            logger_raw(f"Taster {taster}: R² = {avg_r2:.2f}")
            all_r2_values.extend(r2_list)

        overall_avg_r2 = np.mean(all_r2_values) if len(all_r2_values) > 0 else float('nan')
        logger_raw(f"Overall average R² across all tasters: {overall_avg_r2:.3f}")
    else:
        logger.info("\nSkipped per-taster MAE and R² because test_average_scores=True")

    def plot_single_wine(y_true, y_pred, sample_ids, wine_code):
        import matplotlib.pyplot as plt
        import numpy as np

        indices = np.where(sample_ids == wine_code)[0]
        if len(indices) == 0:
            print(f"No wine with code {wine_code} found in the test set.")
            return

        idx = indices[0]
        true_profile = y_true[idx]
        pred_profile = y_pred[idx]
        mae = np.mean(np.abs(true_profile - pred_profile))

        plt.figure(figsize=(8, 4))
        plt.plot(true_profile, label='True', color='black', linewidth=2)
        plt.plot(pred_profile, label='Predicted', color='red', linestyle='--')
        plt.title(f"Wine {wine_code} – MAE: {mae:.2f}")
        plt.xlabel("Sensory attributes")
        plt.ylabel("Score (0–100)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show(block=False)
    # plot_single_wine(last_y_test, last_y_pred, last_sample_ids, wine_code='141T-N-27')
    plt.show()


    def plot_profiles_grouped_by_taster(y_true, y_pred, sample_ids, taster_ids, n_cols=10):
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import r2_score


        # unique_tasters = sorted(set(taster_ids))
        all_indices = [np.where(taster_ids == t)[0] for t in unique_tasters]
        n_rows = len(unique_tasters)
        max_len = max(len(idx) for idx in all_indices)
        n_cols = min(n_cols, max_len)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2), sharey=True)
        if n_rows == 1:
            axes = [axes]  # make iterable
        axes = np.array(axes).reshape(n_rows, n_cols)

        for row_idx, (taster, indices) in enumerate(zip(unique_tasters, all_indices)):
            for i in range(min(n_cols, len(indices))):
                idx = indices[i]
                ax = axes[row_idx, i]
                ax.plot(y_true[idx], color='black', linewidth=1.5)
                ax.plot(y_pred[idx], color='red', linewidth=1)
                true_profile = y_true[idx]
                pred_profile = y_pred[idx]
                mae_i = np.mean(np.abs(true_profile - pred_profile))
                try:
                    r2_i = r2_score(true_profile, pred_profile)
                except ValueError:
                    r2_i = float('nan')  # in case y_true is constant
                ax.set_title(f"{sample_ids[idx]}; MAE: {mae_i:.2f}; R²: {r2_i:.2f}", fontsize=8, pad=2)
                # ax.set_title(f"{sample_ids[idx]}\n{mae_i:.2f}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for j in range(len(indices), n_cols):
                axes[row_idx, j].axis('off')

        for row_idx, taster in enumerate(unique_tasters):
            axes[row_idx, 0].set_ylabel(f"Taster {taster}", fontsize=9, rotation=0, labelpad=40)

        plt.suptitle("All predicted sensory profiles by taster (true in black, predicted in red)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    if show_taster_predictions:
        logger.info("\nPlotting all predicted sensory profiles grouped by taster (single figure)...")
        plot_profiles_grouped_by_taster(
            y_true=last_y_test,
            y_pred=last_y_pred,
            sample_ids=last_sample_ids,
            taster_ids=last_taster_ids,
            n_cols=10
        )
        plt.show()



