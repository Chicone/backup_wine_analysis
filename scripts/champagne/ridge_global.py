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
   - Loads a structured sensory dataset (`test.csv`) with wine codes, tasters, and 100+ attributes.
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
if __name__ == "__main__":
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

    # --- Parameters ---
    N_DECIMATION = 10
    N_REPEATS = 10
    TEST_SIZE = 0.2
    RANDOM_SEED = 42
    directory = "/home/luiscamara/Documents/datasets/Champagnes/HETEROCYC"
    # directory = "/home/luiscamara/Documents/datasets/Champagnes/DMS"
    column_indices = None  # or specify which columns to use
    row_start, row_end = 0, None  # if you want to trim rows

    # --- Load chromatograms ---
    row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(directory)
    column_indices = list(range(fc_idx_1, lc_idx_1 + 1))
    data_dict = utils.load_ms_csv_data_from_directories(directory, column_indices, row_start, row_end)

    # --- Load metadata ---
    metadata = pd.read_csv("/home/luiscamara/Documents/datasets/Champagnes/test.csv", skiprows=1)
    metadata = metadata.iloc[1:]  # Remove extra header row
    metadata.columns = [col.strip().lower() for col in metadata.columns]  # Clean headers
    metadata.drop(columns=[col for col in metadata.columns if 'unnamed' in col.lower()], inplace=True)

    # --- Average duplicates ---
    # Convert sensory columns to numeric
    known_metadata = ['code vin', 'taster', 'prod area', 'variety', 'cave', 'age']
    sensory_cols = [col for col in metadata.columns if col not in known_metadata and pd.api.types.is_numeric_dtype(metadata[col])]
    metadata[sensory_cols] = metadata[sensory_cols].apply(pd.to_numeric, errors='coerce')

    # Group by code vin and taster and average
    metadata = metadata.groupby(['code vin', 'taster'], as_index=False)[sensory_cols].mean()

    # --- Build input (X) and output (y) ---
    X_raw = []
    y = []
    taster_ids = []
    sample_ids = []
    skipped_count = 0

    for _, row in metadata.iterrows():
        sample_id = row['code vin']
        taster_id = row['taster']
        try:
            attributes = row[sensory_cols].astype(float).values
        except:
            continue  # skip row if attribute values can't be converted

        # Find all replicate keys in the chromatogram dict that start with the sample_id
        replicate_keys = [k for k in data_dict if k.startswith(sample_id)]

        if not replicate_keys:
            skipped_count += 1
            print(f"Warning: No chromatograms found for sample {sample_id}")
            continue  # <-- SKIP if no chromatograms found

        # Stack and average the decimated chromatograms
        replicates = np.array([data_dict[k][::N_DECIMATION] for k in replicate_keys])
        chromatogram = np.mean(replicates, axis=0).flatten()
        chromatogram = np.nan_to_num(chromatogram, nan=0.0)

        X_raw.append(chromatogram)
        y.append(attributes)
        taster_ids.append(taster_id)
        sample_ids.append(sample_id)


    print(f"\nTotal samples skipped due to missing chromatograms: {skipped_count}")

    X_raw = np.array(X_raw)
    y = np.array(y)

    encoder = OneHotEncoder(sparse_output=False)
    taster_onehot = encoder.fit_transform(np.array(taster_ids).reshape(-1, 1))
    X_input = np.concatenate([X_raw, taster_onehot], axis=1)

    mask = ~np.isnan(X_input).any(axis=1) & ~np.isnan(y).any(axis=1)
    X_input = X_input[mask]
    y = y[mask]
    taster_ids = np.array(taster_ids)[mask]
    sample_ids = np.array(sample_ids)[mask]
    print(f"Removed {np.sum(~mask)} samples with NaNs.")

    all_mae = []
    all_rmse = []
    taster_mae_summary = defaultdict(list)
    last_y_test, last_y_pred, last_sample_ids, last_taster_ids = None, None, None, None
    saved_model_coefs = []

    for repeat in range(N_REPEATS):
        X_train, X_test, y_train, y_test, t_train, t_test, s_train, s_test = train_test_split(
            X_input, y, taster_ids, sample_ids, test_size=TEST_SIZE, random_state=RANDOM_SEED + repeat)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        saved_model_coefs.append(model.coef_.copy())

        mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
        rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

        all_mae.append(mae)
        all_rmse.append(rmse)

        last_y_test, last_y_pred = y_test, y_pred
        last_sample_ids, last_taster_ids = s_test, t_test

        for i, t in enumerate(t_test):
            abs_error = np.abs(y_test[i] - y_pred[i])
            taster_mae_summary[t].append(abs_error)

    from collections import Counter

    wine_counts = Counter(last_sample_ids)
    multi_taster_wines = [wine for wine, count in wine_counts.items() if count > 1]


    def plot_wine_across_tasters(y_true, y_pred, sample_ids, taster_ids, wine_code, sensory_cols):
        import matplotlib.pyplot as plt
        import numpy as np

        indices = np.where(sample_ids == wine_code)[0]
        if len(indices) < 2:
            print(f"Wine {wine_code} is not rated by multiple tasters in the test set.")
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
    if multi_taster_wines:
        wine_code = multi_taster_wines[0]
        print(f"\nShowing profiles for wine {wine_code} across tasters:")
        plot_wine_across_tasters(
            y_true=last_y_test,
            y_pred=last_y_pred,
            sample_ids=last_sample_ids,
            taster_ids=last_taster_ids,
            wine_code=wine_code,
            sensory_cols=sensory_cols
        )
    else:
        print("No wine in the test set was rated by multiple tasters.")

    mean_mae = np.mean(all_mae, axis=0)
    mean_rmse = np.mean(all_rmse, axis=0)
    rmse_pct = mean_rmse

    mean_coef = np.mean(saved_model_coefs, axis=0)  # shape (n_outputs, n_features)

    # Separate chromatogram and taster weights
    n_chromatogram_features = X_raw.shape[1]
    n_taster_features = taster_onehot.shape[1]

    chromatogram_weights = mean_coef[:, :n_chromatogram_features]
    taster_weights = mean_coef[:, n_chromatogram_features:]

    def natural_key(t):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', t)]
    unique_tasters = sorted(set(taster_ids), key=natural_key)

    print("\nPer-attribute average errors over multiple splits:")
    for i, col in enumerate(sensory_cols):
        print(f"{col:>12s}: MAE = {mean_mae[i]:5.2f}, RMSE = {mean_rmse[i]:5.2f} ({rmse_pct[i]:.1f}% of scale)")

    print(f"\nOverall RMSE across repeats: {np.sqrt(np.mean(mean_rmse**2)):.4f}")

    print("\nPer-taster average MAE (across all attributes):")
    for taster in sorted(taster_mae_summary.keys(), key=natural_key):
        all_errors = np.array(taster_mae_summary[taster])
        avg_mae_per_attr = np.mean(all_errors, axis=0)
        overall_avg = np.mean(avg_mae_per_attr)
        print(f"Taster {taster}: MAE = {overall_avg:.2f}")

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
                mae_i = np.mean(np.abs(y_true[idx] - y_pred[idx]))
                ax.set_title(f"{sample_ids[idx]}\n{mae_i:.2f}", fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for j in range(len(indices), n_cols):
                axes[row_idx, j].axis('off')

        for row_idx, taster in enumerate(unique_tasters):
            axes[row_idx, 0].set_ylabel(f"Taster {taster}", fontsize=9, rotation=0, labelpad=40)

        plt.suptitle("All predicted sensory profiles by taster (true in black, predicted in red)", fontsize=14)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show(block=False)

    print("\nPlotting all predicted sensory profiles grouped by taster (single figure)...")
    plot_profiles_grouped_by_taster(
        y_true=last_y_test,
        y_pred=last_y_pred,
        sample_ids=last_sample_ids,
        taster_ids=last_taster_ids,
        n_cols=10
    )
    plt.show()


    n_chem = X_raw.shape[1]
    tasters = list(encoder.categories_[0])
    n_tasters = len(tasters)
    taster_weights = mean_coef[:, n_chem:]  # shape: (n_outputs, n_tasters)

    # Transpose for plotting (tasters as rows, attributes as columns)
    taster_weights_T = taster_weights.T  # shape: (n_tasters, n_outputs)

    plt.figure(figsize=(12, 6))
    im = plt.imshow(taster_weights_T, cmap="bwr", aspect="auto", interpolation="nearest")
    plt.colorbar(im, label="Bias weight")
    plt.xticks(ticks=np.arange(len(sensory_cols)), labels=sensory_cols, rotation=90)
    plt.yticks(ticks=np.arange(len(tasters)), labels=tasters)
    plt.title("Taster-specific bias weights per sensory attribute")
    plt.xlabel("Sensory Attribute")
    plt.ylabel("Taster")
    plt.tight_layout()
    plt.show()


