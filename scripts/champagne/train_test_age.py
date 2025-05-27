"""
Taster Score Classification using the script **train_test_labels.py**

This script trains and evaluates a Ridge classifier to predict categorical labels (e.g. taster identity, wine variety,
cave, age) based on averaged sensory evaluation scores of Champagne wines.
The features correspond to numerical scores given by tasters on different attributes (e.g. acid, balance),
and the labels are drawn from associated metadata.

Data is preprocessed by collapsing replicates per (wine, taster) pair and cleaning non-numeric values.
Stratified K-Fold cross-validation is repeated multiple times to obtain a robust estimate of classification accuracy,
and a normalized confusion matrix is plotted at the end for each classification target to visualize model performance
across classes.

"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm
import matplotlib.pyplot as plt
from gcmswine import utils  # Assumes this is your utility module for GC-MS data loading
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle


if __name__ == "__main__":
    # ------------------ Parameters ------------------
    directory = "/home/luiscamara/Documents/datasets/Champagnes/HETEROCYC"
    csv_path = "/home/luiscamara/Documents/datasets/Champagnes/test.csv"
    n_splits = 10
    n_repeats = 1
    random_seed = 42
    N_DECIMATION = 10
    CHROM_CAP = None
    row_start, row_end = 0, None
    n_bins = 5  # for stratification
    year_tolerance = 3  # years of tolerance for relaxed accuracy

    # ------------------ Load chromatograms ------------------
    row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(directory)
    column_indices = list(range(fc_idx_1, lc_idx_1 + 1))
    data_dict = utils.load_ms_csv_data_from_directories(directory, column_indices, row_start, row_end)
    if CHROM_CAP:
        data_dict = {key: value[:CHROM_CAP] for key, value in data_dict.items()}

    # ------------------ Load and clean metadata ------------------
    df = pd.read_csv(csv_path, skiprows=1)
    df = df.iloc[1:]
    df.columns = [col.strip().lower() for col in df.columns]
    df['age'] = pd.to_numeric(df['age'], errors='coerce')
    df = df.dropna(subset=['age'])

    # ------------------ Prepare samples ------------------
    X, y, groups = [], [], []

    for _, row in df.iterrows():
        code_vin = row['code vin']
        if pd.isna(code_vin):
            continue

        replicate_keys = [k for k in data_dict if k.startswith(code_vin)]
        if not replicate_keys:
            continue

        replicates = np.array([data_dict[k][::N_DECIMATION] for k in replicate_keys])
        chromatogram = np.mean(replicates, axis=0).flatten()
        chromatogram = np.nan_to_num(chromatogram)

        X.append(chromatogram)
        y.append(row['age'])
        groups.append(code_vin)

    X = np.array(X)
    y = np.array(y)
    groups = np.array(groups)

    # ------------------ Prepare unique wine stratification ------------------
    wine_df = pd.DataFrame({'code_vin': groups, 'age': y})
    wine_df_unique = wine_df.drop_duplicates(subset='code_vin').copy()

    bin_encoder = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    wine_df_unique['age_bin'] = bin_encoder.fit_transform(wine_df_unique[['age']]).astype(int).flatten()

    # ------------------ Regression with Ridge ------------------
    mae_list, rmse_list, r2_list, acc_list, acc_tol_list = [], [], [], [], []
    y_true_all, y_pred_all = [], []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for train_idx, test_idx in skf.split(wine_df_unique['code_vin'], wine_df_unique['age_bin']):
        train_codes = wine_df_unique.iloc[train_idx]['code_vin'].values
        test_codes = wine_df_unique.iloc[test_idx]['code_vin'].values

        train_mask = np.isin(groups, train_codes)
        test_mask = np.isin(groups, test_codes)

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Ridge()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mae_list.append(mean_absolute_error(y_test, y_pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_list.append(r2_score(y_test, y_pred))

        acc = np.mean(np.round(y_pred) == np.round(y_test))
        acc_tol = np.mean(np.abs(np.round(y_pred) - np.round(y_test)) <= year_tolerance)

        acc_list.append(acc)
        acc_tol_list.append(acc_tol)

        y_true_all.extend(y_test)
        y_pred_all.extend(y_pred)

    # ------------------ Report performance ------------------
    print("\nCross-validated Ridge Regression Performance:")
    print(f"MAE:  {np.mean(mae_list):.2f} ± {np.std(mae_list):.2f}")
    print(f"RMSE: {np.mean(rmse_list):.2f} ± {np.std(rmse_list):.2f}")
    print(f"R²:   {np.mean(r2_list):.3f} ± {np.std(r2_list):.3f}")
    print(f"Rounded Accuracy (Exact): {np.mean(acc_list):.3f} ± {np.std(acc_list):.3f}")
    print(f"Rounded Accuracy (±{year_tolerance} yrs): {np.mean(acc_tol_list):.3f} ± {np.std(acc_tol_list):.3f}")

    # ------------------ Plot metrics distribution ------------------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(mae_list, bins=10, color="skyblue", edgecolor="k")
    plt.title("MAE Distribution")
    plt.xlabel("MAE")

    plt.subplot(1, 3, 2)
    plt.hist(r2_list, bins=10, color="salmon", edgecolor="k")
    plt.title("R² Distribution")
    plt.xlabel("R²")

    plt.subplot(1, 3, 3)
    plt.hist(acc_tol_list, bins=10, color="lightgreen", edgecolor="k")
    plt.title(f"Accuracy (±{year_tolerance} yrs)")
    plt.xlabel("Accuracy")
    plt.tight_layout()
    plt.show()

    # ------------------ Confusion Matrix ------------------
    y_true_arr = np.round(np.array(y_true_all)).astype(int)
    y_pred_arr = np.round(np.array(y_pred_all)).astype(int)
    unique_labels = np.unique(np.concatenate((y_true_arr, y_pred_arr)))
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=unique_labels, normalize='true')

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_labels)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion Matrix (Rounded Age)")
    plt.tight_layout()
    plt.show()

    # ------------------ Extra: Age Distribution & Scatter ------------------
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(y, bins=np.arange(np.floor(y.min()), np.ceil(y.max()) + 1), color="gray", edgecolor="black")
    plt.title("Age Distribution")
    plt.xlabel("True Age")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.scatter(y_true_all, y_pred_all, alpha=0.4, edgecolors='k')
    plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--', lw=2)
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("True vs Predicted Age")
    plt.grid(True)
    plt.tight_layout()
    plt.show()