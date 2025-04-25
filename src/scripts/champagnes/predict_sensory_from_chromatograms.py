import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import utils  # Your custom module
from collections import defaultdict
import matplotlib.pyplot as plt


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

for repeat in range(N_REPEATS):
    X_train, X_test, y_train, y_test, t_train, t_test, s_train, s_test = train_test_split(
        X_input, y, taster_ids, sample_ids, test_size=TEST_SIZE, random_state=RANDOM_SEED + repeat)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Ridge()
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    rmse = np.sqrt(mean_squared_error(y_test, y_pred, multioutput='raw_values'))

    all_mae.append(mae)
    all_rmse.append(rmse)

    last_y_test, last_y_pred = y_test, y_pred
    last_sample_ids, last_taster_ids = s_test, t_test

    for i, t in enumerate(t_test):
        abs_error = np.abs(y_test[i] - y_pred[i])
        taster_mae_summary[t].append(abs_error)

mean_mae = np.mean(all_mae, axis=0)
mean_rmse = np.mean(all_rmse, axis=0)
rmse_pct = 100 * mean_rmse / 100

print("\nPer-attribute average errors over multiple splits:")
for i, col in enumerate(sensory_cols):
    print(f"{col:>12s}: MAE = {mean_mae[i]:5.2f}, RMSE = {mean_rmse[i]:5.2f} ({rmse_pct[i]:.1f}% of scale)")

print(f"\nOverall RMSE across repeats: {np.sqrt(np.mean(mean_rmse**2)):.4f}")

print("\nPer-taster average MAE (across all attributes):")
for taster, all_errors in taster_mae_summary.items():
    all_errors = np.array(all_errors)
    avg_mae_per_attr = np.mean(all_errors, axis=0)
    overall_avg = np.mean(avg_mae_per_attr)
    print(f"Taster {taster}: MAE = {overall_avg:.2f}")

def plot_profiles_for_taster(y_true, y_pred, sample_ids, taster_ids, taster, n_samples=50, n_cols=10):
    indices = np.where(taster_ids == taster)[0]
    if len(indices) == 0:
        print(f"No samples found for taster {taster}")
        return

    n_to_plot = min(n_samples, len(indices))
    n_rows = int(np.ceil(n_to_plot / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.2, n_rows * 2), sharey=True)
    axes = axes.flatten()

    for i in range(n_to_plot):
        idx = indices[i]
        ax = axes[i]
        ax.plot(y_true[idx], color='black', linewidth=1.5)
        ax.plot(y_pred[idx], color='red', linewidth=1)
        mae_i = np.mean(np.abs(y_true[idx] - y_pred[idx]))
        ax.set_title(f"{sample_ids[idx]}\n{mae_i:.2f}", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n_to_plot, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Taster {taster}: true in black, predicted in red", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

unique_tasters = sorted(set(last_taster_ids))

for taster in unique_tasters:
    print(f"\nPlotting profiles for Taster {taster}...")
    plot_profiles_for_taster(
        y_true=last_y_test,
        y_pred=last_y_pred,
        sample_ids=last_sample_ids,
        taster_ids=last_taster_ids,
        taster=taster,
        n_samples=50  # or use len(...) if you want to show all
    )

# plot_profiles_for_taster(last_y_test, last_y_pred, last_sample_ids, last_taster_ids, taster='D1', n_samples=50)

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
    plt.show()
plot_single_wine(last_y_test, last_y_pred, last_sample_ids, wine_code='141T-N-27')

