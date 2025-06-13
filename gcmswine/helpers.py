import numpy as np
import pandas as pd
from gcmswine import utils  # Your custom module
import os
import yaml
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

try:
    from xgboost import XGBRegressor
    xgb_available = True
except ImportError:
    xgb_available = False
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict, GroupKFold


def load_config(config_path=None):
    if config_path is None:
        raise ValueError("You must provide the config path when calling load_config().")
    with open(os.path.abspath(config_path), "r") as f:
        return yaml.safe_load(f)

def get_model(name, random_seed=42):
    name = name.lower()
    if name == "ridge":
        return Ridge()
    elif name == "lasso":
        return Lasso()
    elif name == "elasticnet":
        return ElasticNet()
    elif name == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=random_seed)
    elif name == "gbr":
        return GradientBoostingRegressor(random_state=random_seed)
    elif name == "hgb":
        return MultiOutputRegressor(HistGradientBoostingRegressor(random_state=random_seed))
    elif name == "svr":
        return MultiOutputRegressor(SVR())
    elif name == "knn":
        return KNeighborsRegressor()
    elif name == "dt":
        return DecisionTreeRegressor(random_state=random_seed)
    elif name == "xgb" and xgb_available:
        return XGBRegressor(random_state=random_seed)
    else:
        raise ValueError(f"Unsupported or unavailable model: {name}")

def load_chromatograms_decimated(directory, row_start, row_end, N_DECIMATION, retention_time_range=None):
    row_end_1, fc_idx_1, lc_idx_1 = utils.find_data_margins_in_csv(directory)
    column_indices = list(range(fc_idx_1, lc_idx_1 + 1))
    data_dict = utils.load_ms_csv_data_from_directories(directory, column_indices, row_start, row_end)
    chrom_length = len(list(data_dict.values())[0])

    for key in data_dict:
        data_dict[key] = data_dict[key][::N_DECIMATION]

    if retention_time_range:
        min_rt = retention_time_range['min'] // N_DECIMATION
        raw_max_rt = retention_time_range['max'] // N_DECIMATION
        max_rt = min(raw_max_rt, chrom_length // N_DECIMATION)
        data_dict = {key: value[min_rt:max_rt] for key, value in data_dict.items()}

    print(f"Chromatogram length: {chrom_length}")
    print(f"Chromatogram length after decimation: {chrom_length // N_DECIMATION}")
    return data_dict, chrom_length

def load_and_clean_metadata(path):
    df = pd.read_csv(path, skiprows=1)
    df = df.iloc[1:]  # Remove extra header row
    df.columns = [col.strip().lower() for col in df.columns]
    df.drop(columns=[col for col in df.columns if 'unnamed' in col.lower()], inplace=True)
    return df

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