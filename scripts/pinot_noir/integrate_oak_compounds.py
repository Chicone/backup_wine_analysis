
import numpy as np
import pandas as pd
import os
import yaml
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis
from gcmswine.logger_setup import logger, logger_raw


# Example compound table (fill with your handwritten note)
COMPOUNDS = [
    {"name": "furfural",        "mz": 96,  "rt_start": 11.06, "rt_stop": 11.41},
    {"name": "5-methylfurfural","mz": 110, "rt_start": 14.06, "rt_stop": 14.33},
    {"name": "5-hydroxymethylfurfural",           "mz": 97,  "rt_start": 16.63, "rt_stop": 16.88},
    {"name": "cis-oak_lactone", "mz": 99,  "rt_start": 22.439,"rt_stop": 22.725},
    {"name": "trans-oak_lactone","mz": 99, "rt_start": 24.119,"rt_stop": 24.401},
    {"name": "guaiacol",        "mz": 124, "rt_start": 21.727,"rt_stop": 22.004},
    {"name": "eugenol",         "mz": 164, "rt_start": 29.241,"rt_stop": 29.545},
    {"name": "isoeugenol",      "mz": 164, "rt_start": 29.40, "rt_stop": 29.55},  # approx shoulder
    {"name": "vanilline",       "mz": 151, "rt_start": 38.826,"rt_stop": 39.697},
    {"name": "acetovanillone",  "mz": 166, "rt_start": 38.822,"rt_stop": 39.496},
    {"name": "syringaldehyde",  "mz": 182, "rt_start": 45.349,"rt_stop": 46.080},
    {"name": "syringol",        "mz": 154, "rt_start": 31.361,"rt_stop": 31.647},
]

def integrate_peak_by_index(intensity, start_idx, stop_idx):
    """Integrate peak area by summing intensities between indices."""
    if start_idx >= len(intensity) or stop_idx < 0:
        return 0.0
    start_idx = max(0, start_idx)
    stop_idx = min(len(intensity)-1, stop_idx)
    return np.sum(intensity[start_idx:stop_idx+1])

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
    config_path = os.path.abspath(config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_gc_ms_dataset(directory):
    """
    Load all samples from a GC–MS dataset folder (.D subdirectories containing CSVs).

    Parameters
    ----------
    directory : str
        Path to dataset folder containing .D subdirectories.

    Returns
    -------
    data_dict : dict
        Mapping sample_id -> chromatogram matrix (T × C).
    sample_names : list
        List of sample IDs.
    rt_axis : np.ndarray
        Retention times (length T, same for all samples).
    mz_values : list
        List of m/z channels corresponding to matrix columns.
    """
    row_start = 1

    # Find where intensity columns start/end
    row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(directory)
    column_indices = list(range(fc_idx, lc_idx + 1))

    # Load chromatograms for all samples in this dataset directory
    data_dict = utils.load_ms_csv_data_from_directories(
        directory, column_indices, row_start, row_end
    )

    # Use first .D/CSV to extract RT axis and m/z values
    first_d = [d for d in os.listdir(directory) if d.endswith(".D")][0]
    first_csv = [f for f in os.listdir(os.path.join(directory, first_d)) if f.endswith(".csv")][0]
    df = pd.read_csv(os.path.join(directory, first_d, first_csv))

    rt_axis = df.iloc[:, 0].values  # first column = RT
    mz_values = [int(c) for c in df.columns[fc_idx:lc_idx + 1]]

    # --- Align all samples to the same length ---
    min_len = min(arr.shape[0] for arr in data_dict.values())
    data_dict = {k: v[:min_len, :] for k, v in data_dict.items()}
    rt_axis = rt_axis[:min_len]

    sample_names = list(data_dict.keys())

    return data_dict, sample_names, rt_axis, mz_values


def load_and_prepare_data(config):
    """
    Load one GC–MS dataset (folder with .D subdirectories containing CSVs),
    return full natural retention time axis (no trimming, no decimation).

    Returns
    -------
    wine_kind : str
        Type of wine inferred from dataset name(s).
    gcms : GCMSDataProcessor
        Wrapper for data_dict.
    data_dict : dict
        Mapping sample_id -> chromatogram matrix (T × C).
    dataset_origins : dict
        Mapping sample_id -> dataset name.
    rt_axis : np.ndarray
        Full retention times (length T).
    mz_values : list
        List of m/z channels corresponding to matrix columns.
    """
    dataset_directories = config["datasets"]
    selected_datasets = config["selected_datasets"]

    if len(selected_datasets) != 1:
        raise ValueError("This simplified loader only supports one dataset at a time.")

    dataset_name = selected_datasets[0]
    dataset_path = dataset_directories[dataset_name]

    wine_kind = utils.infer_wine_kind(selected_datasets, dataset_directories)

    # Load one dataset (all samples in the folder, no trimming)
    data_dict, sample_names, rt_axis, mz_values = load_gc_ms_dataset(dataset_path)

    # Ensure RT is in minutes
    rt_axis = rt_axis / 60000.0

    # Remove zero-variance channels
    data_dict, _ = utils.remove_zero_variance_channels(data_dict)

    gcms = GCMSDataProcessor(data_dict)

    return wine_kind, gcms, data_dict, sample_names, rt_axis, mz_values

def quantify_compounds(data_dict, rt_axis, mz_values, compounds):
    """
    Quantify defined compounds by trapezoidal integration.

    Parameters
    ----------
    data_dict : dict
        {sample_name -> chromatogram matrix (T × C)}.
    rt_axis : np.ndarray
        Retention times in minutes (length T).
    mz_values : list[int]
        List of m/z values corresponding to matrix columns.
    compounds : list[dict]
        Each dict must have:
            - "name" : compound name
            - "mz"   : target m/z (int)
            - "rt_start" : start RT [min]
            - "rt_stop"  : stop RT [min]

    Returns
    -------
    pd.DataFrame
        Long-format table with columns:
        ["Sample", "Compound", "Start Time [min]", "Stop Time [min]", "m/z", "Area"]
    """
    results = []

    for sample, matrix in data_dict.items():
        for cmpd in compounds:
            name = cmpd["name"]
            mz = cmpd["mz"]
            rt_start = cmpd.get("rt_start")
            rt_stop = cmpd.get("rt_stop")

            # Find column for this m/z
            try:
                mz_idx = mz_values.index(mz)
            except ValueError:
                area = np.nan
            else:
                y = matrix[:, mz_idx]

                if rt_start is not None and rt_stop is not None:
                    mask = (rt_axis >= rt_start) & (rt_axis <= rt_stop)
                    if np.any(mask):
                        area = np.trapz(y[mask], rt_axis[mask])
                    else:
                        area = 0.0
                else:
                    area = np.nan

            results.append({
                "Sample": sample,
                "Compound": name,
                "Start Time [min]": rt_start,
                "Stop Time [min]": rt_stop,
                "m/z": mz,
                "Area": area
            })

    return pd.DataFrame(results, columns=["Sample", "Compound", "Start Time [min]", "Stop Time [min]", "m/z", "Area"])


def dict_to_array3d(d):
    """Stack per-sample 2D chromatograms into (N, T, C)."""
    arrs = []
    for v in d.values():
        v = np.asarray(v)
        if v.ndim == 1:
            raise ValueError("sotf_mz requires per-channel data (time × channels) per sample.")
        arrs.append(v)
    return np.stack(arrs, axis=0)


if __name__ == "__main__":
    # Load your GC–MS data
    config = load_config()
    wine_kind, gcms, data_dict, sample_names, rt_axis, mz_values = load_and_prepare_data(config)

    # Quantify compounds
    df = quantify_compounds(data_dict, rt_axis, mz_values, COMPOUNDS)

    # Save results
    df.to_csv("integrated_peaks.csv", index=False)
    print("✅ Saved integrated peak areas to integrated_peaks.csv")