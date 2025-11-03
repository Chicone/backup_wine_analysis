from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class RunConfig:
    # Core classification / analysis
    class_by_year: bool = False
    classifier: str = "RGC"
    feature_type: str = "tic_tis"
    region: str = "origin"
    normalize: bool = False
    num_repeats: int = 1
    num_splits: int = 1
    n_decimation: int = 1
    n_rt_bins: int = 25
    n_mz_bins: int = 1
    random_state: int = 42
    cv_type: str = "LOO"
    regressor: Optional[str] = None

    # Plotting / visualization
    plot_projection: bool = False
    projection_method: str = "UMAP"
    projection_source: str | bool = False
    projection_dim: int = 2
    n_neighbors: int = 30
    color_by_country: bool = False
    color_by_origin: bool = False
    color_by_winery: bool = False
    show_sample_names: bool = False
    invert_x: bool = False
    invert_y: bool = False
    rot_axes: bool = False
    sample_display_mode: str = "dots"
    exclude_us: bool = False
    density_plot: bool = False
    show_pred_plot: bool = False
    pred_plot_region: str = "all"
    pred_plot_mode: str = "regression"
    show_confusion_matrix: bool = False
    plot_all_tests: bool = False
    plot_r2: bool = False
    plot_regress_corr: bool = False
    plot_rt_bin_analysis: bool = False
    rt_analysis_filename: Optional[str] = None
    plot_shap: bool = False
    plot_umap: str | bool = False
    show_chromatograms: bool = False
    show_age_histogram: bool = False
    show_sample_names_flag: bool = False  # (dup in yaml as "show_sample_names")
    show_predicted_profiles: bool = False
    show_champ_predicted_profiles: str | bool = False

    # Special analysis flags
    sotf_ret_time: bool = False
    sotf_mz: bool = False
    sotf_remove_2d: bool = False
    sotf_add_2d: bool = False
    reg_acc_map: bool = False
    oak_analysis: bool = False
    oak_mode: Optional[str] = None
    oak_peaks_path: Optional[Path] = "/home/luiscamara/PycharmProjects/wine_analysis/scripts/pinot_noir/changins_integrated_peaks.csv"
    reduce_dims: bool = False
    reduction_dims: int = 2
    reduction_method: str = "pca"
    global_focus_heatmap: bool = False
    taster_focus_heatmap: bool = False
    taster_scaling: bool = False
    taster_vs_mean: bool = False
    test_average_scores: bool = False
    remove_avg_scores: bool = False
    do_classification: bool = False
    shuffle_labels: bool = False
    downsample_group: bool = False
    group_wines: bool = False
    constant_ohe: bool = False

    # Data / labels
    selected_datasets: Optional[List[str]] = None
    datasets: Optional[Dict[str, str]] = None
    label_target: Optional[str] = None
    label_targets: Optional[List[str]] = None
    selected_attribute: Optional[str] = None

    # Ranges
    rt_range: Optional[Dict[str, Any]] = None
    retention_time_range: Optional[Dict[str, Any]] = None

    # UMAP specific
    umap_dim: int = 2
    umap_source: str | bool = False

    # Sync / state
    sync_state: bool = False

    # Extra catch-all for any future keys
    extra: Dict[str, Any] = field(default_factory=dict)


def build_run_config(config: dict) -> RunConfig:
    """Build RunConfig from a raw config dict, filtering + validating keys."""
    allowed = RunConfig.__dataclass_fields__.keys()
    filtered = {k: v for k, v in config.items() if k in allowed}
    extras = {k: v for k, v in config.items() if k not in allowed}

    rc = RunConfig(**filtered)
    rc.extra = extras

    # Special handling for oak_mode
    if rc.oak_mode == "integrated" and rc.oak_peaks_path is None:
        rc.oak_peaks_path = Path(__file__).resolve().parent / "changins_integrated_peaks.csv"

    # Preserve default "color-by" behavior
    if not rc.color_by_origin and not rc.color_by_winery:
        if rc.region == "origin":
            rc.color_by_origin = True
        elif rc.region == "winery":
            rc.color_by_winery = True

    # Validate sotf flags
    if rc.sotf_ret_time and rc.sotf_mz:
        raise ValueError("Only one of 'sotf_ret_time' or 'sotf_mz' can be True.")

    return rc
