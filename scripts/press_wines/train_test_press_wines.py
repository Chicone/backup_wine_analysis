import numpy as np
import os
import yaml
from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified


# Load dataset paths from config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
config_path = os.path.abspath(config_path)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Parameters from config file
dataset_directories = config["datasets"]
selected_datasets = config["selected_datasets"]

# Get the paths corresponding to the selected datasets
selected_paths = [config["datasets"][name] for name in selected_datasets]

# Check if all selected dataset contains "pinot"
if not all("press_wines" in path.lower() for path in selected_paths):
    raise ValueError(
        "The datasets selected in the config.yaml file do not seem to be compatible with this script. "
        "At least one of the selected paths does not contain 'press_wines'."
    )

# Infer wine_kind from selected dataset paths
wine_kind = utils.infer_wine_kind(selected_datasets, config["datasets"])

feature_type = config["feature_type"]
classifier = config["classifier"]
num_splits = config["num_splits"]
normalize = config["normalize"]
n_decimation = config["n_decimation"]
sync_state = config["sync_state"]
region = config["region"]
# wine_kind = config["wine_kind"]

# Create ChromatogramAnalysis instance for optional alignment
cl = ChromatogramAnalysis(ndec=n_decimation)

# Load dataset, removing zero-variance channels
selected_paths = {name: dataset_directories[name] for name in selected_datasets}
data_dict, dataset_origins = utils.join_datasets(selected_datasets, dataset_directories, n_decimation)
data_dict, _ = utils.remove_zero_variance_channels(data_dict)
chrom_length = len(list(data_dict.values())[0])

gcms = GCMSDataProcessor(data_dict)

if sync_state:
    tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)

# Extract data matrix (samples × channels) and associated labels
data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
labels, year_labels = process_labels_by_wine_kind(labels, wine_kind, region, None, None, None)

# Instantiate classifier with data and labels
cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=classifier, wine_kind=wine_kind,
                 year_labels=np.array(year_labels))

# Train and evaluate on all channels. Parameter "feature_type" decides how to aggregate channels
cls.train_and_evaluate_all_channels(
    num_repeats=num_splits,
    random_seed=42,
    test_size=0.2,
    normalize=normalize,
    scaler_type='standard',
    use_pca=False,
    vthresh=0.97,
    region=region,
    print_results=True,
    n_jobs=20,
    feature_type=feature_type,
    classifier_type=classifier,
    LOOPC=True  # whether to use stratified splitting (False) or Leave One Out Per Class (True)
)
