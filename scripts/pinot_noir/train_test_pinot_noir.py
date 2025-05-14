import numpy as np
import os
import yaml
from gcmswine.classification import Classifier
from gcmswine import utils
from gcmswine.wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from gcmswine.utils import string_to_latex_confusion_matrix, string_to_latex_confusion_matrix_modified

# # Use this function to convert the printed confusion matrix to a latex confusion matrix
# # Copy the matrix into data_str using """ """ and create the list of headers, then call the function
# headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
#             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
#             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
# string_to_latex_confusion_matrix_modified(data_str, headers)

# Load dataset paths from config.yaml
config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.yaml")
config_path = os.path.abspath(config_path)
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Parameters from config file
dataset_directories = config["datasets"]
selected_datasets = config["selected_datasets"]

# Check if any selected dataset contains "pinot"
if not any("pinot" in name.lower() for name in selected_datasets):
    raise ValueError(
        "The datasets selected in the config.yaml file do not seem to be compatible with this script. "
        "Please select at least one Pinot Noir dataset."
    )

feature_type = config["feature_type"]
classifier = config["classifier"]
num_splits = config["num_splits"]
normalize = config["normalize"]
n_decimation = config["n_decimation"]
sync_state = config["sync_state"]
region = config["region"]
wine_kind = config["wine_kind"]

# Create ChromatogramAnalysis instance for optional alignment
cl = ChromatogramAnalysis()

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

