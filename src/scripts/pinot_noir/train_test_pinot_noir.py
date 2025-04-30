from classification import Classifier
import utils
import numpy as np
from wine_analysis import GCMSDataProcessor, ChromatogramAnalysis, process_labels_by_wine_kind
from utils import string_to_latex_confusion_matrix

# # use this function to convert the printed confusion matrix to a latex confusion matrix
# # copy the matrix into data_str using """ """
# headers = ['Clos Des Mouches', 'Vigne Enfant J.', 'Les Cailles', 'Bressandes Jadot', 'Les Petits Monts',
#             'Les Boudots', 'Schlumberger', 'Jean Sipp', 'Weinbach', 'Brunner', 'Vin des Croisés',
#             'Villard et Fils', 'République', 'Maladaires', 'Marimar', 'Drouhin']
# string_to_latex_confusion_matrix(data_str, headers)

DATASET_DIRECTORIES = {
    "pinot_noir_isvv_lle": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/ISVV/LLE_SCAN/",
    "pinot_noir_isvv_dllme": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/ISVV/DLLME_SCAN/",
    "pinot_noir_changins": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/Changins/220322_Pinot_Noir_Tom_CDF/",
   }
SELECTED_DATASETS = ["pinot_noir_changins"]
FEATURE_TYPE = 'tic_tis' # 'tic', 'tis', 'tic_tis'
CLASSIFIER = 'RGC'
NUM_SPLITS = 200
NORMALIZE = True
N_DECIMATION = 10
SYNC_STATE = False
REGION = 'winery'
WINE_KIND = 'pinot_noir'

cl = ChromatogramAnalysis()

# Load dataset
selected_paths = {name: DATASET_DIRECTORIES[name] for name in SELECTED_DATASETS}
data_dict, dataset_origins = utils.join_datasets(SELECTED_DATASETS, DATASET_DIRECTORIES, N_DECIMATION)
data_dict, _ = utils.remove_zero_variance_channels(data_dict)

chrom_length = len(list(data_dict.values())[0])
gcms = GCMSDataProcessor(data_dict)

if SYNC_STATE:
    tics, data_dict = cl.align_tics(data_dict, gcms, chrom_cap=30000)

data, labels = np.array(list(gcms.data.values())), np.array(list(gcms.data.keys()))
labels, year_labels = process_labels_by_wine_kind(labels, WINE_KIND, REGION, None, None, None)

cls = Classifier(np.array(list(data)), np.array(list(labels)), classifier_type=CLASSIFIER, wine_kind=WINE_KIND, year_labels=np.array(year_labels))

# Train and evaluate on all channels. FEATURE_TYPE decides how to aggregate channels
cls.train_and_evaluate_all_channels(
    num_repeats=NUM_SPLITS,
    num_outer_repeats=1,
    random_seed=42,
    test_size=0.2,
    normalize=NORMALIZE,
    scaler_type='standard',
    use_pca=False,
    vthresh=0.97,
    region=REGION,
    print_results=True,
    n_jobs=20,
    feature_type=FEATURE_TYPE,
    classifier_type=CLASSIFIER,
    LOOPC=False
)

