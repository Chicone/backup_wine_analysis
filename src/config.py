# Configuration for the Main Analysis Script

# Data Handling Parameters
# ##### PINOT NOIR #####
DATASET_DIRECTORIES = {
    "pinot_noir_isvv_lle": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/ISVV/LLE_SCAN/",
    "pinot_noir_isvv_dllme": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/ISVV/DLLME_SCAN/",
    # "pinot_noir_isvv_lle_sim": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/ISVV/LLE_SIM/",
    # "pinot_noir_isvv_dllme_sim": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/ISVV/DLLME_SIM/",
    "pinot_noir_changins": "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/Changins/220322_Pinot_Noir_Tom_CDF/",
   }
# SELECTED_DATASETS = ["pinot_noir_isvv_lle"]
SELECTED_DATASETS = ["pinot_noir_changins"]

 ##### BORDEAUX #####
# DATASET_DIRECTORIES = {
#     "bordeaux_oak": "/home/luisgcamara/Documents/datasets/3D_data/BORDEAUX_OAK_PAPER/OAK_WOOD/"
# }

 ##### PRESS WINES #####
# DATASET_DIRECTORIES = {
#     "merlot_2021": "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters21/MERLOT/",
#     "merlot_2022": "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/MERLOT/",
#     "merlot_2023": "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters23/MERLOT/",
#     "cab_sauv_2021": "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters21/CABERNET/",
#     "cab_sauv_2022": "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/CABERNET/",
#     "cab_sauv_2023": "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters23/CABERNET/"
#    }

##### CHAMPAGNES #####
# DATASET_DIRECTORIES = {
#     "champagnes": "/home/luiscamara/Documents/datasets/Champagnes/",
#     }
# SELECTED_DATASETS = ["champagnes"]

ROW_START = 50
NUM_SPLITS = 200
N_DECIMATION = 10  # Decimation factor for 3D data
CHROM_CAP = 29000 // N_DECIMATION  # Limit for chromatogram size
VINTAGE = False  # Include vintage data in analysis
WINDOW = 1000
STRIDE = 200

# Analysis parameters
# DATA_TYPE = "GCMS"  # Options: "TIC", "TIS", "TIC-TIS", "GCMS"
CH_TREAT = 'concatenated'  # 'independent',  'concatenated'

# independent, individual, all_channels, greedy_add_ranked, greedy_add, greedy_remove_ranked, greedy_remove,
# greedy_remove_batch, random_subset
CHANNEL_METHOD = 'all_channels'
FEATURE_TYPE = 'tic_tis'  # concatenated tic tis tic_tis
CLASSIFIER='RGC' # "DTC", "GNB", "KNN", "LDA", "LR", "PAC", "PER", "RFC", "RGC", "SGD", "SVM"
SYNC_STATE = False  # Use retention time alignment
NORMALIZE = True
CONCATENATE_TICS = False
CNN_DIM = None  # 1, 2, None
gcms_options = ["RT_DIRECTION", "MS_DIRECTION"]
GCMS_DIRECTION = gcms_options[0]
NUM_AGGR_CHANNELS = 1
DELAY = 0
CLASS_BY_YEAR = False

# PCA and Classification
PCA_STATE = [False]  # Enable PCA for classification

# Region and Labels
# WINE_KIND = (
#     "pinot_noir" if "pinot_noir" in CHEMICAL_NAME.lower() else
#     "press" if "press" in CHEMICAL_NAME.lower() else
#     "bordeaux"
# )
# REGION = "beaume"
REGION = "winery"

# Bayesian Optimization
BAYES_OPTIMIZE = False
NUM_SPLITS_BAYES = 10
BAYES_CALLS = 50
