# Configuration for the Main Analysis Script

# Data Handling Parameters
# ##### PINOT NOIR #####
# DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/LLE_SCAN/"
# CHEMICAL_NAME = "PINOT_NOIR_LLE_SCAN"
# DATA_DIRECTORY= "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/DLLME_SCAN/"
# CHEMICAL_NAME = 'PINOT_NOIR_DLLME_SCAN'
# DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/220322_Pinot_Noir_Tom_CDF/"
# CHEMICAL_NAME = 'PINOT_NOIR_CHANGINS_TOM'

 ##### BORDEAUX #####
# DATA_DIRECTORY = "/home/luisgcamara/Documents/datasets/3D_data/BORDEAUX_OAK_PAPER/OAK_WOOD/"
# CHEMICAL_NAME = 'BORDEAUX_OAK_PAPER_OAK_WOOD'

 ##### PRESS WINES #####
DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/MERLOT/"
CHEMICAL_NAME = 'PRESS_WINES_ESTERS_2022_M'
# DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/CABERNET/"
# CHEMICAL_NAME = 'PRESS_WINES_ESTERS_2022_CS'
DATA_DIRECTORY_2 = "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/CABERNET/"
CHEMICAL_NAME_2 = 'PRESS_WINES_ESTERS_2022_CS'
# DATA_DIRECTORY_2 = "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters23/MERLOT/"
# CHEMICAL_NAME_2 = 'PRESS_WINES_ESTERS_2023_M'
DATA_DIRECTORY_3 = "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters21/MERLOT/"
CHEMICAL_NAME_3 = 'PRESS_WINES_ESTERS_2021_M'


# Dataset Settings
JOIN_DATASETS=True
ROW_START = 1
NUM_SPLITS = 100
CHROM_CAP = 29000  # Limit for chromatogram size
N_DECIMATION = 5  # Decimation factor for 3D data
VINTAGE = False  # Include vintage data in analysis
WINDOW = 1000
STRIDE = 200

# Analysis parameters
DATA_TYPE = "TIC"  # Options: "TIC", "TIS", "TIC-TIS", "GCMS"
CH_TREAT = 'concatenated'  # 'independent',  'concatenated'
CHANNEL_METHOD = 'greedy_add' # greedy_remove, greedy_remove_batch, greedy_add, all_channels, greedy_ranked
FEATURE_TYPE = 'tic_tis'  # concatenated tic_tis
SYNC_STATE = False  # Use retention time alignment
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
WINE_KIND = (
    "pinot_noir" if "pinot_noir" in CHEMICAL_NAME.lower() else
    "press" if "press" in CHEMICAL_NAME.lower() else
    "bordeaux"
)
# REGION = "beaume"
REGION = "winery"

# Bayesian Optimization
BAYES_OPTIMIZE = False
NUM_SPLITS_BAYES = 10
BAYES_CALLS = 50
