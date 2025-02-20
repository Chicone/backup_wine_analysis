# Configuration for the Main Analysis Script

# Data Handling Parameters
DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/LLE_SCAN/"
CHEMICAL_NAME = "PINOT_NOIR_LLE_SCAN"
# DATA_DIRECTORY= "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/DLLME_SCAN/"
# CHEMICAL_NAME = 'PINOT_NOIR_DLLME_SCAN'
# DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/220322_Pinot_Noir_Tom_CDF/"
# CHEMICAL_NAME = 'PINOT_NOIR_CHANGINS_TOM'
# DATA_DIRECTORY = "/home/luisgcamara/Documents/datasets/3D_data/BORDEAUX_OAK_PAPER/OAK_WOOD/"
# CHEMICAL_NAME = 'BORDEAUX_OAK_PAPER_OAK_WOOD'
# DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/PRESS_WINES/Esters22/CABERNET/"
# CHEMICAL_NAME = 'PRESS_WINES_ESTERS_2022_CS'

# Dataset Settings
ROW_START = 1
NUM_SPLITS = 5
CHROM_CAP = 29000  # Limit for chromatogram size
N_DECIMATION = 1  # Decimation factor for 3D data
VINTAGE = False  # Include vintage data in analysis
WINDOW = 1000
STRIDE = 200

# Analysis parameters
DATA_TYPE = "TIC"  # Options: "TIC", "TIS", "TIC-TIS", "GCMS"
CH_TREAT = 'concatenated'  # 'independent',  'concatenated'
CHANNEL_METHOD = 'greedy_remove' # greedy_remove, greedy, all_channels, individual
SYNC_STATE = False  # Use retention time alignment
CONCATENATE_TICS = False
CNN_DIM = None  # 1, 2, None
gcms_options = ["RT_DIRECTION", "MS_DIRECTION"]
GCMS_DIRECTION = gcms_options[0]
NUM_AGGR_CHANNELS = 1
DELAY = 0

# PCA and Classification
PCA_STATE = [False]  # Enable PCA for classification

# CNN Parameters
CROP_SIZE = (500, 128)  # Size of crops for 2D CNN
CROP_STRIDE = (150, 128)  # Overlapping stride for crops
BATCH_SIZE = 64
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001
NCONV = 1 # Number of 1D convolutional layers
MULTICHANNEL = True

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
