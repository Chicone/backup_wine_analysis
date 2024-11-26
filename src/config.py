# Configuration for the Main Analysis Script

# Data Handling Parameters
DATA_DIRECTORY = "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/LLE_SCAN/"
CHEMICAL_NAME = "PINOT_NOIR_LLE_SCAN"
# DATA_DIRECTORY= "/home/luiscamara/Documents/datasets/3D_data/PINOT_NOIR/LLE_SCAN/"
# CHEMICAL_NAME = 'PINOT_NOIR_LLE_SCAN'
# main_directory = "/home/luisgcamara/Documents/datasets/3D_data/PINOT_NOIR/DLLME_SCAN/"
# chem_name = 'PINOT_NOIR_LLE_SCAN'
# main_directory = "/home/luisgcamara/Documents/datasets/3D_data/BORDEAUX_OAK_PAPER/OAK_WOOD/"
# chem_name = 'BORDEAUX_OAK_PAPER_OAK_WOOD'

# Dataset Settings
ROW_START = 1
N_SPLITS = 500
CHROM_CAP = 25000  # Limit for chromatogram size
N_DECIMATION = 5  # Decimation factor for 3D data
VINTAGE = False  # Include vintage data in analysis
WINDOW = 1000
STRIDE = 200

# Analysis parameters
DATA_TYPE = "GCMS"  # Options: "TIC", "TIS", "TIC-TIS", "GCMS"
SYNC_STATE = True  # Use retention time alignment
CONCATENATE_TICS = False
CNN_DIM = None
gcms_options = ["RT_DIRECTION", "MS_DIRECTION"]
GCMS_DIRECTION = gcms_options[0]
NUM_AGGR_CHANNELS = 3

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
WINE_KIND = "pinot_noir" if "pinot_noir" in CHEMICAL_NAME.lower() else "bordeaux"
REGION = "winery"
