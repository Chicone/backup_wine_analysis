from config import (DATA_DIRECTORY, N_DECIMATION, WINE_KIND)
from classification import (Classifier)
from wine_analysis import GCMSDataProcessor
import utils
import numpy as np

N_DECIMATION = 5
DATA_TYPE = "TIS"

# Set rows in columns to read
row_start = 1
row_end, fc_idx, lc_idx = utils.find_data_margins_in_csv(DATA_DIRECTORY)
column_indices = list(range(fc_idx, lc_idx + 1))
data_dict = utils.load_ms_data_from_directories(DATA_DIRECTORY, column_indices, row_start, row_end)
min_length = min(array.shape[0] for array in data_dict.values())
gcms = GCMSDataProcessor(data_dict)

chromatograms = gcms.compute_tiss()
data = chromatograms.values()
labels = chromatograms.keys()

cls_type = 'RGC'
alpha = 1
cls = Classifier(
            np.array(list(data)), np.array(list(labels)), classifier_type=cls_type, wine_kind=WINE_KIND,
            alpha=alpha,
        )

classif_res = cls.train_and_evaluate_balanced(
    num_outer_repeats=10,  # Repeat the outer stratified split 5 times.
    n_inner_repeats=50,  # Use 20 inner CV repeats per outer repetition.
    random_seed=42,
    test_size=0.2,
    normalize=True,  # Enable normalization.
    scaler_type='standard',
    use_pca=False,  # Apply PCA.
    vthresh=0.95,  # Variance threshold for PCA.
    region='winery'  # Set region to 'winery' for a custom confusion matrix order.
)