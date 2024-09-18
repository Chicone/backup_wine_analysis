"""
Main Analysis Script Overview
==================================

This script serves as the primary entry point for analyzing wine-related chromatographic data. It integrates various
components of the `wine_analysis` framework, including data loading, dimensionality reduction, synchronization, and
classification.

Key Features:
-------------
- **Data Loading**: Loads and normalizes chromatographic data from `.npy` or `.xlsx` files using the `DataLoader` class.
- **Chromatogram Analysis**: Uses the `ChromatogramAnalysis` class to resample, synchronize, and merge chromatograms from different datasets.
- **Dimensionality Reduction**: Applies techniques such as PCA, t-SNE, and UMAP to reduce the dimensionality of the data for easier visualization and analysis.
- **Classification**: Implements various classification strategies using the `Classifier` class, including leave-one-out cross-validation and cross-dataset training and testing.
- **Visualization**: Generates visualizations of chromatograms, synchronization results, and dimensionality reduction outputs.

"""

import numpy as np
import os
from data_loader import DataLoader
from dimensionality_reduction import DimensionalityReducer
from wine_analysis import WineAnalysis, ChromatogramAnalysis
from classification import Classifier, process_labels
from visualizer import Visualizer, plot_all_acc_LDA, plot_stacked_chromatograms
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import matplotlib.pyplot as plt
import utils
from scipy.ndimage import gaussian_filter
from wine_analysis import SyncChromatograms

if __name__ == "__main__":
    n_splits = 200
    vintage = False
    pca = False

    # plot_all_acc_LDA(vintage=False)
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2018 7 chateaux Oak Old vintages Masse 5.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 4 new bordeaux Oak Masse 5 NORMALIZED 052022 SM2 .xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 11 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/older data/Datat for paper Sept 2022/2018 7 chateaux Oak Old vintages Masse 5.xlsx')
    xlsx_path = os.path.expanduser('~/PycharmProjects/wine_scheck/data/oak.npy')  #  not normalised
    npy_path = os.path.splitext(xlsx_path)[0] + '.npy'

    #  Comparison of oak from 2018 and 2022
    basedir = '~/Documents/datasets/BordeauxData/DataNov2022/'
    cl = ChromatogramAnalysis(
        # os.path.expanduser('~/PycharmProjects/wine_scheck/data/oak.npy'),
        os.path.expanduser(basedir + '/2018 7 chateaux Oak Old vintages Masse 5.npy'),
        os.path.expanduser(basedir + '/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.npy')
        )
    file_name1 = cl.file_path1.split('/')[-1]
    file_name2 = cl.file_path2.split('/')[-1]

    chromatograms1 = utils.load_chromatograms(basedir + file_name1)
    chromatograms2 = utils.load_chromatograms(basedir + file_name2)

    chromatograms1, chromatograms2 = cl.resample_chromatograms(chromatograms1, chromatograms2, start=100, length=30000)

    # chromatograms1 = utils.normalize_amplitude_dict(chromatograms1)
    # chromatograms2 = utils.normalize_amplitude_dict(chromatograms2)



    mean_c1 = cl.calculate_mean_chromatogram(chromatograms1)
    mean_c2 = cl.calculate_mean_chromatogram(chromatograms2)
    mean_c1 = utils.remove_peak(mean_c1, peak_idx=8910, window_size=30)
    sc_inst = SyncChromatograms(mean_c1, mean_c2, chromatograms2, 10, 1,)
    lag_res = sc_inst.calculate_lag_profile(
        mean_c1, mean_c2, 4000, lag_range=200, hop=2000, sigma=20, distance_metric='l1', init_min_dist=1E6)
    # utils.plot_lag(lag_res[0], lag_res[1])

    synced_chromatograms1 = cl.sync_individual_chromatograms(
        mean_c1, chromatograms1, np.linspace(0.997, 1.003, 30), initial_lag=50
    )
    synced_chromatograms2 = cl.sync_individual_chromatograms(
        mean_c1, chromatograms2, np.linspace(0.980, 1.020, 80), initial_lag=250
    )
    # synced_chromatograms1 = chromatograms1
    # synced_chromatograms2 = chromatograms2
    cut_length = min(
        min(len(lst) for lst in synced_chromatograms1.values()),
        min(len(lst) for lst in synced_chromatograms2.values())
    )
    synced_chromatograms1 = {key: value[:cut_length] for key, value in synced_chromatograms1.items()}
    synced_chromatograms2 = {key: value[:cut_length] for key, value in synced_chromatograms2.items()}
    norm_synced_chromatograms1 = utils.normalize_dict(synced_chromatograms1, scaler='standard')
    norm_synced_chromatograms2 = utils.normalize_dict(synced_chromatograms2, scaler='standard')
    norm_merged_chrom = cl.merge_chromatograms(norm_synced_chromatograms1, norm_synced_chromatograms2)
    # cl.stacked_2D_plots_3D(cl.merge_chromatograms({label: cl.min_max_normalize(chromatogram, 0, 1) for label, chromatogram in chromatograms1.items()}, {label: cl.min_max_normalize(chromatogram, 0, 1) for label, chromatogram in chromatograms2.items()}))
    # cl.stacked_2D_plots_3D(cl.merge_chromatograms(synced_chromatograms1, synced_chromatograms2))
    # cl.umap_analysis(norm_merged_chrom, vintage, "Original data;", neigh_range=range(10, 61, 5), random_states=range(0, 97, 8))
    # cl.plot_chromatograms(mean_c1, mean_c2, file_name1, file_name2, cl)


    if not os.path.exists(npy_path):
        # Load .xlsx file and save into npy the chromatogram signal
        xlsx_loader = DataLoader(xlsx_path, normalize=False)
        np.save(os.path.expanduser(npy_path), xlsx_loader.data)

    # Choose the chromatograms to analyze
    # data1 =chromatograms1.values()
    data1 = synced_chromatograms1.values()
    labels1 = synced_chromatograms1.keys()
    chem_name1 = os.path.splitext(file_name1)[0].split('/')[-1]
    # data2 = chromatograms2.values()
    data2 = synced_chromatograms2.values()
    labels2 = synced_chromatograms2.keys()
    chem_name2 = os.path.splitext(file_name2)[0].split('/')[-1]

    if not pca:
        # # Classification of individual datasets (leave-one-out)
        # print(f"Estimating LOO accuracy on dataset {chem_name1}...")
        # cls = Classifier(np.array(list(data1)), np.array(list(labels1)), classifier_type='LDA')
        # cls.train_and_evaluate(n_splits, vintage=vintage, test_size=None, normalize=True, scaler_type='standard')
        # print(f"Estimating LOO accuracy on dataset {chem_name2}...")
        # cls = Classifier(np.array(list(data2)), np.array(list(labels2)), classifier_type='LDA')
        # cls.train_and_evaluate(n_splits, vintage=vintage, test_size=None, normalize=True, scaler_type='standard')

        # Classification of one dataset training on the other
        print(f"Estimating cross-dataset accuracy...")
        # 'LDA', 'LR', 'RFC', 'PAC', 'PER', 'RGC', 'SGD', 'SVM', 'KNN', 'DTC', 'GNB', and 'GBC'.
        cls = Classifier(data1, labels1, classifier_type='LDA')
        cls.train_and_evaluate_separate_datasets(
            np.array(list(data1)), process_labels(labels1, vintage),
            np.array(list(data2)), np.array(list(labels2)),
            n_splits=200, normalize=True, scaler_type='standard'
        )
    else:
        # PCA-reduce
        cls = Classifier(data1, labels1)  # to use _process_labels
        reducer = DimensionalityReducer(data1)
        pca_dict, cumulative_variance, n_components = reducer.cumulative_variance(
              np.array(labels1), variance_threshold=0.97, plot=True, dataset_name=chem_name1
        )
        reducer.cross_validate_pca_classification(
            cls._process_labels(vintage=vintage), n_splits=n_splits, vthresh=0.97, test_size=None
        )

    # # Example usage of DimensionalityReducer
    # data_loader = DataLoader('/home/luiscamara/PycharmProjects/wine_scheck/data/oak.npy')
    # reducer = DimensionalityReducer(data_loader.get_standardized_data())
    # pca_result = reducer.pca(components=80)
    # tsne_result = reducer.tsne()
    # umap_result = reducer.umap()

    # # pca_dict = reducer.perform_pca_on_dict(analysis.labels, n_components=n_components)
    # analysis = WineAnalysis(data_dict=pca_dict)



