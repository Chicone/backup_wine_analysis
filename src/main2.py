import numpy as np
import os
from data_loader import DataLoader
from dimensionality_reduction import DimensionalityReducer
from wine_analysis import WineAnalysis, ChromatogramAnalysis
from classification import Classifier
from visualizer import Visualizer, plot_all_acc_LDA, plot_stacked_chromatograms
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import matplotlib.pyplot as plt
from utils import normalize_dict, smooth_remove_peak
from scipy.ndimage import gaussian_filter

if __name__ == "__main__":
    n_splits = 100
    vintage = False
    # pca = True
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
        os.path.expanduser('~/PycharmProjects/wine_scheck/data/oak.npy'),
        # os.path.expanduser(basedir + '/2018 7 chateaux Oak Old vintages Masse 5.npy'),
        os.path.expanduser(basedir + '/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.npy')
        )
    file_name1 = cl.file_path1.split('/')[-1]
    file_name2 = cl.file_path2.split('/')[-1]

    analysis1 = WineAnalysis(os.path.expanduser(basedir + '/2018 7 chateaux Oak Old vintages Masse 5.npy'), normalize=False)
    chromatograms1 = analysis1.data_loader.data
    analysis2 = WineAnalysis(os.path.expanduser(basedir + '/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.npy'), normalize=False)
    chromatograms2 = analysis2.data_loader.data

    chromatograms1, chromatograms2 = cl.resample_chromatograms(chromatograms1, chromatograms2, start=100, length=30000)
    mean_c1 = cl.calculate_mean_chromatogram(chromatograms1)
    mean_c2 = cl.calculate_mean_chromatogram(chromatograms2)

    # sync_chrom = SyncChromatograms(
    #     smooth_remove_peak(mean_c1, peak_idx=8910, window_size=30),
    #     gaussian_filter(chromatograms2['B2000'], 5),
    #     1, np.linspace(0.997, 1.003, 30), 1E6, threshold=0.00, max_sep_threshold=50, peak_prominence=0.00
    # )
    # optimized_chrom = sync_chrom.adjust_chromatogram(local_sync=True, initial_lag=400)

    synced_chromatograms1 = cl.sync_individual_chromatograms(mean_c1, chromatograms1, np.linspace(0.997, 1.003, 30), local_sync=True, initial_lag=15)
    synced_chromatograms2 = cl.sync_individual_chromatograms(mean_c1, chromatograms2, np.linspace(0.980, 1.020, 80), local_sync=True, initial_lag=250)
    cut_length = min(min(len(lst) for lst in synced_chromatograms1.values()), min(len(lst) for lst in synced_chromatograms2.values()))
    synced_chromatograms1 = {key: value[:cut_length] for key, value in synced_chromatograms1.items()}
    synced_chromatograms2 = {key: value[:cut_length] for key, value in synced_chromatograms2.items()}
    norm_synced_chromatograms1 = normalize_dict(synced_chromatograms1, scaler='standard')
    norm_synced_chromatograms2 = normalize_dict(synced_chromatograms2, scaler='standard')
    norm_merged_chrom = cl.merge_chromatograms(norm_synced_chromatograms1, norm_synced_chromatograms2)
    # cl.stacked_2D_plots_3D(cl.merge_chromatograms(chromatograms1, chromatograms2))
    # cl.stacked_2D_plots_3D(cl.merge_chromatograms(synced_chromatograms1, synced_chromatograms2))
    # cl.umap_analysis(norm_merged_chrom, vintage, "Original data;", neigh_range=range(10, 61, 5), random_states=range(0, 97, 8))

    if not os.path.exists(npy_path):
        # Load .xlsx file and save into npy the chromatogram signal
        xlsx_loader = DataLoader(xlsx_path, normalize=False)
        np.save(os.path.expanduser(npy_path), xlsx_loader.data)


    # Instance of the class, load data, etc.
    analysis = WineAnalysis(npy_path, normalize=True)

    if pca:
        # PCA-reduce
        cls = Classifier(analysis.data, analysis.labels)  # to use _process_labels
        reducer = DimensionalityReducer(analysis.data)

        pca_dict, cumulative_variance, n_components = reducer.cumulative_variance(
             analysis.labels, variance_threshold=0.97, plot=True, dataset_name=analysis.chem_name)
        reducer.cross_validate_pca_classification(
            cls._process_labels(vintage=vintage), n_splits=n_splits, vthresh=0.97, test_size=None
        )
    else:
        analysis = WineAnalysis(npy_path, normalize=False)
        cls = Classifier(analysis.data, analysis.labels)  # to use _process_labels
        # Classification
        # analysis.train_classifier(classifier_type='LDA', vintage=vintage, n_splits=n_splits, test_size=None)
        # analysis.run_tsne()
        # analysis.run_umap(n_neighbors=5, random_state=90)
        # analysis.run_umap(n_neighbors=30, random_state=52) # 2022 oak old
        # analysis.run_umap(n_neighbors=70, random_state=84)  # 2018 oak old
        # run_tsne_and_evaluate(analysis.data, cls._process_labels(vintage), analysis.chem_name)
        best_perp, best_rs, best_score = run_tsne_and_evaluate(
            analysis,
            cls._process_labels(vintage),
            analysis.chem_name,
            perplexities=range(30, 60, 10),
            random_states=range(0, 64, 16)
        )
        analysis.run_tsne(perplexity=best_perp, random_state=best_rs, plot=True)

        best_neigh, best_rs, best_score = run_umap_and_evaluate(
            analysis,
            cls._process_labels(vintage),
            analysis.chem_name,
            neigh_range=range(30, 100, 10),
            random_states=range(0, 64, 16)
        )
        analysis.run_umap(n_neighbors=best_neigh, random_state=best_rs, plot=True)


    # # Example usage of DimensionalityReducer
    # data_loader = DataLoader('/home/luiscamara/PycharmProjects/wine_scheck/data/oak.npy')
    # reducer = DimensionalityReducer(data_loader.get_standardized_data())
    # pca_result = reducer.pca(components=80)
    # tsne_result = reducer.tsne()
    # umap_result = reducer.umap()

    # # pca_dict = reducer.perform_pca_on_dict(analysis.labels, n_components=n_components)
    # analysis = WineAnalysis(data_dict=pca_dict)



