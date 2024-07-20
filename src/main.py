import numpy as np
import os
from data_loader import DataLoader
from dimensionality_reduction import DimensionalityReducer
from wine_analysis import WineAnalysis, ChromatogramAnalysis
from classification import Classifier
from visualizer import Visualizer, plot_all_acc_LDA, plot_stacked_chromatograms
from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
import matplotlib.pyplot as plt

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
    xlsx_path = os.path.expanduser('~/PycharmProjects/wine_scheck/data/concat.npy')  #  not normalised
    npy_path = os.path.splitext(xlsx_path)[0] + '.npy'


    #  Comparison of oak from 2018 and 2022
    cl = ChromatogramAnalysis(
        os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2018 7 chateaux Oak Old vintages Masse 5.npy'),
        os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.npy')
        )
    file_name1 = cl.file_path1.split('/')[-1]
    file_name2 = cl.file_path2.split('/')[-1]

    chromatograms1 = cl.normalize_all_chromatograms(cl.load_chromatogram(cl.file_path1))
    chromatograms2 = cl.normalize_all_chromatograms(cl.load_chromatogram(cl.file_path2))
    chromatograms1, chromatograms2 = cl.resample_chromatograms(chromatograms1, chromatograms2, start=100)
    mean_c1 = cl.calculate_mean_chromatogram(chromatograms1)
    mean_c2 = cl.calculate_mean_chromatogram(chromatograms2)

    # # Define the range of scaling factors and lags to try
    scale_range = np.linspace(0.8, 1.2, 50)
    # lag_range = range(-100, 101)

    # Synchronize the chromatograms
    # aligned_chrom2, best_scale, best_lag = cl.sync_chromatograms(mean_c1, chromatograms2, scale_range, lag_range)


    # # Plot 3 chromatograms
    # cl.sync_and_plot_chromatograms(chromatograms1, chromatograms2, label_to_plot='F2005', extra_label='A2005')
    # cl.sync_and_plot_chromatograms(chromatograms1, chromatograms2, label_to_plot='G2000', extra_label='T2005')


    # # Perform UMAP analysis on merged chromatograms without synchronization
    # merged_chrom = cl.merge_chromatograms(chromatograms1, chromatograms2)
    # cl.tsne_analysis(merged_chrom, vintage, "Chromatograms without sync")
    # cl.umap_analysis(merged_chrom, vintage, "Chromatograms without sync")


    # # Perform UMAP analysis on mean-synchronized and scaled chromatograms
    # best_scale, best_lag, best_corr = cl.find_best_scale_and_lag(mean_c1[:5000], mean_c2[:5000], np.array((1,)), 500)
    # shifted_chromatograms2 = cl.shift_chromatograms(chromatograms2, best_lag)
    # scaled_chromatograms2 = cl.scale_chromatograms(shifted_chromatograms2, 0.998)
    # # scaled_chromatograms2 = cl.sync_and_scale_chromatograms(cl, chromatograms1, chromatograms2)
    # merged_chrom = cl.merge_chromatograms(chromatograms1, scaled_chromatograms2)
    # cl.umap_analysis(merged_chrom, vintage, "Chromatograms with sync and scale")

    # # Perform UMAP analysis on individually-synchronized and scaled chromatograms
    # # ref_peak_value, ref_peak_position = cl.find_highest_common_peak(chromatograms1, tolerance=10)
    # # ref_peak_value, ref_peak_position = cl.find_second_highest_common_peak(chromatograms1, tolerance=10)
    # synced_chromatograms1 = cl.sync_individual_chromatograms(mean_c1, chromatograms1)
    # synced_chromatograms2 = cl.sync_individual_chromatograms(mean_c1, chromatograms2)
    # merged_chrom = cl.merge_chromatograms(synced_chromatograms1, synced_chromatograms2)
    # cl.umap_analysis(merged_chrom, vintage, "Chromatograms with sync")

    # cl.plot_chromatograms(mean_c1, mean_c2, file_name1, file_name2, cl)

    from wine_analysis import SyncChromatograms
    sync_chrom = SyncChromatograms(
        mean_c1, mean_c2, 1, np.linspace(0.9, 1.1, 50), 2, threshold=0.1, max_sep_threshold=50)
    corrected_c2 = sync_chrom.adjust_chromatogram()
    sync_chrom.plot_chromatograms(corrected_c2)



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
            analysis.data,
            cls._process_labels(vintage),
            analysis.chem_name,
            neigh_range=range(30, 100, 10),
            random_states=range(0, 64, 16)
        )
        analysis.run_umap(n_neighbors=best_perp, random_state=best_rs)

        best_neigh, best_rs, best_score = run_umap_and_evaluate(
            analysis.data,
            cls._process_labels(vintage),
            analysis.chem_name,
            neigh_range=range(30, 100, 10),
            random_states=range(0, 64, 16)
        )
        analysis.run_umap(n_neighbors=best_neigh, random_state=best_rs)


    # # Example usage of DimensionalityReducer
    # data_loader = DataLoader('/home/luiscamara/PycharmProjects/wine_scheck/data/oak.npy')
    # reducer = DimensionalityReducer(data_loader.get_standardized_data())
    # pca_result = reducer.pca(components=80)
    # tsne_result = reducer.tsne()
    # umap_result = reducer.umap()

    # # pca_dict = reducer.perform_pca_on_dict(analysis.labels, n_components=n_components)
    # analysis = WineAnalysis(data_dict=pca_dict)



