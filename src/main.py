import numpy as np
import os
from data_loader import DataLoader, ChromatogramLoader
from dimensionality_reduction import DimensionalityReducer
from wine_analysis import WineAnalysis
from classification import Classifier
from visualizer import Visualizer, plot_all_acc_LDA, plot_stacked_chromatograms
if __name__ == "__main__":

    # plot_all_acc_LDA(vintage=False)
    xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2018 7 chateaux Oak Old vintages Masse 5.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 4 new bordeaux Oak Masse 5 NORMALIZED 052022 SM2 .xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 11 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/older data/Datat for paper Sept 2022/2018 7 chateaux Oak Old vintages Masse 5.xlsx')
    # xlsx_path = os.path.expanduser('~/PycharmProjects/wine_scheck/data/concat.npy')  #  not normalised
    npy_path = os.path.splitext(xlsx_path)[0] + '.npy'

    loader = ChromatogramLoader(
        os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak Old vintages Masse 5 NORMALIZED SM.npy'),
        os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2018 7 chateaux Oak Old vintages Masse 5.npy')
    )
    loader.run()

    if not os.path.exists(npy_path):
        # Load .xlsx file and save into npy the chromatogram signal
        xlsx_loader = DataLoader(xlsx_path, normalize=False)
        np.save(os.path.expanduser(npy_path), xlsx_loader.data)

    n_splits = 100
    vintage = False
    # pca = True
    pca = False

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
        from dimensionality_reduction import run_umap_and_evaluate, run_tsne_and_evaluate
        analysis = WineAnalysis(npy_path, normalize=False)
        cls = Classifier(analysis.data, analysis.labels)  # to use _process_labels
        # Classification
        # analysis.train_classifier(classifier_type='LDA', vintage=vintage, n_splits=n_splits, test_size=None)
        # analysis.run_tsne()
        # analysis.run_umap(n_neighbors=5, random_state=90)
        # analysis.run_umap(n_neighbors=30, random_state=52) # 2022 oak old
        analysis.run_umap(n_neighbors=70, random_state=84)  # 2018 oak old
        # run_tsne_and_evaluate(analysis.data, cls._process_labels(vintage), analysis.chem_name)
        # run_umap_and_evaluate(analysis.data, cls._process_labels(vintage), analysis.chem_name)


    # # Example usage of DimensionalityReducer
    # data_loader = DataLoader('/home/luiscamara/PycharmProjects/wine_scheck/data/oak.npy')
    # reducer = DimensionalityReducer(data_loader.get_standardized_data())
    # pca_result = reducer.pca(components=80)
    # tsne_result = reducer.tsne()
    # umap_result = reducer.umap()

    # # pca_dict = reducer.perform_pca_on_dict(analysis.labels, n_components=n_components)
    # analysis = WineAnalysis(data_dict=pca_dict)



