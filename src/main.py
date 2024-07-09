import numpy as np
import os
from data_loader import DataLoader
from dimensionality_reduction import DimensionalityReducer
from wine_analysis import WineAnalysis
from classification import Classifier
if __name__ == "__main__":

    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 7 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 4 new bordeaux Oak Masse 5 NORMALIZED 052022 SM2 .xlsx')
    # xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/DataNov2022/2022 01 11 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx')
    xlsx_path = os.path.expanduser('~/Documents/datasets/BordeauxData/older data/Datat for paper Sept 2022/2018 7 chateaux Oak Old vintages Masse 5.xlsx')
    npy_path = os.path.splitext(xlsx_path)[0] + '.npy'

    if not os.path.exists(npy_path):
        # Load .xlsx file and save into npy the chromatogram signal
        xlsx_loader = DataLoader(xlsx_path)
        np.save(os.path.expanduser(npy_path), xlsx_loader.data)

    # Instance of the class, load data, etc.
    analysis = WineAnalysis(npy_path)

    # # PCA-reduce
    cls = Classifier(analysis.data, analysis.labels)  # to use _process_labels
    reducer = DimensionalityReducer(analysis.data)


    # pca_dict, cumulative_variance, n_components = reducer.cumulative_variance(
    #      analysis.labels, variance_threshold=0.97, plot=False)
    reducer.cross_validate_pca_classification(cls._process_labels(vintage=False), n_splits=100, test_size=None)

    # # pca_dict = reducer.perform_pca_on_dict(analysis.labels, n_components=n_components)
    # analysis = WineAnalysis(data_dict=pca_dict)

    # Classification
    # analysis.train_classifier(classifier_type='LDA', vintage=False, n_splits=50, test_size=None)
    # analysis.run_tsne()
    # analysis.run_umap()




    # # Example usage of DimensionalityReducer
    # data_loader = DataLoader('../datasets/2018_7_chateaux_Oak_Old_vintages_Masse_5.npy')
    # reducer = DimensionalityReducer(data_loader.get_standardized_data())
    # pca_result = reducer.pca(components=80)
    # tsne_result = reducer.tsne()
    # umap_result = reducer.umap()