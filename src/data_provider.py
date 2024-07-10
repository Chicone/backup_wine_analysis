import numpy as np

class AccuracyDataProvider:
    def __init__(self):
        """
        Initializes the AccuracyDataProvider with predefined datasets.
        """
        self.datasets = {
            'lda': self._accuracies_LDA(),
        }

    def _accuracies_LDA(self, vintage=False):
        """
        Returns raw accuracy data.

        Returns
        -------
        tuple
            A tuple containing categories, classification types, and accuracy data.
        """
        categories = ['2022 01 7 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx',
                      '2022 4 new bordeaux Oak Masse 5 NORMALIZED 052022 SM2 .xlsx',
                      '2022 01 11 chateaux Oak All vintages Masse 5 NORMALIZED SM.xlsx',
                      '2018 7 chateaux Oak Old vintages Masse 5.xlsx',
                      'oak',
                 ]
        preprocessing_types = ['Raw', 'PCA', '3bins', 'PCA 3bins', 'PCA prune', 'PCA prune 3bins']

        if not vintage:
            accuracy = np.array([
                [86.0, 88.3, 00.0, 00.0, 00.0, 00.0],
                [52.7, 70.5, 00.0, 00.0, 00.0, 00.0],
                [71.5, 78.4, 00.0, 00.0, 00.0, 00.0],
                [96.7, 98.6, 00.0, 00.0, 00.0, 00.0],
            ])
        else:
            accuracy = np.array([
                [31.6, 28.9, 00.0, 00.0, 00.0, 00.0],
                [10.2,  8.4, 33.3, 00.0, 00.0, 00.0],  # 10 bins
                [32.4, 32.1, 00.0, 00.0, 00.0, 00.0],
                [37.8, 38.4, 00.0, 00.0, 00.0, 00.0],
            ])
        return categories, preprocessing_types, accuracy