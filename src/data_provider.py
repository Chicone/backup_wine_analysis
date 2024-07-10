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
                      'oak.npy',
                 ]
        preprocessing_types = ['Raw', 'PCA', '3bins', 'PCA 3bins', 'PCA prune', 'PCA prune 3bins']
        #  results for 200 splits
        if not vintage:
            accuracy = np.array([
                [86.6, 90.9, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [52.5, 67.5, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [73.5, 77.5, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [97.3, 97.7, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [97.5, 98.6, 00.0, 00.0, 00.0, 00.0],  # do normalise
            ])
        else:
            accuracy = np.array([
                [31.0, 28.7, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [15.8, 11.2, 33.3, 00.0, 00.0, 00.0],  # do NOT normalise
                [32.6, 32.3, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [38.1, 37.2, 00.0, 00.0, 00.0, 00.0],  # do normalise
                [38.2, 36.6, 60.5, 00.0, 00.0, 00.0],  # do normalise; max at 15bins
            ])
        return categories, preprocessing_types, accuracy