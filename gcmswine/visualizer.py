import numpy as np
from matplotlib import pyplot as plt
from gcmswine.data_provider import AccuracyDataProvider
import pandas as pd
from sklearn.manifold import MDS
from matplotlib.cm import get_cmap
from matplotlib import colormaps
from matplotlib import cm


class Visualizer:
    """
    A class used to visualize data in various forms such as scatter plots and 3D stacked plots.
    """

    @staticmethod
    def assign_color(label):
        """
        Assigns a corresponding color to a label.

        Parameters
        ----------
        label : str
            The original label.

        Returns
        -------
        tuple
            A tuple containing:
            - modified_label (str): The label with the first letter changed.
            - color (str): The color corresponding to the original first letter.
        """
        s = list(label)
        if label[0] == 'C':
            color = 'b'
        elif label[0] == 'D':
            color = 'g'
        elif label[0] == 'E':
            color = 'r'
        elif label[0] == 'H':
            color = 'm'
        elif label[0] == 'J':
            color = 'k'
        elif label[0] == 'K':
            color = 'y'
        elif label[0] == 'L':
            color = 'c'
        elif label[0] == 'M':
            color = 'peru'
        elif label[0] == 'N':
            color = 'limegreen'
        elif label[0] == 'P':
            color = 'cornflowerblue'
        elif label[0] == 'Q':
            color = 'olive'
        elif label[0] == 'R':
            color = 'tomato'
        elif label[0] == 'U':
            color = 'orange'
        elif label[0] == 'W':
            color = 'slateblue'
        elif label[0] == 'X':
            color = 'darkcyan'
        elif label[0] == 'Y':
            color = 'gold'
        elif label[0] == 'Z':
            color = 'firebrick'

        if label == 'France':
            color = 'b'
        elif label == 'Switzerland':
            color = 'g'
        elif label == 'US':
            color = 'r'

        if label == 'Beaune':
            color = 'b'
        elif label == 'Alsace':
            color = 'g'
        elif label == 'Neuchatel':
            color = 'r'
        elif label == 'Genève':
            color = 'k'
        elif label== 'Valais':
            color = 'y'
        elif label == 'Californie':
            color = 'c'
        elif label == 'Oregon':
            color = 'limegreen'

        if label == '09':
            color = 'b'
        elif label == '10':
            color = 'g'
        elif label == '11':
            color = 'r'
        elif label == '12':
            color = 'm'
        elif label == '13':
            color = 'k'
        elif label == '14':
            color = 'y'
        elif label == '15':
            color = 'c'
        elif label == '16':
            color = 'peru'
        elif label == '17':
            color = 'limegreen'
        elif label == '18':
            color = 'cornflowerblue'
        elif label == '19':
            color = 'olive'
        elif label == '20':
            color = 'tomato'
        elif label == '88':
            color = 'orange'
        elif label == '95':
            color = 'slateblue'
        elif label == '01':
            color = 'darkcyan'
        elif label == '08':
            color = 'gold'

        if label == 'NB':
            color = 'b'
        elif label == 'SB':
            color = 'g'


        return color


    @staticmethod
    def change_letter_and_color(label):
        """
        Changes the first letter of a label and assigns a corresponding color.

        Parameters
        ----------
        label : str
            The original label.

        Returns
        -------
        tuple
            A tuple containing:
            - modified_label (str): The label with the first letter changed.
            - color (str): The color corresponding to the original first letter.
        """
        s = list(label)
        if label[0] == 'V':
            s[0] = 'A'
            color = 'b'
        elif label[0] == 'A':
            s[0] = 'B'
            color = 'g'
        elif label[0] == 'S':
            s[0] = 'C'
            color = 'r'
        elif label[0] == 'F':
            s[0] = 'D'
            color = 'm'
        elif label[0] == 'T':
            s[0] = 'E'
            color = 'k'
        elif label[0] == 'G':
            s[0] = 'F'
            color = 'y'
        elif label[0] == 'B':
            s[0] = 'G'
            color = 'c'
        elif label[0] == 'M':
            s[0] = 'F'
            color = 'y'
        elif label[0] == 'H':
            s[0] = 'H'
            color = 'limegreen'
        elif label[0] == 'I':
            s[0] = 'I'
            color = 'cornflowerblue'
        elif label[0] == 'K':
            s[0] = 'K'
            color = 'olive'
        elif label[0] == 'O':
            s[0] = 'O'
            color = 'tomato'
        return "".join(s), color

    @staticmethod
    def plot_2d_results(result, title, xlabel, ylabel, offset=0.02):
        """
        Plots the results of a dimensionality reduction algorithm with different colors for different label groups.

        Parameters
        ----------
        result : pandas.DataFrame
            The results of the dimensionality reduction.
        title : str
            The title of the plot.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        """
        label_groups = {}
        colors = {}
        for label in result.index:
            first_letter = label[0]
            if first_letter not in label_groups:
                label_groups[first_letter] = []
                colors[first_letter] = np.random.rand(3,)
            label_groups[first_letter].append(label)

        plt.figure(figsize=(8, 6))
        for first_letter, color in colors.items():
            labels = label_groups[first_letter]
            plt.scatter([], [], label=first_letter, color=color)  # Empty scatter plot for legend
            for i, label in enumerate(labels):
                if 'pinot' in title.lower():
                    annotation = label[:3]
                    color = Visualizer.assign_color(label)
                else:
                    annotation, color = Visualizer.change_letter_and_color(label)
                # x = result.loc[label, xlabel]
                # y = result.loc[label, ylabel]
                x = result.loc[label, xlabel].iloc[i]
                y = result.loc[label, ylabel].iloc[i]
                plt.scatter(-x, -y, s=32, c=[color])
                plt.annotate(annotation, (-x + offset, -y + offset), fontsize=8, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.show()

def plot_all_acc_LDA(vintage=False):
    """
    Plots the classification accuracy for different preprocessing methods and chemical types using LDA.

    This function generates a bar plot of the classification accuracy for different preprocessing methods
    and chemical types using LDA decoding.

    Parameters
    ----------
    vintage : bool, optional
        Whether to use vintage data for evaluation (default is False).
    """
    if vintage:
        modality = "vintage"
    else:
        modality = "estate"

    provider = AccuracyDataProvider()
    categories, preprocessing_types, accuracy = provider._accuracies_LDA(vintage=vintage)

    all_accuracies = [accuracy[0], accuracy[1], accuracy[2], accuracy[3], accuracy[4]]

    chemical_types = categories

    fig, ax = plt.subplots()

    bar_width = 0.2
    bar_gap = 0.1

    positions = np.arange(len(chemical_types))

    for i, accuracies in enumerate(all_accuracies):
        ax.bar(positions[i], accuracies[0], width=bar_width, color='blue', alpha=0.7, label='Raw data' if i == 0 else "")
        ax.bar(positions[i] + (bar_width), accuracies[1], width=bar_width, color='green', alpha=0.7, label='PCA on raw' if i == 0 else "")
        ax.bar(positions[i] + 2 * (bar_width), accuracies[2], width=bar_width, color='red', alpha=0.7, label='Best 3 bins' if i == 0 else "")
        # ax.bar(positions[i] + 3 * (bar_width), accuracies[3], width=bar_width, color='c', alpha=0.7, label='PCA on best 3 bins' if i == 0 else "")
        # ax.bar(positions[i] + 4 * (bar_width), accuracies[4], width=bar_width, color='m', alpha=0.7, label='PCA prune on raw ' if i == 0 else "")
        # ax.bar(positions[i] + 5 * (bar_width), accuracies[5], width=bar_width, color='y', alpha=0.7, label='PCA prune on best 3 bins' if i == 0 else "")

        ax.text(positions[i], accuracies[0] + 0.05, f'{accuracies[0]:.2f}', ha='center', va='bottom', color='blue', alpha=0.7, fontsize=10, rotation=50)
        ax.text(positions[i] + (bar_width), accuracies[1] + 0.05, f'{accuracies[1]:.2f}', ha='center', va='bottom', color='green', alpha=0.7, fontsize=10, rotation=50)
        ax.text(positions[i] + 2 * (bar_width), accuracies[2] + 0.05, f'{accuracies[2]:.2f}', ha='center', va='bottom', color='red', alpha=0.7, fontsize=10, rotation=50)
        # ax.text(positions[i] + 3 * (bar_width), accuracies[3] + 0.05, f'{accuracies[3]:.2f}', ha='center', va='bottom', color='c', alpha=0.7, fontsize=10, rotation=50)
        # ax.text(positions[i] + 4 * (bar_width), accuracies[4] + 0.05, f'{accuracies[4]:.2f}', ha='center', va='bottom', color='m', alpha=0.7, fontsize=10, rotation=50)
        # ax.text(positions[i] + 5 * (bar_width), accuracies[5] + 0.05, f'{accuracies[5]:.2f}', ha='center', va='bottom', color='y', alpha=0.7, fontsize=10, rotation=50)

    ax.set_xticks(positions)

    # Split the labels into multiple lines
    def split_label(label, max_width=25):
        words = label.split()
        split_labels = []
        current_line = ""
        for word in words:
            if len(current_line) + len(word) <= max_width:
                if current_line:
                    current_line += " "
                current_line += word
            else:
                split_labels.append(current_line)
                current_line = word
        split_labels.append(current_line)
        return "\n".join(split_labels)

    split_labels = [split_label(label) for label in chemical_types]
    ax.set_xticklabels(split_labels)

    ax.set_ylabel('Accuracy (%)')
    ax.set_yticks(np.arange(0, 101, 10))
    plt.ylim(0, 120)
    ax.set_xlabel('Dataset')
    plt.title(f"LDA {modality} decoding for different pre-processing")

    ax.legend()

    plt.show()

def plot_stacked_chromatograms(analysis, labels):
    """
    Plots a list of wine labels in a 2D-stacked fashion.

    Parameters
    ----------
    analysis : WineAnalysis
        The analysis object containing the data loader.
    labels : list of str
        The list of wine labels to plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    len_chrom = len(analysis.data_loader.data[labels[0]])
    x = np.linspace(0, 100, len_chrom)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    num_plots = len(labels)

    for i in range(num_plots):
        z = np.asarray(analysis.data_loader.data[labels[i]]).astype(int)
        y = np.full_like(x, i)
        ax.plot(x, y + i, z, label=f'Plot {i + 1}')

    ax.set_xlabel('Retention time (%)')
    ax.set_ylabel('Amplitude')
    ax.set_zlabel('Wines')
    ax.set_title('Stacked 2D Plots')
    ax.legend()
    plt.show()



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_classification_accuracy():
    """
    Plots classification accuracy for Pinot Noir ISVV (LLE scan) and Pinot Noir ISVV (DLLME scan) across multiple classifiers,
    comparing values for different settings A, B, C, and D.
    """
    # Data for Pinot Noir ISVV (LLE scan) and Pinot Noir ISVV (DLLME scan)
    data = {
        "Classifier": [
            "LDA", "LR", "RFC", "PAC", "PER", "RGC", "SGD", "SVM", "KNN", "DTC", "GNB"
        ],
        "Pinot Noir ISVV (LLE scan) A": [
            0.493, 0.499, 0.403, 0.519, 0.479, 0.555, 0.519, 0.238, 0.224, 0.198, 0.292
        ],
        "Pinot Noir ISVV (LLE scan) B": [
            0.572, 0.561, 0.467, 0.568, 0.575, 0.601, 0.613, 0.291, 0.304, 0.269, 0.329
        ],
        "Pinot Noir ISVV (LLE scan) C": [
            0.463, 0.485, 0.302, 0.541, 0.386, 0.561, 0.404, 0.147, 0.209, 0.230, 0.176
        ],
        "Pinot Noir ISVV (LLE scan) D": [
            0.551, 0.577, 0.476, 0.585, 0.584, 0.612, 0.597, 0.291, 0.293, 0.306, 0.331
        ],
        "Pinot Noir ISVV (DLLME scan) A": [
            0.441, 0.392, 0.323, 0.460, 0.437, 0.463, 0.486, 0.239, 0.244, 0.248, 0.253
        ],
        "Pinot Noir ISVV (DLLME scan) B": [
            0.557, 0.512, 0.419, 0.595, 0.549, 0.592, 0.586, 0.311, 0.361, 0.304, 0.312
        ],
        "Pinot Noir ISVV (DLLME scan) C": [
            0.316, 0.372, 0.278, 0.364, 0.316, 0.421, 0.336, 0.133, 0.217, 0.175, 0.122
        ],
        "Pinot Noir ISVV (DLLME scan) D": [
            0.516, 0.505, 0.418, 0.583, 0.557, 0.544, 0.583, 0.315, 0.379, 0.309, 0.300
        ]
    }
    # Convert to DataFrame
    df = pd.DataFrame(data)
    # Plotting setup
    bar_width = 0.2  # Adjusted for 4 bars per group
    index = np.arange(len(df["Classifier"]))  # Classifier positions
    # Create subplots for Pinot Noir ISVV (LLE scan) and Pinot Noir ISVV (DLLME scan)
    fig, ax = plt.subplots(2, 1, figsize=(14, 12))
    # Standard color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9370DB']  # Blue, Orange, Green, Re
    # Plot for Pinot Noir ISVV (LLE scan)
    ax[0].bar(index, df["Pinot Noir ISVV (LLE scan) A"], bar_width, label="A: Original TIC", color=colors[0])
    ax[0].bar(index + bar_width, df["Pinot Noir ISVV (LLE scan) B"], bar_width, label="B: TIC after Alignment", color=colors[1])
    ax[0].bar(index + 2 * bar_width, df["Pinot Noir ISVV (LLE scan) C"], bar_width, label="C: Total Ion Spectrum (TIS)", color=colors[2])
    ax[0].bar(index + 3 * bar_width, df["Pinot Noir ISVV (LLE scan) D"], bar_width, label="D: Aligned TIC + TIS", color=colors[3])
    ax[0].set_title("Pinot Noir ISVV (LLE scan) - Classification Accuracy", size=14)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_ylim(0, 1.02)
    ax[0].set_xticks(index + 1.5 * bar_width)
    ax[0].set_xticklabels(df["Classifier"], fontsize=12)  # Set x-tick labels and size for the top plot
    ax[0].legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax[0].grid(axis='y')
    # Plot for Pinot Noir ISVV (DLLME scan)
    ax[1].bar(index, df["Pinot Noir ISVV (DLLME scan) A"], bar_width, label="A: Original TIC", color=colors[0])
    ax[1].bar(index + bar_width, df["Pinot Noir ISVV (DLLME scan) B"], bar_width, label="B: TIC after Alignment", color=colors[1])
    ax[1].bar(index + 2 * bar_width, df["Pinot Noir ISVV (DLLME scan) C"], bar_width, label="C: Total Ion Spectrum (TIS)", color=colors[2])
    ax[1].bar(index + 3 * bar_width, df["Pinot Noir ISVV (DLLME scan) D"], bar_width, label="D: Aligned TIC + TIS", color=colors[3])
    ax[1].set_title("Pinot Noir ISVV (DLLME scan) - Classification Accuracy", size=14)
    ax[1].set_xlabel("Classifier")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_ylim(0, 1.02)
    ax[1].set_xticks(index + 1.5 * bar_width)
    ax[1].set_xticklabels(df["Classifier"], fontsize=12)  # Set x-tick labels and size for the bottom plot
    ax[1].legend(loc='upper right', bbox_to_anchor=(1, 1))
    ax[1].grid(axis='y')
    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix_3d(conf_matrix, class_labels, title="3D visualization with MDS"):
    """
    Visualizes a confusion matrix in 3D using MDS, forcing symmetry using the upper triangular part.

    Parameters:
    ----------
    conf_matrix : np.ndarray
        A 2D NumPy array representing the confusion matrix.
    class_labels : list of str
        A list of class labels corresponding to the confusion matrix rows/columns.

    Example Input:
    --------------
    conf_matrix = np.array([
        [50, 10, 5],
        [8, 45, 7],
        [6, 9, 40]
    ])
    class_labels = ['Class 1', 'Class 2', 'Class 3']
    """
    # Ensure the input is a NumPy array
    conf_matrix = np.array(conf_matrix)

    # Force symmetry using the upper triangle
    symmetric_conf_matrix = np.triu(conf_matrix) + np.triu(conf_matrix, k=1).T

    # Normalize using global normalization to preserve symmetry
    dissimilarity_matrix = 1 - symmetric_conf_matrix / symmetric_conf_matrix.sum()

    # Apply MDS for 3D embedding
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    embedding = mds.fit_transform(dissimilarity_matrix)

    # 3D Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Generate a colormap for the classes
    cmap = get_cmap("tab20")  # Use a colormap with many distinct colors
    colors = [cmap(i / len(class_labels)) for i in range(len(class_labels))]

    # Scatter plot with different colors for each class
    for i, label in enumerate(class_labels):
        ax.scatter(embedding[i, 0], embedding[i, 1], embedding[i, 2],
                   color=colors[i], s=100, label=label)

    # Connect points with a line to aid 3D visualization
    for i in range(len(embedding) - 1):
        ax.plot(
            [embedding[i, 0], embedding[i + 1, 0]],
            [embedding[i, 1], embedding[i + 1, 1]],
            [embedding[i, 2], embedding[i + 1, 2]],
            color="gray", linestyle="--", alpha=0.5
        )

    # Add legend and labels
    ax.legend(loc="best", fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_zlabel("Dimension 3")

    plt.show()


def plot_accuracy_vs_channels():
    """
    Plots a graph where the x-axis represents the number of contiguous aggregated channels,
    and the y-axis represents the accuracy. Data is hardcoded.
    """
    # Hardcoded data
    aggr_channels =            [1    , 2    , 3    , 6    , 10   , 15   , 22   , 30   , 45   , 60   , 90   , 181  ]  # Number of aggregated channels
    accuracy_isvv =            [0.846, 0.890, 0.886, 0.883, 0.802, 0.619, 0.600, 0.526, 0.456, 0.424, 0.358, 0.580]  # Accuracy values
    accuracy_changins =        [0.786, 0.874, 0.906, 0.852, 0.814, 0.657, 0.612, 0.600, 0.553, 0.505, 0.510, 0.744]  # Accuracy values

    # New values with dynamically learned alpha
    accuracy_isvv_alpha =      [0.830, 0.889, 0.861, 0.884, 0.790, 0.613, 0.590, 0.537, 0.475, 0.424, 0.381, 0.551]  # Accuracy values
    accuracy_changins_alpha =  [0.811, 0.889, 0.924, 0.856, 0.810, 0.649, 0.657, 0.595, 0.536, 0.550, 0.571, 0.794]  # Accuracy values

    # Values with concatenated m/z profiles
    accuracy_isvv_conc =      [0.646, 0.616, 0.613, 0.595, 0.584, 0.527, 0.520, 0.500, 0.434, 0.486, 0.515, 0.447]  # Accuracy values
    accuracy_changins_conc =  [0.771, 0.773, 0.786, 0.760, 0.728, 0.746, 0.712, 0.747, 0.701, 0.702, 0.710, 0.707]  # Accuracy values

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(aggr_channels, accuracy_changins, marker='o', linestyle='-', color='red', label=r'Changins $\alpha=500$')
    plt.plot(aggr_channels, accuracy_changins_alpha, marker='s', linestyle='--', color='red', markerfacecolor='none', label=r'Changins adapt. $\alpha$')
    plt.plot(aggr_channels, accuracy_changins_conc, marker='^', linestyle=':', color='red', markerfacecolor='none', label=r'Changins concat. m/z profiles')
    plt.plot(aggr_channels, accuracy_isvv, marker='o', linestyle='-', color='blue', label=r'ISVV $\alpha=500$')
    plt.plot(aggr_channels, accuracy_isvv_alpha, marker='s', linestyle='--', color='blue', markerfacecolor='none', label=r'ISVV adapt. $\alpha$')
    plt.plot(aggr_channels, accuracy_isvv_conc, marker='^', linestyle=':', color='blue', markerfacecolor='none', label=r'ISVV  concat. m/z profiles')




    # Customize the plot
    plt.title("Accuracy vs. Number of Aggregated Channels")
    plt.xlabel("Number of Aggregated Channels (n)")
    plt.ylabel("Accuracy")
    plt.xticks(aggr_channels)  # Set x-ticks to match the data points
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_accuracy_vs_channels_split_by_sample():
    """
    Plots a graph where the x-axis represents the number of contiguous aggregated channels,
    and the y-axis represents the accuracy. Train-test splitting is by sample, not channels.
     Data is hardcoded.
    """
    # Hardcoded data
    final_channels =            [1    , 2    , 3    , 6    , 10   , 15   , 22   , 30   , 45   , 60   , 90   , 181  ]  # Number of aggregated channels
    accuracy_isvv =            [0.470, 0.484, 0.458, 0.440, 0.418, 0.383, 0.360, 0.315, 0.276, 0.245, 0.213, 0.178]  # Accuracy values
    accuracy_changins =        [0.736, 0.704, 0.686, 0.630, 0.569, 0.529, 0.466, 0.392, 0.328, 0.286, 0.247, 0.220]  # Accuracy values


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(final_channels, accuracy_changins, marker='o', linestyle='-', color='red', label=r'Changins' )
    plt.plot(final_channels, accuracy_isvv, marker='o', linestyle='-', color='blue', label=r'ISVV')



    # Customize the plot
    plt.title("Accuracy vs. Total Number of Channels after aggregation (train-test split by sample)")
    plt.xlabel("Number of channels", size=14)
    plt.ylabel("Accuracy", size=14)
    plt.xticks(final_channels)  # Set x-ticks to match the data points
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_channels_concatenated():
    """
    Plots a graph where the x-axis represents the number of contiguous aggregated channels,
    and the y-axis represents the accuracy. Train-test splitting is by sample, not channels.
     Data is hardcoded.
    """
    # Hardcoded data
    final_channels =           [1    , 2    , 3    , 6    , 10   , 15   , 20   , 30   , 45   , 60   , 90   , 181  ]  # Number of aggregated channels
    accuracy_isvv =            [0.458, 0.472, 0.491, 0.502, 0.512, 0.515, 0.527, 0.526, 0.532, 0.540, 0.538, 0.567]  # Accuracy values
    accuracy_changins =        [0.751, 0.762, 0.773, 0.778, 0.788, 0.783, 0.796, 0.778, 0.776, 0.773, 0.760, 0.763]  # Accuracy values


    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(final_channels, accuracy_changins, marker='o', linestyle='-', color='red', label=r'Changins' )
    plt.plot(final_channels, accuracy_isvv, marker='o', linestyle='-', color='blue', label=r'ISVV')



    # Customize the plot
    plt.title("Accuracy vs. Total Number of Channels after aggregation (concatenated channels)")
    plt.xlabel("Number of channels", size=14)
    plt.ylabel("Accuracy", size=14)
    plt.xticks(final_channels)  # Set x-ticks to match the data points
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


def plot_2d(embedding, title, labels, label_dict, group_by_country=False):
    """
    Plot a 2D scatter plot of embedded data with labeled points.
    Automatically applies Bordeaux-specific coloring if labels match Bordeaux format.

    Parameters
    ----------
    embedding : np.ndarray
        2D coordinates of points (n_samples, 2).

    title : str
        Title for the plot.

    labels : np.ndarray
        Array of label codes corresponding to each point.

    label_dict : dict
        Mapping from label codes to human-readable names.

    group_by_country : bool
        If True, color by country (only used for winery/burgundy-style labels).

    Returns
    -------
    None. Displays a matplotlib plot.
    """
    labels = np.array(labels)
    plt.figure(figsize=(8, 6))
    # Handle case where label_dict is a list, not a dict
    if isinstance(label_dict, list):
        label_dict = {label: label for label in label_dict}
    markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
    color_map = colormaps.get_cmap("tab20")

    # Auto-detect Bordeaux based on label prefixes
    bordeaux_prefixes = {'V', 'A', 'S', 'F', 'T', 'G', 'B', 'M', 'H', 'I', 'K', 'O'}
    is_bordeaux = all(str(label)[0] in bordeaux_prefixes for label in labels)

    if is_bordeaux:
        sorted_indices = np.argsort(labels)
        labels = labels[sorted_indices]
        embedding = embedding[sorted_indices]
        used_colors = set()

        for i, label in enumerate(labels):
            x, y = embedding[i]
            mod_label, color = change_letter_and_color_bordeaux(label)

            if color not in used_colors:
                plt.scatter([], [], color=color, label=mod_label[0])
                used_colors.add(color)

            plt.scatter(x, y, color=color, s=60, alpha=0.9)
            plt.annotate(mod_label, (x, y),  xytext=(3, 3), textcoords='offset points',fontsize=8, color=color)
        # Sort legend alphabetically by label
        handles, labels_ = plt.gca().get_legend_handles_labels()
        if handles and labels_:
            sorted_legend = sorted(zip(labels_, handles), key=lambda x: x[0])
            labels_, handles = zip(*sorted_legend)
            plt.legend(handles, labels_, title="Mapped group", loc="best", fontsize='large')
        plt.title(title, fontsize=16)
        plt.tight_layout()
        plt.show(block=False)

    else:
        label_keys = list(label_dict.keys())
        if group_by_country:
            countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
            country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}

        for i, code in enumerate(label_keys):
            mask = labels == code
            readable_label = label_dict[code]
            marker = markers[i % len(markers)]

            if group_by_country:
                country = readable_label.split("(")[-1].strip(")")
                color = country_colors[country]
            else:
                color = color_map(i / len(label_keys))

            plt.scatter(*embedding[mask].T, label=readable_label, alpha=0.9, s=80,
                        color=color, marker=marker)

        plt.title(title, fontsize=16)
        plt.legend(fontsize='large', loc='best')
        plt.tight_layout()
        plt.show(block=False)


# def plot_2d(embedding, title, region, labels, label_dict, group_by_country=False):
#     """
#        Plot a 2D scatter plot of embedded data with labeled points.
#
#        Parameters:
#            embedding (np.ndarray): 2D coordinates of points (n_samples, 2).
#            title (str): Title for the plot.
#            region (str): Type of region grouping (e.g., 'winery', 'burgundy', etc.).
#            labels (np.ndarray): Array of integer or categorical labels corresponding to each point.
#            label_dict (dict): Mapping from label codes to human-readable names.
#            group_by_country (bool): If True, use country-based coloring instead of winery-level.
#
#        Returns:
#            None. Displays a matplotlib plot.
#        """
#     labels = np.array(labels)
#     plt.figure(figsize=(8, 6))
#     markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
#     color_map = colormaps.get_cmap("tab20")
#
#     if region == "winery" or region == "burgundy":
#         label_keys = list(label_dict.keys())
#         if group_by_country:
#             countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
#             country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}
#
#         for i, code in enumerate(label_keys):
#             mask = labels == code
#             readable_label = label_dict[code]
#             marker = markers[i % len(markers)]
#             color = (country_colors[readable_label.split("(")[-1].strip(")")]
#                      if group_by_country else color_map(i / len(label_keys)))
#             plt.scatter(*embedding[mask].T, label=readable_label, alpha=0.9, s=80,
#                         color=color, marker=marker)
#     else:
#         for i, label in enumerate(np.unique(labels)):
#             mask = labels == label
#             plt.scatter(*embedding[mask].T, label=str(label), alpha=0.9, s=80,
#                         marker=markers[i % len(markers)])
#
#     plt.title(title, fontsize=16)
#     plt.legend(fontsize='large', loc='best')
#     plt.tight_layout()
#     plt.show(block=False)


def plot_3d(embedding, title, labels, label_dict, group_by_country=False):
    """
    Plot a 3D scatter plot of embedded data with labeled points.

    Parameters:
        embedding (np.ndarray): 3D coordinates of points (n_samples, 3).
        title (str): Title for the plot.
        labels (np.ndarray): Array of integer or categorical labels corresponding to each point.
        label_dict (dict): Mapping from label codes to human-readable names.
        group_by_country (bool): If True, use country-based coloring instead of winery-level.

    Returns:
        None. Displays a matplotlib 3D plot.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    from gcmswine.visualizer import change_letter_and_color_bordeaux

    labels = np.array(labels)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
    color_map = colormaps.get_cmap("tab20")

    # Handle case where label_dict is a list, not a dict
    if isinstance(label_dict, list):
        label_dict = {label: label for label in label_dict}

    # Auto-detect Bordeaux based on label prefixes
    bordeaux_prefixes = {'V', 'A', 'S', 'F', 'T', 'G', 'B', 'M', 'H', 'I', 'K', 'O'}
    is_bordeaux = all(str(label)[0] in bordeaux_prefixes for label in labels)

    if is_bordeaux:
        sorted_indices = np.argsort(labels)
        labels = labels[sorted_indices]
        embedding = embedding[sorted_indices]
        used_colors = set()

        for i, label in enumerate(labels):
            x, y, z = embedding[i]
            mod_label, color = change_letter_and_color_bordeaux(label)

            if color not in used_colors:
                ax.scatter([], [], [], color=color, label=mod_label[0])
                used_colors.add(color)

            ax.scatter(x, y, z, color=color, s=60, alpha=0.9)

        handles, labels_ = ax.get_legend_handles_labels()
        if handles and labels_:
            sorted_legend = sorted(zip(labels_, handles), key=lambda x: x[0])
            labels_, handles = zip(*sorted_legend)
            ax.legend(handles, labels_, title="Mapped group", loc="best", fontsize='medium')

    else:
        label_keys = list(label_dict.keys())
        if group_by_country:
            countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
            country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}

        for i, code in enumerate(label_keys):
            mask = labels == code
            readable_label = label_dict[code]
            marker = markers[i % len(markers)]
            color = (country_colors[readable_label.split("(")[-1].strip(")")]
                     if group_by_country else color_map(i / len(label_keys)))
            ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
                       label=readable_label, alpha=0.9, s=80, color=color, marker=marker)

    ax.set_title(title)
    ax.set_xlabel(f"{title.split()[0]} 1")
    ax.set_ylabel(f"{title.split()[0]} 2")
    ax.set_zlabel(f"{title.split()[0]} 3")
    plt.tight_layout()
    plt.show(block=False)


# def plot_3d(embedding, title, region, labels, label_dict, group_by_country=False):
#     """
#       Plot a 3D scatter plot of embedded data with labeled points.
#
#       Parameters:
#           embedding (np.ndarray): 3D coordinates of points (n_samples, 3).
#           title (str): Title for the plot.
#           region (str): Type of region grouping (e.g., 'winery').
#           labels (np.ndarray): Array of integer or categorical labels corresponding to each point.
#           label_dict (dict): Mapping from label codes to human-readable names.
#           group_by_country (bool): If True, use country-based coloring instead of winery-level.
#
#       Returns:
#           None. Displays a matplotlib 3D plot.
#       """
#     labels = np.array(labels)
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     markers = ['o', 's', '^', 'v', 'D', 'X', '*', 'P', 'h', '8', '<', '>', 'p', 'H', 'd', '1']
#     color_map = colormaps.get_cmap("tab20")
#
#     if region == "winery":
#         label_keys = list(label_dict.keys())
#         if group_by_country:
#             countries = sorted(set(label.split("(")[-1].strip(")") for label in label_dict.values()))
#             country_colors = {country: color_map(i / len(countries)) for i, country in enumerate(countries)}
#
#         for i, code in enumerate(label_keys):
#             mask = labels == code
#             readable_label = label_dict[code]
#             marker = markers[i % len(markers)]
#             color = (country_colors[readable_label.split("(")[-1].strip(")")]
#                      if group_by_country else color_map(i / len(label_keys)))
#             ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
#                        label=readable_label, alpha=0.9, s=80, color=color, marker=marker)
#     else:
#         for i, label in enumerate(np.unique(labels)):
#             mask = labels == label
#             ax.scatter(embedding[mask, 0], embedding[mask, 1], embedding[mask, 2],
#                        label=str(label), alpha=0.9, s=80, marker=markers[i % len(markers)])
#
#     ax.set_title(title)
#     ax.set_xlabel(f"{title.split()[0]} 1")
#     ax.set_ylabel(f"{title.split()[0]} 2")
#     ax.set_zlabel(f"{title.split()[0]} 3")
#     ax.legend(fontsize='medium', loc='best')
#     plt.tight_layout()
#     plt.show(block=False)


def change_letter_and_color_bordeaux(label):
    """
    Modifies the first letter of a given label and assigns a corresponding color code.

    This function is used to remap the initial character of a label to a different character,
    typically for anonymization or categorization purposes, and to assign a color used in plotting
    or visual grouping.

    Parameters
    ----------
    label : str
        The original string label. The function only considers the first character of this string.

    Returns
    -------
    tuple of (str, str)
        - modified_label : str
            The updated label with the first character changed based on predefined rules.
        - color : str
            A color code (matplotlib-compatible string) associated with the original first character.
            For example:
            - 'b' for blue
            - 'g' for green
            - 'r' for red
            - 'm' for magenta
            - 'k' for black
            - 'y' for yellow
            - 'c' for cyan
            - or specific color names like 'limegreen', 'cornflowerblue', etc.

    Examples
    --------
    >>> change_letter_and_color_bordeaux("Vin123")
    ('Ain123', 'b')

    >>> change_letter_and_color_bordeaux("Apples")
    ('Bpples', 'g')

    Notes
    -----
    - The function does not validate whether `label` is empty. Use with non-empty strings only.
    - The mapping is hardcoded and may require updates for different datasets or applications.
    """
    s = list(label)
    if label[0] == 'V':
        s[0] = 'A'
        color = 'b'
    elif label[0] == 'A':
        s[0] = 'B'
        color = 'g'
    elif label[0] == 'S':
        s[0] = 'C'
        color = 'r'
    elif label[0] == 'F':
        s[0] = 'D'
        color = 'm'
    elif label[0] == 'T':
        s[0] = 'E'
        color = 'k'
    elif label[0] == 'G':
        s[0] = 'F'
        color = 'y'
    elif label[0] == 'B':
        s[0] = 'G'
        color = 'c'
    elif label[0] == 'M':
        s[0] = 'F'
        color = 'y'
    elif label[0] == 'H':
        s[0] = 'H'
        color = 'limegreen'
    elif label[0] == 'I':
        s[0] = 'I'
        color = 'cornflowerblue'
    elif label[0] == 'K':
        s[0] = 'K'
        color = 'olive'
    elif label[0] == 'O':
        s[0] = 'O'
        color = 'tomato'
    return "".join(s), color