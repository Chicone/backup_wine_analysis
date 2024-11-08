import numpy as np
from matplotlib import pyplot as plt
from data_provider import AccuracyDataProvider
import pandas as pd

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
