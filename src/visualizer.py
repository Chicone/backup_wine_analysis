import numpy as np
from matplotlib import pyplot as plt
from data_provider import AccuracyDataProvider

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


        # this is for year
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


        # this is for Beaume
        if label == 'NB':
            color = 'b'
        elif label == 'SB':
            color = 'g'

        # this is for merlot22
        if label[0] == 'A':
            color = 'b'
        elif label[0] == 'B':
            color = 'g'
        elif label[0] == 'C':
            color = 'r'


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
                elif 'merlot' in title.lower():
                    annotation = label[:3]
                    color = Visualizer.assign_color(label)
                elif 'cabernet' in title.lower():
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