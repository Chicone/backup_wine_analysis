import numpy as np
from matplotlib import pyplot as plt
class Visualizer:
    """
    A class used to visualize data in various forms such as scatter plots and 3D stacked plots.
    """

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
    def plot_results(result, title, xlabel, ylabel):
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
            for label in labels:
                plt.scatter(-result.loc[label, xlabel],
                            -result.loc[label, ylabel],
                            s=28,
                            c=[Visualizer.change_letter_and_color(label)[1]])
                plt.annotate(Visualizer.change_letter_and_color(label)[0],
                             (-result.loc[label, xlabel],
                              -result.loc[label, ylabel]),
                             fontsize=9,
                             color=Visualizer.change_letter_and_color(label)[1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.show()