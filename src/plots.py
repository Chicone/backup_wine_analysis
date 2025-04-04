import matplotlib.pyplot as plt
import numpy as np
import re
import matplotlib
matplotlib.use('TkAgg')

def plot_channel_selection_performance_changins():

    # Test accuracy data (y-axis)
    test_accuracy = [
        0.7219, 0.7328, 0.7630, 0.7589, 0.7781, 0.7776, 0.7875, 0.7875, 0.7865, 0.7875,
        0.7906, 0.8016, 0.8052, 0.8021, 0.7937, 0.7885, 0.7932, 0.8089, 0.8047, 0.8052,
        0.8057, 0.8068, 0.8026, 0.8036, 0.8073, 0.8026, 0.8031, 0.8036, 0.8115, 0.8161,
        0.8167, 0.8167, 0.8141, 0.8130, 0.8120, 0.8125, 0.8130, 0.8135, 0.8146, 0.8151,
        0.8141, 0.8125, 0.8130, 0.8099, 0.8104, 0.8073, 0.8063, 0.8057, 0.8047, 0.8042,
        0.8047
    ]

    # Validation accuracy data
    validation_accuracy = [
        0.7161, 0.7453, 0.7635, 0.7755, 0.7906, 0.7927, 0.8042, 0.8063, 0.8104, 0.8130,
        0.8167, 0.8172, 0.8156, 0.8146, 0.8125, 0.8109, 0.8161, 0.8120, 0.8177, 0.8229,
        0.8255, 0.8281, 0.8281, 0.8292, 0.8318, 0.8313, 0.8286, 0.8297, 0.8286, 0.8328,
        0.8323, 0.8344, 0.8365, 0.8359, 0.8359, 0.8365, 0.8380, 0.8380, 0.8391, 0.8417,
        0.8396, 0.8396, 0.8401, 0.8406, 0.8422, 0.8427, 0.8411, 0.8401, 0.8391, 0.8385,
        0.8406
    ]

    # Channels added at each step
    channels_added = [
        27, 111, 44, 81, 10, 103, 21, 136, 29, 8,
        48, 58, 52, 119, 31, 99, 105, 109, 113, 6,
        75, 66, 70, 47, 114, 45, 17, 89, 158, 71,
        57, 3, 24, 5, 141, 42, 96, 170, 85, 107,
        7, 79, 80, 117, 61, 127, 28, 33, 137, 126,
        36
    ]
    channels_added = [
        13, 127, 3, 44, 53, 158, 45, 27, 80, 46,
        72, 64, 160, 168, 142, 19, 109, 48, 95, 71,
        26, 125, 99, 173, 47, 42, 110
    ]

    # Number of selected channels (x-axis)
    num_channels = list(range(1, len(channels_added) + 1))  # From step 1 to step 51

    # Plotting the data
    plt.figure(figsize=(12, 8))

    # Plot test accuracy with markers
    plt.plot(num_channels, test_accuracy, marker='o', label='Test Accuracy', color='b')

    # Plot validation accuracy with markers and dashed line
    plt.plot(num_channels, validation_accuracy, marker='x', linestyle='--', label='Validation Accuracy', color='r')

    # Annotate test accuracy points with corresponding channel numbers
    for i, (x, y, ch) in enumerate(zip(num_channels, test_accuracy, channels_added)):
        plt.annotate(str(ch), (x, y), textcoords="offset points", xytext=(5,5), ha='right', fontsize=6, color='blue')

    plt.xlabel('Number of Selected Channels')
    plt.ylabel('Accuracy')
    plt.title('Greedy Forward Selection Accuracy for Pinot noir (Changins dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_channel_selection_performance_isvv():

    # Test accuracy data (y-axis)
    test_accuracy = [
        0.5125, 0.5437, 0.5578, 0.5620, 0.5729, 0.5927, 0.5844, 0.5979, 0.6010, 0.6099,
        0.6167, 0.6083, 0.6208, 0.6161, 0.6151, 0.6177, 0.6245, 0.6203, 0.6208, 0.6104,
        0.6052, 0.6052, 0.6036, 0.6073, 0.6047, 0.6094, 0.6109
    ]

    # Validation accuracy data
    validation_accuracy = [
        0.5161, 0.5286, 0.5474, 0.5667, 0.5870, 0.5917, 0.6005, 0.6042, 0.6167, 0.6208,
        0.6286, 0.6307, 0.6354, 0.6370, 0.6359, 0.6339, 0.6370, 0.6401, 0.6411, 0.6417,
        0.6427, 0.6422, 0.6422, 0.6401, 0.6417, 0.6391, 0.6401
    ]

    # Channels added at each step
    channels_added = [
        13, 127, 3, 44, 53, 158, 45, 27, 80, 46,
        72, 64, 160, 168, 142, 19, 109, 48, 95, 71,
        26, 125, 99, 173, 47, 42, 110
    ]

    # Number of selected channels (x-axis)
    num_channels = list(range(1, len(channels_added) + 1))  # From step 1 to step 51

    # Plotting the data
    plt.figure(figsize=(12, 8))

    # Plot test accuracy with markers
    plt.plot(num_channels, test_accuracy, marker='o', label='Test Accuracy', color='b')

    # Plot validation accuracy with markers and dashed line
    plt.plot(num_channels, validation_accuracy, marker='x', linestyle='--', label='Validation Accuracy', color='r')

    # Annotate test accuracy points with corresponding channel numbers
    for i, (x, y, ch) in enumerate(zip(num_channels, test_accuracy, channels_added)):
        plt.annotate(str(ch), (x, y), textcoords="offset points", xytext=(5,5), ha='right', fontsize=6, color='blue')

    plt.xlabel('Number of Selected Channels')
    plt.ylabel('Accuracy')
    plt.title('Greedy Forward Selection Accuracy for Pinot noir (ISVV dataset)')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_channel_selection_thresholds(data):
    """
    Parses the given log data to extract test accuracy progression and plots the results.

    Parameters:
        data (str): The multi-line string containing step-by-step results of the channel selection process.
    """
    # Regex pattern to extract correlation threshold and test accuracies
    threshold_pattern = re.compile(r'Processing correlation_threshold = ([\d\.]+)')
    step_pattern = re.compile(r'Step \d+: Added Channel \d+ - Validation Accuracy: ([\d\.]+), Test Accuracy: ([\d\.]+)')

    accuracy_progressions = {}
    current_threshold = None

    for line in data.splitlines():
        threshold_match = threshold_pattern.search(line)
        step_match = step_pattern.search(line)

        if threshold_match:
            current_threshold = float(threshold_match.group(1))
            accuracy_progressions[current_threshold] = []

        if step_match and current_threshold is not None:
            test_accuracy = float(step_match.group(2))  # Extracting test accuracy
            accuracy_progressions[current_threshold].append(test_accuracy)

    # Plot the results
    plt.figure(figsize=(12, 8))
    for correlation_threshold, accuracies in accuracy_progressions.items():
        plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-',
                 label=f'Threshold {correlation_threshold:.2f}')

    plt.xlabel("Number of Selected Channels")
    plt.ylabel("Balanced Test Accuracy")
    plt.title("Incremental Channel Selection Performance Across Correlation Thresholds (ISVV)")
    plt.legend(title="Correlation Threshold", loc="lower right")
    plt.grid()
    plt.tight_layout()
    plt.show()

    return accuracy_progressions


def plot_accuracy_all_methods():
    """
    Plots accuracy results from the given LaTeX table using a bar chart.
    """

    # Methods and their corresponding accuracies for ISVV and Changins datasets
    methods = [
        "TIC",
        "TIS",
        "TIC ∘ TIS",
        "Channel Averaging",
        "Single Channels",
        "Greedy Ranking-based",
        "Greedy Forward",
        "Correlation Filtering"
    ]

    # Accuracy values from the LaTeX table
    isvv_accuracies = [0.534, 0.484, 0.529, 0.000, 0.535, 0.566, 0.625, 0.628]
    changins_accuracies = [0.695, 0.715, 0.666, 0.000, 0.718, 0.756, 0.817, 0.000]

    x = np.arange(len(methods))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar plots for ISVV and Changins datasets
    bars1 = ax.bar(x - width / 2, isvv_accuracies, width, label='ISVV', color='skyblue')
    bars2 = ax.bar(x + width / 2, changins_accuracies, width, label='Changins', color='lightcoral')

    # Adding accuracy values on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

    # Formatting the plot
    ax.set_xlabel('Methods')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of Accuracy Across Methods and Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.legend(title="Dataset")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

def plot_accuracy_vs_decimation(wine_type):
    """
    Plots accuracy vs. decimation factor for TIC, TIS, and TIC-TIS feature extraction methods.
    The data is hardcoded within the function.
    """
    import matplotlib
    matplotlib.use('TkAgg')  # Ensure TkAgg backend is used (better for debugging)



    # Hardcoded accuracy data for each method
    if wine_type == 'merlot':
        # Merlot
        decimation_factors = np.array([    1,     2,     3,     4,     5,    10,    20,    30,    40,    50,   100,   500,  1000])
        accuracy_tic =       np.array([0.843, 0.841, 0.841, 0.832, 0.850, 0.851, 0.862, 0.842, 0.825, 0.837, 0.807, 0.692, 0.665])
        accuracy_tis =       np.array([0.640, 0.630, 0.633, 0.611, 0.621, 0.641, 0.706, 0.844, 0.774, 0.795, 0.779, 0.683, 0.646])
        accuracy_tic_tis =   np.array([0.843, 0.840, 0.839, 0.829, 0.848, 0.849, 0.873, 0.850, 0.827, 0.833, 0.799, 0.696, 0.711])
        accuracy_concat =    np.array([0.844, 0.843, 0.832, 0.838, 0.841, 0.838, 0.845, 0.830, 0.871, 0.850, 0.853, 0.756, 0.764])
        title = "Accuracy vs. Decimation Factor (Merlot)"
    elif wine_type == 'cabernet_sauvignon':
        # # Cabernet Sauvignon
        decimation_factors = np.array([    1,     2,     3,     4,     5,    10,    20,    30,    40,    50,   100,   500,  1000])
        accuracy_tic =       np.array([0.673, 0.676, 0.669, 0.681, 0.661, 0.648, 0.657, 0.616, 0.586, 0.692, 0.673, 0.633, 0.645])
        accuracy_tis =       np.array([0.496, 0.495, 0.484, 0.460, 0.508, 0.473, 0.461, 0.565, 0.571, 0.569, 0.621, 0.593, 0.568])
        accuracy_tic_tis =   np.array([0.677, 0.674, 0.675, 0.666, 0.659, 0.640, 0.635, 0.623, 0.563, 0.685, 0.684, 0.613, 0.563])
        accuracy_concat =    np.array([0.703, 0.689, 0.696, 0.682, 0.688, 0.692, 0.688, 0.688, 0.696, 0.715, 0.736, 0.520, 0.445])
        title = "Accuracy vs. Decimation Factor (Cabernet Sauvignon)"
    else:
        raise ValueError("Invalid wine type. Use 'merlot' or 'cabernet_sauvignon'.")

    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(decimation_factors, accuracy_tic, marker='o', linestyle='-', label="TIC", linewidth=2)
    plt.plot(decimation_factors, accuracy_tis, marker='s', linestyle='--', label="TIS", linewidth=2)
    plt.plot(decimation_factors, accuracy_tic_tis, marker='^', linestyle='-.', label="TIC-TIS", linewidth=2)
    plt.plot(decimation_factors, accuracy_concat, marker='d', linestyle=':', label="All m/z", linewidth=2)  # Changed marker for differentiation

    # Labels and formatting
    plt.xlabel("Decimation Factor", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title(title, fontsize=14)
    # plt.title("Accuracy vs. Decimation Factor (Cabernet Sauvignon)", fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.xscale("log")  # Log scale for better visualization
    plt.ylim(0.4, 0.9)  # Adjust y-axis for better visibility

    # Show plot
    plt.show()


def plot_press_wines_accuracies():
    import matplotlib
    matplotlib.use('TkAgg')  # Ensure TkAgg backend is used (better for debugging)
    # Data
    algorithms = ["TIC", "TIS", "TIC-TIS", "All m/z", "Best m/z",
                  "Greedy ranked (TIC-TIS)", "Greedy ranked (concat.)",
                  "Greedy add (TIC-TIS)", "Greedy add (concat.)",
                  "Greedy remove (TIC-TIS)", "Greedy remove (concat.)"]

    merlot_acc = [0.844, 0.612, 0.839, 0.833, 0.789,
                  0.853, 0.839, 0.844, 0.836, 0.850, 0.833]

    csauv_acc = [0.637, 0.524, 0.637, 0.679, 0.699,
                 0.730, 0.729, 0.704, 0.724, 0.714, 0.696]

    x = np.arange(len(algorithms))  # Label locations

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.4

    ax.bar(x - bar_width/2, merlot_acc, bar_width, label="Merlot", color='royalblue', alpha=0.8)
    ax.bar(x + bar_width/2, csauv_acc, bar_width, label="C. Sauv.", color='tomato', alpha=0.8)

    # Labels and formatting
    ax.set_xlabel("Algorithm", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Comparison of Classification Accuracy for Different Algorithms", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms, rotation=45, ha="right")
    ax.legend()

    # Grid and layout adjustments
    ax.set_ylim(0, 1)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # Show plot
    plt.show()


def plot_histogram_correlation(file1, file2, wine1="Wine 1", wine2="Wine 2", threshold=0, show_plots=True):
    """
    Load two CSV files containing histogram data from different runs, construct histograms,
    plot histograms for each dataset, and examine their correlation.

    Args:
        file1 (str): Path to the first CSV file.
        file2 (str): Path to the second CSV file.
        wine1 (str): Name of the first wine.
        wine2 (str): Name of the second wine.
        threshold (int): Minimum count required for a bin to be considered in the analysis.

    Returns:
        None
    """
    import pandas as pd
    import ast
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    def load_and_flatten_csv(file_path):
        """Load a CSV file, parse lists from strings, and flatten to a single NumPy array."""
        df = pd.read_csv(file_path, skiprows=1, header=None)  # Skip the first row (header)
        parsed_data = df[0].apply(ast.literal_eval)  # Convert string representation to lists
        flattened_data = np.concatenate(parsed_data.values)  # Flatten the list of lists
        return flattened_data

    # Load data from CSV files
    data1 = load_and_flatten_csv(file1)
    data2 = load_and_flatten_csv(file2)

    # Bin edges covering all channels from both datasets
    all_data = np.concatenate([data1, data2])
    bins = np.arange(all_data.min(), all_data.max() + 2) - 0.5

    # Compute histograms
    hist1, _ = np.histogram(data1, bins=bins)
    hist2, _ = np.histogram(data2, bins=bins)

    # Apply threshold: Select only bins where at least one dataset exceeds the threshold
    mask = (hist1 > threshold) | (hist2 > threshold)
    bin_centers = (bins[:-1] + bins[1:]) / 2  # Compute the center of each bin
    filtered_channels = bin_centers[mask]     # Use bin_centers instead of unique_channels
    hist1 = hist1[mask]
    hist2 = hist2[mask]

    X = hist1.reshape(-1, 1)
    y = hist2

    # Fit linear regression line
    X = hist1.reshape(-1, 1)
    y = hist2
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r_squared = reg.score(X, y)


    if show_plots:

        # Create figure with improved aesthetics
        fig, axes = plt.subplots(1, 3, figsize=(22, 6), gridspec_kw={'width_ratios': [1, 1, 1.2]})
        plt.subplots_adjust(wspace=0.4)

        # Colors
        color1 = '#4C72B0'  # Blue
        color2 = '#C44E52'  # Red
        scatter_color = '#8172B3'  # Purple

        # Histogram for File 1
        axes[0].bar(filtered_channels, hist1, width=1, color=color1, alpha=0.8, edgecolor='black', label=f"{wine1} Histogram")
        # axes[0].set_title(f"Histogram of Channels ({wine1})", fontsize=14)
        axes[0].set_title(wine1, fontsize=14)
        axes[0].set_xlabel("Channel Index", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].grid(axis='y', linestyle='--', alpha=0.5)
        axes[0].tick_params(axis='x', rotation=45)  # Rotate x labels for better readability
        # axes[0].legend()

        # Histogram for File 2
        axes[1].bar(filtered_channels, hist2, width=1, color=color2, alpha=0.8, edgecolor='black', label=f"{wine2} Histogram")
        # axes[1].set_title(f"Histogram of Channels ({wine2})", fontsize=14)
        axes[1].set_title(wine2, fontsize=14)
        axes[1].set_xlabel("Channel Index", fontsize=12)
        axes[1].set_ylabel("Count", fontsize=12)
        axes[1].grid(axis='y', linestyle='--', alpha=0.5)
        axes[1].tick_params(axis='x', rotation=45)
        # axes[1].legend()

        # Scatter plot for histogram correlation
        axes[2].scatter(hist1, hist2, alpha=0.8, color=scatter_color, edgecolors='black', label="Histogram Correlation")
        axes[2].plot([0, max(hist1.max(), hist2.max())], [0, max(hist1.max(), hist2.max())], 'r--',
                     label="y = x (perfect correlation)")
        axes[2].set_xlabel(f"Histogram Counts in wine 1", fontsize=12)
        axes[2].set_ylabel(f"Histogram Counts in wine 2", fontsize=12)
        # axes[2].set_title(f"Histogram Correlation: {wine1} vs {wine2}", fontsize=14)
        axes[2].set_title(f"Histogram Correlation", fontsize=14)
        axes[2].legend()
        axes[2].grid(True)

        # Plot regression line
        x_vals = np.linspace(hist1.min(), hist1.max(), 100).reshape(-1, 1)
        y_fit = reg.predict(x_vals)
        axes[2].plot(x_vals, y_fit, 'g-', label='Linear Fit')

    # Compute and print Pearson correlation coefficient
    correlation = np.corrcoef(hist1, hist2)[0, 1]

    if show_plots:
        print(f"Pearson Correlation Between {wine1} and {wine2}: {correlation:.3f}")
        axes[2].text(0.65 * max(hist1), 0.9 * max(hist2),  # Position it near the top-left of the plot
                     f"Pearson: {correlation:.3f}", fontsize=16, color='r',
                     bbox=dict(facecolor='white', alpha=0.7))  # Background box for visibility
        plt.tight_layout()
        plt.show()

        a = reg.coef_[0]
        b = reg.intercept_
        axes[2].text(
            0.05 * max(hist1), 0.90 * max(hist2),
            f"$y = {a:.2f}x + {b:.2f}$\n$R = {r_squared**0.5:.3f}$",
            fontsize=14, color='g',
            bbox=dict(facecolor='white', alpha=0.7)
        )

        plt.tight_layout()
        plt.show()

    return correlation
