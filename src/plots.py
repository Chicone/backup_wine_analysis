import matplotlib.pyplot as plt
import numpy as np
import re

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
