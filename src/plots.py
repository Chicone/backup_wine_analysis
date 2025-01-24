import matplotlib.pyplot as plt
import numpy as np

def plot_channel_selection_performance_changins():
    # Number of selected channels (x-axis)
    num_channels = list(range(1, 52))  # From step 1 to step 51

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
    plt.title('Incremental Channel Selection Performance (Changins dataset)')
    plt.legend(title='Metrics')
    plt.grid(True)
    plt.show()
