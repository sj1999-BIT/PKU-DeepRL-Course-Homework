import numpy as np
import matplotlib.pyplot as plt
from data import load_array_from_file



def plot_value_distribution(array, num_bins=10, title="Value Distribution"):
    """
    Plots a histogram/bar graph showing the distribution of values in an array.

    Parameters:
    -----------
    array : list or numpy.ndarray
        The array of values to analyze
    num_bins : int, optional
        Number of bins/bars to use (default is 10)
    title : str, optional
        Title for the plot (default is "Value Distribution")
    """
    # Convert to numpy array if it's not already
    if not isinstance(array, np.ndarray):
        array = np.array(array)

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create histogram/bar plot
    counts, bins, patches = ax.hist(array, bins=num_bins, edgecolor='black', alpha=0.7)

    # Add count labels above each bar
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, x in zip(counts, bin_centers):
        # Only add label if the count is greater than 0
        if count > 0:
            ax.annotate(f'{int(count)}',
                        xy=(x, count),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    # Add labels and title
    ax.set_xlabel('Values')
    ax.set_ylabel('Frequency')
    ax.set_title(title)

    # Add grid lines for better readability
    ax.grid(axis='y', alpha=0.75, linestyle='--')

    # Show plot
    plt.tight_layout()
    plt.show()


def plot_progress_data(data, save_plot=False, plot_file_title=None):
    # After training is complete, plot the loss graph

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title('Training Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_plot:
        if plot_file_title is None:
            plot_file_title = "plot.png"
        plt.savefig(plot_file_title)
    plt.show()


def plot_frequency(A, B, save_plot=False):
    # Count frequency of each value in B that appears in A
    frequencies = []
    for value in B:
        frequency = np.sum(np.array(A) == value)
        frequencies.append(frequency)

    # Create bar graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(B, frequencies)

    # Add labels and title
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.title('Frequency of Values in Array A')

    # Add frequency labels on top of bars
    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{freq}', ha='center', va='bottom')

    # Display the plot
    plt.xticks(B)
    plt.tight_layout()
    if save_plot:
        plt.savefig("frequency.png")

    plt.show()


# Example usage:
if __name__ == "__main__":
    policy_loss_arr = load_array_from_file("./results and data/2_1900_epoch/policy_loss.txt")
    plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")

    policy_loss_arr = load_array_from_file("./results and data/2_1900_epoch/value_loss.txt")
    plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")

    policy_loss_arr = load_array_from_file("./results and data/2_1900_epoch/reward_plot.txt")
    plot_progress_data(policy_loss_arr, save_plot=True, plot_file_title="policy_loss")
