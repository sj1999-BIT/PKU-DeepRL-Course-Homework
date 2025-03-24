import numpy as np
import matplotlib.pyplot as plt



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


def plot_progress_data(data, save_plot=False):
    # After training is complete, plot the loss graph

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.title('Training Loss over Time')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.grid(True)
    if save_plot:
        plt.savefig('loss_plot.png')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Generate a sample array with random values
    # This could be your actual array
    sample_array = np.random.normal(loc=50, scale=15, size=500).astype(int)

    # Plot the distribution
    plot_value_distribution(sample_array, num_bins=15, title="Distribution of Random Values")
