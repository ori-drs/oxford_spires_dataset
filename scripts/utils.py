import csv
from matplotlib.colors import LinearSegmentedColormap


def get_BGYR_colourmap():
    colours = [
        (0, 0, 255),  # Blue
        (0, 255, 0),  # Green (as specified)
        (255, 255, 0),  # Yellow (as specified)
        (255, 0, 0),  # Red
    ]
    colours = [(r / 255, g / 255, b / 255) for r, g, b in colours]

    # Create the custom colormap
    n_bins = 100  # Number of color segments
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colours, N=n_bins)
    return cmap