import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.cm import get_cmap


def visualize_patches(
    tile_coordinates, tile_width, tile_height, tile_labels, save_path
):
    cmap_colours = get_cmap("tab10").colors
    keys = list(range(13))
    colourmap = dict(zip(keys, cmap_colours))

    min_x, min_y = tile_coordinates[:, 0].min(), tile_coordinates[:, 1].min()
    max_x = tile_coordinates[:, 0].max() + tile_width
    max_y = tile_coordinates[:, 1].max() + tile_height

    if max_x > max_y:
        max_y = max_x
    else:
        max_x = max_y

    fig, ax = plt.subplots(1, figsize=(8, 8))
    for i, tile in enumerate(tile_coordinates):
        x, y = tile[0], tile[1]
        rect = patches.Rectangle(
            (x, y),
            tile_width,
            tile_height,
            alpha=0.7,
            facecolor=colourmap[tile_labels[i]],
        )
        ax.add_patch(rect)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)

    plt.gca().invert_yaxis()
    plt.axis("off")
    fig.tight_layout()
    plt.savefig(save_path)
