import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

# === Configuration ===
GRID_WIDTH = 64
GRID_HEIGHT = 36
N_LED_TOP = 144

INITIAL_GROUP_SIZE = 4
INITIAL_ALONG_SPREAD = 1.2
INITIAL_INWARD_SPREAD = 0.18


# === Normalized Pixel Coordinates ===
x = (np.arange(GRID_WIDTH, dtype=np.float32) + 0.5) / GRID_WIDTH
y = (np.arange(GRID_HEIGHT, dtype=np.float32) + 0.5) / GRID_HEIGHT

X, Y = np.meshgrid(x, y)


def calculate_weights(
    group_size: int,
    along_spread: float,
    inward_spread: float,
) -> np.ndarray:

    top_zones = N_LED_TOP // group_size

    # Center the Gaussian on the middle of the top strip.
    center_x = 0.5

    # Gaussian standard deviations.
    sigma_x = along_spread / top_zones
    sigma_y = inward_spread

    # Squared normalized distance.
    distance = (
        ((X - center_x) / sigma_x) ** 2
        + (Y / sigma_y) ** 2
    )

    return np.exp(-0.5 * distance)


def main():
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 7))
    fig.canvas.manager.set_window_title("Gaussian LED Sampling")

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.90,
        bottom=0.28,
    )

    fig.suptitle("Gaussian LED Color Extraction")

    weights = calculate_weights(
        INITIAL_GROUP_SIZE,
        INITIAL_ALONG_SPREAD,
        INITIAL_INWARD_SPREAD,
    )

    image = ax.imshow(
        weights,
        cmap="inferno",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        extent=(0, 1, 1, 0),
        aspect="equal",
    )

    # Mark the Gaussian sampling center.
    ax.plot(
        [0.5],
        [0],
        marker="o",
        color="cyan",
        markersize=8,
    )

    ax.set_xlabel("Normalized screen width")
    ax.set_ylabel("Normalized screen height")

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Relative pixel weight")

    # === Sliders ===
    ax_along = fig.add_axes((0.18, 0.17, 0.68, 0.03))
    ax_inward = fig.add_axes((0.18, 0.11, 0.68, 0.03))
    ax_group = fig.add_axes((0.18, 0.05, 0.68, 0.03))

    slider_along = Slider(
        ax=ax_along,
        label="Along spread",
        valmin=0.3,
        valmax=5.0,
        valinit=INITIAL_ALONG_SPREAD,
        valstep=0.1,
    )

    slider_inward = Slider(
        ax=ax_inward,
        label="Inward spread",
        valmin=0.02,
        valmax=0.6,
        valinit=INITIAL_INWARD_SPREAD,
        valstep=0.01,
    )

    # Valid divisors of 144 LEDs.
    valid_group_sizes = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24]

    slider_group = Slider(
        ax=ax_group,
        label="LEDs per group",
        valmin=min(valid_group_sizes),
        valmax=max(valid_group_sizes),
        valinit=INITIAL_GROUP_SIZE,
        valstep=valid_group_sizes,
        valfmt="%d",
    )

    def update(_):
        group_size = int(slider_group.val)
        top_zones = N_LED_TOP // group_size

        weights = calculate_weights(
            group_size,
            slider_along.val,
            slider_inward.val,
        )

        image.set_data(weights)

        ax.set_title(
            f"LEDs/group: {group_size} | "
            f"Top zones: {top_zones} | "
            f"σ along: {slider_along.val:.2f} | "
            f"σ inward: {slider_inward.val:.2f}"
        )

        fig.canvas.draw_idle()

    slider_along.on_changed(update)
    slider_inward.on_changed(update)
    slider_group.on_changed(update)

    update(None)
    plt.show()


if __name__ == "__main__":
    main()
