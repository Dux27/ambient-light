import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ambient_light import config


def createGaussianWeights(
    grid_width: int,
    grid_height: int,
    top_zones: int,
    side_zones: int,
) -> np.ndarray:
    """
    Precompute normalized Gaussian weights for every LED color zone.

    Zone order:
        1. Left side: top to bottom
        2. Top: left to right
        3. Right side: top to bottom

    Returns:
        (N_ZONES, GRID_WIDTH * GRID_HEIGHT) float32 array.
    """

    # Normalized coordinates of pixel centers in the color grid.
    x = (np.arange(grid_width, dtype=np.float32) + 0.5) / grid_width
    y = (np.arange(grid_height, dtype=np.float32) + 0.5) / grid_height

    X, Y = np.meshgrid(x, y)

    weights = []

    inward_spread = config.GAUSSIAN_INWARD_SPREAD
    along_spread = config.GAUSSIAN_ALONG_SPREAD

    # === Left Side ===
    for i in range(side_zones):
        center_y = (i + 0.5) / side_zones

        sigma_x = inward_spread
        sigma_y = along_spread / side_zones

        distance = (
            (X / sigma_x) ** 2
            + ((Y - center_y) / sigma_y) ** 2
        )

        weights.append(np.exp(-0.5 * distance))

    # === Top ===
    for i in range(top_zones):
        center_x = (i + 0.5) / top_zones

        sigma_x = along_spread / top_zones
        sigma_y = inward_spread

        distance = (
            ((X - center_x) / sigma_x) ** 2
            + (Y / sigma_y) ** 2
        )

        weights.append(np.exp(-0.5 * distance))

    # === Right Side ===
    for i in range(side_zones):
        center_y = (i + 0.5) / side_zones

        sigma_x = inward_spread
        sigma_y = along_spread / side_zones

        distance = (
            ((X - 1.0) / sigma_x) ** 2
            + ((Y - center_y) / sigma_y) ** 2
        )

        weights.append(np.exp(-0.5 * distance))

    # Shape: (N_ZONES, GRID_HEIGHT, GRID_WIDTH)
    weights = np.asarray(weights, dtype=np.float32)

    # Flatten screen dimensions.
    weights = weights.reshape(len(weights), -1)

    # Normalize each zone's weights so their sum equals 1.
    weights /= weights.sum(axis=1, keepdims=True)

    return weights


def extractLedColors(
    color_grid: np.ndarray,
    weights: np.ndarray,
    top_zones: int,
    side_zones: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate weighted RGB colors and expand them to physical LEDs.

    Returns:
        left_leds:  (N_LED_SIDE, 3)
        top_leds:   (N_LED_TOP, 3)
        right_leds: (N_LED_SIDE, 3)
    """

    # Convert RGB image into an array of RGB pixels.
    # Example: (18, 32, 3) -> (576, 3)
    pixels = color_grid.reshape(-1, 3).astype(np.float32)

    # Weighted average of every RGB channel for every zone.
    # Shape: (N_ZONES, 3)
    zone_colors = weights @ pixels

    zone_colors = np.clip(
        np.rint(zone_colors), 0, 255
    ).astype(np.uint8)

    # Split into left, top and right zones.
    left_colors = zone_colors[:side_zones]

    top_colors = zone_colors[
        side_zones : side_zones + top_zones
    ]

    right_colors = zone_colors[
        side_zones + top_zones :
    ]

    # Assign each calculated color to four physical LEDs.
    left_leds = np.repeat(
        left_colors,
        config.LEDS_PER_GROUP,
        axis=0,
    )

    top_leds = np.repeat(
        top_colors,
        config.LEDS_PER_GROUP,
        axis=0,
    )

    right_leds = np.repeat(
        right_colors,
        config.LEDS_PER_GROUP,
        axis=0,
    )

    return left_leds, top_leds, right_leds


def main():
    # === Capture Initialization ===
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError("Cannot open /dev/video0")

    # FOURCC - Four Character Code
    # *"MJPG" - Motion JPEG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.REQUESTED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify actual capture settings.
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Capture format: {codec}")
    print(f"Capture FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
    print(f"Capture resolution: {width}x{height}")

    if config.REQUESTED_HEIGHT != height or config.REQUESTED_WIDTH != width:
        print(
            f"Invalid resolution. Auto adjusting: "
            f"{config.REQUESTED_WIDTH}x{config.REQUESTED_HEIGHT} "
            f"-> {width}x{height}"
        )

    # === LED Group Configuration ===
    if (
        config.N_LED_TOP % config.LEDS_PER_GROUP != 0
        or config.N_LED_SIDE % config.LEDS_PER_GROUP != 0
    ):
        raise ValueError(
            "LED counts must be divisible by LEDS_PER_GROUP"
        )

    top_zones = config.N_LED_TOP // config.LEDS_PER_GROUP
    side_zones = config.N_LED_SIDE // config.LEDS_PER_GROUP

    total_zones = top_zones + 2 * side_zones

    print(f"LED color zones: {total_zones}")
    print(f"Top zones: {top_zones}")
    print(f"Side zones: {side_zones} each")

    # === Precompute Gaussian Weights ===
    weights = createGaussianWeights(
        config.GRID_WIDTH,
        config.GRID_HEIGHT,
        top_zones,
        side_zones,
    )

    print(f"Gaussian weights shape: {weights.shape}")

    # === Visualization Initialization ===
    fig = None

    visualization_interval = 1.0 / config.VISUALIZATION_FPS
    last_visualization_time = 0.0

    if config.ENABLE_VISUALIZATION:
        plt.ion()

        fig = plt.figure(figsize=(17, 9))

        gs = fig.add_gridspec(
            2, 6,
            height_ratios=[3, 2],
            hspace=0.25,
            wspace=0.15,
        )

        ax_video = fig.add_subplot(gs[0, 0:3])
        ax_color = fig.add_subplot(gs[0, 3:6])

        ax_left = fig.add_subplot(gs[1, 0])
        ax_top = fig.add_subplot(gs[1, 1:5])
        ax_right = fig.add_subplot(gs[1, 5])

        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            top=0.92,
            bottom=0.05,
        )

        fig.canvas.manager.set_window_title("HDMI Capture")
        fig.suptitle("Ambient Light - Gaussian Color Extraction")

        # === Video Axes ===
        video_image = ax_video.imshow(
            np.zeros((height, width, 3), dtype=np.uint8),
            interpolation="nearest",
            aspect="equal",
        )

        ax_video.set_title(f"Raw image: {width}x{height}")
        ax_video.axis("off")

        # === Color Grid Axes ===
        color_image = ax_color.imshow(
            np.zeros(
                (config.GRID_HEIGHT, config.GRID_WIDTH, 3),
                dtype=np.uint8,
            ),
            interpolation="nearest",
            aspect="equal",
        )

        ax_color.set_title(
            f"Color grid: {config.GRID_WIDTH}x{config.GRID_HEIGHT}"
        )
        ax_color.axis("off")

        # === Left LED Axes ===
        left_image = ax_left.imshow(
            np.zeros((config.N_LED_SIDE, 1, 3), dtype=np.uint8),
            interpolation="nearest",
            aspect="auto",
        )

        ax_left.set_title(f"Left: {config.N_LED_SIDE} LEDs")
        ax_left.axis("off")

        # === Top LED Axes ===
        top_image = ax_top.imshow(
            np.zeros((1, config.N_LED_TOP, 3), dtype=np.uint8),
            interpolation="nearest",
            aspect="auto",
        )

        ax_top.set_title(f"Top: {config.N_LED_TOP} LEDs")
        ax_top.axis("off")

        # === Right LED Axes ===
        right_image = ax_right.imshow(
            np.zeros((config.N_LED_SIDE, 1, 3), dtype=np.uint8),
            interpolation="nearest",
            aspect="auto",
        )

        ax_right.set_title(f"Right: {config.N_LED_SIDE} LEDs")
        ax_right.axis("off")

        fig.canvas.manager.full_screen_toggle()
        plt.show(block=False)

    # === Performance Measurements ===
    fail_counter = 0

    frame_times = deque(maxlen=101)
    capture_times = deque(maxlen=100)
    processing_times = deque(maxlen=100)
    rendering_times = deque(maxlen=100)

    try:
        while fig is None or plt.fignum_exists(fig.number):

            # === Capture ===
            t0 = time.perf_counter()

            ret, frame = cap.read()

            t1 = time.perf_counter()

            if not ret:
                fail_counter += 1
                print(
                    f"\nCapture card failed to provide a frame. "
                    f"Fail Counter: {fail_counter}"
                )
                continue

            # === Color Extraction ===
            # Downsample original BGR image to the color grid.
            color_grid = cv2.resize(
                frame,
                (config.GRID_WIDTH, config.GRID_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            color_grid = cv2.cvtColor(
                color_grid,
                cv2.COLOR_BGR2RGB,
            )

            # === Gaussian LED Extraction ===
            left_leds, top_leds, right_leds = extractLedColors(
                color_grid,
                weights,
                top_zones,
                side_zones,
            )

            # Physical RGB arrays:
            # left_leds  -> (72, 3)
            # top_leds   -> (144, 3)
            # right_leds -> (72, 3)

            t2 = time.perf_counter()

            # === Visualization ===
            if config.ENABLE_VISUALIZATION:
                now = time.perf_counter()

                if now - last_visualization_time >= visualization_interval:
                    # Full-frame RGB conversion only for preview.
                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB,
                    )

                    video_image.set_data(rgb)
                    color_image.set_data(color_grid)

                    left_image.set_data(
                        left_leds.reshape(config.N_LED_SIDE, 1, 3)
                    )

                    top_image.set_data(
                        top_leds.reshape(1, config.N_LED_TOP, 3)
                    )

                    right_image.set_data(
                        right_leds.reshape(config.N_LED_SIDE, 1, 3)
                    )

                    fig.canvas.draw_idle()
                    fig.canvas.flush_events()

                    last_visualization_time = time.perf_counter()

            t3 = time.perf_counter()

            # === Performance Measurements ===
            capture_times.append((t1 - t0) * 1000)
            processing_times.append((t2 - t1) * 1000)
            rendering_times.append((t3 - t2) * 1000)

            frame_times.append(t3)

            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed

                capture_ms = sum(capture_times) / len(capture_times)
                processing_ms = sum(processing_times) / len(processing_times)
                rendering_ms = sum(rendering_times) / len(rendering_times)

                print(
                    f"\033[2K\r"
                    f"FPS ({len(frame_times) - 1:3d}/100): {fps:6.2f} | "
                    f"Capture: {capture_ms:6.2f} ms | "
                    f"Processing: {processing_ms:6.2f} ms | "
                    f"Rendering: {rendering_ms:6.2f} ms",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()

        if fig is not None:
            plt.close(fig)

        print()


if __name__ == "__main__":
    main()
