import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ambient_light import config


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

    # === Visualization Initialization ===
    fig = None
    visualization_fps = 8
    visualization_interval = 1.0 / visualization_fps
    last_visualization_time = 0.0

    if config.ENABLE_VISUALIZATION:
        plt.ion()

        # Two rows, six columns:
        # Top: original video + color grid
        # Bottom: left strip + top strip + right strip
        fig = plt.figure(figsize=(17, 9))

        gs = fig.add_gridspec(
            2, 6,
            height_ratios=[3, 2],
            width_ratios=[1, 1, 1, 1, 1, 1],
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
        fig.suptitle("Ambient Light Visualization")

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

    # 101 timestamps represent 100 frame intervals.
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
            # Downsample original BGR frame to the configured grid.
            color_grid = cv2.resize(
                frame,
                (config.GRID_WIDTH, config.GRID_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            # Convert only the downsampled grid to RGB.
            color_grid = cv2.cvtColor(
                color_grid,
                cv2.COLOR_BGR2RGB,
            )

            # === LED Extraction ===
            # The top strip uses the first row.
            # Side strips start from the second row.
            left_colors = color_grid[1:, 0:1, :]
            top_colors = color_grid[0:1, :, :]
            right_colors = color_grid[1:, -1:, :]

            # Expand extracted colors to the physical LED counts.
            # INTER_NEAREST duplicates source colors without blending.
            left_leds = cv2.resize(
                left_colors,
                (1, config.N_LED_SIDE),
                interpolation=cv2.INTER_NEAREST,
            ).reshape(config.N_LED_SIDE, 3)

            top_leds = cv2.resize(
                top_colors,
                (config.N_LED_TOP, 1),
                interpolation=cv2.INTER_NEAREST,
            ).reshape(config.N_LED_TOP, 3)

            right_leds = cv2.resize(
                right_colors,
                (1, config.N_LED_SIDE),
                interpolation=cv2.INTER_NEAREST,
            ).reshape(config.N_LED_SIDE, 3)

            # LED arrays always available:
            # left_leds  -> (72, 3)
            # top_leds   -> (144, 3)
            # right_leds -> (72, 3)

            t2 = time.perf_counter()

            # === Visualization ===
            if config.ENABLE_VISUALIZATION:
                now = time.perf_counter()

                if now - last_visualization_time >= visualization_interval:
                    # Full-resolution RGB is needed only for video preview.
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # Update original video and color grid.
                    video_image.set_data(rgb)
                    color_image.set_data(color_grid)

                    # Display the physical LED arrays.
                    # Preserve a single column/row dimension for imshow().
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
