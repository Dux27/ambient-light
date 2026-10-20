import time
from collections import deque

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ambient_light import config


def main():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2)  # V4L2 - Video4Linux2

    if not cap.isOpened():
        raise RuntimeError("Cannot open /dev/video0")

    # FOURCC - Four Character Code
    # *"MJPG" - "M", "J", "P", "G" - Motion JPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.REQUESTED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify the actual capture format and FPS.
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Capture format: {codec}")
    print(f"Capture FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")

    if config.REQUESTED_HEIGHT != height or config.REQUESTED_WIDTH != width:
        print(
            f"Invalid resolution. Auto adjusting: "
            f"{config.REQUESTED_WIDTH}x{config.REQUESTED_HEIGHT} "
            f"-> {width}x{height}"
        )

    # Physical LED layout: 144 top + 72 left + 72 right.
    # Extra row allows side strips to start below the top strip.
    led_preview_width = config.N_LED_TOP
    led_preview_height = config.N_LED_SIDE + 1

    # Initialize visualization only when enabled.
    fig = None

    if config.ENABLE_VISUALIZATION:
        plt.ion()  # Interactive mode on

        # Creating Figure and three child Axes objects.
        fig, (ax_video, ax_color, ax_led) = plt.subplots(1, 3)

        fig.canvas.manager.full_screen_toggle()

        fig.subplots_adjust(
            left=0.01,
            right=0.99,
            wspace=0.05,
        )

        fig.canvas.manager.set_window_title("HDMI Capture")
        fig.suptitle("Ambient Light Visualization")

        # === Video Axes ===
        video_image = ax_video.imshow(
            np.zeros((height, width, 3), dtype=np.uint8),
            interpolation="nearest",
            aspect="equal",
        )

        ax_video.set_title(f"Raw img resolution: {width}x{height}")
        ax_video.axis("off")

        # === Color Axes ===
        color_image = ax_color.imshow(
            np.zeros(
                (config.GRID_HEIGHT, config.GRID_WIDTH, 3),
                dtype=np.uint8,
            ),
            interpolation="nearest",
            aspect="equal",
        )

        ax_color.set_title(
            f"Downsampled img resolution: "
            f"{config.GRID_WIDTH}x{config.GRID_HEIGHT}"
        )
        ax_color.axis("off")

        # === LED Axes ===
        led_image = ax_led.imshow(
            np.zeros(
                (led_preview_height, led_preview_width, 3),
                dtype=np.uint8,
            ),
            interpolation="nearest",
            aspect="equal",
        )

        ax_led.set_title(
            f"LED layout: {config.N_LED_TOP} top + "
            f"{config.N_LED_SIDE} left + "
            f"{config.N_LED_SIDE} right"
        )
        ax_led.axis("off")

    fail_counter = 0

    # 101 timestamps represent 100 frame intervals.
    frame_times = deque(maxlen=101)

    # Track stage timings over the last 100 frames.
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
            color_grid = cv2.resize(
                frame,
                (config.GRID_WIDTH, config.GRID_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            color_grid = cv2.cvtColor(
                color_grid,
                cv2.COLOR_BGR2RGB,
            )

            # === LED Extraction ===
            # The first row belongs to the top strip.
            # Side strips start from the second row.
            # Extract colors while preserving the RGB image dimensions.
            left_colors = color_grid[1:, 0:1, :]    # (17, 1, 3)
            top_colors = color_grid[0:1, :, :]      # (1, 32, 3)
            right_colors = color_grid[1:, -1:, :]  # (17, 1, 3)

            # Map the extracted colors to the physical LED counts.
            # INTER_NEAREST duplicates colors without blending.
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

            # Extracted LED arrays:
            # left_leds  -> (72, 3)
            # top_leds   -> (144, 3)
            # right_leds -> (72, 3)

            t2 = time.perf_counter()

            # === Visualization ===
            if config.ENABLE_VISUALIZATION:

                # Convert full-resolution frame only for video preview.
                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                # Create U-shaped LED visualization matrix.
                # 73 rows x 144 columns, with a black interior.
                led_strip = np.zeros(
                    (led_preview_height, led_preview_width, 3),
                    dtype=np.uint8,
                )

                # Top: all 144 LEDs, including both corners.
                led_strip[0, :, :] = top_leds

                # Left and right: 72 LEDs each.
                led_strip[1:, 0, :] = left_leds
                led_strip[1:, -1, :] = right_leds

                # Update image Artists.
                video_image.set_data(rgb)
                color_image.set_data(color_grid)
                led_image.set_data(led_strip)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            t3 = time.perf_counter()

            # === Performance Measurements ===
            capture_times.append((t1 - t0) * 1000)
            processing_times.append((t2 - t1) * 1000)
            rendering_times.append((t3 - t2) * 1000)

            frame_times.append(t3)

            # Calculate rolling FPS and average stage timings.
            if len(frame_times) >= 2:
                elapsed = frame_times[-1] - frame_times[0]
                fps = (len(frame_times) - 1) / elapsed

                capture_ms = sum(capture_times) / len(capture_times)
                processing_ms = sum(processing_times) / len(processing_times)
                rendering_ms = sum(rendering_times) / len(rendering_times)

                # Clear terminal line and print updated measurements.
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
