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

    # Initialize visualization only when enabled.
    fig = None
    video_image = None
    color_image = None

    if config.ENABLE_VISUALIZATION:
        plt.ion()  # Interactive mode on

        # Creating Figure and two child Axes objects.
        fig, (ax_video, ax_color) = plt.subplots(
            1, 2, figsize=(12, 5)
        )

        fig.canvas.manager.set_window_title("HDMI Capture")
        fig.suptitle("Ambient Light Visualization")

        # Create an image Artist for the captured video.
        video_image = ax_video.imshow(
            np.zeros((height, width, 3), dtype=np.uint8)
        )

        ax_video.set_title(f"Raw Image Resolution: {width}x{height}")
        ax_video.axis("off")

        # Create an image Artist for the extracted color grid.
        color_image = ax_color.imshow(
            np.zeros(
                (config.GRID_HEIGHT, config.GRID_WIDTH, 3),
                dtype=np.uint8,
            ),
            interpolation="nearest",
        )

        ax_color.set_title(
            f"Downsampled Image Resolution: "
            f"{config.GRID_WIDTH}x{config.GRID_HEIGHT}"
        )
        ax_color.axis("off")

    fail_counter = 0

    # 101 timestamps represent 100 frame intervals.
    frame_times = deque(maxlen=101)

    # Track stage timings over the last 100 frames.
    capture_times = deque(maxlen=100)
    processing_times = deque(maxlen=100)
    rendering_times = deque(maxlen=100)

    try:
        while fig is None or plt.fignum_exists(fig.number):
            # Capture frame.
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

            color_preview = cv2.resize(
                frame,
                (config.GRID_WIDTH, config.GRID_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            color_preview = cv2.cvtColor(
                color_preview, cv2.COLOR_BGR2RGB
            )

            # Full-resolution RGB conversion is only needed for visualization.
            if config.ENABLE_VISUALIZATION:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            t2 = time.perf_counter()

            # Visualization.
            if config.ENABLE_VISUALIZATION:
                video_image.set_data(rgb)
                color_image.set_data(color_preview)

                fig.canvas.draw_idle()
                fig.canvas.flush_events()

            t3 = time.perf_counter()

            # Store measurements in milliseconds.
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

                # f"\033[2K\r" - clears terminal line
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


if __name__ == "__main__":
    main()
