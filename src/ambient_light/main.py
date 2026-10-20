import cv2
import matplotlib.pyplot as plt
import numpy as np

from ambient_light import config, layouts


def main():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2) # V4LP - Video4Linux2

    if not cap.isOpened():
        raise RuntimeError("Cannot open /dev/video0")

    # FOURCC - Four Character Code
    # *"MJPG" - "M", "J", "P", "G" - Motion JPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.REQUESTED_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    plt.ion() # interactive mode on

    # Creating Figure and two child components of it (two Axes objects)
    # rows, columns, size of the the figure in inches (depends on DPI - dot per inch)
    fig, (ax_video, ax_color) = plt.subplots(
        1, 2, figsize=(12, 5)
    )
    fig.canvas.manager.set_window_title("HDMI Capture")
    fig.suptitle("Ambient Light - Mean Value Test")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (config.REQUESTED_HEIGHT != height or config.REQUESTED_WIDTH != width):
        print(f"Invalid resolution. Auto adjusting: {config.REQUESTED_WIDTH}x{config.REQUESTED_HEIGHT} -> {width}x{height}")

    # Create an image Artist in ax_video and store a reference to it.
    video_image = ax_video.imshow(
        np.zeros((height, width, 3), dtype=np.uint8)
    )

    ax_video.set_title(f"Resolution: {width}x{height}")
    ax_video.axis("off")

    color_image = ax_color.imshow(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    color_title = ax_color.set_title("Mean RGB: (0, 0, 0)")
    ax_color.axis("off")

    fail_counter = 0
    try:
        # fig.number atribute is and identifier of the figure
        while plt.fignum_exists(fig.number):
            # ret - boolean indicating whether reading the frame succeeded
            # frame - the captured image, represented as a NumPy array
            ret, frame = cap.read()

            if not ret:
                fail_counter += 1
                print(f"Capture card failed to provide a frame. Fail Counter: {fail_counter}")
                continue

            # OpenCV uses BGR; convert to RGB.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb = cv2.resize(
                rgb,
                None,
                fx=config.DOWNSCALE_FACTOR,
                fy=config.DOWNSCALE_FACTOR,
                interpolation=cv2.INTER_AREA,
            )

            extracted_rgb = layouts.extractColors(rgb, config.EXTRACTION_TYPE)

            # Update live preview.
            video_image.set_data(rgb)

            # Determine preview grid dimensions.
            if config.EXTRACTION_TYPE.endswith("x") and config.EXTRACTION_TYPE[:-1].isdigit():
                columns = int(config.EXTRACTION_TYPE[:-1])
                rows = max(1, round(columns * rgb.shape[0] / rgb.shape[1]))
            else:
                columns = len(extracted_rgb)
                rows = 1

            # Create color preview.
            block_size = config.BLOCK_SIZE
            color_preview = np.zeros(
                (rows * block_size, columns * block_size, 3),
                dtype=np.uint8,
            )

            # Fill each grid cell with its extracted RGB color.
            for i, color in enumerate(extracted_rgb):
                row = i // columns
                col = i % columns

                y_start = row * block_size
                x_start = col * block_size

                color_preview[
                    y_start : y_start + block_size,
                    x_start : x_start + block_size,
                ] = color

            color_image.set_data(color_preview)
            color_title.set_text(f"RGB Extraction: {config.EXTRACTION_TYPE}")

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        plt.close(fig)
    finally:
        cap.release()
        plt.close(fig)


if __name__ == "__main__":
    main()
