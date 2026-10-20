import cv2
import matplotlib.pyplot as plt
import numpy as np

# 16:9 options: 1920x1080, 1280x720
# 4:3 options: 1600x1200, 1280x960, 1024x768, 800x600, 640x480
# Check supported modes - v4l2-ctl -d /dev/video0 --list-formats-ext
REQUESTED_WIDTH: int = 1280
REQUESTED_HEIGHT: int = 720

def main():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2) # V4LP - Video4Linux2

    if not cap.isOpened():
        raise RuntimeError("Cannot open /dev/video0")

    # FOURCC - Four Character Code
    # *"MJPG" - "M", "J", "P", "G" - Motion JPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT)
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
    if (REQUESTED_HEIGHT != height or REQUESTED_WIDTH != width):
        print(f"Invalid resolution. Auto adjusting: {REQUESTED_WIDTH}x{REQUESTED_HEIGHT} -> {width}x{height}")

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

            # Mean RGB value of all pixels.
            mean_rgb = rgb.mean(axis=(0, 1))
            mean_rgb_uint8 = np.clip(mean_rgb, 0, 255).astype(np.uint8)

            # Update live preview.
            video_image.set_data(rgb)

            # Display the mean color.
            color_image.set_data(
                np.full((100, 100, 3), mean_rgb_uint8, dtype=np.uint8)
            )
            color_title.set_text(
                f"Mean RGB: ({mean_rgb_uint8[0]}, {mean_rgb_uint8[1]}, {mean_rgb_uint8[2]})"
            )

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    except KeyboardInterrupt:
        plt.close(fig)
    finally:
        cap.release()
        plt.close(fig)


if __name__ == "__main__":
    main()
