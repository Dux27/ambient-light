import cv2
import matplotlib.pyplot as plt
import numpy as np


def main():
    cap = cv2.VideoCapture("/dev/video0", cv2.CAP_V4L2) # V4LP - Video4Linux2

    if not cap.isOpened():
        raise RuntimeError("Cannot open /dev/video0")


    # FOURCC - Four Character Code
    # *"MJPG" - "M", "J", "P", "G" - Motion JPG
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # 16:9 options: 1920x1080, 1600x900, 1280x720, 960x540, 480x270
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    plt.ion()

    fig, (ax_video, ax_color) = plt.subplots(
        1, 2, figsize=(12, 5)
    )
    fig.canvas.manager.set_window_title("HDMI Capture - Mean Color")

    video_image = ax_video.imshow(
        np.zeros((1080, 1920, 3), dtype=np.uint8)
    )
    ax_video.set_title("Capture card")
    ax_video.axis("off")

    color_image = ax_color.imshow(
        np.zeros((100, 100, 3), dtype=np.uint8)
    )
    color_title = ax_color.set_title("Mean RGB: (0, 0, 0)")
    ax_color.axis("off")

    try:
        while plt.fignum_exists(fig.number):
            ret, frame = cap.read()

            if not ret:
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
        pass
    finally:
        cap.release()
        plt.close(fig)


if __name__ == "__main__":
    main()
