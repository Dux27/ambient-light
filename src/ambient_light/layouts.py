import numpy as np


def extractColors(rgb: np.ndarray, layout: str) -> list[np.ndarray]:
    height, width, _ = rgb.shape

    match layout:
        case "MEAN":
            mean_rgb = rgb.mean(axis=(0, 1))
            return [np.clip(mean_rgb, 0, 255).astype(np.uint8)]

        case _ if layout.endswith("x") and layout[:-1].isdigit():
            columns = int(layout[:-1])

            if columns < 1:
                raise ValueError("Number of columns must be positive")

            rows = max(1, round(columns * height / width))
            colors = []

            for row in range(rows):
                for col in range(columns):
                    y_start = row * height // rows
                    y_end = (row + 1) * height // rows
                    x_start = col * width // columns
                    x_end = (col + 1) * width // columns

                    region = rgb[y_start:y_end, x_start:x_end]
                    mean_rgb = region.mean(axis=(0, 1))

                    colors.append(
                        np.clip(mean_rgb, 0, 255).astype(np.uint8)
                    )

            return colors

        case _:
            raise ValueError(f"Unknown layout: {layout}")
