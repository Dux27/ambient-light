# 16:9 options: 1920x1080, 1280x720
# 4:3 options: 1600x1200, 1280x960, 1024x768, 800x600, 640x480
# Check supported modes - v4l2-ctl -d /dev/video0 --list-formats-ext
REQUESTED_WIDTH: int = 640
REQUESTED_HEIGHT: int = 480

# LED layout: 72 left side + 144 top + 72 right side (288 LEDs total).
N_LED_TOP: int = 144
N_LED_SIDE: int = 72

# Downsampled image resolution for ambient light color extraction.
# Screen aspect ratio: 16:9.
# 4 LEDs per color group:
# - 18 vertical zones on each side (72 / 4).
# - 36 horizontal zones along the top (144 / 4).
# - 72 independent color zones in total.
#
# Recommended resolutions to keep 16:9 aspect ratio:
# 32x18   - Fast, lower spatial precision, matches vertical zones.
# 64x36   - Recommended: balanced detail and performance.
# 128x72  - Higher precision for more accurate color mapping.
#
# LED colors will be mapped proportionally to the image grid.
# Grid dimensions do not need to match the number of LED groups.
GRID_WIDTH: int = 32
GRID_HEIGHT: int = 18

ENABLE_VISUALIZATION: bool = True
