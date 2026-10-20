# 16:9 options: 1920x1080, 1280x720
# 4:3 options: 1600x1200, 1280x960, 1024x768, 800x600, 640x480
# Check supported modes - v4l2-ctl -d /dev/video0 --list-formats-ext
REQUESTED_WIDTH: int = 640
REQUESTED_HEIGHT: int = 480

# Downsampled image resolution for ambient light color extraction.
# Screen aspect ratio: 16:9.
# LED layout: 144 left + 288 top + 144 right (576 LEDs total).
# 4 LEDs per color group:
# - 36 vertical zones on each side.
# - 72 horizontal zones along the top.
#
# Recommended resolutions:
# 32x18   - Fast, lower spatial precision.
# 64x36   - Recommended: exact 16:9, matches vertical zones.
# 128x72  - Higher precision, covers all LED zones.
#
# LED colors will be mapped proportionally to the image grid.
# Grid dimensions do not need to match the number of LED groups.
GRID_WIDTH: int = 32
GRID_HEIGHT: int = 18

ENABLE_VISUALIZATION: bool = False
