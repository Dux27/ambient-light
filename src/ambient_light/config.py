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

# === Gaussian Color Extraction ===

# Number of physical LEDs sharing one calculated RGB color.
# Example for 288 LEDs:
# 2 = 144 independent color zones (72 top, 36 per side).
# 4 = 72 independent color zones (36 top, 18 per side).
# 8 = 36 independent color zones (18 top, 9 per side).
# Must evenly divide both N_LED_TOP and N_LED_SIDE.
LEDS_PER_GROUP: int = 4

# Gaussian standard deviation along the LED strip.
# Unit: LED color-zone widths.
#
# Controls how strongly pixels in neighboring zones influence a given zone.
# 0.5 = Pixels one zone away have 13.5% relative weight.
# 1.0 = Pixels one zone away have 60.7% relative weight.
# 1.2 = Pixels one zone away have 70.7% relative weight.
# 2.0 = Pixels one zone away have 88.2% relative weight.
# Larger values produce wider influence and smoother color transitions.
# These percentages are relative weights before normalization.
GAUSSIAN_ALONG_SPREAD: float = 1.3

# Gaussian standard deviation inward from the screen edge.
# Unit: fraction of screen height (top) or width (sides).
#
# Controls how strongly pixels deeper inside the screen influence the LEDs.
# Relative pixel weight at 20% depth:
# 0.10 = 13.5%
# 0.18 = 53.9%
# 0.30 = 80.1%
# Larger values allow deeper screen pixels to contribute more strongly.
# At a depth equal to the configured value, weight is always 60.7%.
# These are relative weights before normalization, not a hard cutoff.
GAUSSIAN_INWARD_SPREAD: float = 0.22

# === Visualization ===
# Maximum Matplotlib refresh rate, independent of capture FPS.
VISUALIZATION_FPS: int = 8
ENABLE_VISUALIZATION: bool = True
