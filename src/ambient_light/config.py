# 16:9 options: 1920x1080, 1280x720
# 4:3 options: 1600x1200, 1280x960, 1024x768, 800x600, 640x480
# Check supported modes - v4l2-ctl -d /dev/video0 --list-formats-ext
REQUESTED_WIDTH: int = 1280
REQUESTED_HEIGHT: int = 720

# Scale factor applied to image width and height (0.0–1.0).
# 1.0 = original size; 0.5 = half width and height (25% of pixels),
# 0.25 = quarter width and height (6.25% of pixels).
DOWNSCALE_FACTOR: float = 0.75

# Color extraction types:
# "MEAN" - Mean RGB value of all pixels.
# "Nx"   - Grid with N columns; rows are calculated automatically
#          to approximate the input image's aspect ratio of 16:9.
#          Examples: "16x", "32x".
EXTRACTION_TYPE: str = "MEAN"

# Size of the matplotlib visualization block
BLOCK_SIZE = 100
