# 16:9 options: 1920x1080, 1280x720
# 4:3 options: 1600x1200, 1280x960, 1024x768, 800x600, 640x480
# Check supported modes - v4l2-ctl -d /dev/video0 --list-formats-ext
REQUESTED_WIDTH: int = 640
REQUESTED_HEIGHT: int = 480

# Downsampled image resolution
# Recomended values are 
GRID_WIDTH: int = 16
GRID_HEIGHT: int = 9

ENABLE_VISUALIZATION: bool = True
