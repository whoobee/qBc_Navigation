"""
Pi Camera intrinsic parameters for IBVS computation.

Default values for Pi Camera v3 (Sony IMX708) at 1920x1080.
These can be replaced with calibrated values for better accuracy.
"""

# Image resolution
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# Focal length in pixels (approximate for Pi Camera v3 at 1920x1080)
FX = 1330.0
FY = 1330.0

# Principal point (image center)
CX = IMG_WIDTH / 2.0   # 960.0
CY = IMG_HEIGHT / 2.0  # 540.0


def pixel_to_normalized(u, v):
    """Convert pixel coordinates to normalized image coordinates.

    Normalized coords have origin at principal point, scaled by focal length.
    Used as input to the IBVS interaction matrix.

    Returns:
        (nx, ny): normalized coordinates where (0, 0) is the image center.
    """
    return (u - CX) / FX, (v - CY) / FY


def normalized_to_pixel(nx, ny):
    """Convert normalized image coordinates back to pixel coordinates."""
    return nx * FX + CX, ny * FY + CY


def image_coords_to_pixel(x, y):
    """Convert normalized image-space coords (0-1 range) to pixel coords.

    In image-space: (0, 0) = top-left, (1, 1) = bottom-right.
    """
    return x * IMG_WIDTH, y * IMG_HEIGHT


def pixel_to_image_coords(u, v):
    """Convert pixel coordinates to normalized image-space coords (0-1 range)."""
    return u / IMG_WIDTH, v / IMG_HEIGHT
