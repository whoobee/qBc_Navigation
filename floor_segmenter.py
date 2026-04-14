"""
Floor segmentation for reactive navigation.

Uses HSV histogram backprojection to identify drivable floor area in each
camera frame.  The bottom strip of the image is assumed to be floor and used
as the color reference.  The result is a binary floor mask and per-corridor
openness scores for steering decisions.

Algorithm:
    1. Downsample frame for speed
    2. Convert to HSV
    3. Build hue-saturation histogram from bottom reference strip
    4. Back-project histogram onto full frame → floor probability map
    5. Threshold + morphological cleanup
    6. Keep only the connected component touching the bottom (actual floor)
    7. Score vertical corridors by openness and depth

Dependencies: opencv-contrib-python, numpy (already in the project)
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger("qBc_Nav.floor_seg")

# Processing scale (0.5 = half resolution for speed)
SEGMENT_SCALE = 0.5

# Bottom fraction used as floor color reference
REF_FRACTION = 0.15

# HSV histogram parameters
H_BINS = 30
S_BINS = 32

# Backprojection threshold (0-255)
BP_THRESHOLD = 50

# Morphological cleanup kernel size
MORPH_KERNEL = 11

# Corridor scoring region (normalized y range — avoid sky/ceiling)
CORRIDOR_TOP_Y = 0.30
CORRIDOR_BOTTOM_Y = 1.0

# Default number of corridors
NUM_CORRIDORS = 7

# Minimum row-floor fraction to count as "floor present" in depth scan
DEPTH_ROW_THRESHOLD = 0.3


class FloorSegmenter:
    """Segments drivable floor from a monocular camera frame."""

    def __init__(self, num_corridors=NUM_CORRIDORS):
        self._num_corridors = num_corridors

    def segment(self, frame):
        """Produce a binary floor mask from a BGR camera frame.

        Args:
            frame: BGR numpy array (any resolution).

        Returns:
            Boolean numpy array (same H, W as input) — True where floor.
        """
        h, w = frame.shape[:2]

        # Downsample for speed
        sw = int(w * SEGMENT_SCALE)
        sh = int(h * SEGMENT_SCALE)
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Floor reference: bottom strip
        ref_top = int(sh * (1.0 - REF_FRACTION))
        ref_region = hsv[ref_top:sh, :, :]

        # Hue-saturation histogram of the floor reference
        hist = cv2.calcHist(
            [ref_region], [0, 1], None,
            [H_BINS, S_BINS],
            [0, 180, 0, 256],
        )
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)

        # Back-project onto the full (downsampled) frame
        bp = cv2.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

        # Disc filter to smooth the backprojection (OpenCV recommendation)
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bp = cv2.filter2D(bp, -1, disc)

        # Threshold
        _, mask = cv2.threshold(bp, BP_THRESHOLD, 255, cv2.THRESH_BINARY)

        # Morphological cleanup — close gaps then remove noise
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MORPH_KERNEL, MORPH_KERNEL),
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Keep only the floor component (connected to bottom of frame)
        mask = self._keep_floor_component(mask)

        # Upsample back to original resolution
        mask_full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        return mask_full > 0

    def compute_corridors(self, floor_mask):
        """Score vertical corridors by floor openness and forward depth.

        Divides the scoring region of the floor mask into vertical strips
        and computes how open each corridor is.

        Args:
            floor_mask: boolean array (H, W).

        Returns:
            List of dicts, one per corridor (left to right):
                center_x: corridor center in normalized image coords (0-1)
                openness:  fraction of corridor area that is floor (0-1)
                depth:     how far forward the floor extends (0-1, 1=maximum)
        """
        h, w = floor_mask.shape
        n = self._num_corridors
        cw = w // n  # corridor pixel width

        top_row = int(h * CORRIDOR_TOP_Y)
        bot_row = int(h * CORRIDOR_BOTTOM_Y)
        roi = floor_mask[top_row:bot_row, :]
        roi_h = roi.shape[0]

        corridors = []
        for i in range(n):
            x1 = i * cw
            x2 = (x1 + cw) if i < n - 1 else w
            strip = roi[:, x1:x2]
            strip_w = strip.shape[1]

            # Openness: fraction of pixels that are floor
            openness = float(strip.sum()) / strip.size if strip.size > 0 else 0.0

            # Depth: how far up does floor extend in this corridor?
            # Scan from top of ROI downward — find the first row where
            # floor is present.  Higher up = more clear space ahead.
            depth = 0.0
            for row in range(roi_h):
                row_frac = strip[row, :].sum() / strip_w if strip_w > 0 else 0.0
                if row_frac >= DEPTH_ROW_THRESHOLD:
                    depth = 1.0 - (row / roi_h)
                    break

            center_x = (x1 + x2) / 2.0 / w

            corridors.append({
                "center_x": float(center_x),
                "openness": float(openness),
                "depth": float(depth),
            })

        return corridors

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _keep_floor_component(mask):
        """Keep only connected components that touch the bottom of the frame.

        Eliminates same-colored objects (e.g. furniture) that aren't
        part of the contiguous floor surface.
        """
        h, w = mask.shape
        num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
        if num_labels <= 1:
            return mask

        # Labels present in the bottom 10% of the frame
        bottom_strip = labels[int(h * 0.90):h, :]
        bottom_labels = set(np.unique(bottom_strip)) - {0}

        if not bottom_labels:
            # No floor detected at bottom — return empty mask
            return np.zeros_like(mask)

        result = np.zeros_like(mask)
        for lbl in bottom_labels:
            result[labels == lbl] = 255
        return result
