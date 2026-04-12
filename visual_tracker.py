"""
Visual feature tracker for waypoint navigation.

Primary: ORB descriptor matching (fast, rotation-invariant).
Fallback: Template matching (works on textureless surfaces).

The tracker automatically falls back to template matching when ORB
cannot find enough features in the waypoint region.
"""

import logging

import cv2
import numpy as np

from camera_model import IMG_WIDTH, IMG_HEIGHT

logger = logging.getLogger("qBc_Nav.tracker")

# ORB parameters
ORB_FEATURES = 200          # Max keypoints to detect per ROI
KEEP_TOP_N = 30             # Store top-N strongest descriptors
ROI_SIZE = 256              # Pixel size of region around waypoint
SEARCH_SCALE = 2.0          # Search window = ROI_SIZE * SEARCH_SCALE
HAMMING_THRESHOLD = 60       # Max Hamming distance for valid match
MIN_MATCHES = 3             # Below this, tracking is considered lost
MIN_INIT_FEATURES = 5       # Minimum ORB features to use ORB mode

# Template matching parameters
TEMPLATE_SIZE = 128          # Size of the template patch
TEMPLATE_SEARCH_SCALE = 3.0  # Search window multiplier for template matching
TEMPLATE_THRESHOLD = 0.4     # Minimum correlation to consider valid (0-1)

# Tracking mode
MODE_ORB = "orb"
MODE_TEMPLATE = "template"


class VisualTracker:
    """Hybrid ORB + template tracker for a single waypoint target."""

    def __init__(self):
        self._orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Target state
        self._mode = None
        self._target_descriptors = None   # ORB mode
        self._target_template = None      # Template mode
        self._last_pixel_x = 0.0
        self._last_pixel_y = 0.0
        self._feature_count = 0
        self._initialized = False

    def initialize_target(self, frame, waypoint_x, waypoint_y):
        """Extract tracking features around a waypoint in the given frame.

        Tries ORB first. If the region is too textureless, falls back to
        template matching automatically.

        Args:
            frame: BGR image (numpy array).
            waypoint_x: Waypoint x in normalized image coords (0-1).
            waypoint_y: Waypoint y in normalized image coords (0-1).

        Returns:
            True if tracking was initialized successfully.
        """
        self._initialized = False
        self._mode = None
        h, w = frame.shape[:2]

        cx = int(waypoint_x * w)
        cy = int(waypoint_y * h)
        self._last_pixel_x = float(cx)
        self._last_pixel_y = float(cy)

        # --- Try ORB first ---
        if self._try_orb_init(frame, cx, cy):
            return True

        # --- Fallback: template matching ---
        if self._try_template_init(frame, cx, cy):
            return True

        logger.warning(
            "Both ORB and template init failed at (%.2f, %.2f)",
            waypoint_x, waypoint_y,
        )
        return False

    def update(self, frame):
        """Track the target in a new frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            (x, y, valid): Updated position in normalized image coords (0-1)
            and whether tracking is still valid.
        """
        if not self._initialized:
            return 0.0, 0.0, False

        if self._mode == MODE_ORB:
            return self._update_orb(frame)
        elif self._mode == MODE_TEMPLATE:
            return self._update_template(frame)

        return 0.0, 0.0, False

    def get_feature_count(self):
        """Return the number of matched features from the last update."""
        return self._feature_count

    def get_last_pixel_position(self):
        """Return last known pixel position (for debug visualization)."""
        return self._last_pixel_x, self._last_pixel_y

    def get_mode(self):
        """Return the current tracking mode ('orb', 'template', or None)."""
        return self._mode

    # ------------------------------------------------------------------
    # ORB initialization and tracking
    # ------------------------------------------------------------------

    def _try_orb_init(self, frame, cx, cy):
        """Try to initialize ORB tracking. Returns True on success."""
        roi, offset_x, offset_y = self._extract_roi(frame, cx, cy, ROI_SIZE)
        if roi is None:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        n_found = len(keypoints) if keypoints else 0
        if descriptors is None or n_found < MIN_INIT_FEATURES:
            logger.info(
                "ORB: only %d features (need %d) — will try template",
                n_found, MIN_INIT_FEATURES,
            )
            return False

        # Keep top-N by response strength
        if len(keypoints) > KEEP_TOP_N:
            indices = np.argsort([-kp.response for kp in keypoints])[:KEEP_TOP_N]
            keypoints = [keypoints[i] for i in indices]
            descriptors = descriptors[indices]

        self._target_descriptors = descriptors
        self._feature_count = len(keypoints)
        self._mode = MODE_ORB
        self._initialized = True

        logger.info(
            "ORB tracker initialized: %d features at pixel (%d, %d)",
            self._feature_count, cx, cy,
        )
        return True

    def _update_orb(self, frame):
        """Track using ORB descriptor matching."""
        if self._target_descriptors is None:
            return 0.0, 0.0, False

        h, w = frame.shape[:2]
        cx = int(self._last_pixel_x)
        cy = int(self._last_pixel_y)

        search_size = int(ROI_SIZE * SEARCH_SCALE)
        roi, offset_x, offset_y = self._extract_roi(frame, cx, cy, search_size)
        if roi is None:
            return self._last_pixel_x / w, self._last_pixel_y / h, False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self._orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) == 0:
            self._feature_count = 0
            return self._last_pixel_x / w, self._last_pixel_y / h, False

        matches = self._bf.match(self._target_descriptors, descriptors)
        good_matches = [m for m in matches if m.distance < HAMMING_THRESHOLD]

        if len(good_matches) < MIN_MATCHES:
            self._feature_count = len(good_matches)
            logger.debug("ORB tracking weak: %d matches", len(good_matches))
            return self._last_pixel_x / w, self._last_pixel_y / h, False

        # Median position of matched keypoints
        matched_pts = np.array([
            keypoints[m.trainIdx].pt for m in good_matches
        ])
        median_x = np.median(matched_pts[:, 0])
        median_y = np.median(matched_pts[:, 1])

        new_px = max(0, min(w - 1, offset_x + median_x))
        new_py = max(0, min(h - 1, offset_y + median_y))

        self._last_pixel_x = float(new_px)
        self._last_pixel_y = float(new_py)
        self._feature_count = len(good_matches)

        return new_px / w, new_py / h, True

    # ------------------------------------------------------------------
    # Template matching initialization and tracking
    # ------------------------------------------------------------------

    def _try_template_init(self, frame, cx, cy):
        """Initialize template matching as fallback. Returns True on success."""
        roi, offset_x, offset_y = self._extract_roi(frame, cx, cy, TEMPLATE_SIZE)
        if roi is None:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Check that the template has some variance (not completely flat)
        std = gray.std()
        if std < 5.0:
            logger.info(
                "Template region too flat (std=%.1f) at pixel (%d, %d)",
                std, cx, cy,
            )
            return False

        self._target_template = gray.copy()
        self._mode = MODE_TEMPLATE
        self._feature_count = 1  # Template counts as 1 "feature"
        self._initialized = True

        logger.info(
            "Template tracker initialized: %dx%d patch at pixel (%d, %d), std=%.1f",
            gray.shape[1], gray.shape[0], cx, cy, std,
        )
        return True

    def _update_template(self, frame):
        """Track using normalized cross-correlation template matching."""
        if self._target_template is None:
            return 0.0, 0.0, False

        h, w = frame.shape[:2]
        cx = int(self._last_pixel_x)
        cy = int(self._last_pixel_y)

        # Search in a larger window
        search_size = int(TEMPLATE_SIZE * TEMPLATE_SEARCH_SCALE)
        roi, offset_x, offset_y = self._extract_roi(frame, cx, cy, search_size)
        if roi is None:
            return self._last_pixel_x / w, self._last_pixel_y / h, False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        th, tw = self._target_template.shape[:2]

        # Ensure search region is larger than template
        if gray.shape[0] < th or gray.shape[1] < tw:
            return self._last_pixel_x / w, self._last_pixel_y / h, False

        result = cv2.matchTemplate(gray, self._target_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < TEMPLATE_THRESHOLD:
            self._feature_count = 0
            logger.debug("Template match weak: correlation=%.3f", max_val)
            return self._last_pixel_x / w, self._last_pixel_y / h, False

        # max_loc is the top-left of the best match; compute center
        match_cx = max_loc[0] + tw / 2.0
        match_cy = max_loc[1] + th / 2.0

        new_px = max(0, min(w - 1, offset_x + match_cx))
        new_py = max(0, min(h - 1, offset_y + match_cy))

        self._last_pixel_x = float(new_px)
        self._last_pixel_y = float(new_py)
        self._feature_count = 1  # Template is a single "feature"

        # Update template to handle gradual appearance changes
        # Blend 80% old + 20% new to prevent drift
        new_roi, _, _ = self._extract_roi(frame, int(new_px), int(new_py), TEMPLATE_SIZE)
        if new_roi is not None:
            new_gray = cv2.cvtColor(new_roi, cv2.COLOR_BGR2GRAY)
            if new_gray.shape == self._target_template.shape:
                self._target_template = cv2.addWeighted(
                    self._target_template, 0.8, new_gray, 0.2, 0,
                )

        return new_px / w, new_py / h, True

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_roi(frame, cx, cy, size):
        """Extract a square ROI centered on (cx, cy).

        Returns:
            (roi, offset_x, offset_y) where offset is the top-left corner
            of the ROI in full-image coordinates. None if out of bounds.
        """
        h, w = frame.shape[:2]
        half = size // 2

        x1 = max(0, cx - half)
        y1 = max(0, cy - half)
        x2 = min(w, cx + half)
        y2 = min(h, cy + half)

        if x2 - x1 < 32 or y2 - y1 < 32:
            return None, 0, 0

        return frame[y1:y2, x1:x2], x1, y1
