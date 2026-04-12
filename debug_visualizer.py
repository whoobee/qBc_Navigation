"""
Debug visualization — draws waypoints, path, and tracking state on camera frames.

Generates annotated images saved to the debug/ directory for display
in the ConfigurationManager Navigation applet.
"""

import logging
import os
import time
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("qBc_Nav.debug_viz")

DEBUG_DIR = Path(__file__).parent / "debug"
TEMP_DIR = Path(__file__).parent / "temp"
LATEST_FRAME = str(TEMP_DIR / "latest_frame.jpg")

# Colors (BGR)
COLOR_WAYPOINT = (0, 200, 255)       # Orange — pending waypoint
COLOR_CURRENT = (0, 255, 0)          # Green — current target
COLOR_REACHED = (200, 200, 200)      # Gray — reached waypoint
COLOR_PATH_LINE = (255, 180, 0)      # Cyan-ish — path line
COLOR_TRACKING = (0, 0, 255)         # Red — current tracked position
COLOR_TEXT = (255, 255, 255)         # White — labels
COLOR_DEADZONE = (0, 180, 255)      # Orange — neck deadzone band
COLOR_HORIZON = (0, 255, 255)       # Yellow — detected floor boundary

WAYPOINT_RADIUS = 16
CURRENT_RADIUS = 24
TRACKING_RADIUS = 12
PATH_THICKNESS = 3
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 2


class DebugVisualizer:
    """Generates annotated debug images for the navigation system."""

    def __init__(self):
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        self._base_frame = None
        self._waypoints = []
        self._current_wp_index = 0
        self._reached_indices = set()
        self._tracked_pos = None
        self._last_image_path = None
        self._neck_deadzone = 0.03  # normalized camera coords
        self._floor_boundary = None  # normalized y (0-1), set externally

    def set_waypoints(self, frame_path, waypoints):
        """Load the base frame and store waypoint list.

        Args:
            frame_path: Path to the camera frame image.
            waypoints: List of {"x": float, "y": float} in image-space (0-1).
        """
        if frame_path and os.path.isfile(frame_path):
            self._base_frame = cv2.imread(frame_path)
        else:
            # Create a blank frame if original not available
            self._base_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
            logger.warning("Base frame not found: %s, using blank", frame_path)

        self._waypoints = waypoints
        self._current_wp_index = 0
        self._reached_indices = set()
        self._tracked_pos = None
        return self._render_and_save()

    def update_progress(self, current_index, tracked_x=None, tracked_y=None):
        """Update navigation progress and re-render.

        Args:
            current_index: Index of the current target waypoint.
            tracked_x: Current tracked position x (image-space 0-1), or None.
            tracked_y: Current tracked position y (image-space 0-1), or None.
        """
        # Mark all waypoints before current as reached
        for i in range(current_index):
            self._reached_indices.add(i)
        self._current_wp_index = current_index

        if tracked_x is not None and tracked_y is not None:
            self._tracked_pos = (tracked_x, tracked_y)
        else:
            self._tracked_pos = None

        return self._render_and_save()

    def mark_complete(self):
        """Mark all waypoints as reached."""
        for i in range(len(self._waypoints)):
            self._reached_indices.add(i)
        self._tracked_pos = None
        return self._render_and_save()

    def update_live_frame(self, frame, current_index, tracked_x=None, tracked_y=None):
        """Annotate a live camera frame and write to latest_frame.jpg.

        On live frames, only the tracked position is meaningful — it shows
        where the current waypoint's visual features are in the new camera view.
        Reached waypoints are behind the robot; future ones aren't tracked yet.

        Args:
            frame: BGR numpy array (the raw camera frame).
            current_index: Current target waypoint index.
            tracked_x: Tracked position x (image-space 0-1), or None.
            tracked_y: Tracked position y (image-space 0-1), or None.
        """
        if frame is None:
            return
        self._current_wp_index = current_index
        for i in range(current_index):
            self._reached_indices.add(i)
        if tracked_x is not None and tracked_y is not None:
            self._tracked_pos = (tracked_x, tracked_y)
        img = self._annotate_live(frame)
        cv2.imwrite(LATEST_FRAME, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

    @staticmethod
    def cleanup_temp():
        """Remove all files from the temp directory."""
        if TEMP_DIR.exists():
            for f in TEMP_DIR.iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass

    @staticmethod
    def write_passthrough_frame(frame):
        """Write a raw camera frame to latest_frame.jpg (no annotation).

        Used when navigation is idle to provide a live camera feed.
        """
        if frame is None:
            return
        cv2.imwrite(LATEST_FRAME, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

    def get_last_image_path(self):
        """Return the path of the last generated debug image."""
        return self._last_image_path

    def _annotate(self, frame):
        """Draw waypoints, path, and tracking state onto a frame. Returns annotated copy."""
        img = frame.copy()
        h, w = img.shape[:2]

        if not self._waypoints:
            return img

        wp_pixels = [(int(wp["x"] * w), int(wp["y"] * h)) for wp in self._waypoints]

        # Path lines
        for i in range(len(wp_pixels) - 1):
            cv2.line(img, wp_pixels[i], wp_pixels[i + 1],
                     COLOR_PATH_LINE, PATH_THICKNESS, cv2.LINE_AA)

        # Waypoints
        for i, (px, py) in enumerate(wp_pixels):
            if i in self._reached_indices:
                cv2.circle(img, (px, py), WAYPOINT_RADIUS, COLOR_REACHED, -1, cv2.LINE_AA)
                cv2.line(img, (px - 6, py), (px - 1, py + 5), COLOR_TEXT, 2, cv2.LINE_AA)
                cv2.line(img, (px - 1, py + 5), (px + 8, py - 6), COLOR_TEXT, 2, cv2.LINE_AA)
            elif i == self._current_wp_index:
                cv2.circle(img, (px, py), CURRENT_RADIUS, COLOR_CURRENT, 3, cv2.LINE_AA)
                cv2.circle(img, (px, py), 4, COLOR_CURRENT, -1, cv2.LINE_AA)
                cv2.line(img, (px - CURRENT_RADIUS - 4, py), (px + CURRENT_RADIUS + 4, py),
                         COLOR_CURRENT, 1, cv2.LINE_AA)
                cv2.line(img, (px, py - CURRENT_RADIUS - 4), (px, py + CURRENT_RADIUS + 4),
                         COLOR_CURRENT, 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (px, py), WAYPOINT_RADIUS, COLOR_WAYPOINT, -1, cv2.LINE_AA)
            label = f"WP{i + 1}"
            cv2.putText(img, label, (px + WAYPOINT_RADIUS + 4, py + 5),
                        LABEL_FONT, LABEL_SCALE, COLOR_TEXT, LABEL_THICKNESS, cv2.LINE_AA)

        # Tracked position
        if self._tracked_pos is not None:
            tx = int(self._tracked_pos[0] * w)
            ty = int(self._tracked_pos[1] * h)
            cv2.circle(img, (tx, ty), TRACKING_RADIUS, COLOR_TRACKING, 2, cv2.LINE_AA)
            cv2.drawMarker(img, (tx, ty), COLOR_TRACKING,
                           cv2.MARKER_CROSS, TRACKING_RADIUS, 2, cv2.LINE_AA)

        # Status overlay
        status = f"Waypoint {self._current_wp_index + 1}/{len(self._waypoints)}"
        cv2.putText(img, status, (20, 40), LABEL_FONT, 1.0, COLOR_TEXT, 2, cv2.LINE_AA)

        return img

    def _annotate_live(self, frame):
        """Draw live tracking overlay on a camera frame.

        Only draws the tracked position (where the current waypoint's features
        are NOW in the camera view) and a crosshair at image center (the target).
        """
        img = frame.copy()
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2

        # Draw image-center crosshair (the IBVS target — where we're steering toward)
        cross_size = 30
        cv2.line(img, (cx - cross_size, cy), (cx + cross_size, cy),
                 COLOR_CURRENT, 1, cv2.LINE_AA)
        cv2.line(img, (cx, cy - cross_size), (cx, cy + cross_size),
                 COLOR_CURRENT, 1, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 4, COLOR_CURRENT, -1, cv2.LINE_AA)

        # Draw waypoint path and markers (static overlay — no tracking cost)
        if self._waypoints:
            wp_px = [(int(wp["x"] * w), int(wp["y"] * h)) for wp in self._waypoints]

            # Path lines
            for i in range(len(wp_px) - 1):
                cv2.line(img, wp_px[i], wp_px[i + 1],
                         COLOR_PATH_LINE, 2, cv2.LINE_AA)

            # Waypoint markers
            for i, (px, py) in enumerate(wp_px):
                if i in self._reached_indices:
                    # Reached — gray filled with checkmark
                    cv2.circle(img, (px, py), WAYPOINT_RADIUS, COLOR_REACHED, -1, cv2.LINE_AA)
                    cv2.line(img, (px - 6, py), (px - 1, py + 5), COLOR_TEXT, 2, cv2.LINE_AA)
                    cv2.line(img, (px - 1, py + 5), (px + 8, py - 6), COLOR_TEXT, 2, cv2.LINE_AA)
                elif i == self._current_wp_index:
                    # Current target — green ring with crosshairs
                    cv2.circle(img, (px, py), CURRENT_RADIUS, COLOR_CURRENT, 2, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 4, COLOR_CURRENT, -1, cv2.LINE_AA)
                else:
                    # Pending — orange filled
                    cv2.circle(img, (px, py), WAYPOINT_RADIUS, COLOR_WAYPOINT, -1, cv2.LINE_AA)

                cv2.putText(img, f"WP{i + 1}", (px + WAYPOINT_RADIUS + 4, py + 5),
                            LABEL_FONT, LABEL_SCALE, COLOR_TEXT, 1, cv2.LINE_AA)

        # Draw detected floor boundary (horizon line)
        if self._floor_boundary is not None and 0.0 < self._floor_boundary < 1.0:
            hy = int(self._floor_boundary * h)
            # Dashed horizontal line
            dash_len = 20
            gap_len = 12
            x = 0
            while x < w:
                x_end = min(x + dash_len, w)
                cv2.line(img, (x, hy), (x_end, hy), COLOR_HORIZON, 1, cv2.LINE_AA)
                x += dash_len + gap_len
            # Label
            cv2.putText(img, "FLOOR", (8, hy - 6),
                        LABEL_FONT, 0.4, COLOR_HORIZON, 1, cv2.LINE_AA)

        # Draw neck deadzone band — full-height vertical band (neck only pans horizontally)
        if self._neck_deadzone > 0:
            # Deadzone is in normalized camera coords: nx = (u - CX) / FX
            # Convert back to pixel offset: dz_px = deadzone * FX
            from camera_model import FX
            dz_px = int(self._neck_deadzone * FX)

            # Full-height vertical edge lines
            cv2.line(img, (cx - dz_px, 0), (cx - dz_px, h),
                     COLOR_DEADZONE, 1, cv2.LINE_AA)
            cv2.line(img, (cx + dz_px, 0), (cx + dz_px, h),
                     COLOR_DEADZONE, 1, cv2.LINE_AA)

            # Subtle filled band overlay spanning full frame height
            overlay = img.copy()
            cv2.rectangle(overlay, (cx - dz_px, 0), (cx + dz_px, h),
                          COLOR_DEADZONE, -1)
            cv2.addWeighted(overlay, 0.06, img, 0.94, 0, img)

            # Label at top
            cv2.putText(img, "DEADZONE", (cx - dz_px + 4, 20),
                        LABEL_FONT, 0.4, COLOR_DEADZONE, 1, cv2.LINE_AA)

        # Draw tracked waypoint position (where the features actually are)
        if self._tracked_pos is not None:
            tx = int(self._tracked_pos[0] * w)
            ty = int(self._tracked_pos[1] * h)

            # Ring + crosshair at tracked position
            cv2.circle(img, (tx, ty), CURRENT_RADIUS, COLOR_TRACKING, 3, cv2.LINE_AA)
            cv2.drawMarker(img, (tx, ty), COLOR_TRACKING,
                           cv2.MARKER_CROSS, CURRENT_RADIUS, 2, cv2.LINE_AA)

            # Line from tracked to center (the error vector the IBVS is correcting)
            cv2.line(img, (tx, ty), (cx, cy), COLOR_PATH_LINE, 2, cv2.LINE_AA)

            # Label
            wp_label = f"WP{self._current_wp_index + 1}"
            cv2.putText(img, wp_label, (tx + CURRENT_RADIUS + 6, ty + 6),
                        LABEL_FONT, LABEL_SCALE, COLOR_TEXT, LABEL_THICKNESS, cv2.LINE_AA)

            # Error distance
            error = ((tx - cx) ** 2 + (ty - cy) ** 2) ** 0.5
            error_norm = error / ((w ** 2 + h ** 2) ** 0.5)
            err_text = f"err: {error_norm:.3f}"
            cv2.putText(img, err_text, (tx + CURRENT_RADIUS + 6, ty + 24),
                        LABEL_FONT, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

        # Status overlay
        total = len(self._waypoints)
        reached = len(self._reached_indices)
        status = f"WP {self._current_wp_index + 1}/{total}  |  reached: {reached}"
        cv2.putText(img, status, (20, 40), LABEL_FONT, 1.0, COLOR_TEXT, 2, cv2.LINE_AA)

        return img

    def _render_and_save(self):
        """Render the annotated image and save to debug directory."""
        if self._base_frame is None:
            return None

        img = self._annotate(self._base_frame)

        # Save timestamped snapshot
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"nav_debug_{timestamp}.jpg"
        filepath = str(DEBUG_DIR / filename)
        cv2.imwrite(filepath, img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        self._last_image_path = filepath

        # Also update the latest frame for streaming
        cv2.imwrite(LATEST_FRAME, img, [cv2.IMWRITE_JPEG_QUALITY, 85])

        logger.debug("Debug image saved: %s", filepath)
        return filepath
