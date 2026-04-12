"""
Navigation controller — state machine, IBVS visual servoing, neck stabilization.

Consumes waypoints from the exploration handler, tracks them visually with ORB,
and computes differential-drive motor commands using Image-Based Visual Servoing.

The neck servo provides fast horizontal stabilization to keep the tracked
feature centered while the wheel base turns more slowly.

State machine:
    IDLE → ACQUIRING → SERVOING → NEXT → (loop or IDLE)
    RECOVERY on tracking loss, IDLE on safety trigger.
"""

import json
import logging
import math
import os
import threading
import time

import cv2
import numpy as np

from visual_tracker import VisualTracker
from debug_visualizer import DebugVisualizer
from camera_model import (
    pixel_to_normalized, image_coords_to_pixel,
    IMG_WIDTH, IMG_HEIGHT,
)

logger = logging.getLogger("qBc_Nav.controller")

# MQTT topics
TOPIC_NAV_WAYPOINTS = "robot/navigation/waypoints"
TOPIC_NAV_CMD = "robot/navigation/cmd"
TOPIC_NAV_STATE = "robot/navigation/state"
TOPIC_VISION_CMD = "robot/vision/cmd"
TOPIC_FRAME_READY = "robot/vision/frame_ready"
TOPIC_SAFETY_STATUS = "robot/safety/status"
TOPIC_WHEELS_CMD = "robot/wheels/cmd"
TOPIC_JOINTS_CMD = "robot/joints/cmd"
TOPIC_ODOMETRY = "robot/odometry/pose"
TOPIC_STREAM_REQ = "robot/navigation/stream/request"
TOPIC_STREAM_VIEWER = "robot/navigation/stream/viewer"
VIEWER_TIMEOUT = 3.0  # Seconds without heartbeat before stopping captures

# Debug topics
TOPIC_DEBUG_PATH_IMAGE = "robot/navigation/debug/path_image"
TOPIC_DEBUG_MOTOR = "robot/navigation/debug/motor"
TOPIC_DEBUG_LOG = "robot/navigation/debug/log"

# Navigation states
STATE_IDLE = "idle"
STATE_ACQUIRING = "acquiring"
STATE_SERVOING = "servoing"
STATE_NEXT = "next"
STATE_RECOVERY = "recovery"

# IBVS parameters
LAMBDA_GAIN = 0.5           # IBVS control gain (lower = slower/safer)
MAX_RPM = 30.0              # Maximum wheel RPM
LINEAR_SCALE = 80.0         # Normalized velocity → RPM (forward)
ANGULAR_SCALE = 40.0        # Normalized angular velocity → RPM (yaw)
Z_MIN = 0.5                 # Minimum estimated depth (meters)
Z_MAX = 4.0                 # Maximum estimated depth (meters)

# Waypoint reached condition
REACHED_THRESHOLD = 0.08    # Normalized distance from image center
REACHED_FRAMES = 3          # Consecutive frames to confirm reached

# Neck stabilization
NECK_GAIN = 120.0           # Horizontal error → neck degrees (nx=0.5 → 60 deg)
NECK_RECENTER_GAIN = 1.0    # Wheel yaw driven by neck angle (deg → angular rate)
NECK_SPEED = 120.0          # Neck servo speed in deg/s (fast response)
NECK_MAX = 80.0             # Limit neck range (don't use full ±90)

# Frame capture
FRAME_TIMEOUT = 0.5         # Seconds to wait for frame
SERVO_LOOP_HZ = 10          # Target servoing frequency

# Recovery
MAX_RECOVERY_ATTEMPTS = 3


class NavigationController:
    """Visual servoing navigation controller with neck stabilization."""

    subscriptions = [
        (TOPIC_NAV_WAYPOINTS, 1),
        (TOPIC_NAV_CMD, 1),
        (TOPIC_FRAME_READY, 1),
        (TOPIC_SAFETY_STATUS, 1),
        (TOPIC_ODOMETRY, 0),
        (TOPIC_STREAM_REQ, 0),
        (TOPIC_STREAM_VIEWER, 0),
    ]

    def __init__(self, mqtt_client):
        self._client = mqtt_client
        self._tracker = VisualTracker()
        self._debug_viz = DebugVisualizer()
        self._lock = threading.Lock()

        # State
        self._state = STATE_IDLE
        self._waypoints = []
        self._frame_path_for_init = None
        self._current_wp_index = 0
        self._reached_counter = 0
        self._recovery_attempts = 0
        self._neck_deg = 0.0

        # Safety
        self._safety_ok = True

        # Frame synchronization
        self._waiting_for_frame = False
        self._frame_path = None
        self._frame_event = threading.Event()

        # Navigation thread
        self._nav_thread = None
        self._stop_event = threading.Event()

        # Viewer heartbeat — only capture idle frames when a viewer is active
        self._viewer_last_seen = 0.0

    def register_callbacks(self, client):
        """Register per-topic MQTT callbacks."""
        client.message_callback_add(TOPIC_NAV_WAYPOINTS, self._on_waypoints)
        client.message_callback_add(TOPIC_NAV_CMD, self._on_command)
        client.message_callback_add(TOPIC_FRAME_READY, self._on_frame_ready)
        client.message_callback_add(TOPIC_SAFETY_STATUS, self._on_safety)
        client.message_callback_add(TOPIC_STREAM_REQ, self._on_stream_request)
        client.message_callback_add(TOPIC_STREAM_VIEWER, self._on_viewer_heartbeat)

    def subscribe(self, client):
        """Subscribe to all required topics."""
        for topic, qos in self.subscriptions:
            client.subscribe(topic, qos=qos)

    def publish_state(self):
        """Publish current navigation state."""
        self._client.publish(
            TOPIC_NAV_STATE,
            json.dumps({
                "status": "online",
                "nav_state": self._state,
                "current_waypoint": self._current_wp_index,
                "total_waypoints": len(self._waypoints),
                "tracking_features": self._tracker.get_feature_count(),
                "tracking_mode": self._tracker.get_mode(),
                "neck_deg": self._neck_deg,
            }),
            qos=1, retain=True,
        )

    def shutdown(self):
        """Clean shutdown — stop navigation and zero motors."""
        self._stop_event.set()
        if self._nav_thread and self._nav_thread.is_alive():
            self._nav_thread.join(timeout=3.0)
        self._stop_wheels()
        self._center_neck()
        self._client.publish(
            TOPIC_NAV_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )

    # ------------------------------------------------------------------
    # On-demand stream frame (only captures when a viewer requests it)
    # ------------------------------------------------------------------

    def _on_viewer_heartbeat(self, client, userdata, msg):
        """Track when a viewer is active on the Navigation page."""
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}

        if data.get("active") is False:
            # Viewer explicitly left — clean up temp immediately
            self._viewer_last_seen = 0.0
            DebugVisualizer.cleanup_temp()
            logger.info("Viewer left — cleaned navigation temp")
            return

        self._viewer_last_seen = time.monotonic()

    def _has_active_viewer(self):
        """Check if a viewer heartbeat was received recently."""
        return (time.monotonic() - self._viewer_last_seen) < VIEWER_TIMEOUT

    def _on_stream_request(self, client, userdata, msg):
        """Handle a stream frame request — capture one frame if idle and viewer active."""
        if self._state != STATE_IDLE:
            return
        if not self._has_active_viewer():
            return
        threading.Thread(target=self._capture_stream_frame, daemon=True).start()

    def _capture_stream_frame(self):
        """Capture a single frame for the live stream (runs in a thread)."""
        frame = self._capture_frame()
        if frame is not None:
            DebugVisualizer.write_passthrough_frame(frame)

    # ------------------------------------------------------------------
    # MQTT callbacks
    # ------------------------------------------------------------------

    def _on_waypoints(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        waypoints = data.get("waypoints", [])
        frame_path = data.get("frame")

        if not waypoints:
            logger.warning("Received empty waypoints")
            return

        logger.info("Received %d waypoints, frame: %s", len(waypoints), frame_path)

        with self._lock:
            self._waypoints = waypoints
            self._frame_path_for_init = frame_path
            self._current_wp_index = 0
            self._reached_counter = 0
            self._recovery_attempts = 0

        # Generate initial debug image
        debug_path = self._debug_viz.set_waypoints(frame_path, waypoints)
        if debug_path:
            self._publish_debug_path_image(debug_path)

        self._publish_debug_log("waypoints_received",
                                f"Received {len(waypoints)} waypoints")

        # Start navigation thread
        self._stop_event.set()
        if self._nav_thread and self._nav_thread.is_alive():
            self._nav_thread.join(timeout=2.0)

        self._stop_event.clear()
        self._nav_thread = threading.Thread(target=self._navigation_loop, daemon=True)
        self._nav_thread.start()

    def _on_command(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        command = data.get("command")
        if command == "cancel":
            logger.info("Navigation cancelled")
            self._stop_event.set()
            self._stop_wheels()
            self._center_neck()
            self._set_state(STATE_IDLE)
            self._publish_debug_log("cancel", "Navigation cancelled by user")
        elif command == "pause":
            logger.info("Navigation paused")
            self._stop_wheels()
            self._publish_debug_log("pause", "Navigation paused")
        elif command == "resume":
            logger.info("Navigation resumed")
            self._publish_debug_log("resume", "Navigation resumed")

    def _on_frame_ready(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        file_path = data.get("file")
        if not file_path:
            return

        if self._waiting_for_frame:
            self._frame_path = file_path
            self._frame_event.set()

    def _on_safety(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return

        # Check any safety flag
        unsafe = any([
            data.get("obstacle", False),
            data.get("tilt", False),
            data.get("pickup", False),
            data.get("low_battery", False),
            data.get("watchdog", False),
            data.get("overtemp", False),
        ])

        if unsafe and self._safety_ok:
            logger.warning("Safety triggered — stopping navigation")
            self._safety_ok = False
            self._stop_wheels()
            self._stop_event.set()
            self._set_state(STATE_IDLE)
            self._publish_debug_log("safety_stop",
                                    f"Safety triggered: {json.dumps(data)}")
        elif not unsafe:
            self._safety_ok = True

    # ------------------------------------------------------------------
    # Navigation loop (runs in a thread)
    # ------------------------------------------------------------------

    def _navigation_loop(self):
        """Main navigation state machine loop."""
        self._set_state(STATE_ACQUIRING)

        while not self._stop_event.is_set():
            if not self._safety_ok:
                self._stop_wheels()
                self._set_state(STATE_IDLE)
                break

            state = self._state

            if state == STATE_ACQUIRING:
                self._do_acquiring()
            elif state == STATE_SERVOING:
                self._do_servoing()
            elif state == STATE_NEXT:
                self._do_next()
            elif state == STATE_RECOVERY:
                self._do_recovery()
            elif state == STATE_IDLE:
                break

        # Clean up on exit
        self._stop_wheels()
        self._center_neck()
        logger.info("Navigation loop ended in state: %s", self._state)

    def _do_acquiring(self):
        """Acquire ORB features for the current waypoint."""
        with self._lock:
            if self._current_wp_index >= len(self._waypoints):
                self._set_state(STATE_IDLE)
                return
            wp = self._waypoints[self._current_wp_index]
            frame_path = self._frame_path_for_init

        logger.info("Acquiring features for waypoint %d: (%.2f, %.2f)",
                     self._current_wp_index + 1, wp["x"], wp["y"])
        self._publish_debug_log(
            "acquiring",
            f"Acquiring WP{self._current_wp_index + 1} at ({wp['x']:.2f}, {wp['y']:.2f})",
        )

        # Use the initial frame or capture a fresh one
        if frame_path and os.path.isfile(frame_path):
            frame = cv2.imread(frame_path)
        else:
            frame = self._capture_frame()

        if frame is None:
            logger.error("Failed to get frame for acquisition")
            self._set_state(STATE_IDLE)
            return

        if self._tracker.initialize_target(frame, wp["x"], wp["y"]):
            self._reached_counter = 0
            self._set_state(STATE_SERVOING)
            mode = self._tracker.get_mode() or "unknown"
            self._publish_debug_log(
                "acquired",
                f"Locked WP{self._current_wp_index + 1} [{mode}] "
                f"with {self._tracker.get_feature_count()} features",
            )
        else:
            logger.warning("Feature acquisition failed — not enough features")
            self._recovery_attempts += 1
            if self._recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
                self._set_state(STATE_IDLE)
                self._publish_debug_log("acquisition_failed",
                                        "Feature acquisition failed after max retries")
            else:
                # Try capturing a new frame
                self._frame_path_for_init = None
                self._set_state(STATE_RECOVERY)

    def _do_servoing(self):
        """One cycle of the visual servoing loop."""
        period = 1.0 / SERVO_LOOP_HZ
        start = time.monotonic()

        # Capture a new frame
        frame = self._capture_frame()
        if frame is None:
            logger.debug("Frame capture failed during servoing")
            # Don't immediately fail — try again next cycle
            elapsed = time.monotonic() - start
            if elapsed < period:
                self._stop_event.wait(period - elapsed)
            return

        # Update tracker
        track_x, track_y, valid = self._tracker.update(frame)

        if not valid:
            logger.warning("Tracking lost — entering recovery")
            self._stop_wheels()
            self._set_state(STATE_RECOVERY)
            self._publish_debug_log("tracking_lost",
                                    f"Tracking lost at WP{self._current_wp_index + 1}")
            return

        # Update debug visualization — only write to disk if someone is watching
        if self._has_active_viewer():
            self._debug_viz.update_live_frame(
                frame, self._current_wp_index, track_x, track_y,
            )

        # Convert tracked position to normalized camera coords for IBVS
        track_px, track_py = image_coords_to_pixel(track_x, track_y)
        nx, ny = pixel_to_normalized(track_px, track_py)

        # Target: image center in normalized coords = (0, 0)
        target_nx, target_ny = 0.0, 0.0

        # Depth heuristic from vertical image position
        z_est = Z_MIN + (1.0 - track_y) * (Z_MAX - Z_MIN)

        # Compute IBVS control
        left_rpm, right_rpm, neck_cmd = self._compute_control(
            nx, ny, target_nx, target_ny, z_est,
        )

        # Publish wheel command
        self._publish_wheel_cmd(left_rpm, right_rpm)

        # Publish neck command
        self._publish_neck_cmd(neck_cmd)

        # Publish motor debug
        self._publish_debug_motor(left_rpm, right_rpm, neck_cmd)

        # Check waypoint reached
        error = math.sqrt(nx * nx + ny * ny)
        if error < REACHED_THRESHOLD:
            self._reached_counter += 1
            if self._reached_counter >= REACHED_FRAMES:
                self._set_state(STATE_NEXT)
                self._publish_debug_log(
                    "waypoint_reached",
                    f"WP{self._current_wp_index + 1} reached (error={error:.3f})",
                )
        else:
            self._reached_counter = 0

        # Pace the loop
        elapsed = time.monotonic() - start
        if elapsed < period:
            self._stop_event.wait(period - elapsed)

    def _do_next(self):
        """Advance to the next waypoint."""
        self._stop_wheels()

        with self._lock:
            self._current_wp_index += 1
            self._recovery_attempts = 0
            # After the first waypoint, always capture fresh frames
            self._frame_path_for_init = None

        # Update debug visualization
        if self._current_wp_index >= len(self._waypoints):
            debug_path = self._debug_viz.mark_complete()
            if debug_path:
                self._publish_debug_path_image(debug_path)
            self._publish_debug_log("navigation_complete",
                                    f"All {len(self._waypoints)} waypoints reached")
            self._center_neck()
            self._set_state(STATE_IDLE)
        else:
            debug_path = self._debug_viz.update_progress(self._current_wp_index)
            if debug_path:
                self._publish_debug_path_image(debug_path)
            self._publish_debug_log(
                "next_waypoint",
                f"Advancing to WP{self._current_wp_index + 1}/{len(self._waypoints)}",
            )
            self._set_state(STATE_ACQUIRING)

    def _do_recovery(self):
        """Attempt to re-acquire tracking after loss."""
        self._stop_wheels()
        self._recovery_attempts += 1

        if self._recovery_attempts > MAX_RECOVERY_ATTEMPTS:
            logger.error("Max recovery attempts reached — aborting navigation")
            self._center_neck()
            self._set_state(STATE_IDLE)
            self._publish_debug_log("recovery_failed",
                                    "Max recovery attempts — navigation aborted")
            return

        self._publish_debug_log(
            "recovery",
            f"Recovery attempt {self._recovery_attempts}/{MAX_RECOVERY_ATTEMPTS}",
        )

        # Wait a moment for the robot to settle
        self._stop_event.wait(0.5)
        if self._stop_event.is_set():
            return

        # Capture fresh frame and try to re-acquire
        self._frame_path_for_init = None
        self._set_state(STATE_ACQUIRING)

    # ------------------------------------------------------------------
    # IBVS + Neck control
    # ------------------------------------------------------------------

    def _compute_control(self, nx, ny, target_nx, target_ny, z_est):
        """Compute wheel velocities and neck angle using IBVS.

        Control architecture:
          - NECK: fast proportional response to center the feature horizontally.
            Reacts to the image-space horizontal error (ex).
          - WHEELS (yaw): slow body re-alignment driven by the NECK ANGLE,
            not by the image error. The goal is to bring the body in line
            with where the neck is pointing, so the neck can return to center.
          - WHEELS (forward): driven by vertical error to approach the waypoint.

        This separation means the neck handles fast disturbances (body rotation)
        while the wheels gradually align the body to follow.

        Args:
            nx, ny: Current feature in normalized camera coords.
            target_nx, target_ny: Desired feature position (0, 0 = center).
            z_est: Estimated depth to feature in meters.

        Returns:
            (left_rpm, right_rpm, neck_deg)
        """
        ex = nx - target_nx  # Horizontal error (positive = feature is right of center)
        ey = ny - target_ny  # Vertical error (positive = feature is below center)

        # --- Neck: fast proportional tracking of horizontal error ---
        # Directly maps image error to neck angle.
        # Feature left of center (ex < 0) → neck turns left (negative degrees).
        # Feature right of center (ex > 0) → neck turns right (positive degrees).
        # Servo convention: negative = left, positive = right (from idle BT).
        neck_cmd = ex * NECK_GAIN
        neck_cmd = max(-NECK_MAX, min(NECK_MAX, neck_cmd))
        self._neck_deg = neck_cmd

        # --- Wheels: forward velocity from vertical error ---
        vz = -LAMBDA_GAIN * (-ny / z_est) * ey

        # --- Wheels: yaw driven by NECK ANGLE (not image error) ---
        # The neck angle tells us how far the body is misaligned from
        # the direction of travel. Wheel yaw works to reduce this angle,
        # bringing the body in line with where the camera is looking.
        # neck_deg negative = neck turned left = body should turn left
        wz = -self._neck_deg * NECK_RECENTER_GAIN * LAMBDA_GAIN / NECK_MAX

        # Map to differential drive
        v_linear = vz * LINEAR_SCALE
        v_angular = wz * ANGULAR_SCALE

        left_rpm = v_linear - v_angular
        right_rpm = v_linear + v_angular

        # Clamp
        left_rpm = max(-MAX_RPM, min(MAX_RPM, left_rpm))
        right_rpm = max(-MAX_RPM, min(MAX_RPM, right_rpm))

        return left_rpm, right_rpm, neck_cmd

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _capture_frame(self):
        """Request a frame from vision service and load it.

        Returns:
            BGR numpy array or None on failure.
        """
        self._frame_event.clear()
        self._frame_path = None
        self._waiting_for_frame = True

        self._client.publish(
            TOPIC_VISION_CMD,
            json.dumps({"command": "capture_frame"}),
            qos=1,
        )

        if not self._frame_event.wait(timeout=FRAME_TIMEOUT):
            self._waiting_for_frame = False
            return None

        self._waiting_for_frame = False
        path = self._frame_path

        if not path or not os.path.isfile(path):
            return None

        frame = cv2.imread(path)
        return frame

    # ------------------------------------------------------------------
    # Motor / servo commands
    # ------------------------------------------------------------------

    def _publish_wheel_cmd(self, left_vel, right_vel):
        """Publish wheel velocity command."""
        self._client.publish(
            TOPIC_WHEELS_CMD,
            json.dumps({"left_vel": round(left_vel, 2),
                        "right_vel": round(right_vel, 2)}),
            qos=1,
        )

    def _stop_wheels(self):
        """Immediately stop both wheels."""
        self._publish_wheel_cmd(0.0, 0.0)

    def _publish_neck_cmd(self, angle_deg):
        """Command the neck servo to a specific angle."""
        self._client.publish(
            TOPIC_JOINTS_CMD,
            json.dumps({
                "type": "joint_move_request",
                "joint_name": "neck",
                "target_position": round(angle_deg, 1),
                "speed": NECK_SPEED,
                "movement_type": "linear",
            }),
            qos=1,
        )

    def _center_neck(self):
        """Return neck to center position."""
        self._neck_deg = 0.0
        self._publish_neck_cmd(0.0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state):
        """Update state and publish."""
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.info("State: %s → %s", old, new_state)
        self.publish_state()

    # ------------------------------------------------------------------
    # Debug publishing
    # ------------------------------------------------------------------

    def _publish_debug_path_image(self, filepath):
        """Publish path to the annotated debug image."""
        self._client.publish(
            TOPIC_DEBUG_PATH_IMAGE,
            json.dumps({"file": filepath}),
            qos=1,
        )

    def _publish_debug_motor(self, left_vel, right_vel, neck_deg):
        """Publish human-readable motor control debug info."""
        avg_speed = (abs(left_vel) + abs(right_vel)) / 2.0
        diff = right_vel - left_vel

        # Determine direction description
        if avg_speed < 0.5:
            direction_desc = "stopped"
        elif abs(diff) < 1.0:
            direction_desc = "forward" if (left_vel + right_vel) > 0 else "backward"
        elif diff > 0:
            direction_desc = "forward-right" if (left_vel + right_vel) > 0 else "backward-left"
        else:
            direction_desc = "forward-left" if (left_vel + right_vel) > 0 else "backward-right"

        # Direction angle from differential
        if avg_speed > 0.5:
            direction_deg = math.degrees(math.atan2(diff, (left_vel + right_vel) / 2.0))
        else:
            direction_deg = 0.0

        self._client.publish(
            TOPIC_DEBUG_MOTOR,
            json.dumps({
                "left_vel": round(left_vel, 2),
                "right_vel": round(right_vel, 2),
                "linear": round((left_vel + right_vel) / 2.0, 2),
                "angular": round(diff, 2),
                "direction_deg": round(direction_deg, 1),
                "speed_magnitude": round(avg_speed, 2),
                "neck_deg": round(neck_deg, 1),
                "description": direction_desc,
            }),
            qos=0,
        )

    def _publish_debug_log(self, action, details):
        """Publish a human-readable navigation log entry."""
        # Format as decoded human-readable command info
        if action == "wheel_cmd":
            # This case is handled in _publish_debug_motor
            pass

        self._client.publish(
            TOPIC_DEBUG_LOG,
            json.dumps({
                "timestamp": time.time(),
                "action": action,
                "details": details,
            }),
            qos=0,
        )
        logger.info("[NAV LOG] %s: %s", action, details)
