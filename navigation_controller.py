"""
Navigation controller — state machine, IBVS visual servoing, neck stabilization.

Consumes waypoints from the exploration handler, tracks them visually with ORB,
and computes differential-drive motor commands using Image-Based Visual Servoing.

The neck servo provides fast horizontal stabilization to keep the tracked
feature centered while the wheel base turns more slowly.

State machine:
    IDLE → ACQUIRING → SERVOING → NEXT → (loop or IDLE)
    SERVOING → DEADRECKON on exit-gate detection (floor/ceiling target exit).
    RECOVERY on genuine tracking loss, with odometry backtrack.
    ToF collision guard can short-circuit to NEXT from SERVOING/DEADRECKON.
    IDLE on safety trigger.
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
TOPIC_IMU = "robot/imu/orientation"
TOPIC_TOF = "robot/sensors/tof"
TOPIC_SETTINGS_NAV = "robot/settings/navigation"
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
STATE_DEADRECKON = "deadreckon"

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

# Neck stabilization (PD controller — tuned for ~200ms frame latency)
NECK_KP = 10.0              # Proportional gain (degrees per unit error per cycle)
NECK_KD = 12.0              # Derivative gain (strong damping for high-latency loop)
NECK_RECENTER_GAIN = 1.0    # Wheel yaw driven by neck angle (deg → angular rate)
NECK_RECENTER_RATE = 0.92   # Decay factor for recentering inside deadzone
NECK_SPEED = 120.0          # Neck servo speed in deg/s (fast response)
NECK_MAX = 80.0             # Limit neck range (don't use full ±90)
NECK_DEADZONE = 0.03        # Normalized horizontal error below which neck holds still
NECK_INVERT = False         # Invert neck direction (True if camera is mirrored)

# Frame capture
FRAME_TIMEOUT = 0.5         # Seconds to wait for on-demand frame (acquisition only)
SHM_NAV_FRAME = "/dev/shm/qb_nav_frame.jpg"  # Shared-memory frame from vision stream
SERVO_LOOP_HZ = 15          # Target servoing frequency (achievable with SHM stream)

# Recovery
MAX_RECOVERY_ATTEMPTS = 3
RECOVERY_DRIVE_RPM = 15.0       # Conservative speed for backtracking
RECOVERY_MAX_BACKTRACK_MM = 500  # Cap backtrack distance

# Exit-gate detection (floor/ceiling target exits camera FOV)
EXIT_BOTTOM_THRESHOLD = 0.85    # track_y above this → bottom exit (floor target)
EXIT_TOP_THRESHOLD = 0.15       # track_y below this → top exit (ceiling target)
PITCH_LEVEL_TOLERANCE = 10.0    # Degrees — pitch within this of 0° is "level"
DEADRECKON_DEPTH_FACTOR = 0.3   # Fraction of last depth estimate as remaining distance
DEADRECKON_SPEED_RPM = 15.0     # Conservative forward speed during dead-reckoning
DEADRECKON_TIMEOUT = 5.0        # Safety timeout for dead-reckoning (seconds)

# ToF collision guard
TOF_COLLISION_MM = 150          # Mark reached if front distance below this


class NavigationController:
    """Visual servoing navigation controller with neck stabilization."""

    subscriptions = [
        (TOPIC_NAV_WAYPOINTS, 1),
        (TOPIC_NAV_CMD, 1),
        (TOPIC_FRAME_READY, 1),
        (TOPIC_SAFETY_STATUS, 1),
        (TOPIC_ODOMETRY, 0),
        (TOPIC_IMU, 0),
        (TOPIC_TOF, 0),
        (TOPIC_SETTINGS_NAV, 1),
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

        # Sensor state (from Teensy telemetry, updated via MQTT callbacks)
        self._odom_x_mm = 0.0
        self._odom_y_mm = 0.0
        self._odom_heading_deg = 0.0
        self._imu_pitch_deg = 0.0
        self._tof_front_mm = float('inf')

        # Acquisition pose (recorded when entering SERVOING for backtrack recovery)
        self._acq_pose = None  # (x_mm, y_mm, heading_deg)

        # Last valid tracking state (for exit-gate classification on loss)
        self._last_track_y = 0.5
        self._last_z_est = 1.0

        # PD controller state for neck
        self._prev_ex = 0.0

        # Frame rate measurement
        self._last_frame_time = 0.0
        self._actual_fps = 0.0

        # Dead-reckoning state (for driving remaining distance after exit-gate)
        self._deadreckon_target_mm = 0.0

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
        client.message_callback_add(TOPIC_ODOMETRY, self._on_odometry)
        client.message_callback_add(TOPIC_IMU, self._on_imu)
        client.message_callback_add(TOPIC_TOF, self._on_tof)
        client.message_callback_add(TOPIC_SETTINGS_NAV, self._on_settings)
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
                "fps": round(self._actual_fps, 1),
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

        floor_boundary = data.get("floor_boundary")
        logger.info("Received %d waypoints, frame: %s, floor: %s",
                     len(waypoints), frame_path, floor_boundary)

        with self._lock:
            self._waypoints = waypoints
            self._frame_path_for_init = frame_path
            self._current_wp_index = 0
            self._reached_counter = 0
            self._recovery_attempts = 0

        # Pass floor boundary to debug visualizer
        if floor_boundary is not None:
            self._debug_viz._floor_boundary = float(floor_boundary)

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
    # Sensor callbacks (odometry, IMU, ToF)
    # ------------------------------------------------------------------

    def _on_odometry(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        self._odom_x_mm = data.get("x_mm", self._odom_x_mm)
        self._odom_y_mm = data.get("y_mm", self._odom_y_mm)
        self._odom_heading_deg = data.get("heading_deg", self._odom_heading_deg)

    def _on_imu(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        self._imu_pitch_deg = data.get("pitch", self._imu_pitch_deg)

    def _on_tof(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        left = data.get("left_mm", float('inf'))
        right = data.get("right_mm", float('inf'))
        # Use minimum of forward-facing sensors as front distance proxy
        self._tof_front_mm = min(left, right)

    def _on_settings(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        global NECK_DEADZONE, NECK_INVERT
        if "neck_deadzone" in data:
            NECK_DEADZONE = max(0.0, min(0.72, float(data["neck_deadzone"])))
            self._debug_viz._neck_deadzone = NECK_DEADZONE
            logger.info("Neck deadzone updated: %.3f", NECK_DEADZONE)
        if "neck_invert" in data:
            NECK_INVERT = bool(data["neck_invert"])
            logger.info("Neck invert updated: %s", NECK_INVERT)

    # ------------------------------------------------------------------
    # Navigation loop (runs in a thread)
    # ------------------------------------------------------------------

    def _navigation_loop(self):
        """Main navigation state machine loop."""
        # Start continuous capture stream for low-latency servoing
        self._start_nav_stream()
        # Brief delay for the stream to start producing frames
        time.sleep(0.3)

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
            elif state == STATE_DEADRECKON:
                self._do_deadreckon()
            elif state == STATE_IDLE:
                break

        # Clean up on exit
        self._stop_wheels()
        self._center_neck()
        self._stop_nav_stream()
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

        # Use the initial frame, SHM stream, or MQTT capture (in priority order)
        if frame_path and os.path.isfile(frame_path):
            frame = cv2.imread(frame_path)
        else:
            frame = self._read_shm_frame()
            if frame is None:
                frame = self._capture_frame()

        if frame is None:
            logger.error("Failed to get frame for acquisition")
            self._set_state(STATE_IDLE)
            return

        if self._tracker.initialize_target(frame, wp["x"], wp["y"]):
            self._reached_counter = 0
            self._acq_pose = (self._odom_x_mm, self._odom_y_mm, self._odom_heading_deg)
            self._set_state(STATE_SERVOING)
            mode = self._tracker.get_mode() or "unknown"
            self._publish_debug_log(
                "acquired",
                f"Locked WP{self._current_wp_index + 1} [{mode}] "
                f"with {self._tracker.get_feature_count()} features",
            )
        else:
            logger.warning("Feature acquisition failed — not enough features")
            # Don't increment _recovery_attempts here — _do_recovery handles the counting.
            # Just transition to RECOVERY which will retry or abort.
            self._frame_path_for_init = None
            self._set_state(STATE_RECOVERY)

    def _do_servoing(self):
        """One cycle of the visual servoing loop."""
        period = 1.0 / SERVO_LOOP_HZ
        start = time.monotonic()

        # ── ToF collision guard ──
        if self._tof_front_mm < TOF_COLLISION_MM:
            self._stop_wheels()
            self._publish_debug_log(
                "tof_reached",
                f"ToF collision guard: {self._tof_front_mm:.0f}mm < {TOF_COLLISION_MM}mm "
                f"— marking WP{self._current_wp_index + 1} reached",
            )
            self._set_state(STATE_NEXT)
            return

        # Read latest frame from shared memory (no MQTT round-trip)
        frame = self._read_shm_frame()
        if frame is None:
            # Fall back to MQTT-based capture if SHM not available
            frame = self._capture_frame()
        if frame is None:
            logger.debug("Frame capture failed during servoing")
            elapsed = time.monotonic() - start
            if elapsed < period:
                self._stop_event.wait(period - elapsed)
            return

        # Measure actual frame rate
        now = time.monotonic()
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                self._actual_fps = 1.0 / dt
        self._last_frame_time = now

        # Update tracker
        track_x, track_y, valid = self._tracker.update(frame)

        if valid:
            # Store tracking state every cycle for exit-gate analysis on loss
            self._last_track_y = track_y
            self._last_z_est = Z_MIN + (1.0 - track_y) * (Z_MAX - Z_MIN)

        if not valid:
            self._stop_wheels()

            # ── Exit-gate detection ──
            exit_type = self._classify_exit(self._last_track_y)

            if exit_type:
                # Target exited the frame naturally — dead-reckon remaining distance
                remaining_mm = self._last_z_est * DEADRECKON_DEPTH_FACTOR * 1000.0
                self._deadreckon_target_mm = remaining_mm
                self._publish_debug_log(
                    f"exit_gate_{exit_type}",
                    f"Target exited {exit_type} (y={self._last_track_y:.2f}, "
                    f"pitch={self._imu_pitch_deg:.1f}deg) — "
                    f"dead-reckoning {remaining_mm:.0f}mm forward",
                )
                self._set_state(STATE_DEADRECKON)
            else:
                # Genuine tracking loss — recovery
                logger.warning("Tracking lost — entering recovery")
                self._set_state(STATE_RECOVERY)
                self._publish_debug_log(
                    "tracking_lost",
                    f"Tracking lost at WP{self._current_wp_index + 1} "
                    f"(last_y={self._last_track_y:.2f})",
                )
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
        z_est = self._last_z_est

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
        """Attempt to re-acquire tracking after loss, with odometry backtracking."""
        self._stop_wheels()
        self._recovery_attempts += 1

        if self._recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
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

        # Backtrack toward acquisition pose if available
        if self._acq_pose is not None:
            acq_x, acq_y, _ = self._acq_pose
            dx = acq_x - self._odom_x_mm
            dy = acq_y - self._odom_y_mm
            dist_mm = math.sqrt(dx * dx + dy * dy)

            if dist_mm > 20.0:
                backtrack_mm = min(dist_mm, RECOVERY_MAX_BACKTRACK_MM)
                self._publish_debug_log(
                    "recovery_backtrack",
                    f"Reversing {backtrack_mm:.0f}mm toward acquisition pose",
                )
                self._drive_distance(-RECOVERY_DRIVE_RPM, backtrack_mm)

        # Wait a moment for the robot to settle
        self._stop_event.wait(0.5)
        if self._stop_event.is_set():
            return

        # Capture fresh frame and try to re-acquire
        self._frame_path_for_init = None
        self._set_state(STATE_ACQUIRING)

    # ------------------------------------------------------------------
    # Exit-gate detection and dead-reckoning
    # ------------------------------------------------------------------

    def _classify_exit(self, last_y):
        """Classify whether tracking loss is an exit-gate or genuine loss.

        Uses the last known vertical image position and IMU pitch to determine
        if the target naturally exited the camera FOV (floor/ceiling) vs a
        genuine tracking failure (mid-frame loss or side exit).

        Returns:
            'bottom', 'top', or None (genuine loss).
        """
        pitch = self._imu_pitch_deg

        # Bottom exit: floor target dropped below camera FOV
        if last_y > EXIT_BOTTOM_THRESHOLD:
            # Accept if pitch is level or nose-down (positive or near-zero)
            if pitch > -PITCH_LEVEL_TOLERANCE:
                return "bottom"

        # Top exit: ceiling target rose above camera FOV
        if last_y < EXIT_TOP_THRESHOLD:
            # Accept if pitch is level or nose-up (negative or near-zero)
            if pitch < PITCH_LEVEL_TOLERANCE:
                return "top"

        return None

    def _do_deadreckon(self):
        """Drive forward a computed distance after exit-gate, then mark reached.

        Uses odometry to measure actual distance traveled rather than
        pure time-based estimation.
        """
        target_mm = self._deadreckon_target_mm
        if target_mm <= 0:
            self._set_state(STATE_NEXT)
            return

        start_x = self._odom_x_mm
        start_y = self._odom_y_mm
        start_time = time.monotonic()

        self._publish_debug_log("deadreckon_start",
                                f"Driving forward {target_mm:.0f}mm")

        self._publish_wheel_cmd(DEADRECKON_SPEED_RPM, DEADRECKON_SPEED_RPM)

        while not self._stop_event.is_set():
            if not self._safety_ok:
                self._stop_wheels()
                self._set_state(STATE_IDLE)
                return

            # ToF collision guard during dead-reckoning
            if self._tof_front_mm < TOF_COLLISION_MM:
                self._stop_wheels()
                self._publish_debug_log(
                    "tof_reached",
                    f"ToF collision during dead-reckon: {self._tof_front_mm:.0f}mm",
                )
                self._set_state(STATE_NEXT)
                return

            # Check distance traveled via odometry
            dx = self._odom_x_mm - start_x
            dy = self._odom_y_mm - start_y
            traveled = math.sqrt(dx * dx + dy * dy)

            if traveled >= target_mm:
                self._stop_wheels()
                self._publish_debug_log(
                    "deadreckon_reached",
                    f"Traveled {traveled:.0f}mm — WP{self._current_wp_index + 1} reached",
                )
                self._set_state(STATE_NEXT)
                return

            # Timeout safety
            if (time.monotonic() - start_time) > DEADRECKON_TIMEOUT:
                self._stop_wheels()
                self._publish_debug_log(
                    "deadreckon_timeout",
                    f"Timeout after {DEADRECKON_TIMEOUT:.0f}s — marking reached",
                )
                self._set_state(STATE_NEXT)
                return

            self._stop_event.wait(0.1)

        self._stop_wheels()

    def _drive_distance(self, rpm, distance_mm):
        """Drive at a given RPM until odometry shows the specified distance traveled.

        Args:
            rpm: Wheel speed (negative = reverse).
            distance_mm: Target distance in mm.
        """
        start_x = self._odom_x_mm
        start_y = self._odom_y_mm
        start_time = time.monotonic()

        self._publish_wheel_cmd(rpm, rpm)

        while not self._stop_event.is_set():
            dx = self._odom_x_mm - start_x
            dy = self._odom_y_mm - start_y
            traveled = math.sqrt(dx * dx + dy * dy)

            if traveled >= distance_mm:
                break
            if (time.monotonic() - start_time) > DEADRECKON_TIMEOUT:
                break
            if not self._safety_ok:
                break

            self._stop_event.wait(0.05)

        self._stop_wheels()

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
        if NECK_INVERT:
            ex = -ex           # Flip for mirrored camera mounting
        ey = ny - target_ny  # Vertical error (positive = feature is below center)

        # --- Neck: PD controller with dead-zone ---
        # Proportional term drives toward center, derivative term dampens
        # oscillation caused by latency between command and camera frame.
        # Inside the dead-zone, gently recenter toward 0°.
        if abs(ex) < NECK_DEADZONE:
            # Decay toward center — keeps target in the middle of the band
            neck_cmd = self._neck_deg * NECK_RECENTER_RATE
            if abs(neck_cmd) < 0.5:
                neck_cmd = 0.0
            self._prev_ex = ex
        else:
            # PD: correction = Kp * error + Kd * d(error)/dt
            d_ex = ex - self._prev_ex
            correction = NECK_KP * ex + NECK_KD * d_ex
            neck_cmd = self._neck_deg + correction
            self._prev_ex = ex
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

    def _read_shm_frame(self):
        """Read the latest frame from shared memory (fast, no MQTT round-trip).

        Returns:
            BGR numpy array or None if not available.
        """
        try:
            if not os.path.isfile(SHM_NAV_FRAME):
                return None
            frame = cv2.imread(SHM_NAV_FRAME)
            return frame
        except Exception:
            return None

    def _capture_frame(self):
        """Request a single frame from vision service via MQTT (for acquisition).

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

    def _start_nav_stream(self):
        """Tell the vision service to start continuous capture to shared memory."""
        self._client.publish(
            TOPIC_VISION_CMD,
            json.dumps({"command": "start_nav_stream"}),
            qos=1,
        )

    def _stop_nav_stream(self):
        """Tell the vision service to stop continuous capture."""
        self._client.publish(
            TOPIC_VISION_CMD,
            json.dumps({"command": "stop_nav_stream"}),
            qos=1,
        )

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
