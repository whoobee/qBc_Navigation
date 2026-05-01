"""
Navigation controller — reactive free-space steering with odometry goals.

Consumes waypoints from the exploration handler (qBc_Ai), converts them to
odometry-frame goals, and navigates by steering toward open floor corridors
in the direction of each goal.

Instead of tracking specific visual features (ORB/template), the controller
segments the drivable floor in every frame and steers toward the largest
open corridor that is closest to the desired heading.  Waypoint arrival is
determined by odometry distance, not visual convergence.

State machine:
    IDLE → NAVIGATING → NEXT → (loop or IDLE)
    Safety triggers short-circuit to IDLE.
    ToF collision guard marks the current waypoint reached (→ NEXT).
"""

import json
import logging
import math
import os
import threading
import time

import cv2
import numpy as np

from floor_segmenter import FloorSegmenter
from debug_visualizer import DebugVisualizer
from camera_model import IMG_WIDTH, IMG_HEIGHT

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
VIEWER_TIMEOUT = 3.0

# Debug topics
TOPIC_DEBUG_PATH_IMAGE = "robot/navigation/debug/path_image"
TOPIC_DEBUG_MOTOR = "robot/navigation/debug/motor"
TOPIC_DEBUG_LOG = "robot/navigation/debug/log"

# Navigation states
STATE_IDLE = "idle"
STATE_NAVIGATING = "navigating"
STATE_NEXT = "next"

# Camera FOV (Pi Camera v3, degrees)
H_FOV_DEG = 66.0

# Waypoint depth estimation
Z_MIN = 0.3     # meters — depth when waypoint is at bottom of frame
Z_MAX = 3.0     # meters — depth when waypoint is at top of frame

# Corridor steering
NUM_CORRIDORS = 7
MIN_CORRIDOR_OPENNESS = 0.12    # Below this the corridor is "blocked"
GOAL_WEIGHT = 0.4               # Preference toward goal heading
OPEN_WEIGHT = 0.6               # Preference toward open corridors
BLOCKED_THRESHOLD = 0.10        # If best corridor below this, stop completely

# Wheel control
MAX_RPM = 30.0
STEER_GAIN = 50.0               # Corridor offset → angular RPM
FORWARD_GAIN = 25.0             # Center openness → forward RPM
MIN_FORWARD_OPENNESS = 0.15     # Don't drive forward if center is too blocked

# Waypoint reached
REACHED_MARGIN_MM = 200.0       # Mark waypoint reached within this distance

# Neck (kept centered for now — body steers)
NECK_SPEED = 120.0

# Frame capture
SHM_NAV_FRAME = "/dev/shm/qb_nav_frame.jpg"
SERVO_LOOP_HZ = 15

# ToF collision guard defaults (overridable via robot/settings/navigation)
TOF_COLLISION_MM_DEFAULT = 150
# Final-waypoint ToF stop distance — keep more clearance from the destination
# object so the bot doesn't look like it's about to ram it.
TOF_FINAL_TARGET_MM_DEFAULT = 300


class NavigationController:
    """Reactive free-space navigation controller."""

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
        self._segmenter = FloorSegmenter(num_corridors=NUM_CORRIDORS)
        self._debug_viz = DebugVisualizer()
        self._lock = threading.Lock()

        # State
        self._state = STATE_IDLE
        self._waypoints = []
        self._goals = []           # odometry-frame goals [{x_mm, y_mm}]
        self._current_wp_index = 0

        # Per-waypoint odometry tracking
        self._wp_start_x = 0.0
        self._wp_start_y = 0.0
        self._wp_target_dist_mm = 0.0

        # Safety
        self._safety_ok = True

        # Sensor state
        self._odom_x_mm = 0.0
        self._odom_y_mm = 0.0
        self._odom_heading_deg = 0.0
        self._imu_pitch_deg = 0.0
        self._tof_front_mm = float('inf')

        # Bench test mode — robot suspended, ignore odometry-based arrival
        # and ToF collision guard so phantom wheel-encoder travel doesn't
        # falsely complete waypoints.
        self._bench_test_mode = False

        # ToF stop thresholds (configurable via robot/settings/navigation)
        self._tof_intermediate_mm = TOF_COLLISION_MM_DEFAULT
        self._tof_final_target_mm = TOF_FINAL_TARGET_MM_DEFAULT

        # Starting pose (recorded when navigation begins)
        self._start_heading_deg = 0.0

        # Frame rate measurement
        self._last_frame_time = 0.0
        self._actual_fps = 0.0

        # Frame synchronization (for on-demand capture fallback)
        self._waiting_for_frame = False
        self._frame_path = None
        self._frame_event = threading.Event()

        # Navigation thread
        self._nav_thread = None
        self._stop_event = threading.Event()

        # Viewer heartbeat
        self._viewer_last_seen = 0.0

    def register_callbacks(self, client):
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
        for topic, qos in self.subscriptions:
            client.subscribe(topic, qos=qos)

    def publish_state(self):
        self._client.publish(
            TOPIC_NAV_STATE,
            json.dumps({
                "status": "online",
                "nav_state": self._state,
                "current_waypoint": self._current_wp_index,
                "total_waypoints": len(self._waypoints),
                "fps": round(self._actual_fps, 1),
            }),
            qos=1, retain=True,
        )

    def shutdown(self):
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
    # Viewer stream (idle camera feed for debug applet)
    # ------------------------------------------------------------------

    def _on_viewer_heartbeat(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            data = {}
        if data.get("active") is False:
            self._viewer_last_seen = 0.0
            DebugVisualizer.cleanup_temp()
            return
        self._viewer_last_seen = time.monotonic()

    def _has_active_viewer(self):
        return (time.monotonic() - self._viewer_last_seen) < VIEWER_TIMEOUT

    def _on_stream_request(self, client, userdata, msg):
        if self._state != STATE_IDLE:
            return
        if not self._has_active_viewer():
            return
        threading.Thread(target=self._capture_stream_frame, daemon=True).start()

    def _capture_stream_frame(self):
        frame = self._read_shm_frame()
        if frame is None:
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
        logger.info("Received %d waypoints, frame: %s", len(waypoints), frame_path)

        with self._lock:
            self._waypoints = waypoints
            self._current_wp_index = 0

        # Convert image-space waypoints to odometry-frame goals
        self._start_heading_deg = self._odom_heading_deg
        self._goals = self._waypoints_to_goals(waypoints)
        logger.info("Converted %d waypoints to odometry goals", len(self._goals))

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
        self._nav_thread = threading.Thread(
            target=self._navigation_loop, daemon=True,
        )
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
            self._stop_wheels()
            self._publish_debug_log("pause", "Navigation paused")
        elif command == "resume":
            self._publish_debug_log("resume", "Navigation resumed")

    def _on_frame_ready(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        file_path = data.get("file")
        if file_path and self._waiting_for_frame:
            self._frame_path = file_path
            self._frame_event.set()

    def _on_safety(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
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
    # Sensor callbacks
    # ------------------------------------------------------------------

    def _on_odometry(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        # Null / NaN means the firmware has no confident reading right now —
        # keep the previous value rather than corrupting state.
        x = data.get("x_mm")
        y = data.get("y_mm")
        h = data.get("heading_deg")
        if x is not None:
            self._odom_x_mm = x
        if y is not None:
            self._odom_y_mm = y
        if h is not None:
            self._odom_heading_deg = h

    def _on_imu(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        pitch = data.get("pitch")
        if pitch is not None:
            self._imu_pitch_deg = pitch

    def _on_tof(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        # Use the dedicated front TOF (replaced the YDLIDAR). The bridge sends
        # null when the firmware has no confident reading — treat as +inf so
        # the collision guard doesn't fire on missing data.
        front = data.get("front_mm")
        if front is None or front <= 0:
            self._tof_front_mm = float('inf')
        else:
            self._tof_front_mm = float(front)

    def _on_settings(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if "bench_test_mode" in data:
            new_val = bool(data["bench_test_mode"])
            if new_val != self._bench_test_mode:
                logger.info("Bench test mode: %s → %s",
                            self._bench_test_mode, new_val)
                self._publish_debug_log(
                    "bench_test_mode",
                    f"Bench test mode {'ENABLED' if new_val else 'disabled'}",
                )
            self._bench_test_mode = new_val

        if "tof_intermediate_mm" in data:
            try:
                self._tof_intermediate_mm = int(data["tof_intermediate_mm"])
            except (TypeError, ValueError):
                pass

        if "tof_final_target_mm" in data:
            try:
                self._tof_final_target_mm = int(data["tof_final_target_mm"])
            except (TypeError, ValueError):
                pass

    # ------------------------------------------------------------------
    # Waypoint → odometry goal conversion
    # ------------------------------------------------------------------

    def _waypoints_to_goals(self, waypoints):
        """Convert image-space waypoints to odometry-frame goal positions.

        Each waypoint (x, y) in normalised image coords is converted to a
        heading angle (from camera FOV) and distance (from vertical position),
        then projected into the world frame using the current odometry pose.

        The goals are cumulative: each waypoint's body-frame offset is computed
        relative to the previous goal, not all from the starting pose.
        """
        goals = []
        # Accumulate in world frame starting from current odometry pose
        cur_x = self._odom_x_mm
        cur_y = self._odom_y_mm
        cur_heading_rad = math.radians(self._odom_heading_deg)

        for wp in waypoints:
            # Heading offset from camera center
            angle_rad = (wp["x"] - 0.5) * math.radians(H_FOV_DEG)

            # Distance estimate from vertical position in frame
            z_est = Z_MIN + (1.0 - wp["y"]) * (Z_MAX - Z_MIN)
            dist_mm = z_est * 1000.0

            # Body-frame offset: forward (+y) and lateral (+x = right)
            body_fwd = dist_mm * math.cos(angle_rad)
            body_lat = dist_mm * math.sin(angle_rad)

            # Rotate into world frame and offset from current position
            world_dx = (body_fwd * math.cos(cur_heading_rad)
                        - body_lat * math.sin(cur_heading_rad))
            world_dy = (body_fwd * math.sin(cur_heading_rad)
                        + body_lat * math.cos(cur_heading_rad))

            goal_x = cur_x + world_dx
            goal_y = cur_y + world_dy

            goals.append({
                "x_mm": goal_x,
                "y_mm": goal_y,
                "dist_mm": dist_mm,
            })

            # Next waypoint is relative to this goal
            cur_x = goal_x
            cur_y = goal_y
            cur_heading_rad += angle_rad

        return goals

    # ------------------------------------------------------------------
    # Navigation loop
    # ------------------------------------------------------------------

    def _navigation_loop(self):
        """Main navigation loop — reactive free-space steering."""
        self._start_nav_stream()
        time.sleep(0.3)  # let stream produce a few frames

        # Initialise first waypoint tracking
        self._begin_waypoint()
        self._set_state(STATE_NAVIGATING)

        while not self._stop_event.is_set():
            if not self._safety_ok:
                self._stop_wheels()
                self._set_state(STATE_IDLE)
                break

            state = self._state
            if state == STATE_NAVIGATING:
                self._do_navigating()
            elif state == STATE_NEXT:
                self._do_next()
            elif state == STATE_IDLE:
                break

        self._stop_wheels()
        self._center_neck()
        self._stop_nav_stream()
        logger.info("Navigation loop ended in state: %s", self._state)

    def _begin_waypoint(self):
        """Record odometry at the start of a new waypoint leg."""
        self._wp_start_x = self._odom_x_mm
        self._wp_start_y = self._odom_y_mm

        if self._current_wp_index < len(self._goals):
            goal = self._goals[self._current_wp_index]
            self._wp_target_dist_mm = goal["dist_mm"]
        else:
            self._wp_target_dist_mm = 0.0

        self._publish_debug_log(
            "begin_waypoint",
            f"WP{self._current_wp_index + 1}: "
            f"target_dist={self._wp_target_dist_mm:.0f}mm",
        )

    def _do_navigating(self):
        """One cycle of the reactive navigation loop."""
        period = 1.0 / SERVO_LOOP_HZ
        start = time.monotonic()

        # ── ToF collision guard ──
        # Skipped in bench test mode — robot is suspended so ToF readings
        # may be misleading and we don't want phantom completions.
        is_final_wp = self._current_wp_index >= len(self._waypoints) - 1
        tof_threshold = (self._tof_final_target_mm if is_final_wp
                         else self._tof_intermediate_mm)
        if (not self._bench_test_mode
                and self._tof_front_mm < tof_threshold):
            self._stop_wheels()
            self._publish_debug_log(
                "tof_reached",
                f"ToF collision guard: {self._tof_front_mm:.0f}mm "
                f"(threshold {tof_threshold}mm) "
                f"— marking WP{self._current_wp_index + 1} reached",
            )
            self._set_state(STATE_NEXT)
            return

        # ── Read frame ──
        frame = self._read_shm_frame()
        if frame is None:
            frame = self._capture_frame()
        if frame is None:
            elapsed = time.monotonic() - start
            if elapsed < period:
                self._stop_event.wait(period - elapsed)
            return

        # Measure FPS
        now = time.monotonic()
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                self._actual_fps = 1.0 / dt
        self._last_frame_time = now

        # ── Segment floor ──
        floor_mask = self._segmenter.segment(frame)
        corridors = self._segmenter.compute_corridors(floor_mask)

        # ── Compute goal bearing ──
        goal_bearing_deg = self._compute_goal_bearing()

        # ── Select best corridor ──
        selected_idx = self._select_corridor(corridors, goal_bearing_deg)

        # ── Compute wheel commands ──
        if selected_idx is None:
            # All corridors blocked — stop
            self._stop_wheels()
            self._publish_debug_log("blocked", "All corridors blocked — waiting")
        else:
            left_rpm, right_rpm = self._compute_wheel_cmd(
                corridors, selected_idx,
            )
            self._publish_wheel_cmd(left_rpm, right_rpm)
            self._publish_debug_motor(left_rpm, right_rpm, 0.0)

        # ── Debug visualization ──
        if self._has_active_viewer():
            self._debug_viz.update_live_frame_reactive(
                frame,
                self._current_wp_index,
                floor_mask=floor_mask,
                corridors=corridors,
                goal_bearing_deg=goal_bearing_deg,
                selected_corridor=selected_idx,
            )

        # ── Check waypoint reached (odometry distance) ──
        # Skipped in bench test mode — wheel encoders integrate phantom
        # travel while the robot is suspended, which would falsely complete
        # waypoints almost instantly.
        if not self._bench_test_mode:
            dx = self._odom_x_mm - self._wp_start_x
            dy = self._odom_y_mm - self._wp_start_y
            traveled_mm = math.sqrt(dx * dx + dy * dy)

            if traveled_mm >= self._wp_target_dist_mm - REACHED_MARGIN_MM:
                self._publish_debug_log(
                    "waypoint_reached",
                    f"WP{self._current_wp_index + 1} reached "
                    f"(traveled={traveled_mm:.0f}mm, "
                    f"target={self._wp_target_dist_mm:.0f}mm)",
                )
                self._set_state(STATE_NEXT)
                return

        # Pace the loop
        elapsed = time.monotonic() - start
        if elapsed < period:
            self._stop_event.wait(period - elapsed)

    def _do_next(self):
        """Advance to the next waypoint or finish."""
        self._stop_wheels()

        with self._lock:
            self._current_wp_index += 1

        if self._current_wp_index >= len(self._waypoints):
            debug_path = self._debug_viz.mark_complete()
            if debug_path:
                self._publish_debug_path_image(debug_path)
            self._publish_debug_log(
                "navigation_complete",
                f"All {len(self._waypoints)} waypoints reached",
            )
            self._center_neck()
            self._set_state(STATE_IDLE)
        else:
            debug_path = self._debug_viz.update_progress(self._current_wp_index)
            if debug_path:
                self._publish_debug_path_image(debug_path)
            self._publish_debug_log(
                "next_waypoint",
                f"Advancing to WP{self._current_wp_index + 1}"
                f"/{len(self._waypoints)}",
            )
            self._begin_waypoint()
            self._set_state(STATE_NAVIGATING)

    # ------------------------------------------------------------------
    # Corridor selection and steering
    # ------------------------------------------------------------------

    def _compute_goal_bearing(self):
        """Compute bearing from current odometry pose to the current goal.

        Returns the angle in degrees relative to the robot's current heading.
        Positive = goal is to the right, negative = left.
        """
        if self._current_wp_index >= len(self._goals):
            return 0.0

        goal = self._goals[self._current_wp_index]
        dx = goal["x_mm"] - self._odom_x_mm
        dy = goal["y_mm"] - self._odom_y_mm

        # Absolute bearing to goal (world frame)
        goal_bearing_abs = math.degrees(math.atan2(dy, dx))

        # Relative to robot's current heading
        relative = goal_bearing_abs - self._odom_heading_deg

        # Normalise to [-180, 180]
        while relative > 180.0:
            relative -= 360.0
        while relative < -180.0:
            relative += 360.0

        return relative

    def _select_corridor(self, corridors, goal_bearing_deg):
        """Pick the best corridor: open + close to goal heading.

        Returns the index of the selected corridor, or None if all blocked.
        """
        n = len(corridors)
        if n == 0:
            return None

        # Map goal bearing to a target corridor index
        # Bearing 0° = center corridor, ±(H_FOV/2) = edge corridors
        goal_frac = goal_bearing_deg / H_FOV_DEG + 0.5  # 0-1 range
        goal_frac = max(0.0, min(1.0, goal_frac))

        best_score = -1.0
        best_idx = None

        for i, c in enumerate(corridors):
            if c["openness"] < MIN_CORRIDOR_OPENNESS:
                continue

            # Proximity to goal (1.0 = exactly the goal corridor)
            corridor_frac = c["center_x"]
            proximity = 1.0 - abs(corridor_frac - goal_frac)

            score = (OPEN_WEIGHT * c["openness"]
                     + GOAL_WEIGHT * proximity)

            if score > best_score:
                best_score = score
                best_idx = i

        # If the best corridor is barely open, treat as blocked
        if best_idx is not None:
            if corridors[best_idx]["openness"] < BLOCKED_THRESHOLD:
                return None

        return best_idx

    def _compute_wheel_cmd(self, corridors, selected_idx):
        """Convert selected corridor to differential-drive wheel RPMs.

        Steering: proportional to offset between selected corridor center
        and image center (0.5).
        Forward speed: proportional to the selected corridor's openness.
        """
        c = corridors[selected_idx]
        center_idx = len(corridors) // 2

        # Steering: how far off-center is the selected corridor?
        steer_error = c["center_x"] - 0.5  # positive = right
        angular = steer_error * STEER_GAIN

        # Forward speed: proportional to selected corridor openness
        # Also scale down if we're steering hard
        steer_factor = max(0.3, 1.0 - abs(steer_error) * 2.0)
        forward_openness = corridors[center_idx]["openness"]

        if forward_openness < MIN_FORWARD_OPENNESS:
            # Center blocked — slow way down but still allow turning
            forward = FORWARD_GAIN * 0.2
        else:
            forward = FORWARD_GAIN * forward_openness * steer_factor

        left_rpm = forward - angular
        right_rpm = forward + angular

        # Clamp
        left_rpm = max(-MAX_RPM, min(MAX_RPM, left_rpm))
        right_rpm = max(-MAX_RPM, min(MAX_RPM, right_rpm))

        return left_rpm, right_rpm

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _read_shm_frame(self):
        try:
            if not os.path.isfile(SHM_NAV_FRAME):
                return None
            return cv2.imread(SHM_NAV_FRAME)
        except Exception:
            return None

    def _capture_frame(self):
        self._frame_event.clear()
        self._frame_path = None
        self._waiting_for_frame = True

        self._client.publish(
            TOPIC_VISION_CMD,
            json.dumps({"command": "capture_frame"}),
            qos=1,
        )

        if not self._frame_event.wait(timeout=0.5):
            self._waiting_for_frame = False
            return None

        self._waiting_for_frame = False
        path = self._frame_path
        if not path or not os.path.isfile(path):
            return None
        return cv2.imread(path)

    def _start_nav_stream(self):
        self._client.publish(
            TOPIC_VISION_CMD,
            json.dumps({"command": "start_nav_stream"}),
            qos=1,
        )

    def _stop_nav_stream(self):
        self._client.publish(
            TOPIC_VISION_CMD,
            json.dumps({"command": "stop_nav_stream"}),
            qos=1,
        )

    # ------------------------------------------------------------------
    # Motor / servo commands
    # ------------------------------------------------------------------

    def _publish_wheel_cmd(self, left_vel, right_vel):
        self._client.publish(
            TOPIC_WHEELS_CMD,
            json.dumps({"left_vel": round(left_vel, 2),
                        "right_vel": round(right_vel, 2)}),
            qos=1,
        )

    def _stop_wheels(self):
        self._publish_wheel_cmd(0.0, 0.0)

    def _publish_neck_cmd(self, angle_deg):
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
        self._publish_neck_cmd(0.0)

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _set_state(self, new_state):
        old = self._state
        self._state = new_state
        if old != new_state:
            logger.info("State: %s → %s", old, new_state)
        self.publish_state()

    # ------------------------------------------------------------------
    # Debug publishing
    # ------------------------------------------------------------------

    def _publish_debug_path_image(self, filepath):
        self._client.publish(
            TOPIC_DEBUG_PATH_IMAGE,
            json.dumps({"file": filepath}),
            qos=1,
        )

    def _publish_debug_motor(self, left_vel, right_vel, neck_deg):
        avg_speed = (abs(left_vel) + abs(right_vel)) / 2.0
        diff = right_vel - left_vel

        if avg_speed < 0.5:
            direction_desc = "stopped"
        elif abs(diff) < 1.0:
            direction_desc = "forward" if (left_vel + right_vel) > 0 else "backward"
        elif diff > 0:
            direction_desc = "forward-right" if (left_vel + right_vel) > 0 else "backward-left"
        else:
            direction_desc = "forward-left" if (left_vel + right_vel) > 0 else "backward-right"

        if avg_speed > 0.5:
            direction_deg = math.degrees(
                math.atan2(diff, (left_vel + right_vel) / 2.0),
            )
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
