#!/usr/bin/env python3
"""
qBc_Navigation — Visual servoing navigation service for qB Companion.

Consumes waypoints from the exploration handler (qBc_Ai), tracks them
using ORB features, and navigates toward them using Image-Based Visual
Servoing (IBVS) with neck servo stabilization.

MQTT topics:
    Subscribe:
        robot/navigation/waypoints      Waypoints from exploration handler
        robot/navigation/cmd            Navigation commands (cancel, pause, resume)
        robot/vision/frame_ready        Camera frame notifications
        robot/safety/status             Safety flags from Teensy
    Publish:
        robot/navigation/state          (RETAIN) Service + navigation state
        robot/system/heartbeat/navigation   Keepalive (1 Hz)
        robot/wheels/cmd                Wheel velocity commands
        robot/joints/cmd                Neck servo commands
        robot/navigation/debug/*        Debug visualization topics

Prerequisites:
    pip install opencv-contrib-python numpy paho-mqtt

Usage:
    python3 main.py [--mqtt-broker localhost] [--mqtt-port 1883]
"""

import argparse
import json
import logging
import signal
import threading

import paho.mqtt.client as mqtt

from navigation_controller import NavigationController

logger = logging.getLogger("qBc_Nav")

TOPIC_STATE = "robot/navigation/state"
TOPIC_HEARTBEAT = "robot/system/heartbeat/navigation"


class NavigationService:
    """Navigation service — manages MQTT lifecycle and the navigation controller."""

    def __init__(self, broker="localhost", port=1883):
        self.broker = broker
        self.port = port

        # MQTT client
        self.mqtt = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2,
            client_id="qbc_navigation",
        )
        self.mqtt.on_connect = self._on_connect
        self.mqtt.on_disconnect = self._on_disconnect
        self.mqtt.will_set(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )

        # Navigation controller
        self._controller = NavigationController(self.mqtt)
        self._controller.register_callbacks(self.mqtt)

    def _on_connect(self, client, userdata, connect_flags, reason_code, properties):
        if reason_code.is_failure:
            logger.error("MQTT connection failed: %s", reason_code)
            return
        logger.info("Connected to MQTT broker %s:%d", self.broker, self.port)

        # Subscribe controller topics
        self._controller.subscribe(client)

        # Publish online state
        client.publish(
            TOPIC_STATE,
            json.dumps({"status": "online", "nav_state": "idle"}),
            qos=1, retain=True,
        )
        self._controller.publish_state()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        if reason_code.is_failure:
            logger.warning("Disconnected from MQTT broker: %s", reason_code)

    def run(self):
        """Main service loop — connect MQTT and run heartbeat."""
        self.mqtt.connect(self.broker, self.port)
        self.mqtt.loop_start()

        logger.info("qBc_Navigation service running on MQTT %s:%d",
                     self.broker, self.port)

        stop = threading.Event()
        signal.signal(signal.SIGINT, lambda *_: stop.set())
        signal.signal(signal.SIGTERM, lambda *_: stop.set())

        while not stop.is_set():
            self.mqtt.publish(TOPIC_HEARTBEAT, b"1", qos=0)
            stop.wait(1.0)

        logger.info("Shutting down...")
        self._controller.shutdown()
        self.mqtt.publish(
            TOPIC_STATE,
            json.dumps({"status": "offline"}),
            qos=1, retain=True,
        )
        self.mqtt.loop_stop()
        self.mqtt.disconnect()


def main():
    parser = argparse.ArgumentParser(description="qBc_Navigation — Visual Servoing Navigation")
    parser.add_argument("--mqtt-broker", default="localhost",
                        help="MQTT broker address")
    parser.add_argument("--mqtt-port", type=int, default=1883,
                        help="MQTT broker port")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    service = NavigationService(
        broker=args.mqtt_broker,
        port=args.mqtt_port,
    )
    service.run()


if __name__ == "__main__":
    main()
