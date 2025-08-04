#!/usr/bin/env python3
"""
Manual control example with target detection overlay
Shows camera feed with detected targets and obstacles
"""

import sys
import cv2
import time
import threading

sys.path.append("../src")

from jetbot import Camera
from target_detection import TargetDetector
from obstacle_avoidance import ObstacleDetector
from jetbot_controller import JetBotController


class ManualControlWithVision:
    def __init__(self):
        self.camera = Camera.instance(width=224, height=224)
        self.target_detector = TargetDetector()
        self.obstacle_detector = ObstacleDetector()
        self.robot_controller = JetBotController()

        self.latest_image = None
        self.running = True

        # Set target to look for
        target = input("Enter target description (default: red object): ").strip()
        if not target:
            target = "red object"
        self.target_detector.set_target_description(target)

        # Camera callback
        self.camera.observe(self._camera_callback, names="value")

    def _camera_callback(self, change):
        self.latest_image = change["new"]

    def run(self):
        print("Manual Control with Vision")
        print("==========================")
        print("Controls:")
        print("w/s: forward/backward")
        print("a/d: left/right")
        print("space: stop")
        print("q: quit")
        print("v: toggle vision display")

        show_vision = True

        try:
            while self.running:
                # Get keyboard input (simplified - in real implementation use proper keyboard handling)
                if self.latest_image is not None:
                    display_image = self.latest_image.copy()

                    # Detect target
                    target_bbox = self.target_detector.detect_target(display_image)
                    if target_bbox:
                        x, y, w, h = target_bbox
                        cv2.rectangle(
                            display_image, (x, y), (x + w, y + h), (0, 255, 0), 2
                        )
                        cv2.putText(
                            display_image,
                            "TARGET",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )

                    # Detect obstacles
                    obstacles = self.obstacle_detector.detect_obstacles_from_rgb(
                        display_image
                    )
                    for obs in obstacles:
                        x, y, w, h = obs
                        cv2.rectangle(
                            display_image, (x, y), (x + w, y + h), (0, 0, 255), 2
                        )
                        cv2.putText(
                            display_image,
                            "OBSTACLE",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            2,
                        )

                    # Show path clear status
                    path_clear = self.obstacle_detector.is_path_clear(display_image)
                    status_color = (0, 255, 0) if path_clear else (0, 0, 255)
                    status_text = "PATH CLEAR" if path_clear else "OBSTACLE AHEAD"
                    cv2.putText(
                        display_image,
                        status_text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        status_color,
                        2,
                    )

                    if show_vision:
                        cv2.imshow("JetBot Vision", display_image)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("w"):
                        self.robot_controller.set_speeds(0.3, 0.0)
                    elif key == ord("s"):
                        self.robot_controller.set_speeds(-0.3, 0.0)
                    elif key == ord("a"):
                        self.robot_controller.set_speeds(0.0, 1.0)
                    elif key == ord("d"):
                        self.robot_controller.set_speeds(0.0, -1.0)
                    elif key == ord(" "):
                        self.robot_controller.set_speeds(0.0, 0.0)
                    elif key == ord("v"):
                        show_vision = not show_vision
                        if not show_vision:
                            cv2.destroyAllWindows()
                    elif key == ord("q"):
                        break

                time.sleep(0.05)

        except KeyboardInterrupt:
            pass
        finally:
            self.robot_controller.stop()
            cv2.destroyAllWindows()


def main():
    controller = ManualControlWithVision()
    controller.run()


if __name__ == "__main__":
    main()
