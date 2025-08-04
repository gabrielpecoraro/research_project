import cv2
import numpy as np
import torch
from typing import Tuple, Optional, List
from transformers import pipeline
import time


class TargetDetector:
    """
    Target detection using computer vision
    Supports both color-based and AI-based detection
    """

    def __init__(self, camera_width=224, camera_height=224):
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Initialize YOLO for object detection (if available)
        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO("yolov8n.pt")
            self.use_yolo = True
        except ImportError:
            self.use_yolo = False
            print("YOLO not available, using color-based detection")

        # Color ranges for different targets (HSV)
        self.color_ranges = {
            "red": ((0, 120, 70), (10, 255, 255)),
            "blue": ((100, 150, 0), (140, 255, 255)),
            "green": ((40, 40, 40), (80, 255, 255)),
            "yellow": ((20, 100, 100), (30, 255, 255)),
        }

        # Target description for AI-based detection
        self.target_description = ""

    def set_target_description(self, description: str):
        """Set target description for AI-based detection"""
        self.target_description = description.lower()

    def detect_by_color(
        self, image: np.ndarray, color: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect target by color

        Args:
            image: Input BGR image
            color: Color name ('red', 'blue', 'green', 'yellow')

        Returns:
            (x, y, width, height) of detected target or None
        """
        if color not in self.color_ranges:
            return None

        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Create mask for target color
        lower, upper = self.color_ranges[color]
        mask = cv2.inRange(hsv, lower, upper)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Filter by minimum area
        if cv2.contourArea(largest_contour) < 500:
            return None

        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        return (x, y, w, h)

    def detect_by_yolo(
        self, image: np.ndarray, target_class: str
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect target using YOLO object detection

        Args:
            image: Input BGR image
            target_class: YOLO class name (e.g., 'person', 'car', 'bottle')

        Returns:
            (x, y, width, height) of detected target or None
        """
        if not self.use_yolo:
            return None

        try:
            results = self.yolo_model(image, verbose=False)

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class name
                        class_id = int(box.cls[0])
                        class_name = self.yolo_model.names[class_id]

                        if class_name.lower() == target_class.lower():
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            return None

        except Exception as e:
            print(f"YOLO detection error: {e}")
            return None

    def detect_target(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Main target detection method
        Uses description to determine detection method

        Returns:
            (x, y, width, height) of detected target or None
        """
        if not self.target_description:
            return None

        # Check for color-based detection
        for color in self.color_ranges.keys():
            if color in self.target_description:
                result = self.detect_by_color(image, color)
                if result:
                    return result

        # Check for YOLO classes
        yolo_classes = ["person", "car", "truck", "bus", "bottle", "cup", "ball"]
        for yolo_class in yolo_classes:
            if yolo_class in self.target_description:
                result = self.detect_by_yolo(image, yolo_class)
                if result:
                    return result

        return None

    def get_target_center_world_position(
        self, bbox: Tuple[int, int, int, int], depth_image: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Convert bounding box to world position

        Args:
            bbox: (x, y, width, height) bounding box
            depth_image: Optional depth image for distance estimation

        Returns:
            (x, y) world position in meters relative to robot
        """
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2

        # Convert pixel coordinates to world coordinates
        # Assuming camera FOV and simple pinhole model
        camera_fov_horizontal = 62.2  # degrees (typical for JetBot camera)
        camera_fov_vertical = 48.8  # degrees

        # Normalize to [-1, 1]
        norm_x = (center_x - self.camera_width / 2) / (self.camera_width / 2)
        norm_y = (center_y - self.camera_height / 2) / (self.camera_height / 2)

        # Convert to angles
        angle_x = norm_x * (camera_fov_horizontal / 2) * np.pi / 180
        angle_y = norm_y * (camera_fov_vertical / 2) * np.pi / 180

        # Estimate distance (default if no depth available)
        if depth_image is not None and depth_image.shape[:2] == (
            self.camera_height,
            self.camera_width,
        ):
            distance = depth_image[center_y, center_x] / 1000.0  # Convert mm to meters
            if distance == 0 or distance > 10:  # Invalid depth
                distance = 2.0  # Default distance
        else:
            distance = 2.0  # Default distance in meters

        # Calculate world position
        world_x = distance * np.tan(angle_x)
        world_y = distance * np.tan(angle_y)

        return (world_x, -world_y)  # Negative y because image y increases downward
