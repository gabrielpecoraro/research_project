"""
JetBot Navigation System

A complete navigation system for JetBot that includes:
- Target detection and tracking
- Obstacle avoidance
- A* pathfinding
- Lane following
- Real-time environment mapping
"""

from .environment import RealWorldEnvironment
from .pathfinding import AStarPathfinder
from .target_detection import TargetDetector
from .obstacle_avoidance import ObstacleDetector
from .lane_following import LaneFollower
from .jetbot_controller import JetBotController
from .main_navigation import NavigationSystem

__version__ = "1.0.0"
__all__ = [
    "RealWorldEnvironment",
    "AStarPathfinder",
    "TargetDetector",
    "ObstacleDetector",
    "LaneFollower",
    "JetBotController",
    "NavigationSystem",
]
