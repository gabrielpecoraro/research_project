#!/usr/bin/env python3
"""
Basic navigation example
Demonstrates target detection and navigation to target
"""

import sys
import time

sys.path.append("../src")

from main_navigation import NavigationSystem


def main():
    print("Basic Navigation Example")
    print("========================")

    # Create navigation system looking for a red object
    nav_system = NavigationSystem("red object")

    print("Starting navigation in 3 seconds...")
    time.sleep(3)

    try:
        # Start autonomous navigation
        nav_system.start_navigation()

    except KeyboardInterrupt:
        print("\nStopping navigation...")
    finally:
        nav_system.stop_navigation()


if __name__ == "__main__":
    main()
