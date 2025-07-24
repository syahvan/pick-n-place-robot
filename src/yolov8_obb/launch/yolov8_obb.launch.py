# This launch file is used to start the yolov8_obb_publisher node.
# It uses ROS2 launch system to define and start the node.

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generates the launch description for the yolov8_obb_publisher node.

    Returns:
        LaunchDescription: Contains the node to be launched.
    """
    return LaunchDescription(
        [
            Node(
                package="yolov8_obb",  # ROS2 package name
                executable="yolov8_obb_publisher.py",  # Python script to execute
                output="screen",  # Output logs to screen
            ),
        ]
    )
