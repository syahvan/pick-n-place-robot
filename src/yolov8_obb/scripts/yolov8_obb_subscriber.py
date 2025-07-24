#!/usr/bin/env python3

"""
This script defines two ROS2 nodes:
- Camera_subscriber: Subscribes to raw camera images and stores them.
- Yolo_subscriber: Subscribes to YOLOv8 inference results, draws bounding boxes on the image,
  and publishes the annotated image.
"""

import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()

img = np.zeros([480, 640, 3])  # Global image buffer


class Camera_subscriber(Node):
    """
    Subscribes to raw camera images and updates the global image buffer.
    """

    def __init__(self):
        super().__init__("camera_subscriber")

        self.subscription = self.create_subscription(
            Image, "image_raw", self.camera_callback, 10
        )
        self.subscription

    def camera_callback(self, data):
        """
        Callback for camera image subscription.
        Updates the global image buffer.
        """
        global img
        img = bridge.imgmsg_to_cv2(data, "bgr8")


class Yolo_subscriber(Node):
    """
    Subscribes to YOLOv8 inference results, draws bounding boxes on the image,
    and publishes the annotated image.
    """

    def __init__(self):
        super().__init__("yolo_subscriber")

        self.subscription = self.create_subscription(
            Yolov8Inference, "/Yolov8_Inference", self.yolo_callback, 10
        )
        self.subscription

        self.img_pub = self.create_publisher(Image, "/inference_result_cv2", 1)

    def yolo_callback(self, data):
        """
        Callback for YOLOv8 inference subscription.
        Draws bounding boxes and publishes the result.
        """
        global img
        for r in data.yolov8_inference:
            class_name = r.class_name
            points = np.array(r.coordinates).astype(np.int32).reshape([4, 2])
            # Draw polygon for OBB
            cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        img_msg = bridge.cv2_to_imgmsg(img)
        self.img_pub.publish(img_msg)


if __name__ == "__main__":
    # Initialize ROS2, create nodes, and run with multithreaded executor
    rclpy.init(args=None)
    yolo_subscriber = Yolo_subscriber()
    camera_subscriber = Camera_subscriber()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(yolo_subscriber)
    executor.add_node(camera_subscriber)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = yolo_subscriber.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()
