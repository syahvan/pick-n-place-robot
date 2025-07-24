#!/usr/bin/env python3

"""
This script defines a ROS2 node that subscribes to a camera image topic,
runs YOLOv8 OBB (Oriented Bounding Box) inference on the images,
and publishes the inference results and annotated images.
"""

import copy
import os

import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
from yolov8_msgs.msg import InferenceResult, Yolov8Inference

bridge = CvBridge()  # Used for converting ROS Image messages to OpenCV images


class Camera_subscriber(Node):
    """
    ROS2 Node that subscribes to camera images, runs YOLOv8 OBB inference,
    and publishes results.
    """

    def __init__(self):
        super().__init__("camera_subscriber")

        # Load YOLOv8 model from a specified path
        self.model = YOLO(
            os.environ["HOME"] + "/moveit2_obb/src/yolov8_obb/scripts/best.pt"
        )

        # Message to store inference results
        self.yolov8_inference = Yolov8Inference()

        # Subscribe to raw camera images
        self.subscription = self.create_subscription(
            Image, "/image_raw", self.camera_callback, 10
        )
        self.subscription

        # Publisher for inference results (custom message)
        self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
        # Publisher for annotated images
        self.img_pub = self.create_publisher(Image, "/inference_result", 1)

    def camera_callback(self, data):
        """
        Callback function for image subscription.
        Runs YOLOv8 inference and publishes results.
        Args:
            data (sensor_msgs.msg.Image): Incoming image message.
        """
        img = bridge.imgmsg_to_cv2(data, "bgr8")  # Convert ROS image to OpenCV format
        results = self.model(
            img, conf=0.90
        )  # Run YOLOv8 inference with confidence threshold

        # Set header info for inference message
        self.yolov8_inference.header.frame_id = "inference"
        self.yolov8_inference.header.stamp = (
            camera_subscriber.get_clock().now().to_msg()
        )

        # Iterate through results and extract OBBs
        for r in results:
            if r.obb is not None:
                boxes = r.obb
                for box in boxes:
                    self.inference_result = InferenceResult()
                    b = (
                        box.xyxyxyxy[0].to("cpu").detach().numpy().copy()
                    )  # Get OBB coordinates
                    c = box.cls  # Class index
                    self.inference_result.class_name = self.model.names[
                        int(c)
                    ]  # Class name
                    a = b.reshape(1, 8)
                    self.inference_result.coordinates = copy.copy(
                        a[0].tolist()
                    )  # List of 8 coordinates
                    self.yolov8_inference.yolov8_inference.append(self.inference_result)
            else:
                camera_subscriber.get_logger().info("no_results")

        # Publish inference results
        self.yolov8_pub.publish(self.yolov8_inference)
        self.yolov8_inference.yolov8_inference.clear()

        # Publish annotated image
        annotated_frame = results[0].plot()
        img_msg = bridge.cv2_to_imgmsg(annotated_frame)
        self.img_pub.publish(img_msg)


if __name__ == "__main__":
    # Initialize ROS2 and start the node
    rclpy.init(args=None)
    camera_subscriber = Camera_subscriber()
    rclpy.spin(camera_subscriber)
    rclpy.shutdown()
