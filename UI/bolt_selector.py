#! /usr/bin/python3
# -*- coding: utf-8 -*-

"""
This script defines a PyQt5 GUI for selecting bolts using YOLOv8 OBB inference results.
It subscribes to camera and inference topics, displays results, and publishes target points
for robot control.
"""

import math
import sys

import cv2
import numpy as np

# ROS imports
import rclpy
from bolt_selector_window import Ui_Form
from cv_bridge import CvBridge
from PyQt5.Qt import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from yolov8_msgs.msg import Yolov8Inference

bridge = CvBridge()

img = np.zeros([480, 640, 3])  # Global image buffer


class GraphicsScene(QGraphicsScene):
    """
    Custom QGraphicsScene to handle mouse events for selecting bolts.
    Stores mouse position and click status.
    """

    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.mouse_x = 0
        self.mouse_y = 0
        self.click_mouse = False

    def mouseMoveEvent(self, event):
        # Update mouse position
        self.mouse_x = event.scenePos().x()
        self.mouse_y = event.scenePos().y()

    def mousePressEvent(self, event):
        # Set click flag when mouse is pressed
        self.click_mouse = True


class GUI(QDialog):
    """
    Main GUI class for the bolt selector.
    Handles ROS subscriptions, image display, and user interaction.
    """

    def __init__(self, parent=None):
        super(GUI, self).__init__(parent)
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        # Set up graphics scene for displaying images and overlays
        self.scene = GraphicsScene(self.ui.graphicsView)
        self.ui.graphicsView.setScene(self.scene)
        self.ui.graphicsView.setMouseTracking(True)

        # ROS2 initialization and subscriptions
        rclpy.init(args=None)
        self.camera_subscriber = Node("image_subscriber")
        self.sub = self.camera_subscriber.create_subscription(
            Image, "/image_raw", self.camera_callback, 10
        )

        self.yolo_subscriber = Node("yolo_subscriber")
        self.sub = self.yolo_subscriber.create_subscription(
            Yolov8Inference, "/Yolov8_Inference", self.yolo_callback, 10
        )

        # Publisher for target points (for robot control)
        self.pub_node = Node("pub_path")
        self.pub = self.pub_node.create_publisher(
            Float64MultiArray, "/target_point", 10
        )

        # Timer for updating the GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(10)

        self.brush = QBrush(QColor(255, 255, 255, 255))
        self.target_point = [0, 0, 0]  # Target point to be published
        # Camera intrinsic parameters and offsets
        self.fx = 253.93635749816895
        self.fy = 253.93635749816895
        self.cx = 320
        self.cy = 240
        self.z = 0.7
        self.init_x = 0.2  # Camera position offset (robot arm link 0 initial position)
        self.init_y = 0.6

    def camera_callback(self, data):
        """
        Callback for camera image subscription.
        Updates the global image buffer.
        """
        global img
        img = bridge.imgmsg_to_cv2(data, "bgr8")

    def yolo_callback(self, data):
        """
        Callback for YOLOv8 inference subscription.
        Draws polygons for detected bolts, highlights selected bolt,
        and publishes target point if selected.
        """
        global img

        self.scene.clear()
        # Convert image to RGB for Qt display
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_image.shape
        q_image = QImage(rgb_image.data, width, height, 3 * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(pixmap_item)

        for r in data.yolov8_inference:
            # Get OBB points and calculate middle point
            points = np.array(r.coordinates).astype(np.int32).reshape([4, 2])
            middle_point = np.sum(points, 0) / 4
            dist = math.sqrt(
                (self.scene.mouse_x - middle_point[0]) ** 2
                + (self.scene.mouse_y - middle_point[1]) ** 2
            )
            qpoly = QPolygonF([QPointF(p[0], p[1]) for p in points])

            if dist < 15:
                # Highlight selected bolt
                self.scene.addPolygon(
                    qpoly, QPen(QColor(255, 0, 0, 255)), QBrush(QColor(255, 0, 0, 100))
                )

                if self.scene.click_mouse:
                    # Calculate target point in robot coordinates
                    self.target_point[0] = (
                        -self.z * (middle_point[1] - self.cy) / self.fy + self.init_x
                    )
                    self.target_point[1] = (
                        -self.z * (middle_point[0] - self.cx) / self.fx + self.init_y
                    )
                    dist1 = math.sqrt(
                        (points[0][0] - points[1][0]) ** 2
                        + (points[0][1] - points[1][1]) ** 2
                    )
                    dist2 = math.sqrt(
                        (points[1][0] - points[2][0]) ** 2
                        + (points[1][1] - points[2][1]) ** 2
                    )

                    # Calculate orientation angle
                    if dist1 > dist2:
                        denominator = points[0][0] - points[1][0]
                        if denominator == 0:
                            angle = math.pi / 2
                        else:
                            angle = math.atan2(points[0][1] - points[1][1], denominator)
                    else:
                        denominator = points[1][0] - points[2][0]
                        if denominator == 0:
                            angle = math.pi / 2
                        else:
                            angle = math.atan2(points[1][1] - points[2][1], denominator)

                    self.target_point[2] = math.pi / 2 - angle
                    # Publish target point for robot control
                    target_point_pub = Float64MultiArray(data=self.target_point)
                    self.pub.publish(target_point_pub)
                    self.scene.click_mouse = False
            else:
                # Draw non-selected bolts in blue
                self.scene.addPolygon(
                    qpoly, QPen(QColor(0, 0, 255, 255)), QBrush(QColor(0, 0, 255, 100))
                )

            # Draw middle point of the bolt
            self.scene.addEllipse(
                middle_point[0] - 2,
                middle_point[1] - 2,
                4,
                4,
                QPen(Qt.green),
                QBrush(Qt.green),
            )

    def update(self):
        """
        Periodically called by the timer to process ROS messages.
        """
        rclpy.spin_once(self.camera_subscriber)
        rclpy.spin_once(self.yolo_subscriber)


if __name__ == "__main__":
    # Start the Qt application and show the GUI
    app = QApplication(sys.argv)
    window = GUI()
    window.show()
    sys.exit(app.exec_())
