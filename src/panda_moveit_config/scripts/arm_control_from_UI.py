#!/usr/bin/env python3

"""
This script defines a ROS2 node that controls a Panda robot arm using MoveIt.
It listens for target points published by the GUI, plans and executes motions,
and controls the gripper for pick-and-place operations.
"""

import threading
import time

import rclpy

# set pose goal with PoseStamped message
from geometry_msgs.msg import PoseStamped
from moveit.core.kinematic_constraints import construct_joint_constraint

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
)
from rclpy.logging import get_logger
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def plan_and_execute(
    robot,
    planning_component,
    logger,
    single_plan_parameters=None,
    multi_plan_parameters=None,
    sleep_time=0.0,
):
    """
    Helper function to plan and execute a motion using MoveIt.
    Args:
        robot: MoveItPy instance.
        planning_component: Planning component for the robot.
        logger: Logger for output.
        single_plan_parameters: Optional single pipeline parameters.
        multi_plan_parameters: Optional multi pipeline parameters.
        sleep_time: Time to sleep after execution.
    """
    # plan to goal
    logger.info("Planning trajectory")
    if multi_plan_parameters is not None:
        plan_result = planning_component.plan(
            multi_plan_parameters=multi_plan_parameters
        )
    elif single_plan_parameters is not None:
        plan_result = planning_component.plan(
            single_plan_parameters=single_plan_parameters
        )
    else:
        plan_result = planning_component.plan()

    # execute the plan
    if plan_result:
        logger.info("Executing plan")
        robot_trajectory = plan_result.trajectory
        robot.execute(robot_trajectory, controllers=[])
    else:
        logger.error("Planning failed")

    time.sleep(sleep_time)


class Controller(Node):
    """
    ROS2 Node for controlling the Panda robot arm.
    Listens for target points and performs pick-and-place actions.
    """

    def __init__(self):
        super().__init__("commander")
        # Subscribe to target point topic from GUI
        self.subscription = self.create_subscription(
            Float64MultiArray, "/target_point", self.listener_callback, 10
        )
        self.subscription

        # Pose goal for robot arm
        self.pose_goal = PoseStamped()
        self.pose_goal.header.frame_id = "panda_link0"
        # Instantiate MoveItPy and get planning components
        self.panda = MoveItPy(node_name="moveit_py")
        self.panda_arm = self.panda.get_planning_component("panda_arm")
        self.panda_hand = self.panda.get_planning_component("hand")
        self.logger = get_logger("moveit_py.pose_goal")

        robot_model = self.panda.get_robot_model()
        self.robot_state = RobotState(robot_model)

        # Heights and initial angle for pick-and-place
        self.height = 0.18
        self.pick_height = 0.126
        self.carrying_height = 0.3
        self.init_angle = -0.3825

    def move_to(self, x, y, z, xo, yo, zo, wo):
        """
        Moves the robot arm to the specified pose.
        Args:
            x, y, z: Position coordinates.
            xo, yo, zo, wo: Orientation quaternion.
        """
        self.pose_goal.pose.position.x = x
        self.pose_goal.pose.position.y = y
        self.pose_goal.pose.position.z = z
        self.pose_goal.pose.orientation.x = xo
        self.pose_goal.pose.orientation.y = yo
        self.pose_goal.pose.orientation.z = zo
        self.pose_goal.pose.orientation.w = wo
        self.panda_arm.set_goal_state(
            pose_stamped_msg=self.pose_goal, pose_link="panda_link8"
        )
        plan_and_execute(self.panda, self.panda_arm, self.logger, sleep_time=3.0)

    def gripper_action(self, action):
        """
        Controls the gripper to open or close.
        Args:
            action (str): 'open' or 'close'
        """
        self.panda_hand.set_start_state_to_current_state()

        if action == "open":
            joint_values = {"panda_finger_joint1": 0.03}
        elif action == "close":
            joint_values = {"panda_finger_joint1": 0.001}
        else:
            self.get_logger().info("no such action")

        self.robot_state.joint_positions = joint_values
        joint_constraint = construct_joint_constraint(
            robot_state=self.robot_state,
            joint_model_group=self.panda.get_robot_model().get_joint_model_group(
                "hand"
            ),
        )
        self.panda_hand.set_goal_state(motion_plan_constraints=[joint_constraint])
        plan_and_execute(self.panda, self.panda_hand, self.logger, sleep_time=3.0)

    def listener_callback(self, data):
        """
        Callback for target point subscription.
        Executes pick-and-place sequence based on received target point.
        Args:
            data (Float64MultiArray): [x, y, angle]
        """
        self.get_logger().info(f"{data}")

        # Move above the target
        self.move_to(
            data.data[0],
            data.data[1],
            self.height,
            1.0,
            self.init_angle + data.data[2],
            0.0,
            0.0,
        )

        # Open gripper
        self.gripper_action("open")

        # Move down to pick
        self.move_to(
            data.data[0],
            data.data[1],
            self.pick_height,
            1.0,
            self.init_angle + data.data[2],
            0.0,
            0.0,
        )

        # Close gripper to grasp
        self.gripper_action("close")

        # Lift up
        self.move_to(
            data.data[0],
            data.data[1],
            self.carrying_height,
            1.0,
            self.init_angle + data.data[2],
            0.0,
            0.0,
        )

        # Move to drop location
        self.move_to(
            0.3,
            -0.3,
            self.carrying_height,
            1.0,
            self.init_angle + data.data[2],
            0.0,
            0.0,
        )

        # Open gripper to release
        self.gripper_action("open")


if __name__ == "__main__":
    # Initialize ROS2, create controller node, and run with multithreaded executor
    rclpy.init(args=None)

    controller = Controller()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(controller)

    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()

    rate = controller.create_rate(2)
    try:
        while rclpy.ok():
            rate.sleep()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
    executor_thread.join()
