# Pick and Place Simulation ROS2

## Overview
This project simulates a pick-and-place robot using ROS2. It demonstrates basic robotic manipulation tasks in a simulated environment, including object detection, GUI-based selection, and robot arm control.


## Installation

```bash
git clone https://github.com/syahvan/pick-n-place-robot.git
cd pick-n-place-robot

# Install Python dependencies
pip install -r requirements.txt

# Install ROS2 dependencies
sudo apt install \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-gz-ros2-control \
  ros-jazzy-gripper-controllers \
  ros-jazzy-moveit-py

# Build custom messages and packages
colcon build --symlink-install
source install/setup.bash
```

## Usage

### 1. Launch the Robot Arm Controller
```bash
ros2 launch panda_moveit_config moveit_gazebo_obb.py
```

### 2. Launch YOLOv8 OBB Publisher Node
```bash
ros2 launch yolov8_obb yolov8_obb.launch.py
```

### 3. Run the Bolt Selector GUI
```bash
python3 UI/bolt_selector.py
```

## ROS2 Nodes

- **YOLOv8 OBB Publisher**: Subscribes to camera images, runs YOLOv8 OBB inference, publishes results and annotated images.
- **YOLOv8 OBB Subscriber**: Receives inference results, draws bounding boxes, and publishes annotated images.
- **Bolt Selector GUI**: Displays camera and inference results, allows user to select bolts, publishes target points for robot control.
- **Arm Controller**: Listens for target points, plans and executes pick-and-place motions using MoveIt.