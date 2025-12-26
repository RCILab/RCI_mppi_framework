# MPPI ROS2 + MuJoCo Integration

This directory contains the ROS2 packages and integration layer that sit on top of the core `mppi_framework` Python baseline.

It provides a real-time MPPI controller for the Franka Panda arm running in MuJoCo, with trajectory visualization in RViz.
This layer corresponds to the ROS2 / robotic side of the baseline proposed in:

**“Unifying Model Predictive Path Integral Control:  
From Stochastic Theory to Real-Time Implementation”**



## Repository Layout

mppi_framework/  Core MPPI Python baseline  
├── core/         Core components (registry)  
├── defaults/     Built-in sampling, dynamics, costs, configs  
├── interfaces/   Common abstract interfaces (sampling, dynamics, costs, etc.)  
├── examples/     Python simulation & plotting demos (Matplotlib)  
└── utils/        Helpers and utilities  

ros2/  ROS2 packages and integration layer  
├── mppi_ros2/          MPPI ROS2 controller node  
├── mujoco_ros2/        MuJoCo <-> ROS2 simulation bridge  
└── robot_description/  URDF, meshes, and robot model resources  


## ROS2 Demo — Franka Manipulator (MPPI + MuJoCo + RViz)

The ROS2 layer provides real-time MPPI control running on MuJoCo,
while RViz visualizes:

- sampled trajectories (blue)
- optimal trajectory (red)
- end-effector motion

<p align="center">
  <img src="../mppi_framework/mppi_framework/outputs/manipulator_ros2.gif" width="700">
</p>

*This demo is generated using the baseline configuration included in this repository.*



## 1. Prerequisites

- ROS2 Humble (Ubuntu 22.04)
- System Python (same Python used by ROS2)
- PyTorch (GPU optional but recommended)
- MuJoCo
- Pinocchio
- RViz2

For PyTorch, use the official installer and select the correct CUDA version:

https://pytorch.org/get-started/locally/

We recommend installing into **system / user site-packages** (not conda), to avoid ROS2 conflicts.



## 2. Create Workspace & Clone

```bash
mkdir -p ~/mppi_framework_ws/src
cd ~/mppi_framework_ws/src
```
### Clone this repo (name is arbitrary)
```bash
git clone <repository_url>
```
After clone you should have:
```bash
 ~/mppi_framework_ws/src/<repo_name>/
```


## 3. Install Python Dependencies (system python)

### 3.1 Install MPPI framework

```bash
cd <repo_name>
pip install -e ./mppi_framework
```

### 3.2 Install PyTorch

(Choose the command from the PyTorch website — depends on your CUDA)

### 3.3 MuJoCo + Pinocchio

```bash
pip3 install mujoco
pip3 install pinocchio
```


## 4. Build ROS2 packages
```bash
cd ~/mppi_framework_ws
colcon build --symlink-install
source install/setup.bash
```


## 5. Running the Demo

### Single launch

Only one command is needed — this launches:

- MuJoCo simulation bridge
- MPPI control node
- RViz visualization
- robot description

```bash
ros2 launch mppi_ros2 franka_mppi.launch.py
```
In RViz you can see:

- sampled trajectories (blue)
- optimal trajectory (red)
- end-effector motion in real time
- robot model
- tf



## 6. Optional: Periodic Goal Publisher

If you want the goal to change automatically:

```bash
ros2 run mppi_ros2 goal_publisher.py
```

This publishes `/mppi/goal` periodically with predefined waypoints.



## 7. Config Files

All configs are editable:

```bash
ros2/mppi_ros2/config/
ros2/mujoco_ros2/config/
```
You can modify:

- MPPI horizon, sample, noise
- cost
- visualization stride
- MuJoCo timestep / viewer on/off
- robot frames & topics



## 8. Extending

This ROS2 layer follows the same MPPI baseline architecture.

You can extend it to:

- quadrupeds
- humanoids
- mobile manipulators
- new cost functions
- alternative sampling strategies
- contact-aware models
- various sensors
