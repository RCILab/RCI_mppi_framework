# MPPI Unified Control Framework

This repository provides the baseline implementation accompanying the paper:

**“Unifying Model Predictive Path Integral Control:
From Stochastic Theory to Real-Time Implementation”**


The full paper is available here: <br>
👉 [Download the PDF](paper/Unifying_Model_Predictive_Path_Integral_Control.pdf)

The framework is designed as a unified experimental platform for developing,
testing, and extending MPPI-based controllers — from simple simulations to
full ROS2 robotic systems.

Built around a modular architecture, the framework separates MPPI into
independent components:

- sampling
- dynamics
- cost definitions
- rollout & visualization

Each module can be replaced or extended without modifying the others, enabling:

- rapid prototyping
- fair benchmarking
- real-time experiments
- research reproducibility

## Repository Structure

mppi_framework/  Core MPPI Python baseline <br>
├── core/         Core components (registry) <br>
├── defaults/     Built-in sampling, dynamics, costs, configs <br>
├── interfaces/   Common abstract interfaces (sampling, dynamics, costs, etc.) <br>
├── examples/     Python simulation & plotting demos (Matplotlib) <br>
└── utils/        Helpers and utilities <br>


ros2/  ROS2 packages and integration layer <br>
├── mppi_ros2/          MPPI ROS2 controller node <br>
├── mujoco_ros2/        MuJoCo <-> ROS2 simulation bridge <br>
└──  robot_description/   URDF, meshes, and robot model resources <br>



## MPPI Baseline (Core Framework)

The `mppi_framework/` directory contains the research baseline implementation.

Features include:

- GPU-accelerated rollout engine
- modular cost & sampling interfaces
- ready-to-use dynamics models
- visualization utilities
- clean API for new MPPI variants


## Environment Setup

There are two typical workflows depending on how you plan to use the framework.


### 1) Python-only simulations (Matplotlib examples)

Use a virtual environment (recommended).

This mode is intended for:

- algorithm experiments
- visualization
- debugging and benchmarking

Full installation steps:

➡ See [mppi_framework/README.md](mppi_framework/)


### 2) ROS2 + MuJoCo + Real-time MPPI

Use a ROS2 workspace (system-level install recommended).

This mode is intended for:

- real robots
- simulation pipelines
- integration with ROS2 tools

Full installation and launch instructions:

➡ See [ros2/README.md](ros2/)


## Examples

Python simulations live under:

mppi_framework/examples/


They provide simple MPPI demos using Matplotlib, showing:

- trajectory rollout
- controller behavior

## Demo
MPPI rollouts generated using the framework:

### Cartpole
![cartpole](mppi_framework/mppi_framework/outputs/cartpole.gif)

### Mobile 2D
![mobile2d](mppi_framework/mppi_framework/outputs/mobile2d.gif)

### Quadrotor 3D
![quad3d](mppi_framework/mppi_framework/outputs/quad3d.gif)

### Manipulator (ROS2)
![manipulator](mppi_framework/mppi_framework/outputs/manipulator_ros2.gif)


## Citation
comming soon


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Maintainer: [Leesai Park] (leesai2000@khu.ac.kr)<br>
Lab: [RCI Lab @ Kyung Hee University](https://rcilab.khu.ac.kr)