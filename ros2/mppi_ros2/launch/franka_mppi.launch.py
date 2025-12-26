from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # configs
    mujoco_config = os.path.join(
        get_package_share_directory("mujoco_ros2"),
        "config",
        "mujoco_ros2.yaml",
    )
    mppi_config = os.path.join(
        get_package_share_directory("mppi_ros2"),
        "config",
        "panda_mppi.yaml",
    )
    mppi_viz_config = os.path.join(
        get_package_share_directory("mppi_ros2"),
        "config",
        "mppi_viz.yaml"
    )

    # RViz config path
    rviz_config = os.path.join(
        get_package_share_directory("franka_description"),
        "rviz",
        "panda.rviz",
    )

    # ---- URDF path (reuse the same package:// path you already use) ----
    # franka_description share dir
    franka_share = get_package_share_directory("franka_description")
    urdf_path = os.path.join(franka_share, "robots", "panda", "panda.urdf")

    # robot_description: read URDF file contents
    robot_description = Command(["cat ", urdf_path])

    # 1) MuJoCo bridge
    robot_node = Node(
        package="mujoco_ros2",
        executable="sim_node",
        name="mujoco_ros2_bridge",
        output="screen",
        parameters=[mujoco_config],
    )

    # 2) MPPI node
    mppi_node = Node(
        package="mppi_ros2",
        executable="mppi_node",
        name="mppi_ros2_node",
        output="screen",
        parameters=[mppi_config],
    )

    viz_node = Node(
        package="mppi_ros2",
        executable="mppi_viz_node",
        name="mppi_viz_node",
        output="screen",
        parameters=[mppi_viz_config],
    )

    # 3) robot_state_publisher (TF from joint states + robot_description)
    # MuJoCo publishes: /robot/joint_states  -> RSP expects: /joint_states
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
        remappings=[
            ("/joint_states", "/robot/joint_states"),
        ],
    )

    # 4) RViz2 (no config = empty RViz; you can save a .rviz later)
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d",rviz_config],
    )

    return LaunchDescription([
        robot_node,
        mppi_node,
        viz_node,
        rsp_node,
        rviz_node,
    ])
