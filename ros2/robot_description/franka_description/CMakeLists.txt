cmake_minimum_required(VERSION 3.5)
project(franka_description)

# Required packages
find_package(ament_cmake REQUIRED)
find_package(xacro REQUIRED)

# Install targets: meshes and robots directories
install(DIRECTORY
  meshes
  robots
  rviz
  DESTINATION share/${PROJECT_NAME}
)

# Tests (optional)
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()

  # You can add pytest or launch tests here
endif()

ament_package()
