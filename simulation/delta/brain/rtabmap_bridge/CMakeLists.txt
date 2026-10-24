# The bridge between the brain (Python) and RTAB-Map (C++).
# Built by setup/setup_rtabmap.sh, inside the `rtabmap` conda env.

cmake_minimum_required(VERSION 3.16)
project(rtabmap_bridge CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(RTABMap REQUIRED COMPONENTS core)

add_executable(rtabmap_bridge bridge.cpp)
target_link_libraries(rtabmap_bridge rtabmap::core)

# Run straight from the build tree, finding the env's libraries without
# anyone having to activate it.
set_target_properties(rtabmap_bridge PROPERTIES
    BUILD_RPATH "$ENV{CONDA_PREFIX}/lib"
)
