cmake_minimum_required(VERSION 3.14)

project(ReclaimedGridShells)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include FetchContent module
include(FetchContent)

# Fetch vcglib from GitHub
FetchContent_Declare(
    vcglib
    GIT_REPOSITORY https://github.com/cnr-isti-vclab/vcglib.git
    GIT_TAG main
)
FetchContent_MakeAvailable(vcglib)

set(VCGDIR ${vcglib_SOURCE_DIR})

# Add executable for draw_shell
add_executable(draw_shell draw_shell.cpp ${vcglib_SOURCE_DIR}/wrap/ply/plylib.cpp)
target_include_directories(draw_shell PRIVATE ${vcglib_SOURCE_DIR})
target_include_directories(draw_shell PRIVATE ${vcglib_SOURCE_DIR}/eigenlib)
target_link_libraries(draw_shell PRIVATE ${vcglib_SOURCE_DIR})

# Add executable for draw_color_shell
add_executable(draw_color_shell draw_color_shell.cpp ${vcglib_SOURCE_DIR}/wrap/ply/plylib.cpp)
target_include_directories(draw_color_shell PRIVATE ${vcglib_SOURCE_DIR})
target_include_directories(draw_color_shell PRIVATE ${vcglib_SOURCE_DIR}/eigenlib)
target_link_libraries(draw_color_shell PRIVATE ${vcglib_SOURCE_DIR})

# Copy executables next to the cpp files after build
add_custom_command(TARGET draw_shell POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:draw_shell> ${CMAKE_SOURCE_DIR}
)

add_custom_command(TARGET draw_color_shell POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:draw_color_shell> ${CMAKE_SOURCE_DIR}
)
