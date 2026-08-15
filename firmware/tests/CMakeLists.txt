cmake_minimum_required(VERSION 3.10)
project(firmware_tests C)

include(FetchContent)
FetchContent_Declare(
  unity
  GIT_REPOSITORY https://github.com/ThrowTheSwitch/Unity.git
  GIT_TAG        master
)
FetchContent_MakeAvailable(unity)

add_executable(run_host_tests test_main_host.c)
target_link_libraries(run_host_tests unity m)
