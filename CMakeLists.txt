cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(examples)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

add_executable(example-mnist example-mnist.cpp)
target_link_libraries(example-mnist "${TORCH_LIBRARIES}")
set_property(TARGET example-mnist PROPERTY CXX_STANDARD 17)

add_executable(example-autograd example-autograd.cpp)
target_link_libraries(example-autograd "${TORCH_LIBRARIES}")
set_property(TARGET example-autograd PROPERTY CXX_STANDARD 17)

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(example-custom-dataset "example-custom-dataset.cpp")
target_link_libraries(example-custom-dataset "${OpenCV_LIBS}")
target_link_libraries(example-custom-dataset "${TORCH_LIBRARIES}")
set_property(TARGET example-custom-dataset PROPERTY CXX_STANDARD 17)

configure_file("info.txt" "info.txt" COPYONLY)

add_executable(example-dcgan example-dcgan.cpp)
target_link_libraries(example-dcgan "${TORCH_LIBRARIES}")
set_property(TARGET example-dcgan PROPERTY CXX_STANDARD 17)
