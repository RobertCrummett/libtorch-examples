cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(examples)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

find_package(MPI REQUIRED)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# MNIST example
add_executable(example-mnist example-mnist.cpp)
target_link_libraries(example-mnist "${TORCH_LIBRARIES}")
set_property(TARGET example-mnist PROPERTY CXX_STANDARD 17)

# Autograd example
add_executable(example-autograd example-autograd.cpp)
target_link_libraries(example-autograd "${TORCH_LIBRARIES}")
set_property(TARGET example-autograd PROPERTY CXX_STANDARD 17)

# Training on custom dataset - loading the dataset with OpenCV
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(example-custom-dataset "example-custom-dataset.cpp")
target_link_libraries(example-custom-dataset "${OpenCV_LIBS}")
target_link_libraries(example-custom-dataset "${TORCH_LIBRARIES}")
set_property(TARGET example-custom-dataset PROPERTY CXX_STANDARD 17)

configure_file("info.txt" "info.txt" COPYONLY)

# DCGAN example
add_executable(example-dcgan example-dcgan.cpp)
target_link_libraries(example-dcgan "${TORCH_LIBRARIES}")
set_property(TARGET example-dcgan PROPERTY CXX_STANDARD 17)

# # Distributed MNIST example
# include_directories(SYSTEM ${MPI_C_INCLUDE_PATH} ${MPI_CXX_INCLUDE_PATH})
# include_directories(SYSTEM "~/Desktop/libtorch/include/torch/csrc/api/include")
# # /Users/robertcrummett/Desktop/libtorch/include/torch/csrc/api
# add_executable(example-distributed-mnist example-distributed-mnist.cpp)
# target_link_libraries(example-distributed-mnist ${TORCH_LIBRARIES})
# target_link_libraries(example-distributed-mnist ${MPI_LIBRARIES})
# target_link_libraries(example-distributed-mnist "${CMAKE_PREFIX_PATH}/lib/libc10.dylib")
# set_property(TARGET example-distributed-mnist PROPERTY CXX_STANDARD 17)
# 
# if (MPI_COMPILE_FLAGS)
# 	set_target_properties(example-distributed-mnist PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
# endif (MPI_COMPILE_FLAGS)
# 
# if (MPI_LINK_FLAGS)
# 	set_target_properties(example-distributed-mnist PROPERTIES LINK_FLAGS "${MPI_COMPILE_FLAGS}")
# endif (MPI_LINK_FLAGS)

# Regression example
add_executable(example-regression example-regression.cpp)
target_link_libraries(example-regression "${TORCH_LIBRARIES}")
set_property(TARGET example-regression PROPERTY CXX_STANDARD 17)
