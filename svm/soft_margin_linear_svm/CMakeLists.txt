cmake_minimum_required(VERSION 3.19)
set(PROJECT_NAME svm)
project(PROJECT_NAME)

# 添加编译选项
SET(CMAKE_CXX_FLAGS "-std=c++17 -O1")

# 生成文件的输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# 设置 Python 的路径, 这是必须的, 最好在 anaconda 的 base 环境中; 不然会出现 encoding 的运行错误
set(ENV{PYTHONHOME} "D:/environments/Miniconda")
set(ENV{PYTHONPATH} "D:/environments/Miniconda")

# include 我自己的目录
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 加入 matplotlib.h 的路径
include_directories(D:/environments/C++/3rdparty/matplotlib)

# 添加 Python3 和 numpy
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)


# 添加可执行文件
add_executable(
    PROJECT_NAME
    src/svm.cpp
)

# 加入 python 的动态库路径
target_include_directories(PROJECT_NAME PRIVATE ${Python3_INCLUDE_DIRS})

# 设置 OpenCV
set(OpenCV_DIR "D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/")
# 寻找 opencv 的库
find_package(OpenCV COMPONENTS core highgui imgproc imgcodecs)
# 添加 opencv 的头文件
include_directories(${OpenCV_INCLUDE_DIRS} )
# 链接 opencv 动态库路径
link_directories(${OpenCV_LIBRARY_DIRS})

## 设置库文件
target_link_libraries(PROJECT_NAME ${OpenCV_LIBS} ${Python3_LIBRARIES} Python3::NumPy)