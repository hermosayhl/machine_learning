# CMAKE generated file: DO NOT EDIT!
# Generated by "MinGW Makefiles" Generator, CMake Version 3.19

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

SHELL = cmd.exe

# The CMake executable.
CMAKE_COMMAND = D:\software\editor\CLion\bin\cmake\win\bin\cmake.exe

# The command to remove a file.
RM = D:\software\editor\CLion\bin\cmake\win\bin\cmake.exe -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = D:\work\crane\machine_learning\naive_bayers

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\work\crane\machine_learning\naive_bayers\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/naive_bayers.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/naive_bayers.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/naive_bayers.dir/flags.make

CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.obj: CMakeFiles/naive_bayers.dir/flags.make
CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.obj: CMakeFiles/naive_bayers.dir/includes_CXX.rsp
CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.obj: ../src/naive_bayers_continuous.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\work\crane\machine_learning\naive_bayers\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.obj"
	D:\environments\C++\TDM-GCC\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\naive_bayers.dir\src\naive_bayers_continuous.cpp.obj -c D:\work\crane\machine_learning\naive_bayers\src\naive_bayers_continuous.cpp

CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.i"
	D:\environments\C++\TDM-GCC\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\work\crane\machine_learning\naive_bayers\src\naive_bayers_continuous.cpp > CMakeFiles\naive_bayers.dir\src\naive_bayers_continuous.cpp.i

CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.s"
	D:\environments\C++\TDM-GCC\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\work\crane\machine_learning\naive_bayers\src\naive_bayers_continuous.cpp -o CMakeFiles\naive_bayers.dir\src\naive_bayers_continuous.cpp.s

# Object files for target naive_bayers
naive_bayers_OBJECTS = \
"CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.obj"

# External object files for target naive_bayers
naive_bayers_EXTERNAL_OBJECTS =

bin/naive_bayers.exe: CMakeFiles/naive_bayers.dir/src/naive_bayers_continuous.cpp.obj
bin/naive_bayers.exe: CMakeFiles/naive_bayers.dir/build.make
bin/naive_bayers.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_highgui452.dll.a
bin/naive_bayers.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_videoio452.dll.a
bin/naive_bayers.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_imgcodecs452.dll.a
bin/naive_bayers.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_imgproc452.dll.a
bin/naive_bayers.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_core452.dll.a
bin/naive_bayers.exe: CMakeFiles/naive_bayers.dir/linklibs.rsp
bin/naive_bayers.exe: CMakeFiles/naive_bayers.dir/objects1.rsp
bin/naive_bayers.exe: CMakeFiles/naive_bayers.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\work\crane\machine_learning\naive_bayers\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable bin\naive_bayers.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\naive_bayers.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/naive_bayers.dir/build: bin/naive_bayers.exe

.PHONY : CMakeFiles/naive_bayers.dir/build

CMakeFiles/naive_bayers.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\naive_bayers.dir\cmake_clean.cmake
.PHONY : CMakeFiles/naive_bayers.dir/clean

CMakeFiles/naive_bayers.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\work\crane\machine_learning\naive_bayers D:\work\crane\machine_learning\naive_bayers D:\work\crane\machine_learning\naive_bayers\cmake-build-debug D:\work\crane\machine_learning\naive_bayers\cmake-build-debug D:\work\crane\machine_learning\naive_bayers\cmake-build-debug\CMakeFiles\naive_bayers.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/naive_bayers.dir/depend

