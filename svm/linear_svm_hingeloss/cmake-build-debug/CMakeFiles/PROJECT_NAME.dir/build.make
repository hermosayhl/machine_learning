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
CMAKE_SOURCE_DIR = D:\work\crane\machine_learning\svm\linear_svm_hingeloss

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = D:\work\crane\machine_learning\svm\linear_svm_hingeloss\cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/PROJECT_NAME.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/PROJECT_NAME.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PROJECT_NAME.dir/flags.make

CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.obj: CMakeFiles/PROJECT_NAME.dir/flags.make
CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.obj: CMakeFiles/PROJECT_NAME.dir/includes_CXX.rsp
CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.obj: ../src/svm_by_hingeloss.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=D:\work\crane\machine_learning\svm\linear_svm_hingeloss\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.obj"
	D:\environments\C++\TDM-GCC\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles\PROJECT_NAME.dir\src\svm_by_hingeloss.cpp.obj -c D:\work\crane\machine_learning\svm\linear_svm_hingeloss\src\svm_by_hingeloss.cpp

CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.i"
	D:\environments\C++\TDM-GCC\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E D:\work\crane\machine_learning\svm\linear_svm_hingeloss\src\svm_by_hingeloss.cpp > CMakeFiles\PROJECT_NAME.dir\src\svm_by_hingeloss.cpp.i

CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.s"
	D:\environments\C++\TDM-GCC\bin\g++.exe $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S D:\work\crane\machine_learning\svm\linear_svm_hingeloss\src\svm_by_hingeloss.cpp -o CMakeFiles\PROJECT_NAME.dir\src\svm_by_hingeloss.cpp.s

# Object files for target PROJECT_NAME
PROJECT_NAME_OBJECTS = \
"CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.obj"

# External object files for target PROJECT_NAME
PROJECT_NAME_EXTERNAL_OBJECTS =

../bin/PROJECT_NAME.exe: CMakeFiles/PROJECT_NAME.dir/src/svm_by_hingeloss.cpp.obj
../bin/PROJECT_NAME.exe: CMakeFiles/PROJECT_NAME.dir/build.make
../bin/PROJECT_NAME.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_highgui452.dll.a
../bin/PROJECT_NAME.exe: D:/environments/Miniconda/libs/python37.lib
../bin/PROJECT_NAME.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_videoio452.dll.a
../bin/PROJECT_NAME.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_imgcodecs452.dll.a
../bin/PROJECT_NAME.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_imgproc452.dll.a
../bin/PROJECT_NAME.exe: D:/environments/C++/OpenCV/opencv-4.5.2/build_TDM-GCC/install/x64/mingw/lib/libopencv_core452.dll.a
../bin/PROJECT_NAME.exe: D:/environments/Miniconda/libs/python37.lib
../bin/PROJECT_NAME.exe: CMakeFiles/PROJECT_NAME.dir/linklibs.rsp
../bin/PROJECT_NAME.exe: CMakeFiles/PROJECT_NAME.dir/objects1.rsp
../bin/PROJECT_NAME.exe: CMakeFiles/PROJECT_NAME.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=D:\work\crane\machine_learning\svm\linear_svm_hingeloss\cmake-build-debug\CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable ..\bin\PROJECT_NAME.exe"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles\PROJECT_NAME.dir\link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PROJECT_NAME.dir/build: ../bin/PROJECT_NAME.exe

.PHONY : CMakeFiles/PROJECT_NAME.dir/build

CMakeFiles/PROJECT_NAME.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles\PROJECT_NAME.dir\cmake_clean.cmake
.PHONY : CMakeFiles/PROJECT_NAME.dir/clean

CMakeFiles/PROJECT_NAME.dir/depend:
	$(CMAKE_COMMAND) -E cmake_depends "MinGW Makefiles" D:\work\crane\machine_learning\svm\linear_svm_hingeloss D:\work\crane\machine_learning\svm\linear_svm_hingeloss D:\work\crane\machine_learning\svm\linear_svm_hingeloss\cmake-build-debug D:\work\crane\machine_learning\svm\linear_svm_hingeloss\cmake-build-debug D:\work\crane\machine_learning\svm\linear_svm_hingeloss\cmake-build-debug\CMakeFiles\PROJECT_NAME.dir\DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/PROJECT_NAME.dir/depend

