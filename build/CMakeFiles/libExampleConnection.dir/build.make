# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/colin-wsl/leap-motion/LeapSDK/samples

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/colin-wsl/leap-motion/build

# Include any dependencies generated for this target.
include CMakeFiles/libExampleConnection.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/libExampleConnection.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/libExampleConnection.dir/flags.make

CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o: CMakeFiles/libExampleConnection.dir/flags.make
CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o: /home/colin-wsl/leap-motion/LeapSDK/samples/ExampleConnection.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/colin-wsl/leap-motion/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o   -c /home/colin-wsl/leap-motion/LeapSDK/samples/ExampleConnection.c

CMakeFiles/libExampleConnection.dir/ExampleConnection.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/libExampleConnection.dir/ExampleConnection.c.i"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/colin-wsl/leap-motion/LeapSDK/samples/ExampleConnection.c > CMakeFiles/libExampleConnection.dir/ExampleConnection.c.i

CMakeFiles/libExampleConnection.dir/ExampleConnection.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/libExampleConnection.dir/ExampleConnection.c.s"
	/usr/bin/gcc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/colin-wsl/leap-motion/LeapSDK/samples/ExampleConnection.c -o CMakeFiles/libExampleConnection.dir/ExampleConnection.c.s

libExampleConnection: CMakeFiles/libExampleConnection.dir/ExampleConnection.c.o
libExampleConnection: CMakeFiles/libExampleConnection.dir/build.make

.PHONY : libExampleConnection

# Rule to build all files generated by this target.
CMakeFiles/libExampleConnection.dir/build: libExampleConnection

.PHONY : CMakeFiles/libExampleConnection.dir/build

CMakeFiles/libExampleConnection.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/libExampleConnection.dir/cmake_clean.cmake
.PHONY : CMakeFiles/libExampleConnection.dir/clean

CMakeFiles/libExampleConnection.dir/depend:
	cd /home/colin-wsl/leap-motion/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/colin-wsl/leap-motion/LeapSDK/samples /home/colin-wsl/leap-motion/LeapSDK/samples /home/colin-wsl/leap-motion/build /home/colin-wsl/leap-motion/build /home/colin-wsl/leap-motion/build/CMakeFiles/libExampleConnection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/libExampleConnection.dir/depend
