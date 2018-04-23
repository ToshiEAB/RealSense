cmake_minimum_required(VERSION 2.8)

project(RealSense)

### Windows ###
if(WIN32)
	# Modify the following two codes if using a different version of RealSense SDK
	set(REALSENSE_INCLUDE C:/librealsense-2.10.3/include)
	set(REALSENSE_LIB C:/librealsense-2.10.3/build/Release/realsense2.lib)
	
	# Look for OpenCV. Make sure that the root directory of OpenCV is set an environment path before using the following code
	find_package(OpenCV REQUIRED)

	# Combine and add the Include directories
	set(INCLUDE_DIR ${REALSENSE_INCLUDE} ${OpenCV_INCLUDE_DIRS})
	include_directories("${INCLUDE_DIR}")

	# Build the main source code
	add_executable(RealSense RealSense.cpp)

	# Combine and add the library directories
	set(LIB_DIR ${REALSENSE_LIB} ${OpenCV_LIBS})
	target_link_libraries(RealSense ${LIB_DIR})

	# Limit the configuration only to Debug and Release modes (Optional)
	set(CMAKE_CONFIGURATION_TYPES "Debug;Release" CACHE STRING "limited configs" FORCE)

	# Omit ZERO_CHECK
	set(CMAKE_SUPPRESS_REGENERATION true)
endif()


### Unix/Linux ###
if(UNIX AND NOT APPLE)
	# Modify the following two codes if using a different version of RealSense SDK
	set(REALSENSE_INCLUDE ~/Documents/librealsense-2.10.3/include)
	set(REALSENSE_LIB ~/Documents/librealsense-2.10.3/build/librealsense2.so)	

	# Look for OpenCV. Make sure that the root directory of OpenCV is set an environment path before using the following code
	find_package(OpenCV REQUIRED)

	# Combine and add the Include directories of RealSense SDK and OpenCV
	set(INCLUDE_DIR ${REALSENSE_INCLUDE} ${OpenCV_INCLUDE_DIRS})
	include_directories("${INCLUDE_DIR}")

	# Build the main source code
	add_executable(RealSense RealSense.cpp)

	# Combine and add the library directories
	set(LIB_DIR ${REALSENSE_LIB} ${OpenCV_LIBS})
	target_link_libraries(RealSense ${LIB_DIR})

	# Check the availability of C++ compilers
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	if(COMPILER_SUPPORTS_CXX11)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	elseif(COMPILER_SUPPORTS_CXX0X)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	endif()
endif()


### MacOS ###
if(APPLE)

endif()



