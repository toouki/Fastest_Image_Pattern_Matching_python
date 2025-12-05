cmake_minimum_required(VERSION 3.11)

set(project_name "MatchTool")

project(${project_name})

set(CMAKE_CXX_STANDARD 17)

# Platform-specific configuration
if(WIN32)
    # Windows MFC configuration
    add_definitions(-D_AFXDLL)
    set(CMAKE_MFC_FLAG 2)
    add_definitions(-DUNICODE -D_UNICODE)
    
    # Create Windows executable with MFC
    add_executable(${project_name} WIN32
                                   MatchTool/framework.h 
                                   MatchTool/MatchTool.h
                                   MatchTool/MatchToolDlg.h
                                   MatchTool/pch.h
                                   MatchTool/resource.h
                                   MatchTool/targetver.h
                                   MatchTool/MatchTool.cpp
                                   MatchTool/MatchToolDlg.cpp
                                   MatchTool/pch.cpp
                                   MatchTool/MatchTool.rc
                                   MatchTool/MatchTool.Lang
                                   MatchTool/res/MatchTool.ico
                                   MatchTool/res/MatchTool.rc2
    )
    
    # Copy MatchTool.Lang to the output directory
    add_custom_command(TARGET ${project_name} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/MatchTool/MatchTool.Lang
            $<TARGET_FILE_DIR:${project_name}>
    )
else()
    # Linux/Unix configuration - build core library
    add_definitions(-DOPENCV_4X)
    
    # Create a shared library for Linux
    add_library(${project_name} SHARED
                                   MatchTool/x86_ubuntu_linux.cpp
    )
    
    # Create a simple test executable
    add_executable(${project_name}_test
        MatchTool/test_main.cpp
    )
    
    # Link OpenCV for test executable will be handled below
endif()

# Find OpenCV - with helpful error messages
find_package(OpenCV QUIET)

# Set a unified variable for OpenCV availability
set(OPENCV_AVAILABLE FALSE)
set(OPENCV_LIBS_TO_USE "")
set(OPENCV_INCS_TO_USE "")

if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    set(OPENCV_AVAILABLE TRUE)
    set(OPENCV_LIBS_TO_USE ${OpenCV_LIBRARIES})
    set(OPENCV_INCS_TO_USE ${OpenCV_INCLUDE_DIRS})
else()
    message(STATUS "OpenCV not found in standard locations")
    message(STATUS "Trying alternative search methods...")
    
    # Try alternative find methods
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENCV QUIET opencv4)
        if(NOT OPENCV_FOUND)
            pkg_check_modules(OPENCV QUIET opencv)
        endif()
        
        if(OPENCV_FOUND)
            message(STATUS "Found OpenCV via pkg-config: ${OPENCV_VERSION}")
            set(OPENCV_AVAILABLE TRUE)
            set(OPENCV_LIBS_TO_USE ${OPENCV_LIBRARIES})
            set(OPENCV_INCS_TO_USE ${OPENCV_INCLUDE_DIRS})
        endif()
    endif()
endif()

# If still not found, provide helpful installation instructions
if(NOT OPENCV_AVAILABLE)
    message(FATAL_ERROR "\nOpenCV not found! Please install OpenCV first:\n"
        "  Linux:   ./install_dependencies.sh\n"
        "           or: sudo apt install libopencv-dev\n"
        "  macOS:    brew install opencv\n"
        "  Windows:  vcpkg install opencv4[contrib]\n"
        "\nOr run: ./install_dependencies.sh\n")
endif()

# Link OpenCV libraries
if(WIN32)
    target_link_libraries(${project_name} PRIVATE opencv_core opencv_imgproc opencv_highgui)
else()
    target_link_libraries(${project_name} PRIVATE ${OPENCV_LIBS_TO_USE})
    target_include_directories(${project_name} PRIVATE ${OPENCV_INCS_TO_USE})
    
    # Also link for test executable
    target_link_libraries(${project_name}_test PRIVATE ${project_name} ${OPENCV_LIBS_TO_USE})
    target_include_directories(${project_name}_test PRIVATE ${OPENCV_INCS_TO_USE})
endif()

