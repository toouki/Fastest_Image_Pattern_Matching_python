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
    
    target_link_libraries(${project_name}_test PRIVATE ${project_name} opencv_core opencv_imgproc opencv_highgui)
endif()

# Find OpenCV
find_package(OpenCV CONFIG REQUIRED)

# Link OpenCV libraries
target_link_libraries(${project_name} PRIVATE opencv_core opencv_imgproc opencv_highgui)

