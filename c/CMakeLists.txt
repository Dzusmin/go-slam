cmake_minimum_required(VERSION 2.8)
project( c-slam )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( c-slam main.cpp )
target_link_libraries( c-slam ${OpenCV_LIBS} )

