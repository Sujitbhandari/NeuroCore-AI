cmake_minimum_required(VERSION 3.12)
project(NeuroCore VERSION 1.0.0 LANGUAGES CXX)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable OpenMP for parallel computing
# On macOS, OpenMP might be installed via Homebrew
if(APPLE)
    # Try to find OpenMP in Homebrew's location
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -lomp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
endif()

find_package(OpenMP REQUIRED)

# Compiler flags for optimization
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra")

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/main.cpp
    src/Matrix.cpp
    src/Network.cpp
)

# Header files
set(HEADERS
    include/Matrix.h
    include/Network.h
)

# Create executable
add_executable(NeuroCore ${SOURCES} ${HEADERS})

# Link OpenMP
target_link_libraries(NeuroCore PUBLIC OpenMP::OpenMP_CXX)

# Set output directory
set_target_properties(NeuroCore PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

