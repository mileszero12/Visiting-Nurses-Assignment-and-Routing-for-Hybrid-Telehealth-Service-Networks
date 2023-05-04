cmake_minimum_required(VERSION 3.17)
project(code)
set(CMAKE_CXX_STANDARD 17)

option(CXX "enable C++ compilation" ON)
if (CXX)
    enable_language(CXX)
endif ()

set(ENV{GUROBI_HOME} /Library/gurobi1001/macos_universal2)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR})
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif ()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#run profile
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")


find_package(GUROBI REQUIRED)
#find_package(xgboost REQUIRED)
#find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${GUROBI_INCLUDE_DIRS})
#cvrpsep
#link_directories(${PROJECT_SOURCE_DIR}/lib)
#include_directories(${PROJECT_SOURCE_DIR}/include)

#find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(Python REQUIRED Development)
# message(STATUS "Python_VERSION: ${Python_INCLUDE_DIRS}")
include_directories(/Users/pikay/miniconda/lib/python3.10)

#include_directories(${Python_INCLUDE_DIRS})
#add_executable(CGLR main.cpp)
#target_include_directories(CGLR PRIVATE ${Python3_INCLUDE_DIRS})
#target_link_libraries(CGLR PRIVATE ${Python3_LIBRARIES})
# find_package(REQUIRED COMPONENTS iostreams system filesystem)

# list source files here
set(code main.cpp
       NurseVrp.cpp NurseVrp.hpp Solver.cpp Solver.hpp MaxClique.cpp Pivot.cpp MACRO.hpp InitHeuristic.cpp Label.cpp
        localSearch.cpp
        )

add_executable(${CMAKE_PROJECT_NAME} ${code})

target_include_directories(code PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${Boost_INCLUDE_DIRS})

target_link_libraries(${CMAKE_PROJECT_NAME}
        ${GUROBI_LIBRARIES}
        #-lpython3.10
        Python::Python
        #Python::Module
        #Python::NumPy
        )




if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
    include(FeatureSummary)
    feature_summary(WHAT ALL)
endif ()


