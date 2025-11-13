add_executable(test_problem gwo_main.cpp)
target_include_directories(test_problem PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(test_problem PRIVATE Eigen3::Eigen)