cmake_minimum_required (VERSION 3.2.0)

include("../cmake/GetGTest.cmake")

add_executable(unittest)
target_sources(unittest PRIVATE
  main.cpp
  IntegrationTest.cpp
  TestAssembleLoadVector.cpp
  TestAssembleStiffnessMatrix.cpp
  TestCoordinateTransform.cpp
  TestDirichletBoundary.cpp
  TestElementVector.cpp
  TestGradientShapeFunction.cpp
  TestShapeFunction.cpp
  TestStiffnessMatrix.cpp
)
target_include_directories(unittest PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(unittest PUBLIC Eigen IGL GTest)
target_compile_features(unittest PUBLIC cxx_std_11)
target_compile_definitions(unittest PUBLIC -D_USE_MATH_DEFINES)
