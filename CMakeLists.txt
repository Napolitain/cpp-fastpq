cmake_minimum_required(VERSION 3.24)

project(
    cpp_pq
    VERSION 0.1.0
    DESCRIPTION "Bounded-priority multi-queue priority queue with benchmarks"
    LANGUAGES CXX
)

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(CPP_PQ_CXX_STANDARD_FEATURE cxx_std_23)
if("cxx_std_26" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CPP_PQ_CXX_STANDARD_FEATURE cxx_std_26)
elseif("cxx_std_23" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CPP_PQ_CXX_STANDARD_FEATURE cxx_std_23)
elseif("cxx_std_20" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    set(CPP_PQ_CXX_STANDARD_FEATURE cxx_std_20)
else()
    message(FATAL_ERROR "cpp_pq requires at least C++20 support.")
endif()

if(PROJECT_IS_TOP_LEVEL)
    message(STATUS "cpp_pq requesting ${CPP_PQ_CXX_STANDARD_FEATURE}")
endif()

option(CPP_PQ_BUILD_BENCHMARKS "Build the benchmark executable" ${PROJECT_IS_TOP_LEVEL})
option(CPP_PQ_BUILD_TESTS "Build the test executable" ${PROJECT_IS_TOP_LEVEL})

add_library(cpp_pq INTERFACE)
add_library(cpp_pq::cpp_pq ALIAS cpp_pq)

target_compile_features(cpp_pq INTERFACE ${CPP_PQ_CXX_STANDARD_FEATURE})
target_include_directories(
    cpp_pq
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if(CPP_PQ_BUILD_BENCHMARKS)
    add_executable(cpp_pq_benchmark benchmarks/benchmark_priority_queues.cpp)
    target_link_libraries(cpp_pq_benchmark PRIVATE cpp_pq::cpp_pq)
    set_target_properties(cpp_pq_benchmark PROPERTIES OUTPUT_NAME benchmark_priority_queues)

    add_executable(cpp_pq_dynamic_variant_benchmark benchmarks/benchmark_dynamic_variants.cpp)
    target_link_libraries(cpp_pq_dynamic_variant_benchmark PRIVATE cpp_pq::cpp_pq)
    set_target_properties(cpp_pq_dynamic_variant_benchmark PROPERTIES OUTPUT_NAME benchmark_dynamic_variants)
endif()

if(CPP_PQ_BUILD_TESTS)
    include(CTest)

    if(BUILD_TESTING)
        add_executable(cpp_pq_tests tests/multi_queue_priority_queue_tests.cpp)
        target_link_libraries(cpp_pq_tests PRIVATE cpp_pq::cpp_pq)
        add_test(NAME cpp_pq.multi_queue_priority_queue COMMAND cpp_pq_tests)
    endif()
endif()

install(
    TARGETS cpp_pq
    EXPORT cpp_pqTargets
    INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

set(CPP_PQ_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/cpp_pq")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpp_pqConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_pqConfig.cmake"
    INSTALL_DESTINATION "${CPP_PQ_INSTALL_CMAKEDIR}"
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_pqConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
)

install(
    FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cpp_pqConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cpp_pqConfigVersion.cmake"
    DESTINATION "${CPP_PQ_INSTALL_CMAKEDIR}"
)

install(
    EXPORT cpp_pqTargets
    FILE cpp_pqTargets.cmake
    NAMESPACE cpp_pq::
    DESTINATION "${CPP_PQ_INSTALL_CMAKEDIR}"
)
