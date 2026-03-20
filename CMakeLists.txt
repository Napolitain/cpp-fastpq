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
option(CPP_PQ_ENABLE_FOLLY_FBVECTOR "Fetch Folly sources and enable the fbvector benchmark variant" OFF)

add_library(cpp_pq INTERFACE)
add_library(cpp_pq::cpp_pq ALIAS cpp_pq)

target_compile_features(cpp_pq INTERFACE ${CPP_PQ_CXX_STANDARD_FEATURE})
target_include_directories(
    cpp_pq
    INTERFACE
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

if(CPP_PQ_BUILD_BENCHMARKS AND CPP_PQ_ENABLE_FOLLY_FBVECTOR)
    include(FetchContent)

    if(PROJECT_IS_TOP_LEVEL)
        message(STATUS "cpp_pq fetching Folly sources for fbvector benchmarks")
    endif()

    if(POLICY CMP0169)
        cmake_policy(PUSH)
        cmake_policy(SET CMP0169 OLD)
    endif()

    FetchContent_Declare(
        folly_source
        GIT_REPOSITORY https://github.com/facebook/folly.git
        GIT_TAG 101f1817c499860252db4799acbb3a66b47c8929
        GIT_SHALLOW TRUE
    )
    FetchContent_Declare(
        boost_preprocessor_source
        GIT_REPOSITORY https://github.com/boostorg/preprocessor.git
        GIT_TAG boost-1.88.0
        GIT_SHALLOW TRUE
    )
    FetchContent_Declare(
        fmt_source
        GIT_REPOSITORY https://github.com/fmtlib/fmt.git
        GIT_TAG 11.0.2
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(folly_source)
    if(NOT folly_source_POPULATED)
        FetchContent_Populate(folly_source)
    endif()
    FetchContent_GetProperties(boost_preprocessor_source)
    if(NOT boost_preprocessor_source_POPULATED)
        FetchContent_Populate(boost_preprocessor_source)
    endif()
    FetchContent_GetProperties(fmt_source)
    if(NOT fmt_source_POPULATED)
        FetchContent_Populate(fmt_source)
    endif()
    if(POLICY CMP0169)
        cmake_policy(POP)
    endif()

    find_package(Threads REQUIRED)
    set(FOLLY_HAVE_PTHREAD "${CMAKE_USE_PTHREADS_INIT}")
    set(FOLLY_HAVE_PTHREAD_ATFORK FALSE)
    set(FOLLY_HAVE_LIBGFLAGS FALSE)
    set(FOLLY_HAVE_LIBGLOG FALSE)
    set(FOLLY_USE_JEMALLOC FALSE)
    set(FOLLY_HAVE_ACCEPT4 FALSE)
    set(FOLLY_HAVE_GETRANDOM FALSE)
    set(FOLLY_HAVE_PREADV FALSE)
    set(FOLLY_HAVE_PWRITEV FALSE)
    set(FOLLY_HAVE_CLOCK_GETTIME FALSE)
    set(FOLLY_HAVE_PIPE2 FALSE)
    set(FOLLY_HAVE_IFUNC FALSE)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64|i[3-6]86|x86)$")
        set(FOLLY_HAVE_UNALIGNED_ACCESS TRUE)
    else()
        set(FOLLY_HAVE_UNALIGNED_ACCESS FALSE)
    endif()
    set(FOLLY_HAVE_VLA FALSE)
    set(FOLLY_HAVE_WEAK_SYMBOLS FALSE)
    set(FOLLY_HAVE_LINUX_VDSO FALSE)
    set(FOLLY_HAVE_MALLOC_USABLE_SIZE FALSE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        set(FOLLY_HAVE_INT128_T TRUE)
    else()
        set(FOLLY_HAVE_INT128_T FALSE)
    endif()
    set(FOLLY_HAVE_WCHAR_SUPPORT TRUE)
    set(FOLLY_HAVE_EXTRANDOM_SFMT19937 FALSE)
    set(HAVE_VSNPRINTF_ERRORS FALSE)
    set(FOLLY_HAVE_LIBUNWIND FALSE)
    set(FOLLY_HAVE_DWARF FALSE)
    set(FOLLY_HAVE_ELF FALSE)
    set(FOLLY_HAVE_SWAPCONTEXT FALSE)
    set(FOLLY_HAVE_BACKTRACE FALSE)
    set(FOLLY_USE_SYMBOLIZER FALSE)
    set(FOLLY_HAVE_SHADOW_LOCAL_WARNINGS FALSE)
    set(FOLLY_HAVE_LIBLZ4 FALSE)
    set(FOLLY_HAVE_LIBLZMA FALSE)
    set(FOLLY_HAVE_LIBSNAPPY FALSE)
    set(FOLLY_HAVE_LIBZ FALSE)
    set(FOLLY_HAVE_LIBZSTD FALSE)
    set(FOLLY_HAVE_LIBBZ2 FALSE)
    set(FOLLY_LIBRARY_SANITIZE_ADDRESS FALSE)
    set(FOLLY_SUPPORT_SHARED_LIBRARY FALSE)
    set(FOLLY_HAVE_LIBRT FALSE)
    set(FOLLY_HAVE_VSOCK FALSE)
    configure_file(
        "${folly_source_SOURCE_DIR}/CMake/folly-config.h.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/generated-folly/folly/folly-config.h"
    )

    add_library(cpp_pq_folly_fbvector_headers INTERFACE)
    target_include_directories(
        cpp_pq_folly_fbvector_headers
        SYSTEM
        INTERFACE
            "${CMAKE_CURRENT_BINARY_DIR}/generated-folly"
            "${folly_source_SOURCE_DIR}"
            "${boost_preprocessor_source_SOURCE_DIR}/include"
            "${fmt_source_SOURCE_DIR}/include"
    )
    target_compile_definitions(
        cpp_pq_folly_fbvector_headers
        INTERFACE
            CPP_PQ_HAS_FOLLY_FBVECTOR=1
            FOLLY_ASSUME_NO_JEMALLOC=1
            FOLLY_ASSUME_NO_TCMALLOC=1
            FMT_HEADER_ONLY=1
    )
endif()

if(CPP_PQ_BUILD_BENCHMARKS)
    add_executable(cpp_pq_benchmark benchmarks/benchmark_priority_queues.cpp)
    target_link_libraries(cpp_pq_benchmark PRIVATE cpp_pq::cpp_pq)
    if(TARGET cpp_pq_folly_fbvector_headers)
        target_sources(cpp_pq_benchmark PRIVATE benchmarks/folly_fbvector_shims.cpp)
        target_link_libraries(cpp_pq_benchmark PRIVATE cpp_pq_folly_fbvector_headers)
    endif()
    set_target_properties(cpp_pq_benchmark PROPERTIES OUTPUT_NAME benchmark_priority_queues)
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
