set(INSTALL_GTEST OFF)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG v1.15.2
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

link_libraries(GTest::gtest_main)
add_executable(test_alloc test_alloc.cu)
gtest_discover_tests(test_alloc)
add_executable(test_lanczos test_lanczos.cu)
gtest_discover_tests(test_lanczos)
