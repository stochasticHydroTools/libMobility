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

add_executable(test_alloc test_alloc.cu)
target_link_libraries(test_alloc PRIVATE GTest::gtest_main spdlog::spdlog)
gtest_discover_tests(test_alloc)
