# SPDX-FileCopyrightText: 2020 CERN
# SPDX-License-Identifier: Apache-2.0

# In Allen, there are "allen_add_..." CMake functions that wrap the
# needed steps
# Compile for host/device as specified
if(TARGET_DEVICE STREQUAL "CPU")
  # Not clear why this is needed, but results in link errors if not...
  add_definitions("-x c++")
  set_source_files_properties(test_saxpy.cu PROPERTIES LANGUAGE CXX)
  set_source_files_properties(test_mandel.cu PROPERTIES LANGUAGE CXX)
endif()

add_executable(test_saxpy test_saxpy.cu)
target_link_libraries(test_saxpy PRIVATE CopCore::CopCore)
add_test(NAME test_saxpy COMMAND test_saxpy)

add_executable(test_mandel test_mandel.cu)
target_link_libraries(test_mandel PRIVATE CopCore::CopCore)
add_test(NAME test_mandel COMMAND test_mandel)
