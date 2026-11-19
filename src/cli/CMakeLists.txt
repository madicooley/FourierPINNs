
add_library(fourierpinns_cli_runtime STATIC
	CLI.cpp
	../solvers/SolverFactory.cpp
)
target_link_libraries(fourierpinns_cli_runtime PUBLIC fourierpinns_models)
target_include_directories(fourierpinns_cli_runtime PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(fourierpinns_cli_runtime PUBLIC cxx_std_17)

add_executable(fourierpinns Main.cpp)
target_link_libraries(fourierpinns PRIVATE fourierpinns_cli_runtime)
set_target_properties(fourierpinns PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET fourierpinns
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:fourierpinns>)
endif (MSVC)
