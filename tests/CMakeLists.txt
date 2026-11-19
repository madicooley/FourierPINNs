cmake_minimum_required(VERSION 3.15)

include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.2
)

FetchContent_MakeAvailable(Catch2)

# Catch2 passes C++ standard library types across the test executable's ABI.
target_compile_options(Catch2 PRIVATE ${FOURIERPINNS_TORCH_FLAGS})
target_compile_options(Catch2WithMain PRIVATE ${FOURIERPINNS_TORCH_FLAGS})

add_executable(fourierpinns_tests
  FactoryTests.cpp
  FourierBasisBuilderTests.cpp
  FourierNetTests.cpp
  FourierLayerTests.cpp
  FinalLinearLayerTests.cpp
  IntegrationSmokeTests.cpp
  LoadedDomainTests.cpp
  NeuralNetTests.cpp
  StatsLoggerTests.cpp
  Section65Tests.cpp
  PDEBasisTests.cpp
  TrainerTests.cpp
  TrainerLstsqTests.cpp
  TrainerUtilsTests.cpp
  ValidateConfigCLITests.cpp
)

target_link_libraries(fourierpinns_tests PRIVATE
  Catch2::Catch2WithMain
  fourierpinns_cli_runtime
)

target_include_directories(fourierpinns_tests PRIVATE
  ${PROJECT_SOURCE_DIR}/src/cli
)

target_compile_features(fourierpinns_tests PRIVATE cxx_std_17)

add_test(NAME fourierpinns_tests COMMAND fourierpinns_tests)
set_tests_properties(fourierpinns_tests PROPERTIES LABELS "unit;smoke")
