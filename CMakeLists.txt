CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

PROJECT(FourierPINNs VERSION 0.1 LANGUAGES C CXX)

INCLUDE(GNUInstallDirs)
INCLUDE(CTest)
INCLUDE(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

# nlohmann/json (header-only)
set(JSON_Install OFF CACHE BOOL "Install the bundled JSON library")
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
FetchContent_MakeAvailable(nlohmann_json)

# Build tests?
OPTION(FOURIERPINNS_ENABLE_TESTING "Should CTest be enabled?" ON)

add_library(FOURIERPINNS_HEADERS INTERFACE)
target_include_directories(FOURIERPINNS_HEADERS INTERFACE
  "${PROJECT_SOURCE_DIR}/include"
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(FOURIERPINNS_HEADERS INTERFACE
  ${TORCH_LIBRARIES}
  nlohmann_json::nlohmann_json)
separate_arguments(FOURIERPINNS_TORCH_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(FOURIERPINNS_HEADERS INTERFACE ${FOURIERPINNS_TORCH_FLAGS})

# Internal interface targets:
#   fourierpinns_core -> fourierpinns_training -> fourierpinns_models
add_library(fourierpinns_core INTERFACE)
target_link_libraries(fourierpinns_core INTERFACE FOURIERPINNS_HEADERS)

add_library(fourierpinns_training INTERFACE)
target_link_libraries(fourierpinns_training INTERFACE fourierpinns_core)

add_library(fourierpinns_models INTERFACE)
target_link_libraries(fourierpinns_models INTERFACE fourierpinns_training)

add_library(FourierPINNs::core ALIAS fourierpinns_core)


# Build and install the CLI; no installed C++ package is provided.
ADD_SUBDIRECTORY(src/cli)
set_target_properties(fourierpinns PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS fourierpinns RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

IF(FOURIERPINNS_ENABLE_TESTING)
  ENABLE_TESTING()
  ADD_SUBDIRECTORY(tests)
ENDIF()
