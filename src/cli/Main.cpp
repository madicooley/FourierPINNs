#include "fourierpinns/utils/utils.h"
#include "CLI.h"

// Keep main() separate so tests can compile CLI.cpp without colliding
// with Catch2's own main().
int main(int argc, char* argv[]) {
    return runFourierPINNsCLI(argc, argv);
}
