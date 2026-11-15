// Source-compiled HIP device identity: no architecture spoofing or guessed struct ABI.
#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  char* end = nullptr;
  long index = std::strtol(argv[1], &end, 10);
  if (!end || *end || index < 0 || index > 1024) return 2;
  hipDeviceProp_t properties{};
  if (hipGetDeviceProperties(&properties, static_cast<int>(index)) != hipSuccess) return 1;
  std::cout << properties.gcnArchName << '\n';
}
