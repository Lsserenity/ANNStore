#pragma once

namespace annfs::tests {

int RunHnswFsIoMain(int argc, char** argv, bool enable_flops_count);
int RunHnswFsOpenOnlyMain(int argc, char** argv);
int RunHnswFsIOAsyncMain(int argc, char** argv);

} // namespace annfs::tests
