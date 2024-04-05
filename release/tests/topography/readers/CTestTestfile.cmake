# CMake generated Testfile for 
# Source directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/readers
# Build directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/topography/readers
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_topography_serial_reader "/sw/summit/spack-envs/summit-plus/opt/gcc-8.5.0/spectrum-mpi-10.4.0.6-20230210-f3ouht4ckff2qogy74bwki5ovljfou36/bin/mpiexec" "-n" "6" "--oversubscribe" "test_topography_serial_reader")
set_tests_properties(test_topography_serial_reader PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/readers/CMakeLists.txt;17;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/readers/CMakeLists.txt;0;")
