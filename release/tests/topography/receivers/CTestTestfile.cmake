# CMake generated Testfile for 
# Source directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/receivers
# Build directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/topography/receivers
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_topography_receivers "/sw/summit/spack-envs/summit-plus/opt/gcc-8.5.0/spectrum-mpi-10.4.0.6-20230210-f3ouht4ckff2qogy74bwki5ovljfou36/bin/mpiexec" "-n" "4" "--oversubscribe" "test_topography_receivers" "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/fixtures/receiver.txt")
set_tests_properties(test_topography_receivers PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/receivers/CMakeLists.txt;15;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/receivers/CMakeLists.txt;0;")
