# CMake generated Testfile for 
# Source directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/mpi
# Build directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/mpi
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_indexed "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/mpi/test_indexed")
set_tests_properties(test_indexed PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/mpi/CMakeLists.txt;14;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/mpi/CMakeLists.txt;0;")
add_test(test_mpi_io "/sw/summit/spack-envs/summit-plus/opt/gcc-8.5.0/spectrum-mpi-10.4.0.6-20230210-f3ouht4ckff2qogy74bwki5ovljfou36/bin/mpiexec" "-n" "4" "--oversubscribe" "test_mpi_io")
set_tests_properties(test_mpi_io PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/mpi/CMakeLists.txt;32;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/mpi/CMakeLists.txt;0;")
