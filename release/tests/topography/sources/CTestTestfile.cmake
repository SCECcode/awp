# CMake generated Testfile for 
# Source directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources
# Build directory: /ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/topography/sources
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(test_topography_sources_dm "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/topography/sources/test_topography_sources_dm")
set_tests_properties(test_topography_sources_dm PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources/CMakeLists.txt;41;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources/CMakeLists.txt;0;")
add_test(test_topography_source_distribution "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/release/tests/topography/sources/test_topography_source_distribution")
set_tests_properties(test_topography_source_distribution PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources/CMakeLists.txt;44;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources/CMakeLists.txt;0;")
add_test(test_topography_sources "/sw/summit/spack-envs/summit-plus/opt/gcc-8.5.0/spectrum-mpi-10.4.0.6-20230210-f3ouht4ckff2qogy74bwki5ovljfou36/bin/mpiexec" "-n" "4" "--oversubscribe" "test_topography_sources" "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/fixtures/source.txt")
set_tests_properties(test_topography_sources PROPERTIES  _BACKTRACE_TRIPLES "/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources/CMakeLists.txt;47;add_test;/ccs/home/dean316/AWP_code_archive/gpu/awp_topo/awp/tests/topography/sources/CMakeLists.txt;0;")
