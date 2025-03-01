#!/bin/bash

module unload intel
module load cmake gcc impi cuda

rm -r release
mkdir -p release

cd release
export CC=$(which mpigcc)
export CXX=$(which mpigxx)
export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib:$LD_LIBRARY_PATH
export CPATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/include:$CPATH
#export PATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin:$PATH
module list
env | grep "PATH"

cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make
