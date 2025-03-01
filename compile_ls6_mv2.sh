#!/bin/bash

module unload intel impi
module load cmake gcc mvapich2 cuda
module list

rm -r release
mkdir -p release

cd release
#export CC=$(which mpicc)
#export CXX=$(which mpicxx)
export LD_LIBRARY_PATH=/opt/apps/gcc11_2/mvapich2/2.3.7/lib:$LD_LIBRARY_PATH
export CPATH=/opt/apps/gcc11_2/mvapich2/2.3.7/include:$CPATH
#export LD_LIBRARY_PATH=/.../mvapich2/lib:$LD_LIBRARY_PATH
#export CPATH=/.../mvapich2/include:$CPATH
cmake -DCMAKE_VERBOSE_MAKEFILE=ON ..
make
