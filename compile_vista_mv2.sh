#!/bin/bash


module unload cmake
#module unload gcc 
#ml reset
#ml nvhpc-hpcx-cuda11/23.7
#module load intel cmake impi cuda
module unload openmpi
module use /scratch/00494/tg457572/packages/modulefiles

module load e4s
module load mvapich
module load tau 
ml cuda/12.4

rm -r release
mkdir -p release

cd release
#export CXX

#export MPI_HOME="/home1/07936/tg872351/mvp-pre-rc-ofi-cuda12.5"
export MPI_HOME="/home1/07936/tg872351/mvp-pre-rc-zfp-cuda12.4"

export PATH=${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export CPATH=${MPI_HOME}/include:$CPATH
export C_INCLUDE_PATH=${MPI_HOME}/include:$C_INCLUDE_PATH

export LD_LIBRARY_PATH=/home1/apps/nvidia/Linux_aarch64/24.7/cuda/12.5/lib64:$LD_LIBRARY_PATH

export CC=$(which mpicc)
export CXX=$(which mpicxx)
export FC=$(which mpifort)
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)
export MPI_INCLUDE_PATH=${MPI_HOME}/include


#export CXX
#export LD_LIBRARY_PATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/lib:$LD_LIBRARY_PATH
#export CPATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/include:$CPATH
#export C_INCLUDE_PATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/include:$C_INCLUDE_PATH
#export PATH=/opt/intel/compilers_and_libraries_2020.4.304/linux/mpi/intel64/bin:$PATH

echo -e "\n"
echo "======== PATH=============="
echo $PATH | tr : '\n'

echo -e "\n"
echo "======== INCLUDE =============="
echo $INCLUDE | tr : '\n'

echo -e "\n"
echo "======== LD_LIBRARY_PATH=============="
echo $LD_LIBRARY_PATH | tr : '\n'

module list
echo -e "mpicc = `which mpicc`"
echo ""

echo "LD_PRELOAD=$LD_PRELOAD"

#export MPI_HOME=${TACC_MPI_DIR}

echo "TACC_IMPI_INC=$TACC_IMPI_INC"
echo "MPI_HOME=$MPI_HOME"


cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DCMAKE_C_COMPILER=`which mpicc` -DCMAKE_CXX_COMPILER=`which mpicxx` -DMY_INCLUDE_PATH=$MPI_INCLUDE_PATH ..
make
