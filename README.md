# Installation on TACC Vista
  - Updated 12/18/2025 by Ke Xu based on suggestions from Wenyang Zhang (TACC) and Te-Yang Yeh (USC)

### TACC Vista compilation: using mvapich 
  - interactive mode need for mvapich version
  - module options are included in this compile script

```
idev -p gh -t 1:00:00 -N 1
./compile_mvp.sh
```

### TACC Vista compilation: using openmpi

```
module reset
module load cuda
./compile.sh
```

### Maximize performance when running AWP on Vista.

  - 1/ Add the -bynode flag when running the code using mpirun & openmpi.
  - 2/ Add export OMPI_MCA_io=^ompio before running the job. This will significantly improve the performance if your case has a large IO.
 
  -  And here is an example job script,

```
awpexec=/scratch/08458/wz_tacc/AWP/awp-topo-cmake-gh/release/src/awp/pmcl3d
cd $SLURM_SUBMIT_DIR
echo STARTING `date`

module reset
module load cuda

export OMPI_MCA_io=^ompio

rm -r output_sfc
mkdir -p output_sfc
rm -r output_rec
mkdir -p output_rec
rm -r debug
mkdir -p debug
rm -r output_ckp
mkdir -p output_ckp

mpirun -np 1440 -bynode  $awpexec \
  -X 1536 -Y 2700 -Z 288,32,480 -x 16 -y 30 -G 3 \
  --TMAX 150.0 --DH 90 --DT 0.00075 \
  --NPC 0 --ND 40 --ARBC 0.92 \
  --NSRC 0,0,0 --SOURCEFILE source_txt \
  --IDYNA 0  --NVE 1 --NVAR 3 \
  --MEDIASTART 2 --INVEL mesh \
  --INTOPO topo_bin \
  -c output_ckp/CHKP --OUT output_sfc \
  --NTISKP 50 --WRITE_STEP 100 \
  --NBGX 1,1,1 --NEDX 13824,4608,1536 \
  --NBGY 1,1,1 --NEDY 24300,8100,2700 \
  --NBGZ 1,1,1 --NEDZ 1,1,1 \
  --FAC 1.0 --Q0 150. --EX 0.4 --FP 1.0 \
  --QSI 0.05 --QPQSR 2.0 --VMIN 250.0 --MAXVPVSR 10. --DMIN 1500. \
  --FOLLOWBATHY 1 --SoCalQ 0
echo ENDING `date`
```


## Dependencies for Summit (old version)
The following dependencies must be met in order to compile the source code:
* C
* CUDA
* MPI
* CMake

## Installation on Summmit (old version)

To compile the source code on Summit, it is necessary to first load some
modules:
```bash
module load cmake gcc cuda spectrum-mpi
```

Compile AWP using CMake:
```bash
$ mkdir release
$ cd release
$ cmake ..
$ make

```
If the source is successfully compiled, the main executable `pmcl3d` is placed in `release/src/awp`.  

## Tests
For test environments that have *CMake*, *MPI*, *CUDA*-enabled devices available,
testing can be done by calling 
```
make test
```
For verbose output, call *CTest* directly
```
ctest --verbose
```

**Warning**: Many of the tests will fail on Summit. See the
[awp-benchmarks](https://github.com/SCECcode/awp-benchmarks) repository for
running tests on Summit.

