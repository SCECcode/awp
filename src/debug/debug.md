# AWP DEBUG utilities
These notes describe a set of scripts that are helpful for both debugging and further
developing AWP. Before you use any of these scripts, please commit your work beforehand so
that you can easily revert the changes in case something goes wrong.

## Debugging
It is not uncommon for segmentation faults to occur when running AWP. Sometimes,
these are caused by user errors and other times they are caused by bugs in the
program. In either case, these errors can be time consuming to identity without
a systematic approach and proper tools. This guide is meant to show you one
effective way of catching segmentation fault using the gdb debugger.

If you can read this document,
then chances are that you have access to a sufficiently recent version of AWP
that enables gdb to attach to one of your runs.

1. Search for `GDB_ATTACH`
at the top of `pmcl3d.c` and uncomment the line. If you cannot find this macro,
go ahead and copy and paste the following block of code after the `MPI_Init,
MPI_Comm_rank, MPI_Comm_size` calls.

```C
    if ( rank == 0) {
        volatile int i = 0;
        printf("Process ID %d is ready for attach\n", getpid());
        fflush(stdout);
        while (0 == i)
            sleep(5);
    }

```

2. Compile AWP in debug mode:
```
$ cd build
$ make clean
$ cmake -DCMAKE_BUILD_TYPE=Debug ..
$ make

```
3. Launch an interactive job and load gdb. On summit, see the user guide for
   launching interactive jobs: https://docs.olcf.ornl.gov/systems/summit_user_guide.html#interactive-jobs
   To use gdb on Summit, you need to load it: `module load gdb`.
4. Run AWP: Once your interactive jobs has started, call AWP with its usual input
   arguments associated with your particular run. Put `&` at the end of the
   command to spawn it in a background process so that you regain control of the
   terminal. For e.g,
   ```
        jsrun -n 4 -a 3 -c 3 -g 1 -r 4 -d cyclic pmcl3d [ARGS]&
   ```
   After a while, you should see:
   ```
        Process ID 23969 is ready for attach
   ```
5. Run gdb: `gdb ---pid PID`, but replace PID with process ID displayed in the
   previous step. Press `n` followed by the return key until you see
   ```
   while (0 == i)
   ```
   Run the command:
   `set var i = 1`
   This command will cause gdb do modify the variable `i` and therefore exit the
   while loop. Next, you can proceed using gdb as you please. To simplify go to
   the next error, type `c` and gdb should run until the error occurs and tell
   you what statement in the source code that caused the error.


## Memory issues
We have found that certain bugs in AWP are due to memory errors. Tools such as `valgrind`
and `cuda-memcheck` are excellent for reporting many memory related issues. You are highly
encouraged to use these tools during development. So please use them to detect and fix
memory related issues before the code is used in production. As a precautionary measure,
we can avoid uninitialized memory errors by always initializing the allocated memory
immediately after it has been allocated.

### Uninitialized memory
The bash script `addmemset.sh` can be used to automatically add a statement that will
zero initialize memory after each device allocation call. **Warning:** Running this
script can modify all .c and .cu files in the source directory. Please commit any changes
and any untracked files before proceeding. No special instructions are required to run
this script, call
```bash
$ bash addmemset.sh
```

### Counting function calls
The number of calls to `malloc` should match the number of calls to `free`.  A simply way
to check that this is the case is to count number of matches from a grep search. For
convenience, the bash script `numcalls.sh` will count the number of occurrences for some
functions residing in either .c or .cu files. Example usage:
```bash
 bash numcalls.sh malloc free cudaMalloc cudaFree cudaMallocHost cudaFreeHost cudaMemset
malloc: 19
free: 19
cudaMalloc: 173
cudaFree: 188
cudaMallocHost: 40
cudaFreeHost: 40
cudaMemset: 204
```

## MPI and CUDA errors
In many cases it is a lot easier to identify problems if errors are reported as soon as
they occur. However, in practice it can be quite tedious to check all error codes. When
it comes to cuda and MPI error codes, it is quite easy to check them because each CUDA
function starts with `cuda...` and returns an error code. The same holds true for MPI
functions. With the help of the preprocessor macros `CUCHK` and MPICHK` we can intercept
any error from these libraries and report the error message, as well as file, function,
and line number at which it occured. These macros are defined in `pmcl3d_cons.h`. Both of
these macros can be disabled by defining `NDEBUG`. To avoid having to modify the source,
you can instead modify the Makefile by adding the following to it,
```bash
CFLAGS = ... -DNDEBUG
GFLAGS = ... -DNDEBUG
```
The dots `...` represent any already existing arguments.


### Automatic checking
The
makefile `Makefile.debug` can modify the source to enable or disable these additional
checks. Both commands will modify source files, but the disabling call listed below should return to the
source files to their original state. To be safe, make sure to commit any changes
(including untracked files) before proceeding.

To enable, call
```bash
$ make -f Makefile.debug enable
```
To disable, call
```bash
$ make -f Makefile.debug disable
```

### Manual checking
You can also add these checks manually. Here is an example that demonstrates how to check
if a call to cudaMalloc is successful. We add the following statements to `pmcl3d.c`
```
float *var;
CUCHK(cudaMalloc((void**)&var, -1));
```

In this example, the number of bytes is invalid and will therefore generate an error. The
error message is:
```
CUDA error in pmcl3d.c:776 main(): invalid argument.
```

Since MPI exits by default on any error, it is necessary to first enable error handling.
After `MPI_Init(...)` in `main()`, add the statement
```
MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
```
After doing that, you can check for MPI errors by wrapping the MPI function call of
interest in `MPICHK` (similar to `CUCHK`).






