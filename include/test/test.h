#ifndef TEST_H
#define TEST_H
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Unit testing module.
 *
 * To use this module, first call it for each MPI Rank `rank` as follows.
 *
 * test_t test;
 * int err = 0;
 * test = test_init("test description", rank, size); 
 *
 * The above code will initialize the testing facility. The variable `err` will
 * be used to keep track of the most recent error code that occurred.
 *
 * Next, add some test statements and keep track of their result using an error
 * code. That is, err = 0 (SUCCESS) otherwise FAILURE. To get a message if the
 * condition failed, use the `mpi_assert` function. Here's an example:
 *
 * err |= mpi_assert(1 > 0, rank); // Condition is true -> err = 0
 * err |= mpi_assert(1 < 0, rank); // Condition is false -> err = 1. 
 *
 * The call to mpi_assert will print a message if the condition failed. You can
 * also use:
 * * mpi_assert(pass, rank)
 * * mpi_no_except(x, rank)
 * Read their documentation below for more details.
 *
 * err = test_finalize(&test, err);
 * The final call will share whether the test passed or not among all ranks, and
 * display if test status as well as timing information. The test will only pass
 * if all ranks have err = 0.
 *
 */


#include <time.h>
#include <errno.h>
#include <mpi.h>
#include <string.h>

#ifndef ADDLINENUM
#define ADDLINENUM 0
#endif

#ifndef LINEFORMAT
#define LINEFORMAT "%03d"
#endif

#ifndef RANKFORMAT
#define RANKFORMAT "%d"
#endif

#ifndef ADDRANK
#define ADDRANK 0
#endif

#ifndef RANK
#define RANK 0
#endif


/*  Display the rank for which the condition 'x' failed on, and in what
 *  function, and at what line. Displays nothing if the condition is true.
 *  Return error code SUCCESS = 0 on success.
 */
#define mpi_assert(x, rank)__extension__ ({                                    \
        if (!(x)) printf("Assert "#x                                           \
                         " failed for rank %d at line %d in %s.\n",            \
                         rank, __LINE__, __func__);                            \
        !(x);                                                                  \
        })


/*  Display the rank for which the condition 'x' failed on, and in what
 *  function, and at what line. Displays nothing if the condition is true.
 *  Use this function to check when an error code is expected to be anything
 *  else than SUCCESS.
 *  Return error code SUCCESS = 0 on success.
 */
#define mpi_no_except(x, rank)__extension__ ({                                 \
        if (!(x)) printf("Exception "                                          \
                          #x" not raised for rank %d at line %d in %s.\n"      \
                         , rank, __LINE__, __func__);                          \
        !(x);                                                                  \
        })                                                                     

//Serial versions of the above functions
#define s_assert(x)__extension__ ({                                            \
        if (!(x)) printf("Assert "#x" failed at line %d in %s.\n",             \
                         __LINE__, __func__);                                  \
        !(x);                                                                  \
        })

#define s_no_except(x)__extension__ ({                                         \
        if (!(x)) printf("Exception "#x" not raised at line %d in %s.\n",      \
                         __LINE__, __func__);                                  \
        !(x);                                                                  \
        })                                                                     

#define inspect(fmt, x) printf("%s = " #fmt "\n", #x, x);

#define __line__ {\
        if (ADDLINENUM) {\
                printf(LINEFORMAT ": ", __LINE__);\
        }\
}

#define __sprintf_line__(x) {\
        if (ADDLINENUM) {\
                sprintf(x, "%s" LINEFORMAT ": ", (x), __LINE__);\
        }\
}

#define __rank__ {\
        if (ADDRANK) {\
                printf("%d: ", RANK);\
        }\
}

#define __sprintf_rank__(x) {\
        if (ADDRANK) {\
                sprintf(x, "%s" RANKFORMAT ": ", (x), RANK);\
        }\
}

#ifndef CUCHK
#define CUCHK(call) {                                    \
  cudaError_t err = call;                                                    \
  if( cudaSuccess != err) {                                                \
  fprintf(stderr, "CUDA error: %s:%s():%i %s.\n",        \
          __FILE__, __func__, __LINE__, cudaGetErrorString( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }
#endif

#define CCHK(err) {                                    \
  if( (err) != 0) {                                                \
  fprintf(stderr, "C error: %s:%s():%i %s.\n",        \
          __FILE__, __func__, __LINE__, strerror( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

#define MPICHK2(err, rank) {                                    \
 if (err != MPI_SUCCESS) {  \
 char error_string[2048]; \
 int length_of_error_string;\
 MPI_Error_string((err), error_string, &length_of_error_string);\
 fprintf(stderr, "MPI error: %s:%s():%i %3d: %s\n", \
         __FILE__, __func__, __LINE__, rank, error_string);\
 MPI_Abort(MPI_COMM_WORLD, err); \
}       \
}

#ifndef NDEBUG
#define MPICHK(err) {                                                         \
 if (err != MPI_SUCCESS) {                                                    \
 char error_string[2048];                                                     \
 int length_of_error_string;                                                  \
 MPI_Error_string((err), error_string, &length_of_error_string);              \
 fprintf(stderr, "MPI error: %s:%i %s(): %s\n",                               \
         __FILE__, __LINE__, __func__, error_string);                         \
 MPI_Abort(MPI_COMM_WORLD, err);                                              \
 fflush(stderr);                                                              \
 exit(EXIT_FAILURE);                                                          \
}                                                                             \
}
#else
#define MPICHK(err) {}
#endif

#define CHK(err) {                                    \
  if( err != 0) {                                                \
  fprintf(stderr, "Error: %s:%s():%i %s.\n",        \
          __FILE__, __func__, __LINE__, error_print( err) );              \
  fflush(stderr); \
  exit(EXIT_FAILURE);                                                  \
  } }

#define print_if(cond, ...) {\
        if (cond) printf(__VA_ARGS__);\
}



#define KNRM  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"



#define dstrcat(x, y) { \
        char cur = x[0]; \
        int i = 0;\
    while(cur != '\0')\
    {\
        cur = x[i];\
            i+=1;\
    }\
    int j = 0;\
    cur = y[0];\
    if (i > 0) {\
    i--;\
    }\
    while(cur != '\0')\
    {\
            x[i] = y[j];\
            i += 1;\
            j += 1;\
            cur = y[j];\
    }\
        x[i] = '\0';\
}

#define inspect_dfa(x, n) {\
        char tmp[2048];\
        dstrcat(tmp, "%s[0:%d] =");\
        for(int k = 0; k < n; ++k) {\
                dstrcat(tmp, " %g");\
        }\
        dstrcat(tmp, "\n");\
        if (n == 1) {\
                printf(tmp, #x, n, x[0]);\
        }\
        if (n == 2) {\
                printf(tmp, #x, n, x[0], x[1]);\
        }\
        if (n == 3) {\
                printf(tmp, #x, n, x[0], x[1], x[2]);\
        }\
        if (n == 4) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3] );\
        }\
        if (n == 5) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3], x[4] );\
        }\
        if (n == 6) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3], x[4], x[5] );\
        }\
        if (n == 7) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3], x[4], x[5], x[6] );\
        }\
        if (n == 8) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] );\
        }\
        if (n == 9) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8] );\
        }\
        if (n == 10) {\
                printf(tmp, #x, n, x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9] );\
        }\
}

#define print(...) __line__; printf(__VA_ARGS__);
#define mpi_print(...) __line__;__rank__; printf(__VA_ARGS__);
#define print_master(...) if (RANK == 0) { __line__; printf(__VA_ARGS__);}

#define inspect_fmta(fmt, x, n) inspect_a(fmt, x, 0, (n))
#define inspect_la(x, n) inspect_a("%ld", x, 0, (n))
#define inspect_da(x, n) inspect_a("%d", x, 0, (n))
#define inspect_fa(x, n) inspect_a("%f", x, 0, (n))
#define inspect_ga(x, n) inspect_a("%g", x, 0, (n))

#define inspect_fmtar(fmt, x, n0, n1) inspect_a(fmt, x, (n0), (n1))
#define inspect_lar(x, n0, n1)        inspect_a("%ld", x, (n0), (n1))
#define inspect_dar(x, n0, n1)        inspect_a("%d", x, (n0), (n1))
#define inspect_far(x, n0, n1)        inspect_a("%f", x, (n0), (n1))
#define inspect_gar(x, n0, n1)        inspect_a("%g", x, (n0), (n1))

#define inspect_fmtm(fmt, x, m, n) inspect_m(fmt, x, 0, (m), 0, (n), (n))
#define inspect_lm(x, m, n)        inspect_m("%ld", x, 0, (m), 0, (n), (n))
#define inspect_dm(x, m, n)        inspect_m("%d", x, 0, (m), 0, (n), (n))
#define inspect_fm(x, m, n)        inspect_m("%f", x, 0, (m), 0, (n), (n))
#define inspect_gm(x, m, n)        inspect_m("%g", x, 0, (m), 0, (n), (n))

#define inspect_s(x)  printf("%s = %s\n", #x, x)
#define inspect_d(x)  printf("%s = %d\n", #x, (int)x)
#define inspect_f(x)  printf("%s = %f\n", #x, x)
#define inspect_g(x)  printf("%s = %g\n", #x, x)
#define inspect_d3(q) printf("%s = {%d %d %d}\n", #q, (q).x, (q).y, (q).z)
#define inspect_f3(q) printf("%s = {%f %f %f}\n", #q, (q).x, (q).y, (q).z)
#define inspect_g3(q) printf("%s = {%g %g %g}\n", #q, (q).x, (q).y, (q).z)

#define inspect_a(fmt, x, n0, n1) { \
        __line__;\
        __rank__;\
        char __tmp__[2048];\
        sprintf(__tmp__, "%s[%d:%d] =", #x, (int)n0, (int)n1);\
        for (int i = (int)n0; i < (int)n1; ++i) {\
                sprintf(__tmp__, "%s " fmt, __tmp__, x[i]);\
        }\
        printf("%s\n", __tmp__);\
}\

#define inspect_m(fmt, x, m0, m1, n0, n1, n) { \
        char tmp[2048];\
        __line__;\
        __rank__;\
        sprintf(tmp, "%s[%d:%d,%d:%d] =\n", #x, (int)m0, (int)m1, \
                                                 (int)n0, (int)n1);\
        for (int i = (int)m0; i < (int)m1; ++i) {\
                __sprintf_line__(tmp);\
                __sprintf_rank__(tmp);\
                sprintf(tmp, "%s ", tmp);\
        for (int j = (int)n0; j < (int)n1; ++j) {\
                sprintf(tmp, "%s " fmt, tmp, x[j + i * (int)n]);\
        }\
                sprintf(tmp, "%s\n", tmp);\
        }\
        printf("%s", tmp);\
}


typedef struct
{
        const char *label;
        int rank;
        int size;
        int line;
        clock_t start;
        double elapsed;

} test_t;



#define DIVIDER_LEN 75
#define DISP_MAX_LEN 50
#define LABEL_MAX_LEN 40

#define test_init(label, rank, size) _test_init((label), __LINE__, (rank), (size))
test_t _test_init(const char *label, int line, int rank, int size);
int test_finalize(test_t *test, int x);
// Get last error.
int test_last_error(void);

// Signal that a test passed if all MPI ranks passed the test.
int test_all(int x, int rank);

void test_divider(void);
void print_status(int x);



#ifdef __cplusplus
}
#endif
#endif

