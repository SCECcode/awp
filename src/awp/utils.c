#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <mpi.h>

#include <awp/pmcl3d_cons.h>
#include <awp/utils.h>

const double   micro = 1.0e-6;

double gethrtime(void)
{
    struct timeval TV;
    int RC = gettimeofday(&TV,NULL);

    if (RC == -1){
       printf("Bad call to gettimeofday\n");
       return(-1);
    }

    return ( ((double)TV.tv_sec ) + micro * ((double)  TV.tv_usec));
}

void error_check(int ierr, char *message){
   char errmsg[500];
   int errlen;
   if (ierr != MPI_SUCCESS) {
      fprintf(stderr, "%d: Error in %s\n", ierr, message);
      MPICHK(MPI_Error_string(ierr, errmsg, &errlen));
      fprintf(stderr, "%s", errmsg);
   }
}

