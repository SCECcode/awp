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

int copyfile(const char *output, const char *input)
{
        FILE *fin = fopen(input, "r"); 
        FILE *fout = fopen(output, "w"); 
        int count = -1;

        if (fin == NULL) {
                fprintf(stderr, "Cannot open file %s. \n", input);
                return count;
        }

        if (fout == NULL) {
                fprintf(stderr, "Cannot write to file %s. \n", output);
                return count;
        }
  
        char ch;
        while ((ch = fgetc(fin)) != EOF)
                fputc(ch, fout);
        fclose(fin);
        fclose(fout);
        return count;
}

