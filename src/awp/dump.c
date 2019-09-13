#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <awp/dump.h>

void dump_all_data(_prec *d_u1, _prec *d_v1, _prec *d_w1, 
    _prec *d_xx, _prec *d_yy, _prec *d_zz,_prec *d_xz,_prec *d_yz,_prec *d_xy, 
    int nel, int tstep, int tsub, int d_i, int rank, int ncpus){

    #ifdef DUMP_SNAPSHOTS
    dump_all_vels(d_u1, d_v1, d_w1, nel, d_i, tstep, tsub, rank, ncpus);
    dump_all_stresses(d_xx, d_yy, d_zz, d_xz, d_yz, d_xy, nel, d_i, tstep, tsub, rank, ncpus);
    #endif

}

void dump_local_variable(_prec *var, long int nel, char *varname, char desc, int tstep, int tsub, int rank, int ncpus){
   #ifdef DUMP_SNAPSHOTS
   FILE *fid;
   char outfile[200];
   sprintf(outfile, "output_dbg.%1d/%s_%c_%07d-%02d.r%1d", ncpus, varname, desc, tstep, tsub, rank);

   fid=fopen(outfile, "w");
   if (fid ==NULL){
      fprintf(stderr, "could not open %s\n", outfile);
   }
   
   fwrite(var, nel, sizeof(_prec), fid);
   fclose(fid);
   #endif
}

void dump_nonzeros(_prec *var, int nx, int ny, int nz, char *varname, int desc, int tstep, int tsub, 
     int rank, int ncpus){
   #ifdef DUMP_NONZERO
   FILE *fid;
   char outfile[200];
   sprintf(outfile, "output_dbg.%1d/%s_%1d_%07d-%02d_r%1d.txt", ncpus, varname, desc, tstep, tsub, rank);
   _prec *buf;
   int i, j, k;
   long int nel;
   long int pos;

   fid=fopen(outfile, "w");
   if (fid ==NULL){
      fprintf(stderr, "could not open %s\n", outfile);
   }
   
   nel = (long int) nx * ny * nz;
   buf=(_prec* ) calloc(nel, sizeof(_prec));
   CUCHK(cudaMemcpy(buf, var, nel*sizeof(_prec), cudaMemcpyDeviceToHost));

   for (i=0; i <nx; i++){
      for (j=0; j<ny; j++){
         for (k=0; k<nz; k++){ 
           pos= i*ny*nz + j*nz + k;
           if (buf[pos] != 0.0f) fprintf(fid, "%d %d %d: %e\n", i, j, k, buf[pos]);
         }
      }
   }
   
   fclose(fid);

   free(buf);
   #endif
}


void dump_variable(_prec *var, long int nel, char *varname, int desc, int tstep, int tsub, int rank, int ncpus){
   FILE *fid;
   char outfile[200];
   sprintf(outfile, "output_dbg.%1d/%s_%d_%07d-%1d.r%1d", ncpus, varname, desc, tstep, tsub, rank);
   _prec *buf;

   fid=fopen(outfile, "w");
   if (fid ==NULL){
      fprintf(stderr, "could not open %s\n", outfile);
   }
   
   buf=(_prec* ) calloc(nel, sizeof(_prec));
   CUCHK(cudaMemcpy(buf, var, nel*sizeof(_prec), cudaMemcpyDeviceToHost));
   fwrite(buf, nel, sizeof(_prec), fid);
   fclose(fid);

   free(buf);
}

void dump_all_stresses(_prec *d_xx, _prec *d_yy, _prec *d_zz, _prec *d_xz, _prec *d_yz, _prec *d_xy, 
    long int nel, int desc, int tstep, int tsub, int rank, int ncpus){

   dump_variable(d_xx, nel, "xx", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_yy, nel, "yy", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_zz, nel, "zz", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_xz, nel, "xz", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_yz, nel, "yz", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_xy, nel, "xy", desc, tstep, tsub, rank, ncpus);
}

void dump_all_vels(_prec *d_u1, _prec *d_v1, _prec *d_w1, long int nel, int desc, int tstep, int tsub, int rank, int ncpus){

   dump_variable(d_u1, nel, "u1", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_v1, nel, "v1", desc, tstep, tsub, rank, ncpus);
   dump_variable(d_w1, nel, "w1", desc, tstep, tsub, rank, ncpus);
}

