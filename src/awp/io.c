#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <awp/pmcl3d.h>
#include <awp/utils.h>

int writeCHK(char *chkfile, int ntiskp, _prec dt, _prec *dh, 
      int *nxt, int *nyt, int *nzt,
      int nt, _prec arbc, int npc, int nve,
      _prec fac, _prec q0, _prec ex, _prec fp, 
      _prec **vse, _prec **vpe, _prec **dde, 
      int ngrids){

  FILE *fchk;
  int p;

  fchk = fopen(chkfile,"w");
  for (p=0; p<ngrids; p++){
     fprintf(fchk,"-------------------------------------------------------------\n");
     fprintf(fchk,"GRID NUMBER:\t%d\n",p);
     fprintf(fchk,"STABILITY CRITERIA .5 > CMAX*DT/DX:\t%f\n",vpe[p][1]*dt/dh[p]);
     fprintf(fchk,"# OF X,Y,Z NODES PER PROC:\t%d, %d, %d\n",nxt[p],nyt[p],nzt[p]);
     fprintf(fchk,"DISCRETIZATION IN SPACE:\t%f\n",dh[p]);
     fprintf(fchk,"HIGHEST P-VELOCITY ENCOUNTERED:\t%f\n",vpe[p][1]);
     fprintf(fchk,"LOWEST P-VELOCITY ENCOUNTERED:\t%f\n",vpe[p][0]);
     fprintf(fchk,"HIGHEST S-VELOCITY ENCOUNTERED:\t%f\n",vse[p][1]);
     fprintf(fchk,"LOWEST S-VELOCITY ENCOUNTERED:\t%f\n",vse[p][0]);
     fprintf(fchk,"HIGHEST DENSITY ENCOUNTERED:\t%f\n",dde[p][1]);
     fprintf(fchk,"LOWEST  DENSITY ENCOUNTERED:\t%f\n",dde[p][0]);
     fprintf(fchk,"-------------------------------------------------------------\n");
  }
  fprintf(fchk,"# OF TIME STEPS:\t%d\n",nt);
  fprintf(fchk,"DISCRETIZATION IN TIME:\t%f\n",dt);
  fprintf(fchk,"PML REFLECTION COEFFICIENT:\t%f\n",arbc);
  fprintf(fchk,"SKIP OF SEISMOGRAMS IN TIME (LOOP COUNTER):\t%d\n",ntiskp);
  fprintf(fchk,"ABC CONDITION, PML=1 OR CERJAN=0:\t%d\n",npc);
  fprintf(fchk,"FD SCHEME, ELASTIC=0,VISCO=1,ELASTO-PLASTIC=2,VISCO-ELASTO-PLASTIC=3:\t%d\n",nve);
  fprintf(fchk,"Q, FAC,Q0,EX,FP:\t%f, %f, %f, %f\n",fac,q0,ex,fp);
  fclose(fchk);

return 0;
}

void background_output_writer(int rank, int size, int nout, int wstep, int ntiskp, int ngrids, char *OUT,
   MPI_Comm MCI, int NVE){
   char filename[50];
   char filenamebasex[50];
   char filenamebasey[50];
   char filenamebasez[50];
   char filenamebaseeta[50];
   char filenamebaseep[50];

   int n, p;
   long int cur_step;
   time_t time1, time2;

   int rec_nxt[MAXGRIDS], rec_nyt[MAXGRIDS], rec_nzt[MAXGRIDS];
   int rec_NX[MAXGRIDS], rec_NY[MAXGRIDS], rec_NZ[MAXGRIDS];

   Grid1D *Bufx=NULL, *Bufy=NULL, *Bufz=NULL, *Bufeta=NULL;

   int i;
   MPI_Datatype filetype[MAXGRIDS];
   MPI_File fh;
   int maxNX_NY_NZ_WS, **ones;
   MPI_Aint **dispArray;
   MPI_Offset displacement[MAXGRIDS];
   int err;
   int tmpSize;
   MPI_Status filestatus;
   int master;

   int outsize[MAXGRIDS];
   int grid_output[MAXGRIDS];

   master = size*3-1;

   /*Initialize MPI Filetype */
   dispArray=(MPI_Aint**) calloc(ngrids, sizeof(MPI_Aint*));
   ones=(int**) calloc(ngrids, sizeof(int*));
   MPICHK(MPI_Recv(rec_nxt, ngrids, MPI_INT, rank-2*size, MPIRANKIO, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(rec_nyt, ngrids, MPI_INT, rank-2*size, MPIRANKIO+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(rec_nzt, ngrids, MPI_INT, rank-2*size, MPIRANKIO+2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(rec_NX, ngrids, MPI_INT, rank-2*size, MPIRANKIO+3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(rec_NY, ngrids, MPI_INT, rank-2*size, MPIRANKIO+4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(rec_NZ, ngrids, MPI_INT, rank-2*size, MPIRANKIO+5, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(grid_output, ngrids, MPI_INT, rank-2*size, MPIRANKIO+6, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
   MPICHK(MPI_Recv(displacement, ngrids, MPI_OFFSET, rank-2*size, MPIRANKIO+7, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

   fprintf(stdout, "I/O: ngrids=%d\n", ngrids);
   fprintf(stdout, "I/O: rec_nxt,rec_nyt,rec_nzt=%d,%d,%d\n", rec_nxt[0], rec_nyt[0], rec_nzt[0]);
   fprintf(stdout, "I/O: rec_NX,rec_NY,rec_NZ=%d,%d,%d\n", rec_NX[0], rec_NY[0], rec_NZ[0]);

   for (p=0; p<ngrids; p++){
       maxNX_NY_NZ_WS = (rec_NX[p]>rec_NY[p]?rec_NX[p]:rec_NY[p]);
       maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>rec_NZ[p]?maxNX_NY_NZ_WS:rec_NZ[p]);
       maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>wstep?maxNX_NY_NZ_WS:wstep);
       fprintf(stdout, "I/O: maxNX_NY_NZ_WS=%d\n", maxNX_NY_NZ_WS);
       ones[p]=(int*) calloc(maxNX_NY_NZ_WS, sizeof(int));
       for(i=0;i<maxNX_NY_NZ_WS;++i) ones[p][i] = 1;
       dispArray[p] = (MPI_Aint*) calloc(maxNX_NY_NZ_WS, sizeof(MPI_Aint));

       err = MPI_Type_contiguous(rec_nxt[p], _mpi_prec, &filetype[p]);
       err = MPI_Type_commit(&filetype[p]);
       for(i=0;i<rec_nyt[p];i++){
	 dispArray[p][i] = sizeof(_prec);
	 dispArray[p][i] = dispArray[p][i]*rec_NX[p]*i;
       }
       err = MPI_Type_create_hindexed(rec_nyt[p], ones[p], dispArray[p], filetype[p], &filetype[p]);
       err = MPI_Type_commit(&filetype[p]);
       for(i=0;i<rec_nzt[p];i++){
	 dispArray[p][i] = sizeof(_prec);
	 dispArray[p][i] = dispArray[p][i]*rec_NY[p]*rec_NX[p]*i;
       }
       err = MPI_Type_create_hindexed(rec_nzt[p], ones[p], dispArray[p], filetype[p], &filetype[p]);
       err = MPI_Type_commit(&filetype[p]);
       for(i=0;i<wstep;i++){
	 dispArray[p][i] = sizeof(_prec);
	 dispArray[p][i] = dispArray[p][i]*rec_NZ[p]*rec_NY[p]*rec_NX[p]*i;
       }
       err = MPI_Type_create_hindexed(wstep, ones[p], dispArray[p], filetype[p], &filetype[p]);
       err = MPI_Type_commit(&filetype[p]);
       MPICHK(MPI_Type_size(filetype[p], &tmpSize));
       if(rank==master) printf("filetype size grid %d (supposedly=rec_nxt*nyt*nzt*WS*4=%ld) =%d\n", 
          p, rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*wstep*sizeof(_prec),tmpSize);
   }

   sprintf(filenamebasex,"%s/SX",OUT);
   sprintf(filenamebasey,"%s/SY",OUT);
   sprintf(filenamebasez,"%s/SZ",OUT);
   sprintf(filenamebaseeta,"%s/Eta",OUT);
   if (NVE==3) sprintf(filenamebaseep,"%s/EP",OUT);

   for (p=0; p<ngrids; p++){
      if (grid_output[p]) outsize[p] = rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*wstep;
      else outsize[p]=0;
   }
   for (p=0; p<ngrids; p++) fprintf(stdout, "I/O %d displacement[%d] = %lld\n", rank, p, displacement[p]);

   Bufx = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
   Bufy = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
   Bufz = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
   Bufeta = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
   for (p=0; p<ngrids; p++){
      if (outsize[p] > 0){
	 Bufx[p]  = Alloc1D(outsize[p]);
	 Bufy[p]  = Alloc1D(outsize[p]);
	 Bufz[p]  = Alloc1D(outsize[p]);
	 if (NVE == 3) Bufeta[p] = Alloc1D(outsize[p]);
      }
   }

   for (n=0; n<nout; n++){
      cur_step = (n+1)*wstep*ntiskp;
      for (p=0; p<ngrids; p++){
         if (outsize[p] > 0){
            MPICHK(MPI_Recv(Bufx[p], outsize[p], _mpi_prec, rank-2*size, MPIRANKIO+30, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            MPICHK(MPI_Recv(Bufy[p], outsize[p], _mpi_prec, rank-2*size, MPIRANKIO+31, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            MPICHK(MPI_Recv(Bufz[p], outsize[p], _mpi_prec, rank-2*size, MPIRANKIO+32, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            if (NVE ==3) 
               MPICHK(MPI_Recv(Bufeta[p], outsize[p], _mpi_prec, rank-2*size, MPIRANKIO+33, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
         }
      }

      MPICHK(MPI_Barrier(MCI));
      time(&time1);

      for (p=0; p<ngrids; p++){
	 sprintf(filename, "%s_%1d_%07ld", filenamebasex, p, cur_step);
	 err = MPI_File_open(MCI,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
	 error_check(err, "MPI_File_open X");
	 err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
	 error_check(err, "MPI_File_set_view X");
	 err = MPI_File_write_all(fh, Bufx[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*wstep, _mpi_prec, &filestatus);
	 error_check(err, "MPI_File_write X");

	 err = MPI_File_close(&fh);
	 error_check(err, "MPI_File_close X");

	 sprintf(filename, "%s_%1d_%07ld", filenamebasey, p, cur_step);
	 err = MPI_File_open(MCI,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
	 err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
	 err = MPI_File_write_all(fh, Bufy[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*wstep, _mpi_prec, &filestatus);
	 err = MPI_File_close(&fh);
	 sprintf(filename, "%s_%1d_%07ld", filenamebasez, p, cur_step);
	 err = MPI_File_open(MCI,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
	 err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
	 err = MPI_File_write_all(fh, Bufz[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*wstep, _mpi_prec, &filestatus);
	 err = MPI_File_close(&fh);
	 //saves the plastic shear work
	 if (NVE == 3) {
	    sprintf(filename, "%s_%1d_%07ld", filenamebaseeta, p, cur_step);
	    err = MPI_File_open(MCI,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
	    err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
	    err = MPI_File_write_all(fh, Bufeta[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*wstep, _mpi_prec, &filestatus);
	    err = MPI_File_close(&fh);
	 }
      }
      MPICHK(MPI_Barrier(MCI));
      time(&time2);
      if (rank==master) fprintf(stdout, "Time for output (): %4.f seconds.\n", difftime(time2, time1));
   }
}


