#define _GNU_SOURCE
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <errno.h>
#include <awp/pmcl3d.h>

#define MPIRANKSRC 300000
#define FNAMELEN 400

void errhandle(int ierr, char *where);
void read_awp_subvolume(MPI_File fh, int *griddims, int *extent, int *timerange, 
    MPI_Comm comm, int seismio, _prec *buf);

int read_src_ifault_2(int rank, int READ_STEP, 
    char *INSRC, char *INSRC_I2, 
    int maxdim, int *coords, int NZ,
    int nxt, int nyt, int nzt,
    int *NPSRC, int *SRCPROC, 
    PosInf *psrc, Grid1D *axx, Grid1D *ayy, Grid1D *azz, 
    Grid1D *axz, Grid1D *ayz, Grid1D *axy,
    int idx){

  FILE *f;
  char fname[50];
  int dummy[2], i, j;
  int nbx, nby;
  PosInf tpsrc = NULL;
  Grid1D taxx=NULL, tayy=NULL, tazz=NULL; 
  Grid1D taxy=NULL, taxz=NULL, tayz=NULL;

  // First time entering this function
  if(idx == 1){
    sprintf(fname, "%s%07d", INSRC, rank);
    printf("SOURCE reading first time: %s\n",fname);
    f = fopen(fname, "rb");
    if(f == NULL){
      *SRCPROC = -1;
      *NPSRC = 0;
      return 0;
    }
    *SRCPROC = rank;
    /*changed from 2 to 6, as we have 8 ghost cell layers - Daniel */
    nbx     = nxt*coords[0] + 1 - 2*loop; 
    nby     = nyt*coords[1] + 1 - 2*loop;
    // not sure what happens if maxdim != 3
    fread(NPSRC, sizeof(int), 1, f);
    fread(dummy, sizeof(int), 2, f);

    printf("SOURCE I am, rank=%d npsrc=%d\n",rank,*NPSRC);    

    tpsrc = Alloc1P((*NPSRC)*maxdim);
    fread(tpsrc, sizeof(int), (*NPSRC)*maxdim, f);
    // assuming nzt=NZ
    for(i=0; i<*NPSRC; i++){
      //tpsrc[i*maxdim] = (tpsrc[i*maxdim]-1)%nxt+1;
      //tpsrc[i*maxdim+1] = (tpsrc[i*maxdim+1]-1)%nyt+1;
      //tpsrc[i*maxdim+2] = NZ+1 - tpsrc[i*maxdim+2];
      tpsrc[i*maxdim]   = tpsrc[i*maxdim] - nbx - 1;
      tpsrc[i*maxdim+1] = tpsrc[i*maxdim+1] - nby - 1;
      tpsrc[i*maxdim+2] = NZ + 1 - tpsrc[i*maxdim+2];
    }
    *psrc = tpsrc;
    fclose(f);
  }
  if(*NPSRC > 0){
    sprintf(fname, "%s%07d_%03d", INSRC_I2, rank, idx);
    printf("SOURCE reading: %s\n",fname);
    f = fopen(fname, "rb");
    if(f == NULL){
      printf("ERROR! Rank=%d: Cannot open file: %s\n", rank, fname);
      return -1;
    }
    // fastest axis in partitioned source is npsrc, then read_step (see source.f)
    // fastest axis in axx is time
    Grid1D tmpta;
    tmpta = Alloc1D((*NPSRC)*READ_STEP);
    if(idx==1){
      taxx = Alloc1D((*NPSRC)*READ_STEP);
      tayy = Alloc1D((*NPSRC)*READ_STEP);
      tazz = Alloc1D((*NPSRC)*READ_STEP);
      taxy = Alloc1D((*NPSRC)*READ_STEP);
      taxz = Alloc1D((*NPSRC)*READ_STEP);
      tayz = Alloc1D((*NPSRC)*READ_STEP);
    }
    else{
      taxx = *axx;
      tayy = *ayy;
      tazz = *azz;
      taxy = *axy;
      taxz = *axz;
      tayz = *ayz;
    }
    fread(tmpta, sizeof(_prec), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        taxx[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(_prec), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        tayy[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(_prec), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        tazz[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(_prec), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        taxz[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(_prec), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        tayz[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fread(tmpta, sizeof(_prec), (*NPSRC)*READ_STEP, f);
    for(i=0; i<*NPSRC; i++)
      for(j=0; j<READ_STEP; j++)
        taxy[i*READ_STEP+j] = tmpta[j*(*NPSRC)+i];
    fclose(f);
    *axx = taxx;
    *ayy = tayy;
    *azz = tazz;
    *axz = taxz;
    *ayz = tayz;
    *axy = taxy;
    Delloc1D(tmpta);
  }

return 0;
}

int inisource(int      rank,    int     IFAULT, int     NSRC,   int     READ_STEP, int     NST,     int     *SRCPROC, int    NZ, 
              MPI_Comm MCW,     int     nxt,    int     nyt,    int     nzt,       int     *coords, int     maxdim,   int    *NPSRC,
              PosInf   *ptpsrc, Grid1D  *ptaxx, Grid1D  *ptayy, Grid1D  *ptazz,    Grid1D  *ptaxz,  Grid1D  *ptayz,   Grid1D *ptaxy, char *INSRC, char *INSRC_I2)
{
   int i, j, k, npsrc, srcproc, master=0;
   int nbx, nex, nby, ney, nbz, nez;
   PosInf tpsrc=NULL, tpsrcp =NULL;
   Grid1D taxx =NULL, tayy   =NULL, tazz =NULL, taxz =NULL, tayz =NULL, taxy =NULL;
   Grid1D taxxp=NULL, tayyp  =NULL, tazzp=NULL, taxzp=NULL, tayzp=NULL, taxyp=NULL;
   if(NSRC<1) return 0;

   npsrc   = 0;
   srcproc = -1;
// Indexing is based on 1: [1, nxt], etc. Include 1st layer ghost cells
   nbx     = nxt*coords[0] + 1 - 2;
   nex     = nbx + nxt + 4*loop - 1;
   nby     = nyt*coords[1] + 1 - 2;
   ney     = nby + nyt + 4*loop - 1;
   nbz     = 1;
   nez     = nzt;
   // IFAULT=1 has bug! READ_STEP does not work, it tries to read NST all at once - Efe
   if(IFAULT==1 || IFAULT==5)
   {
      tpsrc = Alloc1P(NSRC*maxdim);
      taxx  = Alloc1D(NSRC*READ_STEP);
      tayy  = Alloc1D(NSRC*READ_STEP);
      tazz  = Alloc1D(NSRC*READ_STEP);
      taxz  = Alloc1D(NSRC*READ_STEP);
      tayz  = Alloc1D(NSRC*READ_STEP);
      taxy  = Alloc1D(NSRC*READ_STEP);

      if(rank==master)
      {
      	 FILE   *file;
         int    tmpsrc[3];
         Grid1D tmpta = Alloc1D(NST*6);
         file = fopen(INSRC,"rb");
         if(!file)
         {
            printf("can't open file %s", INSRC);
	    return 0;
         }
         for(i=0;i<NSRC;i++)
         { 
            if(fread(tmpsrc,sizeof(int),3,file) && fread(tmpta,sizeof(_prec),NST*6,file))
            {
               tpsrc[i*maxdim]   = tmpsrc[0];
               tpsrc[i*maxdim+1] = tmpsrc[1];
               tpsrc[i*maxdim+2] = nzt+1-tmpsrc[2];
               for(j=0;j<READ_STEP;j++)
               {
                  taxx[i*READ_STEP+j] = tmpta[j*6];
                  tayy[i*READ_STEP+j] = tmpta[j*6+1];
                  tazz[i*READ_STEP+j] = tmpta[j*6+2];
                  taxz[i*READ_STEP+j] = tmpta[j*6+3];
                  tayz[i*READ_STEP+j] = tmpta[j*6+4];
                  taxy[i*READ_STEP+j] = tmpta[j*6+5];
               }
            }
         }
         Delloc1D(tmpta);
         fclose(file);
      }
      MPICHK(MPI_Bcast(tpsrc, NSRC*maxdim,    MPI_INT,  master, MCW));
      MPICHK(MPI_Bcast(taxx,  NSRC*READ_STEP, MPI_FLOAT, master, MCW)); 
      MPICHK(MPI_Bcast(tayy,  NSRC*READ_STEP, MPI_FLOAT, master, MCW));
      MPICHK(MPI_Bcast(tazz,  NSRC*READ_STEP, MPI_FLOAT, master, MCW));
      MPICHK(MPI_Bcast(taxz,  NSRC*READ_STEP, MPI_FLOAT, master, MCW));
      MPICHK(MPI_Bcast(tayz,  NSRC*READ_STEP, MPI_FLOAT, master, MCW));
      MPICHK(MPI_Bcast(taxy,  NSRC*READ_STEP, MPI_FLOAT, master, MCW));
      for(i=0;i<NSRC;i++)
      {
          if( tpsrc[i*maxdim]   >= nbx && tpsrc[i*maxdim]   <= nex && tpsrc[i*maxdim+1] >= nby 
           && tpsrc[i*maxdim+1] <= ney && tpsrc[i*maxdim+2] >= nbz && tpsrc[i*maxdim+2] <= nez)
          {
              srcproc = rank;
              npsrc ++;
          }
      }
      if(npsrc > 0)
      {
          tpsrcp = Alloc1P(npsrc*maxdim);
          taxxp  = Alloc1D(npsrc*READ_STEP);
          tayyp  = Alloc1D(npsrc*READ_STEP);
          tazzp  = Alloc1D(npsrc*READ_STEP);
          taxzp  = Alloc1D(npsrc*READ_STEP);
          tayzp  = Alloc1D(npsrc*READ_STEP);
          taxyp  = Alloc1D(npsrc*READ_STEP);
          k      = 0;
          for(i=0;i<NSRC;i++)
          {
              if( tpsrc[i*maxdim]   >= nbx && tpsrc[i*maxdim]   <= nex && tpsrc[i*maxdim+1] >= nby
               && tpsrc[i*maxdim+1] <= ney && tpsrc[i*maxdim+2] >= nbz && tpsrc[i*maxdim+2] <= nez)
               {
                 tpsrcp[k*maxdim]   = tpsrc[i*maxdim]   - nbx - 1;
                 tpsrcp[k*maxdim+1] = tpsrc[i*maxdim+1] - nby - 1;
                 tpsrcp[k*maxdim+2] = tpsrc[i*maxdim+2] - nbz + 1;
                 for(j=0;j<READ_STEP;j++)
                 {
                    taxxp[k*READ_STEP+j] = taxx[i*READ_STEP+j];
                    tayyp[k*READ_STEP+j] = tayy[i*READ_STEP+j];
                    tazzp[k*READ_STEP+j] = tazz[i*READ_STEP+j];
                    taxzp[k*READ_STEP+j] = taxz[i*READ_STEP+j];
                    tayzp[k*READ_STEP+j] = tayz[i*READ_STEP+j];
                    taxyp[k*READ_STEP+j] = taxy[i*READ_STEP+j];
                  }
                  k++;
               }
          }
      }
      Delloc1D(taxx);
      Delloc1D(tayy);
      Delloc1D(tazz);
      Delloc1D(taxz);
      Delloc1D(tayz);
      Delloc1D(taxy); 
      Delloc1P(tpsrc);       

      *SRCPROC = srcproc;
      *NPSRC   = npsrc;
      *ptpsrc  = tpsrcp;
      *ptaxx   = taxxp;
      *ptayy   = tayyp;
      *ptazz   = tazzp;
      *ptaxz   = taxzp;
      *ptayz   = tayzp;
      *ptaxy   = taxyp;
   }
   else if(IFAULT == 2){
      return read_src_ifault_2(rank, READ_STEP,
        INSRC, INSRC_I2,
        maxdim, coords, nzt,
        nxt, nyt, nzt,
        NPSRC, SRCPROC,
        ptpsrc, ptaxx, ptayy, ptazz,
        ptaxz, ptayz, ptaxy, 1);
   }
   return 0;
}

void addsrc(int i,      _prec DH,   _prec DT,   int NST,    int npsrc,  int READ_STEP, int dim, PosInf psrc,
            Grid1D axx, Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
            Grid3D xx,  Grid3D yy,  Grid3D zz,  Grid3D xy,  Grid3D yz,  Grid3D xz)
{
  _prec vtst;
  int idx, idy, idz, j;
  vtst = (_prec)DT/(DH*DH*DH);
#if VERBOSE
  fprintf(stdout, "vtst=%.15g\n", vtst);
#endif

  i   = i - 1;
  for(j=0;j<npsrc;j++) 
  {
     idx = psrc[j*dim]   + 1 + ngsl;
     idy = psrc[j*dim+1] + 1 + ngsl;
     idz = psrc[j*dim+2] + align - 1;
#if VERBOSE
     fprintf(stdout, "adding source %e to %d,%d,%d\n", vtst*axx[j*READ_STEP+i], idx, idy, idz);
#endif
     xx[idx][idy][idz] = xx[idx][idy][idz] - vtst*axx[j*READ_STEP+i];
     yy[idx][idy][idz] = yy[idx][idy][idz] - vtst*ayy[j*READ_STEP+i];
     zz[idx][idy][idz] = zz[idx][idy][idz] - vtst*azz[j*READ_STEP+i];
     xz[idx][idy][idz] = xz[idx][idy][idz] - vtst*axz[j*READ_STEP+i];
     yz[idx][idy][idz] = yz[idx][idy][idz] - vtst*ayz[j*READ_STEP+i];
     xy[idx][idy][idz] = xy[idx][idy][idz] - vtst*axy[j*READ_STEP+i];
/*
     printf("xx=%1.6g\n",xx[idx][idy][idz]);
     printf("yy=%1.6g\n",yy[idx][idy][idz]);
     printf("zz=%1.6g\n",zz[idx][idy][idz]);
     printf("xz=%1.6g\n",xz[idx][idy][idz]);
     printf("yz=%1.6g\n",yz[idx][idy][idz]);
     printf("xy=%1.6g\n",xy[idx][idy][idz]);
*/
  }
  return;
}

/* this routine interpolates between adjacent values */
void frcvel(int i,      _prec DH,   _prec DT,   int NST,    int npsrc,  int READ_STEP, int tskp, int dim, PosInf psrc,
            Grid1D axx, Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
            Grid3D u1,  Grid3D v1,  Grid3D w1, int rank)
{
  int idx, idy, idz, j;
  int i0, i1;
  _prec u1_p, u1_n, v1_p, v1_n, w1_p, w1_n;
  _prec pfact;

  i   = i - 1;

  i0=(int) tskp * floorf(((_prec) i + 1.) / (_prec) tskp);
  i1=(int) tskp * ceilf(((_prec) i + 1.) / (_prec) tskp);

  pfact = (_prec) (i - i0) / (_prec) tskp;

  fprintf(stdout, "i=%d, i0=%d, i1=%d\n", i, i0, i1);

  i0 /= tskp;
  i1 /= tskp;

  for(j=0;j<npsrc;j++) 
  {
     idx = psrc[j*dim]   + 1 + ngsl;
     idy = psrc[j*dim+1] + 1 + ngsl;
     idz = psrc[j*dim+2] + align - 1;

     u1_p = axx[i0*npsrc+j];
     v1_p = ayy[i0*npsrc+j];
     w1_p = azz[i0*npsrc+j];

     u1_n = axx[i1*npsrc+j];
     v1_n = ayy[i1*npsrc+j];
     w1_n = azz[i1*npsrc+j];

     /* Linear interpolation */
     u1[idx][idy][idz] = u1_p + (u1_n - u1_p) * pfact;
     v1[idx][idy][idz] = v1_p + (v1_n - v1_p) * pfact;
     w1[idx][idy][idz] = w1_p + (w1_n - w1_p) * pfact;
  }
  return;
}

int max(int a, int b){
   return (a > b ? a: b);
}

int min(int a, int b){
   return (a < b ? a: b);
}

void errhandle(int ierr, char *where){
   int errlen;
   char *errstr;

   if (ierr != 0) {
      fprintf(stderr, "error in %s\n", where);
      errstr=calloc(500, sizeof(char));
      MPICHK(MPI_Error_string(ierr, errstr, &errlen));
      fprintf(stderr, "%s", errstr);
      MPICHK(MPI_Finalize());
   }
}


/* read time segment ouf of subvolume of AWP output.
 *  * Note: here, indices start at 0 */
void read_awp_subvolume(MPI_File fh, int *griddims, int *extent, int *timerange, 
    MPI_Comm comm, int seismio, _prec *buf){
    int nx, ny, nz;
    int i0, i1, j0, j1, k0, k1;
    int t0, t1, nt;

    int nx1, ny1, nz1, p=0; /* dimension of subvolume being read */
    size_t np, npt, nel;
    int t, j, k, kp;

    int ierr;
    MPI_Datatype filetype;
    MPI_Offset disp=0;
    MPI_Aint *map, nskp, toff;
    int *blen;

    nx=griddims[0];
    ny=griddims[1];
    nz=griddims[2];

    i0=extent[0];
    i1=extent[1];
    j0=extent[2];
    j1=extent[3];
    k0=extent[4];
    k1=extent[5];

    t0=timerange[0];
    t1=timerange[1];
    nt=timerange[1] - t0;

    nx1=i1 - i0;
    ny1=j1 - j0;
    nz1=k1 - k0;

    np = (size_t) nt * ny1 * nz1;
    nel = (size_t) nx * ny * nz;
    npt = nx1 * np;
    blen=(int*) calloc(np, sizeof(int));
    map=(MPI_Aint*) calloc (np, sizeof(MPI_Aint));

    for (t=t0; t<t1; t++){
       toff = (MPI_Aint) nel * (MPI_Aint) sizeof(_mpi_prec) * (MPI_Aint) t;
       
       if (seismio) { /* order in files is from bottom of mesh to top */
	  for (k=k0; k<k1; k++){
	    for (j=j0; j<j1; j++){
	       blen[p] = nx1; 
	       nskp = (MPI_Aint) k * nx * ny + (MPI_Aint) j*nx + (MPI_Aint) i0;
	       map[p] = nskp * (MPI_Aint) sizeof(_mpi_prec) + toff;
	       if (map[p] < 0) fprintf(stderr, "Error.  Displacement %ld < 0\n", map[p]);
	       p++;
	    }
	  }
       }
       else { /* order is from top to bottom */
    	  for (k=(k1-1); k>=k0; k--){
	    for (j=j0; j<j1; j++){
	       blen[p] = nx1; 
               kp = nz - k - 1;
	       nskp = (MPI_Aint) kp * nx * ny + (MPI_Aint) j*nx + (MPI_Aint) i0;
	       map[p] = nskp * (MPI_Aint) sizeof(_mpi_prec) + toff;
	       if (map[p] < 0) fprintf(stderr, "Error.  Displacement %ld < 0\n", map[p]);
	       p++;
	    }
	  }
       }
    }
    ierr=MPI_Type_create_hindexed(np, blen, map, _mpi_prec, &filetype);
    errhandle(ierr, "MPI_Type_create_indexed_block");

    ierr=MPI_Type_commit(&filetype);
    errhandle(ierr, "MPI_Type_commit");

    ierr=MPI_File_set_view(fh, disp, _mpi_prec, filetype, "native", MPI_INFO_NULL);
    errhandle(ierr, "MPI_File_set_view");

    ierr=MPI_File_read_all(fh, buf, npt, _mpi_prec, MPI_STATUS_IGNORE);
    errhandle(ierr, "MPI_File_read_all");

    ierr=MPI_Type_free(&filetype);
    errhandle(ierr, "MPI_Type_free");

    free(blen);
    free(map);
}

/* this is for fault boundary condition.  Velocities in pre-determined
 * zone are prescriped from output of previous dynamic simulation 
 *
 * in this version, reading is done by separate MPI process
 * Daniel Roten, March 2016 */
int read_src_ifault_4 (int rank, int READ_STEP, char *INSRC, 
    int maxdim, int *coords, int NZ, int nxt, int nyt, int nzt,
    int *NPSRC, int *SRCPROC, PosInf *psrc, Grid1D *axx, Grid1D *ayy, Grid1D *azz, 
    int idx, int *fbc_ext, int *fbc_off, char *fbc_pmask, int *fbc_extl, int *fbc_dim,
    int *fbc_seismio, int *fbc_tskp, int nst, int size) {

    int npsrc;
    int i, j, k, l;
    int nbx, nby;
    int ierr;

    int eastm, westm, southm, northm;
    PosInf tpsrc = NULL;
    Grid1D taxx=NULL, tayy=NULL, tazz=NULL; 
    char xfile[FNAMELEN], yfile[FNAMELEN], zfile[FNAMELEN];

    time_t time1, time2;

    if (idx == 1) { /* initialization during first call */
       if (rank==0) {
	  FILE *fid;
	  size_t linecap;
	  char *nline=NULL;

	  fid=fopen(INSRC, "r");
	  if (fid == NULL) {
	     perror("could not open INSRC file");
	     return(-1); 
	  }

	  /* extent of volume where velocities are prescribed. X, */
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d %d", fbc_ext, fbc_ext+1);

	  /* Y, */
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d %d", fbc_ext+2, fbc_ext+3);

	  /* and Z direction.*/
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d %d", fbc_ext+4, fbc_ext+5);

	  /* dimensions of dynamic grid output */
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d %d %d", fbc_dim, fbc_dim+1, fbc_dim+2);

	  /* offset of these coordinates, with respect to dynamic grid */
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d %d %d\n", fbc_off, fbc_off+1, fbc_off+2);

	  /* directory and filename of previous output, e.g. "../dyn/output_sfc/S%c" */
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%s\n", fbc_pmask);
          fprintf(stdout, "fbc_pmask=\"%s\"\n", fbc_pmask);

	  /* flag controlling if seismio was used in dynamic run */
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d\n", fbc_seismio);
          fprintf(stdout, "fbc_seismio=%d\n", (*fbc_seismio));

	  /* integer with ntiskp used for saving output*/
	  getline(&nline, &linecap, fid);
	  sscanf(nline, "%d\n", fbc_tskp);
          fprintf(stdout, "fbc_tskp=%d\n", (*fbc_tskp));

	  fclose(fid);
       }

       MPICHK(MPI_Bcast(fbc_ext, 6, MPI_INT, 0, MPI_COMM_WORLD));
       MPICHK(MPI_Bcast(fbc_off, 3, MPI_INT, 0, MPI_COMM_WORLD));
       MPICHK(MPI_Bcast(fbc_pmask, 100, MPI_CHAR, 0, MPI_COMM_WORLD));
       MPICHK(MPI_Bcast(fbc_dim, 3, MPI_INT, 0, MPI_COMM_WORLD));
       MPICHK(MPI_Bcast(fbc_seismio, 1, MPI_INT, 0, MPI_COMM_WORLD));
       MPICHK(MPI_Bcast(fbc_tskp, 1, MPI_INT, 0, MPI_COMM_WORLD));

       nbx     = nxt*coords[0] + 1 - 2*loop; 
       nby     = nyt*coords[1] + 1 - 2*loop;

       westm=1-ngsl;
       eastm=nxt+ngsl;
       southm=1-ngsl;
       northm=nyt+ngsl;

       fbc_extl[0] = max(fbc_ext[0] - nbx - 1, westm);
       fbc_extl[1] = min(fbc_ext[1] - nbx - 1, eastm);

       fbc_extl[2] = max(fbc_ext[2] - nby - 1, southm);
       fbc_extl[3] = min(fbc_ext[3] - nby - 1, northm);
       fbc_extl[4] = fbc_ext[4];
       fbc_extl[5] = fbc_ext[5];

       if ((fbc_extl[0] < eastm) && (fbc_extl[1] > westm) &&
	  (fbc_extl[2] < northm) && (fbc_extl[3] > southm)){
	 npsrc= (fbc_extl[1] - fbc_extl[0] + 1) * 
		(fbc_extl[3] - fbc_extl[2] + 1) * 
		(fbc_extl[5] - fbc_extl[4] + 1);
       }
       else {
	 npsrc=-1;
       }

       if (npsrc > 0){
	 tpsrc = Alloc1P(npsrc*maxdim);

	 l=0;
         /*when SEISM-IO is used, output is saved from bottom to top*/
	 if ((*fbc_seismio)) {
	    for (k=fbc_extl[5]; k>=fbc_extl[4]; k--){
	       for (j=fbc_extl[2]; j<=fbc_extl[3]; j++){
		  for (i=fbc_extl[0]; i<=fbc_extl[1]; i++){
		      tpsrc[l*maxdim]   = i;
		      tpsrc[l*maxdim+1] = j;
		      tpsrc[l*maxdim+2] = NZ - k + 1;
		      l++;
		  }
	       }
	    }
	 }
         else { /*otherwise from top to bottom */
	    for (k=fbc_extl[4]; k<=fbc_extl[5]; k++) {
       	       for (j=fbc_extl[2]; j<=fbc_extl[3]; j++){
		  for (i=fbc_extl[0]; i<=fbc_extl[1]; i++){
		      tpsrc[l*maxdim]   = i;
		      tpsrc[l*maxdim+1] = j;
		      tpsrc[l*maxdim+2] = NZ - k + 1;
		      l++;
		  }
	       }
	    }
         }
	 if (l != npsrc) fprintf(stdout, "Error on %d: l=%d, npsrc=%d\n", rank, l, npsrc);
	 *SRCPROC = rank;
	 *NPSRC = npsrc;
	 *psrc = tpsrc;
         
	 fprintf(stdout, "(%d) npsrc=%d\n", rank, npsrc);
	 fprintf(stdout, "(%d) fbc_extl=%d to %d, %d to %d, %d to %d\n", 
	       rank, fbc_extl[0], fbc_extl[1], fbc_extl[2], fbc_extl[3], 
	       fbc_extl[4], fbc_extl[5]);
	 fprintf(stdout, "(%d) fbc_dim=%d,%d,%d\n", rank, fbc_dim[0], fbc_dim[1], fbc_dim[2]);

         /* Allocating arrays */
         /* here, array will also hold last value from previous read operations, size is READ_STEP + 1 */
         taxx = (Grid1D) Alloc1P(npsrc*(READ_STEP+1));
         tayy = (Grid1D) Alloc1P(npsrc*(READ_STEP+1));
         tazz = (Grid1D) Alloc1P(npsrc*(READ_STEP+1));

         (*axx) = taxx;
         (*ayy) = tayy;
         (*azz) = tazz;

       }
       else {
	 *SRCPROC = -1;
	 *NPSRC = 0;
	 fprintf(stdout, "(%d) npsrc=%d\n", rank, npsrc);
       }

       sprintf(xfile, fbc_pmask, 'X');
       fprintf(stdout, "(%d) xfile=%s\n", rank, xfile);
       sprintf(yfile, fbc_pmask, 'Y');
       sprintf(zfile, fbc_pmask, 'Z');

       ierr=MPI_Send(&npsrc, 1, MPI_INT, rank+size, MPIRANKSRC, MPI_COMM_WORLD);
       errhandle(ierr, "MPI_Send 0");
       ierr=MPI_Send(fbc_extl, 6, MPI_INT, rank+size, MPIRANKSRC+1, MPI_COMM_WORLD);
       errhandle(ierr, "MPI_Send 1");
       ierr=MPI_Send(xfile, FNAMELEN, MPI_CHAR, rank+size, MPIRANKSRC+2, MPI_COMM_WORLD);
       errhandle(ierr, "MPI_Send 2");
       ierr=MPI_Send(yfile, FNAMELEN, MPI_CHAR, rank+size, MPIRANKSRC+3, MPI_COMM_WORLD);
       errhandle(ierr, "MPI_Send 3");
       ierr=MPI_Send(zfile, FNAMELEN, MPI_CHAR, rank+size, MPIRANKSRC+4, MPI_COMM_WORLD);
       errhandle(ierr, "MPI_Send 4");

       ierr=MPI_Send(&nxt, 1, MPI_INT, rank+size, MPIRANKSRC+5, MPI_COMM_WORLD);
       ierr=MPI_Send(&nyt, 1, MPI_INT, rank+size, MPIRANKSRC+6, MPI_COMM_WORLD);
       ierr=MPI_Send(&nzt, 1, MPI_INT, rank+size, MPIRANKSRC+7, MPI_COMM_WORLD);
       ierr=MPI_Send(coords, 3, MPI_INT, rank+size, MPIRANKSRC+8, MPI_COMM_WORLD);

    } /* end of initialization */

    if ((*NPSRC) > 0){
      taxx = *axx;
      tayy = *ayy;
      tazz = *azz;

      npsrc = *NPSRC;

      /* copy last npsrc velocities to beginning of arrays, or set to zero */
      if (idx == 1) {
	 for (l=0; l<npsrc; l++){
	    taxx[l] = 0.;
	    tayy[l] = 0.;
	    tazz[l] = 0.;
	 }
      }
      else { 
	 for (l=0; l<npsrc; l++){
	    taxx[l] = taxx[npsrc*READ_STEP+l];
	    tayy[l] = tayy[npsrc*READ_STEP+l];
	    tazz[l] = tazz[npsrc*READ_STEP+l];
	 }
      }

      time(&time1);
      MPICHK(MPI_Recv(taxx+npsrc, npsrc*READ_STEP, _mpi_prec, rank+size, MPIRANKSRC+100, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      MPICHK(MPI_Recv(tayy+npsrc, npsrc*READ_STEP, _mpi_prec, rank+size, MPIRANKSRC+101, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      MPICHK(MPI_Recv(tazz+npsrc, npsrc*READ_STEP, _mpi_prec, rank+size, MPIRANKSRC+102, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
      time(&time2);
      fprintf(stdout, "(%d) Time for MPI_Recv(): %4.f seconds.\n", rank, difftime(time2, time1));

      (*axx) = taxx;
      (*ayy) = tayy;
      (*azz) = tazz;

    }

    return(0);
}

int background_velocity_reader(int rank, int size, int NST, int READ_STEP, MPI_Comm MCS){
   int fbc_ext[6], fbc_off[3], fbc_pmask[100], fbc_dim[3];
   int npsrc, fbc_extl[6], fbc_extg[6], fbc_seismio=0;
   int coords[2];
   char xfile[FNAMELEN], yfile[FNAMELEN], zfile[FNAMELEN];
   int nxt, nyt, nzt;
   int ierr;
   MPI_File srcfh[3];
   int color=0;
   MPI_Comm fbc_comm;
   int idx, trange[2];
   Grid1D taxx=NULL, tayy=NULL, tazz=NULL; 
   time_t time1, time2;
   int fbc_tskp;

   MPICHK(MPI_Bcast(fbc_ext, 6, MPI_INT, 0, MPI_COMM_WORLD));
   MPICHK(MPI_Bcast(fbc_off, 3, MPI_INT, 0, MPI_COMM_WORLD));
   MPICHK(MPI_Bcast(fbc_pmask, 100, MPI_CHAR, 0, MPI_COMM_WORLD));
   MPICHK(MPI_Bcast(fbc_dim, 3, MPI_INT, 0, MPI_COMM_WORLD));
   MPICHK(MPI_Bcast(&fbc_seismio, 1, MPI_INT, 0, MPI_COMM_WORLD));
   MPICHK(MPI_Bcast(&fbc_tskp, 1, MPI_INT, 0, MPI_COMM_WORLD));

   fprintf(stdout, "(%d) fbc_off=%d, %d, %d\n", rank, fbc_off[0], fbc_off[1], fbc_off[2]);

   ierr=MPI_Recv(&npsrc, 1, MPI_INT, rank-size, MPIRANKSRC, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   errhandle(ierr, "MPI_Recv 0");
   ierr=MPI_Recv(fbc_extl, 6, MPI_INT, rank-size, MPIRANKSRC+1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   errhandle(ierr, "MPI_Recv 1");
   ierr=MPI_Recv(xfile, FNAMELEN, MPI_CHAR, rank-size, MPIRANKSRC+2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   errhandle(ierr, "MPI_Recv 2");
   ierr=MPI_Recv(yfile, FNAMELEN, MPI_CHAR, rank-size, MPIRANKSRC+3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   errhandle(ierr, "MPI_Recv 3");
   ierr=MPI_Recv(zfile, FNAMELEN, MPI_CHAR, rank-size, MPIRANKSRC+4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   errhandle(ierr, "MPI_Recv 4");

   ierr=MPI_Recv(&nxt, 1, MPI_INT, rank-size, MPIRANKSRC+5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   ierr=MPI_Recv(&nyt, 1, MPI_INT, rank-size, MPIRANKSRC+6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   ierr=MPI_Recv(&nzt, 1, MPI_INT, rank-size, MPIRANKSRC+7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   ierr=MPI_Recv(coords, 3, MPI_INT, rank-size, MPIRANKSRC+8, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

   if (npsrc > 0) color=1;
   MPICHK(MPI_Comm_split(MCS, color, 1, &fbc_comm));

   if (npsrc > 0){

      taxx = Alloc1D(npsrc*READ_STEP);
      tayy = Alloc1D(npsrc*READ_STEP);
      tazz = Alloc1D(npsrc*READ_STEP);

      fprintf(stdout, "(%d) xfile=%s\n", rank, xfile);

      /* convert local grid extent to extent of dynamic grid */
      fbc_extg[0] = fbc_extl[0] + coords[0]*nxt + fbc_off[0] - 1;
      fbc_extg[1] = fbc_extl[1] + coords[0]*nxt + fbc_off[0];
      /* need to add 1, as the last point is not included otherwise */

      fbc_extg[2] = fbc_extl[2] + coords[1]*nyt + fbc_off[1] - 1;
      fbc_extg[3] = fbc_extl[3] + coords[1]*nyt + fbc_off[1];

      fbc_extg[4] = fbc_extl[4] + fbc_off[2] - 1;
      fbc_extg[5] = fbc_extl[5] + fbc_off[2];

      fprintf(stdout, "(%d) fbc_extg = %d to %d, %d to %d, %d to %d\n", rank,
	 fbc_extg[0], fbc_extg[1], fbc_extg[2], fbc_extg[3], fbc_extg[4], fbc_extg[5]);

      ierr=MPI_File_open(fbc_comm, xfile, MPI_MODE_RDONLY, MPI_INFO_NULL, srcfh);
      errhandle(ierr, "MPI_File_open X");
      ierr=MPI_File_open(fbc_comm, yfile, MPI_MODE_RDONLY, MPI_INFO_NULL, srcfh+1);
      errhandle(ierr, "MPI_File_open Y");
      ierr=MPI_File_open(fbc_comm, zfile, MPI_MODE_RDONLY, MPI_INFO_NULL, srcfh+2);
      errhandle(ierr, "MPI_File_open Z");

      fprintf(stdout, "entering loop: NST=%d, tskp=%d, read_step=%d\n", NST, fbc_tskp, READ_STEP);
      //for (idx=0; idx<(NST/fbc_tskp); idx+=(READ_STEP/fbc_tskp)){
      for (idx=0; idx<NST; idx+=READ_STEP){

	 /*trange[0] = idx / tskp;
	 trange[1] = (idx/tskp)+READ_STEP;*/
	 trange[0] = idx;
	 //trange[1] = idx+READ_STEP/fbc_tskp;
	 trange[1] = idx+READ_STEP;

	 fprintf(stdout, "(%d) trange = %d to %d\n", rank, trange[0], trange[1]);

         time(&time1);
         read_awp_subvolume(srcfh[0], fbc_dim, fbc_extg, trange, fbc_comm, fbc_seismio, taxx);
         read_awp_subvolume(srcfh[1], fbc_dim, fbc_extg, trange, fbc_comm, fbc_seismio, tayy);
         read_awp_subvolume(srcfh[2], fbc_dim, fbc_extg, trange, fbc_comm, fbc_seismio, tazz);
         time(&time2);
         fprintf(stdout, "(%d) Time for reading(): %4.f seconds.\n", rank, difftime(time2, time1));

         MPICHK(MPI_Send(taxx, npsrc*READ_STEP, _mpi_prec, rank-size, MPIRANKSRC+100, MPI_COMM_WORLD));
         MPICHK(MPI_Send(tayy, npsrc*READ_STEP, _mpi_prec, rank-size, MPIRANKSRC+101, MPI_COMM_WORLD));
         MPICHK(MPI_Send(tazz, npsrc*READ_STEP, _mpi_prec, rank-size, MPIRANKSRC+102, MPI_COMM_WORLD));

         MPICHK(MPI_Barrier(fbc_comm));
     
      }
      fprintf(stdout, "done.\n");
   }


   return(0);

}

/* This routine initializes plane wave input.  It just reads the 3-C time series from an ASCII file
 * names INSRc and saves it in the variables taxx, tayy, taxx for the X, Y- and Z velocity, respectively */
int ini_plane_wave(int rank, MPI_Comm MCW, char *INSRC, int NST, Grid1D* taxx, Grid1D* tayy, Grid1D* tazz){
   Grid1D velx=NULL, vely=NULL, velz=NULL;
   FILE *fid;
   char errmsg[100];
   int i;
   int err=0, nread; 
  
   velx  = Alloc1D(NST);
   vely  = Alloc1D(NST);
   velz  = Alloc1D(NST);

   if (rank==0) {
      fid=fopen(INSRC, "r");
      if (fid==NULL){
         sprintf(errmsg, "Could not open %s", INSRC);
         perror(errmsg);
         err=1;
      } 
      for (i=0; i<NST; i++) {
         nread=fscanf(fid, "%e %e %e\n", velx+i, vely+i, velz+i);
         if (nread < 3) err=2;
      }
      fclose(fid);
   }
  
   MPICHK(MPI_Bcast(velx, NST, MPI_FLOAT, 0, MCW));
   MPICHK(MPI_Bcast(vely, NST, MPI_FLOAT, 0, MCW));
   MPICHK(MPI_Bcast(velz, NST, MPI_FLOAT, 0, MCW));

   *taxx=velx;
   *tayy=vely;
   *tazz=velz;

   //fprintf(stdout, "%d: inside ini_plane_wave.  taxx[%d]=%e\n", rank, NST-1, taxx[NST-1]);

   return(err);
}

