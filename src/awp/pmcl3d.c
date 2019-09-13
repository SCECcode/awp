/*  
********************************************************************************
* pmcl3d.c                                                                     *
* programming in C&CUDA language                                                    *
* Author: Jun Zhou                                                             * 
* First Version: Cerjan Mode and Homogenous                                    *
********************************************************************************
*/  
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <awp/kernel_launch.h>
#include <awp/pmcl3d.h>
#include <awp/swap.h>
#include <awp/utils.h>
#include <awp/dump.h>
#include <awp/seism.h>
#include <awp/debug.h>
#include <awp/calc.h>
#include <topography/topography.h>
#include <topography/topography.cuh>
#include <topography/sources/sources.h>
#include <topography/receivers/receivers.h>
#include <buffers/buffer.h>
#include <checksum/checksum.h>

#define FORCE_HIGH_Q 0


int main(int argc,char **argv)
{
//  variable definition begins
    _prec TMAX, DH[MAXGRIDS], DT, ARBC, PHT;
    int   NPC, ND, NSRC[MAXGRIDS], NST;
    int   NVE, NVAR, MEDIASTART, IFAULT, READ_STEP, READ_STEP_GPU;
    int   NX, NY, NZ[MAXGRIDS], PX, PY, IDYNA, SoCalQ, FOLLOWBATHY;
    int   NBGX[MAXGRIDS], NEDX[MAXGRIDS], NSKPX[MAXGRIDS]; 
    int   NBGY[MAXGRIDS], NEDY[MAXGRIDS], NSKPY[MAXGRIDS]; 
    int   NBGZ[MAXGRIDS], NEDZ[MAXGRIDS], NSKPZ[MAXGRIDS];
    int   nxt[MAXGRIDS], nyt[MAXGRIDS], nzt[MAXGRIDS];
    MPI_Offset displacement[MAXGRIDS];
    _prec FAC, Q0, EX, FP; 
    char  INSRC[50], INVEL[50], OUT[50], INSRC_I2[50], CHKFILE[50];
    char  insrcgrid[52], insrc_i2_grid[50];
    double GFLOPS = 1.0;
    double GFLOPS_SUM = 0.0;
    Grid3D *u1=NULL, *v1=NULL, *w1=NULL;
    Grid3D *d1=NULL, *mu=NULL, *lam=NULL;
    Grid3D *xx=NULL, *yy=NULL, *zz=NULL, *xy=NULL, *yz=NULL, *xz=NULL;
    Grid3D *r1=NULL, *r2=NULL, *r3=NULL, *r4=NULL, *r5=NULL, *r6=NULL;
    Grid3D *qp=NULL, *qs=NULL;
    PosInf *tpsrc=NULL;
    Grid1D *taxx=NULL, *tayy=NULL, *tazz=NULL, *taxz=NULL, *tayz=NULL, *taxy=NULL; 
    Grid1D *Bufx=NULL,coeff=NULL;
    Grid1D *Bufy=NULL, *Bufz=NULL;
    //Plasticity output buffers
    Grid1D *Bufeta=NULL, *Bufeta2=NULL;
    Grid3D *vx1=NULL,   *vx2=NULL,  *wwo=NULL,  *lam_mu=NULL;
    Grid3Dww *ww=NULL;
    Grid1D *dcrjx, *dcrjy, *dcrjz;
    _prec **vse, **vpe, **dde;
    FILE *fchk;
    // plasticity variables
    Grid3D *sigma2=NULL;
    Grid3D *cohes=NULL, *phi=NULL;
    Grid3D *yldfac=NULL, *neta=NULL;
    /*Grid3D EPxx=NULL, EPyy=NULL, EPzz=NULL;
    Grid3D EPxy=NULL, EPyz=NULL, EPxz=NULL;*/
    // topography variables
    int usetopo = 0;
    char INTOPO[IN_FILE_LEN];

    checksum_t checksum;
    int usechecksum = 0;
    char CHECKSUM[IN_FILE_LEN];

    int usesourcefile = 0;
    char SOURCEFILE[IN_FILE_LEN];

    int userecvfile = 0;
    char RECVFILE[IN_FILE_LEN];

//  GPU variables
    long int num_bytes;
    _prec* * d_d1;
    _prec* * d_u1;
    _prec* * d_v1;
    _prec* * d_w1;
    _prec* * d_f_u1;
    _prec* * d_f_v1;
    _prec* * d_f_w1;
    _prec* * d_b_u1;
    _prec* * d_b_v1;
    _prec* * d_b_w1;
    _prec* * d_dcrjx;
    _prec* * d_dcrjy;
    _prec* * d_dcrjz;
    _prec* * d_lam;
    _prec* * d_mu;
    _prec* * d_qp;
    _prec*  d_coeff;
    _prec* * d_qs;
    _prec* * d_vx1;
    _prec* * d_vx2;
    int** d_ww;
    _prec* * d_wwo;
    _prec* * d_xx;
    _prec* * d_yy;
    _prec* * d_zz;
    _prec* * d_xy;
    _prec* * d_xz;
    _prec* * d_yz;
    _prec* * d_r1;
    _prec* * d_r2;
    _prec* * d_r3;
    _prec* * d_r4;
    _prec* * d_r5;
    _prec* * d_r6;
    _prec* * d_lam_mu;
    int **d_tpsrc;
    _prec* * d_taxx;
    _prec* * d_tayy;
    _prec* * d_tazz;
    _prec* * d_taxz;
    _prec* * d_tayz;
    _prec* * d_taxy;
    // plasticity
    _prec **d_sigma2;
    _prec **d_yldfac,**d_cohes, **d_phi, **d_neta;
//  end of GPU variables
    int i,j,k,idx,idy,idz;
    long int idtmp;
    long int tmpInd;
    const int maxdim = 3;
    _prec taumax, taumin, tauu;
    Grid3D tau=NULL, tau1=NULL, tau2=NULL;
    Grid3D weights=NULL; 
    int npsrc[MAXGRIDS];
    long int nt, cur_step, source_step;
    double time_un = 0.0;
    // time_src and time_mesh measures the time spent
    // in source and mesh reading 
    double time_src = 0.0, time_src_tmp = 0.0, time_mesh = 0.0; 
    // time_fileio and time_gpuio measures the time spent
    // in file system IO and gpu memory copying for IO 
    double time_fileio = 0.0, time_gpuio = 0.0;
    double time_fileio_tmp = 0.0, time_gpuio_tmp = 0.0; 
//  MPI+CUDA variables
    cudaError_t cerr;
    size_t cmemfree, cmemtotal;
    cudaStream_t stream_1, /*stream_1b,*/ stream_2, /*stream_2b,*/ stream_i, stream_i2;;
    int   rank, size, err, srcproc[MAXGRIDS], rank_gpu;
    int   dim[2], period[2], coord[2], reorder;
    int   x_rank_L  = -1,  x_rank_R  = -1,  y_rank_F = -1,  y_rank_B = -1;
    MPI_Comm MCW, MC1;
    MPI_Request  request_x[MAXGRIDS][4], request_y[MAXGRIDS][4];
    MPI_Status   status_x[MAXGRIDS][4],  status_y[MAXGRIDS][4], filestatus;
    MPI_File fh;
    int maxNX_NY_NZ_WS; 
    #ifdef NOBGIO
    /*int   fmtype[3], fptype[3], foffset[3];*/
    int **ones;
    MPI_Aint **dispArray;
    MPI_Datatype filetype[MAXGRIDS];
    #endif

    int   msg_v_size_x[MAXGRIDS], msg_v_size_y[MAXGRIDS], count_x[MAXGRIDS], count_y[MAXGRIDS];
    int   xls[MAXGRIDS], xre[MAXGRIDS], xvs[MAXGRIDS], xve[MAXGRIDS], xss1[MAXGRIDS]; 
    int   xse1[MAXGRIDS], xss2[MAXGRIDS], xse2[MAXGRIDS], xss3[MAXGRIDS], xse3[MAXGRIDS];
    int   yfs[MAXGRIDS], yfe[MAXGRIDS], ybs[MAXGRIDS], ybe[MAXGRIDS], yls[MAXGRIDS],  yre[MAXGRIDS];
    /* Added by Daniel for plasticity computation boundaries */
    int  xlsp[MAXGRIDS], xrep[MAXGRIDS], ylsp[MAXGRIDS], yrep[MAXGRIDS];
    _prec* * SL_vel;     // Velocity to be sent to   Left  in x direction (u1,v1,w1)
    _prec* * SR_vel;     // Velocity to be Sent to   Right in x direction (u1,v1,w1)
    _prec* * RL_vel;     // Velocity to be Recv from Left  in x direction (u1,v1,w1)
    _prec* * RR_vel;     // Velocity to be Recv from Right in x direction (u1,v1,w1)
    _prec* * SF_vel;     // Velocity to be sent to   Front in y direction (u1,v1,w1)
    _prec* * SB_vel;     // Velocity to be Sent to   Back  in y direction (u1,v1,w1)
    _prec* * RF_vel;     // Velocity to be Recv from Front in y direction (u1,v1,w1)
    _prec* * RB_vel;     // Velocity to be Recv from Back  in y direction (u1,v1,w1)

//  variable definition ends    

    int tmpSize;
    int WRITE_STEP;
    int NTISKP;
    int rec_NX[MAXGRIDS];
    int rec_NY[MAXGRIDS];
    int rec_NZ[MAXGRIDS];
    int rec_nxt[MAXGRIDS];
    int rec_nyt[MAXGRIDS];
    int rec_nzt[MAXGRIDS];
    int rec_nbgx[MAXGRIDS];   // 0-based indexing, however NBG* is 1-based
    int rec_nedx[MAXGRIDS];   // 0-based indexing, however NED* is 1-based
    int rec_nbgy[MAXGRIDS];   // 0-based indexing
    int rec_nedy[MAXGRIDS];   // 0-based indexing
    int rec_nbgz[MAXGRIDS];   // 0-based indexing
    int rec_nedz[MAXGRIDS];   // 0-based indexing
    char filename[50];
    #ifdef NOBGIO
    char filenamebasex[50];
    char filenamebasey[50];
    char filenamebasez[50];
    char filenamebaseeta[50];
    char filenamebaseep[50];
    #endif

    // moving initial stress computation to GPU
    _prec fmajor=0, fminor=0, strike[3], dip[3], Rz[9], RzT[9];

    // variables for fault boundary condition (Daniel)
    int fbc_ext[6], fbc_off[3], fbc_extl[6], fbc_dim[3], fbc_seismio, fbc_tskp=1;
    char fbc_pmask[200];
    long int nel[MAXGRIDS];

    int ranktype=0, size_tot;
    MPI_Comm MCT, MCS, MCI;

    /*Daniel - Buffers for exchange of yield factors, same naming as with velocity */
    _prec **SL_yldfac, **SR_yldfac, **RL_yldfac, **RR_yldfac; 
    _prec **SF_yldfac, **SB_yldfac, **RF_yldfac, **RB_yldfac; 
    _prec **d_SL_yldfac, **d_SR_yldfac, **d_RL_yldfac, **d_RR_yldfac; 
    _prec **d_SF_yldfac, **d_SB_yldfac, **d_RF_yldfac, **d_RB_yldfac; 
    int *yldfac_msg_size_x, *yldfac_msg_size_y;
    long int num_bytes2;
    MPI_Request  request_x_yldfac[MAXGRIDS][4], request_y_yldfac[MAXGRIDS][4];
    MPI_Status   status_x_yldfac[MAXGRIDS][4], status_y_yldfac[MAXGRIDS][4];
    int   count_x_yldfac[MAXGRIDS], count_y_yldfac[MAXGRIDS];
    int yls2[MAXGRIDS], yre2[MAXGRIDS];

    /* DM variables added by Daniel */
    int p;
    int ngrids;
    int grdfct[MAXGRIDS]; /* Horizontal grid extent with respect to coarsest grid */

    //int dm = 53;
    //int dm = 41;

    /*buffers for overlap zone variables */
    _prec **SL_swap, **SR_swap, **RL_swap, **RR_swap; 
    _prec **SF_swap, **SB_swap, **RF_swap, **RB_swap; 
    _prec **d_SL_swap, **d_SR_swap, **d_RL_swap, **d_RR_swap;
    _prec **d_SF_swap, **d_SB_swap, **d_RF_swap, **d_RB_swap;
    int *swp_msg_size_x; //*swp_msg_size_x_l;
    int *swp_msg_size_y; //*swp_msg_size_y_l;
    MPI_Request  request_x_swp[MAXGRIDS][4], request_y_swp[MAXGRIDS][4];
    MPI_Status   status_x_swp[MAXGRIDS][4], status_y_swp[MAXGRIDS][4];
    int   count_x_swp[MAXGRIDS], count_y_swp[MAXGRIDS];
    int intlev[MAXGRIDS], swaplevmin, swaplevmax; 
    int grid_output[MAXGRIDS];
    int islowest=0;

    /*computation of moment and magnitude for kinematic source */
    _prec **mom, **d_mom, tmom=0.0f, gmom, mag;
    int n;

    #ifdef SEISMIO
    //int ghostx=ngsl+2, ghosty=ngsl+2, ghostz=align;
    int ghostx=0, ghosty=0, ghostz=0;
    char seism_method[]="mpiio";
    int nx, ny, PZ=1;
    int seism_regGridID[MAXGRIDS];
    int seism_filex[MAXGRIDS], seism_filey[MAXGRIDS], seism_filez[MAXGRIDS];
    int seism_fileeta[MAXGRIDS], seism_fileep[MAXGRIDS];
    int one=1;
    #endif

    // Variables for filtering of source-time-function (Daniel)
    int filtorder=-1;
    /* filter parameters b and a, and state variable d */
    double srcfilt_b[MAXFILT], srcfilt_a[MAXFILT], **d_srcfilt_d;  
    FILE *fltfid;

    int outsize, nout;
    time_t time1, time2;

    /* for FOLLOWBATHY option - save surface output on ocean floor */
    int **bathy, ko;
    _prec tmpvs;

    int main_err = 0;


//  variable initialization begins 
    //NZ=(int*) calloc(MAXGRIDS, sizeof(int));
    command(argc, argv, &TMAX, DH, &DT, &ARBC, &PHT, &NPC, &ND, NSRC, &NST,
            &NVAR, &NVE, &MEDIASTART, &IFAULT, &READ_STEP, &READ_STEP_GPU,
            &NTISKP, &WRITE_STEP, &NX, &NY, NZ, &PX, &PY, NBGX, NEDX, NSKPX,
            NBGY, NEDY, NSKPY, NBGZ, NEDZ, NSKPZ, &FAC, &Q0, &EX, &FP, &IDYNA,
            &SoCalQ, INSRC, INVEL, OUT, INSRC_I2, CHKFILE, &ngrids,
            &FOLLOWBATHY, INTOPO, &usetopo, CHECKSUM, &usechecksum, SOURCEFILE,
            &usesourcefile, RECVFILE, &userecvfile);


#ifndef SEISMIO
     #ifdef NOBGIO
      sprintf(filenamebasex,"%s/SX",OUT);
      sprintf(filenamebasey,"%s/SY",OUT);
      sprintf(filenamebasez,"%s/SZ",OUT);
      sprintf(filenamebaseeta,"%s/Eta",OUT);
      sprintf(filenamebaseep,"%s/EP",OUT);
     #endif
    #endif


    MPICHK(MPI_Init(&argc,&argv));
     MPICHK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    MPICHK(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
    MPICHK(MPI_Comm_size(MPI_COMM_WORLD,&size_tot));

    if (rank==0) fprintf(stdout, "AWP-ODC-DM: Number of grid resolutions = %d\n", ngrids);
    fflush(stdout);

    #ifndef NOBGIO
    if ((size_tot % 3) != 0){
       if (rank==0) fprintf(stderr, "Error. Number of CPUs %d must be divisible by 3.\n", size_tot);
       MPICHK(MPI_Finalize());
       return(0);
    }
    fflush(stdout);
    size = size_tot / 3;

    if ((NX % PX) != 0) {
        if (rank==0) fprintf(stderr, "NX on grid %d (%d) is not divisible by PX (%d)\n", 
           ngrids-1, NX, PX);
        MPICHK(MPI_Finalize());
        return(0);
    }

    if ((NY % PY) != 0) {
        if (rank==0) fprintf(stderr, "NY on grid %d (%d) is not divisible by PY (%d)\n", 
           ngrids-1, NY, PY);
        MPICHK(MPI_Finalize());
    }
    fflush(stdout);

    MPICHK(MPI_Comm_dup(MPI_COMM_WORLD, &MCT ));
    /* The communicator MCW includes all ranks involved in GPU computations */
    /* colors for MPI_Comm_split: 0=launches kernels; 1=source I/O for IFAULT=4; 2=time series output*/
    if (rank < size) ranktype=0;
    else if (rank < size*2) ranktype=1;
    else ranktype=2;
    MPICHK(MPI_Comm_split(MCT, ranktype, 0, &MCW));
    MPICHK(MPI_Comm_split(MCT, ranktype, 1, &MCS));
    MPICHK(MPI_Comm_split(MCT, ranktype, 2, &MCI));

    for (p=0; p<ngrids; p++) count_y_yldfac[p] = count_x_yldfac[p] = 0;

    MPICHK(MPI_Barrier(MCT));

    /* Business as usual for these ranks */
    if (ranktype==0) {
    #else
    size = size_tot ;
    MPICHK(MPI_Comm_dup(MPI_COMM_WORLD, &MCW ));
    #endif

    grdfct[ngrids-1]=1;
    for (p=ngrids-2; p>-1; p--) grdfct[p] = grdfct[p+1] * 3;

    DH[ngrids-1] = DH[0];
    for (p=0; p<ngrids; p++){
       DH[p] = DH[ngrids-1] / grdfct[p];
       nxt[p] = NX/PX * grdfct[p];
       nyt[p] = NY/PY * grdfct[p];
    }

    for (p=0; p<ngrids; p++){
       nzt[p] = NZ[p];
       if ((nzt[p] % BLOCK_SIZE_Z) != 0){
	  if (rank==0) {
	      fprintf(stderr, "NZT for grid %d is not divisble by BLOCK_SIZE_Z.\n", p);
	      fprintf(stderr, "NZT = %d, BLOCK_SIZE_Z=%d\n", nzt[p], BLOCK_SIZE_Z);
	      fprintf(stderr, "Aborting.  Please change NZT or change BLOCK_SIZE_Z in pmcl3d_cons.h and recompile.\n");
	  }
	  MPICHK(MPI_Finalize());
	  return(0);
       }
    }
    fflush(stdout);

    nt        = (int)(TMAX/DT) + 1;
    dim[0]    = PX;
    dim[1]    = PY;
    if (NPC < 2) { 
       period[0] = 0;
       period[1] = 0;
    }
    else { /* Periodic PCs - Daniel */
       period[0] = 1;
       period[1] = 1;
    }
    reorder   = 1;
    err       = MPI_Cart_create(MCW, 2, dim, period, reorder, &MC1);
    err       = MPI_Cart_shift(MC1, 0,  1,  &x_rank_L, &x_rank_R );
    err       = MPI_Cart_shift(MC1, 1,  1,  &y_rank_F, &y_rank_B ); 
    err       = MPI_Cart_coords(MC1, rank, 2, coord);
    err       = MPI_Barrier(MCW);

    // If any neighboring rank is out of bounds, then MPI_Cart_shift sets the
    // destination argument to a negative number. We use the convention that -1 
    // denotes ranks out of bounds.
    if (x_rank_L < 0) {
            x_rank_L = -1;
    }    
    if (x_rank_R < 0 ) {
            x_rank_R = -1;
    }    
    if (y_rank_F < 0) {
            y_rank_F = -1;
    }    
    if (y_rank_B < 0) {
            y_rank_B = -1;
    }    

    rank_gpu = init_gpu_rank(rank);



printf("\n\nrank=%d) RS=%d, RSG=%d, NST=%d, IF=%d\n\n\n", 
rank, READ_STEP, READ_STEP_GPU, NST, IFAULT);
    fflush(stdout);

    for (p=0; p<ngrids; p++){
       if (p==0){
	  if(NEDX[p]==-1) NEDX[p] = NX*grdfct[p];
	  if(NEDY[p]==-1) NEDY[p] = NY*grdfct[p];
	  if(NEDZ[p]==-1) NEDZ[p] = NZ[p];
          grid_output[p] = 1;
       }
       // make NED's a record point
       // for instance if NBGX:NSKPX:NEDX = 1:3:9
       // then we have 1,4,7 but NEDX=7 is better
       NEDX[p] = NEDX[p]-(NEDX[p]-NBGX[p])%NSKPX[p];
       NEDY[p] = NEDY[p]-(NEDY[p]-NBGY[p])%NSKPY[p];
       NEDZ[p] = NEDZ[p]-(NEDZ[p]-NBGZ[p])%NSKPZ[p];
       if (NEDX[p] > -1 && NEDY[p] > -1 && NEDZ[p] > -1) grid_output[p] = 1;
       fprintf(stdout, "%d: X: %d:%d:%d.  Y: %d:%d:%d.  Z:%d:%d:%d\n",
          p, NBGX[p], NSKPX[p], NEDX[p],  NBGY[p], NSKPY[p], NEDY[p], NBGZ[p], NSKPZ[p], NEDZ[p]);
    }
    #ifndef SEISMIO
    // number of recording points in total
    for (p=0; p<ngrids; p++){
       rec_NX[p] = (NEDX[p]-NBGX[p])/NSKPX[p]+1;
       rec_NY[p] = (NEDY[p]-NBGY[p])/NSKPY[p]+1;
       rec_NZ[p] = (NEDZ[p]-NBGZ[p])/NSKPZ[p]+1;

       // specific to each processor:
       calcRecordingPoints(&rec_nbgx[p], &rec_nedx[p], &rec_nbgy[p], &rec_nedy[p], 
	 &rec_nbgz[p], &rec_nedz[p], &rec_nxt[p], &rec_nyt[p], &rec_nzt[p], &displacement[p],
	 (long int)nxt[p],(long int)nyt[p],(long int)nzt[p], rec_NX[p], rec_NY[p], rec_NZ[p], 
	 NBGX[p],NEDX[p],NSKPX[p], NBGY[p],NEDY[p],NSKPY[p], NBGZ[p],NEDZ[p],NSKPZ[p], coord);
       printf(
           "%d = (%d,%d)) "
           "NX,NY,NZ=%d,%d,%d\n"
           "nxt,nyt,nzt=%d,%d,%d\nrec_N=(%d,%d,%d)\nrec_nxt,"
           "=%d,%d,%d\nNBGX,SKP,END=(%d:%d:%d),(%d:%d:%d),(%d:%d:%d)\nrec_nbg,"
           "ed=(%d,%d),(%d,%d),(%d,%d)\ndisp=%ld\n",
           rank, coord[p], coord[1], NX, NY, NZ[p], nxt[p], nyt[p], nzt[p],
           rec_NX[p], rec_NY[p], rec_NZ[p], rec_nxt[p], rec_nyt[p], rec_nzt[p],
           NBGX[p], NSKPX[p], NEDX[p], NBGY[p], NSKPY[p], NEDY[p], NBGZ[p],
           NSKPZ[p], NEDZ[p], rec_nbgx[p], rec_nedx[p], rec_nbgy[p],
           rec_nedy[p], rec_nbgz[p], rec_nedz[p], (long int)displacement[p]);
    }

    #ifndef NOBGIO
    MPICHK(MPI_Send(rec_nxt, ngrids, MPI_INT, rank+2*size, MPIRANKIO, MPI_COMM_WORLD));
    MPICHK(MPI_Send(rec_nyt, ngrids, MPI_INT, rank+2*size, MPIRANKIO+1, MPI_COMM_WORLD));
    MPICHK(MPI_Send(rec_nzt, ngrids, MPI_INT, rank+2*size, MPIRANKIO+2, MPI_COMM_WORLD));
    MPICHK(MPI_Send(rec_NX, ngrids, MPI_INT, rank+2*size, MPIRANKIO+3, MPI_COMM_WORLD));
    MPICHK(MPI_Send(rec_NY, ngrids, MPI_INT, rank+2*size, MPIRANKIO+4, MPI_COMM_WORLD));
    MPICHK(MPI_Send(rec_NZ, ngrids, MPI_INT, rank+2*size, MPIRANKIO+5, MPI_COMM_WORLD));
    MPICHK(MPI_Send(grid_output, ngrids, MPI_INT, rank+2*size, MPIRANKIO+6, MPI_COMM_WORLD));
    MPICHK(MPI_Send(displacement, ngrids, MPI_OFFSET, rank+2*size, MPIRANKIO+7, MPI_COMM_WORLD));
    #else
    dispArray=(MPI_Aint**) calloc(ngrids, sizeof(MPI_Aint*));
    ones=(int**) calloc(ngrids, sizeof(int*));
    for (p=0; p<ngrids; p++){
       maxNX_NY_NZ_WS = (rec_NX[p]>rec_NY[p]?rec_NX[p]:rec_NY[p]);
       maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>rec_NZ[p]?maxNX_NY_NZ_WS:rec_NZ[p]);
       maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>WRITE_STEP?maxNX_NY_NZ_WS:WRITE_STEP);
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
       for(i=0;i<WRITE_STEP;i++){
	 dispArray[p][i] = sizeof(_prec);
	 dispArray[p][i] = dispArray[p][i]*rec_NZ[p]*rec_NY[p]*rec_NX[p]*i;
       }
       err = MPI_Type_create_hindexed(WRITE_STEP, ones[p], dispArray[p], filetype[p], &filetype[p]);
       err = MPI_Type_commit(&filetype[p]);
       MPICHK(MPI_Type_size(filetype[p], &tmpSize));
       if(rank==0) printf("filetype size grid %d (supposedly=rec_nxt*nyt*nzt*WS*4=%ld) =%d\n", 
          p, rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP*sizeof(_prec),tmpSize);

    }
    fflush(stdout);

    /*
    fmtype[0]  = WRITE_STEP;
    fmtype[1]  = NY;
    fmtype[2]  = NX;
    fptype[0]  = WRITE_STEP;
    fptype[1]  = nyt[0];
    fptype[2]  = nxt[0];
    foffset[0] = 0;
    foffset[1] = nyt*coord[1];
    foffset[2] = nxt*coord[0];
    err = MPI_Type_create_subarray(3, fmtype, fptype, foffset, MPI_ORDER_C, _mpi_prec, &filetype);
    err = MPI_Type_commit(&filetype);*/

    #endif

    #else
    err = 0;
    // 2 <= maxdim <= 3
    for (p=0; p<ngrids; p++){
       if (grid_output[p]){
	  nx=NX*grdfct[p];
	  ny=NY*grdfct[p];
	  if (rank == 0) fprintf(stdout, "|    initializing SEISM-IO for grid %d\n", p);
	  seism_init(&MC1,&rank,coord,(int*)&maxdim,&nx,&ny,&nzt[p],&nxt[p],&nyt[p],&nzt[p],
		     &ghostx,&ghosty,&ghostz,&PX,&PY,&PZ,seism_method,&err);
	  if (err != 0) {
	      fprintf(stderr, "|    SEISM ERROR! Init failed for grid %d!\n", p);
	      MPICHK(MPI_Abort(MCW, 1));
	      MPICHK(MPI_Finalize());
	  }
	  if (rank == 0) fprintf(stdout, "|    done initializing SEISM-IO for grid %d\n", p);

	  seism_createRegularGrid(NBGX+p, NEDX+p, NSKPX+p, NBGY+p, NEDY+p, NSKPY+p,
				  NBGZ+p, NEDZ+p, NSKPZ+p, seism_regGridID+p, &err);

	  sprintf(filenamebasex,"%s/SX_%d", OUT, p);
	  sprintf(filenamebasey,"%s/SY_%d", OUT, p);
	  sprintf(filenamebasez,"%s/SZ_%d", OUT, p);
	  sprintf(filenamebaseeta,"%s/Eta_%d",OUT, p);
	  //sprintf(filenamebaseep,"%s/EP",OUT);

          seism_file_open(filenamebasex, "w", &WRITE_STEP, "float", seism_regGridID+p, seism_filex+p, &err);
          seism_file_open(filenamebasey, "w", &WRITE_STEP, "float", seism_regGridID+p, seism_filey+p, &err);
          seism_file_open(filenamebasez, "w", &WRITE_STEP, "float", seism_regGridID+p, seism_filez+p, &err);
          if (NVE == 3)
             seism_file_open(filenamebaseeta, "w", &WRITE_STEP, "float", seism_regGridID+p, seism_fileeta+p, &err);
       }
    }
    #endif 

    for (p=0; p<ngrids; p++){

       if(x_rank_L<0) {
	  xls[p] = 2+ngsl;
	  xlsp[p] = xls[p];
       }
       else {
	  xls[p] = 4;
	  xlsp[p] = xls[p] -1;
       }

       if(x_rank_R<0) {
	  xre[p] = nxt[p]+ngsl+1;
	  xrep[p] = xre[p];
       }
       else {
	  xre[p] = nxt[p] + ngsl2 - 1;
	  xrep[p] = xre[p] + 1;
       }

       xvs[p]   = 2+ngsl;
       xve[p]   = nxt[p]+ngsl+1;

       xss1[p]  = xls[p];
       xse1[p]  = ngsl+3;
       xss2[p]  = ngsl+4;
       xse2[p]  = nxt[p]+ngsl-1;
       xss3[p]  = nxt[p]+ngsl;
       xse3[p]  = xre[p];

       if(y_rank_F<0) {
	  yls[p] = 2+ngsl;
	  ylsp[p] = yls[p];
       }
       else {
	  yls[p] = 4;
	  ylsp[p] = yls[p] -1;
       }

       if(y_rank_B<0) {
	  yre[p] = nyt[p]+ngsl+1;
	  yrep[p] = yre[p];
       }
       else {
	  yre[p] = nyt[p] + ngsl2 - 1;
	  yrep[p] = yre[p] + 1;
       }

       /*margins for division of inner stress region*/
       yls2[p]=yls[p] + (int) (yre[p]-yls[p])*0.25;
       if (yls2[p] % 2 !=0) yls2[p]=yls2[p]+1;  /* yls2 must be even */
       yre2[p]=yls[p] + (int) (yre[p]-yls[p])*0.75;
       if (yre2[p] % 2 ==0) yre2[p]=yre2[p]-1; /* yre2 must be uneven */

       yls2[p]=max(yls2[p], ylsp[p]+ngsl+2);
       yre2[p]=min(yre2[p], yrep[p]-ngsl-2);

       if (rank == 0)
         fprintf(stdout, "%d: yls[%d]=%d, yls2[%d]=%d, yre2[%d]=%d, yre[%d]=%d\n", rank, 
            p, yls[p], p, yls2[p], p, yre2[p], p, yre[p]);

       yfs[p]  = 2+ngsl;
       yfe[p]  = 2+ngsl2-1;   
       ybs[p]  = nyt[p]+2;
       ybe[p]  = nyt[p]+ngsl+1;   
    
    }
    fflush(stdout);

    time_src -= gethrtime();

    if(rank==0) printf("Before inisource\n");

    if (rank==0) {
       if (access("sourcefilter.dat", F_OK) != -1){
          fltfid=fopen("sourcefilter.dat", "r");
          fscanf(fltfid, "%d\n", &filtorder);
          fprintf(stdout, "Order of source filter: %d.  Parameters:\n", filtorder);
          for (k=0; k<filtorder+1; k++){
             fscanf(fltfid, "%le %le\n", srcfilt_b+k, srcfilt_a+k);
             fprintf(stdout, "b[%d]=%le, a[%d]=%le\n", k, srcfilt_b[k], k, srcfilt_a[k]);
          }
          fclose(fltfid); 
       }
       else {
          fprintf(stdout, "File sourcefilter.dat not found, no STF filtering applied.\n");
       }
    }
    MPICHK(MPI_Bcast(&filtorder, 1, MPI_INT, 0, MCW));
    
    if (filtorder > 0){
       MPICHK(MPI_Bcast(srcfilt_b, filtorder+1, MPI_DOUBLE, 0, MCW));
       MPICHK(MPI_Bcast(srcfilt_a, filtorder+1, MPI_DOUBLE, 0, MCW));
    }
    fflush(stdout);

    SetDeviceFilterParameters(filtorder, srcfilt_b, srcfilt_a);

    tpsrc = (PosInf*) calloc(ngrids, sizeof(PosInf));
    taxx = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    tayy = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    tazz = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    taxy = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    taxz = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    tayz = (Grid1D*) calloc(ngrids, sizeof(Grid1D));

    for (p=0; p<ngrids; p++) {
       npsrc[p] = 0;
       tpsrc[p] = NULL;
       taxx[p] = tayy[p] = tazz[p] = taxy[p] = taxz[p] = tayz[p] = NULL;
    }

    if (IFAULT == 5){
       if (rank==0) fprintf(stdout, "Using IFAULT=5: kinematic source.\n");
       if ((NST != 2) || (READ_STEP != 2)) {
             if (rank==0) fprintf(stderr, "IFAULT=5 requires NST = READ_STEP =2.\nQuitting.");
          MPICHK(MPI_Finalize());
          return(-1);
          } 
    }

    if (IFAULT < 3 || IFAULT == 5) {
        for (p=0; p<ngrids; p++){
           if (NSRC[p] > 0) {
	      sprintf(insrcgrid, "%s_%d", INSRC, p);
	      sprintf(insrc_i2_grid, "%s_%d", INSRC_I2, p);
	      fprintf(stdout, "opening %s\n", insrcgrid);
	      err = inisource(rank,   IFAULT, NSRC[p],  READ_STEP, NST,   srcproc+p, NZ[p], MCW, nxt[p], nyt[p], nzt[p], 
		 coord, maxdim, npsrc+p, tpsrc+p, taxx+p, tayy+p, tazz+p, taxz+p, tayz+p, taxy+p, insrcgrid, insrc_i2_grid);
           }
           else srcproc[p] = -1;
       }
    }
    else if(IFAULT == 4){
	err = read_src_ifault_4(rank, READ_STEP,
	INSRC, maxdim, coord, NZ[0],
	nxt[0], nyt[0], nzt[0],
	&npsrc[0], &srcproc[0],
	&tpsrc[0], &taxx[0], &tayy[0], &tazz[0], 1, 
	fbc_ext, fbc_off, fbc_pmask, fbc_extl, fbc_dim, 
	&fbc_seismio, &fbc_tskp, NST, size);
    }

    if (IFAULT == 5){
        mom=(_prec* *) calloc(ngrids, sizeof(_prec* ));
        d_mom=(_prec* *) calloc(ngrids, sizeof(_prec* ));
        for (p=0; p<ngrids; p++) {
           if (rank==srcproc[p]) {
              num_bytes = npsrc[p] * sizeof(_prec);
              mom[p] = (_prec* ) calloc(npsrc[p], sizeof(_prec));
              CUCHK(cudaMalloc((void**) &d_mom[p], num_bytes));
              CUCHK(cudaMemset(d_mom[p], 0, num_bytes));
	      CUCHK(cudaMemcpy(d_mom[p], mom[p], num_bytes, cudaMemcpyHostToDevice));
           }
        }
        /*if (srcproc[0] == rank){
           for (n=0; n<npsrc[0]; n++) fprintf(stdout, "src at rank %d: %d,%d,%d\n", 
              rank, tpsrc[0][n*3], tpsrc[0][n*3+1], tpsrc[0][n*3+2]);
        }*/

        /* allocate state variables required for filtering */
        d_srcfilt_d = (double**) calloc(ngrids, sizeof(double*));
        if (filtorder > 0){
           for (p=0; p<ngrids; p++){
              num_bytes = npsrc[p] * (filtorder+1) * sizeof(double);
              CUCHK(cudaMalloc((void**) &d_srcfilt_d[p], num_bytes));
              CUCHK(cudaMemset(d_srcfilt_d[p], 0, num_bytes));
           }
        }
    }

    if (IFAULT == 6){
       if (rank==0) fprintf(stdout, "Using plane wave input at grid position %d in grid %d\n", NZ[ngrids-1]-ND-1, ngrids-1);
       if (READ_STEP != NST) {
          if (rank==0) fprintf(stderr, "Error.  READ_STEP should be equal NST for IFAULT=6\n");
          MPICHK(MPI_Finalize());
       }
       if (NST < 1) {
          if (rank==0) fprintf(stderr, "Error.  NST=%d, but should be > 0 for IFAULT=6.\n", NST);
          MPICHK(MPI_Finalize());
       }
       for (p=0; p<ngrids-1; p++) srcproc[p] = -1;
       srcproc[ngrids-1]=rank;
       err=ini_plane_wave(rank, MCW, INSRC, NST, taxx+ngrids-1, tayy+ngrids-1, tazz+ngrids-1);
       if (rank==0) fprintf(stdout, "taxx[%d]=%e\n", NST-1, taxx[ngrids-1][NST-1]);
    }

    if(err)
    {
       printf("source initialization failed\n");
       return -1;
    }
    time_src += gethrtime(); 
    if(rank==0) printf("After inisource. Time elapsed (seconds): %lf\n", time_src); 
    fflush(stdout);

    d_tpsrc = (int**) calloc(ngrids, sizeof(int*));
    d_taxx = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_tayy = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_tazz = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_taxy = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_taxz = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_tayz = (_prec* *) calloc(ngrids, sizeof(_prec* ));

    for (p=0; p<ngrids; p++){
       if(rank==srcproc[p]) {
	  printf("rank=%d, grid=%d, source rank, npsrc=%d, srcproc=%d\n", rank, p, npsrc[p], srcproc[p]);
	  /* here, we allocate data for keeping prevoius timestep */
          if (IFAULT == 4) num_bytes = sizeof(_prec)*npsrc[p]*(READ_STEP_GPU+1);
          else if (IFAULT == 6) num_bytes = sizeof(_prec)*NST;
          else num_bytes = sizeof(_prec)*npsrc[p]*READ_STEP_GPU;
	  CUCHK(cudaMalloc((void**)&d_taxx[p], num_bytes));
	  CUCHK(cudaMemset(d_taxx[p], 0, num_bytes));
	  CUCHK(cudaMalloc((void**)&d_tayy[p], num_bytes));
	  CUCHK(cudaMemset(d_tayy[p], 0, num_bytes));
	  CUCHK(cudaMalloc((void**)&d_tazz[p], num_bytes));
	  CUCHK(cudaMemset(d_tazz[p], 0, num_bytes));
	  /*Added by Daniel for fault B.C. and plane wave*/
	  if (IFAULT != 4 && IFAULT != 6){
	     CUCHK(cudaMalloc((void**)&d_taxz[p], num_bytes));
	     CUCHK(cudaMemset(d_taxz[p], 0, num_bytes));
	     CUCHK(cudaMalloc((void**)&d_tayz[p], num_bytes));
	     CUCHK(cudaMemset(d_tayz[p], 0, num_bytes));
	     CUCHK(cudaMalloc((void**)&d_taxy[p], num_bytes));
	     CUCHK(cudaMemset(d_taxy[p], 0, num_bytes));
	  }
	  CUCHK(cudaMemcpy(d_taxx[p],taxx[p],num_bytes,cudaMemcpyHostToDevice));
	  CUCHK(cudaMemcpy(d_tayy[p],tayy[p],num_bytes,cudaMemcpyHostToDevice));
	  CUCHK(cudaMemcpy(d_tazz[p],tazz[p],num_bytes,cudaMemcpyHostToDevice));
	  /*Added by Daniel for fault B.C.*/
	  if (IFAULT != 4 && IFAULT != 6) {
	     CUCHK(cudaMemcpy(d_taxz[p],taxz[p],num_bytes,cudaMemcpyHostToDevice));
	     CUCHK(cudaMemcpy(d_tayz[p],tayz[p],num_bytes,cudaMemcpyHostToDevice));
	     CUCHK(cudaMemcpy(d_taxy[p],taxy[p],num_bytes,cudaMemcpyHostToDevice));
	  }
          if (IFAULT !=6) {
	     num_bytes = sizeof(int)*npsrc[p]*maxdim;
	     CUCHK(cudaMalloc((void**)&d_tpsrc[p], num_bytes));
	     CUCHK(cudaMemset(d_tpsrc[p], 0, num_bytes));
	     CUCHK(cudaMemcpy(d_tpsrc[p],tpsrc[p],num_bytes,cudaMemcpyHostToDevice));
          }
       }
    }

    d1 = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    mu = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    lam = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    lam_mu = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    for (p=0; p<ngrids; p++){
       d1[p]     = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align); 
       mu[p]     = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       lam[p]    = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       lam_mu[p] = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, 1); 
    }

    qp = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    qs = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    if(NVE==1 || NVE==3)
       for (p=0; p<ngrids; p++){
       { 
	  qp[p]   = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	  qs[p]   = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       }
       tau  = Alloc3D(2, 2, 2);
       tau1 = Alloc3D(2, 2, 2); 
       tau2 = Alloc3D(2, 2, 2); 
       weights = Alloc3D(2, 2, 2); 
       coeff = Alloc1D(16); 
       weights_sub(weights,coeff, EX, FAC);  
    }
    time_mesh -= gethrtime(); 

    if(NVE==3){
       sigma2 = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
       cohes = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
       phi = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
       yldfac = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
       neta = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
       for (p=0; p<ngrids; p++){
	  sigma2[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	  cohes[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	  phi[p]    = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);

	  yldfac[p] = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	  neta[p]   = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);

	 // initialize
	 for(i=0;i<nxt[p]+4+ngsl2;i++) 
	   for(j=0;j<nyt[p]+4+ngsl2;j++)
	     for(k=0;k<nzt[p]+2*align;k++){
	       neta[p][i][j][k] = 0.;
	       yldfac[p][i][j][k] = 1.;
	     }
       }
    }
    fflush(stdout);

    if(rank==0) printf("Before inimesh\n");
    vpe = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    vse = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    dde = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    for (p=0; p<ngrids; p++){
       char INVEL2[52];
       int corrected;
       sprintf(INVEL2, "%s_%d", INVEL, p);
       //if (rank==0) fprintf(stdout, "opening %s\n", INVEL2);
       vpe[p] = (_prec* ) calloc(2, sizeof(_prec));
       vse[p] = (_prec* ) calloc(2, sizeof(_prec));
       dde[p] = (_prec* ) calloc(2, sizeof(_prec));
       inimesh(rank, MEDIASTART, d1[p], mu[p], lam[p], qp[p], qs[p], &taumax, &taumin, tau, 
               weights,coeff, NVAR, FP, FAC, Q0, EX, 
	       nxt[p], nyt[p], nzt[p], PX, PY, NX*grdfct[p], NY*grdfct[p], nzt[p], coord, MCW, IDYNA, NVE, 
               SoCalQ, INVEL2, vse[p], vpe[p], dde[p]);
       if (p > 0) {
          corrected=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], d1[p], d1[p-1], p, p-1, "d1");
          corrected+=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], mu[p], mu[p-1], p, p-1, "mu");
          corrected+=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], lam[p], lam[p-1], p, p-1, "lam");
          corrected+=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], qp[p], qp[p-1], p, p-1, "qp");
          corrected+=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], qs[p], qs[p-1], p, p-1, "qs");
          if (corrected > 0) fprintf(stdout, "Warning: Inconsistent material constants between mesh %d and %d corrected.\n", p-1, p);
       }

    }
    fflush(stdout);

    if (FOLLOWBATHY == 1) {
       bathy = (int**) calloc(nxt[0]+4+ngsl2, sizeof(int*));
       for (i=0; i<nxt[0]+4+ngsl2; i++) bathy[i] = (int*) calloc(nyt[0]+4+ngsl2, sizeof(int));

       FILE *bathyfid;
       char bathyofname[200];
       sprintf(bathyofname, "debug/bathy.%04d", rank);
       bathyfid=fopen(bathyofname, "w");
       for (i=0; i<nxt[0]+4+ngsl2; i++){
          for (j=0; j<nyt[0]+4+ngsl2; j++){
             for (k=nzt[0] + align - 1; k > align; k--){
                //if (mu[0][i][j][k] < 1.e7) {
                tmpvs=sqrt(1./(mu[0][i][j][k] * d1[0][i][j][k]));
                if (tmpvs > 0.f){
                   bathy[i][j] = k;
                   fprintf(bathyfid, "%d %d %d %e\n", i, j, k, tmpvs);
                   break;
       }}}} 
       fclose(bathyfid);
    } 

    time_mesh += gethrtime();  
    if(rank==0) printf("After inimesh. Time elapsed (seconds): %lf\n", time_mesh);  
    fflush(stdout);
    if(rank==0)
      writeCHK(CHKFILE, NTISKP, DT, DH, nxt, nyt, nzt,
	       nt, ARBC, NPC, NVE, FAC, Q0, EX, FP, vse, vpe, dde, ngrids);

    for (p=0; p<ngrids; p++){
       mediaswap(d1[p], mu[p], lam[p], qp[p], qs[p], rank, x_rank_L, x_rank_R, y_rank_F, y_rank_B, 
          nxt[p], nyt[p], nzt[p], MCW, p);

       for(i=xls[p];i<xre[p]+1;i++)
	 for(j=yls[p];j<yre[p]+1;j++)
	 {
	    _prec t_xl, t_xl2m;
	    t_xl             = 1.0/lam[p][i][j][nzt[p]+align-1];
	    t_xl2m           = 2.0/mu[p][i][j][nzt[p]+align-1] + t_xl; 
	    lam_mu[p][i][j][0]  = t_xl/t_xl2m;
	 }

       if(NVE==3){
	 printf("%d) Computing initial stress\n",rank);
	 inidrpr_hoekbrown_light(nxt[p], nyt[p], nzt[p], NVE, coord, DH[p], rank, mu[p], lam[p], d1[p],
	     sigma2[p], cohes[p], phi[p], &fmajor, &fminor, strike, dip, MCW, p);
	 rotation_matrix(strike, dip, Rz, RzT);
       }
    }
    fflush(stdout);

    if(usechecksum) {
            char tmp[512];
            sprintf(tmp,"%s_%d%d", CHECKSUM,  coord[0], coord[1]);
            checksum_init(&checksum, tmp);
    }

    /*set a zone without plastic yielding around source nodes*/
    MPICHK(MPI_Barrier(MCW));
    if ((NVE > 1) && (IFAULT < 4 || IFAULT == 5)){
    fprintf(stdout, "removing plasticity from source nodes\n");
    for (p=0; p<ngrids; p++){
       for (j=0; j<npsrc[p]; j++){
	  idx = tpsrc[p][j*maxdim]   + 1 + ngsl;
	  idy = tpsrc[p][j*maxdim+1] + 1 + ngsl;
	  idz = tpsrc[p][j*maxdim+2] + align - 1;
	  int xi, yi, zi;
	  int dox, doy, doz;
	  for (xi=idx-1; xi<idx+2;xi++){
	    for (yi=idy-2; yi<idy+2;yi++){ // because we are adding slip on two sides of the fault 
	       for (zi=idz-1; zi<idz+2;zi++){
		  dox=doy=doz=0;
		  if ((xi>=0) && (xi < (nxt[0] + ngsl2 +1))) dox = 1;
		  if ((yi>=0) && (yi < (nyt[0] + ngsl2 +1))) doy = 1;
		  if ((zi>=0) && (yi < (nzt[0] + ngsl2 +1))) doz = 1;
		  if ((dox && doy) && doz ) cohes[p][xi][yi][zi]=1.e18;
	       }
	     }
	  } 
       } 
    }
    fprintf(stdout, "done\n");
    }
    MPICHK(MPI_Barrier(MCW));
    fflush(stdout);

    /*set a zone with high Q around source nodes for two-step method*/
    MPICHK(MPI_Barrier(MCW));
#if FORCE_HIGH_Q
    if (((NVE == 1) || (NVE == 3)) && (IFAULT < 4) && (NPC < 2)){
    fprintf(stdout, "forcing high Q around source nodes\n");
    for (p=0; p<ngrids; p++){
       for (j=0; j<npsrc[p]; j++){
	  idx = tpsrc[p][j*maxdim]   + 1 + ngsl;
	  idy = tpsrc[p][j*maxdim+1] + 1 + ngsl;
	  idz = tpsrc[p][j*maxdim+2] + align - 1;
	  int xi, yi, zi;
	  int dox, doy, doz;
	  for (xi=idx-2; xi<idx+3;xi++){
	    for (yi=idy-2; yi<idy+3;yi++){
	       for (zi=idz-2; zi<idz+3;zi++){
		  dox=doy=doz=0;
		  if ((xi>=0) && (xi < (nxt[0] + ngsl2 +1))) dox = 1;
		  if ((yi>=0) && (yi < (nyt[0] + ngsl2 +1))) doy = 1;
		  if ((zi>=0) && (yi < (nzt[0] + ngsl2 +1))) doz = 1;
		  if ((dox && doy) && doz ) {
		     qp[p][xi][yi][zi]=7.88313861E-04;  //Q of 10,000 before inimesh
		     qs[p][xi][yi][zi]=7.88313861E-04;
		     //qp[p][xi][yi][zi]=0.;  //Q of 10,000 before inimesh
		     //qs[p][xi][yi][zi]=0.;
		  }
	       }
	     }
	  }
       }
    }
    fprintf(stdout, "done\n");
    fflush(stdout);
    }
#endif
    MPICHK(MPI_Barrier(MCW));

    vx1 = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    vx2 = (Grid3D*) calloc(ngrids, sizeof(Grid3D));
    ww = (Grid3Dww*) calloc(ngrids, sizeof(Grid3Dww));
    wwo = (Grid3D*) calloc(ngrids, sizeof(Grid3D));

    d_lam_mu = (_prec* *) calloc(ngrids, sizeof(_prec* ));

    for (p=0; p<ngrids; p++){
       num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2);
       CUCHK(cudaMalloc((void**)&d_lam_mu[p], num_bytes));
       CUCHK(cudaMemset(d_lam_mu[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_lam_mu[p],&lam_mu[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));

       vx1[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       vx2[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       ww[p]   = Alloc3Dww(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align); 
       wwo[p]   = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align); 
    }

    //fprintf(stdout, "sizeof Grid1D: %ld, sizeof Grid1D*: %ld\n", sizeof(Grid1D), sizeof(Grid1D*));
    dcrjx = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    dcrjy = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    dcrjz = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    if((NPC==0) || (NPC == 2)){
       for (p=0; p<ngrids; p++){
	   dcrjx[p] = Alloc1D(nxt[p]+4+ngsl2);
	   dcrjy[p] = Alloc1D(nyt[p]+4+ngsl2);
	   dcrjz[p] = Alloc1D(nzt[p]+2*align);

	   for(i=0;i<nxt[p]+4+ngsl2;i++)
	      dcrjx[p][i]  = 1.0;
	   for(j=0;j<nyt[p]+4+ngsl2;j++)
	      dcrjy[p][j]  = 1.0;
	   for(k=0;k<nzt[p]+2*align;k++)
	      dcrjz[p][k]  = 1.0;

           if (p == ngrids-1) islowest = 1;
           else islowest = 0;
	   inicrj(ARBC, coord, nxt[p], nyt[p], nzt[p], NX*grdfct[p], NY*grdfct[p], ND*grdfct[p], dcrjx[p], dcrjy[p], dcrjz[p], islowest, NPC);

           /*DM: disable ABCs at bottom unless it's the lowest grid*/
           //if (p < ngrids-1) for(k=0;k<nzt[p]+2*align;k++) dcrjz[p][k]  = 1.0;
       }
    }

    if(NVE==1 || NVE==3)
    {
        //_prec dt1 = 1.0/DT;
        for(i=0;i<2;i++)
          for(j=0;j<2;j++)
            for(k=0;k<2;k++)
            {
               tauu          = tau[i][j][k];
	       tau2[i][j][k] = exp(-DT/tauu);
	       tau1[i][j][k] = 0.5*(1.-tau2[i][j][k]); 
            }

        for (p=0; p<ngrids; p++){
	   init_texture(nxt[p], nyt[p], nzt[p], tau1, tau2, 
		vx1[p], vx2[p], weights, ww[p],wwo[p], xls[p], xre[p], yls[p], yre[p]);  
 	   if (p > 0) {
             int corrected;
	     corrected=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], vx1[p], vx1[p-1], p, p-1, "vx1");
	     corrected+=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], vx2[p], vx2[p-1], p, p-1, "vx2");
	     corrected+=checkmesh(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], wwo[p], wwo[p-1], p, p-1, "wwo");
	     corrected+=checkmesh_ww(nxt[p], nyt[p], nzt[p], nxt[p-1], nyt[p-1], nzt[p-1], ww[p], ww[p-1], p, p-1, "ww");
             if (corrected > 0) fprintf(stdout, "Warning: Inconsistent texture variables between mesh %d and %d corrected.\n", 
                p-1, p);
           }
        }


        Delloc3D(tau);
        Delloc3D(tau1);
        Delloc3D(tau2);
    }

    if(rank==0) printf("Allocate device media pointers and copy.\n");
    fflush(stdout);
    d_d1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_lam = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_mu = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_qp = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_qs = (_prec* *) calloc(ngrids, sizeof(_prec* ));

    d_vx1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_vx2 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_ww = (int**) calloc(ngrids, sizeof(int*));
    d_wwo = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    for (p=0; p<ngrids; p++){
       num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
       CUCHK(cudaMalloc((void**)&d_d1[p], num_bytes));
       CUCHK(cudaMemset(d_d1[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_d1[p],&d1[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_lam[p], num_bytes));
       CUCHK(cudaMemset(d_lam[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_lam[p],&lam[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_mu[p], num_bytes));
       CUCHK(cudaMemset(d_mu[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_mu[p],&mu[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_qp[p], num_bytes));
       CUCHK(cudaMemset(d_qp[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_qp[p],&qp[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));

       num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align); 
       CUCHK(cudaMalloc((void**)&d_qs[p], num_bytes));
       CUCHK(cudaMemset(d_qs[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_qs[p],&qs[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_vx1[p], num_bytes));
       CUCHK(cudaMemset(d_vx1[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_vx1[p],&vx1[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_vx2[p], num_bytes));
       CUCHK(cudaMemset(d_vx2[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_vx2[p],&vx2[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       num_bytes = sizeof(int)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align); 
       if (rank==0) fprintf(stdout, "Allocating d_ww and d_wwo, num_bytes=%ld\n", num_bytes);
       CUCHK(cudaMalloc((void**)&d_ww[p], num_bytes));
       CUCHK(cudaMemset(d_ww[p], 0, num_bytes)); 
       CUCHK(cudaMemcpy(d_ww[p],&ww[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align); 
       CUCHK(cudaMalloc((void**)&d_wwo[p], num_bytes));
       CUCHK(cudaMemset(d_wwo[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_wwo[p],&wwo[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));  

    }
    num_bytes = sizeof(_prec)*(16);  
    CUCHK(cudaMalloc((void**)&d_coeff, num_bytes));
    CUCHK(cudaMemset(d_coeff, 0, num_bytes));
    CUCHK(cudaMemcpy(d_coeff,&coeff[0],num_bytes,cudaMemcpyHostToDevice));


    if((NPC==0) || (NPC == 2)) {
       d_dcrjx = (_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_dcrjy = (_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_dcrjz = (_prec* *) calloc(ngrids, sizeof(_prec* ));
       for (p=0; p<ngrids; p++){
	  num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2);
	  CUCHK(cudaMalloc((void**)&d_dcrjx[p], num_bytes));
	  CUCHK(cudaMemset(d_dcrjx[p], 0, num_bytes));
	  CUCHK(cudaMemcpy(d_dcrjx[p],dcrjx[p],num_bytes,cudaMemcpyHostToDevice));
	  num_bytes = sizeof(_prec)*(nyt[p]+4+ngsl2);
	  CUCHK(cudaMalloc((void**)&d_dcrjy[p], num_bytes));
	  CUCHK(cudaMemset(d_dcrjy[p], 0, num_bytes));
	  CUCHK(cudaMemcpy(d_dcrjy[p],dcrjy[p],num_bytes,cudaMemcpyHostToDevice));
	  num_bytes = sizeof(_prec)*(nzt[p]+2*align);
	  CUCHK(cudaMalloc((void**)&d_dcrjz[p], num_bytes));
	  CUCHK(cudaMemset(d_dcrjz[p], 0, num_bytes));
	  CUCHK(cudaMemcpy(d_dcrjz[p],dcrjz[p],num_bytes,cudaMemcpyHostToDevice));
       }
    }

    if(rank==0) printf("Allocate host velocity and stress pointers.\n");
    fflush(stdout);
    u1=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    v1=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    w1=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    xx=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    yy=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    zz=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    xy=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    yz=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    xz=(Grid3D*) calloc(ngrids, sizeof(Grid3D));

    r1=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    r2=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    r3=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    r4=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    r5=(Grid3D*) calloc(ngrids, sizeof(Grid3D));
    r6=(Grid3D*) calloc(ngrids, sizeof(Grid3D));

    for (p=0; p<ngrids; p++){
       u1[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       v1[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       w1[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       xx[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       yy[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       zz[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       xy[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       yz[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       xz[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       if(NVE==1 || NVE==3)
       {
	   r1[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	   r2[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	   r3[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	   r4[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	   r5[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
	   r6[p]  = Alloc3D(nxt[p]+4+ngsl2, nyt[p]+4+ngsl2, nzt[p]+2*align);
       }
    }

    source_step = 1;
    if (IFAULT < 4) {
       for (p=0; p<ngrids; p++){
	  if(rank==srcproc[p]) {
	     printf("%d) add initial src\n", rank);
		addsrc(source_step, DH[p], DT, NST, npsrc[p], READ_STEP, maxdim, tpsrc[p], taxx[p], tayy[p], tazz[p], taxz[p], \
		   tayz[p], taxy[p], xx[p], yy[p], zz[p], xy[p], yz[p], xz[p]);
	  }
       }
    }
    else if (IFAULT == 4) {
       if(rank==srcproc[0]) {
 	 frcvel(source_step, DH[0], DT, NST, npsrc[p], READ_STEP, fbc_tskp, maxdim, tpsrc[0], taxx[0], tayy[0], tazz[0], taxz[0], \
     	    tayz[0], taxy[0], u1[0], v1[0], w1[0], rank);
       }
    }

    if(rank==0) printf("Allocate device velocity and stress pointers and copy.\n");
    fflush(stdout);
    
    d_u1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_v1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_w1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_xx = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_yy = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_zz = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_xy = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_xz = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_yz = (_prec* *) calloc(ngrids, sizeof(_prec* ));

    d_r1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_r2 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_r3 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_r4 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_r5 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_r6 = (_prec* *) calloc(ngrids, sizeof(_prec* ));

    if (NVE==3){
      d_sigma2 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
      d_yldfac = (_prec* *) calloc(ngrids, sizeof(_prec* ));
      d_cohes = (_prec* *) calloc(ngrids, sizeof(_prec* ));
      d_phi = (_prec* *) calloc(ngrids, sizeof(_prec* ));
      d_neta = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    }

    for (p=0; p<ngrids; p++){
       num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
       CUCHK(cudaMalloc((void**)&d_u1[p], num_bytes));
       CUCHK(cudaMemset(d_u1[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_u1[p],&u1[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_v1[p], num_bytes));
       CUCHK(cudaMemset(d_v1[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_v1[p],&v1[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_w1[p], num_bytes));
       CUCHK(cudaMemset(d_w1[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_w1[p],&w1[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_xx[p], num_bytes));
       CUCHK(cudaMemset(d_xx[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_xx[p],&xx[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_yy[p], num_bytes));
       CUCHK(cudaMemset(d_yy[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_yy[p],&yy[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_zz[p], num_bytes));
       CUCHK(cudaMemset(d_zz[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_zz[p],&zz[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_xy[p], num_bytes));
       CUCHK(cudaMemset(d_xy[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_xy[p],&xy[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_xz[p], num_bytes));
       CUCHK(cudaMemset(d_xz[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_xz[p],&xz[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       CUCHK(cudaMalloc((void**)&d_yz[p], num_bytes));
       CUCHK(cudaMemset(d_yz[p], 0, num_bytes));
       CUCHK(cudaMemcpy(d_yz[p],&yz[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       if(NVE==1 || NVE==3)
       {
	 if(rank==0) printf("Allocate additional device pointers (r) and copy.\n");
	   CUCHK(cudaMalloc((void**)&d_r1[p], num_bytes));
	   CUCHK(cudaMemset(d_r1[p], 0, num_bytes));
	   CUCHK(cudaMemcpy(d_r1[p],&r1[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	   CUCHK(cudaMalloc((void**)&d_r2[p], num_bytes));
	   CUCHK(cudaMemset(d_r2[p], 0, num_bytes));
	   CUCHK(cudaMemcpy(d_r2[p],&r2[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	   CUCHK(cudaMalloc((void**)&d_r3[p], num_bytes));
	   CUCHK(cudaMemset(d_r3[p], 0, num_bytes));
	   CUCHK(cudaMemcpy(d_r3[p],&r3[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	   CUCHK(cudaMalloc((void**)&d_r4[p], num_bytes));
	   CUCHK(cudaMemset(d_r4[p], 0, num_bytes));
	   CUCHK(cudaMemcpy(d_r4[p],&r4[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	   CUCHK(cudaMalloc((void**)&d_r5[p], num_bytes));
	   CUCHK(cudaMemset(d_r5[p], 0, num_bytes));
	   CUCHK(cudaMemcpy(d_r5[p],&r5[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	   CUCHK(cudaMalloc((void**)&d_r6[p], num_bytes));
	   CUCHK(cudaMemset(d_r6[p], 0, num_bytes));
	   CUCHK(cudaMemcpy(d_r6[p],&r6[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
       }
       if(NVE==3){
	 if(rank==0) printf("Allocate plasticity variables since NVE=3\n");
	 num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
	 CUCHK(cudaMalloc((void**)&d_sigma2[p], num_bytes));
	 CUCHK(cudaMemset(d_sigma2[p], 0, num_bytes));
	 CUCHK(cudaMemcpy(d_sigma2[p],&sigma2[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	 CUCHK(cudaMalloc((void**)&d_yldfac[p], num_bytes));
	 CUCHK(cudaMemset(d_yldfac[p], 0, num_bytes));
	 CUCHK(cudaMemcpy(d_yldfac[p],&yldfac[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	 CUCHK(cudaMalloc((void**)&d_cohes[p], num_bytes));
	 CUCHK(cudaMemset(d_cohes[p], 0, num_bytes));
	 CUCHK(cudaMemcpy(d_cohes[p],&cohes[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	 CUCHK(cudaMalloc((void**)&d_phi[p], num_bytes));
	 CUCHK(cudaMemset(d_phi[p], 0, num_bytes));
	 CUCHK(cudaMemcpy(d_phi[p],&phi[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));
	 CUCHK(cudaMalloc((void**)&d_neta[p], num_bytes));
	 CUCHK(cudaMemset(d_neta[p], 0, num_bytes));
	 CUCHK(cudaMemcpy(d_neta[p],&neta[p][0][0][0],num_bytes,cudaMemcpyHostToDevice));

	 /*cudaMalloc((void**)&d_yldfac_L, msg_yldfac_size_x*sizeof(_prec));
	 CUCHK(cudaMalloc((void**)&d_yldfac_R, msg_yldfac_size_x*sizeof(_prec)));
	 CUCHK(cudaMalloc((void**)&d_yldfac_F, msg_yldfac_size_y*sizeof(_prec)));
	 CUCHK(cudaMalloc((void**)&d_yldfac_B, msg_yldfac_size_y*sizeof(_prec)));
	 CUCHK(cudaMalloc((void**)&d_yldfac_FL, sizeof(_prec)));
	 CUCHK(cudaMalloc((void**)&d_yldfac_FR, sizeof(_prec)));
	 CUCHK(cudaMalloc((void**)&d_yldfac_BL, sizeof(_prec)));
	 CUCHK(cudaMalloc((void**)&d_yldfac_BR, sizeof(_prec)));*/
       }
       nel[p]=(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
    }
    fflush(stdout);

    /*dump_variable(d_d1[0], nel[0], "d1", 0, 0, 0, rank, size);
    dump_variable(d_qs[0], nel[0], "qs", 0, 0, 0, rank, size);*/

    //  variable initialization ends
    #ifndef SEISMIO
    Bufx = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    Bufy = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    Bufz = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
    if (NVE == 3) Bufeta = (Grid1D*) calloc(ngrids, sizeof(Grid1D));

    for (p=0; p<ngrids; p++){
       if (grid_output[p]){
	  if(rank==0) printf("Allocate buffers of #elements: %d\n",rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP);
	  Bufx[p]  = Alloc1D(rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP);
	  Bufy[p]  = Alloc1D(rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP);
	  Bufz[p]  = Alloc1D(rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP);
	  //  Allocate buffers for plasticity output
	  if (NVE == 3) Bufeta[p] = Alloc1D(rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP);
       }
    }
    #endif

    SL_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    SR_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    RL_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    RR_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    SF_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    SB_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    RF_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    RB_vel = (_prec* *) calloc(ngrids, sizeof(_prec* ));

    d_f_u1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_f_v1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_f_w1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_b_u1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_b_v1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_b_w1 = (_prec* *) calloc(ngrids, sizeof(_prec* ));
    for (p=0; p<ngrids; p++){
       num_bytes = sizeof(_prec)*3*(ngsl)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
       CUCHK(cudaMallocHost((void**)&SL_vel[p], num_bytes));
       CUCHK(cudaMemset(SL_vel[p], 0, num_bytes));
       CUCHK(cudaMallocHost((void**)&SR_vel[p], num_bytes));
       CUCHK(cudaMemset(SR_vel[p], 0, num_bytes));
       CUCHK(cudaMallocHost((void**)&RL_vel[p], num_bytes));
       CUCHK(cudaMemset(RL_vel[p], 0, num_bytes));
       CUCHK(cudaMallocHost((void**)&RR_vel[p], num_bytes));
       CUCHK(cudaMemset(RR_vel[p], 0, num_bytes));
       num_bytes = sizeof(_prec)*3*(ngsl)*(nxt[p]+4+ngsl2)*(nzt[p]+2*align);
       CUCHK(cudaMallocHost((void**)&SF_vel[p], num_bytes));
       CUCHK(cudaMemset(SF_vel[p], 0, num_bytes));
       CUCHK(cudaMallocHost((void**)&SB_vel[p], num_bytes));
       CUCHK(cudaMemset(SB_vel[p], 0, num_bytes));
       CUCHK(cudaMallocHost((void**)&RF_vel[p], num_bytes));
       CUCHK(cudaMemset(RF_vel[p], 0, num_bytes));
       CUCHK(cudaMallocHost((void**)&RB_vel[p], num_bytes));
       CUCHK(cudaMemset(RB_vel[p], 0, num_bytes));
       num_bytes = sizeof(_prec)*(ngsl)*(nxt[p]+4+ngsl2)*(nzt[p]+2*align);
       CUCHK(cudaMalloc((void**)&d_f_u1[p], num_bytes));
       CUCHK(cudaMemset(d_f_u1[p], 0, num_bytes));
       CUCHK(cudaMalloc((void**)&d_f_v1[p], num_bytes));
       CUCHK(cudaMemset(d_f_v1[p], 0, num_bytes));
       CUCHK(cudaMalloc((void**)&d_f_w1[p], num_bytes));
       CUCHK(cudaMemset(d_f_w1[p], 0, num_bytes));
       CUCHK(cudaMalloc((void**)&d_b_u1[p], num_bytes));
       CUCHK(cudaMemset(d_b_u1[p], 0, num_bytes));
       CUCHK(cudaMalloc((void**)&d_b_v1[p], num_bytes));
       CUCHK(cudaMemset(d_b_v1[p], 0, num_bytes));
       CUCHK(cudaMalloc((void**)&d_b_w1[p], num_bytes));
       CUCHK(cudaMemset(d_b_w1[p], 0, num_bytes));
       msg_v_size_x[p] = 3*(ngsl)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
       msg_v_size_y[p] = 3*(ngsl)*(nxt[p]+4+ngsl2)*(nzt[p]+2*align);
    }

    SetDeviceConstValue(DH, DT, nxt, nyt, nzt, ngrids, fmajor, fminor, Rz, RzT);
    print_const_H(ngrids);
    fprintf(stdout, "fmajor in main = %f\n", fmajor);
    fflush(stdout);

    CUCHK(cudaStreamCreate(&stream_1));
    CUCHK(cudaStreamCreate(&stream_2));
    //cudaStreamCreate(&stream_2b);
    CUCHK(cudaStreamCreate(&stream_i));
    CUCHK(cudaStreamCreate(&stream_i2));
//    Delloc3D(tau); 

    /*Daniel - yield factor exchange*/
    if (NVE == 3){
       SL_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       SR_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       RL_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       RR_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       SF_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       SB_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       RF_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       RB_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));

       d_SL_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_SR_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_RL_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_RR_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_SF_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_SB_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_RF_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));
       d_RB_yldfac=(_prec* *) calloc(ngrids, sizeof(_prec* ));

       yldfac_msg_size_x = (int*) calloc(ngrids, sizeof(int));
       yldfac_msg_size_y = (int*) calloc(ngrids, sizeof(int));
       for (p=0; p<ngrids; p++){
	  yldfac_msg_size_x[p] = ngsl*(nyt[p]+ngsl2)*nzt[p];
	  num_bytes2 = yldfac_msg_size_x[p]*sizeof(_prec);
	  /*fprintf(stdout, "swp_msg_size_x=%d, num_bytes2=%d\n", swp_msg_size_x, num_bytes2);*/
	  CUCHK(cudaMallocHost((void**)&SL_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(SL_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMallocHost((void**)&SR_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(SR_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMallocHost((void**)&RL_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(RL_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMallocHost((void**)&RR_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(RR_yldfac[p], 0, num_bytes2));

	  CUCHK(cudaMalloc((void**) &d_SL_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_SL_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMalloc((void**) &d_SR_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_SR_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMalloc((void**) &d_RL_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_RL_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMalloc((void**) &d_RR_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_RR_yldfac[p], 0, num_bytes2));

	  yldfac_msg_size_y[p] = nxt[p]*ngsl*nzt[p];
	  num_bytes2 = yldfac_msg_size_y[p]*sizeof(_prec);
	  CUCHK(cudaMallocHost((void**)&SF_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(SF_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMallocHost((void**)&SB_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(SB_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMallocHost((void**)&RF_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(RF_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMallocHost((void**)&RB_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(RB_yldfac[p], 0, num_bytes2));

	  CUCHK(cudaMalloc((void**) &d_SF_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_SF_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMalloc((void**) &d_SB_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_SB_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMalloc((void**) &d_RF_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_RF_yldfac[p], 0, num_bytes2));
	  CUCHK(cudaMalloc((void**) &d_RB_yldfac[p], num_bytes2));
	  CUCHK(cudaMemset(d_RB_yldfac[p], 0, num_bytes2));

       }
    }

    /* Daniel: overlap zone variable exchange for DM */
    SL_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    SR_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    RL_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    RR_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    SF_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    SB_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    RF_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    RB_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));

    d_SL_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_SR_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_RL_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_RR_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_SF_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_SB_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_RF_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));
    d_RB_swap=(_prec* *) calloc(ngrids, sizeof(_prec* ));

    swp_msg_size_x = (int*) calloc(ngrids, sizeof(int));
    swp_msg_size_y = (int*) calloc(ngrids, sizeof(int));

    for (p=0; p<ngrids; p++){
       swp_msg_size_x[p] = 9*(2+ngsl+WWL)*(nyt[p]+4+ngsl2+2*WWL)*6;
       num_bytes2 = swp_msg_size_x[p]*sizeof(_prec);
       
       CUCHK(cudaMallocHost((void**)&SL_swap[p], num_bytes2));
       CUCHK(cudaMemset(SL_swap[p], 0, num_bytes2));
       CUCHK(cudaMallocHost((void**)&SR_swap[p], num_bytes2));
       CUCHK(cudaMemset(SR_swap[p], 0, num_bytes2));
       CUCHK(cudaMallocHost((void**)&RL_swap[p], num_bytes2));
       CUCHK(cudaMemset(RL_swap[p], 0, num_bytes2));
       CUCHK(cudaMallocHost((void**)&RR_swap[p], num_bytes2));
       CUCHK(cudaMemset(RR_swap[p], 0, num_bytes2));

       CUCHK(cudaMalloc((void**) &d_SL_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_SL_swap[p], 0, num_bytes2));
       CUCHK(cudaMalloc((void**) &d_SR_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_SR_swap[p], 0, num_bytes2));
       CUCHK(cudaMalloc((void**) &d_RL_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_RL_swap[p], 0, num_bytes2));
       CUCHK(cudaMalloc((void**) &d_RR_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_RR_swap[p], 0, num_bytes2));

       for (k=0; k<num_bytes2/sizeof(_prec); k++) SL_swap[p][k] = SR_swap[p][k] = RL_swap[p][k] = RR_swap[p][k] = 0.f;

       //copy zero-allocated arrays to device if GPU arrays remain uninitialized otherwise
       if (x_rank_L < 0) cudaMemcpy(d_RL_swap[p], RL_swap[p], num_bytes2, cudaMemcpyHostToDevice);
       if (x_rank_R < 0) cudaMemcpy(d_RR_swap[p], RR_swap[p], num_bytes2, cudaMemcpyHostToDevice);

       swp_msg_size_y[p] = 9*(nxt[p]+4+ngsl2)*(2+ngsl+WWL)*6;
       num_bytes2 = swp_msg_size_y[p]*sizeof(_prec);
       CUCHK(cudaMallocHost((void**)&SF_swap[p], num_bytes2));
       CUCHK(cudaMemset(SF_swap[p], 0, num_bytes2));
       CUCHK(cudaMallocHost((void**)&SB_swap[p], num_bytes2));
       CUCHK(cudaMemset(SB_swap[p], 0, num_bytes2));
       CUCHK(cudaMallocHost((void**)&RF_swap[p], num_bytes2));
       CUCHK(cudaMemset(RF_swap[p], 0, num_bytes2));
       CUCHK(cudaMallocHost((void**)&RB_swap[p], num_bytes2));
       CUCHK(cudaMemset(RB_swap[p], 0, num_bytes2));

       CUCHK(cudaMalloc((void**) &d_SF_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_SF_swap[p], 0, num_bytes2));
       CUCHK(cudaMalloc((void**) &d_SB_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_SB_swap[p], 0, num_bytes2));
       CUCHK(cudaMalloc((void**) &d_RF_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_RF_swap[p], 0, num_bytes2));
       CUCHK(cudaMalloc((void**) &d_RB_swap[p], num_bytes2));
       CUCHK(cudaMemset(d_RB_swap[p], 0, num_bytes2));

       for (k=0; k<num_bytes2/sizeof(_prec); k++) SF_swap[p][k] = SB_swap[p][k] = RF_swap[p][k] = RB_swap[p][k] = 0.f;

       //copy zero-allocated arrays to device if GPU arrays remain uninitialized otherwise
       if (y_rank_F < 0) cudaMemcpy(d_RF_swap[p], RF_swap[p], num_bytes2, cudaMemcpyHostToDevice);
       if (y_rank_B < 0) cudaMemcpy(d_RB_swap[p], RB_swap[p], num_bytes2, cudaMemcpyHostToDevice);

       /*swp_msg_size_x_l = 9*(2+ngsl+WWL)*(nytl+4+ngsl2+2*WWL); 
       swp_msg_size_y_l = 9*(nxtl+4+8*loop)*(2+4*loop+WWL)*1;*/

       intlev[p] = nzt[p] + align - 3;
    }
    swaplevmin = align + 2;
    swaplevmax = align + 7;

#if TOPO
            // Initialize grids
            grids_t grids[MAXGRIDS];
            for (p = 0; p < ngrids; p++) {
                    grids[p] = grids_init(nxt[p], nyt[p], nzt[p], coord[0],
                                          coord[1], 0, usetopo, DH[p]);
            }

            topo_t T = topo_init(usetopo, INTOPO, 
                                 rank, 
                                 x_rank_L, x_rank_R,
                                 y_rank_F, y_rank_B, coord,
                                 dim[0], dim[1],
                                 nxt[0], nyt[0], nzt[0], 
                                 DT, *DH,
                                 stream_1, stream_2, stream_i
                                 );
            topo_bind(&T, d_d1[0], d_lam[0], d_mu[0], d_xx[0], d_yy[0], d_zz[0],
                      d_xy[0], d_xz[0], d_yz[0], d_r1[0], d_r2[0], d_r3[0],
                      d_r4[0], d_r5[0], d_r6[0], d_u1[0], d_v1[0], d_w1[0],
                      d_f_u1[0], d_f_v1[0], d_f_w1[0], d_b_u1[0], d_b_v1[0],
                      d_b_w1[0], d_dcrjx[0], d_dcrjy[0], d_dcrjz[0]);
            topo_init_metrics(&T);
            int topo_vtk_mode = 0;
            if (T.use) {
                topo_init_grid(&T);

#if TOPO_USE_INCLINE_PLANE
                printf("Using incline plane geometry\n");
                _prec phi_x = 0.3;
                _prec phi_y = 0.0;
                topo_init_incline_plane(&T, phi_x, phi_y);
                topo_write_geometry_vtk(&T, 1);
#elif TOPO_USE_GAUSSIAN_HILL_AND_CANYON_XZ
                printf("Using Gaussian hill and canyon (xz) geometry\n");
                _prec3_t hill_width = {.x = 3000, .y = 0, .z = 0};
                _prec3_t hill_height = {.x = 5000, .y = 0, .z = 0};
                _prec3_t hill_center = {.x = -1e4, .y = 0, .z = 0};
                _prec3_t canyon_width = {.x = 3000, .y = 0, .z = 0};
                _prec3_t canyon_height = {.x = 5000, .y = 0, .z = 0};
                _prec3_t canyon_center = {.x = 1e4, .y = 0, .z = 0};
                topo_init_gaussian_hill_and_canyon_xz(
                    &T, hill_width, hill_height, hill_center, canyon_width,
                    canyon_height, canyon_center);
#elif TOPO_USE_GAUSSIAN_HILL_AND_CANYON
                printf("Using Gaussian hill and canyon geometry\n");
                _prec3_t hill_width = {.x = 3000, .y = 3000, .z = 0};
                _prec3_t hill_height = 5000;
                _prec3_t hill_center = {.x = -1e4, .y = -1e4, .z = 0};
                _prec3_t canyon_width = {.x = 3000, .y = 3000, .z = 0};
                _prec3_t canyon_height = 5000;
                _prec3_t canyon_center = {.x = 1e4, .y = 1e4, .z = 0};
                topo_vtk_mode = 0;
                topo_init_gaussian_hill_and_canyon(
                    &T, hill_width, hill_height, hill_center, canyon_width,
                    canyon_height, canyon_center);
#elif TOPO_USE_GAUSSIAN_HILL
                printf("Using Gaussian hill geometry\n");
                _prec3_t hill_width = {.x = 1000, .y = 1000, .z = 0};
                _prec hill_height = 1000;
                _prec3_t hill_center = {.x = 0, .y = 0, .z = 0};
                // No canyon
                _prec3_t canyon_width = {.x = 2000, .y = 2000, .z = 0};
                _prec canyon_height = 0;
                _prec3_t canyon_center = {.x = 0, .y = 0, .z = 0};
                topo_vtk_mode = 0;
                topo_init_gaussian_hill_and_canyon(
                    &T, hill_width, hill_height, hill_center, canyon_width,
                    canyon_height, canyon_center);
#elif TOPO_USE_GAUSSIAN_HILL_XZ
                printf("Using Gaussian hill geometry\n");
                _prec3_t hill_width = {.x = 1000, .y = 0, .z = 0};
                _prec hill_height = 1000;
                _prec3_t hill_center = {.x = 0, .y = 0, .z = 0};
                // No canyon
                _prec3_t canyon_width = {.x = 2000, .y = 0, .z = 0};
                _prec canyon_height = 0;
                _prec3_t canyon_center = {.x = 0, .y = 0, .z = 0};
                topo_vtk_mode = 0;
                topo_init_gaussian_hill_and_canyon_xz(
                    &T, hill_width, hill_height, hill_center, canyon_width,
                    canyon_height, canyon_center);
#elif TOPO_USE_GAUSSIAN
                printf("Using Gaussian geometry\n");
                _prec amplitude =  100 * 10;
                _prec3_t width = {.x = 1000, .y = 1e3, .z = 0};
                _prec3_t center = {.x = 0, .y = 0, .z = 0};
                 topo_init_gaussian_geometry(&T, amplitude, width, center);
#endif
                topo_init_geometry(&T);
                topo_write_geometry_vtk(&T, topo_vtk_mode);
                topo_build(&T);

            }



            if(rank == 0)printf("Initialize source and receivers\n");
            fflush(stdout);


            f_grid_t *metrics_f = NULL;
            if (T.use) {
                metrics_f = &T.metrics_f;
            }

            sources_init(SOURCEFILE, grids, ngrids, metrics_f, MCW, rank,
                         size_tot);
            receivers_init(RECVFILE, grids, ngrids, metrics_f, MCW, rank,
                           size_tot);
            if(rank == 0)printf("done.\n");
            fflush(stdout);




#if TOPO_TEST
                    topo_test_t Tt = topo_test_init(&T);
#endif
#endif

      //Initialize all variables to zero
      for (p=0; p<ngrids; p++){
        int mx = nxt[p] + 4 + ngsl2;
        int my = nyt[p] + 4 + ngsl2;
        int mz = nzt[p] + 2 * align;
        int byte_size = mx * my * mz * sizeof(_prec);
        float *dbg = malloc(byte_size);
        zeros(dbg, mx, my, mz);
        CUCHK(cudaMemcpy(d_u1[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_v1[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_w1[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_xx[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_yy[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_zz[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_xy[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_xz[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_yz[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_r1[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_r2[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_r3[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_r4[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_r5[p],dbg,byte_size,cudaMemcpyHostToDevice));
        CUCHK(cudaMemcpy(d_r6[p],dbg,byte_size,cudaMemcpyHostToDevice));

        CUCHK(cudaMemcpy(dbg,d_u1[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("u1", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_v1[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("v1", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_w1[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("w1", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_xx[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("xx", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_yy[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("yy", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_zz[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("zz", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_xy[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("xy", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_xz[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("xz", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_yz[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("yz", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_r1[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("r1", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_r2[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("r2", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_r3[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("r3", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_r4[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("r4", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_r5[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("r5", dbg, mx, my, mz, rank); 
        CUCHK(cudaMemcpy(dbg,d_r6[p],byte_size,cudaMemcpyDeviceToHost));
        check_values("r6", dbg, mx, my, mz, rank); 
        free(dbg);
     }

    size_t  cmemfreeMin;
    CUCHK(cudaMemGetInfo(&cmemfree, &cmemtotal));
    if(sizeof(size_t)==8)  {
      MPICHK(MPI_Reduce(&cmemfree, &cmemfreeMin, 1, MPI_UINT64_T, MPI_MIN, 0, MCW));
    }
    else {
      MPICHK(MPI_Reduce(&cmemfree, &cmemfreeMin, 1, MPI_UINT32_T, MPI_MIN, 0, MCW));
    }
    if(rank==0) printf("CUDA MEMORY: free = %ld\ttotal = %ld\n",cmemfreeMin,cmemtotal);
    fflush(stdout);


    if(rank==0){
      CUCHK(cudaMemGetInfo(&cmemfree, &cmemtotal));
      printf("CUDA MEMORY: Total=%ld\tAvailable=%ld\n",cmemtotal,cmemfree);
    }
    fflush(stdout);

    // FIXME: Add support for DM
    receivers_write(d_u1[0], d_v1[0], d_w1[0], 0, nt);
    if(rank ==0) printf("Read sources\n");
    sources_read(0);
    fflush(stdout);

    if(rank==0)
      fchk = fopen(CHKFILE,"a+");
//  Main Loop Starts
    if( ((NPC==0) || (NPC==2)) && (NVE==1 || NVE==3))
    {
       time_un  -= gethrtime();
       //This loop has no loverlapping because there is source input
       for(cur_step=1;cur_step<=nt;cur_step++)
       {
         if(rank==0 && cur_step%100==0) printf("Time Step =                   %ld    OF  Total Timesteps = %ld\n", cur_step, nt); 
         if(cur_step%100==0 && rank==0) printf("Time per timestep:\t%f seconds\n",(gethrtime()+time_un)/cur_step);
         fflush(stdout);
         CUCHK(cudaGetLastError());
         //cerr=cudaGetLastError();
         
         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_u1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "u1", p, cur_step, 6, rank, size);
	    dump_nonzeros(d_v1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "v1", p, cur_step, 6, rank, size);
	    dump_nonzeros(d_w1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "w1", p, cur_step, 6, rank, size);
         }
         
         if(cerr!=cudaSuccess) printf("CUDA ERROR! rank=%d before timestep: %s\n",rank,cudaGetErrorString(cerr));
         fflush(stdout);
	 //pre-post MPI Message
	 for (p=0; p<ngrids; p++){
	    PostRecvMsg_Y(RF_vel[p], RB_vel[p], MCW, request_y[p], &count_y[p], msg_v_size_y[p], y_rank_F, y_rank_B, p);
	    //PostRecvMsg_X(RL_vel[p], RR_vel[p], MCW, request_x[p], &count_x[p], msg_v_size_x[p], x_rank_L, x_rank_R, p);
	    //velocity computation in y boundary, two ghost cell regions
            if (!usetopo || p > 0) {
	    dvelcy_H(d_u1[p], d_v1[p], d_w1[p], d_xx[p],   d_yy[p],   d_zz[p],   d_xy[p],       
                     d_xz[p], d_yz[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
		     d_d1[p], nxt[p],  nzt[p],  d_f_u1[p], d_f_v1[p], d_f_w1[p], 
                     stream_1,   yfs[p],  yfe[p], y_rank_F, p);
            } else {
        
            #if TOPO_TEST
                topo_test_velfront(&Tt, &T);
            #elif TOPO
                topo_velocity_front_H(&T);
            #endif
            }
	    Cpy2Host_VY(d_f_u1[p], d_f_v1[p], d_f_w1[p],  SF_vel[p], nxt[p], nzt[p], stream_1, y_rank_F);
            if (!usetopo || p > 0) {
	    dvelcy_H(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     d_dcrjx[p], d_dcrjy[p], d_dcrjz[p], d_d1[p], nxt[p],  nzt[p],  
                     d_b_u1[p], d_b_v1[p], d_b_w1[p], stream_2, ybs[p], ybe[p], y_rank_B, p);
            } 
            else {
            #if TOPO_TEST
                topo_test_velback(&Tt, &T);
            #elif TOPO
                topo_velocity_back_H(&T);
            #endif
            }

	    Cpy2Host_VY(d_b_u1[p], d_b_v1[p], d_b_w1[p], SB_vel[p], nxt[p], nzt[p], stream_2, y_rank_B);

            if (!usetopo || p > 0) {
	    dvelcx_H_opt(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     d_dcrjx[p], d_dcrjy[p], d_dcrjz[p], d_d1[p], nyt[p], nzt[p], stream_i, xvs[p],  xve[p], p, ngrids);
            } 
            else {
            #if TOPO_TEST
                topo_test_velx(&Tt, &T);
            #elif TOPO
                topo_velocity_interior_H(&T);
            #endif

            }
         }

         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_u1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "u1", p, cur_step, 7, rank, size);
	    dump_nonzeros(d_v1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "v1", p, cur_step, 7, rank, size);
	    dump_nonzeros(d_w1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "w1", p, cur_step, 7, rank, size);
         }

         //MPI overlapping velocity computation

         //velocity communication in y direction
         CUCHK(cudaStreamSynchronize(stream_1));
	 for (p=0; p<ngrids; p++){
	    PostSendMsg_Y(SF_vel[p], SB_vel[p], MCW, request_y[p], &count_y[p], 
               msg_v_size_y[p], y_rank_F, y_rank_B, rank, Front, p);
         }
	 CUCHK(cudaStreamSynchronize(stream_2));
	 for (p=0; p<ngrids; p++){
	    PostSendMsg_Y(SF_vel[p], SB_vel[p], MCW, request_y[p], &count_y[p], 
               msg_v_size_y[p], y_rank_F, y_rank_B, rank, Back, p);
	    MPICHK(MPI_Waitall(count_y[p], request_y[p], status_y[p]));
	    Cpy2Device_VY(d_u1[p], d_v1[p], d_w1[p], d_f_u1[p], d_f_v1[p], d_f_w1[p], d_b_u1[p], d_b_v1[p], d_b_w1[p],  
                          RF_vel[p], RB_vel[p], nxt[p], nyt[p], 
			  nzt[p], stream_1, stream_2, y_rank_F, y_rank_B, p);
         }


         CUCHK(cudaDeviceSynchronize());
         /* DM: 2nd order velocity update */
	 for (p=0; p<ngrids-1; p++){
            dvelc2_H(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                d_dcrjx[p], d_dcrjy[p], d_dcrjz[p], d_d1[p], nxt[p], nyt[p], stream_i, p);
         }
         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_u1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "u1", p, cur_step, 0, rank, size);
	    dump_nonzeros(d_v1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "v1", p, cur_step, 0, rank, size);
	    dump_nonzeros(d_w1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "w1", p, cur_step, 0, rank, size);
         }

         CUCHK(cudaDeviceSynchronize());
         /*swap transition zone data on coarse grid(s)*/
	 for (p=1; p<ngrids; p++){
            Cpy2Host_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     SF_swap[p], SB_swap[p], d_SF_swap[p], d_SB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     intlev[p], intlev[p], p);
            CUCHK(cudaThreadSynchronize());
            PostRecvMsg_Y(RF_swap[p], RB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, p);
            PostSendMsg_Y(SF_swap[p], SB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, 
                     rank, Both, p);
            MPICHK(MPI_Waitall(count_y_swp[p], request_y_swp[p], status_y_swp[p]));
            Cpy2Device_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     RF_swap[p], RB_swap[p], d_RF_swap[p], d_RB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     intlev[p], intlev[p], p);
            Cpy2Host_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     SL_swap[p], SR_swap[p], d_SL_swap[p], d_SR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     intlev[p], intlev[p], p);
            CUCHK(cudaThreadSynchronize());
            PostRecvMsg_X(RL_swap[p], RR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, x_rank_R, p);
            PostSendMsg_X(SL_swap[p], SR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, x_rank_R, 
                     rank, Both, p);
            MPICHK(MPI_Waitall(count_x_swp[p], request_x_swp[p], status_x_swp[p]));
            Cpy2Device_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     RL_swap[p], RR_swap[p], d_RL_swap[p], d_RR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     intlev[p], intlev[p], p);
         }

	 for (p=1; p<ngrids; p++){
            intp3d_H(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
                   d_u1[p-1], d_v1[p-1], d_w1[p-1], d_xx[p-1], d_yy[p-1], d_zz[p-1], d_xy[p-1], d_xz[p-1], d_yz[p-1],
                   nxt[p], nyt[p], rank, stream_i, p);
         }
         for (p=0; p<ngrids; p++)
            dump_nonzeros(d_w1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "w1", p, cur_step, 1, rank, size);

         for (p=usetopo; p<ngrids-1; p++){
	    Cpy2Host_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     SF_swap[p], SB_swap[p], d_SF_swap[p], d_SB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     swaplevmin, swaplevmax, p);
	    CUCHK(cudaThreadSynchronize());
	    PostRecvMsg_Y(RF_swap[p], RB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, p);
	    PostSendMsg_Y(SF_swap[p], SB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, 
                     rank, Both, p);
	    MPICHK(MPI_Waitall(count_y_swp[p], request_y_swp[p], status_y_swp[p]));
	    Cpy2Device_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     RF_swap[p], RB_swap[p], d_RF_swap[p], d_RB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     swaplevmin, swaplevmax, p);

	    Cpy2Host_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     SL_swap[p], SR_swap[p], d_SL_swap[p], d_SR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     swaplevmin, swaplevmax, p);
	    CUCHK(cudaThreadSynchronize());
	    swaparea_update_corners(SL_swap[p], SR_swap[p], RF_swap[p], RB_swap[p], 6, WWL, nxt[p], nyt[p]);
	    PostRecvMsg_X(RL_swap[p], RR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, x_rank_R, p);
	    PostSendMsg_X(SL_swap[p], SR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, 
                     x_rank_R, rank, Both, p);
	    MPICHK(MPI_Waitall(count_x_swp[p], request_x_swp[p], status_x_swp[p]));
	    Cpy2Device_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     RL_swap[p], RR_swap[p], d_RL_swap[p], d_RR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     swaplevmin, swaplevmax, p);
         }

	 for (p=usetopo; p<ngrids-1; p++){
            swap_H(d_xx[p+1], d_yy[p+1], d_zz[p+1], d_xy[p+1], d_xz[p+1], d_yz[p+1], d_u1[p+1], d_v1[p+1], d_w1[p+1],
                   d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_u1[p], d_v1[p], d_w1[p],
                   nxt[p+1],  nyt[p+1], d_RL_swap[p], d_RR_swap[p], d_RF_swap[p], d_RB_swap[p], rank, stream_i, p);
         }

         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_u1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "u1", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_v1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "v1", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_w1[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "w1", p, cur_step, 2, rank, size);
         }

         CUCHK(cudaStreamSynchronize(stream_i));

	 for (p=0; p<ngrids; p++){
            dump_all_data(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xz[p], d_yz[p], d_xy[p], 
                     nel[p], cur_step, 0, p, rank, size);
         }

         if((rank==srcproc[0]) && (IFAULT == 4))
         {
            fprintf(stdout, "calling frcvel_H\n");
            ++source_step;
            frcvel_H(source_step, READ_STEP_GPU, maxdim, d_tpsrc[0], npsrc[0], fbc_tskp, stream_i, d_taxx[0], d_tayy[0], 
                 d_tazz[0], d_taxz[0], d_tayz[0], d_taxy[0], d_u1[0], d_v1[0], d_w1[0], -1, -1, 0);
         }
         CUCHK(cudaStreamSynchronize(stream_i));
         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_xx[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xx", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_yy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yy", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_zz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "zz", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_xy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xy", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_xz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xz", p, cur_step, 2, rank, size);
	    dump_nonzeros(d_yz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yz", p, cur_step, 2, rank, size);
         }

         for (p=0; p<ngrids; p++){ 
   	    PostRecvMsg_X(RL_vel[p], RR_vel[p], MCW, request_x[p], &count_x[p], msg_v_size_x[p], x_rank_L, x_rank_R, p);
         }


         if (NVE < 3){
            // Only called if no plasticity
   	    //stress computation in full inside region
   	    for (p=usetopo; p<ngrids; p++){
               CUCHK(cudaThreadSynchronize());
	       dstrqc_H_new(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
			d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p],
			d_u1[p], d_v1[p], d_w1[p], d_lam[p],
			d_mu[p], d_qp[p], d_coeff, d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
			nyt[p],  nzt[p],  stream_1, d_lam_mu[p],
			d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p],
			NX*grdfct[p], NPC,  coord[0], coord[1],   xss2[p],  xse2[p],
			yls[p],  yre[p], p);
            }

#if TOPO_TEST
            topo_test_stress_interior(&Tt, &T);
#elif TOPO
            topo_stress_interior_H(&T);
#endif
         }
         else {
            // FIXME: Does not work with the DM yet because not applied in the
            // swap zone.  cohesion and friction value are set to large values
            // and will therefore not change anything. However, kernels still
            // run.
   	    for (p=usetopo; p<ngrids; p++){
	       dstrqc_H_new(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
			d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p],
			d_u1[p], d_v1[p], d_w1[p], d_lam[p],
			d_mu[p], d_qp[p],d_coeff, d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
			nyt[p],  nzt[p],  stream_i, d_lam_mu[p],
			d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p],
			NX*grdfct[p], NPC,  coord[0], coord[1],   xss2[p],  xse2[p],
			yls[p],  yls2[p]-1, p);
	       dstrqc_H_new(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
			d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p],
			d_u1[p], d_v1[p], d_w1[p], d_lam[p],
			d_mu[p], d_qp[p],d_coeff, d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
			nyt[p],  nzt[p],  stream_i2, d_lam_mu[p],
			d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p],
			NX*grdfct[p], NPC,  coord[0], coord[1],   xss2[p],  xse2[p],
			yre2[p]+1, yre[p], p);
           }
         }

         //dump_all_stresses(d_xx, d_yy, d_zz, d_xz, d_yz, d_xy, nel, 'u', cur_step, 0, rank, size);

   	 for (p=0; p<ngrids; p++){
	    Cpy2Host_VX(d_u1[p], d_v1[p], d_w1[p], SL_vel[p], nxt[p], nyt[p], nzt[p], stream_1, x_rank_L, Left);
	    Cpy2Host_VX(d_u1[p], d_v1[p], d_w1[p], SR_vel[p], nxt[p], nyt[p], nzt[p], stream_2, x_rank_R, Right);
         }

         //velocity communication in x direction
         CUCHK(cudaStreamSynchronize(stream_1));
   	 for (p=0; p<ngrids; p++){
	    PostSendMsg_X(SL_vel[p], SR_vel[p], MCW, request_x[p], &count_x[p], 
	       msg_v_size_x[p], x_rank_L, x_rank_R, rank, Left, p);
         }
         CUCHK(cudaStreamSynchronize(stream_2));



   	 for (p=0; p<ngrids; p++){
	    PostSendMsg_X(SL_vel[p], SR_vel[p], MCW, request_x[p], &count_x[p], 
               msg_v_size_x[p], x_rank_L, x_rank_R, rank, Right, p);
	    MPICHK(MPI_Waitall(count_x[p], request_x[p], status_x[p]));

	    Cpy2Device_VX(d_u1[p], d_v1[p], d_w1[p], RL_vel[p], RR_vel[p], nxt[p], nyt[p], nzt[p], 
               stream_1, stream_2, x_rank_L, x_rank_R);

            if (!usetopo || p > 0) { 
            // Stress computation is split left to right. This kernel handles
            // left boundary
	    dstrqc_H_new(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
		     d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p],
		     d_u1[p], d_v1[p], d_w1[p], d_lam[p],
		     d_mu[p], d_qp[p],d_coeff, d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
		     nyt[p],  nzt[p],  stream_1, d_lam_mu[p],
		     d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p],
		     NX*grdfct[p], NPC,  coord[0], coord[1],   xss1[p],  xse1[p],
		     yls[p],  yre[p], p);
            // Stress computation for the right boundary. Look at the xss3, xse3
            // bounds.
	    dstrqc_H_new(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
		     d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p],
		     d_u1[p], d_v1[p], d_w1[p], d_lam[p],
		     d_mu[p], d_qp[p],d_coeff, d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
		     nyt[p],  nzt[p],  stream_2, d_lam_mu[p],
		     d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p],
		     NX*grdfct[p], NPC,  coord[0], coord[1],   xss3[p],  xse3[p],
		     yls[p],  yre[p], p);
            } else {
#if TOPO_TEST
                    topo_test_stress_sides(&Tt, &T);
#elif TOPO
                    topo_stress_left_H(&T);
                    topo_stress_right_H(&T);
#endif
            }
         }
         CUCHK(cudaDeviceSynchronize());
         
         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_xx[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xx", p, cur_step, 8, rank, size);
	    dump_nonzeros(d_yy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yy", p, cur_step, 8, rank, size);
	    dump_nonzeros(d_zz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "zz", p, cur_step, 8, rank, size);
	    dump_nonzeros(d_xy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xy", p, cur_step, 8, rank, size);
	    dump_nonzeros(d_xz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xz", p, cur_step, 8, rank, size);
	    dump_nonzeros(d_yz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yz", p, cur_step, 8, rank, size);
         }

         for (p=0; p<ngrids-1; p++){
            if (!usetopo || p > 0) { 
                // Not applied in the last grid. Second order stress update in
                // the interface regions 
                dstrqc2_H(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],    
                          d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p], d_u1[p], d_v1[p], d_w1[p],
                          d_lam[p], d_mu[p], d_qp[p], d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p], 
                          nxt[p], nyt[p], stream_i, d_coeff, d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p], 
                          xss1[p],  xse3[p], yls[p],  yre[p], p);
            }
	    /*print_nan_H(d_xx[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), "xx");
	    print_nan_H(d_yy[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), "yy");
	    print_nan_H(d_zz[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), "zz");
	    print_nan_H(d_xy[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), "xy");
	    print_nan_H(d_xz[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), "xz");
	    print_nan_H(d_yz[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), "yz");*/
         }

         #if TOPO_TEST
             topo_test_stress(&Tt, &T);
         #endif

         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_xx[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xx", p, cur_step, 3, rank, size);
	    dump_nonzeros(d_yy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yy", p, cur_step, 3, rank, size);
	    dump_nonzeros(d_zz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "zz", p, cur_step, 3, rank, size);
	    dump_nonzeros(d_xy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xy", p, cur_step, 3, rank, size);
	    dump_nonzeros(d_xz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xz", p, cur_step, 3, rank, size);
	    dump_nonzeros(d_yz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yz", p, cur_step, 3, rank, size);
         }

         CUCHK(cudaDeviceSynchronize());

         /*swap transition zone data on coarse grid(s)*/
	 for (p=1; p<ngrids; p++){
            Cpy2Host_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     SF_swap[p], SB_swap[p], d_SF_swap[p], d_SB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     intlev[p], intlev[p], p);
            CUCHK(cudaThreadSynchronize());
            PostRecvMsg_Y(RF_swap[p], RB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, p);
            PostSendMsg_Y(SF_swap[p], SB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, 
                     rank, Both, p);
            MPICHK(MPI_Waitall(count_y_swp[p], request_y_swp[p], status_y_swp[p]));
            Cpy2Device_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     RF_swap[p], RB_swap[p], d_RF_swap[p], d_RB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     intlev[p], intlev[p], p);
            Cpy2Host_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     SL_swap[p], SR_swap[p], d_SL_swap[p], d_SR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     intlev[p], intlev[p], p);
            CUCHK(cudaThreadSynchronize());
            PostRecvMsg_X(RL_swap[p], RR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, x_rank_R, p);
            PostSendMsg_X(SL_swap[p], SR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, x_rank_R, 
                     rank, Both, p);
            MPICHK(MPI_Waitall(count_x_swp[p], request_x_swp[p], status_x_swp[p]));
            Cpy2Device_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
                     RL_swap[p], RR_swap[p], d_RL_swap[p], d_RR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     intlev[p], intlev[p], p);
         }

         sources_read(cur_step);
         if (T.use) {
                sources_add_curvilinear(d_xx[0], d_yy[0], d_zz[0], d_xy[0], d_xz[0],
                               d_yz[0], cur_step, DH[0], DT, &T.metrics_f, &T.metrics_g);
         } else {
         sources_add_cartesian(d_xx[0], d_yy[0], d_zz[0], d_xy[0], d_xz[0],
                               d_yz[0], cur_step, DH[0], DT);
         }

         //update source input
         if ((IFAULT < 4) && (cur_step<NST)) {
            CUCHK(cudaThreadSynchronize());
            ++source_step;
            for (p=0; p<ngrids; p++){ 
               if (rank==srcproc[p])
	       addsrc_H(source_step, READ_STEP_GPU, maxdim, d_tpsrc[p], npsrc[p], stream_i, 
			d_taxx[p], d_tayy[p], d_tazz[p], d_taxz[p], d_tayz[p], d_taxy[p],
			d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_yz[p], d_xz[p], p);
            }
	    for (p=0; p<ngrids; p++){
	       dump_nonzeros(d_xx[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xx", p, cur_step, 1, rank, size);
	       dump_nonzeros(d_yy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yy", p, cur_step, 1, rank, size);
	       dump_nonzeros(d_zz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "zz", p, cur_step, 1, rank, size);
	       dump_nonzeros(d_xy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xy", p, cur_step, 1, rank, size);
	       dump_nonzeros(d_xz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xz", p, cur_step, 1, rank, size);
	       dump_nonzeros(d_yz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yz", p, cur_step, 1, rank, size);
	    }
         }
         else if (IFAULT == 5) {
            CUCHK(cudaThreadSynchronize());
            for (p=usetopo; p<ngrids; p++){ 
               if (rank==srcproc[p])
	       addkinsrc_H(cur_step, maxdim, d_tpsrc[p], npsrc[p], stream_i, d_mu[p],
			d_taxx[p], d_tayy[p], d_tazz[p], d_taxz[p], d_tayz[p], d_taxy[p],
			d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_yz[p], d_xz[p], d_mom[p], d_srcfilt_d[p], p);
            }
         }
         else if ((IFAULT == 6) && (cur_step<NST)) {
            CUCHK(cudaThreadSynchronize());
	    p=ngrids-1;
	    addplanesrc_H(cur_step, maxdim, NST, stream_i, d_mu[p],d_lam[p], ND*grdfct[p], nxt[p], nyt[p],
		     d_taxx[p], d_tayy[p], d_tazz[p],
		     d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_yz[p], d_xz[p], p);
         }

         CUCHK(cudaThreadSynchronize());

	 for (p=1; p<ngrids; p++){
            intp3d_H(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
                   d_u1[p-1], d_v1[p-1], d_w1[p-1], d_xx[p-1], d_yy[p-1], d_zz[p-1], d_xy[p-1], d_xz[p-1], d_yz[p-1],
                   nxt[p], nyt[p], rank, stream_i, p);
         }
 
         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_xz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xz", p, cur_step, 4, rank, size);
	    dump_nonzeros(d_yz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yz", p, cur_step, 4, rank, size);
         }

         for (p=0; p<ngrids-1; p++){
	    Cpy2Host_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     SF_swap[p], SB_swap[p], d_SF_swap[p], d_SB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     swaplevmin, swaplevmax, p);
	    CUCHK(cudaThreadSynchronize());
	    PostRecvMsg_Y(RF_swap[p], RB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, p);
	    PostSendMsg_Y(SF_swap[p], SB_swap[p], MCW, request_y_swp[p], count_y_swp+p, swp_msg_size_y[p], y_rank_F, y_rank_B, 
                     rank, Both, p);
	    MPICHK(MPI_Waitall(count_y_swp[p], request_y_swp[p], status_y_swp[p]));
	    Cpy2Device_swaparea_Y(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     RF_swap[p], RB_swap[p], d_RF_swap[p], d_RB_swap[p], nxt[p], stream_i, stream_i, y_rank_F, y_rank_B, 
                     swaplevmin, swaplevmax, p);

	    Cpy2Host_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     SL_swap[p], SR_swap[p], d_SL_swap[p], d_SR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     swaplevmin, swaplevmax, p);
	    CUCHK(cudaThreadSynchronize());
	    swaparea_update_corners(SL_swap[p], SR_swap[p], RF_swap[p], RB_swap[p], 6, WWL, nxt[p], nyt[p]);
	    PostRecvMsg_X(RL_swap[p], RR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, x_rank_R, p);
	    PostSendMsg_X(SL_swap[p], SR_swap[p], MCW, request_x_swp[p], count_x_swp+p, swp_msg_size_x[p], x_rank_L, 
                     x_rank_R, rank, Both, p);
	    MPICHK(MPI_Waitall(count_x_swp[p], request_x_swp[p], status_x_swp[p]));
	    Cpy2Device_swaparea_X(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], 
		     RL_swap[p], RR_swap[p], d_RL_swap[p], d_RR_swap[p], nyt[p], stream_i, stream_i, x_rank_L, x_rank_R, 
                     swaplevmin, swaplevmax, p);
         }

         /*dump_local_variable(SF_swap[0], swp_msg_size_y[0], "SF_swap", 'h', cur_step, 4, rank, size);
         dump_local_variable(RB_swap[0], swp_msg_size_y[0], "RB_swap", 'h', cur_step, 4, rank, size);*/

	 for (p=0; p<ngrids-1; p++){
            swap_H(d_xx[p+1], d_yy[p+1], d_zz[p+1], d_xy[p+1], d_xz[p+1], d_yz[p+1], d_u1[p+1], d_v1[p+1], d_w1[p+1],
                   d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_u1[p], d_v1[p], d_w1[p],
                   nxt[p+1],  nyt[p+1], d_RL_swap[p], d_RR_swap[p], d_RF_swap[p], d_RB_swap[p], rank, stream_i, p);
         }

         for (p=0; p<ngrids; p++){
	    dump_nonzeros(d_xx[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xx", p, cur_step, 5, rank, size);
	    dump_nonzeros(d_yy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yy", p, cur_step, 5, rank, size);
	    dump_nonzeros(d_zz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "zz", p, cur_step, 5, rank, size);
	    dump_nonzeros(d_xy[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xy", p, cur_step, 5, rank, size);
	    dump_nonzeros(d_xz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "xz", p, cur_step, 5, rank, size);
	    dump_nonzeros(d_yz[p], nxt[p]+4+8*loop, nyt[p]+4+8*loop, nzt[p]+2*align, "yz", p, cur_step, 5, rank, size);
         }

         CUCHK(cudaStreamSynchronize(stream_i));

	 for (p=0; p<ngrids-1; p++){
            dump_all_data(d_u1[p], d_v1[p], d_w1[p], d_xx[p], d_yy[p], d_zz[p], d_xz[p], d_yz[p], d_xy[p], 
                     nel[p], cur_step, 1, p, rank, size);
         }

         // plasticity related calls:
         if(NVE==3){
           
           CUCHK(cudaDeviceSynchronize());

           //cudaStreamSynchronize(stream_i);
           for (p=usetopo; p<ngrids; p++){
	      PostRecvMsg_Y(RF_yldfac[p], RB_yldfac[p], MCW, request_y_yldfac[p], &count_y_yldfac[p], 
		 yldfac_msg_size_y[p], y_rank_F, y_rank_B, p);
	      PostRecvMsg_X(RL_yldfac[p], RR_yldfac[p], MCW, request_x_yldfac[p], &count_x_yldfac[p], 
		 yldfac_msg_size_x[p], x_rank_L, x_rank_R, p);

	      //yield factor computation, front and back
	      drprecpc_calc_H_opt(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_mu[p], d_d1[p],
		 d_sigma2[p], d_yldfac[p],d_cohes[p], d_phi[p], d_neta[p], 
		 nzt[p], xlsp[p], xrep[p], ylsp[p], ylsp[p]+ngsl, stream_1, p);
	      drprecpc_calc_H_opt(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_mu[p], d_d1[p],
		 d_sigma2[p], d_yldfac[p],d_cohes[p], d_phi[p], d_neta[p], 
		 nzt[p], xlsp[p], xrep[p], yrep[p]-ngsl, yrep[p], stream_2, p);
	      update_yldfac_buffer_y_H(d_yldfac[p], d_SF_yldfac[p], d_SB_yldfac[p], nxt[p], nzt[p], 
		 stream_1, stream_2, y_rank_F, y_rank_B, p);
           }
           CUCHK(cudaStreamSynchronize(stream_1));
           CUCHK(cudaStreamSynchronize(stream_2));

           for (p=usetopo; p<ngrids; p++){
	      Cpy2Host_yldfac_Y(d_yldfac[p],  SF_yldfac[p], SB_yldfac[p], d_SF_yldfac[p], d_SB_yldfac[p], 
		   nxt[p], nzt[p], stream_1, stream_2, y_rank_F, y_rank_B, p);
           }

	   /*compute Stress in remaining part of inner region*/
           for (p=usetopo; p<ngrids; p++){
	      dstrqc_H_new(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p],
			d_r1[p], d_r2[p], d_r3[p], d_r4[p], d_r5[p], d_r6[p],
			d_u1[p], d_v1[p], d_w1[p], d_lam[p],
			d_mu[p], d_qp[p],d_coeff, d_qs[p], d_dcrjx[p], d_dcrjy[p], d_dcrjz[p],
			nyt[p],  nzt[p],  stream_i, d_lam_mu[p],
			d_vx1[p], d_vx2[p], d_ww[p], d_wwo[p],
			NX*grdfct[p], NPC,  coord[0], coord[1],   xss2[p],  xse2[p],
			yls2[p], yre2[p], p);
           }
 
           CUCHK(cudaStreamSynchronize(stream_1));
           CUCHK(cudaStreamSynchronize(stream_2));
           for (p=usetopo; p<ngrids; p++){
	      PostSendMsg_Y(SF_yldfac[p], SB_yldfac[p], MCW, request_y_yldfac[p], &count_y_yldfac[p], 
		 yldfac_msg_size_y[p], y_rank_F, y_rank_B, rank, Both, p);
	      MPICHK(MPI_Waitall(count_y_yldfac[p], request_y_yldfac[p], status_y_yldfac[p]));

	      //cudaStreamSynchronize(stream_i);
	      //left and right
	      drprecpc_calc_H_opt(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_mu[p], d_d1[p],
		 d_sigma2[p], d_yldfac[p],d_cohes[p], d_phi[p], d_neta[p], 
		 nzt[p], xlsp[p], xlsp[p]+ngsl, ylsp[p], yrep[p], stream_1, p);
	      drprecpc_calc_H_opt(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_mu[p], d_d1[p],
		 d_sigma2[p], d_yldfac[p],d_cohes[p], d_phi[p], d_neta[p], 
		 nzt[p], xrep[p]-ngsl, xrep[p], ylsp[p], yrep[p], stream_2, p);

	      Cpy2Device_yldfac_Y(d_yldfac[p], RF_yldfac[p], RB_yldfac[p], d_RF_yldfac[p], d_RB_yldfac[p], 
                         nxt[p], nzt[p], stream_1, stream_2, y_rank_F, y_rank_B, p);
	      update_yldfac_buffer_x_H(d_yldfac[p], d_SL_yldfac[p], d_SR_yldfac[p], nyt[p], nzt[p], 
                 stream_1, stream_2, x_rank_L, x_rank_R, p);
	   }

           CUCHK(cudaDeviceSynchronize());
           
           for (p=usetopo; p<ngrids; p++){
	      Cpy2Host_yldfac_X(d_yldfac[p],  SL_yldfac[p], SR_yldfac[p], d_SL_yldfac[p], d_SR_yldfac[p], 
		   nyt[p], nzt[p], stream_1, stream_2, x_rank_L, x_rank_R, p);

	      //compute yield factor in inside of subdomain
	      drprecpc_calc_H_opt(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_mu[p], d_d1[p],
		 d_sigma2[p], d_yldfac[p],d_cohes[p], d_phi[p], d_neta[p], 
		 nzt[p], xlsp[p]+ngsl, xrep[p]-ngsl, ylsp[p]+ngsl, yrep[p]-ngsl, stream_i, p); /*xrep-ngsl*/
           }

           //dump_variable(d_yldfac, nel, "yldfac", 'u', cur_step, 0, rank, size);

           //dump_variable(d_yldfac, nel, "yldfac", 'u', cur_step, 1, rank, size);

           CUCHK(cudaStreamSynchronize(stream_1));
           CUCHK(cudaStreamSynchronize(stream_2));
           //cudaStreamSynchronize(stream_2b);
           //cudaThreadSynchronize();
           for (p=usetopo; p<ngrids; p++){
	      PostSendMsg_X(SL_yldfac[p], SR_yldfac[p], MCW, request_x_yldfac[p], &count_x_yldfac[p], 
		 yldfac_msg_size_x[p], x_rank_L, x_rank_R, rank, Both, p);
	      MPICHK(MPI_Waitall(count_x_yldfac[p], request_x_yldfac[p], status_x_yldfac[p]));
	      Cpy2Device_yldfac_X(d_yldfac[p], RL_yldfac[p], RR_yldfac[p], d_RL_yldfac[p], d_RR_yldfac[p], 
		 nyt[p], nzt[p], stream_1, stream_2, x_rank_L, x_rank_R, p);
           }

           //wait until all streams have completed, including stream_i working on the inside part
           CUCHK(cudaThreadSynchronize());
           //dump_variable(d_yldfac, nel, "yldfac", 'u', cur_step, 2, rank, size);

           for (p=usetopo; p<ngrids; p++){
	      drprecpc_app_H(d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_xz[p], d_yz[p], d_mu[p],
		 d_sigma2[p], d_yldfac[p], 
		 nzt[p], xlsp[p], xrep[p], ylsp[p], yrep[p], stream_i, p);
           }
         }

         //update source input
         /*if ((IFAULT < 4) && (cur_step<NST)) {
            CUCHK(cudaThreadSynchronize());
            ++source_step;
            for (p=0; p<ngrids; p++){ 
               if (rank==srcproc[p])
	       addsrc_H(source_step, READ_STEP_GPU, maxdim, d_tpsrc[p], npsrc[p], stream_i, 
			d_taxx[p], d_tayy[p], d_tazz[p], d_taxz[p], d_tayz[p], d_taxy[p],
			d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_yz[p], d_xz[p], p);
            }
         }
         if (IFAULT == 5) {
            CUCHK(cudaThreadSynchronize());
            for (p=0; p<ngrids; p++){ 
               if (rank==srcproc[p])
	       addkinsrc_H(cur_step, maxdim, d_tpsrc[p], npsrc[p], stream_i, d_mu[p],
			d_taxx[p], d_tayy[p], d_tazz[p], d_taxz[p], d_tayz[p], d_taxy[p],
			d_xx[p], d_yy[p], d_zz[p], d_xy[p], d_yz[p], d_xz[p], d_mom[p], p);
            }
         }*/

         CUCHK(cudaThreadSynchronize());
         if (cur_step < -1){
            for (p=usetopo; p<ngrids; p++)
   	       //print_nonzero_H(d_xx[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), p);
   	       print_nonzero_mat_H(d_xx[p], (nxt[p]+4+ngsl2), (nyt[p]+4+ngsl2), (nzt[p]+2*align), p, 
                 d_d1[p], d_mu[p], d_lam[p], d_qp[p], d_qs[p], rank);
         }
 
         //apply free surface boundary conditions (Daniel)
         CUCHK(cudaDeviceSynchronize());

         if (!usetopo) {
         fstr_H(d_zz[0], d_xz[0], d_yz[0], stream_i, xls[0], xre[0], yls[0], yre[0]);
         }
         CUCHK(cudaDeviceSynchronize());

         // FIXME: Add support for DM
         receivers_write(d_u1[0], d_v1[0], d_w1[0], cur_step, nt);

         if(cur_step%NTISKP == 0){
#ifdef TOPO
    topo_write_vtk(&T, cur_step, topo_vtk_mode);
#endif


          #ifndef SEISMIO

          for (p=0; p<ngrids; p++){
          //for (p=0; p<1; p++){
             if (grid_output[p]){
		num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
		if(!rank) time_gpuio_tmp = -gethrtime();
		CUCHK(cudaMemcpy(&u1[p][0][0][0],d_u1[p],num_bytes,cudaMemcpyDeviceToHost));
		CUCHK(cudaMemcpy(&v1[p][0][0][0],d_v1[p],num_bytes,cudaMemcpyDeviceToHost));
		CUCHK(cudaMemcpy(&w1[p][0][0][0],d_w1[p],num_bytes,cudaMemcpyDeviceToHost));

		//added for plasticity
		if (NVE == 3) 
		   CUCHK(cudaMemcpy(&neta[p][0][0][0],d_neta[p],num_bytes,cudaMemcpyDeviceToHost));

                if(usechecksum) {
                        char msg[512];
                        sprintf(msg, "%06ld ", cur_step);
                        checksum_update(&checksum, &u1[p][0][0][0], num_bytes);
                        checksum_update(&checksum, &v1[p][0][0][0], num_bytes);
                        checksum_update(&checksum, &w1[p][0][0][0], num_bytes);
                        checksum_write(&checksum, msg); 
                }


		idtmp = ((cur_step/NTISKP+WRITE_STEP-1)%WRITE_STEP);
		idtmp = idtmp*rec_nxt[p]*rec_nyt[p]*rec_nzt[p];
		tmpInd = idtmp;
		//for(k=nzt[p]+align-1 - rec_nbgz[p]; k>=nzt[p]+align-1 - rec_nedz[p]; k=k-NSKPZ[p])
		for(k=rec_nbgz[p]; k<=rec_nedz[p]; k=k+NSKPZ[p])
		  for(j=2+ngsl + rec_nbgy[p]; j<=2+ngsl + rec_nedy[p]; j=j+NSKPY[p])
		    for(i=2+ngsl + rec_nbgx[p]; i<=2+ngsl + rec_nedx[p]; i=i+NSKPX[p])
		    {
                      if (FOLLOWBATHY==1 && p == 0) ko=bathy[i][j] - k;
                      else ko=nzt[p]+align-1-k;
		      Bufx[p][tmpInd] = u1[p][i][j][ko];
		      Bufy[p][tmpInd] = v1[p][i][j][ko];
		      Bufz[p][tmpInd] = w1[p][i][j][ko];
		      if (NVE == 3) {
			 Bufeta[p][tmpInd] = neta[p][i][j][ko];
		      }
		      tmpInd++;
		    }
		if((cur_step/NTISKP)%WRITE_STEP == 0){
		  CUCHK(cudaThreadSynchronize());
                  #ifndef NOBGIO
                  outsize=rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP;
                  time(&time1);
                  MPICHK(MPI_Send(Bufx[p], outsize, _mpi_prec, rank+2*size, MPIRANKIO+30, MPI_COMM_WORLD));
                  time(&time2);
                  if (rank==0 && p==0)
                     fprintf(stdout, "Wait time for sending output (): %5.f seconds.\n", difftime(time2, time1));
                  MPICHK(MPI_Send(Bufy[p], outsize, _mpi_prec, rank+2*size, MPIRANKIO+31, MPI_COMM_WORLD));
                  MPICHK(MPI_Send(Bufz[p], outsize, _mpi_prec, rank+2*size, MPIRANKIO+32, MPI_COMM_WORLD));
                  if (NVE ==3) 
                    MPICHK(MPI_Send(Bufeta[p], outsize, _mpi_prec, rank+2*size, MPIRANKIO+33, MPI_COMM_WORLD));
                  #else
		  sprintf(filename, "%s_%1d_%07ld", filenamebasex, p, cur_step);
		  err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
                  //error_check(err, "MPI_File_open X");
		  err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
                  //error_check(err, "MPI_File_set_view X");
		  err = MPI_File_write_all(fh, Bufx[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP, _mpi_prec, &filestatus);
                  //error_check(err, "MPI_File_write X");

		  err = MPI_File_close(&fh);
                  //error_check(err, "MPI_File_close X");

		  sprintf(filename, "%s_%1d_%07ld", filenamebasey, p, cur_step);
		  err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
		  err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
		  err = MPI_File_write_all(fh, Bufy[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP, _mpi_prec, &filestatus);
		  err = MPI_File_close(&fh);
		  sprintf(filename, "%s_%1d_%07ld", filenamebasez, p, cur_step);
		  err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
		  err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
		  err = MPI_File_write_all(fh, Bufz[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP, _mpi_prec, &filestatus);
		  err = MPI_File_close(&fh);
		  //saves the plastic shear work
		  if (NVE == 3) {
		     sprintf(filename, "%s_%1d_%07ld", filenamebaseeta, p, cur_step);
		     err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
		     err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype[p], "native", MPI_INFO_NULL);
		     err = MPI_File_write_all(fh, Bufeta[p], rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*WRITE_STEP, _mpi_prec, &filestatus);
		     err = MPI_File_close(&fh);
		  }
                  #endif
		}
          }
          //else 
            //cudaThreadSynchronize();
          #else
          for (p=0; p<ngrids; p++){
	     num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
	     if(!rank && p==0) time_gpuio_tmp = -gethrtime();
	     CUCHK(cudaMemcpy(&u1[p][0][0][0],d_u1[p],num_bytes,cudaMemcpyDeviceToHost));
	     CUCHK(cudaMemcpy(&v1[p][0][0][0],d_v1[p],num_bytes,cudaMemcpyDeviceToHost));
	     CUCHK(cudaMemcpy(&w1[p][0][0][0],d_w1[p],num_bytes,cudaMemcpyDeviceToHost));
	     //added for plasticity
	     if (NVE == 3) 
		CUCHK(cudaMemcpy(&neta[p][0][0][0],d_neta[p],num_bytes,cudaMemcpyDeviceToHost));

	     num_bytes = sizeof(_prec)*(nxt[p])*(nyt[p])*(nzt[p]);
             Bufx[0]=(_prec* ) malloc(num_bytes);
             Bufy[0]=(_prec* ) malloc(num_bytes);
             Bufz[0]=(_prec* ) malloc(num_bytes);
             Bufeta[0]=(_prec* ) malloc(num_bytes);

	     tmpInd = 0;
	     for(k=nzt[p]+align-1; k>=align; k--) {
		for(j=2+ngsl; j<2+ngsl+nyt[p]; j++) {
		   for(i=2+ngsl; i<2+ngsl+nxt[p]; i++) {
		      Bufx[0][tmpInd] = u1[p][i][j][k];
		      Bufy[0][tmpInd] = v1[p][i][j][k];
		      Bufz[0][tmpInd] = w1[p][i][j][k];
		      if (NVE == 3) Bufeta[0][tmpInd] = neta[p][i][j][k];
		      tmpInd++;
		   }
		}
	     }
 
             /*seism_write(&seism_filex[p], &u1[p][0][0][0], &err);
             seism_write(&seism_filey[p], &v1[p][0][0][0], &err);
             seism_write(&seism_filez[p], &w1[p][0][0][0], &err);
             if (NVE == 3) seism_write(&seism_fileeta[p], &neta[p][0][0][0], &err);*/

             seism_write(&seism_filex[p], Bufx[0], &err);
             seism_write(&seism_filey[p], Bufy[0], &err);
             seism_write(&seism_filez[p], Bufz[0], &err);
             if (NVE == 3) seism_write(&seism_fileeta[p], &neta[p][0][0][0], &err);
             
             free(Bufx[0]);
             free(Bufy[0]);
             free(Bufz[0]);
             if (NVE == 3) free(Bufeta[0]);
          }
          #endif

          // write-statistics to chk file:
          if(rank==0){
            if (NPC < 2){ /* for periodic BCs, ND may be larger than nxt - Daniel */
	       i = ND+2+ngsl;
	       j = i;
            }
            else i = j = 2+ngsl;
            
            k = nzt[0]+align-1-ND;
            fprintf(fchk,"%ld :\t%e\t%e\t%e\n",cur_step,u1[0][i][j][k],v1[0][i][j][k],w1[0][i][j][k]);
            fflush(fchk);
          }
         }
         }
         //else
          //cudaThreadSynchronize();

          if((cur_step<(NST*fbc_tskp)-1) && (IFAULT >= 2) && 
                   ((cur_step+1)%(READ_STEP_GPU*fbc_tskp)== 0)){
            printf("%d) Read new source from CPU.\n",rank);
            if((cur_step+1)%(READ_STEP*fbc_tskp) == 0){
	       printf("%d) Read new source from file.\n",rank);
	       if (IFAULT == 2) 
		  for (p=0; p<ngrids; p++){
		     if (rank==srcproc[p]) {
			sprintf(insrcgrid, "%s_%d", INSRC, p);
			sprintf(insrc_i2_grid, "%s_%d", INSRC_I2, p);
			read_src_ifault_2(rank, READ_STEP,
			  insrcgrid, insrc_i2_grid,
			  maxdim, coord, NZ[p],
			  nxt[p], nyt[p], nzt[p],
			  npsrc+p, srcproc+p,
			  tpsrc+p, taxx+p, tayy+p, tazz+p,
			  taxz+p, tayz+p, taxy+p, (cur_step+1)/READ_STEP+1);
		     }
		 }
	       else if ((IFAULT == 4) && (rank==srcproc[0])) {
		  read_src_ifault_4(rank, READ_STEP,
		    INSRC, maxdim, coord, NZ[0],
		    nxt[0], nyt[0], nzt[0],
		    npsrc, srcproc,
		    tpsrc, taxx, tayy, tazz, cur_step+2,
		    fbc_ext, fbc_off, fbc_pmask, fbc_extl, fbc_dim, 
		    &fbc_seismio, &fbc_tskp, NST, size);
	       }
            }
            if (rank==srcproc[0]) printf("%d) SOURCE: taxx,yy,zz:%e,%e,%e\n",rank,
                taxx[0][cur_step%READ_STEP],tayy[0][cur_step%READ_STEP],tazz[0][cur_step%READ_STEP]);
            // Synchronous copy!
            
            for (p=0; p<ngrids; p++){
               if (rank == srcproc[p])
		  Cpy2Device_source(npsrc[p], READ_STEP_GPU,
		    (cur_step+1) % (READ_STEP*fbc_tskp) / fbc_tskp,
		    taxx[p], tayy[p], tazz[p],
		    taxz[p], tayz[p], taxy[p],
		    d_taxx[p], d_tayy[p], d_tazz[p],
		    d_taxz[p], d_tayz[p], d_taxy[p], IFAULT);
            }
            source_step = 0;
          }       
       }
       time_un += gethrtime();
    } 

    if (IFAULT == 5){
       for (p=usetopo; p<ngrids; p++){
	  if (rank==srcproc[p]) {
	     num_bytes = npsrc[p]*sizeof(_prec);
	     fprintf(stdout, "num_bytes=%ld\n", num_bytes);
	     CUCHK(cudaMemcpy(mom[p], d_mom[p], num_bytes, cudaMemcpyDeviceToHost));
	     for (n=0; n<npsrc[p]; n++) {
		   /*fprintf(stdout, "mom[%d]=%e\n", n, mom[p][n]);*/
		   tmom += mom[p][n];
		}
	     }
       }
       fprintf(stdout, "rank %d: moment=%e\n", rank, tmom);
       MPICHK(MPI_Allreduce(&tmom, &gmom, 1, _mpi_prec, MPI_SUM, MCW));
       mag= 2./3. * (log10f(gmom) - 9.1);
       if (rank==0) fprintf(stdout, "Total M0=%e, Mw=%4.1f\n", gmom, mag);
       //if (rank==0) fprintf(stdout, "moment of source node 19132: %e\n", mom[0][19132]);
    }

    if(rank==0){
      fprintf(fchk,"END\n");
      fclose(fchk);
    }

    #ifdef SEISMIO
    for (p=0; p<ngrids; p++){
       seism_file_close(seism_filex+p, &err);
       seism_file_close(seism_filey+p, &err);
       seism_file_close(seism_filez+p, &err);
       seism_file_close(seism_fileeta+p, &err);
    }
    #endif

    //This should save the final plastic strain tensor at the end of the simulation
    if (NVE == 3){
      #ifndef SEISMIO
      Bufeta = (Grid1D*) calloc(ngrids, sizeof(Grid1D));
      for (p=usetopo; p<ngrids; p++){
	 fprintf(stdout, "copying plastic strain back to CPU\n");
	 num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
	 CUCHK(cudaMemcpy(&neta[p][0][0][0],d_neta[p],num_bytes,cudaMemcpyDeviceToHost));
	 tmpInd = 0;

	 rec_NZ[p] = (NEDZ_EP-NBGZ[p])/NSKPZ[p]+1;
	 calcRecordingPoints(&rec_nbgx[p], &rec_nedx[p], &rec_nbgy[p], &rec_nedy[p], 
	   &rec_nbgz[p], &rec_nedz[p], &rec_nxt[p], &rec_nyt[p], &rec_nzt[p], &displacement[p],
	   (long int)nxt[p],(long int)nyt[p],(long int)nzt[p], rec_NX[p], rec_NY[p], rec_NZ[p], 
	   NBGX[p],NEDX[p],NSKPX[p], NBGY[p],NEDY[p],NSKPY[p], NBGZ[p],NEDZ_EP,NSKPZ[p], coord);
	printf("%d = (%d,%d)) NX,NY,NZ=%d,%d,%d\nnxt,nyt,nzt=%d,%d,%d\nrec_N=(%d,%d,%d)\nrec_nxt,=%d,%d,%d\nNBGX,SKP,END=(%d:%d:%d),(%d:%d:%d),(%d:%d:%d)\nrec_nbg,ed=(%d,%d),(%d,%d),(%d,%d)\ndisp=%ld\n",
	   rank,coord[0],coord[1],NX*grdfct[p],NY*grdfct[p],nzt[p],nxt[p],nyt[p],nzt[p],
	   rec_NX[p], rec_NY[p], rec_NZ[p], rec_nxt[p], rec_nyt[p], rec_nzt[p],
	   NBGX[p],NSKPX[2],NEDX[2],NBGY[2],NSKPY[2],NEDY[2],NBGZ[2],NSKPZ[2],NEDZ_EP,
	   rec_nbgx[p],rec_nedx[p],rec_nbgy[p],rec_nedy[p],rec_nbgz[p],rec_nedz[p],(long int)displacement[p]);


	 /*this should save the final plastic strain down to NEDZ_EP grip points*/
	 Bufeta2[p] = Alloc1D(rec_nxt[p]*rec_nyt[p]*rec_nzt[p]);

	 for(k=nzt[p]+align-1 - rec_nbgz[p]; k>=nzt[p]+align-1 - rec_nedz[p]; k=k-NSKPZ[p])
	   for(j=2+ngsl + rec_nbgy[p]; j<=2+ngsl + rec_nedy[p]; j=j+NSKPY[p])
	     for(i=2+ngsl + rec_nbgx[p]; i<=2+ngsl + rec_nedx[p]; i=i+NSKPX[p]) {
	       if (tmpInd >= (rec_nxt[p]*rec_nyt[p]*rec_nzt[p])) 
		  fprintf(stdout, "tmpind=%ld (allocated %d)\n", tmpInd, (rec_nxt[p]*rec_nyt[p]*rec_nzt[p]));
	       Bufeta2[p][tmpInd] = neta[p][i][j][k];
	       tmpInd++;
	     }

	 MPI_Datatype filetype2;


       maxNX_NY_NZ_WS = (maxNX_NY_NZ_WS>rec_NZ[p]?maxNX_NY_NZ_WS:rec_NZ[p]);
       int ones2[maxNX_NY_NZ_WS];
       MPI_Aint dispArray2[maxNX_NY_NZ_WS];
       for(i=0;i<maxNX_NY_NZ_WS;++i){
	 ones2[i] = 1;
       }
      
       err = MPI_Type_contiguous(rec_nxt[p], _mpi_prec, &filetype2);
       err = MPI_Type_commit(&filetype2);
       for(i=0;i<rec_nyt[p];i++){
	 dispArray2[i] = sizeof(_prec);
	 dispArray2[i] = dispArray2[i]*rec_NX[p]*i;
       }
       err = MPI_Type_create_hindexed(rec_nyt[p], ones2, dispArray2, filetype2, &filetype2);
       err = MPI_Type_commit(&filetype2);
       for(i=0;i<rec_nzt[p];i++){
	 dispArray2[i] = sizeof(_prec);
	 dispArray2[i] = dispArray2[i]*rec_NY[p]*rec_NX[p]*i;
       }
       err = MPI_Type_create_hindexed(rec_nzt[p], ones2, dispArray2, filetype2, &filetype2);
       err = MPI_Type_commit(&filetype2);
       MPICHK(MPI_Type_size(filetype2, &tmpSize));
       printf("filetype size (supposedly=rec_nxt*rec_nyt*rec_nzt*4=%ld) =%d\n", 
                  rec_nxt[p]*rec_nyt[p]*rec_nzt[p]*sizeof(_prec),tmpSize);

	 sprintf(filename, "Finaleta_%d_%07ld", p, cur_step);
	 err = MPI_File_open(MCW,filename,MPI_MODE_CREATE|MPI_MODE_WRONLY,MPI_INFO_NULL,&fh);
	 err = MPI_File_set_view(fh, displacement[p], _mpi_prec, filetype2, "native", MPI_INFO_NULL);
	 if (err != MPI_SUCCESS) {
	    fprintf(stderr, "MPI error in MPI_File_set_view():\n");
	    char errstr[200];
	    int strlen;
	    MPICHK(MPI_Error_string(err, errstr, &strlen));
	    fprintf(stderr, "MPI error in MPI_File_set_view(): %s\n", errstr);
	 }
	 err = MPI_File_write_all(fh, Bufeta2, rec_nxt[p]*rec_nyt[p]*rec_nzt[p], _mpi_prec, &filestatus);
	 if (err != MPI_SUCCESS) {
	    char errstr[200];
	    int strlen;
	    MPICHK(MPI_Error_string(err, errstr, &strlen));
	    fprintf(stderr, "MPI error in MPI_File_write_all(): %s\n", errstr);
	 }
	 err = MPI_File_close(&fh);
      }
      #else
      for (p=usetopo; p<ngrids; p++){
	 nx=NX*grdfct[p];
	 ny=NX*grdfct[p];
         num_bytes = sizeof(_prec)*(nxt[p]+4+ngsl2)*(nyt[p]+4+ngsl2)*(nzt[p]+2*align);
         CUCHK(cudaMemcpy(&neta[p][0][0][0],d_neta[p],num_bytes,cudaMemcpyDeviceToHost));
	 seism_createRegularGrid(&one, &nx, &one, &one, &ny, &one,
				 &one, nzt+p, &one, seism_regGridID+p, &err);

	 sprintf(filenamebaseep,"%s/EP_%d",OUT, p);

	 seism_file_open(filenamebaseep, "w", &one, "float", seism_regGridID+p, seism_fileep+p, &err);
         seism_write(&seism_fileep[p], &neta[p][0][0][0], &err);
         seism_file_close(seism_fileep+p, &err);
      }
      #endif

    }

#if TOPO
#if TOPO_TEST
    main_err |= topo_test_finalize(&Tt, &T);
#endif
    topo_free(&T);

    receivers_finalize();
    sources_finalize();
#endif

if(usechecksum) {
        checksum_finalize(&checksum);
}

    CUCHK(cudaStreamDestroy(stream_1));
    //cudaStreamDestroy(stream_1b);
    CUCHK(cudaStreamDestroy(stream_2));
    //cudaStreamDestroy(stream_2b);
    CUCHK(cudaStreamDestroy(stream_i));
    CUCHK(cudaStreamDestroy(stream_i2));
    for (p=0; p<ngrids; p++){
       CUCHK(cudaFreeHost(SL_vel[p]));
       CUCHK(cudaFreeHost(SR_vel[p]));
       CUCHK(cudaFreeHost(RL_vel[p]));
       CUCHK(cudaFreeHost(RR_vel[p]));
       CUCHK(cudaFreeHost(SF_vel[p]));
       CUCHK(cudaFreeHost(SB_vel[p]));
       CUCHK(cudaFreeHost(RF_vel[p]));
       CUCHK(cudaFreeHost(RB_vel[p]));
       if(NVE==3){
	 CUCHK(cudaFreeHost(SL_yldfac[p]));
	 CUCHK(cudaFreeHost(SR_yldfac[p]));
	 CUCHK(cudaFreeHost(RL_yldfac[p]));
	 CUCHK(cudaFreeHost(RR_yldfac[p]));
	 CUCHK(cudaFreeHost(SF_yldfac[p]));
	 CUCHK(cudaFreeHost(SB_yldfac[p]));
	 CUCHK(cudaFreeHost(RF_yldfac[p]));
	 CUCHK(cudaFreeHost(RB_yldfac[p]));

	 CUCHK(cudaFree(d_SL_yldfac[p]));
	 CUCHK(cudaFree(d_SR_yldfac[p]));
	 CUCHK(cudaFree(d_RL_yldfac[p]));
	 CUCHK(cudaFree(d_RR_yldfac[p]));
	 CUCHK(cudaFree(d_SF_yldfac[p]));
	 CUCHK(cudaFree(d_SB_yldfac[p]));
	 CUCHK(cudaFree(d_RF_yldfac[p]));
	 CUCHK(cudaFree(d_RB_yldfac[p]));

       }
    }
    GFLOPS  = 1.0;
    if (NVE < 2) GFLOPS  = GFLOPS*307.0*(xre - xls)*(yre-yls)*nzt[0];
    else GFLOPS  = GFLOPS*511.0*(xre - xls)*(yre-yls)*nzt[0];
    GFLOPS  = GFLOPS/(1000*1000*1000);
    time_un = time_un/(cur_step-READ_STEP);
    GFLOPS  = GFLOPS/time_un;
    MPICHK(MPI_Allreduce( &GFLOPS, &GFLOPS_SUM, 1, MPI_DOUBLE, MPI_SUM, MCW ));
    if(rank==0)
    {
        printf("GPU benchmark size (fine grid) NX=%d, NY=%d, NZ=%d, ReadStep=%d\n", NX, NY, NZ[0], READ_STEP);
    	printf("GPU computing flops=%1.18f GFLOPS, time = %1.18f secs per timestep\n", GFLOPS_SUM, time_un);
    }	
//  Main Loop Ends
//
 
//  program ends, free all memories
    for (p=0; p<ngrids; p++){
       Delloc3D(u1[p]);
       Delloc3D(v1[p]);
       Delloc3D(w1[p]); 
       Delloc3D(xx[p]);
       Delloc3D(yy[p]);
       Delloc3D(zz[p]);
       Delloc3D(xy[p]);
       Delloc3D(yz[p]);
       Delloc3D(xz[p]);
       Delloc3D(vx1[p]);
       Delloc3D(vx2[p]);
       Delloc3Dww(ww[p]);
       Delloc3D(wwo[p]); 

       CUCHK(cudaFree(d_u1[p]));
       CUCHK(cudaFree(d_v1[p]));
       CUCHK(cudaFree(d_w1[p]));
       CUCHK(cudaFree(d_f_u1[p]));
       CUCHK(cudaFree(d_f_v1[p]));
       CUCHK(cudaFree(d_f_w1[p]));
       CUCHK(cudaFree(d_b_u1[p]));
       CUCHK(cudaFree(d_b_v1[p]));
       CUCHK(cudaFree(d_b_w1[p]));
       CUCHK(cudaFree(d_xx[p]));
       CUCHK(cudaFree(d_yy[p]));
       CUCHK(cudaFree(d_zz[p]));
       CUCHK(cudaFree(d_xy[p]));
       CUCHK(cudaFree(d_yz[p]));
       CUCHK(cudaFree(d_xz[p]));
       CUCHK(cudaFree(d_vx1[p]));
       CUCHK(cudaFree(d_vx2[p]));

       if(NVE==1 || NVE==3)
       {
	  Delloc3D(r1[p]);
	  Delloc3D(r2[p]);
	  Delloc3D(r3[p]);
	  Delloc3D(r4[p]);
	  Delloc3D(r5[p]);
	  Delloc3D(r6[p]);
	  CUCHK(cudaFree(d_r1[p]));
	  CUCHK(cudaFree(d_r2[p]));
	  CUCHK(cudaFree(d_r3[p]));
	  CUCHK(cudaFree(d_r4[p]));
	  CUCHK(cudaFree(d_r5[p]));
	  CUCHK(cudaFree(d_r6[p]));

	  Delloc3D(qp[p]);
	  Delloc3D(qs[p]);
	  CUCHK(cudaFree(d_qp[p]));
	  CUCHK(cudaFree(d_qs[p]));
       }
       if(NVE==3){
	 Delloc3D(sigma2[p]);
	 Delloc3D(cohes[p]);
	 Delloc3D(phi[p]);
	 Delloc3D(yldfac[p]);
	 Delloc3D(neta[p]);
       }

       if((NPC==0) || (NPC==2))
       {
	   Delloc1D(dcrjx[p]);
	   Delloc1D(dcrjy[p]);
	   Delloc1D(dcrjz[p]);
	   CUCHK(cudaFree(d_dcrjx[p]));
	   CUCHK(cudaFree(d_dcrjy[p]));
	   CUCHK(cudaFree(d_dcrjz[p]));
       }

       Delloc3D(d1[p]);
       Delloc3D(mu[p]);
       Delloc3D(lam[p]);
       Delloc3D(lam_mu[p]);
       CUCHK(cudaFree(d_d1[p]));
       CUCHK(cudaFree(d_mu[p]));
       CUCHK(cudaFree(d_lam[p]));
       CUCHK(cudaFree(d_lam_mu[p]));
    }

    if(NVE==1 || NVE==3) {
       Delloc1D(coeff);  
       CUCHK(cudaFree(d_coeff));
    }

    for (p=0; p<ngrids; p++){
       if(rank==srcproc[p]) {
	  Delloc1D(taxx[p]);
	  Delloc1D(tayy[p]);
	  Delloc1D(tazz[p]);
	  Delloc1D(taxz[p]);
	  Delloc1D(tayz[p]);
	  Delloc1D(taxy[p]); 
	  CUCHK(cudaFree(d_taxx[p]));
	  CUCHK(cudaFree(d_tayy[p]));
	  CUCHK(cudaFree(d_tazz[p]));
	  CUCHK(cudaFree(d_taxz[p]));
	  CUCHK(cudaFree(d_tayz[p]));
	  CUCHK(cudaFree(d_taxy[p]));
	  Delloc1P(tpsrc[p]);
	  CUCHK(cudaFree(d_tpsrc[p]));
       }
    }
    MPICHK(MPI_Comm_free( &MC1 ));

    #ifndef NOBGIO
    } /* end of if (rank < size) */

    else if (ranktype==1) {
       if (IFAULT == 4) background_velocity_reader(rank, size, NST, READ_STEP, MCS);
    }
    else {
       nt = (int)(TMAX/DT) + 1;
       nout= nt / (WRITE_STEP * NTISKP);
       background_output_writer(rank, size, nout, WRITE_STEP, NTISKP, ngrids, OUT, MCI, NVE);
    }
    #endif


    MPICHK(MPI_Barrier(MCT));
    MPICHK(MPI_Finalize());
    return main_err;
}

