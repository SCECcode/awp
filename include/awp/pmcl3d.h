/*  
********************************************************************************
* pmcl3d.h                                                                     *
* programming in C language                                                    *
* all pmcl3d data types are defined here                                       * 
********************************************************************************
*/
#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <awp/pmcl3d_cons.h>

#ifdef __RESTRICT 
#define RESTRICT restrict 
#else
#define RESTRICT 
#endif

#ifndef _PMCL3D_H
#define _PMCL3D_H

typedef _prec *RESTRICT *RESTRICT *RESTRICT Grid3D;
typedef int *RESTRICT *RESTRICT *RESTRICT Grid3Dww;
typedef _prec *RESTRICT Grid1D;
typedef int   *RESTRICT PosInf;

void command(int argc, char **argv, _prec *TMAX, _prec *DH, _prec *DT,
             _prec *ARBC, _prec *PHT, int *NPC, int *ND, int *NSRC, int *NST,
             int *NVAR, int *NVE, int *MEDIASTART, int *IFAULT, int *READ_STEP,
             int *READ_STEP_GPU, int *NTISKP, int *WRITE_STEP, int *NX, int *NY,
             int *NZ, int *PX, int *PY, int *NBGX, int *NEDX, int *NSKPX,
             int *NBGY, int *NEDY, int *NSKPY, int *NBGZ, int *NEDZ, int *NSKPZ,
             _prec *FAC, _prec *Q0, _prec *EX, _prec *FP, int *IDYNA,
             int *SoCalQ, char *INSRC, char *INVEL, char *OUT, char *INSRC_I2,
             char *CHKFILE, int *NGRIDS, int *FOLLOWBATHY, char *INTOPO,
             int *USETOPO, char *SOURCEFILE,
             int *USESOURCEFILE, char *RECVFILE, int *USERECVFILE,
             char *FORCEFILE, int *USEFORCEFILE,
             char *SGTFILE, int *USESGTFILE, char *MMSFILE, int *USEMMSFILE, float *DHB, float *DHT,
             char *ENERGYFILE, int *USEENERGYFILE, 
             _prec *QSI, _prec *QPQSR, _prec *MAXVPVSR, _prec *VMIN, _prec *VMAX, _prec *DMIN);

int read_src_ifault_2(int rank, int READ_STEP, 
    char *INSRC, char *INSRC_I2, 
    int maxdim, int *coords, int NZ,
    int nxt, int nyt, int nzt,
    int *NPSRC, int *SRCPROC, 
    PosInf *psrc, Grid1D *axx, Grid1D *ayy, Grid1D *azz, 
    Grid1D *axz, Grid1D *ayz, Grid1D *axy,
    int idx);

int read_src_ifault_4 (int rank, int READ_STEP, char *INSRC, 
    int maxdim, int *coords, int NZ, int nxt, int nyt, int nzt,
    int *NPSRC, int *SRCPROC, PosInf *psrc, Grid1D *axx, Grid1D *ayy, Grid1D *azz, 
    int idx, int *fbc_ext, int *fbc_off, char *fbc_pmask, int *fbc_extl, int *fbc_dim, 
    int *fbc_seismio, int *fbc_tskp, int nst, int size);

int inisource(int      rank,    int     IFAULT, int     NSRC,   int     READ_STEP, 
              int     NST,     int     *SRCPROC, int    NZ,
              MPI_Comm MCW,     int     nxt,    int     nyt,    int     nzt,       
              int     *coords, int     maxdim,   int    *NPSRC,
              PosInf   *ptpsrc, Grid1D  *ptaxx, Grid1D  *ptayy, Grid1D  *ptazz,    
              Grid1D  *ptaxz,  Grid1D  *ptayz,   Grid1D *ptaxy, char *INSRC, char *INSRC_I2);

void addsrc(int i,      _prec DH,   _prec DT,   int NST,    int npsrc,  int READ_STEP, int dim, PosInf psrc,
            Grid1D axx, Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
            Grid3D xx,  Grid3D yy,  Grid3D zz,  Grid3D xy,  Grid3D yz,  Grid3D xz);

void frcvel(int i,      _prec DH,   _prec DT,   int NST,    int npsrc,  int tpsrc, int READ_STEP, int dim, PosInf psrc,
            Grid1D axx, Grid1D ayy, Grid1D azz, Grid1D axz, Grid1D ayz, Grid1D axy,
            Grid3D u1,  Grid3D v1,  Grid3D w1, int rank);

void inimesh(int rank, int MEDIASTART, Grid3D d1, Grid3D mu, Grid3D lam, Grid3D qp, Grid3D qs, _prec *taumax, _prec *taumin,
	     Grid3D tau, Grid3D weights,Grid1D coeff,
	     int nvar, _prec FP,  _prec FAC, _prec Q0, _prec EX, int nxt, int nyt, int nzt, int PX, int PY, int NX, int NY,
             int NZ, int *coords, MPI_Comm MCW, int IDYNA, int NVE, int SoCalQ, char *INVEL,
            _prec qsi, _prec qpqsr, _prec maxvpvsr, _prec vmin, _prec vmax, _prec dmin,
             _prec *vse, _prec *vpe, _prec *dde);

int checkmesh(int nxtl, int nytl, int nztl, int nxth, int nyth, int nzth, Grid3D varl, Grid3D varh,
    int pl, int ph, char *varname);

int checkmesh_ww(int nxtl, int nytl, int nztl, int nxth, int nyth, int nzth, Grid3Dww varl, Grid3Dww varh,
    int pl, int ph, char *varname);

void inidrpr_hoekbrown_light(int nxt, int nyt, int nzt, int nve, int *coords,
    _prec dh, int rank, 
    Grid3D mu, Grid3D lam, Grid3D d1, 
    Grid3D sigma2,
    Grid3D cohes, Grid3D phi, _prec *fmajor, _prec *fminor, 
    _prec *strike, _prec *dip, MPI_Comm MCW, int d_i);

void rotation_matrix(_prec *strike, _prec *dip, _prec *Rz, _prec *RzT);

int writeCHK(char *chkfile, int ntiskp, _prec dt, _prec *dh, 
      int *nxt, int *nyt, int *nzt,
      int nt, _prec arbc, int npc, int nve,
      _prec fac, _prec q0, _prec ex, _prec fp, 
      _prec **vse, _prec **vpe, _prec **dde, int ngrids);


void tausub( Grid3D tau, _prec taumin,_prec taumax);

void weights_sub(Grid3D weights,Grid1D coeff, _prec ex, _prec fac);

void inicrj(_prec ARBC, int *coords, int nxt, int nyt, int nzt, int NX, int NY, int ND, Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz, int islowest, int NPC);

void init_texture(int nxt,  int nyt,  int nzt,  Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2,
		  Grid3D weights, Grid3Dww ww,Grid3D wwo,
                  int xls,  int xre,  int yls,  int yre);


Grid3D Alloc3D(int nx, int ny, int nz);
Grid3Dww Alloc3Dww(int nx, int ny, int nz); 
Grid1D Alloc1D(long nx);    
PosInf Alloc1P(int nx);

void Delloc3D(Grid3D U);
void Delloc3Dww(Grid3Dww U);
void Delloc1D(Grid1D U);
void Delloc1P(PosInf U);

int max(int a, int b);

int min(int a, int b);

int background_velocity_reader(int rank, int size, int NST, int READ_STEP, MPI_Comm MCS);


void background_output_writer(int rank, int size, int nout, int wstep, int ntiskp, int ngrids, char *OUT,
   MPI_Comm MCI, int NVE);

int ini_plane_wave(int rank, MPI_Comm MCW, char *INSRC, int NST, Grid1D* taxx, Grid1D* tayy, Grid1D* tazz);
#endif

