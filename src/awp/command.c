/*
**************************************************************************************************************** 
*  command.c						                                                       *
*  Process Command Line	                                                                                       *
*                                                                                                              *
*  Name         Type        Command             Description 	                                                 *
*  TMAX         <FLOAT>       -T              propagation time	                	                             *
*  DH           <FLOAT>       -H              spatial step for x, y, z (meters)                                *
*  DT           <FLOAT>       -t              time step (seconds)                                              *
*  ARBC         <FLOAT>       -A              oefficient for PML (3-4), or Cerjan (0.90-0.96)                  *
*  PHT          <FLOAT>       -P                                                                               *
*  NPC          <INTEGER>     -M              PML or Cerjan ABC (1=PML, 0=Cerjan, 2=Periodic)                  *
*  ND           <INTEGER>     -D              ABC thickness (grid-points) PML <= 20, Cerjan >= 20              *
*  NSRC         <INTEGER>     -S              number of source nodes on fault                                  *
*  NST          <INTEGER>     -N              number of time steps in rupture functions                        *
*  NVAR         <INTEGER>     -n              number of variables in a grid point                              *
*  NVE          <INTEGER>     -V              visco or elastic scheme (1=visco, 0=elastic)                     *
*  MEDIASTART   <INTEGER>     -B              initial media restart option(0=homogenous)                       *
*  IFAULT       <INTEGER>     -I              mode selection and fault or initial stress setting (1 or 2)      *
*  READ_STEP    <INTEGER>     -R                                                                               *
*  READ_STEP_GPU<INTEGER>     -Q              CPU reads larger chunks and sends to GPU at every READ_STEP_GPU  *
*                                               (IFAULT=2) READ_STEP must be divisible by READ_STEP_GPU        *
*  NX           <INTEGER>     -X              x model dimension in nodes                                       *      
*  NY           <INTEGER>     -Y              y model dimension in nodes                                       *
*  NZ           <INTEGER>     -Z              z model dimension in nodes                                       *
*  PX           <INTEGER>     -x              number of procs in the x direction                               *
*  PY           <INTEGER>     -y              number of procs in the y direction                               * 
*  NBGX         <INTEGER>                     index (starts with 1) to start recording points in X             *
*  NEDX         <INTEGER>                     index to end recording points in X (-1 for all)                  *
*  NSKPX        <INTEGER>                     #points to skip in recording points in X                         *
*  NBGY         <INTEGER>                     index to start recording points in Y                             *
*  NEDY         <INTEGER>                     index to end recording points in Y (-1 for all)                  *
*  NSKPY        <INTEGER>                     #points to skip in recording points in Y                         *
*  NBGZ         <INTEGER>                     index to start recording points in Z                             *
*  NEDZ         <INTEGER>                     index to end recording points in Z (-1 for all)                  *
*  NSKPZ        <INTEGER>                     #points to skip in recording points in Z                         *
*  IDYNA        <INTEGER>     -i              mode selection of dynamic rupture model                          *
*  SoCalQ       <INTEGER>     -s              Southern California Vp-Vs Q relationship enabling flag           *
*  FAC          <FLOAT>       -l              Q                                                                * 
*  Q0           <FLOAT>       -h              Q                                                                * 
*  EX           <FLOAT>       -x              Q                                                                *
*  NTISKP       <INTEGER>     -r              # timesteps to skip to copy velocities from GPU to CPU           *
*  WRITE_STEP   <INTEGER>     -W              # timesteps to write the buffer to the files                     *
*                                               (written timesteps are n*NTISKP*WRITE_STEP for n=1,2,...)      *
*  INSRC        <STRING>                      source input file (if IFAULT=2, then this is prefix of tpsrc)    *
*  INVEL        <STRING>                      mesh input file                                                  *
*  INSRC_I2     <STRING>                      split source input file prefix for IFAULT=2 option               *
*  CHKFILE      <STRING>      -c              Checkpoint statistics file to write to                           *
*  FOLLOWBATHY  <STRING>                      surface output follows ocean bathymetry
*  INTOPO       <STRING>                      topography input file
*
*  SOURCEFILE   <STRING>                      Source input file that uses
*                                                coordinates instead of indices
*                                                to specify the position
*  RECVFILE     <STRING>                      Receiver output file
*  FORCEFILE    <STRING>                      Boundary point force input file
*  SGTFILE      <STRING>                      Strain Green's tensor output file
*  MMSFILE      <STRING>                      MMS input file
*  DHB          <FLOAT>                       Grid spacing at the bottom of the curvilinear block  
*  DHT          <FLOAT>                       Grid spacing at the top of the curvilinear block  
*  ENERGYFILE  <STRING>                       File to write energy information at each time step to  
****************************************************************************************************************
*/

#include <stdlib.h>
#include <stdio.h>
#include <getopt.h>
#include <math.h>
#include <string.h>
#include <awp/pmcl3d.h>
#include <awp/pmcl3d_cons.h>
#include <assert.h>

// Default IN3D Values
const _prec def_TMAX       = 20.00;
const _prec def_DH         = 200.0;
const _prec def_DT         = 0.01; 
const _prec def_ARBC       = 0.92;
const _prec def_PHT        = 0.1;

const int   def_NPC        = 0;
const int   def_ND         = 20;
const int   def_NSRC       = 1;
const int   def_NST        = 91;
const int   def_NVAR       = 3;

const int   def_NVE        = 1;
const int   def_MEDIASTART = 0;
const int   def_IFAULT     = 1; 
const int   def_READ_STEP  = 91;
const int   def_READ_STEP_GPU = 91;

const int   def_NTISKP     = 1;
const int   def_WRITE_STEP = 10;

const int   def_NX         = 256;
const int   def_NY         = 256; 
const int   def_NZ         = 1024;

const int   def_PX         = 1;
const int   def_PY         = 1;

const int   def_NBGX       = 1;
const int   def_NEDX       = -1;   // use -1 for all
const int   def_NSKPX      = 1;
const int   def_NBGY       = 1;
const int   def_NEDY       = -1;   // use -1 for all
const int   def_NSKPY      = 1;
const int   def_NBGZ       = 1;
const int   def_NEDZ       = 1;    // only surface
const int   def_NSKPZ      = 1;

const int   def_NGRIDS     = 1; /* number of grid resolutions */

const int   def_IDYNA      = 0;
const int   def_SoCalQ     = 1;

const _prec def_FAC        = 0.005;
const _prec def_Q0         = 5.0; 
const _prec def_EX         = 0.0; 
const _prec def_FP         = 2.5; 

const char  def_INSRC[50]  = "input/FAULTPOW";
const char  def_INVEL[50]  = "input/media";

const char  def_OUT[50] = "output_sfc";

const char  def_INSRC_TPSRC[50] = "input_rst/srcpart/tpsrc/tpsrc";
const char  def_INSRC_I2[50]  = "input_rst/srcpart/split_faults/fault";

const char  def_CHKFILE[50]   = "output_ckp/CHKP";

const int   def_FOLLOWBATHY     = 0;

const char def_INTOPO[IN_FILE_LEN] = "input/topography";

const char def_SOURCEFILE[IN_FILE_LEN] = "";
const char def_RECVFILE[IN_FILE_LEN] = "";
const char def_FORCEFILE[IN_FILE_LEN] = "";
const char def_SGTFILE[IN_FILE_LEN] = "";
const char def_MMSFILE[IN_FILE_LEN] = "";
const char def_ENERGYFILE[IN_FILE_LEN] = "";

const _prec def_DHB = -1.0;
const _prec def_DHT = -1.0;

void parsemultiple(char *optarg, int *val);

void parsemultiple(char *optarg, int *val){
    int k;
    char *token;
    token = strtok (optarg,",");
    k = 0;
    while (token != NULL && k < MAXGRIDS) {
     val[k] = atoi(token);
     token = strtok(NULL, ",");
     k++;
    } 
}

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
             char *ENERGYFILE, int *USEENERGYFILE)
{
        int p;

        // Fill in default values
        *TMAX = def_TMAX;
        *DH = def_DH;
        *DT = def_DT;
        *ARBC = def_ARBC;
        *PHT = def_PHT;

        *NPC = def_NPC;
        *ND = def_ND;
        NSRC[0] = def_NSRC;
        *NST = def_NST;

        *NVE = def_NVE;
        *MEDIASTART = def_MEDIASTART;
        *NVAR = def_NVAR;
        *IFAULT = def_IFAULT;
        *READ_STEP = def_READ_STEP;
        *READ_STEP_GPU = def_READ_STEP_GPU;

        *NTISKP = def_NTISKP;
        *WRITE_STEP = def_WRITE_STEP;

        *NX = def_NX;
        *NY = def_NY;
        NZ[0] = def_NZ;
        *PX = def_PX;
        *PY = def_PY;

        for (p = 0; p < MAXGRIDS; p++) {
                NBGX[p] = def_NBGX;
                NEDX[p] = def_NEDX;
                NSKPX[p] = def_NSKPX;
                NBGY[p] = def_NBGY;
                NEDY[p] = def_NEDY;
                NSKPY[p] = def_NSKPY;
                NBGZ[p] = def_NBGZ;
                NEDZ[p] = def_NEDZ;
                NSKPZ[p] = def_NSKPZ;
        }

        *IDYNA = def_IDYNA;
        *SoCalQ = def_SoCalQ;
        *FAC = def_FAC;
        *Q0 = def_Q0;
        *EX = def_EX;
        *FP = def_FP;
        *NGRIDS = def_NGRIDS;
        *FOLLOWBATHY = def_FOLLOWBATHY;

        strcpy(INSRC, def_INSRC);
        strcpy(INVEL, def_INVEL);
        strcpy(OUT, def_OUT);
        strcpy(INSRC_I2, def_INSRC_I2);
        strcpy(CHKFILE, def_CHKFILE);
        strcpy(INTOPO, def_INTOPO);
        strcpy(SOURCEFILE, def_SOURCEFILE);
        strcpy(RECVFILE, def_RECVFILE);
        strcpy(MMSFILE, def_MMSFILE);


        extern char *optarg;
        static const char *optstring =
            "-T:H:t:A:P:M:D:S:N:V:B:n:I:R:Q:X:Y:Z:x:y:G:z:i:l:h:30:p:s:r:W:1:2:"
            "3:11:12:13:21:22:23:100:101:102:103:106:107:109:9:14:o:c:15:16:17:";
        static struct option long_options[] = {
            {"TMAX", required_argument, NULL, 'T'},
            {"DH", required_argument, NULL, 'H'},
            {"DT", required_argument, NULL, 't'},
            {"ARBC", required_argument, NULL, 'A'},
            {"PHT", required_argument, NULL, 'P'},
            {"NPC", required_argument, NULL, 'M'},
            {"ND", required_argument, NULL, 'D'},
            {"NSRC", required_argument, NULL, 'S'},
            {"NST", required_argument, NULL, 'N'},
            {"NVE", required_argument, NULL, 'V'},
            {"MEDIASTART", required_argument, NULL, 'B'},
            {"NVAR", required_argument, NULL, 'n'},
            {"IFAULT", required_argument, NULL, 'I'},
            {"READ_STEP", required_argument, NULL, 'R'},
            {"READ_STEP_GPU", required_argument, NULL, 'Q'},
            {"NX", required_argument, NULL, 'X'},
            {"NY", required_argument, NULL, 'Y'},
            {"NZ", required_argument, NULL, 'Z'},
            {"PX", required_argument, NULL, 'x'},
            {"PY", required_argument, NULL, 'y'},
            {"NGRIDS", required_argument, NULL, 'G'},
            {"NBGX", required_argument, NULL, 1},
            {"NEDX", required_argument, NULL, 2},
            {"NSKPX", required_argument, NULL, 3},
            {"NBGY", required_argument, NULL, 11},
            {"NEDY", required_argument, NULL, 12},
            {"NSKPY", required_argument, NULL, 13},
            {"NBGZ", required_argument, NULL, 21},
            {"NEDZ", required_argument, NULL, 22},
            {"NSKPZ", required_argument, NULL, 23},
            {"IDYNA", required_argument, NULL, 'i'},
            {"SoCalQ", required_argument, NULL, 's'},
            {"FAC", required_argument, NULL, 'l'},
            {"Q0", required_argument, NULL, 'h'},
            {"EX", required_argument, NULL, 30},
            {"FP", required_argument, NULL, 'p'},
            {"NTISKP", required_argument, NULL, 'r'},
            {"WRITE_STEP", required_argument, NULL, 'W'},
            {"INSRC", required_argument, NULL, 100},
            {"INVEL", required_argument, NULL, 101},
            {"OUT", required_argument, NULL, 'o'},
            {"INSRC_I2", required_argument, NULL, 102},
            {"CHKFILE", required_argument, NULL, 'c'},
            {"FOLLOWBATHY", required_argument, NULL, 7},
            {"INTOPO", required_argument, NULL, 103},
            {"SOURCEFILE", required_argument, NULL, 107},
            {"RECVFILE", required_argument, NULL, 109},
            {"FORCEFILE", required_argument, NULL, 9},
            {"SGTFILE", required_argument, NULL, 10},
            {"MMSFILE", required_argument, NULL, 14},
            {"DHB", required_argument, NULL, 15},
            {"DHT", required_argument, NULL, 16},
            {"ENERGYFILE", required_argument, NULL, 17},
        };


        // If IFAULT=2 and INSRC is not set, then *INSRC = def_INSRC_TPSRC, not
        // def_INSRC
        int insrcIsSet = 0;
        // If IFAULT=1 and READ_STEP_GPU is not set, it should be = READ_STEP
        int readstepGpuIsSet = 0;
        int c;

        while ((c = getopt_long(argc, argv, optstring, long_options, NULL)) !=

               -1) {
                switch (c) {
                        case 'T':
                                *TMAX = atof(optarg);
                                break; 
                        case 'H':
                                *DH = atof(optarg);
                                break;
                        case 't':
                                *DT = atof(optarg);
                                break;
                        case 'A':
                                *ARBC = atof(optarg);
                                break;
                        case 'P':
                                *PHT = atof(optarg);
                                break;
                        case 'M':
                                *NPC = atoi(optarg);
                                break;
                        case 'D':
                                *ND = atoi(optarg);
                                break;
                        case 'S':
                                parsemultiple(optarg, NSRC);
                                break;
                        case 'N':
                                *NST = atoi(optarg);
                                break;
                        case 'V':
                                *NVE = atoi(optarg);
                                break;
                        case 'B':
                                *MEDIASTART = atoi(optarg);
                                break;
                        case 'n':
                                *NVAR = atoi(optarg);
                                break;
                        case 'I':
                                *IFAULT = atoi(optarg);
                                break;
                        case 'R':
                                *READ_STEP = atoi(optarg);
                                break;
                        case 'Q':
                                readstepGpuIsSet = 1;
                                *READ_STEP_GPU = atoi(optarg);
                                break;
                        case 'X':
                                *NX = atoi(optarg);
                                break;
                        case 'Y':
                                *NY = atoi(optarg);
                                break;
                        case 'Z':
                                parsemultiple(optarg, NZ);
                                break;
                        case 'x':
                                *PX = atoi(optarg);
                                break;
                        case 'y':
                                *PY = atoi(optarg);
                                break;
                        case 1:
                                parsemultiple(optarg, NBGX);
                                break;
                        case 2:
                                parsemultiple(optarg, NEDX);
                                break;
                        case 3:
                                parsemultiple(optarg, NSKPX);
                                break;
                        case 11:
                                parsemultiple(optarg, NBGY);
                                break;
                        case 12:
                                parsemultiple(optarg, NEDY);
                                break;
                        case 13:
                                parsemultiple(optarg, NSKPY);
                                break;
                        case 21:
                                parsemultiple(optarg, NBGZ);
                                break;
                        case 22:
                                parsemultiple(optarg, NEDZ);
                                break;
                        case 23:
                                parsemultiple(optarg, NSKPZ);
                                break;
                        case 'i':
                                *IDYNA = atoi(optarg);
                                break;
                        case 's':
                                *SoCalQ = atoi(optarg);
                                break;
                        case 'l':
                                *FAC = atof(optarg);
                                break;
                        case 'h':
                                *Q0 = atof(optarg);
                                break;
                        case 30:
                                *EX = atof(optarg);
                                break;
                        case 'p':
                                *FP = atof(optarg);
                                break;
                        case 'r':
                                *NTISKP = atoi(optarg);
                                break;
                        case 'W':
                                *WRITE_STEP = atoi(optarg);
                                break;
                        case 100:
                                insrcIsSet = 1;
                                strcpy(INSRC, optarg);
                                break;
                        case 101:
                                strcpy(INVEL, optarg);
                                break;
                        case 'o':
                                strcpy(OUT, optarg);
                                break;
                        case 102:
                                strcpy(INSRC_I2, optarg);
                                break;
                        case 'c':
                                strcpy(CHKFILE, optarg);
                                break;
                        case 'G':
                                *NGRIDS = atoi(optarg);
                                break;
                        case 7:
                                *FOLLOWBATHY = atoi(optarg);
                                break;
                        case 103:
                                strcpy(INTOPO, optarg);
                                *USETOPO = 1;
                                break;
                        case 107:
                                strcpy(SOURCEFILE, optarg);
                                *USESOURCEFILE = 1;
                                break;
                        case 109:
                                strcpy(RECVFILE, optarg);
                                *USERECVFILE = 1;
                                break;
                        case 9:
                                strcpy(FORCEFILE, optarg);
                                *USEFORCEFILE = 1;
                                break;
                        case 10:
                                strcpy(SGTFILE, optarg);
                                *USESGTFILE = 1;
                                break;
                        case 14:
                                strcpy(MMSFILE, optarg);
                                *USEMMSFILE = 1;
                                break;
                        case 15:
                                *DHB = atof(optarg);
                                break;
                        case 16:
                                *DHT = atof(optarg);
                                break;
                        case 17:
                                strcpy(ENERGYFILE, optarg);
                                *USEENERGYFILE = 1;
                                break;
                        default:
                                printf(
                                    "Usage: %s \nOptions:\n\t[(-T | --TMAX) "
                                    "<TMAX>]\n\t[(-H | --DH) <DH>]\n\t[(-t | "
                                    "--DT) <DT>]\n\t[(-A | --ARBC) "
                                    "<ARBC>]\n\t[(-P | --PHT) <PHT>]\n\t[(-M | "
                                    "--NPC) <NPC>]\n\t[(-D | --ND) "
                                    "<ND>]\n\t[(-S | --NSRC) <NSRC>]\n\t[(-N | "
                                    "--NST) <NST>]\n",
                                    argv[0]);
                                printf(
                                    "\n\t[(-V | --NVE) <NVE>]\n\t[(-B | "
                                    "--MEDIASTART) <MEDIASTART>]\n\t[(-n | "
                                    "--NVAR) <NVAR>]\n\t[(-I | --IFAULT) "
                                    "<IFAULT>]\n\t[(-R | --READ_STEP) <x "
                                    "READ_STEP for CPU>]\n\t[(-Q | "
                                    "--READ_STEP_GPU) <READ_STEP for GPU>]\n");
                                printf(
                                    "\n\t[(-X | --NX) <x length]\n\t[(-Y | "
                                    "--NY) <y length>]\n\t[(-Z | --NZ) <z "
                                    "length 0,z length 1,...]\n\t[(-x | --NPX) "
                                    "<x processors]\n\t[(-y | --NPY) <y "
                                    "processors>]\n\t[(-z | --NPZ) <z "
                                    "processors>]\n");
                                printf(
                                    "\n\t[(-1 | --NBGX) <starting point to "
                                    "record in X>]\n\t[(-2 | --NEDX) <ending "
                                    "point to record in X>]\n\t[(-3 | --NSKPX) "
                                    "<skipping points to record in "
                                    "X>]\n\t[(-11 | --NBGY) <starting point to "
                                    "record in Y>]\n\t[(-12 | --NEDY) <ending "
                                    "point to record in Y>]\n\t[(-13 | "
                                    "--NSKPY) <skipping points to record in "
                                    "Y>]\n\t[(-21 | --NBGZ) <starting point to "
                                    "record in Z>]\n\t[(-22 | --NEDZ) <ending "
                                    "point to record in Z>]\n\t[(-23 | "
                                    "--NSKPZ) <skipping points to record in "
                                    "Z>]\n");
                                printf(
                                    "\n\t[(-i | --IDYNA) <i IDYNA>]\n\t[(-s | "
                                    "--SoCalQ) <s SoCalQ>]\n\t[(-l | --FAC) <l "
                                    "FAC>]\n\t[(-h | --Q0) <h Q0>]\n\t[(-30 | "
                                    "--EX) <e EX>]\n\t[(-p | --FP) <p "
                                    "FP>]\n\t[(-r | --NTISKP) <time skipping "
                                    "in writing>]\n\t[(-W | --WRITE_STEP) "
                                    "<time aggregation in writing>]\n");
                                printf(
                                    "\n\t[(-100 | --INSRC) <source "
                                    "file>]\n\t[(-101 | --INVEL) <mesh "
                                    "file>]\n\t[(-o | --OUT) <output "
                                    "file>]\n\t[(-102 | --INSRC_I2) <split "
                                    "source file prefix (IFAULT=2)>]\n\t[(-c | "
                                    "--CHKFILE) <checkpoint file to write "
                                    "statistics>]\n");
                                printf(
                                    "\n\t[(-G | --NGRIDS) <number of "
                                    "grids>]\n\n");
                                printf(
                                    "\n\t[(-104 | --FOLLOWBATHY) 0|1]\n\n");  // FIXME: Looks strange, I think id 104 is unused. Id 7 is used earlier <ooreilly@usc.edu>
                                printf(
                                    "\n\t[(-103 | --INTOPO) <topography "
                                    "file>]\n\n");
                                printf(
                                    "\n\t[(-107 | --SOURCEFILE) <source "
                                    "file>]\n\n");
                                printf(
                                    "\n\t[(-109 | --RECVFILE) <receiver "
                                    "file>]\n\n");
                                printf(
                                    "\n\t[(-9 | --FORCEFILE) <force "
                                    "file>]\n\n");
                                printf(
                                    "\n\t[(-10 | --SGTFILE) <SGT "
                                    "file>]\n\n");
                                printf(
                                    "\n\t[(-14 | --MMSFILE) <MMS "
                                    "file>]\n\n");
                                printf(
                                    "\n\t[(-15 | --DHB) <Bottom grid spacing> ]\n\n");
                                printf(
                                    "\n\t[(-16 | --DHT) <Top grid spacing> ]\n\n");
                                printf(
                                    "\n\t[(-17 | --ENERGYFILE) <File to output energy information to> ]\n\n");
                                exit(-1);
                }
        }



        // If IFAULT=2 and INSRC is not set, then *INSRC = def_INSRC_TPSRC, not
        // def_INSRC
        if (*IFAULT == 2 && !insrcIsSet) {
                strcpy(INSRC, def_INSRC_TPSRC);
        }
        if (!readstepGpuIsSet) {
                *READ_STEP_GPU = *READ_STEP;
        }

        /* if a \ character is present in the command-line arguments, it is
           interpreted as 1 and NBGX[0] is set to 0.  This line prevents that.
         */
        if (NBGX[0] == 0) NBGX[0] = 1;


        return;
}

