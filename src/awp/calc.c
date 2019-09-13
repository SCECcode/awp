#include <stdlib.h>
#include <stdio.h>
#include <awp/pmcl3d.h>
#include <awp/calc.h>

// Calculates recording points for each core
// rec_nbgxyz rec_nedxyz...
// WARNING: Assumes NPZ = 1! Only surface outputs are needed!
void calcRecordingPoints(int *rec_nbgx, int *rec_nedx, 
  int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz, 
  int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
  long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
  int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY, 
  int NBGZ, int NEDZ, int NSKPZ, int *coord){

  *displacement = 0;
  *rec_nbgx = *rec_nedx = *rec_nbgy = *rec_nedy = *rec_nbgz = *rec_nedz = 0;

  if(NBGX > nxt*(coord[0]+1))     *rec_nxt = 0;
  else if(NEDX < nxt*coord[0]+1)  *rec_nxt = 0;
  else{
    if(nxt*coord[0] >= NBGX){
      *rec_nbgx = (nxt*coord[0]+NBGX-1)%NSKPX;
      *displacement += (nxt*coord[0]-NBGX)/NSKPX+1;
    }
    else
      *rec_nbgx = NBGX-nxt*coord[0]-1;  // since rec_nbgx is 0-based
    if(nxt*(coord[0]+1) <= NEDX)
      *rec_nedx = (nxt*(coord[0]+1)+NBGX-1)%NSKPX-NSKPX+nxt;
    else
      *rec_nedx = NEDX-nxt*coord[0]-1;
    *rec_nxt = (*rec_nedx-*rec_nbgx)/NSKPX+1;
  }

  if(NBGY > nyt*(coord[1]+1))     *rec_nyt = 0;
  else if(NEDY < nyt*coord[1]+1)  *rec_nyt = 0;
  else{
    if(nyt*coord[1] >= NBGY){
      *rec_nbgy = (nyt*coord[1]+NBGY-1)%NSKPY;
      *displacement += ((nyt*coord[1]-NBGY)/NSKPY+1)*rec_NX;
    }
    else
      *rec_nbgy = NBGY-nyt*coord[1]-1;  // since rec_nbgy is 0-based
    if(nyt*(coord[1]+1) <= NEDY)
      *rec_nedy = (nyt*(coord[1]+1)+NBGY-1)%NSKPY-NSKPY+nyt;
    else
      *rec_nedy = NEDY-nyt*coord[1]-1;
    *rec_nyt = (*rec_nedy-*rec_nbgy)/NSKPY+1;
  }

  if(NBGZ > nzt) *rec_nzt = 0;
  else{
    *rec_nbgz = NBGZ-1;  // since rec_nbgz is 0-based
    *rec_nedz = NEDZ-1;
    *rec_nzt = (*rec_nedz-*rec_nbgz)/NSKPZ+1;
  }

  if(*rec_nxt == 0 || *rec_nyt == 0 || *rec_nzt == 0){
    *rec_nxt = 0;
    *rec_nyt = 0;
    *rec_nzt = 0;

    /*Added by Daniel, otherwise memory violation occurs later if some subdomains save no output*/
    *rec_nedx = *rec_nbgx - 1;
    *rec_nedy = *rec_nbgy - 1;
  }

  // displacement assumes NPZ=1!
  *displacement *= sizeof(_prec);

  return;
}

