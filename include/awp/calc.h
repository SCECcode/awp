#ifndef CALC_H
#define CALC_H

void calcRecordingPoints(int *rec_nbgx, int *rec_nedx, 
  int *rec_nbgy, int *rec_nedy, int *rec_nbgz, int *rec_nedz, 
  int *rec_nxt, int *rec_nyt, int *rec_nzt, MPI_Offset *displacement,
  long int nxt, long int nyt, long int nzt, int rec_NX, int rec_NY, int rec_NZ,
  int NBGX, int NEDX, int NSKPX, int NBGY, int NEDY, int NSKPY, 
  int NBGZ, int NEDZ, int NSKPZ, int *coord);
#endif 

