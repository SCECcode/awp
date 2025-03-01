/*
********************************************************************************
* Grid3D.c                                                                     *
* programming in C language                                                    *
* 3D data structure                                                            * 
********************************************************************************
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <awp/pmcl3d.h>

Grid3D Alloc3D(int nx, int ny, int nz)
{
   int i, j, k;
   Grid3D U = (Grid3D)malloc(sizeof(_prec* *)*nx + sizeof(_prec *)*nx*ny +sizeof(_prec)*nx*ny*nz);

   if (!U){
       printf("Cannot allocate 3D _prec array\n");
       exit(-1);
   }
   for(i=0;i<nx;i++){
       U[i] = ((_prec* *) U) + nx + i*ny;
    }

   _prec *Ustart = (_prec *) (U[nx-1] + ny);
   for(i=0;i<nx;i++)
       for(j=0;j<ny;j++)
           U[i][j] = Ustart + i*ny*nz + j*nz;

   for(i=0;i<nx;i++)
       for(j=0;j<ny;j++)
           for(k=0;k<nz;k++)
              U[i][j][k] = 0.0f;

   return U;
}

Grid3Dww Alloc3Dww(int nx, int ny, int nz)
{
  int i, j, k;
  Grid3Dww U = (Grid3Dww)malloc(sizeof(int**)*nx + sizeof(int *)*nx*ny +sizeof(int)*nx*ny*nz);
  if (!U){ 
    printf("Cannot allocate 3D int array\n");
    exit(-1);
  }
  for(i=0;i<nx;i++){
    U[i] = ((int**) U) + nx + i*ny;
  }
  int *Ustart = (int *) (U[nx-1] + ny);
  for(i=0;i<nx;i++)
    for(j=0;j<ny;j++)
      U[i][j] = Ustart + i*ny*nz + j*nz;

  for(i=0;i<nx;i++)
    for(j=0;j<ny;j++)
      for(k=0;k<nz;k++)
	U[i][j][k] = 0;
  return U;
}



Grid1D Alloc1D(long nx)
{
   long i;
   Grid1D U = (Grid1D)malloc(sizeof(_prec)*nx);
   if (!U){
       printf("Cannot allocate 2D _prec array\n");
       exit(-1);
   }

   for(i=0;i<nx;i++)
       U[i] = 0.0f;

   return U;
}


PosInf Alloc1P(int nx)
{
   int i;
   PosInf U = (PosInf)malloc(sizeof(int)*nx);

   if (!U){
       printf("Cannot allocate 2D integer array\n");
       exit(-1);
   }

   for(i=0;i<nx;i++)
       U[i] = 0;

   return U;
}

void Delloc3D(Grid3D U)
{
   if (U) 
   {
      free(U);
      U = NULL;
   }

   return;
}

void Delloc3Dww(Grid3Dww U)
{  
  if (U)
    {
      free(U);
      U = NULL;
    }
  return;
}



void Delloc1D(Grid1D U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}

void Delloc1P(PosInf U)
{
   if (U)
   {
      free(U);
      U = NULL;
   }

   return;
}

