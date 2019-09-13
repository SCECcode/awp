#include <math.h>
#include <stdio.h>
#include <awp/pmcl3d.h>

void inicrj(_prec ARBC, int *coords, int nxt, int nyt, int nzt, int NX, int NY, int ND, Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz, int islowest, int NPC)
{
  int nxp, nyp, nzp;
  int i,   j,   k;
  _prec alpha;
  alpha = sqrt(-log(ARBC))/ND;

  nxp   = nxt*coords[0] + 1;
  if ((nxp <= ND) && (NPC < 2))  /* added by Daniel for periodic BCs */
  {
     for(i=0;i<ND;i++)
     {
        nxp        = i + 1;
        dcrjx[i+2+ngsl] = dcrjx[i+2+ngsl]*(exp(-((alpha*(ND-nxp+1))*(alpha*(ND-nxp+1)))));
     } 
  }
  nxp   = nxt*coords[0] + 1;
  if( ((nxp+nxt-1) >= (NX-ND+1)) && (NPC < 2)) /* added by Daniel for periodic BCs */
  {
     for(i=nxt-ND;i<nxt;i++)
     {
        nxp        = i + NX - nxt + 1;
        dcrjx[i+2+ngsl] = dcrjx[i+2+ngsl]*(exp(-((alpha*(ND-(NX-nxp)))*(alpha*(ND-(NX-nxp))))));
     }
  }

  nyp   = nyt*coords[1] + 1;
  if((nyp <= ND) && (NPC < 2)) /* added by Daniel for periodic BCs */
  {
     for(j=0;j<ND;j++)
     {
        nyp        = j + 1;
        dcrjy[j+2+ngsl] = dcrjy[j+2+ngsl]*(exp(-((alpha*(ND-nyp+1))*(alpha*(ND-nyp+1)))));
     }
  }
  nyp   = nyt*coords[1] + 1;
  if(((nyp+nyt-1) >= (NY-ND+1)) && (NPC < 2))
  {
     for(j=nyt-ND;j<nyt;j++)
     {
        nyp        = j + NY - nyt + 1;
        dcrjy[j+2+ngsl] = dcrjy[j+2+ngsl]*(exp(-((alpha*(ND-(NY-nyp)))*((alpha*(ND-(NY-nyp)))))));
     }
  }

  /* in the vertical direction, the Cerjan ABCs are only set for the lowest grid */
  if (islowest){
     nzp = 1;
     if(nzp <= ND)
     {
	for(k=0;k<ND;k++)
	{
	   nzp            = k + 1;
	   dcrjz[k+align] = dcrjz[k+align]*(exp(-((alpha*(ND-nzp+1))*((alpha*(ND-nzp+1))))));
	}
     }
  }
  return;
}  

