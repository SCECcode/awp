#include <math.h>
#include <stdio.h>
#include <awp/pmcl3d.h>

void inicrj(_prec ARBC, int *coords, int nxt, int nyt, int nzt, int NX, int NY, int ND, Grid1D dcrjx, Grid1D dcrjy, Grid1D dcrjz, int islowest, int NPC)
{
 
  int i,   j,   k, ix, iy, iz;
  _prec alpha;
  alpha = sqrt(-log(ARBC))/(ND-1);

    if (NPC < 2)
    {
        for(i=0;i<4+nxt;i++)
        {
            ix = nxt*coords[0] + i + 1 - 2; //ix is one-indexing
            if((ix>=1) && (ix<=ND))
            {
                dcrjx[ngsl+i] = dcrjx[ngsl+i]*(exp(-((alpha*(ND-ix))*(alpha*(ND-ix)))));
            }

            if( (ix>=(NX-ND+1)) && (ix<=NX))
            {
                dcrjx[ngsl+i] = dcrjx[ngsl+i]*(exp(-((alpha*(ix-(NX-ND)-1))*(alpha*(ix-(NX-ND)-1)))));
            }

            if((ix<1) || (ix>NX))
            {
                dcrjx[ngsl+i] = ARBC;
            }
        }

        for(j=0;j<4+nyt;j++)
        {
            iy = nyt*coords[1] + j + 1 - 2; //iy is one-indexing
            if((iy>=1) && (iy<=ND))
            {
                dcrjy[ngsl+j] = dcrjy[ngsl+j]*(exp(-((alpha*(ND-iy))*(alpha*(ND-iy)))));
            }

            if((iy>=(NY-ND+1)) && (iy<NY))
            {
                dcrjy[ngsl+j] = dcrjy[ngsl+j]*(exp(-((alpha*(iy-(NY-ND)-1))*(alpha*(iy-(NY-ND)-1)))));
            }

            if((iy<1) || (iy>NY))
            {
                dcrjy[ngsl+j] = ARBC;
            }
        }
    }



  /* in the vertical direction, the Cerjan ABCs are only set for the lowest grid */
    if (islowest)
    {
        for(k=0;k<ND;k++)
        {
            iz=k+1; //iz is one-indexing
            dcrjz[k+align] = dcrjz[k+align]*(exp(-((alpha*(ND-iz))*(alpha*(ND-iz)))));
        }
    }

  return;
}  

