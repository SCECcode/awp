/**************************************************************************
 * Efecan updated on Sep 30, 2013
 *    Plasticity NVE=3 option is added into inimesh
 * Efecan updated on Oct 4, 2012
 *    MEDIARESTART=3 is added for partitioned large mesh reading
 *
***************************************************************************/
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <awp/pmcl3d.h>
#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif

void inimesh(int rank, int MEDIASTART, Grid3D d1, Grid3D mu, Grid3D lam, Grid3D qp, Grid3D qs, float *taumax, float *taumin,
	     Grid3D tau, Grid3D weights,Grid1D coeff, 
	     int nvar, float FP,  float FAC, float Q0, float EX, int nxt, int nyt, int nzt, int PX, int PY, int NX, int NY, 
             int NZ, int *coords, MPI_Comm MCW, int IDYNA, int NVE, int SoCalQ, char *INVEL, 
             float *vse, float *vpe, float *dde)
{
  int merr;
  int i,j,k,err;
  float vp,vs,dd,pi; 
  int   rmtype[3], rptype[3], roffset[3];
  MPI_Datatype readtype;
  MPI_Status   filestatus;
  MPI_File     fh;
  char mpiErrStr[100];
  int mpiErrStrLen;

  pi      = 4.*atan(1.);
  //  *taumax = 1./(2*pi*0.01)*10.0*FAC;                                                                                                                            
  *taumax = 1./(2*pi*0.01)*1.0*FAC;
  if(EX<0.65 && EX>=0.01) *taumin = 1./(2*pi*10.0)*0.2*FAC;
  else if(EX<0.85 && EX>=0.65) {
    *taumin = 1./(2*pi*12.0)*0.5*FAC;
    *taumax = 1./(2*pi*0.08)*2.0*FAC;
  }
  else if (EX<0.95 && EX>=0.85){
    //(EX<0.95 && EX>=0.85) *taumin = 1./(2*pi*280.0)*0.1*FAC;                                                                                                      
    //  else if(EX<0.01) *taumin = 1./(2*pi*40.0)*0.1*FAC;                                                                                                        
    *taumin = 1./(2*pi*15.0)*0.8*FAC;
    *taumax = 1./(2*pi*0.1)*2.5*FAC;
  }

  else if(EX<0.01) *taumin = 1./(2*pi*10.0)*0.2*FAC;


  tausub(tau, *taumin, *taumax);
  if(!coords[0] && !coords[1]) printf("tau: %e,%e; %e,%e; %e,%e; %e,%e\n",
				      tau[0][0][0],tau[1][0][0],tau[0][1][0],tau[1][1][0],
				      tau[0][0][1],tau[1][0][1],tau[0][1][1],tau[1][1][1]);
  MPI_Comm_rank(MCW,&rank);        
  if(MEDIASTART==0)
  {
    //    *taumax = 1./(2*pi*FL);
    // *taumin = 1./(2*pi*FH);
    if(IDYNA==1) 
    {
       vp=6000.0;
       vs=3464.0;
       dd=2670.0;
    }
    else
    {
       vp=6000.0; // 4800.0;
       vs=3464.0; // 2800.0;
       dd=2670.0; //2500.0;
    }

    for(i=0;i<nxt+4+ngsl2;i++)
      for(j=0;j<nyt+4+ngsl2;j++)
        for(k=0;k<nzt+2*align;k++)
        {
           lam[i][j][k]=1./(dd*(vp*vp - 2.*vs*vs));
           if (vs > 0.0f) mu[i][j][k]=1./(dd*vs*vs);
           else mu[i][j][k]=1.e7;
           mu[i][j][k]=1./(dd*vs*vs);
           d1[i][j][k]=dd;
        }
  }
  else 
  {
      Grid3D tmpvp=NULL, tmpvs=NULL, tmpdd=NULL;
      Grid3D tmppq=NULL, tmpsq=NULL; 
      int var_offset;

      tmpvp = Alloc3D(nxt, nyt, nzt);
      tmpvs = Alloc3D(nxt, nyt, nzt);
      tmpdd = Alloc3D(nxt, nyt, nzt);
      for(i=0;i<nxt;i++)
        for(j=0;j<nyt;j++)
          for(k=0;k<nzt;k++)
          {
             tmpvp[i][j][k]=0.0f;
             tmpvs[i][j][k]=0.0f;
             tmpdd[i][j][k]=0.0f;
          }

      if(NVE==1 || NVE==3)
      {
        tmppq = Alloc3D(nxt, nyt, nzt);
        tmpsq = Alloc3D(nxt, nyt, nzt);
        for(i=0;i<nxt;i++)
          for(j=0;j<nyt;j++)
            for(k=0;k<nzt;k++)
            {
               tmppq[i][j][k]=0.0f;
               tmpsq[i][j][k]=0.0f;
            }
      }

      if(nvar==8) 
      {
          var_offset=3;
      }
      else if(nvar==5)
      {
          var_offset=0;
      }
      else
      {
          var_offset=0;
      }

      if(MEDIASTART>=1 && MEDIASTART<=3)
      {
          char filename[40];
          if(MEDIASTART<3) sprintf(filename,INVEL);
          else if(MEDIASTART==3){
            sprintf(filename,"input_rst/mediapart/media%07d.bin",rank);
            if(rank%100==0) printf("Rank=%d, reading file=%s\n",rank,filename);
          }
          Grid1D tmpta = Alloc1D(nvar*nxt*nyt*nzt);
          if(MEDIASTART==3 || (PX==1 && PY==1))
          {
             FILE   *file;
             file = fopen(filename,"rb");
             if(!file)
             {
                printf("can't open file %s", filename);
                return;
             }
             if(!fread(tmpta,sizeof(float),nvar*nxt*nyt*nzt,file))
             {
                printf("can't read file %s", filename);
                return;
             }
             //printf("%d) 0-0-0,1-10-3=%f, %f\n",rank,tmpta[0],tmpta[1+10*nxt+3*nxt*nyt]);
          }
          else{
	    printf("%d) Media file will be read using MPI-IO\n", rank); 
		    rmtype[0]  = NZ;
    		    rmtype[1]  = NY;
    		    rmtype[2]  = NX*nvar;
    		    rptype[0]  = nzt;
    		    rptype[1]  = nyt;
    		    rptype[2]  = nxt*nvar;
    		    roffset[0] = 0;
    		    roffset[1] = nyt*coords[1];
    		    roffset[2] = nxt*coords[0]*nvar;
    		    err = MPI_Type_create_subarray(3, rmtype, rptype, roffset, MPI_ORDER_C, MPI_FLOAT, &readtype);
		    if(err != MPI_SUCCESS){
		      MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
		      printf("%d) ERROR! MPI-IO mesh reading create subarray: %s\n",rank,mpiErrStr);
		    }
    		    err = MPI_Type_commit(&readtype);
		    if(err != MPI_SUCCESS){
		      MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
		      printf("%d) ERROR! MPI-IO mesh reading commit: %s\n",rank,mpiErrStr);
		    }
		    err = MPI_File_open(MCW,filename,MPI_MODE_RDONLY,MPI_INFO_NULL,&fh);
		    if(err != MPI_SUCCESS){
		      MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
		      printf("%d) ERROR! MPI-IO mesh reading file open: %s\n",rank,mpiErrStr);
		    }
		    err = MPI_File_set_view(fh, 0, MPI_FLOAT, readtype, "native", MPI_INFO_NULL);
		    if(err != MPI_SUCCESS){
		      MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
		      printf("%d) ERROR! MPI-IO mesh reading file set view: %s\n",rank,mpiErrStr);
		    }
		    err = MPI_File_read_all(fh, tmpta, nvar*nxt*nyt*nzt, MPI_FLOAT, &filestatus);
		    if(err != MPI_SUCCESS){
		      MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
		      printf("%d) ERROR! MPI-IO mesh reading file read: %s\n",rank,mpiErrStr);
		    }
		    err = MPI_File_close(&fh);
		    if(err != MPI_SUCCESS){
		      MPI_Error_string(err, mpiErrStr, &mpiErrStrLen);
		      printf("%d) ERROR! MPI-IO mesh reading file close: %s\n",rank,mpiErrStr);
		    }
		    if(!rank) printf("Media file is read using MPI-IO\n");

          }
          for(k=0;k<nzt;k++)
            for(j=0;j<nyt;j++)
              for(i=0;i<nxt;i++){
              	tmpvp[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset];
              	tmpvs[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+1];
               	tmpdd[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+2];
              	if(nvar>3){
                	tmppq[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+3];
                	tmpsq[i][j][k]=tmpta[(k*nyt*nxt+j*nxt+i)*nvar+var_offset+4];
                }
                if(tmpvp[i][j][k]!=tmpvp[i][j][k] ||
                    tmpvs[i][j][k]!=tmpvs[i][j][k] ||
                    tmpdd[i][j][k]!=tmpdd[i][j][k]){
		  printf("%d) tmpvp,vs,dd is NAN!\n",rank);
                      MPI_Abort(MPI_COMM_WORLD,1);
                }
         }
         //printf("%d) vp,vs,dd[0^3]=%f,%f,%f\n",rank,tmpvp[0][0][0],
         //   tmpvs[0][0][0], tmpdd[0][0][0]);
         Delloc1D(tmpta);
      }
      
      if(nvar==3 && (NVE==1 || NVE==3))
      {
        for(i=0;i<nxt;i++)
          for(j=0;j<nyt;j++){
            for(k=0;k<nzt;k++){
                /*if (tmpvs[i][j][k] > 0)
                   tmpsq[i][j][k]=0.05*tmpvs[i][j][k];
                else 
                   tmpsq[i][j][k]=50.;*/
                tmpsq[i][j][k]=0.05*tmpvs[i][j][k];
                tmppq[i][j][k]=2.0*tmpsq[i][j][k];
            }
          }
      }
      
      float w0=0.0f, ww1=0.0f, w2=0.0f, tmp1=0.0f, tmp2=0.0f;
      float qpinv=0.0f, qsinv=0.0f, vpvs=0.0f;
      if(NVE==1 || NVE==3)
      {
         w0=2*pi*FP;
	 //         ww1=2*pi*FL;
         //w2=2*pi*FH;
	 // *taumax=1./ww1;
	 // *taumin=1./w2;
         //tmp1=2./pi*(log((*taumax)/(*taumin)));
         //tmp2=2./pi*log(w0*(*taumin));
         if(!rank) printf("w0 = %g\n",w0); 
      }       

      vse[0] = 1.0e10;
      vpe[0] = 1.0e10;
      dde[0] = 1.0e10;
      vse[1] = -1.0e10;
      vpe[1] = -1.0e10;
      dde[1] = -1.0e10;
      float facex = (float)pow(FAC,EX);
      float mu1, denom;
      float val[2];
      int ii,jj,kk,iii,jjj,kkk,num;
      float weights_los[2][2][2];
      float weights_lop[2][2][2];
      double complex value;
      double complex sqrtm1;
      sqrtm1=1.0 *I;

      for(i=0;i<nxt;i++)
        for(j=0;j<nyt;j++)
          for(k=0;k<nzt;k++)
          {
	    //             tmpvs[i][j][k] = tmpvs[i][j][k]*(1+ ( log(w2/w0) )/(pi*tmpsq[i][j][k]) );
            // tmpvp[i][j][k] = tmpvp[i][j][k]*(1+ ( log(w2/w0) )/(pi*tmppq[i][j][k]) );
        //    tmpsq[i][j][k] = 10000.;
        //    tmppq[i][j][k] = 10000.;
        //      tmpsq[i][j][k] = 200;
        //      tmppq[i][j][k] = 200;
        //   if(tmpvs[i][j][k]>0.0){
        //   if(tmpvs[i][j][k]<2500.0){
        //      vpvs=tmpvp[i][j][k]/tmpvs[i][j][k];
        //      tmpvs[i][j][k]=2500.0;
        //      tmpvp[i][j][k]=tmpvs[i][j][k]*vpvs;
        //      }
        //      }
             /*if(tmpvs[i][j][k]>0.0){
             if(tmpvs[i][j][k]<800.0){
                vpvs=tmpvp[i][j][k]/tmpvs[i][j][k];
                tmpvs[i][j][k]=800.0;
                tmpvp[i][j][k]=tmpvs[i][j][k]*vpvs;
                }
                }*/
	    /*if(tmpvs[i][j][k]<333.33)
	      {
		//		tmpvs[i][j][k]=200.0;
                //tmpvp[i][j][k]=600.0;
		//                 tmpsq[i][j][k] = 20;                                                                                                                             
                // tmppq[i][j][k] = 20;                                                                                                                             
                tmpvp[i][j][k] = 1500.;
                tmpvs[i][j][k] = 0.;
                tmpdd[i][j][k] = 1025.;
                tmpsq[i][j][k] = 50.;
                tmppq[i][j][k] = 100.;
	      }*/
	    /*tmpsq[i][j][k] = 0.05  * tmpvs[i][j][k];
	    tmppq[i][j][k] = 2.0   * tmpsq[i][j][k];*/

	    if(tmppq[i][j][k]>200.0)
	      {
		// QF - VP                                                                                                                                             
		val[0] = 0.0;
		val[1] = 0.0;
		for(ii=0;ii<2;ii++)
		  for(jj=0;jj<2;jj++)
		    for(kk=0;kk<2;kk++){
		      denom = ((w0*w0*tau[ii][jj][kk]*tau[ii][jj][kk]+1.0)*tmppq[i][j][k]*facex);
		      val[0] += weights[ii][jj][kk]/denom;
		      val[1] += -weights[ii][jj][kk]*w0*tau[ii][jj][kk]/denom;
		    }
		mu1 = tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k]/(1.0-val[0]);
	      }
	    else if (tmppq[i][j][k] > 0.)
	      {
		//              if(rank==0) printf("coeff[num]1,2 = %g %g\n",coeff[0],coeff[1]);                                                                   
		num=0;
		for (iii=0;iii<2;iii++)
		  for(jjj=0;jjj<2;jjj++)
		    for(kkk=0;kkk<2;kkk++){
		      weights_lop[iii][jjj][kkk]=coeff[num]/(tmppq[i][j][k]*tmppq[i][j][k])+coeff[num+1]/(tmppq[i][j][k]);
		      num=num+2;
		    }
		//              if(rank==0) printf("weights_lop %g\n",weights_lop[0][0][0]);                                                                       

		value=0.0+0.0*sqrtm1;
		for(ii=0;ii<2;ii++)
		  for(jj=0;jj<2;jj++)
		    for(kk=0;kk<2;kk++){
		      value=value+1./(    1.-weights_lop[ii][jj][kk]/(1+sqrtm1*w0*tau[ii][jj][kk]));
		    }
		value=1./value;
		//              if(rank==0) printf("creal(value) %f\n",creal(value));                                                                              
		// if(rank==0) printf("sqrtm1 %f\n",cimag(sqrtm1));                                                                                                
		mu1=tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k]/(8.*creal(value)) ;
		//      if(rank==0) printf("mu1 %g\n",mu1);                                                                                                        
	      }

	    else {
                mu1=tmpdd[i][j][k]*tmpvp[i][j][k]*tmpvp[i][j][k];
            }

	    tmpvp[i][j][k] = sqrt(mu1/tmpdd[i][j][k]);

	    // QF - VS                                                                                                                                             
	    if(tmpsq[i][j][k]>200.0)
	      {
		val[0] = 0.0;
		val[1] = 0.0;
		for(ii=0;ii<2;ii++)
		  for(jj=0;jj<2;jj++)
		    for(kk=0;kk<2;kk++){
		      denom = ((w0*w0*tau[ii][jj][kk]*tau[ii][jj][kk]+1.0)*tmpsq[i][j][k]*facex);
		      val[0] += weights[ii][jj][kk]/denom;
		      val[1] += -weights[ii][jj][kk]*w0*tau[ii][jj][kk]/denom;
		    }
		mu1 = tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]/(1.0-val[0]);
	      }
	    else if (tmpsq[i][j][k] > 0.)
	      {
		num=0;
		for (iii=0;iii<2;iii++)
		  for(jjj=0;jjj<2;jjj++)
		    for(kkk=0;kkk<2;kkk++){
		      weights_los[iii][jjj][kkk]=coeff[num]/(tmpsq[i][j][k]*tmpsq[i][j][k])+coeff[num+1]/(tmpsq[i][j][k]);
		      num=num+2;
		    }
		value=0.0+0.0*sqrtm1;
		for(ii=0;ii<2;ii++)
		  for(jj=0;jj<2;jj++)
		    for(kk=0;kk<2;kk++){
		      value=value+1./(    1.-weights_los[ii][jj][kk]/(1+sqrtm1*w0*tau[ii][jj][kk]));
		    }
		value=1./value;
		mu1=tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]/(8.*creal(value));

	      }
	    else {
                mu1=tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k];
            }

	    tmpvs[i][j][k] = sqrt(mu1/tmpdd[i][j][k]);
            #ifdef DEBUGVP
	    if ((tmpvp[i][j][k] < 1.e3) || (tmpvs[i][j][k] < 100. && tmpvs[i][j][k] > 0.)) {
	       fprintf(stderr, "rank %d @ %d,%d,%d: vp=%e, vs=%e,rho=%e\n", rank, i, j, k, tmpvp[i][j][k], 
                  tmpvs[i][j][k], tmpdd[i][j][k]);
            }	    
            #endif
	    // QF - end  

            if (tmpvs[i][j][k] == 0.f) {
                tmpvs[i][j][k] = 0.001;
                tmpsq[i][j][k] = 50.;
                tmppq[i][j][k] = 100.;
            }

             if (SoCalQ==1)
             {
                vpvs=tmpvp[i][j][k]/tmpvs[i][j][k];
                if (vpvs<1.45)  tmpvs[i][j][k]=tmpvp[i][j][k]/1.45;
             }
             /*if(tmpvp[i][j][k]>9000.0){
                tmpvs[i][j][k]=5196.0;
                tmpvp[i][j][k]=9000.0;
                }*/

             /*if(tmpvp[i][j][k]>7600.0){
	       tmpvs[i][j][k]=4387.0;
	       tmpvp[i][j][k]=7600.0;
             }*/
           /*if(tmpvs[i][j][k]<500.0){
	      tmpvs[i][j][k]=500.0;
	      if (tmpvp[i][j][k] < 725.0) tmpvp[i][j][k] = 725.0;
           }*/
               //if(tmpdd[i][j][k]<1700.0) tmpdd[i][j][k]=1700.0;


             //if(tmpvs[i][j][k]<400.0)
             /*if(tmpvs[i][j][k]<200.0)
             {
                //tmpvs[i][j][k]=400.0;
                //tmpvp[i][j][k]=1200.0;
                tmpvs[i][j][k]=200.0;
                tmpvp[i][j][k]=600.0;
             }*/
             /*if(tmpvp[i][j][k]>6500.0){
                tmpvs[i][j][k]=3752.0;
                tmpvp[i][j][k]=6500.0;
             }*/
//             if(tmpdd[i][j][k]<1700.0) tmpdd[i][j][k]=1700.0;   
             mu[i+2+ngsl][j+2+ngsl][(nzt+align-1) - k]  = 1./(tmpdd[i][j][k]*tmpvs[i][j][k]*tmpvs[i][j][k]);
             lam[i+2+ngsl][j+2+ngsl][(nzt+align-1) - k] = 1./(tmpdd[i][j][k]*(tmpvp[i][j][k]*tmpvp[i][j][k]
                                                                              -2.*tmpvs[i][j][k]*tmpvs[i][j][k]));
             d1[i+2+ngsl][j+2+ngsl][(nzt+align-1) - k]  = tmpdd[i][j][k];
             if(NVE==1 || NVE==3)
             {
                if(tmppq[i][j][k]<=0.0)
                {
                   qpinv=0.0;
                   qsinv=0.0;
                } 
                else
                {
                   qpinv=1./tmppq[i][j][k];
                   qsinv=1./tmpsq[i][j][k];
                }
		//                tmppq[i][j][k]=tmp1*qpinv/(1.0-tmp2*qpinv);
		// tmpsq[i][j][k]=tmp1*qsinv/(1.0-tmp2*qsinv);
                tmppq[i][j][k] = qpinv/facex;
                tmpsq[i][j][k] = qsinv/facex;
                qp[i+2+ngsl][j+2+ngsl][(nzt+align-1) - k] = tmppq[i][j][k];
                qs[i+2+ngsl][j+2+ngsl][(nzt+align-1) - k] = tmpsq[i][j][k];
             }
             if(tmpvs[i][j][k]<vse[0] && tmpvs[i][j][k] > 1.0f) vse[0] = tmpvs[i][j][k];
             if(tmpvs[i][j][k]>vse[1]) vse[1] = tmpvs[i][j][k];
             if(tmpvp[i][j][k]<vpe[0]) vpe[0] = tmpvp[i][j][k];
             if(tmpvp[i][j][k]>vpe[1]) vpe[1] = tmpvp[i][j][k];
             if(tmpdd[i][j][k]<dde[0]) dde[0] = tmpdd[i][j][k];
             if(tmpdd[i][j][k]>dde[1]) dde[1] = tmpdd[i][j][k];
          }
      Delloc3D(tmpvp);
      Delloc3D(tmpvs);
      Delloc3D(tmpdd);
      if(NVE==1 || NVE==3){
         Delloc3D(tmppq);
         Delloc3D(tmpsq);
      }

      //5 Planes (except upper XY-plane)
      for(j=2+ngsl;j<nyt+2+ngsl;j++)
        for(k=align;k<nzt+align;k++){
          lam[1+ngsl][j][k]     = lam[2+ngsl][j][k];
          lam[nxt+2+ngsl][j][k] = lam[nxt+1+ngsl][j][k];
          mu[1+ngsl][j][k]      = mu[2+ngsl][j][k];
          mu[nxt+2+ngsl][j][k]  = mu[nxt+1+ngsl][j][k];
          d1[1+ngsl][j][k]      = d1[2+ngsl][j][k];
          d1[nxt+2+ngsl][j][k]  = d1[nxt+1+ngsl][j][k];
        }
 
      for(i=2+ngsl;i<nxt+2+ngsl;i++)
        for(k=align;k<nzt+align;k++){
          lam[i][1+ngsl][k]     = lam[i][2+ngsl][k];
          lam[i][nyt+2+ngsl][k] = lam[i][nyt+1+ngsl][k];
          mu[i][1+ngsl][k]      = mu[i][2+ngsl][k];
          mu[i][nyt+2+ngsl][k]  = mu[i][nyt+1+ngsl][k];
          d1[i][1+ngsl][k]      = d1[i][2+ngsl][k];
          d1[i][nyt+2+ngsl][k]  = d1[i][nyt+1+ngsl][k];
        } 

      for(i=2+ngsl;i<nxt+2+ngsl;i++)
        for(j=2+ngsl;j<nyt+2+ngsl;j++){
          lam[i][j][align-1]   = lam[i][j][align];
          mu[i][j][align-1]    = mu[i][j][align];
          d1[i][j][align-1]    = d1[i][j][align];
        }
   
      //12 border lines
      for(i=2+ngsl;i<nxt+2+ngsl;i++){
        lam[i][1+ngsl][align-1]          = lam[i][2+ngsl][align];
        mu[i][1+ngsl][align-1]           = mu[i][2+ngsl][align];
        d1[i][1+ngsl][align-1]           = d1[i][2+ngsl][align];
        lam[i][nyt+2+ngsl][align-1]      = lam[i][nyt+1+ngsl][align];
        mu[i][nyt+2+ngsl][align-1]       = mu[i][nyt+1+ngsl][align];
        d1[i][nyt+2+ngsl][align-1]       = d1[i][nyt+1+ngsl][align];
        lam[i][1+ngsl][nzt+align]        = lam[i][2+ngsl][nzt+align-1];
        mu[i][1+ngsl][nzt+align]         = mu[i][2+ngsl][nzt+align-1];
        d1[i][1+ngsl][nzt+align]         = d1[i][2+ngsl][nzt+align-1];
        lam[i][nyt+2+ngsl][nzt+align]    = lam[i][nyt+1+ngsl][nzt+align-1];
        mu[i][nyt+2+ngsl][nzt+align]     = mu[i][nyt+1+ngsl][nzt+align-1];
        d1[i][nyt+2+ngsl][nzt+align]     = d1[i][nyt+1+ngsl][nzt+align-1];
      }

      for(j=2+ngsl;j<nyt+2+ngsl;j++){
        lam[1+ngsl][j][align-1]          = lam[2+ngsl][j][align];
        mu[1+ngsl][j][align-1]           = mu[2+ngsl][j][align];
        d1[1+ngsl][j][align-1]           = d1[2+ngsl][j][align];
        lam[nxt+2+ngsl][j][align-1]      = lam[nxt+1+ngsl][j][align];
        mu[nxt+2+ngsl][j][align-1]       = mu[nxt+1+ngsl][j][align];
        d1[nxt+2+ngsl][j][align-1]       = d1[nxt+1+ngsl][j][align];
        lam[1+ngsl][j][nzt+align]        = lam[2+ngsl][j][nzt+align-1];
        mu[1+ngsl][j][nzt+align]         = mu[2+ngsl][j][nzt+align-1];
        d1[1+ngsl][j][nzt+align]         = d1[2+ngsl][j][nzt+align-1];
        lam[nxt+2+ngsl][j][nzt+align]    = lam[nxt+1+ngsl][j][nzt+align-1];
        mu[nxt+2+ngsl][j][nzt+align]     = mu[nxt+1+ngsl][j][nzt+align-1];
        d1[nxt+2+ngsl][j][nzt+align]     = d1[nxt+1+ngsl][j][nzt+align-1];
      }

      for(k=align;k<nzt+align;k++){
        lam[1+ngsl][1+ngsl][k]         = lam[2+ngsl][2+ngsl][k];
        mu[1+ngsl][1+ngsl][k]          = mu[2+ngsl][2+ngsl][k];
        d1[1+ngsl][1+ngsl][k]          = d1[2+ngsl][2+ngsl][k];
        lam[nxt+2+ngsl][1+ngsl][k]     = lam[nxt+1+ngsl][2+ngsl][k];
        mu[nxt+2+ngsl][1+ngsl][k]      = mu[nxt+1+ngsl][2+ngsl][k];
        d1[nxt+2+ngsl][1+ngsl][k]      = d1[nxt+1+ngsl][2+ngsl][k];
        lam[1+ngsl][nyt+2+ngsl][k]     = lam[2+ngsl][nyt+1+ngsl][k];
        mu[1+ngsl][nyt+2+ngsl][k]      = mu[2+ngsl][nyt+1+ngsl][k];
        d1[1+ngsl][nyt+2+ngsl][k]      = d1[2+ngsl][nyt+1+ngsl][k];
        lam[nxt+2+ngsl][nyt+2+ngsl][k] = lam[nxt+1+ngsl][nyt+1+ngsl][k];
        mu[nxt+2+ngsl][nyt+2+ngsl][k]  = mu[nxt+1+ngsl][nyt+1+ngsl][k];
        d1[nxt+2+ngsl][nyt+2+ngsl][k]  = d1[nxt+1+ngsl][nyt+1+ngsl][k];
      }

      //8 Corners
      lam[1+ngsl][1+ngsl][align-1]           = lam[2+ngsl][2+ngsl][align];
      mu[1+ngsl][1+ngsl][align-1]            = mu[2+ngsl][2+ngsl][align];
      d1[1+ngsl][1+ngsl][align-1]            = d1[2+ngsl][2+ngsl][align];
      lam[nxt+2+ngsl][1+ngsl][align-1]       = lam[nxt+1+ngsl][2+ngsl][align];
      mu[nxt+2+ngsl][1+ngsl][align-1]        = mu[nxt+1+ngsl][2+ngsl][align];
      d1[nxt+2+ngsl][1+ngsl][align-1]        = d1[nxt+1+ngsl][2+ngsl][align];
      lam[1+ngsl][nyt+2+ngsl][align-1]       = lam[2+ngsl][nyt+1+ngsl][align];
      mu[1+ngsl][nyt+2+ngsl][align-1]        = mu[2+ngsl][nyt+1+ngsl][align];
      d1[1+ngsl][nyt+2+ngsl][align-1]        = d1[2+ngsl][nyt+1+ngsl][align];
      lam[1+ngsl][1+ngsl][nzt+align]         = lam[2+ngsl][2+ngsl][nzt+align-1];
      mu[1+ngsl][1+ngsl][nzt+align]          = mu[2+ngsl][2+ngsl][nzt+align-1];
      d1[1+ngsl][1+ngsl][nzt+align]          = d1[2+ngsl][2+ngsl][nzt+align-1];
      lam[nxt+2+ngsl][1+ngsl][nzt+align]     = lam[nxt+1+ngsl][2+ngsl][nzt+align-1];
      mu[nxt+2+ngsl][1+ngsl][nzt+align]      = mu[nxt+1+ngsl][2+ngsl][nzt+align-1];
      d1[nxt+2+ngsl][1+ngsl][nzt+align]      = d1[nxt+1+ngsl][2+ngsl][nzt+align-1];
      lam[nxt+2+ngsl][nyt+2+ngsl][align-1]   = lam[nxt+1+ngsl][nyt+1+ngsl][align];
      mu[nxt+2+ngsl][nyt+2+ngsl][align-1]    = mu[nxt+1+ngsl][nyt+1+ngsl][align];
      d1[nxt+2+ngsl][nyt+2+ngsl][align-1]    = d1[nxt+1+ngsl][nyt+1+ngsl][align];
      lam[1+ngsl][nyt+2+ngsl][nzt+align]     = lam[2+ngsl][nyt+1+ngsl][nzt+align-1];
      mu[1+ngsl][nyt+2+ngsl][nzt+align]      = mu[2+ngsl][nyt+1+ngsl][nzt+align-1];
      d1[1+ngsl][nyt+2+ngsl][nzt+align]      = d1[2+ngsl][nyt+1+ngsl][nzt+align-1];
      lam[nxt+2+ngsl][nyt+2+ngsl][nzt+align] = lam[nxt+1+ngsl][nyt+1+ngsl][nzt+align-1];
      mu[nxt+2+ngsl][nyt+2+ngsl][nzt+align]  = mu[nxt+1+ngsl][nyt+1+ngsl][nzt+align-1];
      d1[nxt+2+ngsl][nyt+2+ngsl][nzt+align]  = d1[nxt+1+ngsl][nyt+1+ngsl][nzt+align-1];

      k = nzt+align;
      for(i=2+ngsl;i<nxt+2+ngsl;i++)
        for(j=2+ngsl;j<nyt+2+ngsl;j++){
           d1[i][j][k]   = d1[i][j][k-1];
           mu[i][j][k]   = mu[i][j][k-1];
           lam[i][j][k]  = lam[i][j][k-1];
           if(NVE==1 || NVE==3){
             qp[i][j][k] = qp[i][j][k-1];
             qs[i][j][k] = qs[i][j][k-1];
           }
        }

      float tmpvse[2],tmpvpe[2],tmpdde[2];
      merr = MPI_Allreduce(vse,tmpvse,2,MPI_FLOAT,MPI_MAX,MCW);
      merr = MPI_Allreduce(vpe,tmpvpe,2,MPI_FLOAT,MPI_MAX,MCW);
      merr = MPI_Allreduce(dde,tmpdde,2,MPI_FLOAT,MPI_MAX,MCW);
      vse[1] = tmpvse[1];
      vpe[1] = tmpvpe[1];
      dde[1] = tmpdde[1];
      merr = MPI_Allreduce(vse,tmpvse,2,MPI_FLOAT,MPI_MIN,MCW);
      merr = MPI_Allreduce(vpe,tmpvpe,2,MPI_FLOAT,MPI_MIN,MCW);
      merr = MPI_Allreduce(dde,tmpdde,2,MPI_FLOAT,MPI_MIN,MCW);
      vse[0] = tmpvse[0];
      vpe[0] = tmpvpe[0];
      dde[0] = tmpdde[0];
  } 

  // initial stress computation for plasticity
  // moved invocation of drpr to be called after mediaswap - Daniel

  return;
}

float* matmul3(float *a, float *b){
   int i, j, k;
   float *c;
   
   c=(float*) calloc(9, sizeof(float));
   for (i=0; i<3; i++){
      for (j=0; j<3; j++){
         for (k=0; k<3; k++) c[i*3+j]+=a[i*3+k]*b[k*3+j];
      }
   }   
   return(c);
}

float* transpose(float *a){
   float *b;
   int k, l;

   b=(float*) calloc(9, sizeof(float));
   for (l=0; l<3; l++){
      for (k=0; k<3; k++){
          b[l*3+k]=a[k*3+l];
      }
   }

   return(b);
}

float* rotate_principal(float sigma1, float sigma2, float sigma3, float *strike, float *dip){

      float alpha[3], beta[3];
      float *ss, *Rz, *ssp;
      int k;

      ssp=(float*) calloc(9, sizeof(float));
      ss=(float*) calloc(9, sizeof(float));
      Rz=(float*) calloc(9, sizeof(float));

      ss[0] = sigma1;
      ss[4] = sigma2;
      ss[8] = sigma3;

      for (k=0; k<3; k++){
         alpha[k] = strike[k] / 180. * M_PI;
         beta[k] = dip[k] / 180. * M_PI;
      }
  
      Rz[0] = cosf(alpha[0]) * cosf(beta[0]);
      Rz[1] = sinf(alpha[0]) * cosf(beta[0]);
      Rz[2] = sinf(beta[0]);

      Rz[3] = cosf(alpha[1]) * cosf(beta[1]);
      Rz[4] = sinf(alpha[1]) * cosf(beta[1]);
      Rz[5] = sinf(beta[1]);

      Rz[6] = cosf(alpha[2]) * cosf(beta[2]);
      Rz[7] = sinf(alpha[2]) * cosf(beta[2]);
      Rz[8] = sinf(beta[2]);

      ssp = matmul3(matmul3(transpose(Rz), ss), Rz);

      free(Rz);
      free(ss);

      return(ssp);
}


void tausub( Grid3D tau, float taumin,float taumax) 
{
  int idx, idy, idz;
  float tautem[2][2][2];
  float tmp;

  tautem[0][0][0]=1.0;
  tautem[1][0][0]=6.0;
  tautem[0][1][0]=7.0;
  tautem[1][1][0]=4.0;
  tautem[0][0][1]=8.0;
  tautem[1][0][1]=3.0;
  tautem[0][1][1]=2.0;
  tautem[1][1][1]=5.0;

  for(idx=0;idx<2;idx++)
    for(idy=0;idy<2;idy++)
      for(idz=0;idz<2;idz++)  
      {
         tmp = tautem[idx][idy][idz];
         tmp = (tmp-0.5)/8.0;
         tmp = 2.0*tmp - 1.0;
         
         tau[idx][idy][idz] = exp(0.5*(log(taumax*taumin) + log(taumax/taumin)*tmp));         
      }
     
  return;
}


void weights_sub(Grid3D weights,Grid1D coeff, float ex, float fac){

  int i,j,k;

  if(ex<0.15 && ex>=0.01){
    /*    weights[0][0][0] = 0.5574;                                                                                                                                
    weights[1][0][0] = 1.1952;                                                                                                                                      
    weights[0][1][0] = 1.2418;                                                                                                                                      
    weights[1][1][0] = 1.0234;                                                                                                                                      
    weights[0][0][1] = 1.2973;                                                                                                                                      
    weights[1][0][1] = 0.8302;                                                                                                                                      
    weights[0][1][1] = 0.7193;                                                                                                                                      
    weights[1][1][1] = 1.2744;  */

    weights[0][0][0] =0.3273;
    weights[1][0][0] =1.0434;
    weights[0][1][0] =0.044;
    weights[1][1][0] =0.9393;
    weights[0][0][1] =1.7268;
    weights[1][0][1] =0.369;
    weights[0][1][1] =0.8478;
    weights[1][1][1] =0.4474;

    coeff[0] = 7.3781;
    coeff[1]= 4.1655;
    coeff[2]= -83.1627;
    coeff[3]=13.1326;
    coeff[4]=69.0839;
    coeff[5]=0.4981;
    coeff[6]= -37.6966;
    coeff[7]=5.5263;
    coeff[8]=-51.4056;
    coeff[9]=8.1934;
    coeff[10]=13.1865;
    coeff[11]=3.4775;
    coeff[12]=-36.1049;
    coeff[13]=7.2107;
    coeff[14]=12.3809;
    coeff[15]=3.6117;

  }
  else if(ex<0.25 && ex>=0.15){
    /*    weights[0][0][0] = 0.2219;                                                                                                                                
    weights[1][0][0] = 1.1793;                                                                                                                                      
    weights[0][1][0] = 1.2551;                                                                                                                                      
    weights[1][1][0] = 0.8376;                                                                                                                                      
    weights[0][0][1] = 1.2698;                                                                                                                                      
    weights[1][0][1] = 0.5575;                                                                                                                                      
    weights[0][1][1] = 0.4094;                                                                                                                                      
    weights[1][1][1] = 1.3413;  */

    weights[0][0][0] =0.001;
    weights[1][0][0] =1.0349;
    weights[0][1][0] =0.0497;
    weights[1][1][0] =1.0407;
    weights[0][0][1] =1.7245;
    weights[1][0][1] =0.2005;
    weights[0][1][1] =0.804;
    weights[1][1][1] =0.4452;

    coeff[0] = 31.8902;
    coeff[1]= 1.6126;
    coeff[2]= -83.2611;
    coeff[3]=13.0749;
    coeff[4]=65.485;
    coeff[5]=0.5118;
    coeff[6]= -42.02;
    coeff[7]=5.0875;
    coeff[8]=-49.2656;
    coeff[9]=8.1552;
    coeff[10]=25.7345;
    coeff[11]=2.2801;
    coeff[12]=-40.8942;
    coeff[13]=7.9311;
    coeff[14]=7.0206;
    coeff[15]=3.4692;


  }
  else if(ex<0.35 && ex>=0.25){
    /*    weights[0][0][0] = 0.0818;                                                                                                                                
    weights[1][0][0] = 1.1669;                                                                                                                                      
    weights[0][1][0] = 1.2654;                                                                                                                                      
    weights[1][1][0] = 0.6618;                                                                                                                                      
    weights[0][0][1] = 1.2488;                                                                                                                                      
    weights[1][0][1] = 0.3655;                                                                                                                                      
    weights[0][1][1] = 0.2260;                                                                                                                                      
    weights[1][1][1] = 1.3982;  */
    weights[0][0][0] =0.001;
    weights[1][0][0] =1.0135;
    weights[0][1][0] =0.0621;
    weights[1][1][0] =1.1003;
    weights[0][0][1] =1.7198;
    weights[1][0][1] =0.0918;
    weights[0][1][1] =0.6143;
    weights[1][1][1] =0.4659;

    coeff[0] = 43.775;
    coeff[1]= -0.1091;
    coeff[2]= -83.1088;
    coeff[3]=13.0161;
    coeff[4]=60.9008;
    coeff[5]=0.592;
    coeff[6]= -43.4857;
    coeff[7]=4.5869;
    coeff[8]=-45.3315;
    coeff[9]=8.0252;
    coeff[10]=34.3571;
    coeff[11]=1.199;
    coeff[12]=-41.4422;
    coeff[13]=8.399;
    coeff[14]=-2.8772;
    coeff[15]=3.5323;

  }
  else if(ex<0.45 && ex>=0.35){
    /*    weights[0][0][0] = 0.0305;                                                                                                                                
    weights[1][0][0] = 1.1576;                                                                                                                                      
    weights[0][1][0] = 1.2727;                                                                                                                                      
    weights[1][1][0] = 0.4988;                                                                                                                                      
    weights[0][0][1] = 1.2337;                                                                                                                                      
    weights[1][0][1] = 0.2347;                                                                                                                                      
    weights[0][1][1] = 0.1189;                                                                                                                                      
    weights[1][1][1] = 1.4460;  */
    weights[0][0][0] =0.001;
    weights[1][0][0] =0.9782;
    weights[0][1][0] =0.082;
    weights[1][1][0] =1.1275;
    weights[0][0][1] =1.7122;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.4639;
    weights[1][1][1] =0.509;

    coeff[0] = 41.6858;
    coeff[1]= -0.7344;
    coeff[2]= -164.2252;
    coeff[3]=14.9961;
    coeff[4]=103.0301;
    coeff[5]=-0.4199;
    coeff[6]= -41.1157;
    coeff[7]=3.8266;
    coeff[8]=-73.0432;
    coeff[9]=8.5857;
    coeff[10]=38.0868;
    coeff[11]=0.3937;
    coeff[12]=-43.2133;
    coeff[13]=8.6747;
    coeff[14]=5.6362;
    coeff[15]=3.3287;
  }
  else if(ex<0.55 && ex>=0.45){
    /*    weights[0][0][0] = 0.0193;                                                                                                                                
    weights[1][0][0] = 1.1513;                                                                                                                                      
    weights[0][1][0] = 1.2774;                                                                                                                                      
    weights[1][1][0] = 0.3503;                                                                                                                                      
    weights[0][0][1] = 1.2242;                                                                                                                                      
    weights[1][0][1] = 0.1494;                                                                                                                                      
    weights[0][1][1] = 0.0559;                                                                                                                                      
    weights[1][1][1] = 1.4852;  */
    weights[0][0][0] =0.2073;
    weights[1][0][0] =0.912;
    weights[0][1][0] =0.1186;
    weights[1][1][0] =1.081;
    weights[0][0][1] =1.6984;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.1872;
    weights[1][1][1] =0.6016;

    coeff[0] = 20.0539;
    coeff[1]= -0.4354;
    coeff[2]= -81.6068;
    coeff[3]=12.8573;
    coeff[4]=45.9948;
    coeff[5]=1.1528;
    coeff[6]= -23.07;
    coeff[7]=2.6719;
    coeff[8]=-27.8961;
    coeff[9]=7.1927;
    coeff[10]=31.4788;
    coeff[11]=-0.0434;
    coeff[12]=-25.1661;
    coeff[13]=8.245;
    coeff[14]=-45.2178;
    coeff[15]=4.8476;

  }
  else if(ex<0.65 && ex>=0.55){

    weights[0][0][0] = 0.3112  ;
    weights[1][0][0] = 0.8339 ;
    weights[0][1][0] =0.1616 ;
    weights[1][1][0] =1.0117 ;
    weights[0][0][1] =1.6821 ;
    weights[1][0][1] = 0.0001;
    weights[0][1][1] = 0.0001;
    weights[1][1][1] = 0.7123;

    coeff[0] = 8.0848;
    coeff[1]= -0.1968;
    coeff[2]= -79.9715;
    coeff[3]=12.7318;
    coeff[4]=35.7155;
    coeff[5]=1.68;
    coeff[6]= -13.0365;
    coeff[7]=1.8101;
    coeff[8]=-13.2235;
    coeff[9]=6.3697;
    coeff[10]=25.4548;
    coeff[11]=-0.3947;
    coeff[12]=-10.4478;
    coeff[13]=7.657;
    coeff[14]=-75.9179;
    coeff[15]=6.1791;
  }
  else if(ex<0.75 && ex>=0.65){
    /*    weights[0][0][0] = 0.0021;                                                                                                                                
    weights[1][0][0] = 1.0928;                                                                                                                                      
    weights[0][1][0] = 1.3062;                                                                                                                                      
    weights[1][1][0] = 0.1546;                                                                                                                                      
    weights[0][0][1] = 1.1057;                                                                                                                                      
    weights[1][0][1] = 0.0524;                                                                                                                                      
    weights[0][1][1] = 0.0139;                                                                                                                                      
    weights[1][1][1] = 1.5676;  */

    weights[0][0][0] =0.1219;
    weights[1][0][0] =0.001;
    weights[0][1][0] =0.5084;
    weights[1][1][0] =0.2999;
    weights[0][0][1] =1.2197;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.001;
    weights[1][1][1] =1.3635;

    coeff[0] = 1.9975;
    coeff[1]= 0.418;
    coeff[2]= -76.6932;
    coeff[3]=11.3479;
    coeff[4]=40.7406;
    coeff[5]=1.9511;
    coeff[6]= -2.7761;
    coeff[7]=0.5987;
    coeff[8]=0;
    coeff[9]=0;
    coeff[10]=0;
    coeff[11]=0;
    coeff[12]=41.317;
    coeff[13]=2.1874;
    coeff[14]=-88.8095;
    coeff[15]=11.0003;
  }
  else if(ex<0.85 && ex>=0.75){
    /*    weights[0][0][0] = 0.0094;                                                                                                                                
    weights[1][0][0] = 1.0894;                                                                                                                                      
    weights[0][1][0] = 1.3085;                                                                                                                                      
    weights[1][1][0] = 0.0493;                                                                                                                                      
    weights[0][0][1] = 1.1010;                                                                                                                                      
    weights[1][0][1] = 0.0352;                                                                                                                                      
    weights[0][1][1] = 0.0;                                                                                                                                         
    weights[1][1][1] = 1.5931;  */

    weights[0][0][0] = 0.0462 ;
    weights[1][0][0] = 0.001;
    weights[0][1][0] = 0.4157;
    weights[1][1][0] = 0.1585;
    weights[0][0][1] = 1.3005;
    weights[1][0][1] = 0.001;
    weights[0][1][1] = 0.001;
    weights[1][1][1] = 1.4986;

    coeff[0] = 5.1672;
    coeff[1]= 0.2129;
    coeff[2]= -46.506;
    coeff[3]=11.7213;
    coeff[4]=-5.8873;
    coeff[5]=1.4279;
    coeff[6]= -8.2448;
    coeff[7]=0.3455;
    coeff[8]=15.0254;
    coeff[9]=-0.283;
    coeff[10]=0;
    coeff[11]=0;
    coeff[12]=58.975;
    coeff[13]=0.8131;
    coeff[14]=-108.6828;
    coeff[15]=12.4362;

  }
  else if(ex<0.95 && ex>=0.85){
    /*    weights[0][0][0] = 0.0;                                                                                                                                   
    weights[1][0][0] = 1.0532;                                                                                                                                      
    weights[0][1][0] = 1.3298;                                                                                                                                      
    weights[1][1][0] = 0.0;                                                                                                                                         
    weights[0][0][1] = 1.0209;                                                                                                                                      
    weights[1][0][1] = 0.0110;                                                                                                                                      
    weights[0][1][1] = 0.0023;                                                                                                                                      
    weights[1][1][1] = 1.6299;  */

    weights[0][0][0] =0.001;
    weights[1][0][0] =0.001;
    weights[0][1][0] =0.1342;
    weights[1][1][0] =0.1935;
    weights[0][0][1] =1.5755;
    weights[1][0][1] =0.001;
    weights[0][1][1] =0.001;
    weights[1][1][1] =1.5297;

    coeff[0] = -0.8151;
    coeff[1]= 0.1621;
    coeff[2]= -61.9333;
    coeff[3]=12.5014;
    coeff[4]=0.0358;
    coeff[5]=-0.0006;
    coeff[6]= 0;
    coeff[7]=0;
    coeff[8]=22.0291;
    coeff[9]=-0.4022;
    coeff[10]=0;
    coeff[11]=0;
    coeff[12]=56.0043;
    coeff[13]=0.7978;
    coeff[14]=-116.9175;
    coeff[15]=13.0244;
  }
  else if(ex<0.01){/*                                                                                                                                               
    weights[0][0][0] = 1.0655;                                                                                                                                      
    weights[1][0][0] = 1.0122;                                                                                                                                      
    weights[0][1][0] = 1.1027;                                                                                                                                      
    weights[1][1][0] = 1.0276;                                                                                                                                      
    weights[0][0][1] = 1.0587;                                                                                                                                      
    weights[1][0][1] = 1.0140;                                                                                                                                      
    weights[0][1][1] = 1.0997;                                                                                                                                      
    weights[1][1][1] = 1.0301;  */
    weights[0][0][0] = 0.8867;                                                                                                                                     \

    weights[1][0][0] = 1.0440  ;                                                                                                                                   \

    weights[0][1][0] =0.0423  ;                                                                                                                                    \

    weights[1][1][0] =0.8110 ;                                                                                                                                     \

    weights[0][0][1] =1.7275   ;                                                                                                                                   \

    weights[1][0][1] =0.5615 ;                                                                                                                                     \

    weights[0][1][1] =0.8323 ;                                                                                                                                     \

    weights[1][1][1] =0.4641 ;


    coeff[0] = -27.5089;
    coeff[1]= 7.4177;
    coeff[2]=-82.8803;
    coeff[3]=13.1952;
    coeff[4]=72.0312;
    coeff[5]=0.5298;
    coeff[6]=-34.1779;
    coeff[7]=6.0293;
    coeff[8]=-52.2607;
    coeff[9]=8.1754;
    coeff[10]=-1.6270;
    coeff[11]=4.6858;
    coeff[12]=-27.7770;
    coeff[13]=6.2852;
    coeff[14]=14.6295;
    coeff[15]=3.8839;



  }
  /*                                                                                                                                                                  
  int rank=0;                                                                                                                                                       
       if(!rank) printf("weights: %e,%e; %e,%e; %e,%e; %e,%e\n",                                                                                                    
          weights[0][0][0],weights[1][0][0],weights[0][1][0],weights[1][1][0],                                                                                      
          weights[0][0][1],weights[1][0][1],weights[0][1][1],weights[1][1][1]);                                                                                     
  */
  // The following is done in inimesh in the CPU code.                                                                                                              
  double facex = pow(fac,ex);
  for(i=0;i<2;i++)
    for(j=0;j<2;j++)
      for(k=0;k<2;k++)
        weights[i][j][k] = weights[i][j][k]/facex;

  return;
}

//if(EX<0.65 && EX>=0.01) *taumin = 1./(2*pi*400.0)*0.1*FAC;                                                                                                        
// else if(EX<0.85 && EX>=0.65) *taumin = 1./(2*pi*320.0)*0.1*FAC;                                                                                                  


void init_texture(int nxt,  int nyt,  int nzt,  Grid3D tau1,  Grid3D tau2,  Grid3D vx1,  Grid3D vx2,
                  Grid3D weights, Grid3Dww ww,Grid3D wwo,
                  int xls,  int xre,  int yls,  int yre)
{
  int i, j, k, itx, ity, itz;
  itx = 0;
  ity = 0;
  itz = (nzt-1)%2;
  for(i=xls;i<=xre;i++)
    {
      itx = 1 - itx;
      for(j=yls;j<=yre;j++)
	{
          ity = 1 - ity;
          for(k=align;k<nzt+align;k++)
	    {
	      itz           = 1 - itz;
	      vx1[i][j][k]  = tau1[itx][ity][itz];
	      vx2[i][j][k]  = tau2[itx][ity][itz];
	      wwo[i][j][k]   = 8.0*weights[itx][ity][itz];
	      if (itx<0.5 && ity<0.5 && itz<0.5) ww[i][j][k]   = 1;
	      else if (itx<0.5 && ity<0.5 && itz>0.5) ww[i][j][k]   = 2;
	      else if(itx<0.5 && ity>0.5 && itz<0.5) ww[i][j][k]   = 3;
	      else if(itx<0.5 && ity>0.5 && itz>0.5) ww[i][j][k]   = 4;
	      else if(itx>0.5 && ity<0.5 && itz<0.5) ww[i][j][k]   = 5;
	      else if(itx>0.5 && ity<0.5 && itz>0.5) ww[i][j][k]   = 6;
	      else if(itx>0.5 && ity>0.5 && itz<0.5) ww[i][j][k]   = 7;
	      else if(itx>0.5 && ity>0.5 && itz>0.5) ww[i][j][k]   = 8;
	      //               printf("%g %g\n",ww[i][j][k],ww[i][j][k]);                                                                                         

	    }
	}
    }
  return;
}

void hoek_brown(float sigma_0, float sigma_ci, float GSI, float mi, float D, 
    int tunnel, float *phi, float *cohes){


   double mb, s, a, sigma_cm, sigma_3max, sigma_3n, tmp;
   double phi2, cohes2;

   mb=(double) mi * exp(((double) GSI - 100.)/(28. - 14.*D));
   s=exp(((double)GSI - 100.)/(9. - 3.*(double) D));
   a=1./2. + 1./6.*(exp(-(double)GSI /15.) - exp(-20./3.));
  
   sigma_cm = (double) sigma_ci * (mb + 4.*s - a*(mb - 8.*s))*
      pow(mb/4.+s, a-1.) / (2.*(1.+a)*(2.+a)) ;

   if (tunnel == 0) {
      sigma_3max=sigma_cm * 0.72 * pow(sigma_cm/sigma_0, -0.91);
      //fprintf(stdout, "ratio=%f\n", sigma_cm/sigma_0);
      }
   else {
      sigma_3max=sigma_cm * 0.47 * pow(sigma_cm/sigma_0,-0.94);
   }

   sigma_3n=sigma_3max / sigma_ci;
   tmp=6.*a*mb*pow(s+mb*sigma_3n, a-1.) / 
      (2.*(1.+a)*(2.+a) + 6*a*mb*pow(s+mb*sigma_3n,a-1.));

   phi2=asin(tmp)/M_PI * 180.;

   cohes2=(sigma_ci*((1.+2*a)*s + 
      (1.-a)*mb*sigma_3n)*pow(s+mb*sigma_3n,a-1.)) / 
      ((1.+a)*(2.+a)* sqrt(1.+(6*a*mb*pow(s+mb*sigma_3n,a-1.))
      /((1.+a)*(2.+a))));

   *phi=(float)  phi2;
   *cohes=(float) cohes2;

}

// for SAF dynamic rupture simulations
void inidrpr_hoekbrown_light(int nxt, int nyt, int nzt, int nve, int *coords,
    float dh, int rank, 
    Grid3D mu, Grid3D lam, Grid3D d1, 
    Grid3D sigma2,
    Grid3D cohes, Grid3D phi, 
    float *fmajor, float *fminor, float *strike, float *dip, MPI_Comm MCW, int d_i){

  int i,j,k;
  // free surface index
  int sInd = nzt+align-1;
  float depth, tvp, itime;
  //float strike[3], dip[3], *ssp;

  float sigma_ci, GSI, mi, D=0;
  int sigma_ci_type, tunnel=1;
  float gsi100, GSI_d;
  int fltpos;
  /*float gwt;*/
  int ypos;
  float fltdist;
  float GSI_wallrock, GSI_core;

  FILE *fid;
  char *nline=NULL;
  size_t linecap;
  float tvs;
 
  /* the intermediate principal stress is always assumed vertical*/
  strike[1] = 0.;

  dip[0] = 0.;
  dip[1] = 90.;
  dip[2] = 0.;

  /* read principal stress directions and hoek-brown parameters
     from provided input file */

  if (rank == 0){

     fid=fopen("nonlinear.dat", "r");
     if (fid == NULL) {
        perror("could not open nonlinear.dat");
        MPI_Finalize();
     }
     getline(&nline, &linecap, fid);
     sscanf(nline, "%f %f", strike, strike+2);
 
     getline(&nline, &linecap, fid);
     sscanf(nline, "%f %f\n", fmajor, fminor);
 
     /*unconfined compressive strength parameter 
     (0 = constant, 3 or 12 using eq. 3 or 12 in Chang, 2006, respectively */
     getline(&nline, &linecap, fid);
     sscanf(nline, "%d %f\n", &sigma_ci_type, &sigma_ci);
 
     getline(&nline, &linecap, fid);
     sscanf(nline, "%f\n", &GSI); /* Geological strength index*/
 
     getline(&nline, &linecap, fid);
     sscanf(nline, "%f\n", &mi); /* Hoek-Brown parameter */

     getline(&nline, &linecap, fid);
     sscanf(nline, "%f\n", &gsi100); /* depth where GSI reaches 100 */

     getline(&nline, &linecap, fid);
     sscanf(nline, "%d\n", &fltpos); /* fault position along y*/

     /*getline(&nline, &linecap, fid);
     sscanf(nline, "%f\n", &gwt);*/ /* ground water table y*/
 
     fprintf(stdout, "read nonlinear.dat: strike[0]=%f, strike[2]=%f, fmajor=%f, fminor=%f\n",
          strike[0], strike[2], (*fmajor), (*fminor));
     if (sigma_ci_type == 0) 
        fprintf(stdout, "using constant uniaxial compressive strength of %f\n", sigma_ci);
     else if ((sigma_ci_type == 3) || (sigma_ci_type == 12))
        fprintf(stdout, "using eq. %d in Chang et al. (2006) for sigma_ci\n", sigma_ci_type);
     else {
        fprintf(stdout, "Error. sigma_ci_type must be 0, 3 or 12.\n");
        MPI_Finalize();
     }
     fprintf(stdout, "fault position: %d\n", fltpos);
     /*fprintf(stdout, "ground water table: %d\n", gwt);*/
  }

  MPI_Bcast(strike, 3, MPI_REAL, 0, MCW);
  MPI_Bcast(dip, 3, MPI_REAL, 0, MCW);
  MPI_Bcast(fmajor, 1, MPI_REAL, 0, MCW);
  MPI_Bcast(fminor, 1, MPI_REAL, 0, MCW);
  MPI_Bcast(&sigma_ci_type, 1, MPI_INT, 0, MCW);
  MPI_Bcast(&sigma_ci, 1, MPI_REAL, 0, MCW);
  MPI_Bcast(&GSI, 1, MPI_REAL, 0, MCW);
  MPI_Bcast(&mi, 1, MPI_REAL, 0, MCW);
  MPI_Bcast(&gsi100, 1, MPI_REAL, 0, MCW);
  MPI_Bcast(&fltpos, 1, MPI_INT, 0, MCW);
  /*MPI_Bcast(&gwt, 1, MPI_FLOAT, 0, MCW);*/

  for(j=2;j<nyt+2+ngsl2;j++)
    for(i=2;i<nxt+2+ngsl2;i++){
      sigma2[i][j][sInd]   = -dh/2.*d1[i][j][sInd]*9.81;
      sigma2[i][j][sInd+1] = sigma2[i][j][sInd];
      for(k=sInd-1;k>=align-2;k--){
        sigma2[i][j][k] = sigma2[i][j][k+1] - dh*d1[i][j][k]*9.81;
        //fprintf(stdout, "sigma2[%d][%d][%d] = %e\n", i, j, k, sigma2[i][j][k]);
      }
    }

  for(k=align;k<=(sInd+1);k++)
    for(j=2;j<nyt+2+ngsl2;j++)
      for(i=2;i<nxt+2+ngsl2;i++){
        depth = 0.5*dh + dh*(sInd - k);
        /* moved to kernel function */
        /*pfluid[i][j][k] = 9810.*depth;

        // compute stress field from principal stresses 
        sigma1 = (sigma2[i][j][k] + pfluid[i][j][k]) * fmajor  - pfluid[i][j][k];
        sigma3 = (sigma2[i][j][k] + pfluid[i][j][k]) * fminor - pfluid[i][j][k];

        ssp = rotate_principal(sigma1+pfluid[i][j][k], sigma2[i][j][k]+pfluid[i][j][k],
                 sigma3+pfluid[i][j][k], strike, dip);

        inixx[i][j][k] = ssp[0] - pfluid[i][j][k];
        iniyy[i][j][k] = ssp[4] - pfluid[i][j][k];
        inizz[i][j][k] = ssp[8] - pfluid[i][j][k];
        inixy[i][j][k] = ssp[1]; // = ssp[3] 
        iniyz[i][j][k] = ssp[5]; // = ssp[7] 
        inixz[i][j][k] = ssp[2]; // = ssp[6] */

        /* taper GSI from surface value to 100 at specified depth*/
        GSI_wallrock = GSI + (100. - GSI) * (depth - 0.5*dh)/gsi100;
        if (GSI_wallrock > 100) GSI_wallrock = 100;

        if (depth < 4000.) {
           GSI_core = GSI;
        }
        else if ((depth>=4000.) && (depth <=6000.)) {
           GSI_core = GSI + (depth - 4000.)/2000.*(GSI_wallrock - GSI);
        }
        else {
           GSI_core = GSI_wallrock;
        }

        ypos = coords[1] * nyt + j - 2 - ngsl;
        fltdist = fabsf((float) ypos + 0.5 - (float) fltpos) * dh;
        if (fltdist < 225.) {
          GSI_d = GSI_core;
        }
        else if ((fltdist >= 225.) && (fltdist <= 750.))
          GSI_d = GSI_core + (GSI_wallrock - GSI_core)*(fltdist - 225.)/(750.-225.);
        else{
          GSI_d = GSI_wallrock;
        }

        /* compute sigma_ci using equation if requested */
        if (sigma_ci_type == 3) {
           tvp = sqrt((1./lam[i][j][k] + 2./mu[i][j][k])/d1[i][j][k]);
           itime = 1./ tvp / 3.28 * 1.e6; //from m/s to usec/ft
           sigma_ci=1.4138e7 * powf(itime, -3.) * 1.e6; //Chang 2006, eq. 3 in Pa
        }
        else if (sigma_ci_type == 12) {
           tvp = sqrt((1./lam[i][j][k] + 2./mu[i][j][k])/d1[i][j][k]) / 1.e3; /* need km/s*/
           sigma_ci=0.77*pow(tvp, 2.93) * 1.e6; //eq. 12 in Chang 2006, from Horsrud 2001
        }

        /* according to Hoek et al. (2002), this should be 
        changed to the horizontal stress if larger than the vertical stress */
        hoek_brown(-sigma2[i][j][k], sigma_ci, GSI_d, mi, D, tunnel, 
             &phi[i][j][k], &cohes[i][j][k]);

        tvs=sqrtf(1./(mu[i][j][k] * d1[i][j][k]));
        if (tvs > 2000.) cohes[i][j][k] = 1.e12;
        else if (tvs <  750.) {
           cohes[i][j][k] = 0.f;
           phi[i][j][k] = 35.f;
        }
      }

   FILE *dfid;
   char ofname[200];

   sprintf(ofname, "debug/debug-%1d.%04d_%04d", d_i, coords[0], coords[1]);
   dfid=fopen(ofname, "w");
   for (k=align;k<=sInd;k++)
      fprintf(dfid, "%d %e %e %e\n", k, sigma2[19+2+ngsl][19+2+ngsl][k], 
         phi[19+2+ngsl][19+2+ngsl][k], cohes[19+2+ngsl][19+2+ngsl][k]);
   fclose(dfid);
   

return;
}


/* computes rotation matrix used for stress field computation later*/
void rotation_matrix(float *strike, float *dip, float *Rz, float *RzT){
   int k;
   float alpha[3], beta[3];

   for (k=0; k<3; k++){
      alpha[k] = strike[k] / 180. * M_PI;
      beta[k] = dip[k] / 180. * M_PI;
   }

   Rz[0] = cosf(alpha[0]) * cosf(beta[0]);
   Rz[1] = sinf(alpha[0]) * cosf(beta[0]);
   Rz[2] = sinf(beta[0]);

   Rz[3] = cosf(alpha[1]) * cosf(beta[1]);
   Rz[4] = sinf(alpha[1]) * cosf(beta[1]);
   Rz[5] = sinf(beta[1]);

   Rz[6] = cosf(alpha[2]) * cosf(beta[2]);
   Rz[7] = sinf(alpha[2]) * cosf(beta[2]);
   Rz[8] = sinf(beta[2]);

   RzT[0] = Rz[0];
   RzT[1] = Rz[3];
   RzT[2] = Rz[6];
   RzT[3] = Rz[1];
   RzT[4] = Rz[4];
   RzT[5] = Rz[7];
   RzT[6] = Rz[2];
   RzT[7] = Rz[5];
   RzT[8] = Rz[8];

}

/* subroutine to check consistency between fine and coarse mesh */
int checkmesh(int nxtl, int nytl, int nztl, int nxth, int nyth, int nzth, Grid3D varl, Grid3D varh,
    int pl, int ph, char *varname){
   int k, i, j;
   int kh, ih, jh;
   int corrected=0;
   int kpos[3], kposh[3], l;

   kpos[0] = align + nztl - 3;
   kposh[0] = align + 1;

   kpos[1] = align + nztl - 2;
   kposh[1] = align + 4;

   kpos[2] = align + nztl - 1;
   kposh[2] = align + 7;

   for (l=0; l<3; l++){
      k=kpos[l];
      kh=kposh[l];
      for (j=2+ngsl; j<nytl+2+ngsl;j++){
	 for (i=2+ngsl; i<nxtl+2+ngsl;i++){
	    ih = (i-1-1-ngsl)*3+2+ngsl;
	    jh = (j-1-1-ngsl)*3+3+ngsl;
	    if ( varl[i][j][k] != varh[ih][jh][kh]) {
	       /*fprintf(stderr, "inconsistent values for %s between grid %d (%d,%d,%d) and %d @ (%d,%d,%d): %e != %e\n",
		  varname, ph, ih, jh, kh, pl, i, j, k, varl[i][j][k], varh[ih][jh][kh]);*/
               varl[i][j][k] = varh[ih][jh][kh];
	       corrected=1;
	    }
            /*else {
	       fprintf(stderr, "consistent values for %s between grid %d (%d,%d,%d) and %d @ (%d,%d,%d): %e == %e\n",
		  varname, ph, ih, jh, kh, pl, i, j, k, varl[i][j][k], varh[ih][jh][kh]);
            }*/
	       
	 }
      }
   }

   return(corrected);
}

int checkmesh_ww(int nxtl, int nytl, int nztl, int nxth, int nyth, int nzth, Grid3Dww varl, Grid3Dww varh,
    int pl, int ph, char *varname){
   int k, i, j;
   int kh, ih, jh;
   int corrected=0;
   int kpos[3], kposh[3], l;

   kpos[0] = align + nztl - 3;
   kposh[0] = align + 1;

   kpos[1] = align + nztl - 2;
   kposh[1] = align + 4;

   kpos[2] = align + nztl - 1;
   kposh[2] = align + 7;

   for (l=0; l<3; l++){
      k=kpos[l];
      kh=kposh[l];
      for (j=2+ngsl; j<nytl+2+ngsl;j++){
	 for (i=2+ngsl; i<nxtl+2+ngsl;i++){
	    ih = (i-1-1-ngsl)*3+2+ngsl;
	    jh = (j-1-1-ngsl)*3+3+ngsl;
	    if ( varl[i][j][k] != varh[ih][jh][kh]) {
               varl[i][j][k] = varh[ih][jh][kh];
	       corrected=1;
	    }
	 }
      }
   }

   return(corrected);
}
