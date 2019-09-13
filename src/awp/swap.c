#include <stdio.h>
#include <awp/pmcl3d.h>
#include <awp/kernel_launch.h>
#include <awp/swap.h>
#define MPIRANKX        100000
#define MPIRANKY         50000
#define MPIRANKYLDFAC   200000

void mediaswap(Grid3D d1, Grid3D mu,     Grid3D lam,    Grid3D qp,     Grid3D qs, 
               int rank,  int x_rank_L,  int x_rank_R,  int y_rank_F,  int y_rank_B,
               int nxt,   int nyt,       int nzt,       MPI_Comm MCW, int p)
{
	int i, j, k, idx, idy, idz;
	int media_count_x, media_count_y;
	int media_size_x, media_size_y;
        MPI_Request  request_x[4], request_y[4];
        MPI_Status   status_x[4],  status_y[4];
	Grid1D mediaL_S=NULL, mediaR_S=NULL, mediaF_S=NULL, mediaB_S=NULL;
        Grid1D mediaL_R=NULL, mediaR_R=NULL, mediaF_R=NULL, mediaB_R=NULL;

	if(x_rank_L<0 && x_rank_R<0 && y_rank_F<0 && y_rank_B<0)
		return;
	
	if(y_rank_F>=0 || y_rank_B>=0)
	{
		mediaF_S      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
		mediaB_S      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
                mediaF_R      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
                mediaB_R      = Alloc1D(5*ngsl*(nxt+2)*(nzt+2));
                media_size_y  = 5*(ngsl)*(nxt+2)*(nzt+2);
		media_count_y = 0;

                PostRecvMsg_Y(mediaF_R, mediaB_R, MCW, request_y, &media_count_y, media_size_y, y_rank_F, y_rank_B, p);
		
		if(y_rank_F>=0)
		{
	        	for(i=1+ngsl;i<nxt+3+ngsl;i++)
        	  	  for(j=2+ngsl;j<2+ngsl2;j++)
            	    	    for(k=align-1;k<nzt+align+1;k++)
			    {
            			idx = i-1-ngsl;
            			idy = (j-2-ngsl)*5;
	            		idz = k-align+1;
        	    		mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
            			idy++;
            			mediaF_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];            
            	    	    }
		}

		if(y_rank_B>=0)
		{
        		for(i=1+ngsl;i<nxt+3+ngsl;i++)
          	  	  for(j=nyt+2;j<nyt+2+ngsl;j++)
            	    	    for(k=align-1;k<nzt+align+1;k++)
			    {
                		idx = i-1-ngsl;
	                	idy = (j-nyt-2)*5;
        	        	idz = k-align+1;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                		idy++;
                		mediaB_S[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
            	    	     }
		}

       		PostSendMsg_Y(mediaF_S, mediaB_S, MCW, request_y, &media_count_y, media_size_y, y_rank_F, y_rank_B, rank, 
                   Both, p);
                MPICHK(MPI_Waitall(media_count_y, request_y, status_y));

		if(y_rank_F>=0)
		{
                	for(i=1+ngsl;i<nxt+3+ngsl;i++)
                  	  for(j=2;j<2+ngsl;j++)
                    	    for(k=align-1;k<nzt+align+1;k++)
		    	    {
                        	idx = i-1-ngsl;
                        	idy = (j-2)*5;
                        	idz = k-align+1;
                        	d1[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	mu[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	lam[i][j][k] = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	qp[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                        	idy++;
                        	qs[i][j][k]  = mediaF_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                    	    }
		}
		
		if(y_rank_B>=0)
		{
                        for(i=1+ngsl;i<nxt+3+ngsl;i++)
                          for(j=nyt+2+ngsl;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {   
                                idx = i-1-ngsl;
                                idy = (j-nyt-2-ngsl)*5;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                mu[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                lam[i][j][k] = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                qp[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                                idy++;
                                qs[i][j][k]  = mediaB_R[idx*5*(ngsl)*(nzt+2)+idy*(nzt+2)+idz];
                            }
		}
		
		Delloc1D(mediaF_S);
		Delloc1D(mediaB_S);
		Delloc1D(mediaF_R);
		Delloc1D(mediaB_R);		
	}

	if(x_rank_L>=0 || x_rank_R>=0)
	{
                mediaL_S      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                mediaR_S      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                mediaL_R      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                mediaR_R      = Alloc1D(5*ngsl*(nyt+ngsl2)*(nzt+2));
                media_size_x  = 5*(ngsl)*(nyt+ngsl2)*(nzt+2);
                media_count_x = 0;

		PostRecvMsg_X(mediaL_R, mediaR_R, MCW, request_x, &media_count_x, media_size_x, x_rank_L, x_rank_R, p);
                if(x_rank_L>=0)
                {
                        for(i=2+ngsl;i<2+ngsl2;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-2-ngsl)*5;
                                idy = j-2;
                                idz = k-align+1;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                                idx++;
                                mediaL_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
                            }
                }

                if(x_rank_R>=0)
                {
                        for(i=nxt+2;i<nxt+2+ngsl;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-nxt-2)*5;
                                idy = j-2;
                                idz = k-align+1;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = d1[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = mu[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = lam[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qp[i][j][k];
                                idx++;
                                mediaR_S[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz] = qs[i][j][k];
                            }
                }

	        PostSendMsg_X(mediaL_S, mediaR_S, MCW, request_x, &media_count_x, media_size_x, x_rank_L, x_rank_R, rank, 
                   Both, p);
         	MPICHK(MPI_Waitall(media_count_x, request_x, status_x));

                if(x_rank_L>=0)
                {
                        for(i=2;i<2+ngsl;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-2)*5;
                                idy = j-2;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                mu[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                lam[i][j][k] = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qp[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qs[i][j][k]  = mediaL_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                            }
                }

                if(x_rank_R>=0)
                {
                        for(i=nxt+2+ngsl;i<nxt+2+ngsl2;i++)
                          for(j=2;j<nyt+2+ngsl2;j++)
                            for(k=align-1;k<nzt+align+1;k++)
                            {
                                idx = (i-nxt-2-ngsl)*5;
                                idy = j-2;
                                idz = k-align+1;
                                d1[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                mu[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                lam[i][j][k] = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qp[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                                idx++;
                                qs[i][j][k]  = mediaR_R[idx*(nyt+ngsl2)*(nzt+2)+idy*(nzt+2)+idz];
                            }
                }

                Delloc1D(mediaL_S);
                Delloc1D(mediaR_S);
                Delloc1D(mediaL_R);
                Delloc1D(mediaR_R);
	}

	return;
}

void PostRecvMsg_X(_prec*  RL_M, _prec*  RR_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size, int rank_L, 
   int rank_R, int gridnum)
{
	int temp_count = 0;
       
	if(rank_L>=0){
		MPICHK(MPI_Irecv(RL_M, msg_size, _mpi_prec, rank_L, MPIRANKX+rank_L*gridnum, MCW, &request[temp_count]));
		++temp_count;
	}

	if(rank_R>=0){
		MPICHK(MPI_Irecv(RR_M, msg_size, _mpi_prec, rank_R, MPIRANKX+rank_R*gridnum, MCW, &request[temp_count]));
		++temp_count;
	}

	*count = temp_count;
	return;
}

void PostSendMsg_X(_prec*  SL_M, _prec*  SR_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size,
                   int rank_L,  int rank_R,  int rank, int flag, int gridnum)
{
        int temp_count, flag_L=-1, flag_R=-1;
        temp_count = *count;
        if(rank<0)
                return;

        if(flag==Both || flag==Left)  flag_L=1;
        if(flag==Both || flag==Right) flag_R=1;

        if(rank_L>=0 && flag_L==1){
                MPICHK(MPI_Isend(SL_M, msg_size, _mpi_prec, rank_L, MPIRANKX+rank*gridnum, MCW, &request[temp_count]));
                ++temp_count;
        }

        if(rank_R>=0 && flag_R==1){
                MPICHK(MPI_Isend(SR_M, msg_size, _mpi_prec, rank_R, MPIRANKX+rank*gridnum, MCW, &request[temp_count]));
                ++temp_count;
        }

        *count = temp_count;
        return;
}

void PostRecvMsg_Y(_prec*  RF_M, _prec*  RB_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size, int rank_F, int rank_B, int gridnum)
{
        int temp_count = 0;

        if(rank_F>=0){
                MPICHK(MPI_Irecv(RF_M, msg_size, _mpi_prec, rank_F, MPIRANKY+rank_F*gridnum, MCW, &request[temp_count]));
                ++temp_count;
        }

        if(rank_B>=0){
                MPICHK(MPI_Irecv(RB_M, msg_size, _mpi_prec, rank_B, MPIRANKY+rank_B*gridnum, MCW, &request[temp_count]));
                ++temp_count;
        }

        *count = temp_count;
        return;
}

void PostSendMsg_Y(_prec*  SF_M, _prec*  SB_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size,
                   int rank_F,  int rank_B,  int rank, int flag, int gridnum)
{
        int temp_count, flag_F=-1, flag_B=-1;
        temp_count = *count;
        if(rank<0)
                return;

        if(flag==Both || flag==Front)  flag_F=1;
        if(flag==Both || flag==Back)   flag_B=1;

        if(rank_F>=0 && flag_F==1){
                MPICHK(MPI_Isend(SF_M, msg_size, _mpi_prec, rank_F, MPIRANKY+rank*gridnum, MCW, &request[temp_count]));
                ++temp_count;
        }

        if(rank_B>=0 && flag_B==1){
                MPICHK(MPI_Isend(SB_M, msg_size, _mpi_prec, rank_B, MPIRANKY+rank*gridnum, MCW, &request[temp_count]));
                ++temp_count;
        }

        *count = temp_count;
        return;
}

void Cpy2Device_source(int npsrc, int READ_STEP,
      int index_offset,
      Grid1D taxx, Grid1D tayy, Grid1D tazz,
      Grid1D taxz, Grid1D tayz, Grid1D taxy,
      _prec *d_taxx, _prec *d_tayy, _prec *d_tazz,
      _prec *d_taxz, _prec *d_tayz, _prec *d_taxy,
      int IFAULT){

      /* this function works only if READ_STEP_GPU = READ_STEP when IFAULT < 4 */
      /* if IFAULT == 4, order of data in taxx,tayy and tazz is different 
         (npsrc increasing faster), and it works if index_offset is set properly */
      /* Daniel Roten (modified for fault boundary condition */
 
      fprintf(stdout, "inside Cpy2Device_source: index_offset = %d, npsrc=%d\n", index_offset, npsrc);

      if (IFAULT == 4){
         index_offset *= npsrc;
         // add npsrc to index, as first entry in array conains last step from previously
         // read read_wstep source points
         index_offset += npsrc;
      }

      long int num_bytes;
      cudaError_t cerr;
       num_bytes = sizeof(_prec)*npsrc*READ_STEP;
       cerr=cudaMemcpy(d_taxx+npsrc,taxx+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_tayy+npsrc,tayy+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       cerr=cudaMemcpy(d_tazz+npsrc,tazz+index_offset,num_bytes,cudaMemcpyHostToDevice);
       if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       if (IFAULT != 4){
          cerr=cudaMemcpy(d_taxz+npsrc,taxz+index_offset,num_bytes,cudaMemcpyHostToDevice);
	  if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
	  cerr=cudaMemcpy(d_tayz+npsrc,tayz+index_offset,num_bytes,cudaMemcpyHostToDevice);
	  if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
	  cerr=cudaMemcpy(d_taxy+npsrc,taxy+index_offset,num_bytes,cudaMemcpyHostToDevice);
	  if(cerr!=cudaSuccess) printf("CUDA ERROR: Cpy2Device: %s\n",cudaGetErrorString(cerr));
       }
       fprintf(stdout, "taxx[%ld],yy,zz=%f %f %f\n", index_offset+num_bytes/4-4,
          taxx[index_offset+num_bytes/4-1],
          tayy[index_offset+num_bytes/4-1],
          tazz[index_offset+num_bytes/4-1]);

return;
}

void Cpy2Host_VX(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  h_m, int nxt, int nyt, int nzt, cudaStream_t St, int rank, int flag)
{
	int d_offset=0, h_offset=0, msg_size=0;
        if(rank<0 || flag<1 || flag>2)
	        return;

	if(flag==Left)	d_offset = (2+ngsl)*(nyt+4+ngsl2)*(nzt+2*align); 
	if(flag==Right)	d_offset = (nxt+2)*(nyt+4+ngsl2)*(nzt+2*align);
	
        h_offset = (ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(_prec)*(ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
        CUCHK(cudaMemcpyAsync(h_m,            u1+d_offset, msg_size, cudaMemcpyDeviceToHost, St));
        CUCHK(cudaMemcpyAsync(h_m+h_offset,   v1+d_offset, msg_size, cudaMemcpyDeviceToHost, St));
        CUCHK(cudaMemcpyAsync(h_m+h_offset*2, w1+d_offset, msg_size, cudaMemcpyDeviceToHost, St));
	return;
}

void Cpy2Host_VY(_prec*  s_u1, _prec*  s_v1, _prec*  s_w1, _prec*  h_m, int nxt, int nzt, cudaStream_t St, int rank)
{
        int h_offset, msg_size;
        if(rank<0)
                return;

        h_offset = (ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(_prec)*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        CUCHK(cudaGetLastError());
        CUCHK(cudaMemcpyAsync(h_m,            s_u1, msg_size, cudaMemcpyDeviceToHost, St));
        CUCHK(cudaMemcpyAsync(h_m+h_offset,   s_v1, msg_size, cudaMemcpyDeviceToHost, St));
        CUCHK(cudaMemcpyAsync(h_m+h_offset*2, s_w1, msg_size, cudaMemcpyDeviceToHost, St));
        CUCHK(cudaGetLastError());
        return;
}

void Cpy2Device_VX(_prec*  u1, _prec*  v1, _prec*  w1,        _prec*  L_m,       _prec*  R_m, int nxt,
                   int nyt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R)
{
        int d_offset, h_offset, msg_size;

        h_offset = (ngsl)*(nyt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(_prec)*(ngsl)*(nyt+4+ngsl2)*(nzt+2*align);

        if(rank_L>=0){
		d_offset = 2*(nyt+4+ngsl2)*(nzt+2*align);
                CUCHK(cudaMemcpyAsync(u1+d_offset, L_m,            msg_size, cudaMemcpyHostToDevice, St1));
                CUCHK(cudaMemcpyAsync(v1+d_offset, L_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St1));
                CUCHK(cudaMemcpyAsync(w1+d_offset, L_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St1));
	}

        if(rank_R>=0){
		d_offset = (nxt+ngsl+2)*(nyt+4+ngsl2)*(nzt+2*align);
        	CUCHK(cudaMemcpyAsync(u1+d_offset, R_m,            msg_size, cudaMemcpyHostToDevice, St2));
        	CUCHK(cudaMemcpyAsync(v1+d_offset, R_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St2));
        	CUCHK(cudaMemcpyAsync(w1+d_offset, R_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St2));
	}
        return;
}

void Cpy2Device_VY(_prec*  u1,   _prec *v1,  _prec *w1,  _prec*  f_u1, _prec*  f_v1, _prec*  f_w1, _prec*  b_u1,      _prec*  b_v1, 
                   _prec*  b_w1, _prec*  F_m, _prec*  B_m, int nxt,     int nyt,     int nzt,     cudaStream_t St1, cudaStream_t St2, 
                   int rank_F,  int rank_B, int d_i)
{
        int h_offset, msg_size;

        h_offset = (ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        msg_size = sizeof(_prec)*(ngsl)*(nxt+4+ngsl2)*(nzt+2*align);
        CUCHK(cudaGetLastError());
        if(rank_F>=0){
                CUCHK(cudaMemcpyAsync(f_u1, F_m,            msg_size, cudaMemcpyHostToDevice, St1));
                CUCHK(cudaMemcpyAsync(f_v1, F_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St1));
                CUCHK(cudaMemcpyAsync(f_w1, F_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St1));
        }

        if(rank_B>=0){
                CUCHK(cudaMemcpyAsync(b_u1, B_m,            msg_size, cudaMemcpyHostToDevice, St2));
                CUCHK(cudaMemcpyAsync(b_v1, B_m+h_offset,   msg_size, cudaMemcpyHostToDevice, St2));
                CUCHK(cudaMemcpyAsync(b_w1, B_m+h_offset*2, msg_size, cudaMemcpyHostToDevice, St2));
        }

        CUCHK(cudaGetLastError());
        update_bound_y_H(u1, v1, w1, f_u1, f_v1, f_w1, b_u1, b_v1, b_w1, nxt, nzt, St1, St2, rank_F, rank_B, d_i);
        CUCHK(cudaGetLastError());
        return;
}

void Cpy2Host_yldfac_X(_prec*  yldfac, _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyt, int nzt,
   cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int d_i){
   long int msg_size;

   //update_yldfac_buffer_x_H(yldfac, d_buf_L, d_buf_R, nyt, nzt, St1, St2, rank_L, rank_R, d_i);

   msg_size = (long int) ngsl*((long int) nyt+ngsl2)*(nzt)*sizeof(_prec);
   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/

   if (rank_L >= 0) CUCHK(cudaMemcpyAsync(buf_L, d_buf_L, msg_size, cudaMemcpyDeviceToHost, St1));

   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/
   if (rank_R >= 0) CUCHK(cudaMemcpyAsync(buf_R, d_buf_R, msg_size, cudaMemcpyDeviceToHost, St2));
   return;

}

void Cpy2Device_yldfac_X(_prec*  yldfac, _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyt, int nzt,
    cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int d_i){
   long int msg_size;

   msg_size = (long int) ngsl*((long int) nyt+ngsl2)*(nzt)*sizeof(_prec);

   if (rank_L >= 0) CUCHK(cudaMemcpyAsync(d_buf_L, buf_L, msg_size, cudaMemcpyHostToDevice, St1));

   if (rank_R >= 0) CUCHK(cudaMemcpyAsync(d_buf_R, buf_R, msg_size, cudaMemcpyHostToDevice, St2));

   update_yldfac_data_x_H(yldfac, d_buf_L, d_buf_R, nyt, nzt, St1, St2, rank_L, rank_R, d_i);
   return;
}

void Cpy2Host_yldfac_Y(_prec*  yldfac, _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxt, int nzt, 
    cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i){
   long int msg_size;

   //update_yldfac_buffer_y_H(yldfac, d_buf_F, d_buf_B, nxt, nzt, St1, St2, rank_F, rank_B, d_i);

   msg_size = (long int) nxt*ngsl*nzt*sizeof(_prec);

   if (rank_F >= 0) CUCHK(cudaMemcpyAsync(buf_F, d_buf_F, msg_size, cudaMemcpyDeviceToHost, St1));

   if (rank_B >= 0) CUCHK(cudaMemcpyAsync(buf_B, d_buf_B, msg_size, cudaMemcpyDeviceToHost, St2));
   return;

}

void Cpy2Device_yldfac_Y(_prec*  yldfac, _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxt, int nzt,
    cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int d_i){
   long int msg_size;

   msg_size =  (long int) nxt*ngsl*nzt*sizeof(_prec);
   //fprintf(stdout, "msg_size=%d\n", msg_size);

   if (rank_F >= 0) CUCHK(cudaMemcpyAsync(d_buf_F, buf_F, msg_size, cudaMemcpyHostToDevice, St1));

   if (rank_B >= 0) CUCHK(cudaMemcpyAsync(d_buf_B, buf_B, msg_size, cudaMemcpyHostToDevice, St2));

   update_yldfac_data_y_H(yldfac, d_buf_F, d_buf_B, nxt, nzt, St1, St2, rank_F, rank_B, d_i);
   return;
}


void Cpy2Host_swaparea_X(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyt, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, 
    int zs, int ze, int meshtp){
   long int msg_size;

   update_swapzone_buffer_x_H(u1, v1, w1, xx, yy, zz, xy, xz, yz, d_buf_L, d_buf_R, nyt, St1, St2, rank_L, rank_R, zs, ze, meshtp);

   msg_size = (long int) 9*(2+ngsl+WWL)*((long int) nyt+4+ngsl2+2*WWL)*(ze-zs+1)*sizeof(_prec);
   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/

   if (rank_L >= 0) CUCHK(cudaMemcpyAsync(buf_L, d_buf_L, msg_size, cudaMemcpyDeviceToHost, St1));

   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/
   if (rank_R >= 0) CUCHK(cudaMemcpyAsync(buf_R, d_buf_R, msg_size, cudaMemcpyDeviceToHost, St2));
   return;

}

void Cpy2Device_swaparea_X(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyt, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, 
    int zs, int ze, int meshtp){
   long int msg_size;

   msg_size = 9*(2+ngsl+WWL)*((long int) nyt+4+ngsl2+2*WWL)*(ze-zs+1)*sizeof(_prec);

   if (rank_L >= 0) CUCHK(cudaMemcpyAsync(d_buf_L, buf_L, msg_size, cudaMemcpyHostToDevice, St1));

   if (rank_R >= 0) CUCHK(cudaMemcpyAsync(d_buf_R, buf_R, msg_size, cudaMemcpyHostToDevice, St2));

   update_swapzone_data_x_H(u1, v1, w1, xx, yy, zz, xy, xz, yz, d_buf_L, d_buf_R, nyt, St1, St2, rank_L, rank_R, zs, ze, meshtp);
   return;
}

void Cpy2Host_swaparea_Y(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxt, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, 
    int zs, int ze, int meshtp){
   long int msg_size;

   update_swapzone_buffer_y_H(u1, v1, w1, xx, yy, zz, xy, xz, yz, d_buf_F, d_buf_B, nxt, St1, St2, rank_F, rank_B, zs, ze, meshtp);

   msg_size = (long int) 9*((long int) nxt+4+ngsl2)*(2+ngsl+WWL)*(ze-zs+1)*sizeof(_prec);
   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/

   if (rank_F >= 0) CUCHK(cudaMemcpyAsync(buf_F, d_buf_F, msg_size, cudaMemcpyDeviceToHost, St1));

   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/
   if (rank_B >= 0) CUCHK(cudaMemcpyAsync(buf_B, d_buf_B, msg_size, cudaMemcpyDeviceToHost, St2));
   return;

}

void Cpy2Device_swaparea_Y(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxt, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, 
    int zs, int ze, int meshtp){
   long int msg_size;

   msg_size = (long int) 9*((long int) nxt+4+ngsl2)*(2+ngsl+WWL)*(ze-zs+1)*sizeof(_prec);
   /*fprintf(stdout, "msg_size=%d\n", msg_size);*/

   if (rank_F >= 0) CUCHK(cudaMemcpyAsync(d_buf_F, buf_F, msg_size, cudaMemcpyHostToDevice, St1));

   if (rank_B >= 0) CUCHK(cudaMemcpyAsync(d_buf_B, buf_B, msg_size, cudaMemcpyHostToDevice, St2));

   update_swapzone_data_y_H(u1, v1, w1, xx, yy, zz, xy, xz, yz, d_buf_F, d_buf_B, nxt, St1, St2, rank_F, rank_B, zs, ze, meshtp);
   return;
}

void swaparea_update_corners(_prec *SL_swap, _prec *SR_swap, _prec *RF_swap, _prec *RB_swap, int nz, int off,
         int nxt, int nyt){
   //normally, the corners of ghost cell regions are transferred using in-order communication, not explicitly.
   //This is not working for data beyond the ghost region, as this is not copied to local arrays.
   //Here, data points in the corners outside of the local arrays are explicitly copied from received arrays into
   //arrays being sent out. 

   int blr_slice_1, blr_yline_1, blr_offset;
   int bfb_slice_1, bfb_yline_1, bfb_offset;
   int lr_pos, fb_pos;
   int i, j;
   int ipos, jpos, kpos, ipos2, jpos2;
   int tsize=2+ngsl+off; //total width of transferred data; pad + ghost cells + offset) 
   int l;

   /*communication along X goes beyond ghost cell and padding region along Y;
     the communication along Y does only include ghost cells and padding */
   blr_slice_1  = (nyt+4+ngsl2+2*off)*nz;
   blr_yline_1  = nz;
   blr_offset   = tsize*blr_slice_1;

   bfb_slice_1  = tsize*nz;
   bfb_yline_1  = nz;
   bfb_offset   = (nxt+4+ngsl2)*bfb_slice_1;

   //copy lower left corner of received front swap to lower left corner in left swap to be sent
   for (kpos=0; kpos<nz; kpos++){
      for (j=0; j<off; j++){
         for (i=0; i<tsize; i++){
            ipos=2+ngsl+i;
            jpos=j; 
            ipos2=i;
            jpos2=j; 
	    fb_pos = ipos*bfb_slice_1+jpos*bfb_yline_1+kpos;
	    lr_pos = ipos2*blr_slice_1+jpos2*blr_yline_1+kpos;
            for (l=0; l<9; l++) SL_swap[lr_pos+l*blr_offset] = RF_swap[fb_pos+l*bfb_offset];
         }
      }
      for (j=0; j<off; j++){
         for (i=0; i<tsize; i++){
            ipos=2+ngsl+i;
            jpos=tsize-1-j;
            ipos2=i;
            jpos2=nyt+4+ngsl2+2*off-1-j;
    	    fb_pos = ipos*bfb_slice_1+jpos*bfb_yline_1+kpos;
	    lr_pos = ipos2*blr_slice_1+jpos2*blr_yline_1+kpos;
            for (l=0; l<9; l++) SL_swap[lr_pos+l*blr_offset] = RB_swap[fb_pos+l*bfb_offset];
         }
      }
      for (j=0; j<off; j++){
         for (i=0; i<tsize; i++){
            ipos=nxt-off+i;
            jpos=tsize-1-j;
            ipos2=i;
            jpos2=nyt+4+ngsl2+2*off-1-j;
    	    fb_pos = ipos*bfb_slice_1+jpos*bfb_yline_1+kpos;
	    lr_pos = ipos2*blr_slice_1+jpos2*blr_yline_1+kpos;
            for (l=0; l<9; l++) SR_swap[lr_pos+l*blr_offset] = RB_swap[fb_pos+l*bfb_offset];
         }
      }
      for (j=0; j<off; j++){
         for (i=0; i<tsize; i++){
            ipos=nxt-off+i;
            jpos=j; 
            ipos2=i;
            jpos2=j;
    	    fb_pos = ipos*bfb_slice_1+jpos*bfb_yline_1+kpos;
	    lr_pos = ipos2*blr_slice_1+jpos2*blr_yline_1+kpos;
            for (l=0; l<9; l++) SR_swap[lr_pos+l*blr_offset] = RF_swap[fb_pos+l*bfb_offset];
                /*if (ipos==97 && jpos == 0 && l == 3)
                   fprintf(stdout, */
         }
      }
   }
}

