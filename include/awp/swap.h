#ifndef SWAP_H
#define SWAP_H

void mediaswap(Grid3D d1, Grid3D mu,     Grid3D lam,    Grid3D qp,     Grid3D qs,
               int rank,  int x_rank_L,  int x_rank_R,  int y_rank_F,  int y_rank_B,
               int nxt,   int nyt,       int nzt,       MPI_Comm MCW,  int p);

void Cpy2Device_source(int npsrc, int READ_STEP,
      int index_offset,
      Grid1D taxx, Grid1D tayy, Grid1D tazz,
      Grid1D taxz, Grid1D tayz, Grid1D taxy,
      _prec *d_taxx, _prec *d_tayy, _prec *d_tazz,
      _prec *d_taxz, _prec *d_tayz, _prec *d_taxy, int IFAULT);

void Cpy2Host_VX(_prec*  u1, _prec*  v1, _prec*  w1, _prec*  h_m, int nxt, int nyt, int nzt, cudaStream_t St, int rank, int flag);

void Cpy2Host_VY(_prec*  s_u1, _prec*  s_v1, _prec*  s_w1, _prec*  h_m, int nxt, int nzt, cudaStream_t St, int rank);

void Cpy2Device_VX(_prec*  u1, _prec*  v1, _prec*  w1,        _prec*  L_m,       _prec*  R_m, int nxt,    
                   int nyt,   int nzt,   cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R);

void Cpy2Device_VY(_prec*  u1,   _prec *v1,  _prec *w1,  _prec*  f_u1, _prec*  f_v1, _prec*  f_w1, _prec*  b_u1,      _prec*  b_v1,
                   _prec*  b_w1, _prec*  F_m, _prec*  B_m, int nxt,     int nyt,     int nzt,     cudaStream_t St1, cudaStream_t St2,
                   int rank_F,  int rank_B, int d_i);

void Cpy2Host_yldfac(_prec *d_L, _prec *d_R, _prec *d_F, _prec *d_B,
      _prec *d_FL, _prec *d_FR, _prec *d_BL, _prec *d_BR, 
      _prec *SL, _prec *SR, _prec *SF, _prec *SB, 
      _prec *SFL, _prec *SFR, _prec *SBL, _prec *SBR,
      cudaStream_t St, int rank_L, int rank_R, int rank_F, int rank_B, 
      int nxt, int nyt, int d_i);

void Cpy2Device_yldfac(_prec *d_yldfac,
      _prec *d_L, _prec *d_R, _prec *d_F, _prec *d_B,
      _prec *d_FL, _prec *d_FR, _prec *d_BL, _prec *d_BR, 
      _prec *RL, _prec *RR, _prec *RF, _prec *RB, 
      _prec *RFL, _prec *RFR, _prec *RBL, _prec *RBR,
      cudaStream_t St, int rank_L, int rank_R, int rank_F, int rank_B, 
      int nxt, int nyt, int nzt, int d_i);

void PostSendMsg_X(_prec*  SL_M, _prec*  SR_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size,
                   int rank_L,  int rank_R,  int rank,     int flag, int gridnum);

void PostSendMsg_Y(_prec*  SF_M, _prec*  SB_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size,
                   int rank_F,  int rank_B,  int rank,     int flag, int gridnum);

void PostRecvMsg_yldfac(_prec *RL_M, _prec *RR_M, _prec *RF_M, _prec *RB_M, 
      _prec *RFL_M, _prec *RFR_M, _prec *RBL_M, _prec *RBR_M,
      MPI_Comm MCW, MPI_Request *request, int *count, 
      int msg_size_x, int msg_size_y,
      int rank_L,   int rank_R,   int rank_F,   int rank_B,
      int rank_FL,   int rank_FR,   int rank_BL,   int rank_BR);

void PostSendMsg_yldfac(_prec *SL_M, _prec *SR_M, _prec *SF_M, _prec *SB_M, 
      _prec *SFL_M, _prec *SFR_M, _prec *SBL_M, _prec *SBR_M,
      MPI_Comm MCW, MPI_Request *request, int *count, 
      int msg_size_x, int msg_size_y, int rank,
      int rank_L,   int rank_R,   int rank_F,   int rank_B,
      int rank_FL,  int rank_FR,  int rank_BL,  int rank_BR);

void PostRecvMsg_X(_prec*  RL_M, _prec*  RR_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size, int rank_L, int rank_R, int gridnum);

void Cpy2Host_yldfac_X(_prec*  yldfac, _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyt, int nzt, 
   cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int meshtp);

void Cpy2Device_yldfac_X(_prec*  yldfac, _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyt, int nzt,
    cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, int meshtp);

void Cpy2Host_yldfac_Y(_prec*  yldfac, _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxt, int nzt, 
    cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int meshtp);

void Cpy2Device_yldfac_Y(_prec*  yldfac, _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxt, int nzt,
    cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, int meshtp);


void PostRecvMsg_Y(_prec*  RF_M, _prec*  RB_M, MPI_Comm MCW, MPI_Request* request, int* count, int msg_size, int rank_F, int rank_B, int gridnum);

void Cpy2Host_swaparea_X(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyth, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, 
    int zs, int ze, int meshtp);

void Cpy2Device_swaparea_X(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_L, _prec *buf_R, _prec *d_buf_L, _prec *d_buf_R, int nyth, cudaStream_t St1, cudaStream_t St2, int rank_L, int rank_R, 
    int zs, int ze, int meshtp);

void Cpy2Host_swaparea_Y(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxth, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, 
    int zs, int ze, int meshtp);

void Cpy2Device_swaparea_Y(_prec*  u1, _prec*  v1, _prec*  w1, _prec *xx, _prec *yy, _prec *zz, _prec *xy, _prec *xz, _prec *yz,
    _prec *buf_F, _prec *buf_B, _prec *d_buf_F, _prec *d_buf_B, int nxth, cudaStream_t St1, cudaStream_t St2, int rank_F, int rank_B, 
    int zs, int ze, int meshtp);

#endif

