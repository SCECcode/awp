#ifndef _TOPOGRAPHY_MMS_H
#define _TOPOGRAPHY_MMS_H

#ifdef __cplusplus
extern "C" {
#endif
void mms_init(const char *MMSFILE,
        const int *nxt, const int *nyt,
              const int *nzt, const int ngrids, float **d_d1, float **d_lam,
              float **d_mu,
              float **d_qp, float **d_qs,
              float **d_vx, float **d_vy, float **d_vz,
              float **d_xx, float **d_yy, float **d_zz, float **d_xy,
              float **d_xz, float **d_yz, int px, int py, const float *h, const float dt);
#ifdef __cplusplus
}
#endif

#endif // MMS_CUH


