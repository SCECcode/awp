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

void mms_exact_velocity(
              float *d_vx, float *d_vy, float *d_vz,
              const int nx, const int ny, const int nz, 
              const int px, const int py, const int pz, 
              const int bi, const int bj, const int bk, 
              const int ei, const int ej, const int ek, 
              const float h, const float t, const int apply_in_interior);

void mms_exact_stress(
              float *d_xx, float *d_yy, float *d_zz, 
              float *d_xy, float *d_xz, float *d_yz, 
              const int nx, const int ny, const int nz, 
              const int px, const int py, const int pz, 
              const int bi, const int bj, const int bk, 
              const int ei, const int ej, const int ek, 
              const float h, const float t, const int apply_in_interior);


void mms_force_velocity(float *d_vx, float *d_vy, float *d_vz, const int nx,
                        const int ny, const int nz, const float h, const int px,
                        const int py, const int pz, const float t, const float dt);

void mms_force_stress(float *d_xx, float *d_yy, float *d_zz, float *d_xy,
                      float *d_xz, float *d_yz, const int nx, const int ny, const int nz,
                      const float h, const int px, const int py, const int pz, const float t, const float dt);
#ifdef __cplusplus
}
#endif

#endif // MMS_CUH


