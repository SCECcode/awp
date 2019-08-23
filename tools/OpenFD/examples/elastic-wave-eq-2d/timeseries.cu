
__global__ void save_seism(float *v1_out, float *v2_out, float *s11_out, float *s12_out, float *s22_out, const float *v1, const float *v2, const float *s11, const float *s12, const float *s22, const int *idx_x, const int *idx_y, const int nx, const int ny, const int nidx, const int step)
{
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if (i >= nidx) return;
     const int idx = idx_x[i] + idx_y[i]*nx;
     v1_out[i + step*nidx] = v1[idx];
     v2_out[i + step*nidx] = v2[idx];
     s11_out[i + step*nidx] = s11[idx];
     s12_out[i + step*nidx] = s12[idx];
     s22_out[i + step*nidx] = s22[idx];
        
}
/*
 * Compute the Strain Green's tensor e11, e12, e22
 * e11 = d G_1 / dx1
 * e12 = 0.5*(d G_1 / dx2 + d G_2/dx_1)
 * e22 = d G_2 / dx2
 * These strains are obtained by solving the elastodynamic equations due to
 * point forces acting in either the x1-direction, or x2-direction. The strain
 * Green's tensor are obtained by converting the stress field at each receiver
 * location, and at each point in time, to strains using the constitutive law.
 */
__global__ void sgt(float *e11, float *e12, float *e22, const float *s11, const float *s12, const float *s22, const float lam, const float mu, const int *idx_x, const int *idx_y, const int nx, const int ny, const int nidx, const int step)
{
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if (i >= nidx) return;
     const int idx = idx_x[i] + idx_y[i]*nx;
     e11[i + step*nidx] = -0.5*lam*(s11[idx] + s22[idx])/(mu*(2*lam + mu)) +
                           0.5*s11[idx]/mu;
     e22[i + step*nidx] = -0.5*lam*(s11[idx] + s22[idx])/(mu*(2*lam + mu)) + 
                           0.5*s22[idx]/mu;
     e12[i + step*nidx] = 0.5*s12[idx]/mu;
     
}

