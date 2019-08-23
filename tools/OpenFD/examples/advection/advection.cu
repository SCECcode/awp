__global__ void pde_r(float *du, const float *u, const float a, const float ak, const float hi, const int n)
{
<<<<<<< HEAD
     const float dxr[4][6] = {{1.411765, -1.735294, 0.235294, 0.088235, -0.000000, -0.000000}, {0.500000, -0.000000, -0.500000, -0.000000, -0.000000, -0.000000}, {-0.093023, 0.686047, -0.000000, -0.686047, 0.093023, -0.000000}, {-0.030612, -0.000000, 0.602041, -0.000000, -0.653061, 0.081633}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 4) return;
     du[n - 1 - i] = -a*hi*dxr[i][5]*u[n - 6] - a*hi*dxr[i][4]*u[n - 5] - a*hi*dxr[i][3]*u[n - 4] - a*hi*dxr[i][2]*u[n - 3] - a*hi*dxr[i][1]*u[n - 2] - a*hi*dxr[i][0]*u[n - 1] + ak*du[n - 1 - i];
=======
     const float dxr[1][2] = {{1.000000, -1.000000}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 1) return;
     du[n - 1 - i] = -a*hi*dxr[i][1]*u[n - 2] - a*hi*dxr[i][0]*u[n - 1] + ak*du[n - 1 - i];
>>>>>>> ceca10637173c027c198ff7511406fa7b92b910a
     
}

__global__ void pde_l(float *du, const float *u, const float a, const float ak, const float hi, const int n)
{
<<<<<<< HEAD
     const float dxl[4][6] = {{-1.411765, 1.735294, -0.235294, -0.088235, 0.000000, 0.000000}, {-0.500000, 0.000000, 0.500000, 0.000000, 0.000000, 0.000000}, {0.093023, -0.686047, 0.000000, 0.686047, -0.093023, 0.000000}, {0.030612, 0.000000, -0.602041, 0.000000, 0.653061, -0.081633}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 4) return;
     du[i] = -a*hi*dxl[i][0]*u[0] - a*hi*dxl[i][1]*u[1] - a*hi*dxl[i][2]*u[2] - a*hi*dxl[i][3]*u[3] - a*hi*dxl[i][4]*u[4] - a*hi*dxl[i][5]*u[5] + ak*du[i];
=======
     const float dxl[1][2] = {{-1.000000, 1.000000}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 1) return;
     du[i] = -a*hi*dxl[i][0]*u[0] - a*hi*dxl[i][1]*u[1] + ak*du[i];
>>>>>>> ceca10637173c027c198ff7511406fa7b92b910a
     
}

__global__ void pde_i(float *du, const float *u, const float a, const float ak, const float hi, const int n)
{
<<<<<<< HEAD
     const float dx[4] = {0.0833333, -0.666667, 0.666667, -0.0833333};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= n - 8) return;
     du[i + 4] = -a*hi*dx[0]*u[i + 2] - a*hi*dx[1]*u[i + 3] - a*hi*dx[2]*u[i + 5] - a*hi*dx[3]*u[i + 6] + ak*du[i + 4];
=======
     const float dx[2] = {-0.5, 0.5};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= n - 2) return;
     du[i + 1] = -a*hi*dx[0]*u[i] - a*hi*dx[1]*u[i + 2] + ak*du[i + 1];
>>>>>>> ceca10637173c027c198ff7511406fa7b92b910a
     
}

__global__ void bc_l(float *du, const float *u, const float a, const float ck, const float dt, const float hi, const float omega, const float t, const int n)
{
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 1) return;
<<<<<<< HEAD
     du[i] = -2.82352941176471*a*hi*sin(ck*dt*omega + omega*t) - 2.82352941176471*a*hi*u[i] + du[i];
=======
     du[i] = -2.0*a*hi*sin(ck*dt*omega + omega*t) - 2.0*a*hi*u[i] + du[i];
>>>>>>> ceca10637173c027c198ff7511406fa7b92b910a
     
}

__global__ void update_i(float *u, const float *du, const float bk, const float dt, const int n)
{
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= n) return;
     u[i] = bk*dt*du[i] + u[i];
     
}

