__global__ void pde_r(float *du, const float *u, const float a, const float ak, const float hi, const int n)
{
     const float dr[4][6] = {{1.4117647058823530, -1.7352941176470589, 0.2352941176470588, 0.0882352941176471, -0.0000000000000000, -0.0000000000000000}, {0.5000000000000000, -0.0000000000000000, -0.5000000000000000, -0.0000000000000000, -0.0000000000000000, -0.0000000000000000}, {-0.0930232558139535, 0.6860465116279070, -0.0000000000000000, -0.6860465116279070, 0.0930232558139535, -0.0000000000000000}, {-0.0306122448979592, -0.0000000000000000, 0.6020408163265306, -0.0000000000000000, -0.6530612244897959, 0.0816326530612245}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 4) return;
     #define _(i) [(i)]
     #define _du(i) du[(i)]
     #define _u(i) u[(i)]
     _du(n - 1 - i) = -a*hi*(dr[i][5]*_u(n - 6) + dr[i][4]*_u(n - 5) + dr[i][3]*_u(n - 4) + dr[i][2]*_u(n - 3) + dr[i][1]*_u(n - 2) + dr[i][0]*_u(n - 1)) + ak*_du(n - 1 - i);
     #undef _
     #undef _du
     #undef _u
     
}

__global__ void pde_l(float *du, const float *u, const float a, const float ak, const float hi, const int n)
{
     const float dl[4][6] = {{-1.4117647058823530, 1.7352941176470589, -0.2352941176470588, -0.0882352941176471, 0.0000000000000000, 0.0000000000000000}, {-0.5000000000000000, 0.0000000000000000, 0.5000000000000000, 0.0000000000000000, 0.0000000000000000, 0.0000000000000000}, {0.0930232558139535, -0.6860465116279070, 0.0000000000000000, 0.6860465116279070, -0.0930232558139535, 0.0000000000000000}, {0.0306122448979592, 0.0000000000000000, -0.6020408163265306, 0.0000000000000000, 0.6530612244897959, -0.0816326530612245}};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 4) return;
     #define _du(i) du[(i)]
     #define _u(i) u[(i)]
     _du(i) = -a*hi*(dl[i][0]*_u(0) + dl[i][1]*_u(1) + dl[i][2]*_u(2) + dl[i][3]*_u(3) + dl[i][4]*_u(4) + dl[i][5]*_u(5)) + ak*_du(i);
     #undef _du
     #undef _u
     
}

__global__ void pde_i(float *du, const float *u, const float a, const float ak, const float hi, const int n)
{
     const float d[4] = {0.0833333333333333, -0.6666666666666666, 0.6666666666666666, -0.0833333333333333};
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= n - 8) return;
     #define _du(i) du[(i)]
     #define _u(i) u[(i)]
     _du(i + 4) = -a*hi*(d[0]*_u(i + 2) + d[1]*_u(i + 3) + d[2]*_u(i + 5) + d[3]*_u(i + 6)) + ak*_du(i + 4);
     #undef _du
     #undef _u
     
}

__global__ void bc_l(float *du, const float *u, const float a, const float ck, const float dt, const float hi, const float omega, const float t, const int n)
{
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= 1) return;
     #define _du(i) du[(i)]
     #define _u(i) u[(i)]
     _du(i) = -2.82352941176471*a*hi*(sin(omega*(ck*dt + t)) + _u(i)) + _du(i);
     #undef _du
     #undef _u
     
}

__global__ void update_i(float *u, const float *du, const float bk, const float dt, const int n)
{
     const int i = threadIdx.x + blockIdx.x*blockDim.x;
     if ( i >= n) return;
     #define _du(i) du[(i)]
     #define _u(i) u[(i)]
     _u(i) = bk*dt*_du(i) + _u(i);
     #undef _du
     #undef _u
     
}

