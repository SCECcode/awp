#ifndef _OPT_TOPOGRAPHY_LAUNCH_CONFIG_H
#define _OPT_TOPOGRAPHY_LAUNCH_CONFIG_H



// Number of threads per block to use for interior velocity kernel
#ifndef VEL_INT_X
#define VEL_INT_X 32
#endif
#ifndef VEL_INT_Y
#define VEL_INT_Y 4
#endif
#ifndef VEL_INT_Z
#define VEL_INT_Z 1
#endif

// Number of threads per block to use for boundary velocity kernel
#ifndef VEL_BND_X
#define VEL_BND_X 7
#endif
#ifndef VEL_BND_Y
#define VEL_BND_Y 8
#endif
#ifndef VEL_BND_Z
#define VEL_BND_Z 1
#endif

// Launch bounds
 
#ifndef DTOPO_VEL_110_MAX_THREADS_PER_BLOCK
#define DTOPO_VEL_110_MAX_THREADS_PER_BLOCK 32
#endif

#ifndef DTOPO_VEL_110_MIN_BLOCKS_PER_SM
#define DTOPO_VEL_110_MIN_BLOCKS_PER_SM 8
#endif

#ifndef DTOPO_VEL_111_MAX_THREADS_PER_BLOCK
#define DTOPO_VEL_111_MAX_THREADS_PER_BLOCK 256
#endif

#ifndef DTOPO_VEL_111_MIN_BLOCKS_PER_SM
#define DTOPO_VEL_111_MIN_BLOCKS_PER_SM 1
#endif                                  

#ifndef DTOPO_VEL_112_MAX_THREADS_PER_BLOCK
#define DTOPO_VEL_112_MAX_THREADS_PER_BLOCK 256
#endif

#ifndef DTOPO_VEL_112_MIN_BLOCKS_PER_SM
#define DTOPO_VEL_112_MIN_BLOCKS_PER_SM 1
#endif

#ifndef DTOPO_BUF_VEL_110_MAX_THREADS_PER_BLOCK
#define DTOPO_BUF_VEL_110_MAX_THREADS_PER_BLOCK 128
#endif

#ifndef DTOPO_BUF_VEL_110_MIN_BLOCKS_PER_SM
#define DTOPO_BUF_VEL_110_MIN_BLOCKS_PER_SM 1
#endif

#ifndef DTOPO_BUF_VEL_111_MAX_THREADS_PER_BLOCK
#define DTOPO_BUF_VEL_111_MAX_THREADS_PER_BLOCK 128
#endif

#ifndef DTOPO_BUF_VEL_111_MIN_BLOCKS_PER_SM
#define DTOPO_BUF_VEL_111_MIN_BLOCKS_PER_SM 1
#endif

#ifndef DTOPO_BUF_VEL_112_MAX_THREADS_PER_BLOCK
#define DTOPO_BUF_VEL_112_MAX_THREADS_PER_BLOCK 128
#endif

#ifndef DTOPO_BUF_VEL_112_MIN_BLOCKS_PER_SM
#define DTOPO_BUF_VEL_112_MIN_BLOCKS_PER_SM 1
#endif

#endif
