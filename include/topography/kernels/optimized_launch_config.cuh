#ifndef _OPT_TOPOGRAPHY_LAUNCH_CONFIG_H
#define _OPT_TOPOGRAPHY_LAUNCH_CONFIG_H

// Number of threads per block to use for interior velocity kernel
#ifndef VEL_INT_X
#define VEL_INT_X 64
#endif
#ifndef VEL_INT_Y
#define VEL_INT_Y 4
#endif
#ifndef VEL_INT_Z
#define VEL_INT_Z 4
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

// Number of threads per block to use for interior stress kernel
#ifndef STR_INT_X
#define STR_INT_X 32
#endif
#ifndef STR_INT_Y
#define STR_INT_Y 4
#endif
#ifndef STR_INT_Z
#define STR_INT_Z 1
#endif

// Number of threads per block to use for boundary stress kernel
#ifndef STR_BND_X
#define STR_BND_X 7
#endif
#ifndef STR_BND_Y
#define STR_BND_Y 8
#endif
#ifndef STR_BND_Z
#define STR_BND_Z 1
#endif

// Kernel naming convention
// 110: Bottom boundary (only used in debug mode)
// 111: Interior
// 112: Top boundary


// Number of threads per block
// grid dimension (X, Y, Z) refers to CUDA grid indices
#ifndef DTOPO_VEL_110_X
#define DTOPO_VEL_110_X VEL_BND_X
#endif
#ifndef DTOPO_VEL_110_Y
#define DTOPO_VEL_110_Y VEL_BND_Y
#endif
#ifndef DTOPO_VEL_110_Z
#define DTOPO_VEL_110_Z VEL_BND_Z
#endif

#ifndef DTOPO_VEL_111_X
#define DTOPO_VEL_111_X VEL_INT_X
#endif
#ifndef DTOPO_VEL_111_Y
#define DTOPO_VEL_111_Y VEL_INT_Y
#endif
#ifndef DTOPO_VEL_111_Z
#define DTOPO_VEL_111_Z VEL_INT_Z
#endif

#ifndef DTOPO_VEL_112_X
#define DTOPO_VEL_112_X VEL_BND_X
#endif
#ifndef DTOPO_VEL_112_Y
#define DTOPO_VEL_112_Y VEL_BND_Y
#endif
#ifndef DTOPO_VEL_112_Z
#define DTOPO_VEL_112_Z VEL_BND_Z
#endif

#ifndef DTOPO_BUF_VEL_111_X
#define DTOPO_BUF_VEL_111_X VEL_INT_X
#endif
#ifndef DTOPO_BUF_VEL_111_Y
#define DTOPO_BUF_VEL_111_Y VEL_INT_Y
#endif
#ifndef DTOPO_BUF_VEL_111_Z
#define DTOPO_BUF_VEL_111_Z VEL_INT_Z
#endif

#ifndef DTOPO_BUF_VEL_112_X
#define DTOPO_BUF_VEL_112_X VEL_BND_X
#endif
#ifndef DTOPO_BUF_VEL_112_Y
#define DTOPO_BUF_VEL_112_Y VEL_BND_Y
#endif
#ifndef DTOPO_BUF_VEL_112_Z
#define DTOPO_BUF_VEL_112_Z VEL_BND_Z
#endif

#ifndef DTOPO_BUF_VEL_110_X
#define DTOPO_BUF_VEL_110_X VEL_BND_X
#endif
#ifndef DTOPO_BUF_VEL_110_Y
#define DTOPO_BUF_VEL_110_Y VEL_BND_Y
#endif
#ifndef DTOPO_BUF_VEL_110_Z
#define DTOPO_BUF_VEL_110_Z VEL_BND_Z
#endif

#ifndef DTOPO_STR_110_X
#define DTOPO_STR_110_X STR_INT_X
#endif

#ifndef DTOPO_STR_110_Y
#define DTOPO_STR_110_Y STR_INT_Y
#endif

#ifndef DTOPO_STR_110_Z
#define DTOPO_STR_110_Z STR_INT_Z
#endif

#ifndef DTOPO_STR_111_X
#define DTOPO_STR_111_X STR_INT_X
#endif

#ifndef DTOPO_STR_111_Y
#define DTOPO_STR_111_Y STR_INT_Y
#endif

#ifndef DTOPO_STR_111_Z
#define DTOPO_STR_111_Z STR_INT_Z
#endif

#ifndef DTOPO_STR_112_X
#define DTOPO_STR_112_X STR_INT_X
#endif

#ifndef DTOPO_STR_112_Y
#define DTOPO_STR_112_Y STR_INT_Y
#endif

#ifndef DTOPO_STR_112_Z
#define DTOPO_STR_112_Z STR_INT_Z
#endif


// Launch bounds
 
#ifndef DTOPO_VEL_110_MAX_THREADS_PER_BLOCK
#define DTOPO_VEL_110_MAX_THREADS_PER_BLOCK 32
#endif

#ifndef DTOPO_VEL_111_MAX_THREADS_PER_BLOCK
#define DTOPO_VEL_111_MAX_THREADS_PER_BLOCK 1024
#endif

#ifndef DTOPO_VEL_112_MAX_THREADS_PER_BLOCK
#define DTOPO_VEL_112_MAX_THREADS_PER_BLOCK 64
#endif

#ifndef DTOPO_BUF_VEL_110_MAX_THREADS_PER_BLOCK
#define DTOPO_BUF_VEL_110_MAX_THREADS_PER_BLOCK 1024
#endif

#ifndef DTOPO_BUF_VEL_111_MAX_THREADS_PER_BLOCK
#define DTOPO_BUF_VEL_111_MAX_THREADS_PER_BLOCK 1024
#endif

#ifndef DTOPO_BUF_VEL_112_MAX_THREADS_PER_BLOCK
#define DTOPO_BUF_VEL_112_MAX_THREADS_PER_BLOCK 1024
#endif

// Apply loop in kernel
// This option must be compatible with the kernel. If there is no loop in the
// kernel, turn off this option, and vice versa.
#define DTOPO_VEL_110_LOOP_Z 1
#define DTOPO_VEL_111_LOOP_Z 0
#define DTOPO_VEL_112_LOOP_Z 0
#define DTOPO_BUF_VEL_110_LOOP_Z 1
#define DTOPO_BUF_VEL_111_LOOP_Z 0
#define DTOPO_BUF_VEL_112_LOOP_Z 0
#define DTOPO_STR_110_LOOP_Z 1
#define DTOPO_STR_111_LOOP_Z 1
#define DTOPO_STR_112_LOOP_Z 1

#endif
