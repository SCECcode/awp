#include <awp/definitions.h>
#include <assert.h>
// offsets that point to the first grid point in the grid
//const int offset_x = 2 + ngsl;
//const int offset_y = 2 + ngsl;
//const int offset_z = align;
//
//
//int find_block_index(const _prec *z, const int num_grids);
//
//void inbounds( ); 
//
//void fill_x();
//
//
_prec grid_overlap(const _prec h) {
    return 7.0 * h;
}
_prec grid_height(const int nz, const _prec h, const int istopo) {
    return istopo == 1 ? (nz - 2) * h : (nz - 1) * h;
}
void global_to_local(_prec *zloc, int *block_index, const _prec z,
                     const _prec h, const int *nz, const int num_grids,
                     const int istopo) {
    _prec z0 = z;
    _prec bi = -1;

    _prec hloc = h;
    _prec H = 0.0;
    // Go from top grid to bottom grid
    for (int i = 0; i < num_grids; ++i ) {

        if (i > 0) 
            z0 -= grid_overlap(hloc / 3);

        // Check minimum number of grid points per block
        assert(nz[i] >= 7);

        _prec overlap = grid_overlap(hloc);
        
        H = i == 0 ? grid_height(nz[i], hloc, istopo) : grid_height(nz[i], hloc, 0);

        z0 += H;
        hloc *= 3;
        bi = i;

        //printf("z0 + H = %g i = %d \n", z0, i);

        // Check if the coordinate is in the overlap zone, if so, push it to the next grid
        if (z0 > 0 && z0 < grid_overlap(hloc / 3) ) {
            //printf("in overlap zone, z0 = %g i = %d overlap = %g \n", z0, i, overlap);
            continue;
        }

        if (z0 > 0) break;
        
        //printf("next,  z0 = %g i = %d \n", z0, i);

    }

    // Check if the mapping succeeded or not
    if (z0 < 0) {
        printf("WARNING: Failed to map z=%g to a block.\n", z);
    }

    *zloc = z0;
    *block_index = bi;
}
