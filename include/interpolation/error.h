#ifndef INTERP_ERROR_H
#define INTERP_ERROR_H
#ifdef __cplusplus
extern "C" {
#endif

enum error_codes {SUCCESS, 
                  ERR_GRID_OUT_OF_BOUNDS = 400,
                  ERR_LAGRANGE_LOWER_EXTRAPOLATION = 600,
                  ERR_LAGRANGE_UPPER_EXTRAPOLATION = 601,
                  ERR_OUT_OF_BOUNDS_LOWER = 700,
                  ERR_OUT_OF_BOUNDS_UPPER = 701,
                  ERR_INTERP_MALLOC = 900,
                  ERR_INTERP_WRITE_END_OF_FILE = 901,
                  ERR_INCONSISTENT_SIZE = 902
};


const char* error_message(const int err);
void error_print(const int err);

#ifdef __cplusplus
}
#endif
#endif
