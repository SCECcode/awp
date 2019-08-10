#pragma once
#ifndef GRID_ERROR_H
#define GRID_ERROR_H
#ifdef __cplusplus
extern "C" {
#endif

enum error_codes {SUCCESS, 
                  ERR_GRID_OUT_OF_BOUNDS = 400,
                  ERR_OUT_OF_BOUNDS_LOWER = 700,
                  ERR_OUT_OF_BOUNDS_UPPER = 701,
                  ERR_NON_POSITIVE = 800
};

const char* error_message(const int err);
void error_print(const int err);

#ifdef __cplusplus
}
#endif
#endif

