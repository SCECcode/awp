#pragma once
#ifndef FUNCTION_RANDOM_H
#define FUNCTION_RANDOM_H


#include <awp/definitions.h>

#ifdef __cplusplus
extern "C" {
#endif
_prec randomf(void);
void set_seed(int seed);

#ifdef __cplusplus
}
#endif
#endif
