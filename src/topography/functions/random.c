#include <stdlib.h>

#include <awp/pmcl3d_cons.h>
#include <functions/random.h>

_prec randomf(void){
        return (_prec)rand()/(_prec)(RAND_MAX);
}

void set_seed(int s)
{
        srand((unsigned int)s);
}

