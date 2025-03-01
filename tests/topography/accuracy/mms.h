#ifndef MMS_H
#define MMS_H
#include <awp/definitions.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

_prec mms_init_vx(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_vy(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_vz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_sxx(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_syy(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_szz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_sxy(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_sxz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_init_syz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_vx(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_vy(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_vz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_sxx(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_syy(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_szz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_sxy(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_sxz(const _prec x, const _prec y, const _prec z, const _prec *properties);
_prec mms_final_syz(const _prec x, const _prec y, const _prec z, const _prec *properties);

#endif
