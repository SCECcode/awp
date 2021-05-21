#include "mms.h"
_prec mms_init_vx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 1.2)/k;
}

_prec mms_init_vy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.25)/k;
}

_prec mms_init_vz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.40000000000000002)/k;
}

_prec mms_init_sxx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.69999999999999996)/k;
}

_prec mms_init_syy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.29999999999999999)/k;
}

_prec mms_init_szz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.12)/k;
}

_prec mms_init_sxy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.02)/k;
}

_prec mms_init_sxz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.46999999999999997)/k;
}

_prec mms_init_syz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z + 0.33000000000000002)/k;
}

_prec mms_final_vx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 0.46999999999999997) + sin(k*x)*sin(k*z + 0.02)*cos(k*y) + sin(k*y)*sin(k*z + 0.69999999999999996)*cos(k*x);
}

_prec mms_final_vy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 0.33000000000000002) + sin(k*x)*sin(k*z + 0.29999999999999999)*cos(k*y) + sin(k*y)*sin(k*z + 0.02)*cos(k*x);
}

_prec mms_final_vz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 0.12) + sin(k*x)*sin(k*z + 0.33000000000000002)*cos(k*y) + sin(k*y)*sin(k*z + 0.46999999999999997)*cos(k*x);
}

_prec mms_final_sxx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 0.40000000000000002) + sin(k*x)*sin(k*z + 0.25)*cos(k*y) + 3*sin(k*y)*sin(k*z + 1.2)*cos(k*x);
}

_prec mms_final_syy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 0.40000000000000002) + 3*sin(k*x)*sin(k*z + 0.25)*cos(k*y) + sin(k*y)*sin(k*z + 1.2)*cos(k*x);
}

_prec mms_final_szz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return 3*sin(k*x)*sin(k*y)*cos(k*z + 0.40000000000000002) + sin(k*x)*sin(k*z + 0.25)*cos(k*y) + sin(k*y)*sin(k*z + 1.2)*cos(k*x);
}

_prec mms_final_sxy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*z + 1.2)*cos(k*y) + sin(k*y)*sin(k*z + 0.25)*cos(k*x);
}

_prec mms_final_sxz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 1.2) + sin(k*y)*sin(k*z + 0.40000000000000002)*cos(k*x);
}

_prec mms_final_syz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*cos(k*z + 0.25) + sin(k*x)*sin(k*z + 0.40000000000000002)*cos(k*y);
}

