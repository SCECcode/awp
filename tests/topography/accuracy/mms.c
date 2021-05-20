#include "mms.h"
_prec mms_init_vx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_vy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_vz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_sxx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_syy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_szz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_sxy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_sxz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_init_syz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return sin(k*x)*sin(k*y)*sin(k*z);
}

_prec mms_final_vx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*x)*sin(k*z)*cos(k*y) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_vy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*x)*sin(k*z)*cos(k*y) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_vz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*x)*sin(k*z)*cos(k*y) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_sxx(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*x)*sin(k*z)*cos(k*y) + 3*k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_syy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + 3*k*sin(k*x)*sin(k*z)*cos(k*y) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_szz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return 3*k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*x)*sin(k*z)*cos(k*y) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_sxy(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*z)*cos(k*y) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_sxz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*y)*sin(k*z)*cos(k*x);
}

_prec mms_final_syz(const _prec x, const _prec y, const _prec z, const _prec *properties)
{
     _prec k = properties[0];
     return k*sin(k*x)*sin(k*y)*cos(k*z) + k*sin(k*x)*sin(k*z)*cos(k*y);
}

