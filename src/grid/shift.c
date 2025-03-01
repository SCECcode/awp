#include <grid/shift.h>
#include <awp/definitions.h>

void shift_node(int *shift)
{
        shift[0] = 0;
        shift[1] = 0;
        shift[2] = 0;
}

 void shift_u1(int *shift)
{
        shift[0] = 0;
        shift[1] = 1;
        shift[2] = 1;
}


 void shift_u2(int *shift)
{
        shift[0] = 1;
        shift[1] = 0;
        shift[2] = 1;
}

 void shift_u3(int *shift)
{
        shift[0] = 1;
        shift[1] = 1;
        shift[2] = 0;
}

 void shift_xx(int *shift)
{
        shift[0] = 1;
        shift[1] = 1;
        shift[2] = 1;
}

 void shift_yy(int *shift)
{
        shift[0] = 1;
        shift[1] = 1;
        shift[2] = 1;
}

 void shift_zz(int *shift)
{
        shift[0] = 1;
        shift[1] = 1;
        shift[2] = 1;
}

 void shift_xy(int *shift)
{
        shift[0] = 0;
        shift[1] = 0;
        shift[2] = 1;
}

 void shift_xz(int *shift)
{
        shift[0] = 0;
        shift[1] = 1;
        shift[2] = 0;
}

 void shift_yz(int *shift)
{
        shift[0] = 1;
        shift[1] = 0;
        shift[2] = 0;
}


 int3_t grid_node(void)
{
        int3_t out = {.x = 0, .y = 0, .z = 0};
        return out;
}

 int3_t grid_u1(void)
{
        int3_t out = {.x = 0, .y = 1, .z = 1};
        return out;
}

 int3_t grid_u2(void)
{
        int3_t out = {.x = 1, .y = 0, .z = 1};
        return out;
}

 int3_t grid_u3(void)
{
        int3_t out = {.x = 1, .y = 1, .z = 0};
        return out;
}

 int3_t grid_x(void)
{
        return grid_u1();
}

 int3_t grid_y(void)
{
        return grid_u2();
}

 int3_t grid_z(void)
{
        return grid_u3();
}


 int3_t grid_xx(void)
{
        int3_t out = {.x = 1, .y = 1, .z = 1};
        return out;
}

 int3_t grid_yy(void)
{
        int3_t out = {.x = 1, .y = 1, .z = 1};
        return out;
}

 int3_t grid_zz(void)
{
        int3_t out = {.x = 1, .y = 1, .z = 1};
        return out;
}

 int3_t grid_xy(void)
{
        int3_t out = {.x = 0, .y = 0, .z = 1};
        return out;
}

 int3_t grid_xz(void)
{
        int3_t out = {.x = 0, .y = 1, .z = 0};
        return out;
}

 int3_t grid_yz(void)
{
        int3_t out = {.x = 1, .y = 0, .z = 0};
        return out;
}

 int3_t grid_shift(enum eshift gridtype) {
        switch (gridtype) {
                case GRID_U1:
                        return grid_u1();
                case GRID_U2:
                        return grid_u2();
                case GRID_U3:
                        return grid_u3();
                case GRID_XX:
                        return grid_xx();
                case GRID_YY:
                        return grid_yy();
                case GRID_ZZ:
                        return grid_zz();
                case GRID_XY:
                        return grid_xy();
                case GRID_XZ:
                        return grid_xz();
                case GRID_YZ:
                        return grid_yz();
                }
        int3_t out = {0, 0, 0};
        return out;
}

 const char *grid_shift_label(enum eshift gridtype) {
        switch (gridtype) {
                case GRID_U1:
                        return "u1";
                case GRID_U2:
                        return "u2";
                case GRID_U3:
                        return "u3";
                case GRID_XX:
                        return "xx";
                case GRID_YY:
                        return "yy";
                case GRID_ZZ:
                        return "zz";
                case GRID_XY:
                        return "xy";
                case GRID_XZ:
                        return "xz";
                case GRID_YZ:
                        return "yz";
                }
        return "";
}
